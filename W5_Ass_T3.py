
import os
import json
import asyncio
import tiktoken

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


orders = {"1001": "Shipped", "1002": "Delivered"}
customers = {"1": "John", "2": "David"}

tools = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Get order status using order ID",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}
        }, "required": ["order_id"], "additionalProperties": False}
    }},
    {"type": "function", "function": {
        "name": "get_customer",
        "description": "Get customer name using customer ID",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"}
        }, "required": ["customer_id"], "additionalProperties": False}
    }},
    {"type": "function", "function": {
        "name": "calculate_discount",
        "description": "Calculate discount amount",
        "parameters": {"type": "object", "properties": {
            "price": {"type": "number"},
            "percent": {"type": "number"}
        }, "required": ["price", "percent"], "additionalProperties": False}
    }}
]

def get_order_status(order_id):
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    return orders[order_id]

def get_customer(customer_id):
    if customer_id not in customers:
        raise ValueError("Customer not found")
    return customers[customer_id]

def calculate_discount(price, percent):
    if price < 0 or not 0 <= percent <= 100:
        raise ValueError("Invalid price or discount")
    return price * percent / 100

functions = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount
}


async def tool_chat(messages):
    for _ in range(5):
        response = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content

        async def run_tool(call):
            try:
                args = json.loads(call.function.arguments)
                if call.function.name not in functions:
                    raise ValueError("Unknown tool name")
                result = functions[call.function.name](**args)
            except Exception as e:
                result = {"error": str(e)}

            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            }

        results = await asyncio.gather(
            *(run_tool(c) for c in msg.tool_calls)
        )
        print("Tool calls in one round:", len(results))
        messages.extend(results)

    return "Maximum tool rounds reached"

def count_tokens(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(json.dumps(messages, default=str)))

class Conversation:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    async def chat(self, prompt):
        self.messages.append({"role": "user", "content": prompt})

        # Keep recent history and summarize older turns
        if count_tokens(self.messages) > 3000:
            old = self.messages[1:-6]
            summary = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Summarize important facts briefly."},
                    {"role": "user", "content": json.dumps(old, default=str)}
                ]
            )
            self.messages = [
                self.messages[0],
                {"role": "system", "content":
                 "Previous conversation summary: " +
                 summary.choices[0].message.content}
            ] + self.messages[-6:]

        answer = await tool_chat(self.messages)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

async def main():
    # Tool test
    c = Conversation()
    print(await c.chat("Check order 1001, customer 1 and "
                       "calculate 10 percent discount on 500"))

    # Error tests
    for name, args in [
        ("get_order_status", {"order_id": "9999"}),
        ("get_customer", {"customer_id": "99"}),
        ("calculate_discount", {"price": -100, "percent": 10}),
        ("unknown_tool", {})
    ]:
        try:
            result = functions[name](**args)
        except Exception as e:
            result = {"error": str(e)}
        print("Error test:", result)

   
    memory = Conversation()
    for i in range(15):
        answer = await memory.chat(
            "Remember that my favourite colour is blue."
            if i == 0 else f"Tell me something interesting. Turn {i+1}"
        )
        print(f"Turn {i+1}:", answer)

    print("Final history tokens:", count_tokens(memory.messages))

asyncio.run(main())
