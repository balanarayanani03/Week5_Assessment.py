
import os
import time
import json
import tiktoken

from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()
LIMIT = 4000

def count_tokens(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(m["content"])) for m in messages)

def log_usage(model, usage):
    cost = (
        usage.prompt_tokens * 0.15 +
        usage.completion_tokens * 0.60
    ) / 1_000_000

    data = {
        "date": str(date.today()),
        "tokens": usage.total_tokens,
        "cost": cost
    }

    with open("usage.log", "a") as f:
        f.write(json.dumps(data) + "\n")

    return cost

def ask_openai(prompt):
    msg = [{"role": "user", "content": prompt}]

    if count_tokens(msg) > LIMIT:
        raise ValueError("Token limit exceeded")

    r = openai.chat.completions.create(
        model="gpt-4o-mini", messages=msg
    )

    print("Answer:", r.choices[0].message.content)
    print("Model:", r.model)
    print("Tokens:", r.usage.total_tokens)
    print("Cost:", log_usage(r.model, r.usage))

def ask_claude(prompt, system):
    msg = [{"role": "user", "content": prompt}]

    if count_tokens(msg) > LIMIT:
        raise ValueError("Token limit exceeded")

    r = claude.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=500,
        system=system,
        messages=msg
    )

    print("Answer:", r.content[0].text)
    print("Model:", r.model)
    print("Tokens:", r.usage.input_tokens + r.usage.output_tokens)

    with open("usage.log", "a") as f:
        f.write(json.dumps({
            "date": str(date.today()),
            "tokens": r.usage.input_tokens + r.usage.output_tokens,
            "cost": 0
        }) + "\n")

@app.post("/chat/stream")
def chat_stream(data: dict):
    msg = [{"role": "user", "content": data["prompt"]}]

    if count_tokens(msg) > LIMIT:
        raise HTTPException(400, "Token limit exceeded")

    def generate():
        start = time.time()

        stream = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            stream=True
        )

        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                print("Elapsed:", round(time.time()-start, 2))
                yield f"data: {json.dumps({'text': text})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/usage")
def usage():
    tokens = 0
    cost = 0

    if os.path.exists("usage.log"):
        for line in open("usage.log"):
            d = json.loads(line)
            if d["date"] == str(date.today()):
                tokens += d["tokens"]
                cost += d["cost"]

    return {"total_tokens": tokens, "total_cost": round(cost, 6)}
