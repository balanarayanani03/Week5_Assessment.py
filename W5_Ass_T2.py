
import os
import time
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, wait_exponential, stop_after_attempt
from tenacity import retry_if_exception
from tenacity import before_sleep_log
import logging

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)


def retry_error(error):
    return (
        isinstance(error, RateLimitError) or
        (isinstance(error, APIStatusError) and error.status_code >= 500)
    )

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(retry_error),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
    reraise=True
)
async def ask(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


async def test_retry():
    print(await ask("What is Python?"))


async def sequential():
    start = time.time()

    for i in range(10):
        await ask(f"Question {i+1}: Explain Python")

    print("Sequential time:", round(time.time()-start, 2), "seconds")

async def parallel():
    semaphore = asyncio.Semaphore(3)

    async def safe_ask(i):
        async with semaphore:
            try:
                return await ask(f"Question {i}: Explain Python")
            except Exception as e:
                print(f"Question {i} failed:", e)
                return None

    start = time.time()

    results = await asyncio.gather(
        *(safe_ask(i) for i in range(1, 11))
    )

    print("Parallel time:", round(time.time()-start, 2), "seconds")
    print("Successful answers:", sum(r is not None for r in results))


async def main():
    await test_retry()
    await sequential()
    await parallel()

asyncio.run(main())
