"""
Minimal example: token-by-token streaming from Gemini. Requires
GEMINI_API_KEY to be set (see .env.example).

Run with: python -m src.examples.streaming_demo
"""

from __future__ import annotations

import asyncio

from prompts.system_prompts import SYSTEM_PROMPT
from src.llm.gemini_client import GeminiClient


async def main() -> None:
    client = GeminiClient()

    prompt = """
Explain Retrieval-Augmented Generation in simple terms.
"""

    print("\nStreaming Response:\n")

    async for token in client.astream(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    ):
        print(token, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
