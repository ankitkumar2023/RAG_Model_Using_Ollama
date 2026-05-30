from __future__ import annotations

import asyncio

from prompts.system_prompts import SYSTEM_PROMPT
from src.llm.ollama_client import OllamaClient


async def main() -> None:
    client = OllamaClient()

    prompt = """
Explain Retrieval-Augmented Generation in simple terms.
"""

    print("\nStreaming Response:\n")

    async for token in client.stream_generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    ):
        print(token, end="", flush=True)

    print("\n")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())