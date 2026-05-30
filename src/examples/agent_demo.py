from __future__ import annotations

import asyncio

from prompts.system_prompts import (
    SYSTEM_PROMPT,
)
from src.llm.ollama_client import (
    OllamaClient,
)
from src.tools.calculator_tool import (
    CalculatorTool,
)


async def main() -> None:
    client = OllamaClient()

    calculator = CalculatorTool()

    user_query = (
        "What is sqrt(225) + 100?"
    )

    calc_result = (
        await calculator.execute(
            "sqrt(225) + 100"
        )
    )

    prompt = f"""
User Query:
{user_query}

Tool Result:
{calc_result.result}

Generate final answer.
"""

    response = await client.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    print("\nAgent Response:\n")

    print(
        response.get("response", "")
    )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())