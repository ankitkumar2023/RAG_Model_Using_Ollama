"""
Minimal example: a single tool call fed back into Gemini for a final
answer. Requires GEMINI_API_KEY to be set (see .env.example).

Run with: python -m src.examples.agent_demo
"""

from __future__ import annotations

import asyncio

from prompts.system_prompts import SYSTEM_PROMPT
from src.llm.gemini_client import GeminiClient
from src.tools.calculator_tool import CalculatorTool


async def main() -> None:
    client = GeminiClient()
    calculator = CalculatorTool()

    user_query = "What is sqrt(225) + 100?"

    calc_result = await calculator.execute("sqrt(225) + 100")

    prompt = f"""
User Query:
{user_query}

Tool Result:
{calc_result.result}

Generate final answer.
"""

    response = await client.agenerate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    print("\nAgent Response:\n")
    print(response.get("response", ""))


if __name__ == "__main__":
    asyncio.run(main())
