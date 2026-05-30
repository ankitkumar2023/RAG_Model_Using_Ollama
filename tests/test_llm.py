from __future__ import annotations

import pytest

from src.llm.ollama_client import (
    OllamaClient,
)
from src.llm.response_parser import (
    ResponseParser,
)


@pytest.mark.asyncio
async def test_ollama_health() -> None:
    client = OllamaClient()

    healthy = await client.health_check()

    assert isinstance(healthy, bool)

    await client.close()


def test_response_cleaning() -> None:
    raw = """
<think>
hidden reasoning
</think>

Hello world
"""

    cleaned = (
        ResponseParser.clean_response(
            raw
        )
    )

    assert cleaned == "Hello world"


def test_json_extraction() -> None:
    text = """
Some response

{
    "name": "ankit"
}
"""

    result = (
        ResponseParser.extract_json(
            text
        )
    )

    assert result is not None

    assert result["name"] == "ankit"