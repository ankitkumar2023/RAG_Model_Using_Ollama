from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.llm.gemini_client import GeminiClient, GeminiClientError
from src.llm.response_parser import ResponseParser


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.llm import gemini_client as gemini_client_module

    monkeypatch.setattr(
        gemini_client_module.settings,
        "gemini_api_key",
        SecretStr("dummy-test-key"),
    )


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.llm import gemini_client as gemini_client_module

    monkeypatch.setattr(gemini_client_module.settings, "gemini_api_key", None)

    with pytest.raises(GeminiClientError):
        GeminiClient()


def test_health_check_reflects_key_presence() -> None:
    client = GeminiClient()

    assert client.health_check() is True


@pytest.mark.asyncio
async def test_agenerate_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient()

    fake_llm = MagicMock()
    fake_llm.model = "gemini-flash-latest"
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(text="Hello from Gemini"))

    monkeypatch.setattr(client, "_get_llm", lambda **kwargs: fake_llm)

    result = await client.agenerate(prompt="hi")

    assert result["response"] == "Hello from Gemini"
    assert result["model"] == "gemini-flash-latest"


@pytest.mark.asyncio
async def test_astream_yields_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient()

    fake_chunks = [
        MagicMock(text="Hel"),
        MagicMock(text="lo"),
    ]

    async def _fake_astream(messages):
        for chunk in fake_chunks:
            yield chunk

    fake_llm = MagicMock()
    fake_llm.astream = _fake_astream

    monkeypatch.setattr(client, "_get_llm", lambda **kwargs: fake_llm)

    collected = [chunk async for chunk in client.astream(prompt="hi")]

    assert collected == ["Hel", "lo"]


def test_response_cleaning() -> None:
    raw = """
<think>
hidden reasoning
</think>

Hello world
"""

    cleaned = ResponseParser.clean_response(raw)

    assert "hidden reasoning" not in cleaned
    assert "Hello world" in cleaned


def test_extract_json() -> None:
    text = 'Some prefix {"name": "ankit"} some suffix'

    result = ResponseParser.extract_json(text)

    assert result is not None
    assert result["name"] == "ankit"
