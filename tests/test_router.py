from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.routing.query_router import QueryRouter, RouteType


def _make_router() -> tuple[QueryRouter, MagicMock]:
    gemini_client = MagicMock()
    gemini_client.agenerate = AsyncMock(
        return_value={"response": '{"route": "general", "confidence": 0.5, "reason": "test", "params": {}}'}
    )

    return QueryRouter(gemini_client=gemini_client), gemini_client


@pytest.mark.asyncio
async def test_weather_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What is the weather in London?",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.WEATHER
    assert decision.params.get("location", "").lower() == "london"
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_weather_query_with_placeholder_location_leaves_params_empty() -> None:
    """
    Regression test: "weather in my location? I live in Bokaro Steel
    City..." was extracting the literal phrase "my location" as the
    place name and sending that straight to the weather API (404).
    Placeholder phrases must resolve to no params, so
    _gather_weather_context's LLM-based fallback (which can read the
    rest of the sentence) runs instead.
    """

    router, _ = _make_router()

    decision = await router.classify(
        "what is the current weather in my location ? i lives in bokaro steel city ,jharkhand ,india.",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.WEATHER
    assert decision.params == {}


@pytest.mark.asyncio
async def test_finance_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What is the stock price of AAPL?",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.FINANCE
    assert decision.params.get("symbol") == "AAPL"
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_calculator_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What is 12 * 8?",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.TOOL_CALLING
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_web_search_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What is the latest news on OpenAI?",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.WEB_SEARCH
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_document_rag_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What does my uploaded document say about pricing?",
        has_indexed_documents=True,
    )

    assert decision.route == RouteType.DOCUMENT_RAG
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_document_rag_without_indexed_docs_falls_through_to_llm() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "What does my uploaded document say about pricing?",
        has_indexed_documents=False,
    )

    # rag_intent requires has_indexed_documents=True to match
    # deterministically, so this should fall through to the LLM.
    gemini_client.agenerate.assert_called_once()
    assert decision.route == RouteType.GENERAL


@pytest.mark.asyncio
async def test_hybrid_query_routes_without_llm_call() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "Compare my uploaded BRD with the latest AI regulations",
        has_indexed_documents=True,
    )

    assert decision.route == RouteType.HYBRID
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_ambiguous_query_falls_through_to_llm() -> None:
    router, gemini_client = _make_router()

    decision = await router.classify(
        "Explain how neural networks work.",
        has_indexed_documents=False,
    )

    gemini_client.agenerate.assert_called_once()
    assert decision.route == RouteType.GENERAL


@pytest.mark.asyncio
async def test_llm_fallback_parses_structured_response() -> None:
    gemini_client = MagicMock()
    gemini_client.agenerate = AsyncMock(
        return_value={
            "response": json.dumps(
                {
                    "route": "web_search",
                    "confidence": 0.77,
                    "reason": "needs current info",
                    "params": {},
                }
            )
        }
    )

    router = QueryRouter(gemini_client=gemini_client)

    decision = await router.classify(
        "Explain how neural networks work.",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.WEB_SEARCH
    assert decision.confidence == pytest.approx(0.77)


@pytest.mark.asyncio
async def test_llm_fallback_never_selects_rag_without_indexed_docs() -> None:
    gemini_client = MagicMock()
    gemini_client.agenerate = AsyncMock(
        return_value={
            "response": json.dumps(
                {
                    "route": "document_rag",
                    "confidence": 0.9,
                    "reason": "looks like a document question",
                    "params": {},
                }
            )
        }
    )

    router = QueryRouter(gemini_client=gemini_client)

    decision = await router.classify(
        "Explain how neural networks work.",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.GENERAL


@pytest.mark.asyncio
async def test_llm_failure_defaults_to_general() -> None:
    gemini_client = MagicMock()
    gemini_client.agenerate = AsyncMock(side_effect=RuntimeError("boom"))

    router = QueryRouter(gemini_client=gemini_client)

    decision = await router.classify(
        "Explain how neural networks work.",
        has_indexed_documents=False,
    )

    assert decision.route == RouteType.GENERAL
    assert decision.confidence < 0.5


@pytest.mark.asyncio
async def test_router_disabled_defaults_to_document_rag_when_docs_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routing import query_router as query_router_module

    monkeypatch.setattr(query_router_module.settings, "enable_query_router", False)

    router, gemini_client = _make_router()

    decision = await router.classify(
        "anything",
        has_indexed_documents=True,
    )

    assert decision.route == RouteType.DOCUMENT_RAG
    gemini_client.agenerate.assert_not_called()
