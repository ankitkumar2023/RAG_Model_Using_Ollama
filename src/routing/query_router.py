from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from config.settings import get_settings
from prompts.router_prompts import ROUTE_CLASSIFICATION_PROMPT
from src.api.schemas import ChatMessageSchema
from src.llm.gemini_client import GeminiClient
from src.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)

settings = get_settings()


class RouteType(StrEnum):
    DOCUMENT_RAG = "document_rag"
    WEB_SEARCH = "web_search"
    WEATHER = "weather"
    FINANCE = "finance"
    GENERAL = "general"
    HYBRID = "hybrid"
    TOOL_CALLING = "tool_calling"


@dataclass
class RouteDecision:
    """
    The router's decision for a single query.
    """

    route: RouteType
    confidence: float
    reason: str
    params: dict[str, Any] = field(default_factory=dict)


_WEATHER_PATTERN = re.compile(
    r"\b(weather|temperature|forecast)\b",
    re.IGNORECASE,
)

_WEATHER_LOCATION_PATTERN = re.compile(
    r"(?:weather|temperature|forecast)\s*(?:in|at|for)?\s+"
    r"(?:the\s+)?([A-Za-z][A-Za-z\s]{1,40}?)(?:[?.!]|$)",
    re.IGNORECASE,
)

_PLACEHOLDER_LOCATION_PATTERN = re.compile(
    r"^(my|our|this|that)\s+(location|area|city|place|region|"
    r"current\s+location)$|^(here|there)$",
    re.IGNORECASE,
)

_FINANCE_PATTERN = re.compile(
    r"\b(stock price|share price|ticker|market cap|nasdaq|nyse|"
    r"stock quote)\b",
    re.IGNORECASE,
)

_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")

_ARITHMETIC_HINT_PATTERN = re.compile(r"\d+\s*[+\-*/^]\s*\d+")
_CALC_KEYWORD_PATTERN = re.compile(
    r"\b(calculate|compute)\b",
    re.IGNORECASE,
)

_WEB_SEARCH_PATTERN = re.compile(
    r"\b(latest|today|current|recent|breaking|this week|this year|"
    r"right now|news)\b",
    re.IGNORECASE,
)

_RAG_INTENT_PATTERN = re.compile(
    r"\b(my document|uploaded|the brd|this document|this file|"
    r"the report|the pdf|indexed)\b",
    re.IGNORECASE,
)


class QueryRouter:
    """
    Classifies a user query into a route before any retrieval/tool work
    happens.

    A fast deterministic pass handles the obvious cases at near-zero
    latency/cost; genuinely ambiguous queries fall through to a single
    cheap Gemini call that returns a structured route decision. Any
    query that overlaps with document-intent phrasing skips the
    deterministic weather/finance/calculator shortcuts and defers to the
    LLM, to avoid e.g. "weather" appearing inside a document-related
    question being misrouted to the weather tool.
    """

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def _classify_deterministic(
        self,
        query: str,
        has_indexed_documents: bool,
    ) -> RouteDecision | None:
        rag_intent = bool(_RAG_INTENT_PATTERN.search(query)) and has_indexed_documents
        web_intent = bool(_WEB_SEARCH_PATTERN.search(query))

        if rag_intent and web_intent:
            return RouteDecision(
                route=RouteType.HYBRID,
                confidence=0.85,
                reason=("Query references both indexed documents and current/recent information."),
            )

        if rag_intent:
            # Skip the tool-specific keyword checks below when
            # document-intent is present, to avoid a keyword collision
            # (e.g. a document question that happens to mention
            # "weather").
            return RouteDecision(
                route=RouteType.DOCUMENT_RAG,
                confidence=0.8,
                reason=("Query explicitly references the user's uploaded documents."),
            )

        if _WEATHER_PATTERN.search(query):
            location_match = _WEATHER_LOCATION_PATTERN.search(query)
            location = location_match.group(1).strip() if location_match else ""

            if _PLACEHOLDER_LOCATION_PATTERN.match(location):
                # The regex grabbed a placeholder phrase ("my location",
                # "here", ...) rather than an actual place name -- e.g.
                # "weather in my location? I live in Bokaro Steel City"
                # would otherwise be sent to the weather API literally
                # as "my location" and 404. Treat as unresolved so
                # _gather_weather_context's LLM-based extraction (which
                # can read the rest of the sentence) runs instead.
                location = ""

            return RouteDecision(
                route=RouteType.WEATHER,
                confidence=0.9 if location else 0.6,
                reason="Query asks about weather/temperature/forecast.",
                params={"location": location} if location else {},
            )

        if _FINANCE_PATTERN.search(query):
            ticker_match = _TICKER_PATTERN.search(query)

            return RouteDecision(
                route=RouteType.FINANCE,
                confidence=0.85 if ticker_match else 0.6,
                reason=("Query asks about a stock price/ticker/market data."),
                params=({"symbol": ticker_match.group(1)} if ticker_match else {}),
            )

        if _ARITHMETIC_HINT_PATTERN.search(query) or _CALC_KEYWORD_PATTERN.search(query):
            return RouteDecision(
                route=RouteType.TOOL_CALLING,
                confidence=0.8,
                reason="Query appears to be a calculation.",
                params={"expression": query},
            )

        if web_intent:
            return RouteDecision(
                route=RouteType.WEB_SEARCH,
                confidence=0.75,
                reason="Query asks for current/recent information.",
            )

        return None

    async def _classify_with_llm(
        self,
        query: str,
        has_indexed_documents: bool,
        chat_history: list[ChatMessageSchema] | None,
    ) -> RouteDecision:
        history_text = (
            "\n".join(f"{message.role}: {message.content}" for message in chat_history[-4:])
            if chat_history
            else "(none)"
        )

        document_context = (
            "The user has documents indexed in the knowledge base."
            if has_indexed_documents
            else ("The user has NO documents indexed -- do not select document_rag or hybrid.")
        )

        prompt = ROUTE_CLASSIFICATION_PROMPT.format(
            document_context=document_context,
            history=history_text,
            query=query,
        )

        try:
            result = await self.gemini_client.agenerate(
                prompt=prompt,
                model=settings.primary_model,
                temperature=0.0,
            )

            parsed = ResponseParser.extract_json(result.get("response", ""))

            if not parsed:
                raise ValueError("Router LLM returned no parseable JSON.")

            route = RouteType(parsed["route"])

            if route in (RouteType.DOCUMENT_RAG, RouteType.HYBRID) and not has_indexed_documents:
                route = RouteType.WEB_SEARCH if route == RouteType.HYBRID else RouteType.GENERAL

            return RouteDecision(
                route=route,
                confidence=float(parsed.get("confidence", 0.5)),
                reason=parsed.get("reason", "LLM classification."),
                params=parsed.get("params") or {},
            )

        except Exception:
            logger.exception("LLM route classification failed; defaulting to general.")

            return RouteDecision(
                route=RouteType.GENERAL,
                confidence=0.3,
                reason=("Router classification failed; defaulted to general knowledge."),
            )

    async def classify(
        self,
        query: str,
        has_indexed_documents: bool,
        chat_history: list[ChatMessageSchema] | None = None,
    ) -> RouteDecision:
        """
        Classify a query into a route. Deterministic rules run first;
        the LLM fallback only runs for genuinely ambiguous queries.
        """

        if not settings.enable_query_router:
            return RouteDecision(
                route=(RouteType.DOCUMENT_RAG if has_indexed_documents else RouteType.GENERAL),
                confidence=1.0,
                reason="Query routing is disabled.",
            )

        deterministic = self._classify_deterministic(query, has_indexed_documents)

        if deterministic is not None:
            logger.info(
                "Router (deterministic): route=%s confidence=%.2f query=%s",
                deterministic.route.value,
                deterministic.confidence,
                query,
            )

            return deterministic

        decision = await self._classify_with_llm(query, has_indexed_documents, chat_history)

        logger.info(
            "Router (LLM): route=%s confidence=%.2f query=%s",
            decision.route.value,
            decision.confidence,
            query,
        )

        return decision
