from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from config.settings import get_settings
from prompts.router_prompts import TOOL_LOOP_SYSTEM_PROMPT
from prompts.system_prompts import RAG_SYSTEM_PROMPT, SYSTEM_PROMPT
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    HealthResponse,
    SourceCitation,
    ToolInvocationRequest,
    ToolInvocationResponse,
)
from src.llm.gemini_client import GeminiClient
from src.llm.response_parser import ResponseParser
from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retrieval import RetrievedDocument, Retriever
from src.rag.vector_store import VectorStore
from src.routing.query_router import QueryRouter, RouteDecision, RouteType
from src.security.prompt_guard import screen_prompt
from src.tools.calculator_tool import CalculatorError, CalculatorTool
from src.tools.stock_tool import StockTool, StockToolError
from src.tools.weather_tool import WeatherTool, WeatherToolError
from src.tools.web_search_tool import WebSearchError, WebSearchTool

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class _GatherResult:
    """
    Everything decided/gathered before generation -- shared between the
    streaming and non-streaming chat paths.
    """

    route: RouteType
    route_reason: str
    route_confidence: float
    corrective_fallback: bool
    context: str
    retrieved_documents: list[RetrievedDocument]
    web_results_used: int
    tools_used: list[str]
    router_latency_ms: float
    retrieval_latency_ms: float
    # Set only for routes (tool_calling) that produce a complete answer
    # up front rather than something to stream token-by-token.
    precomputed_answer: str | None = None
    precomputed_latency_ms: float = 0.0


class APIRouter:
    """
    Core orchestration routes.
    """

    def __init__(self) -> None:
        self.client = GeminiClient()

        self.vector_store = VectorStore()

        self.embedding_generator = EmbeddingGenerator()

        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            gemini_client=self.client,
        )

        self.query_router = QueryRouter(gemini_client=self.client)

        self.chunker = TextChunker()

        self.calculator_tool = CalculatorTool()

        self.weather_tool = WeatherTool()

        self.stock_tool = StockTool()

        self.web_search_tool = WebSearchTool()

    async def health_check(
        self,
    ) -> HealthResponse:
        """
        Application health route.
        """

        gemini_ok = self.client.health_check()

        vector_count = await self.vector_store.count()

        return HealthResponse(
            status="healthy",
            gemini_connected=gemini_ok,
            vector_store_documents=vector_count,
        )

    async def upload_document(
        self,
        request: DocumentUploadRequest,
    ) -> DocumentUploadResponse:
        """
        Document ingestion pipeline.

        Chunking runs per-page (rather than over one flattened blob) so
        every chunk can carry an accurate page number for citations.
        Non-paginated formats simply pass a single-element `pages` list.
        """

        document_id = str(uuid.uuid4())

        chunks = []
        page_numbers = []

        for page_number, page_text in enumerate(request.pages, start=1):
            if not page_text.strip():
                continue

            page_chunks = self.chunker.split_text(
                text=page_text,
                metadata={
                    "source": request.filename,
                    "page": page_number,
                    **request.metadata,
                },
            )

            chunks.extend(page_chunks)
            page_numbers.extend([page_number] * len(page_chunks))

        if not chunks:
            return DocumentUploadResponse(
                success=False,
                chunks_created=0,
                document_id=document_id,
            )

        texts = [chunk.text for chunk in chunks]

        embeddings = await self.embedding_generator.embed_batch(texts)

        ids = [
            f"{document_id}_p{page_number}_{chunk.chunk_id}"
            for page_number, chunk in zip(page_numbers, chunks, strict=True)
        ]

        metadatas = [
            {
                "document_id": document_id,
                "source": request.filename,
                "chunk_id": chunk.chunk_id,
                "chunk_uid": chunk_uid,
                "page": page_number,
            }
            for page_number, chunk, chunk_uid in zip(page_numbers, chunks, ids, strict=True)
        ]

        await self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("Document uploaded successfully.")

        return DocumentUploadResponse(
            success=True,
            chunks_created=len(chunks),
            document_id=document_id,
        )

    async def invoke_tool(
        self,
        request: ToolInvocationRequest,
    ) -> ToolInvocationResponse:
        """
        Execute registered tools.
        """

        start = time.perf_counter()

        tool_map = {
            "calculator": self.calculator_tool,
            "weather": self.weather_tool,
            "stock": self.stock_tool,
            "web_search": self.web_search_tool,
        }

        tool = tool_map.get(request.tool_name)

        if not tool:
            raise ValueError(f"Unknown tool: {request.tool_name}")

        result = await tool.execute(**request.parameters)

        execution_time = time.perf_counter() - start

        return ToolInvocationResponse(
            tool_name=request.tool_name,
            success=True,
            result=result.__dict__,
            execution_time=execution_time,
        )

    # ==========================================
    # ROUTE CONTEXT GATHERING
    # ==========================================

    async def _extract_param(
        self,
        query: str,
        description: str,
    ) -> str:
        """
        Small fallback extraction call for when the router's fast
        deterministic pass couldn't pull a clean param (e.g. a location
        or ticker) out of the query.
        """

        try:
            result = await self.client.agenerate(
                prompt=(
                    f"Extract ONLY the {description} from this query. "
                    f"Respond with just the value, nothing else.\n\n"
                    f"Query: {query}"
                ),
                model=settings.primary_model,
                temperature=0.0,
            )

            return result.get("response", "").strip()

        except Exception:
            logger.exception("Param extraction failed for '%s'.", description)
            return ""

    async def _gather_document_rag_context(
        self,
        request: ChatRequest,
    ) -> tuple[str, list[RetrievedDocument], float]:
        start = time.perf_counter()

        try:
            docs = await self.retriever.retrieve(
                query=request.query,
                chat_history=request.history,
                top_k=settings.top_k_results,
            )

        except Exception:
            logger.exception("RAG retrieval failed.")
            docs = []

        latency_ms = (time.perf_counter() - start) * 1000

        context = self.retriever.build_context(docs)

        return context, docs, latency_ms

    async def _gather_web_context(
        self,
        query: str,
    ) -> tuple[str, int, float]:
        start = time.perf_counter()

        try:
            results = await self.web_search_tool.execute(query)

        except WebSearchError:
            logger.exception("Web search failed.")
            results = []

        latency_ms = (time.perf_counter() - start) * 1000

        if not results:
            return "", 0, latency_ms

        blocks = [
            f"[Web Result {idx}]\nTitle: {result.title}\nSource: {result.link}\n\n{result.snippet}"
            for idx, result in enumerate(results, start=1)
        ]

        return "\n\n".join(blocks), len(results), latency_ms

    async def _gather_weather_context(
        self,
        query: str,
        params: dict,
    ) -> tuple[str, float]:
        start = time.perf_counter()

        location = params.get("location") or await self._extract_param(query, "city or location name")

        if not location:
            return "", (time.perf_counter() - start) * 1000

        try:
            result = await self.weather_tool.execute(location)

            context = (
                f"Current weather in {result.location}: "
                f"{result.description}, "
                f"{result.temperature_celsius}°C, "
                f"humidity {result.humidity}%, "
                f"wind {result.wind_speed} m/s."
            )

        except WeatherToolError:
            logger.exception("Weather lookup failed.")
            context = ""

        return context, (time.perf_counter() - start) * 1000

    async def _gather_finance_context(
        self,
        query: str,
        params: dict,
    ) -> tuple[str, float]:
        start = time.perf_counter()

        symbol = params.get("symbol") or await self._extract_param(query, "stock ticker symbol")

        if not symbol:
            return "", (time.perf_counter() - start) * 1000

        try:
            result = await self.stock_tool.execute(symbol.strip().upper())

            context = (
                f"{result.symbol}: price ${result.price}, "
                f"change {result.change_percent}, "
                f"volume {result.volume}."
            )

        except StockToolError:
            logger.exception("Stock lookup failed.")
            context = ""

        return context, (time.perf_counter() - start) * 1000

    async def _run_tool_loop(
        self,
        query: str,
    ) -> tuple[str, list[str], float]:
        """
        Bounded ReAct-style loop: Gemini either answers directly or emits
        a `<tool>{json}</tool>` call, the tool executes, the result is
        fed back, and the loop repeats up to `max_tool_iterations` times.
        """

        start = time.perf_counter()

        tools_used: list[str] = []

        transcript = f"{TOOL_LOOP_SYSTEM_PROMPT}\n\nUser: {query}"

        for _ in range(settings.max_tool_iterations):
            try:
                response = await self.client.agenerate(
                    prompt=transcript,
                    model=settings.primary_model,
                    temperature=0.0,
                )

            except Exception:
                logger.exception("Tool loop generation failed.")
                break

            raw = response.get("response", "")
            tool_calls = ResponseParser.extract_tool_calls(raw)

            if not tool_calls:
                return (
                    ResponseParser.clean_response(raw),
                    tools_used,
                    (time.perf_counter() - start) * 1000,
                )

            call = tool_calls[0]
            tool_name = call.get("tool", "")
            tool_input = call.get("input", {}) or {}

            if tool_name == "calculator":
                try:
                    result = await self.calculator_tool.execute(tool_input.get("expression", ""))
                    tool_result_text = f"Result: {result.result}"

                except CalculatorError as exc:
                    tool_result_text = f"Error: {exc}"

                tools_used.append("calculator")

            else:
                tool_result_text = f"Unknown tool '{tool_name}'."

            transcript += (
                f"\n\nAssistant: {raw}\n\n"
                f"Tool Result ({tool_name}): {tool_result_text}\n\n"
                f"Now provide the final answer using this result, in "
                f"plain text (do not call another tool unless truly "
                f"necessary)."
            )

        return (
            ("I wasn't able to complete this request within the allowed number of steps."),
            tools_used,
            (time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _build_prompt(
        route: RouteType,
        query: str,
        context: str,
    ) -> str:
        if route in (RouteType.DOCUMENT_RAG, RouteType.HYBRID):
            return f""" {RAG_SYSTEM_PROMPT} =========================================================== RETRIEVED CONTEXT =========================================================== {context} =========================================================== USER QUESTION =========================================================== {query} =========================================================== RESPONSE GENERATION RULES =========================================================== You are an enterprise-grade AI assistant. Your task is to answer the user's question using ONLY the retrieved context above. IMPORTANT RULES: 1. The answer MUST be grounded in the retrieved context. 2. DO NOT hallucinate or invent information. 3. If the answer exists in the retrieved context: - provide a polished natural-language response - make the response professional and human-friendly - include contextual wording when appropriate - preserve all factual values exactly 4. DO NOT respond with: - raw numbers only - bullet fragments - incomplete phrases 5. Prefer complete sentences. 6. Keep answers concise BUT informative. 7. If the retrieved context mentions: - institution name - course name - organization - program details then naturally incorporate them into the answer. 8. If the answer does NOT exist in the context, respond EXACTLY with: "The uploaded documents do not contain this information." =========================================================== FINAL ANSWER =========================================================== """

        if route == RouteType.WEB_SEARCH:
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"Web search results:\n{context}\n\n"
                f"User question: {query}\n\n"
                f"Answer using the web results above. Mention that the "
                f"information comes from current web sources. If the "
                f"results don't answer the question, say so plainly."
            )

        if route in (RouteType.WEATHER, RouteType.FINANCE):
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"Live data:\n{context}\n\n"
                f"User question: {query}\n\n"
                f"Answer naturally and conversationally using the data "
                f"above. Do not mention 'context' or 'documents'."
            )

        return f"{SYSTEM_PROMPT}\n\nUser question: {query}\n\nAnswer using your own knowledge."

    async def _route_and_gather(
        self,
        request: ChatRequest,
    ) -> _GatherResult:
        """
        Route the query, then gather whatever context/tool output that
        route needs -- shared by both `process_chat` and `stream_chat`.
        """

        vector_count = await self.vector_store.count()
        has_indexed_documents = vector_count > 0

        router_start = time.perf_counter()

        decision: RouteDecision = await self.query_router.classify(
            query=request.query,
            has_indexed_documents=has_indexed_documents and request.use_rag,
            chat_history=request.history,
        )

        if not request.enable_tools and decision.route in (
            RouteType.WEATHER,
            RouteType.FINANCE,
            RouteType.TOOL_CALLING,
        ):
            decision = RouteDecision(
                route=RouteType.GENERAL,
                confidence=1.0,
                reason="Tools disabled; answering from general knowledge.",
            )

        router_latency_ms = (time.perf_counter() - router_start) * 1000

        route = decision.route
        route_reason = decision.reason
        route_confidence = decision.confidence
        corrective_fallback = False

        logger.info(
            "Routed query to '%s' (confidence=%.2f): %s",
            route.value,
            route_confidence,
            route_reason,
        )

        if route == RouteType.TOOL_CALLING:
            answer, tools_used, tool_latency_ms = await self._run_tool_loop(request.query)

            return _GatherResult(
                route=route,
                route_reason=route_reason,
                route_confidence=route_confidence,
                corrective_fallback=False,
                context="",
                retrieved_documents=[],
                web_results_used=0,
                tools_used=tools_used,
                router_latency_ms=router_latency_ms,
                retrieval_latency_ms=0.0,
                precomputed_answer=answer,
                precomputed_latency_ms=tool_latency_ms,
            )

        retrieved_documents: list[RetrievedDocument] = []
        context = ""
        retrieval_latency_ms = 0.0
        web_results_used = 0

        if route == RouteType.DOCUMENT_RAG:
            context, retrieved_documents, retrieval_latency_ms = await self._gather_document_rag_context(
                request
            )

            retrieval_confidence = (
                sum(doc.score for doc in retrieved_documents) / len(retrieved_documents)
                if retrieved_documents
                else 0.0
            )

            if retrieval_confidence < settings.corrective_rag_threshold:
                logger.info(
                    "Retrieval confidence %.2f below corrective threshold %.2f; falling back to web search.",
                    retrieval_confidence,
                    settings.corrective_rag_threshold,
                )

                web_context, web_results_used, web_latency_ms = await self._gather_web_context(request.query)

                retrieval_latency_ms += web_latency_ms
                corrective_fallback = True
                route_reason = (
                    f"{route_reason} Retrieval confidence "
                    f"({retrieval_confidence:.2f}) was below the "
                    f"corrective threshold, so web search results were "
                    f"added."
                )

                context = "\n\n".join(part for part in (context, web_context) if part)

        elif route == RouteType.HYBRID:
            (rag_result, web_result) = await asyncio.gather(
                self._gather_document_rag_context(request),
                self._gather_web_context(request.query),
            )

            rag_context, retrieved_documents, retrieval_latency_ms = rag_result
            web_context, web_results_used, web_latency_ms = web_result

            retrieval_latency_ms += web_latency_ms
            context = "\n\n".join(part for part in (rag_context, web_context) if part)

        elif route == RouteType.WEB_SEARCH:
            context, web_results_used, retrieval_latency_ms = await self._gather_web_context(request.query)

        elif route == RouteType.WEATHER:
            context, retrieval_latency_ms = await self._gather_weather_context(request.query, decision.params)

        elif route == RouteType.FINANCE:
            context, retrieval_latency_ms = await self._gather_finance_context(request.query, decision.params)

        # RouteType.GENERAL: no context needed.

        return _GatherResult(
            route=route,
            route_reason=route_reason,
            route_confidence=route_confidence,
            corrective_fallback=corrective_fallback,
            context=context,
            retrieved_documents=retrieved_documents,
            web_results_used=web_results_used,
            tools_used=[],
            router_latency_ms=router_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    async def process_chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Full conversational orchestration (non-streaming): route ->
        gather context for that route -> generate -> score/cite.
        """

        total_start = time.perf_counter()

        guard_result = screen_prompt(request.query)

        if not guard_result.is_safe:
            logger.info(
                "Input prompt guard blocked query: patterns=%s",
                guard_result.matched_patterns,
            )

            return ChatResponse(
                response="Request blocked by safety guardrails.",
                model=settings.primary_model,
                guardrail_passed=False,
                tools_used=[],
                retrieved_documents=0,
                retrieved_context="",
            )

        gather = await self._route_and_gather(request)

        if gather.precomputed_answer is not None:
            return await self._finalize_response(
                cleaned_response=gather.precomputed_answer,
                route=gather.route,
                route_reason=gather.route_reason,
                route_confidence=gather.route_confidence,
                corrective_fallback=False,
                tools_used=gather.tools_used,
                retrieved_documents=[],
                context="",
                web_results_used=0,
                router_latency_ms=gather.router_latency_ms,
                retrieval_latency_ms=0.0,
                generation_latency_ms=gather.precomputed_latency_ms,
                total_start=total_start,
            )

        prompt_route = RouteType.HYBRID if gather.corrective_fallback else gather.route
        final_prompt = self._build_prompt(prompt_route, request.query, gather.context)

        generation_start = time.perf_counter()

        try:
            response = await self.client.agenerate(
                prompt=final_prompt,
                model=settings.primary_model,
            )

            raw_response = response.get("response", "")

        except Exception:
            logger.exception("LLM generation failed.")

            return ChatResponse(
                response=("Generation failed due to internal system error."),
                model=settings.primary_model,
                guardrail_passed=False,
                tools_used=gather.tools_used,
                retrieved_documents=len(gather.retrieved_documents),
                retrieved_context=gather.context,
                route=gather.route.value,
                route_reason=gather.route_reason,
                route_confidence=gather.route_confidence,
            )

        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        cleaned_response = ResponseParser.clean_response(raw_response)

        if not cleaned_response:
            logger.warning("Model returned empty output; treating as safety-blocked.")

            cleaned_response = "Response blocked by output safety guardrails."

        return await self._finalize_response(
            cleaned_response=cleaned_response,
            route=gather.route,
            route_reason=gather.route_reason,
            route_confidence=gather.route_confidence,
            corrective_fallback=gather.corrective_fallback,
            tools_used=gather.tools_used,
            retrieved_documents=gather.retrieved_documents,
            context=gather.context,
            web_results_used=gather.web_results_used,
            router_latency_ms=gather.router_latency_ms,
            retrieval_latency_ms=gather.retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_start=total_start,
        )

    async def stream_chat(
        self,
        request: ChatRequest,
    ) -> tuple[
        AsyncIterator[str] | None,
        _GatherResult | None,
        float,
        ChatResponse | None,
    ]:
        """
        Streaming counterpart to `process_chat`.

        Returns `(token_generator, gather, total_start, immediate)`.
        `immediate` is set (and `token_generator`/`gather` are None) when
        there's nothing to stream -- the input was blocked, or the route
        (tool_calling) already produced a complete answer up front.
        Otherwise the caller drains `token_generator` (e.g. via
        `st.write_stream`) and then calls `finalize_stream(...)` with the
        accumulated text to get the final `ChatResponse`.
        """

        total_start = time.perf_counter()

        guard_result = screen_prompt(request.query)

        if not guard_result.is_safe:
            logger.info(
                "Input prompt guard blocked query: patterns=%s",
                guard_result.matched_patterns,
            )

            immediate = ChatResponse(
                response="Request blocked by safety guardrails.",
                model=settings.primary_model,
                guardrail_passed=False,
                tools_used=[],
                retrieved_documents=0,
                retrieved_context="",
            )

            return None, None, total_start, immediate

        gather = await self._route_and_gather(request)

        if gather.precomputed_answer is not None:
            immediate = await self._finalize_response(
                cleaned_response=gather.precomputed_answer,
                route=gather.route,
                route_reason=gather.route_reason,
                route_confidence=gather.route_confidence,
                corrective_fallback=False,
                tools_used=gather.tools_used,
                retrieved_documents=[],
                context="",
                web_results_used=0,
                router_latency_ms=gather.router_latency_ms,
                retrieval_latency_ms=0.0,
                generation_latency_ms=gather.precomputed_latency_ms,
                total_start=total_start,
            )

            return None, None, total_start, immediate

        prompt_route = RouteType.HYBRID if gather.corrective_fallback else gather.route
        final_prompt = self._build_prompt(prompt_route, request.query, gather.context)

        async def _generator() -> AsyncIterator[str]:
            async for chunk in self.client.astream(final_prompt, model=settings.primary_model):
                yield chunk

        return _generator(), gather, total_start, None

    async def finalize_stream(
        self,
        gather: _GatherResult,
        accumulated_text: str,
        generation_latency_ms: float,
        total_start: float,
    ) -> ChatResponse:
        """
        Called once the UI has fully drained the token generator from
        `stream_chat`, with the accumulated text -- computes citations,
        confidence, and assembles the final `ChatResponse`.
        """

        cleaned_response = ResponseParser.clean_response(accumulated_text)

        if not cleaned_response:
            cleaned_response = "Response blocked by output safety guardrails."

        return await self._finalize_response(
            cleaned_response=cleaned_response,
            route=gather.route,
            route_reason=gather.route_reason,
            route_confidence=gather.route_confidence,
            corrective_fallback=gather.corrective_fallback,
            tools_used=gather.tools_used,
            retrieved_documents=gather.retrieved_documents,
            context=gather.context,
            web_results_used=gather.web_results_used,
            router_latency_ms=gather.router_latency_ms,
            retrieval_latency_ms=gather.retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_start=total_start,
        )

    async def _finalize_response(
        self,
        *,
        cleaned_response: str,
        route: RouteType,
        route_reason: str,
        route_confidence: float,
        corrective_fallback: bool,
        tools_used: list[str],
        retrieved_documents: list[RetrievedDocument],
        context: str,
        web_results_used: int,
        router_latency_ms: float,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_start: float,
    ) -> ChatResponse:
        """
        Shared tail: citations, confidence scoring, latency, response
        assembly -- used by every route.
        """

        sources = [
            SourceCitation(
                filename=doc.source,
                page=doc.page,
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=doc.score,
            )
            for doc in retrieved_documents
        ]

        confidence_score = 0.0

        if retrieved_documents:
            try:
                retrieval_component = sum(doc.score for doc in retrieved_documents) / len(retrieved_documents)

                grounding_component = await self.retriever.compute_grounding_score(
                    cleaned_response,
                    retrieved_documents,
                )

                confidence_score = round(
                    0.5 * retrieval_component + 0.5 * grounding_component,
                    3,
                )

            except Exception:
                logger.exception("Confidence scoring failed; defaulting to 0.0.")

        total_latency_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Chat orchestration completed. route=%s confidence=%.3f sources=%s total_latency_ms=%.0f",
            route.value,
            confidence_score,
            len(sources),
            total_latency_ms,
        )

        return ChatResponse(
            response=cleaned_response,
            model=settings.primary_model,
            guardrail_passed=True,
            tools_used=tools_used,
            retrieved_documents=len(retrieved_documents),
            retrieved_context=context,
            sources=sources,
            confidence_score=confidence_score,
            route=route.value,
            route_reason=route_reason,
            route_confidence=route_confidence,
            corrective_fallback=corrective_fallback,
            web_results_used=web_results_used,
            reranked=(settings.enable_reranking and bool(retrieved_documents)),
            router_latency_ms=router_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
        )
