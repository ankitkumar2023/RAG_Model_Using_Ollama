
from __future__ import annotations

import logging
import time
import uuid

from config.settings import get_settings
from prompts.system_prompts import (
    GUARDRAIL_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    HealthResponse,
    ToolInvocationRequest,
    ToolInvocationResponse,
)
from src.llm.conversation import Conversation
from src.llm.ollama_client import OllamaClient
from src.llm.response_parser import ResponseParser
from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retrieval import Retriever
from src.rag.vector_store import VectorStore
from src.tools.calculator_tool import CalculatorTool
from src.tools.stock_tool import StockTool
from src.tools.weather_tool import WeatherTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)

settings = get_settings()


class APIRouter:
    """
    Core orchestration routes.
    """

    def __init__(self) -> None:
        self.client = OllamaClient()

        self.vector_store = VectorStore()

        self.embedding_generator = (
            EmbeddingGenerator(self.client)
        )

        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
        )

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

        ollama_ok = (
            await self.client.health_check()
        )

        vector_count = (
            await self.vector_store.count()
        )

        return HealthResponse(
            status="healthy",
            ollama_connected=ollama_ok,
            vector_store_documents=vector_count,
        )

    async def upload_document(
        self,
        request: DocumentUploadRequest,
    ) -> DocumentUploadResponse:
        """
        Document ingestion pipeline.
        """

        chunks = self.chunker.split_text(
            text=request.content,
            metadata={
                "source": request.filename,
                **request.metadata,
            },
        )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = (
            await self.embedding_generator.embed_batch(
                texts
            )
        )

        document_id = str(uuid.uuid4())

        metadatas = [
            {
                "document_id": document_id,
                "source": request.filename,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        ids = [
            f"{document_id}_{chunk.chunk_id}"
            for chunk in chunks
        ]

        await self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(
            "Document uploaded successfully."
        )

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

        tool = tool_map.get(
            request.tool_name
        )

        if not tool:
            raise ValueError(
                f"Unknown tool: {request.tool_name}"
            )

        result = await tool.execute(
            **request.parameters
        )

        execution_time = (
            time.perf_counter() - start
        )

        return ToolInvocationResponse(
            tool_name=request.tool_name,
            success=True,
            result=result.__dict__,
            execution_time=execution_time,
        )

    async def process_chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Full conversational orchestration.
        """

        # ==========================================
        # INPUT GUARDRAIL
        # ==========================================

        guard_prompt = (
            f"{GUARDRAIL_PROMPT}\n\n"
            f"User Input:\n{request.query}"
        )

        moderation_result = (
            await self.client.generate_guard_response(
                guard_prompt
            )
        )

        logger.info(
            "Input moderation result: %s",
            moderation_result,
        )

        if "UNSAFE" in moderation_result:
            return ChatResponse(
                response=(
                    "Request blocked by "
                    "safety guardrails."
                ),
                model=settings.guard_model,
                guardrail_passed=False,
                tools_used=[],
                retrieved_documents=0,
                retrieved_context="",
            )

        # ==========================================
        # CONVERSATION
        # ==========================================

        conversation = Conversation()

        conversation.add_message(
            role="user",
            content=request.query,
        )

        retrieved_documents = []

        rag_context = ""

        rewritten_query = request.query

        # ==========================================
        # QUERY REWRITING
        # ==========================================

        if request.use_rag:
            rewrite_prompt = f""" You are a semantic retrieval optimization system. Rewrite the user's query to improve vector database retrieval quality. Rules: 1. Preserve exact terminology. 2. Do NOT introduce hyphenation. 3. Do NOT rename entities. 4. Fix only obvious grammar issues. 5. Keep original wording whenever possible. 6. Return ONLY rewritten query. User Query: {request.query}

Rewritten Query:
"""

            try:
                rewrite_response = (
                    await self.client.generate(
                        prompt=rewrite_prompt,
                        model=settings.primary_model,
                        stream=False,
                    )
                )

                rewritten_query = (
                    rewrite_response.get(
                        "response",
                        request.query,
                    )
                    .strip()
                )

                logger.info(
                    "Original Query: %s",
                    request.query,
                )

                logger.info(
                    "Rewritten Query: %s",
                    rewritten_query,
                )

            except Exception:
                logger.exception(
                    "Query rewriting failed."
                )

                rewritten_query = request.query

        # ==========================================
        # RAG RETRIEVAL
        # ==========================================

        if request.use_rag:
            try:
                retrieved_documents = (
                    await self.retriever.retrieve(
                        rewritten_query
                    )
                )

                logger.info(
                    "Retrieved %s documents.",
                    len(retrieved_documents),
                )

                rag_context = (
                    self.retriever.build_context(
                        retrieved_documents
                    )
                )

                logger.info(
                    "Retrieved Context:\n%s",
                    rag_context[:3000],
                )

            except Exception:
                logger.exception(
                    "RAG retrieval failed."
                )

                retrieved_documents = []
                rag_context = ""

        # ==========================================
        # PROMPT BUILDING
        # ==========================================

        final_prompt = f"""
{RAG_SYSTEM_PROMPT}

===========================================================
RETRIEVED CONTEXT
===========================================================

{rag_context}

===========================================================
USER QUESTION
===========================================================

{request.query}

===========================================================
CRITICAL INSTRUCTIONS
===========================================================

1. Answer ONLY using retrieved context.

2. If answer exists in retrieved context:
   - answer directly
   - do NOT ask follow-up questions
   - do NOT say information is missing

3. Preserve exact:
   - percentages
   - dates
   - numbers
   - grading values
   - names

4. If answer does NOT exist in context:
   respond with:
   "The uploaded documents do not contain this information."

5. Keep answer concise and factual.

===========================================================
FINAL ANSWER
===========================================================
"""

        logger.info(
            "Final prompt generated successfully."
        )

        # ==========================================
        # LLM GENERATION
        # ==========================================

        try:
            response = (
                await self.client.generate(
                    prompt=final_prompt,
                    model=settings.primary_model,
                    stream=False,
                )
            )

            raw_response = response.get(
                "response",
                "",
            )

            logger.info(
                "Raw model response:\n%s",
                raw_response[:2000],
            )

        except Exception:
            logger.exception(
                "LLM generation failed."
            )

            return ChatResponse(
                response=(
                    "Generation failed due to "
                    "internal system error."
                ),
                model=settings.primary_model,
                guardrail_passed=False,
                tools_used=[],
                retrieved_documents=0,
                retrieved_context="",
            )

        # ==========================================
        # RESPONSE CLEANING
        # ==========================================

        cleaned_response = (
            ResponseParser.clean_response(
                raw_response
            )
        )

        logger.info(
            "Cleaned response:\n%s",
            cleaned_response[:2000],
        )

        # ==========================================
        # OUTPUT GUARDRAIL
        # ==========================================

        output_guard_prompt = (
            f"{GUARDRAIL_PROMPT}\n\n"
            f"Assistant Output:\n"
            f"{cleaned_response}"
        )

        try:
            output_moderation = (
                await self.client.generate_guard_response(
                    output_guard_prompt
                )
            )

            logger.info(
                "Output moderation result: %s",
                output_moderation,
            )

        except Exception:
            logger.exception(
                "Output moderation failed."
            )

            output_moderation = "SAFE"

        if "UNSAFE" in output_moderation:
            cleaned_response = (
                "Response blocked by "
                "output safety guardrails."
            )

        logger.info(
            "Chat orchestration completed successfully."
        )

        return ChatResponse(
            response=cleaned_response,
            model=settings.primary_model,
            guardrail_passed=True,
            tools_used=[],
            retrieved_documents=len(
                retrieved_documents
            ),
            retrieved_context=rag_context,
        )

