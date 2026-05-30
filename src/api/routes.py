from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config.settings import get_settings
from prompts.rag_prompts import (
    ANSWER_SYNTHESIS_PROMPT,
)
from prompts.system_prompts import (
    GUARDRAIL_PROMPT,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
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
from src.tools.calculator_tool import (
    CalculatorTool,
)
from src.tools.stock_tool import StockTool
from src.tools.weather_tool import (
    WeatherTool,
)
from src.tools.web_search_tool import (
    WebSearchTool,
)

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

        self.calculator_tool = (
            CalculatorTool()
        )

        self.weather_tool = WeatherTool()

        self.stock_tool = StockTool()

        self.web_search_tool = (
            WebSearchTool()
        )

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

        # ==========================================
        # RAG RETRIEVAL
        # ==========================================

        if request.use_rag:
            retrieved_documents = (
                await self.retriever.retrieve(
                    request.query
                )
            )

            rag_context = (
                self.retriever.build_context(
                    retrieved_documents
                )
            )

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

- Answer ONLY using retrieved context.
- If answer exists in context, provide it directly.
- Do NOT ask unnecessary follow-up questions.
- Preserve exact percentages and values.
- Keep answer concise and factual.
- If context contains the answer,
  you MUST answer directly.

===========================================================
FINAL ANSWER
===========================================================
"""

        # ==========================================
        # LLM GENERATION
        # ==========================================

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

        cleaned_response = (
            ResponseParser.clean_response(
                raw_response
            )
        )

        # ==========================================
        # OUTPUT GUARDRAIL
        # ==========================================

        output_guard_prompt = (
            f"{GUARDRAIL_PROMPT}\n\n"
            f"Assistant Output:\n"
            f"{cleaned_response}"
        )

        output_moderation = (
            await self.client.generate_guard_response(
                output_guard_prompt
            )
        )

        if "UNSAFE" in output_moderation:
            cleaned_response = (
                "Response blocked by "
                "output safety guardrails."
            )

        logger.info(
            "Chat orchestration completed."
        )

        return ChatResponse(
            response=cleaned_response,
            model=settings.primary_model,
            guardrail_passed=True,
            tools_used=[],
            retrieved_documents=len(
                retrieved_documents
            ),
        )