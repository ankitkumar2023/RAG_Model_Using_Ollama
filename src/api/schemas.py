from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """
    Chat message schema.
    """

    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]

    content: str = Field(
        min_length=1,
    )


class ChatRequest(BaseModel):
    """
    User chat request.
    """

    query: str = Field(
        min_length=1,
        max_length=50000,
    )

    use_rag: bool = True

    enable_tools: bool = True

    stream: bool = True

    # Prior conversation turns, most recent last. Used to condense a
    # follow-up question into a standalone retrieval query -- NOT dumped
    # wholesale into the final answer-generation prompt.
    history: list[ChatMessageSchema] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceCitation(BaseModel):
    """
    A single cited source backing a response.
    """

    filename: str
    page: int | None = None
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    """
    Assistant response.
    """

    retrieved_context: str = ""

    response: str

    model: str

    guardrail_passed: bool

    tools_used: list[str]

    retrieved_documents: int

    sources: list[SourceCitation] = Field(default_factory=list)

    # Blend of retrieval relevance and answer-grounding similarity, in
    # [0, 1]. 0.0 when RAG wasn't used or nothing was retrieved.
    confidence_score: float = 0.0

    # ---- Router diagnostics ("Router Inspector" panel) ----

    route: str = "general"
    route_reason: str = ""
    route_confidence: float = 0.0

    # True when the route started as document_rag/hybrid but retrieval
    # confidence was below the corrective threshold and it fell back to
    # a web search ("corrective RAG").
    corrective_fallback: bool = False

    web_results_used: int = 0
    reranked: bool = False

    router_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolInvocationRequest(BaseModel):
    """
    Tool execution request.
    """

    tool_name: str

    parameters: dict[str, Any]


class ToolInvocationResponse(BaseModel):
    """
    Tool execution response.
    """

    tool_name: str

    success: bool

    result: dict[str, Any] | str

    execution_time: float


class DocumentUploadRequest(BaseModel):
    """
    RAG document upload.
    """

    filename: str

    # One entry per page. Non-paginated formats (.txt, .md, .docx) use a
    # single-element list. Chunking runs per-page so each resulting chunk
    # can be cited with an accurate page number.
    pages: list[str] = Field(
        min_length=1,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """
    Upload result.
    """

    success: bool

    chunks_created: int

    document_id: str


class HealthResponse(BaseModel):
    """
    Health check schema.
    """

    status: str

    gemini_connected: bool

    vector_store_documents: int

    timestamp: datetime = Field(default_factory=datetime.utcnow)
