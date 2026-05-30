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

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


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

    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )


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

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


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

    ollama_connected: bool

    vector_store_documents: int

    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )