from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration metadata for local models.
    """

    name: str
    context_window: int
    supports_tools: bool
    supports_streaming: bool
    temperature: float


PRIMARY_MODEL_CONFIG = ModelConfig(
    name="qwen2.5:7b",
    context_window=32768,
    supports_tools=True,
    supports_streaming=True,
    temperature=0.2,
)

GUARD_MODEL_CONFIG = ModelConfig(
    name="llama-guard3:8b",
    context_window=8192,
    supports_tools=False,
    supports_streaming=False,
    temperature=0.0,
)

EMBEDDING_MODEL_CONFIG = ModelConfig(
    name="nomic-embed-text",
    context_window=8192,
    supports_tools=False,
    supports_streaming=False,
    temperature=0.0,
)