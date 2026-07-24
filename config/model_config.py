from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration metadata for a Gemini model.
    """

    name: str
    context_window: int
    supports_streaming: bool
    temperature: float


PRIMARY_MODEL_CONFIG = ModelConfig(
    name="gemini-flash-latest",
    context_window=1_048_576,
    supports_streaming=True,
    temperature=0.2,
)

PRO_MODEL_CONFIG = ModelConfig(
    name="gemini-pro-latest",
    context_window=1_048_576,
    supports_streaming=True,
    temperature=0.2,
)

EMBEDDING_MODEL_CONFIG = ModelConfig(
    name="models/gemini-embedding-001",
    context_window=2048,
    supports_streaming=False,
    temperature=0.0,
)

AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "gemini-flash-latest": PRIMARY_MODEL_CONFIG,
    "gemini-pro-latest": PRO_MODEL_CONFIG,
}
