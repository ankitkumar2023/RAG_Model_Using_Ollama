from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Centralized application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =====================================================
    # APPLICATION
    # =====================================================

    app_name: str = Field(default="RAPID_PROTOTYPING")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development"
    )

    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # =====================================================
    # OLLAMA
    # =====================================================

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_timeout: int = Field(default=300)

    # =====================================================
    # MODELS
    # =====================================================

    primary_model: str = Field(default="qwen2.5:7b")
    guard_model: str = Field(default="llama-guard3:8b")
    embedding_model: str = Field(default="nomic-embed-text")

    # =====================================================
    # VECTOR STORE
    # =====================================================

    vector_db_path: str = Field(default="./data/vector_store")
    chroma_collection: str = Field(default="rag_documents")

    # =====================================================
    # RAG
    # =====================================================

    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.65)

    # =====================================================
    # GENERATION
    # =====================================================

    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.9)
    top_k: int = Field(default=40)
    max_tokens: int = Field(default=2048)
    repeat_penalty: float = Field(default=1.1)

    # =====================================================
    # STREAMING
    # =====================================================

    enable_streaming: bool = Field(default=True)

    # =====================================================
    # MEMORY
    # =====================================================

    max_chat_history: int = Field(default=20)

    # =====================================================
    # TOOLS
    # =====================================================

    enable_web_search: bool = Field(default=True)
    enable_weather_tool: bool = Field(default=True)
    enable_stock_tool: bool = Field(default=True)
    enable_calculator_tool: bool = Field(default=True)

    # =====================================================
    # OPTIONAL API KEYS
    # =====================================================

    openweather_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    serpapi_api_key: str | None = None

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def vector_db_directory(self) -> Path:
        return BASE_DIR / self.vector_db_path


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.
    """
    return Settings()