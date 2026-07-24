from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_streamlit_secrets_into_env() -> None:
    """
    Bridge Streamlit Community Cloud's secrets manager into the process
    environment, so `Settings` (which reads from the environment / a
    local `.env` file) works unchanged in both places.

    On Cloud, secrets come from the app's Secrets dashboard via
    `st.secrets` -- there's no committed `.env` (it's gitignored) and
    the filesystem starts fresh on every deploy. Locally, `st.secrets`
    simply has nothing configured and this is a no-op, so `.env`
    continues to be the source of truth for local development.
    """

    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:
        # No secrets.toml, not running under Streamlit, or no secrets
        # configured -- fall through to .env / real env vars.
        return

    for key, value in secrets.items():
        env_key = key.upper()

        if env_key not in os.environ:
            os.environ[env_key] = str(value)


_load_streamlit_secrets_into_env()


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
    app_env: Literal["development", "staging", "production"] = Field(default="development")

    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # =====================================================
    # GEMINI (Google Gen AI)
    # =====================================================

    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_timeout: int = Field(default=120)
    gemini_max_retries: int = Field(default=3)

    # =====================================================
    # MODELS
    # =====================================================

    # "-latest" aliases are used instead of pinned versions (e.g.
    # "gemini-2.5-flash") because Google periodically retires older
    # pinned models for new API keys/projects; the alias always resolves
    # to Google's current recommended model in that tier. Override in
    # .env with a specific pinned version if your key needs one.
    primary_model: str = Field(default="gemini-flash-latest")
    pro_model: str = Field(default="gemini-pro-latest")
    embedding_model: str = Field(default="models/gemini-embedding-001")

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

    # Cosine similarity in [0, 1] between the query and a chunk, computed
    # directly against the stored embeddings (see VectorStore, which
    # configures the collection for cosine distance). Chunks scoring below
    # this are dropped before reaching the LLM.
    retrieval_score_threshold: float = Field(default=0.0)

    # =====================================================
    # RETRIEVAL STRATEGY TOGGLES
    # =====================================================

    enable_multi_query: bool = Field(default=True)
    enable_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.5)

    # Gemini-based reranking (an extra lightweight LLM call that judges
    # candidate chunks against the original question) rather than a
    # cross-encoder model, to avoid pulling a torch/sentence-transformers
    # dependency into every deployment target.
    enable_reranking: bool = Field(default=True)
    rerank_candidate_multiplier: int = Field(default=3)

    # How many recent chat turns (user+assistant messages) are used to
    # condense a follow-up question into a standalone one before
    # retrieval. Kept small on purpose so stale conversation topics don't
    # leak into unrelated retrieval queries.
    condense_history_turns: int = Field(default=6)

    # =====================================================
    # QUERY ROUTING
    # =====================================================

    enable_query_router: bool = Field(default=True)

    # Below this RAG confidence score, the router automatically falls
    # back to a web-search route instead of answering "not found"
    # ("corrective RAG").
    corrective_rag_threshold: float = Field(default=0.4)

    # =====================================================
    # AGENT / TOOL LOOP
    # =====================================================

    enable_tool_loop: bool = Field(default=True)
    max_tool_iterations: int = Field(default=3)

    # =====================================================
    # GENERATION
    # =====================================================

    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.9)
    top_k: int = Field(default=40)
    max_tokens: int = Field(default=2048)

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

    # =====================================================
    # SECURITY
    # =====================================================

    max_upload_size_mb: int = Field(default=25)
    allowed_upload_extensions: tuple[str, ...] = Field(default=(".pdf", ".docx", ".txt", ".md"))
    rate_limit_requests: int = Field(default=20)
    rate_limit_window_seconds: int = Field(default=60)

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
