
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pdfplumber
import streamlit as st
from docx import Document

from config.logging_config import setup_logging
from config.settings import get_settings
from src.api.routes import APIRouter
from src.api.schemas import (
    ChatRequest,
    DocumentUploadRequest,
)
from src.utils.gpu_monitor import (
    SystemMonitor,
)

# =========================================================
# WINDOWS ASYNCIO FIX
# =========================================================

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# =========================================================
# INITIALIZATION
# =========================================================

setup_logging()

logger = logging.getLogger(__name__)

settings = get_settings()

router = APIRouter()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAPID_PROTOTYPING",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_SESSION_STATE = {
    "messages": [],
    "uploaded_documents": {},
    "retrieval_debug": [],
    "guardrail_logs": [],
}

for key, value in (
    DEFAULT_SESSION_STATE.items()
):
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# HELPERS
# =========================================================


def normalize_text(
    text: str,
) -> str:
    """
    Normalize extracted text.
    """

    text = re.sub(
        r"\r\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def extract_text_from_pdf(
    file_path: str,
) -> str:
    """
    Enterprise-grade PDF extraction pipeline.

    Features:
    - Layout-aware extraction
    - Table-aware extraction
    - Hybrid parsing
    - Semantic structure preservation
    - Retrieval optimization
    - OCR-friendly normalization
    """

    text_parts: list[str] = []

    logger.info(
        "Starting enterprise PDF extraction."
    )

    try:
        with pdfplumber.open(file_path) as pdf:

            logger.info(
                "PDF contains %s pages.",
                len(pdf.pages),
            )

            for page_idx, page in enumerate(
                pdf.pages,
                start=1,
            ):
                logger.info(
                    "Processing page %s",
                    page_idx,
                )

                page_sections: list[str] = []

                # =====================================================
                # STANDARD TEXT EXTRACTION
                # =====================================================

                try:
                    extracted_text = (
                        page.extract_text(
                            x_tolerance=2,
                            y_tolerance=2,
                            layout=True,
                        )
                        or ""
                    )

                    extracted_text = normalize_text(
                        extracted_text
                    )

                    if extracted_text:
                        page_sections.append(
                            "[TEXT CONTENT]\n\n"
                            f"{extracted_text}"
                        )

                        logger.info(
                            "Extracted %s text chars "
                            "from page %s",
                            len(extracted_text),
                            page_idx,
                        )

                except Exception:
                    logger.exception(
                        "Standard extraction failed "
                        "for page %s",
                        page_idx,
                    )

                # =====================================================
                # TABLE EXTRACTION
                # =====================================================

                try:
                    tables = (
                        page.extract_tables()
                    )

                    logger.info(
                        "Detected %s tables "
                        "on page %s",
                        len(tables),
                        page_idx,
                    )

                    for table_idx, table in enumerate(
                        tables,
                        start=1,
                    ):
                        if not table:
                            continue

                        structured_rows = []

                        for row in table:
                            if not row:
                                continue

                            cleaned_cells = []

                            for cell in row:
                                if cell is None:
                                    cleaned_cells.append(
                                        ""
                                    )
                                else:
                                    cleaned_cells.append(
                                        normalize_text(
                                            str(cell)
                                        )
                                    )

                            if not any(
                                cleaned_cells
                            ):
                                continue

                            row_text = (
                                " | ".join(
                                    cleaned_cells
                                )
                            )

                            structured_rows.append(
                                row_text
                            )

                        if structured_rows:
                            table_block = (
                                f"[TABLE {table_idx}]\n\n"
                                f"{chr(10).join(structured_rows)}"
                            )

                            page_sections.append(
                                table_block
                            )

                            logger.info(
                                "Processed table %s "
                                "with %s rows.",
                                table_idx,
                                len(structured_rows),
                            )

                except Exception:
                    logger.exception(
                        "Table extraction failed "
                        "for page %s",
                        page_idx,
                    )

                # =====================================================
                # PAGE CONSOLIDATION
                # =====================================================

                if page_sections:
                    page_content = (
                        "\n\n".join(
                            page_sections
                        )
                    )

                    page_block = (
                        f"\n\n"
                        f"================ PAGE "
                        f"{page_idx} "
                        f"================\n\n"
                        f"{page_content}"
                    )

                    text_parts.append(
                        page_block
                    )

    except Exception:
        logger.exception(
            "Enterprise PDF extraction failed."
        )

        raise

    # =========================================================
    # FINAL NORMALIZATION
    # =========================================================

    final_text = "\n".join(
        text_parts
    )

    final_text = re.sub(
        r"\n{3,}",
        "\n\n",
        final_text,
    )

    final_text = re.sub(
        r"[ \t]+",
        " ",
        final_text,
    )

    final_text = final_text.strip()

    logger.info(
        "Final extracted text size: %s chars",
        len(final_text),
    )

    if not final_text:
        raise ValueError(
            "No extractable text found in PDF."
        )

    logger.info(
        "Enterprise PDF extraction completed."
    )

    return final_text


def extract_text_from_docx(
    file_path: str,
) -> str:
    """
    Extract DOCX text.
    """

    try:
        document = Document(file_path)

        text = "\n".join(
            [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
        )

        text = normalize_text(text)

        logger.info(
            "Extracted %s characters from DOCX.",
            len(text),
        )

        return text

    except Exception:
        logger.exception(
            "DOCX extraction failed."
        )

        raise


def extract_uploaded_text(
    uploaded_file: Any,
) -> str:
    """
    Process uploaded document safely.
    """

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(
            uploaded_file.read()
        )

        temp_path = temp_file.name

    logger.info(
        "Temporary file created: %s",
        temp_path,
    )

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(
                temp_path
            )

        if suffix == ".docx":
            return extract_text_from_docx(
                temp_path
            )

        text = Path(temp_path).read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return normalize_text(text)

    finally:
        try:
            Path(temp_path).unlink(
                missing_ok=True
            )

        except Exception:
            logger.warning(
                "Temporary file cleanup failed."
            )


def generate_document_hash(
    content: str,
) -> str:
    """
    Generate deterministic hash.
    """

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


async def process_chat_async(
    query: str,
    use_rag: bool,
    enable_tools: bool,
) -> dict:
    """
    Execute async chat orchestration.
    """

    request = ChatRequest(
        query=query,
        use_rag=use_rag,
        enable_tools=enable_tools,
        stream=False,
    )

    response = await router.process_chat(
        request
    )

    return response.model_dump()


async def upload_document_async(
    filename: str,
    content: str,
) -> dict:
    """
    Async document ingestion.
    """

    request = DocumentUploadRequest(
        filename=filename,
        content=content,
    )

    response = await router.upload_document(
        request
    )

    return response.model_dump()


def run_async(coro):
    """
    Execute async safely inside Streamlit.
    """

    try:
        loop = asyncio.get_event_loop()

        if loop.is_closed():
            raise RuntimeError

    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        coro
    )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("⚙️ Control Center")

    st.markdown("---")

    st.subheader("Models")

    st.info(
        f"""
Primary Model:
{settings.primary_model}

Guard Model:
{settings.guard_model}

Embedding Model:
{settings.embedding_model}
"""
    )

    st.subheader("RAG Settings")

    use_rag = st.toggle(
        "Enable RAG",
        value=True,
    )

    enable_tools = st.toggle(
        "Enable Tools",
        value=True,
    )

    st.subheader("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_bytes = (
                    uploaded_file.getvalue()
                )

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                if (
                    file_hash
                    in st.session_state[
                        "uploaded_documents"
                    ]
                ):
                    st.warning(
                        f"{uploaded_file.name} "
                        f"already indexed."
                    )

                    continue

                with st.spinner(
                    f"Processing "
                    f"{uploaded_file.name}..."
                ):
                    text = extract_uploaded_text(
                        uploaded_file
                    )

                    result = run_async(
                        upload_document_async(
                            uploaded_file.name,
                            text,
                        )
                    )

                    st.session_state[
                        "uploaded_documents"
                    ][file_hash] = {
                        "filename": uploaded_file.name,
                        "timestamp": str(
                            datetime.now(UTC)
                        ),
                    }

                    st.success(
                        f"""
Indexed Successfully

File:
{uploaded_file.name}

Chunks:
{result['chunks_created']}
"""
                    )

            except Exception as exc:
                logger.exception(
                    "Upload failed."
                )

                st.error(str(exc))

    st.subheader("Vector Store")

    if st.button(
        "Reset Vector Database",
        type="secondary",
    ):
        try:
            run_async(
                router.vector_store.reset_collection()
            )

            st.success(
                "Vector database cleared."
            )

        except Exception:
            logger.exception(
                "Vector reset failed."
            )

            st.error(
                "Failed to reset vector database."
            )

    st.subheader("System Metrics")

    try:
        metrics = (
            SystemMonitor.collect_metrics()
        )

        st.metric(
            "CPU Usage",
            f"{metrics.cpu_percent:.1f}%",
        )

        st.metric(
            "RAM Usage",
            (
                f"{metrics.ram_used_gb:.1f}/"
                f"{metrics.ram_total_gb:.1f} GB"
            ),
        )

        if metrics.gpu_name:
            st.metric(
                "GPU Usage",
                (
                    f"{metrics.gpu_utilization_percent:.1f}%"
                ),
            )

            st.caption(metrics.gpu_name)

    except Exception:
        logger.exception(
            "Metrics collection failed."
        )

# =========================================================
# MAIN UI
# =========================================================

st.title(
    "🤖 RAPID_PROTOTYPING"
)

st.caption(
    "Enterprise Local Agentic RAG System"
)

try:
    health = run_async(
        router.health_check()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if health.ollama_connected:
            st.success(
                "Ollama Connected"
            )
        else:
            st.error(
                "Ollama Offline"
            )

    with col2:
        st.info(
            f"Indexed Docs: "
            f"{health.vector_store_documents}"
        )

    with col3:
        st.info(
            f"Environment: "
            f"{settings.app_env}"
        )

except Exception:
    logger.exception(
        "Health check failed."
    )

st.markdown("---")

for message in (
    st.session_state.messages
):
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

prompt = st.chat_input(
    "Ask anything..."
)

if prompt:
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": str(
            datetime.now(UTC)
        ),
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message(
        "assistant"
    ):
        response_placeholder = (
            st.empty()
        )

        debug_expander = st.expander(
            "Diagnostics",
            expanded=False,
        )

        try:
            with st.spinner(
                "Running retrieval, "
                "guardrails, and reasoning..."
            ):
                result = run_async(
                    process_chat_async(
                        query=prompt,
                        use_rag=use_rag,
                        enable_tools=enable_tools,
                    )
                )

            final_response = result.get(
                "response",
                "No response generated.",
            )

            response_placeholder.markdown(
                final_response
            )

            with debug_expander:
                st.subheader(
                    "Guardrail Status"
                )

                if result.get(
                    "guardrail_passed",
                    False,
                ):
                    st.success(
                        "Safety checks passed."
                    )
                else:
                    st.error(
                        "Guardrail violation."
                    )

                st.subheader(
                    "Retrieval"
                )

                st.write(
                    f"Retrieved Documents: "
                    f"{result.get('retrieved_documents', 0)}"
                )

                retrieved_context = (
                    result.get(
                        "retrieved_context",
                        "",
                    )
                )

                if retrieved_context:
                    st.code(
                        retrieved_context,
                        language="text",
                    )

                st.subheader(
                    "Generation"
                )

                st.write(
                    f"Model Used: "
                    f"{result.get('model', 'unknown')}"
                )

            assistant_message = {
                "role": "assistant",
                "content": final_response,
                "timestamp": str(
                    datetime.now(UTC)
                ),
            }

            st.session_state.messages.append(
                assistant_message
            )

        except Exception as exc:
            logger.exception(
                "Chat processing failed."
            )

            response_placeholder.error(
                f"System Error: {str(exc)}"
            )

st.markdown("---")

st.caption(
    "Built with Ollama + Qwen2.5 + "
    "Llama Guard + ChromaDB + Streamlit"
)

