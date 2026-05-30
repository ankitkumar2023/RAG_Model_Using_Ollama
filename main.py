from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from docx import Document
from pypdf import PdfReader

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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "retrieval_debug" not in st.session_state:
    st.session_state.retrieval_debug = []

if "guardrail_logs" not in st.session_state:
    st.session_state.guardrail_logs = []

# =========================================================
# HELPERS
# =========================================================


def extract_text_from_pdf(
    file_path: str,
) -> str:
    """
    Extract PDF text.
    """

    reader = PdfReader(file_path)

    text_parts: list[str] = []

    for page in reader.pages:
        text_parts.append(
            page.extract_text() or ""
        )

    return "\n".join(text_parts)


def extract_text_from_docx(
    file_path: str,
) -> str:
    """
    Extract DOCX text.
    """

    document = Document(file_path)

    return "\n".join(
        [
            paragraph.text
            for paragraph in document.paragraphs
        ]
    )


def extract_uploaded_text(
    uploaded_file: Any,
) -> str:
    """
    Process uploaded document.
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

    if suffix == ".pdf":
        return extract_text_from_pdf(
            temp_path
        )

    if suffix == ".docx":
        return extract_text_from_docx(
            temp_path
        )

    return Path(temp_path).read_text(
        encoding="utf-8",
        errors="ignore",
    )


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


def run_async(
    coro,
):
    """
    Execute async functions safely in Streamlit.
    """

    try:
        loop = asyncio.get_event_loop()

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

    # ==========================================
    # MODEL INFO
    # ==========================================

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

    # ==========================================
    # RAG CONTROLS
    # ==========================================

    st.subheader("RAG Settings")

    use_rag = st.toggle(
        "Enable RAG",
        value=True,
    )

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(
            settings.similarity_threshold
        ),
        step=0.05,
    )

    # ==========================================
    # TOOL CONTROLS
    # ==========================================

    st.subheader("Tool Controls")

    enable_tools = st.toggle(
        "Enable Tools",
        value=True,
    )

    enable_weather = st.checkbox(
        "Weather Tool",
        value=True,
    )

    enable_stock = st.checkbox(
        "Stock Tool",
        value=True,
    )

    enable_web = st.checkbox(
        "Web Search Tool",
        value=True,
    )

    enable_calc = st.checkbox(
        "Calculator Tool",
        value=True,
    )

    # ==========================================
    # DOCUMENT UPLOAD
    # ==========================================

    st.subheader("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "txt",
            "pdf",
            "docx",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in (
                st.session_state.uploaded_documents
            ):
                continue

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):
                try:
                    text = extract_uploaded_text(
                        uploaded_file
                    )

                    result = run_async(
                        upload_document_async(
                            uploaded_file.name,
                            text,
                        )
                    )

                    st.success(
                        f"""
Indexed:
{uploaded_file.name}

Chunks:
{result['chunks_created']}
"""
                    )

                    st.session_state.uploaded_documents.append(
                        uploaded_file.name
                    )

                except Exception as exc:
                    logger.exception(
                        "Upload failed."
                    )

                    st.error(str(exc))

    # ==========================================
    # VECTOR DB MANAGEMENT
    # ==========================================

    st.subheader("Vector Store")

    if st.button(
        "Reset Vector Database",
        type="secondary",
    ):
        run_async(
            router.vector_store.reset_collection()
        )

        st.success(
            "Vector database cleared."
        )

    # ==========================================
    # SYSTEM METRICS
    # ==========================================

    st.subheader("System Metrics")

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
            f"{metrics.ram_used_gb:.1f}"
            f"/{metrics.ram_total_gb:.1f} GB"
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

# =========================================================
# MAIN UI
# =========================================================

st.title(
    "🤖 RAPID_PROTOTYPING"
)

st.caption(
    "Production-grade Local Agentic RAG System"
)

# =========================================================
# STATUS BAR
# =========================================================

health = run_async(
    router.health_check()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        (
            "Ollama Connected"
            if health.ollama_connected
            else "Ollama Offline"
        )
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

st.markdown("---")

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# =========================================================
# USER INPUT
# =========================================================

prompt = st.chat_input(
    "Ask anything..."
)

if prompt:
    # ==========================================
    # STORE USER MESSAGE
    # ==========================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "timestamp": str(
                datetime.utcnow()
            ),
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ==========================================
    # ASSISTANT PROCESSING
    # ==========================================

    with st.chat_message(
        "assistant"
    ):
        response_placeholder = (
            st.empty()
        )

        debug_expander = (
            st.expander(
                "Diagnostics",
                expanded=False,
            )
        )

        with st.spinner(
            "Running guardrails, retrieval, "
            "and reasoning..."
        ):
            try:
                result = run_async(
                    process_chat_async(
                        query=prompt,
                        use_rag=use_rag,
                        enable_tools=enable_tools,
                    )
                )

                final_response = result[
                    "response"
                ]

                response_placeholder.markdown(
                    final_response
                )

                with debug_expander:
                    st.subheader(
                        "Guardrail Status"
                    )

                    if result[
                        "guardrail_passed"
                    ]:
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
                        (
                            "Retrieved Documents: "
                            f"{result['retrieved_documents']}"
                        )
                    )

                    st.subheader(
                        "Generation"
                    )

                    st.write(
                        (
                            "Model Used: "
                            f"{result['model']}"
                        )
                    )

                    st.subheader(
                        "Tools"
                    )

                    if result[
                        "tools_used"
                    ]:
                        st.write(
                            result[
                                "tools_used"
                            ]
                        )
                    else:
                        st.write(
                            "No tools used."
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                        "timestamp": str(
                            datetime.utcnow()
                        ),
                    }
                )

            except Exception as exc:
                logger.exception(
                    "Chat processing failed."
                )

                error_message = (
                    f"System Error: {str(exc)}"
                )

                response_placeholder.error(
                    error_message
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with Ollama + Qwen2.5 + "
    "Llama Guard + ChromaDB + Streamlit"
)