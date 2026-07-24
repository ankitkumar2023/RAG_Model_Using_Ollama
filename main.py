
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pdfplumber
import streamlit as st
import streamlit.components.v1 as components
from docx import Document

from config.logging_config import setup_logging
from config.settings import get_settings
from src.api.routes import APIRouter
from src.api.schemas import (
    ChatRequest,
    DocumentUploadRequest,
)
from src.llm.tokenizer import Tokenizer
from src.security.file_validation import validate_upload
from src.security.rate_limiter import RateLimiter
from src.utils.async_bridge import to_sync_iterator
from src.utils.gpu_monitor import SystemMonitor
from ui.components import (
    citation_card,
    diagnostics_dashboard,
    hero_banner,
    loading_pipeline,
    router_timeline_card,
    status_bar_html,
    upload_card,
)
from ui.theme import COLORS, inject_theme, render_aurora_background

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


@st.cache_resource(show_spinner=False)
def get_router() -> APIRouter:
    """
    Build the orchestrator once per process.

    Cached so the Gemini client, embedding client, and vector store are
    not reconstructed on every Streamlit script rerun.
    """

    return APIRouter()


@st.cache_resource(show_spinner=False)
def get_rate_limiter() -> RateLimiter:
    """
    Shared rate limiter instance, cached for the same reason as the
    router above -- an uncached limiter would reset its window on every
    rerun and never actually limit anything.
    """

    return RateLimiter()


@st.cache_resource(show_spinner=False)
def get_tokenizer() -> Tokenizer:
    """
    Approximate token counter for the diagnostics dashboard (tiktoken,
    not Gemini's actual tokenizer -- an estimate, not an exact count).
    """

    return Tokenizer()


router = get_router()
rate_limiter = get_rate_limiter()
tokenizer = get_tokenizer()

# =========================================================
# PAGE CONFIG + DESIGN SYSTEM
# =========================================================

st.set_page_config(
    page_title="AI Workspace",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
render_aurora_background()

# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_SESSION_STATE = {
    "messages": [],
    "uploaded_documents": {},
    "saved_chats": {},
    "session_id": None,
    "last_route": None,
    "last_latency_ms": None,
    "last_confidence": None,
    "api_call_count": 0,
    "regenerate_requested": None,
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state["session_id"] is None:
    st.session_state["session_id"] = uuid.uuid4().hex

# =========================================================
# DOCUMENT EXTRACTION (backend logic -- unstyled)
# =========================================================


def normalize_text(text: str) -> str:
    """
    Normalize extracted text.
    """

    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_text_from_pdf(file_path: str) -> list[str]:
    """
    Enterprise-grade PDF extraction pipeline.

    Returns one text block per page (rather than one flattened string) so
    downstream chunking/ingestion can tag each chunk with an accurate
    page number for citations.
    """

    pages: list[str] = []

    logger.info("Starting enterprise PDF extraction.")

    try:
        with pdfplumber.open(file_path) as pdf:
            logger.info("PDF contains %s pages.", len(pdf.pages))

            for page_idx, page in enumerate(pdf.pages, start=1):
                page_sections: list[str] = []

                try:
                    extracted_text = (
                        page.extract_text(
                            x_tolerance=2,
                            y_tolerance=2,
                            layout=True,
                        )
                        or ""
                    )

                    extracted_text = normalize_text(extracted_text)

                    if extracted_text:
                        page_sections.append(
                            f"[TEXT CONTENT]\n\n{extracted_text}"
                        )

                except Exception:
                    logger.exception(
                        "Standard extraction failed for page %s", page_idx
                    )

                try:
                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables, start=1):
                        if not table:
                            continue

                        structured_rows = []

                        for row in table:
                            if not row:
                                continue

                            cleaned_cells = [
                                normalize_text(str(cell)) if cell else ""
                                for cell in row
                            ]

                            if not any(cleaned_cells):
                                continue

                            structured_rows.append(
                                " | ".join(cleaned_cells)
                            )

                        if structured_rows:
                            page_sections.append(
                                f"[TABLE {table_idx}]\n\n"
                                f"{chr(10).join(structured_rows)}"
                            )

                except Exception:
                    logger.exception(
                        "Table extraction failed for page %s", page_idx
                    )

                if page_sections:
                    page_content = normalize_text(
                        "\n\n".join(page_sections)
                    )

                    if page_content:
                        pages.append(page_content)

    except Exception:
        logger.exception("Enterprise PDF extraction failed.")
        raise

    if not pages:
        raise ValueError("No extractable text found in PDF.")

    logger.info("Enterprise PDF extraction completed: %s pages.", len(pages))

    return pages


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract DOCX text.

    python-docx has no reliable notion of page boundaries (pagination is
    a rendering-time concern), so DOCX is treated as a single page.
    """

    try:
        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        text = normalize_text(text)

        logger.info("Extracted %s characters from DOCX.", len(text))

        return text

    except Exception:
        logger.exception("DOCX extraction failed.")
        raise


def extract_uploaded_text(uploaded_file: Any) -> list[str]:
    """
    Process uploaded document safely.

    Returns one text block per page. Non-paginated formats (.docx, .txt,
    .md) return a single-element list.
    """

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(temp_path)

        if suffix == ".docx":
            return [extract_text_from_docx(temp_path)]

        text = Path(temp_path).read_text(
            encoding="utf-8", errors="ignore"
        )

        return [normalize_text(text)]

    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            logger.warning("Temporary file cleanup failed.")


# =========================================================
# ASYNC ORCHESTRATION HELPERS
# =========================================================


async def stream_chat_async(
    query: str,
    use_rag: bool,
    enable_tools: bool,
    history: list[dict],
):
    request = ChatRequest(
        query=query,
        use_rag=use_rag,
        enable_tools=enable_tools,
        stream=True,
        history=history,
    )

    return await router.stream_chat(request)


async def upload_document_async(filename: str, pages: list[str]) -> dict:
    request = DocumentUploadRequest(filename=filename, pages=pages)
    response = await router.upload_document(request)
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

    return loop.run_until_complete(coro)


def render_copy_button(text: str, key: str) -> None:
    """
    Client-side clipboard copy -- no server round-trip. Tries the modern
    Clipboard API first, falls back to the older execCommand approach
    (more likely to work inside Streamlit's sandboxed component iframe).

    The text is embedded inside a <script> block, not an inline
    `onclick="..."` HTML attribute -- `onclick` is itself double-quote
    delimited, and json.dumps() produces a double-quoted string, so
    embedding it there terminates the attribute at the first embedded
    quote and leaks the rest of the JS onto the page as literal text.
    A <script> block has no such attribute-quoting concern.
    """

    # Defensive: a literal "</script" inside the text would otherwise
    # terminate the script block early.
    safe_text = json.dumps(text).replace("</script", "<\\/script")

    components.html(
        f"""
        <div style="font-family:Inter,sans-serif;">
        <button id="copy-{key}" style="
            background:{COLORS['surface_elevated']}; color:{COLORS['text_secondary']};
            border:1px solid {COLORS['border_strong']}; border-radius:8px;
            padding:4px 12px; font-size:12px; cursor:pointer;
        ">📋 Copy</button>
        </div>
        <script>
        (function() {{
            const text = {safe_text};
            const btn = document.getElementById('copy-{key}');

            function fallbackCopy() {{
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }}

            btn.addEventListener('click', function() {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).catch(fallbackCopy);
                }} else {{
                    fallbackCopy();
                }}
                btn.innerText = '✓ Copied';
                setTimeout(function() {{ btn.innerText = '📋 Copy'; }}, 1500);
            }});
        }})();
        </script>
        """,
        height=38,
    )


def request_regenerate() -> None:
    """
    Drop the last assistant turn and re-queue its user query.
    """

    messages = st.session_state.messages

    if messages and messages[-1]["role"] == "assistant":
        messages.pop()

    if messages and messages[-1]["role"] == "user":
        st.session_state["regenerate_requested"] = messages[-1]["content"]


def render_assistant_extras(
    meta: dict,
    developer_mode: bool = False,
    message_key: str = "",
) -> None:
    """
    Citations, Router Inspector timeline, copy/regenerate, and (in
    Developer Mode) the live diagnostics dashboard for one assistant
    turn -- all using the design-system components.
    """

    action_cols = st.columns([1, 1, 8])

    with action_cols[0]:
        render_copy_button(meta.get("response", ""), key=message_key)

    with action_cols[1]:
        if st.button("🔁 Regenerate", key=f"regen_{message_key}"):
            request_regenerate()
            st.rerun()

    sources = meta.get("sources") or []

    if sources:
        with st.expander(
            f"📚 Sources ({len(sources)})", expanded=False
        ):
            for source in sources:
                st.html(
                    citation_card(
                        filename=source.get("filename", "unknown"),
                        page=source.get("page"),
                        score=source.get("score", 0.0),
                        snippet=source.get("text_snippet", ""),
                    )
                )

    route = meta.get("route")

    if route:
        latency_ms = {
            "Routing": meta.get("router_latency_ms", 0.0),
            "Retrieval": meta.get("retrieval_latency_ms", 0.0),
            "Generation": meta.get("generation_latency_ms", 0.0),
            "Total": meta.get("total_latency_ms", 0.0),
        }

        st.html(
            router_timeline_card(
                route=route,
                reason=meta.get("route_reason", ""),
                confidence=meta.get("route_confidence", 0.0),
                tools_used=meta.get("tools_used", []),
                latency_ms=latency_ms,
                corrective_fallback=meta.get(
                    "corrective_fallback", False
                ),
                reranked=meta.get("reranked", False),
                retrieved_documents=meta.get("retrieved_documents", 0),
                web_results_used=meta.get("web_results_used", 0),
            )
        )

        if developer_mode:
            response_text = meta.get("response", "")
            approx_tokens = (
                tokenizer.count_tokens(response_text)
                if response_text
                else 0
            )
            gen_latency_s = max(
                meta.get("generation_latency_ms", 0.0) / 1000, 0.001
            )
            streaming_speed = approx_tokens / gen_latency_s

            st.html(
                diagnostics_dashboard(
                    latency_ms=meta.get("total_latency_ms", 0.0),
                    approx_tokens=approx_tokens,
                    streaming_speed=streaming_speed,
                    api_calls=st.session_state.api_call_count,
                    retrieved_documents=meta.get(
                        "retrieved_documents", 0
                    ),
                    confidence=meta.get("confidence_score", 0.0),
                    model=meta.get("model", settings.primary_model),
                    route=route,
                    sources=meta.get("sources") or [],
                )
            )


# =========================================================
# SIDEBAR: Workspace | Recent Chats | Knowledge | Uploads | Settings
# =========================================================

with st.sidebar:
    st.html(
        "<div style='font-size:20px; font-weight:800; "
        "background: linear-gradient(135deg,#7C3AED,#00D4FF); "
        "-webkit-background-clip:text; -webkit-text-fill-color:transparent; "
        "margin-bottom:4px;'>✨ AI Workspace</div>"
    )
    st.caption("Gemini + LangChain + ChromaDB")

    st.markdown("---")

    # ---- Workspace ----
    st.html("<div class='nav-section-label'>🗂️ Workspace</div>")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ---- Recent / Pinned Chats ----
    st.html(
        "<div class='nav-section-label'>🕑 Recent &amp; Pinned Chats</div>"
    )

    if st.session_state.messages and st.button(
        "💾 Save current chat", use_container_width=True
    ):
        first_user_msg = next(
            (
                m["content"]
                for m in st.session_state.messages
                if m["role"] == "user"
            ),
            "Untitled chat",
        )

        chat_id = uuid.uuid4().hex[:8]

        st.session_state.saved_chats[chat_id] = {
            "title": first_user_msg[:40],
            "timestamp": datetime.now(UTC).strftime("%H:%M"),
            "messages": list(st.session_state.messages),
            "pinned": False,
        }

    if st.session_state.saved_chats:
        ordered_chats = sorted(
            st.session_state.saved_chats.items(),
            key=lambda kv: not kv[1].get("pinned", False),
        )

        for chat_id, chat in ordered_chats:
            pin_icon = "📌" if chat.get("pinned") else "📍"
            row = st.columns([5, 1])

            with row[0]:
                if st.button(
                    f"💬 {chat['title']} · {chat['timestamp']}",
                    key=f"load_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.messages = list(chat["messages"])
                    st.rerun()

            with row[1]:
                if st.button(pin_icon, key=f"pin_{chat_id}"):
                    chat["pinned"] = not chat.get("pinned", False)
                    st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.markdown("---")

    # ---- Knowledge ----
    st.html("<div class='nav-section-label'>📚 Knowledge Base</div>")

    if st.session_state.uploaded_documents:
        for doc in st.session_state.uploaded_documents.values():
            st.html(
                upload_card(
                    filename=doc["filename"],
                    size_label=f"{doc.get('chunks', '?')} chunks",
                    status_label="indexed",
                )
            )
    else:
        st.caption("No documents indexed yet.")

    # ---- Uploads ----
    st.html("<div class='nav-section-label'>📤 Upload Documents</div>")

    uploaded_files = st.file_uploader(
        "Drag & drop or browse -- PDF, DOCX, TXT",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                if file_hash in st.session_state["uploaded_documents"]:
                    continue

                validation = validate_upload(
                    uploaded_file.name, file_bytes
                )

                if not validation.is_valid:
                    st.error(
                        f"{uploaded_file.name} rejected: "
                        f"{validation.reason}"
                    )
                    continue

                index_placeholder = st.empty()
                stages = [
                    "Extracting text",
                    "Chunking document",
                    "Generating embeddings",
                    "Indexing",
                ]

                index_placeholder.html(loading_pipeline(stages, 0))

                pages = extract_uploaded_text(uploaded_file)

                index_placeholder.html(loading_pipeline(stages, 2))

                result = run_async(
                    upload_document_async(uploaded_file.name, pages)
                )

                index_placeholder.empty()

                st.session_state["uploaded_documents"][file_hash] = {
                    "filename": uploaded_file.name,
                    "chunks": result["chunks_created"],
                    "timestamp": str(datetime.now(UTC)),
                }

                st.success(
                    f"{uploaded_file.name}: "
                    f"{result['chunks_created']} chunks indexed."
                )

            except Exception as exc:
                logger.exception("Upload failed.")
                st.error(str(exc))

    st.markdown("---")

    # ---- Settings ----
    st.html("<div class='nav-section-label'>⚙️ Settings</div>")

    use_rag = st.toggle("Enable RAG", value=True)
    enable_tools = st.toggle("Enable Tools", value=True)
    developer_mode = st.toggle(
        "🛠️ Developer Mode",
        value=False,
        help="Show the live diagnostics dashboard under each response.",
    )

    with st.expander("Model & Router Info", expanded=False):
        st.caption(f"Primary model: `{settings.primary_model}`")
        st.caption(f"Embedding model: `{settings.embedding_model}`")
        st.caption(
            f"Query router: "
            f"{'on' if settings.enable_query_router else 'off'}"
        )
        st.caption(
            f"Corrective RAG threshold: "
            f"{settings.corrective_rag_threshold}"
        )

    if st.button(
        "🗑️ Reset Vector Database", use_container_width=True
    ):
        try:
            run_async(router.vector_store.reset_collection())
            st.session_state.uploaded_documents = {}
            st.success("Vector database cleared.")
        except Exception:
            logger.exception("Vector reset failed.")
            st.error("Failed to reset vector database.")

    with st.expander("System Metrics", expanded=False):
        try:
            metrics = SystemMonitor.collect_metrics()

            st.caption(f"CPU: {metrics.cpu_percent:.1f}%")
            st.caption(
                f"RAM: {metrics.ram_used_gb:.1f}/"
                f"{metrics.ram_total_gb:.1f} GB"
            )

            if metrics.gpu_name:
                st.caption(
                    f"GPU: {metrics.gpu_utilization_percent:.1f}% "
                    f"({metrics.gpu_name})"
                )

        except Exception:
            logger.exception("Metrics collection failed.")

# =========================================================
# MAIN: Hero + Chat Window
# =========================================================

try:
    health = run_async(router.health_check())
    gemini_online = health.gemini_connected
    indexed_docs = health.vector_store_documents
except Exception:
    logger.exception("Health check failed.")
    gemini_online = False
    indexed_docs = 0

st.html(
    hero_banner(
        title="AI Workspace",
        subtitle=(
            "Document QA · Web Search · Weather · Finance · "
            "Reasoning -- routed automatically"
        ),
        status_items=[
            (
                "🟢" if gemini_online else "🔴",
                "Gemini",
                "Online" if gemini_online else "Offline",
            ),
            ("📄", "Knowledge sources", str(indexed_docs)),
            ("🧩", "Model", settings.primary_model),
            (
                "💬",
                "Conversations",
                str(len(st.session_state.saved_chats) + 1),
            ),
            ("🧠", "Memory", f"{len(st.session_state.messages)} msgs"),
            ("📡", "API calls", str(st.session_state.api_call_count)),
            (
                "⚡",
                "Last latency",
                (
                    f"{st.session_state.last_latency_ms:.0f}ms"
                    if st.session_state.last_latency_ms
                    else "--"
                ),
            ),
            ("🌎", "Env", settings.app_env),
        ],
    )
)

for idx, message in enumerate(st.session_state.messages):
    avatar = "✨" if message["role"] == "assistant" else "🧑"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("route"):
            render_assistant_extras(
                message, developer_mode, message_key=f"hist_{idx}"
            )

toolbar_cols = st.columns([1, 1, 10])

with toolbar_cols[0]:
    if st.button("📎", help="Attach a file -- use the sidebar uploader"):
        st.toast("Use the Knowledge Base uploader in the sidebar.")

with toolbar_cols[1]:
    st.button(
        "🎤", help="Voice input -- coming soon", disabled=True
    )

prompt = st.chat_input("Ask anything...")
is_regenerate = False

pending_regenerate = st.session_state.pop("regenerate_requested", None)

if pending_regenerate:
    prompt = pending_regenerate
    is_regenerate = True

if prompt and not rate_limiter.allow(st.session_state["session_id"]):
    st.chat_message("user", avatar="🧑").markdown(prompt)
    st.chat_message("assistant", avatar="✨").error(
        "Rate limit exceeded. Please wait a moment before sending "
        "another message."
    )

elif prompt:
    if is_regenerate:
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
    else:
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "timestamp": str(datetime.now(UTC)),
            }
        )

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        pipeline_placeholder = st.empty()

        pipeline_placeholder.html(
            loading_pipeline(
                ["Routing query", "Gathering context", "Generating answer"],
                0,
            )
        )

        try:
            gen, gather, total_start, immediate = run_async(
                stream_chat_async(
                    query=prompt,
                    use_rag=use_rag,
                    enable_tools=enable_tools,
                    history=conversation_history,
                )
            )

            if immediate is not None:
                pipeline_placeholder.empty()
                st.markdown(immediate.response)
                final_meta = immediate.model_dump()

            else:
                pipeline_placeholder.html(
                    loading_pipeline(
                        [
                            "Routing query",
                            "Gathering context",
                            "Generating answer",
                        ],
                        2,
                    )
                )

                gen_start = time.perf_counter()

                full_text = st.write_stream(
                    to_sync_iterator(lambda: gen)
                )

                generation_latency_ms = (
                    time.perf_counter() - gen_start
                ) * 1000

                pipeline_placeholder.empty()

                final_response = run_async(
                    router.finalize_stream(
                        gather,
                        full_text,
                        generation_latency_ms,
                        total_start,
                    )
                )

                final_meta = final_response.model_dump()

            st.session_state.api_call_count += 1
            st.session_state.last_route = final_meta.get("route")
            st.session_state.last_latency_ms = final_meta.get(
                "total_latency_ms"
            )
            st.session_state.last_confidence = final_meta.get(
                "confidence_score"
            )

            new_message_key = f"live_{len(st.session_state.messages)}"

            render_assistant_extras(
                final_meta, developer_mode, message_key=new_message_key
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_meta.get(
                        "response", "No response generated."
                    ),
                    "timestamp": str(datetime.now(UTC)),
                    **{
                        k: final_meta.get(k)
                        for k in (
                            "route",
                            "route_reason",
                            "route_confidence",
                            "sources",
                            "confidence_score",
                            "tools_used",
                            "retrieved_documents",
                            "retrieved_context",
                            "corrective_fallback",
                            "reranked",
                            "web_results_used",
                            "router_latency_ms",
                            "retrieval_latency_ms",
                            "generation_latency_ms",
                            "total_latency_ms",
                        )
                    },
                }
            )

        except Exception as exc:
            logger.exception("Chat processing failed.")
            pipeline_placeholder.empty()
            st.error(f"System Error: {exc}")

# =========================================================
# STATUS BAR
# =========================================================

st.html(
    status_bar_html(
        gemini_online=gemini_online,
        route=st.session_state.last_route,
        latency_ms=st.session_state.last_latency_ms,
        confidence=st.session_state.last_confidence,
    )
)
