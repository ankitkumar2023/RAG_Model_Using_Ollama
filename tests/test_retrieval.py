from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from src.rag.retrieval import RetrievedDocument, Retriever


def _make_retriever() -> tuple[Retriever, MagicMock, MagicMock, MagicMock]:
    embedding_generator = MagicMock()
    embedding_generator.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    vector_store = MagicMock()
    vector_store.get_embeddings = AsyncMock(return_value={})

    gemini_client = MagicMock()
    gemini_client.agenerate = AsyncMock(return_value={"response": "standalone question"})
    gemini_client.get_langchain_llm = MagicMock()

    retriever = Retriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        gemini_client=gemini_client,
    )

    return retriever, embedding_generator, vector_store, gemini_client


def test_cosine_similarity_identical_vectors() -> None:
    retriever, *_ = _make_retriever()

    score = retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])

    assert score == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    retriever, *_ = _make_retriever()

    score = retriever._cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert score == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_safe() -> None:
    retriever, *_ = _make_retriever()

    score = retriever._cosine_similarity([0.0, 0.0], [1.0, 0.0])

    assert score == 0.0


def test_build_context_empty() -> None:
    retriever, *_ = _make_retriever()

    assert retriever.build_context([]) == ""


def test_build_context_includes_source_and_page() -> None:
    retriever, *_ = _make_retriever()

    docs = [
        RetrievedDocument(
            text="Some chunk text",
            score=0.87,
            source="handbook.pdf",
            page=3,
            metadata={"chunk_uid": "abc"},
        )
    ]

    context = retriever.build_context(docs)

    assert "handbook.pdf" in context
    assert "page 3" in context
    assert "Some chunk text" in context


@pytest.mark.asyncio
async def test_condense_query_without_history_returns_original() -> None:
    retriever, _, _, gemini_client = _make_retriever()

    result = await retriever.condense_query("what about it?", None)

    assert result == "what about it?"
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_condense_query_with_history_calls_llm() -> None:
    retriever, _, _, gemini_client = _make_retriever()

    history = [
        MagicMock(role="user", content="Tell me about the refund policy."),
        MagicMock(role="assistant", content="Refunds are processed in 5 days."),
    ]

    result = await retriever.condense_query("what about international orders?", history)

    assert result == "standalone question"
    gemini_client.agenerate.assert_called_once()


@pytest.mark.asyncio
async def test_condense_query_falls_back_on_error() -> None:
    retriever, _, _, gemini_client = _make_retriever()
    gemini_client.agenerate = AsyncMock(side_effect=RuntimeError("boom"))

    history = [MagicMock(role="user", content="hi")]

    result = await retriever.condense_query("original query", history)

    assert result == "original query"


@pytest.mark.asyncio
async def test_score_documents_uses_stored_embeddings() -> None:
    retriever, _, vector_store, _ = _make_retriever()

    vector_store.get_embeddings = AsyncMock(return_value={"chunk-1": [1.0, 0.0, 0.0]})

    documents = [Document(page_content="text", metadata={"chunk_uid": "chunk-1"})]

    scored = await retriever._score_documents([1.0, 0.0, 0.0], documents)

    assert len(scored) == 1
    assert scored[0][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rerank_disabled_returns_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever, _, _, gemini_client = _make_retriever()

    from src.rag import retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module.settings, "enable_reranking", False)

    doc = Document(page_content="text", metadata={})
    scored_docs = [(doc, 0.5)]

    result = await retriever._rerank("query", scored_docs)

    assert result == scored_docs
    gemini_client.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_compute_grounding_score_no_docs_returns_zero() -> None:
    retriever, *_ = _make_retriever()

    score = await retriever.compute_grounding_score("some answer", [])

    assert score == 0.0
