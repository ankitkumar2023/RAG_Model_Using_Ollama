from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.rag import embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module.settings,
        "gemini_api_key",
        SecretStr("dummy-test-key"),
    )


def test_chunking() -> None:
    chunker = TextChunker(
        chunk_size=20,
        overlap=5,
    )

    text = "This is a long document used for chunking tests."

    chunks = chunker.split_text(text)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_embed_text() -> None:
    generator = EmbeddingGenerator()

    # GoogleGenerativeAIEmbeddings is a pydantic model, so individual
    # methods can't be monkeypatched on the real instance -- swap the
    # whole attribute for a plain mock instead.
    generator.embedding_function = MagicMock(aembed_query=AsyncMock(return_value=[0.1, 0.2, 0.3]))

    embedding = await generator.embed_text("Hello world")

    assert isinstance(embedding, list)
    assert len(embedding) > 0


@pytest.mark.asyncio
async def test_embed_batch() -> None:
    generator = EmbeddingGenerator()

    generator.embedding_function = MagicMock(
        aembed_documents=AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    )

    embeddings = await generator.embed_batch(["a", "b"])

    assert len(embeddings) == 2
