from __future__ import annotations

import pytest

from src.llm.ollama_client import (
    OllamaClient,
)
from src.rag.chunking import (
    TextChunker,
)
from src.rag.embeddings import (
    EmbeddingGenerator,
)


def test_chunking() -> None:
    chunker = TextChunker(
        chunk_size=20,
        overlap=5,
    )

    text = (
        "This is a long document used "
        "for chunking tests."
    )

    chunks = chunker.split_text(
        text
    )

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_embeddings() -> None:
    client = OllamaClient()

    embedding_generator = (
        EmbeddingGenerator(client)
    )

    embedding = (
        await embedding_generator.embed_text(
            "Hello world"
        )
    )

    assert isinstance(
        embedding,
        list,
    )

    assert len(embedding) > 0

    await client.close()