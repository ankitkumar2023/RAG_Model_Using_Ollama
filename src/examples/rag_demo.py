from __future__ import annotations

import asyncio

from src.llm.ollama_client import OllamaClient
from src.rag.chunking import TextChunker
from src.rag.embeddings import (
    EmbeddingGenerator,
)
from src.rag.retrieval import Retriever
from src.rag.vector_store import VectorStore


DOCUMENT = """
Retrieval-Augmented Generation (RAG) is an AI architecture
that combines vector retrieval systems with large language
models. RAG improves factual grounding and reduces
hallucinations by injecting retrieved context into prompts.
"""


async def main() -> None:
    client = OllamaClient()

    chunker = TextChunker()

    vector_store = VectorStore()

    embedding_generator = (
        EmbeddingGenerator(client)
    )

    retriever = Retriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )

    chunks = chunker.split_text(
        DOCUMENT,
        metadata={
            "source": "demo",
        },
    )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = (
        await embedding_generator.embed_batch(
            texts
        )
    )

    await vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": "demo"
            }
            for _ in texts
        ],
    )

    retrieved = await retriever.retrieve(
        "What is RAG?"
    )

    print("\nRetrieved Documents:\n")

    for doc in retrieved:
        print(doc)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())