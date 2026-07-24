"""
Minimal example: chunk a document, embed and index it, then retrieve
against a question. Requires GEMINI_API_KEY to be set (see
.env.example).

Run with: python -m src.examples.rag_demo
"""

from __future__ import annotations

import asyncio
import uuid

from src.llm.gemini_client import GeminiClient
from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retrieval import Retriever
from src.rag.vector_store import VectorStore

DOCUMENT = """
Retrieval-Augmented Generation (RAG) is an AI architecture
that combines vector retrieval systems with large language
models. RAG improves factual grounding and reduces
hallucinations by injecting retrieved context into prompts.
"""


async def main() -> None:
    client = GeminiClient()
    chunker = TextChunker()
    vector_store = VectorStore()
    embedding_generator = EmbeddingGenerator()

    retriever = Retriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        gemini_client=client,
    )

    document_id = str(uuid.uuid4())

    chunks = chunker.split_text(
        DOCUMENT,
        metadata={"source": "demo", "page": 1},
    )

    texts = [chunk.text for chunk in chunks]

    embeddings = await embedding_generator.embed_batch(texts)

    ids = [f"{document_id}_{chunk.chunk_id}" for chunk in chunks]

    await vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "source": "demo",
                "chunk_id": chunk.chunk_id,
                "chunk_uid": chunk_uid,
                "page": 1,
            }
            for chunk, chunk_uid in zip(chunks, ids, strict=True)
        ],
        ids=ids,
    )

    retrieved = await retriever.retrieve("What is RAG?")

    print("\nRetrieved Documents:\n")

    for doc in retrieved:
        print(f"[{doc.score:.2f}] {doc.source}: {doc.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
