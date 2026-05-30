from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class RetrievedDocument:
    """
    Retrieved semantic match.
    """

    text: str
    score: float
    metadata: dict[str, Any]


class Retriever:
    """
    Semantic retrieval orchestrator.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents.
        """

        query_embedding = await (
            self.embedding_generator.embed_text(
                query
            )
        )

        raw_results = await (
            self.vector_store.similarity_search(
                embedding=query_embedding,
                top_k=top_k,
            )
        )

        documents = (
            raw_results.get("documents", [[]])[0]
        )

        metadatas = (
            raw_results.get("metadatas", [[]])[0]
        )

        distances = (
            raw_results.get("distances", [[]])[0]
        )

        retrieved_docs: list[
            RetrievedDocument
        ] = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            similarity_score = 1.0 - float(distance)

            if (
                similarity_score
                < settings.similarity_threshold
            ):
                continue

            retrieved_docs.append(
                RetrievedDocument(
                    text=text,
                    score=similarity_score,
                    metadata=metadata,
                )
            )

        logger.info(
            "Retrieved %s relevant documents.",
            len(retrieved_docs),
        )

        return retrieved_docs

    @staticmethod
    def build_context(
        retrieved_docs: list[RetrievedDocument],
    ) -> str:
        """
        Convert retrieved docs into LLM context.
        """

        context_blocks: list[str] = []

        for idx, doc in enumerate(
            retrieved_docs,
            start=1,
        ):
            source = doc.metadata.get(
                "source",
                "unknown",
            )

            block = (
                f"[Document {idx}]\n"
                f"Source: {source}\n"
                f"Similarity: {doc.score:.4f}\n\n"
                f"{doc.text}"
            )

            context_blocks.append(block)

        return "\n\n".join(context_blocks)