
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
        self.embedding_generator = (
            embedding_generator
        )

        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents.
        """

        logger.info(
            "Starting retrieval for query: %s",
            query,
        )

        # ==========================================
        # GENERATE QUERY EMBEDDING
        # ==========================================

        query_embedding = (
            await self.embedding_generator.embed_text(
                query
            )
        )

        # ==========================================
        # VECTOR SEARCH
        # ==========================================

        raw_results = (
            await self.vector_store.similarity_search(
                embedding=query_embedding,
                top_k=top_k,
            )
        )

        documents = raw_results.get(
            "documents",
            [[]],
        )[0]

        metadatas = raw_results.get(
            "metadatas",
            [[]],
        )[0]

        distances = raw_results.get(
            "distances",
            [[]],
        )[0]

        logger.info(
            "Raw retrieval results count: %s",
            len(documents),
        )

        retrieved_docs: list[
            RetrievedDocument
        ] = []

        # ==========================================
        # PROCESS RETRIEVED DOCUMENTS
        # ==========================================

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            try:
                similarity_score = float(distance)

            except Exception:
                similarity_score = 0.0

            logger.info(
                "Retrieved document | "
                "Score=%.4f | "
                "Source=%s | "
                "Preview=%s",
                similarity_score,
                metadata.get(
                    "source",
                    "unknown",
                ),
                text[:200],
            )

            # IMPORTANT:
            # Do NOT aggressively filter
            # local embedding results.
            #
            # Local embedding models often
            # produce weaker similarity scores
            # than OpenAI embeddings.

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

        if not retrieved_docs:
            return ""

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
                f"Similarity: "
                f"{doc.score:.4f}\n\n"
                f"{doc.text}"
            )

            context_blocks.append(block)

        final_context = "\n\n".join(
            context_blocks
        )

        logger.info(
            "Built context with %s documents.",
            len(retrieved_docs),
        )

        return final_context

