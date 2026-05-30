from __future__ import annotations

import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class VectorStore:
    """
    ChromaDB vector storage manager.
    """

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={
                "description": "Enterprise RAG collection"
            },
        )

        logger.info(
            "Vector store initialized."
        )

    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Add documents into vector database.
        """

        if not ids:
            ids = [
                str(uuid.uuid4())
                for _ in texts
            ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Inserted %s documents into vector store.",
            len(texts),
        )

    async def similarity_search(
        self,
        embedding: list[float],
        top_k: int | None = None,
    ) -> dict:
        """
        Retrieve semantically similar documents.
        """

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k or settings.top_k_results,
        )

        return results

    async def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete vectorized document.
        """

        self.collection.delete(
            ids=[document_id]
        )

    async def reset_collection(self) -> None:
        """
        Remove all stored embeddings.
        """

        self.client.delete_collection(
            settings.chroma_collection
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection
        )

        logger.warning(
            "Vector store reset completed."
        )

    async def count(self) -> int:
        """
        Total vector count.
        """

        return self.collection.count()