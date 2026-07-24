from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_EMBEDDING_MARKER_FILENAME = ".embedding_meta.json"

# Cosine distance is required so that raw Chroma distances translate to a
# clean [0, 1] similarity score for citations/confidence scoring (Chroma's
# default is squared L2, which is unbounded and not directly usable as a
# relevance score).
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class VectorStore:
    """
    ChromaDB vector storage manager, backed by LangChain's Chroma
    integration (embedding function attached so the retriever's
    LangChain retrievers can use this store directly via
    `.as_retriever()`).
    """

    def __init__(self) -> None:
        settings.vector_db_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key.get_secret_value(),
        )

        self.store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embedding_function,
            persist_directory=settings.vector_db_path,
            collection_metadata=_COLLECTION_METADATA,
        )

        # The underlying chromadb collection, used for the precomputed-
        # embedding read/write paths (documents are embedded upstream by
        # EmbeddingGenerator, so we bypass Chroma's own embed-on-write path
        # here to avoid a redundant embedding call).
        self.collection = self.store._collection

        self._guard_embedding_model_compatibility()

        logger.info("Vector store initialized.")

    def _marker_path(self) -> Path:
        return settings.vector_db_directory / _EMBEDDING_MARKER_FILENAME

    def _current_marker(self) -> dict[str, str]:
        return {
            "embedding_model": settings.embedding_model,
            "distance_space": _COLLECTION_METADATA["hnsw:space"],
        }

    def _guard_embedding_model_compatibility(self) -> None:
        """
        Reset the collection if it was populated with a different
        embedding model or distance space than currently configured.

        Different embedding models produce vectors of different
        dimensionality/semantic space, and a different distance space
        changes how raw distances map to similarity scores -- mixing
        either silently corrupts retrieval. This runs once per process
        on startup.
        """

        marker_path = self._marker_path()
        current_marker = self._current_marker()

        try:
            existing_count = self.collection.count()
        except Exception:
            existing_count = 0

        if marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                marker = {}

            if marker == current_marker:
                return

            logger.warning(
                "Vector store was built with %s but %s is now "
                "configured. Resetting the collection to avoid a "
                "dimension/space mismatch.",
                marker,
                current_marker,
            )

            self._reset_collection_sync()

        elif existing_count > 0:
            logger.warning(
                "Vector store contains %s documents with no recorded "
                "embedding metadata (likely from a prior setup). "
                "Resetting the collection.",
                existing_count,
            )

            self._reset_collection_sync()

        marker_path.write_text(
            json.dumps(current_marker),
            encoding="utf-8",
        )

    def _reset_collection_sync(self) -> None:
        client = self.store._client

        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            logger.debug(
                "Collection '%s' did not exist to delete.",
                settings.chroma_collection,
            )

        self._reconnect()

    def _reconnect(self) -> None:
        """
        (Re)connect to the collection without destroying data.

        `Chroma(...)` uses get-or-create semantics, so this is safe to
        call any time -- including as self-healing when the collection
        went missing out from under this instance (e.g. reset by a
        different process sharing the same on-disk store, such as
        another short-lived script calling `reset_collection()` while
        this one stays running).
        """

        self.store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embedding_function,
            persist_directory=settings.vector_db_path,
            collection_metadata=_COLLECTION_METADATA,
        )

        self.collection = self.store._collection

    def _call_with_reconnect(self, operation: Any) -> Any:
        """
        Run a zero-arg chromadb call, self-healing once on
        `NotFoundError` by reconnecting and retrying.
        """

        try:
            return operation()
        except NotFoundError:
            logger.warning(
                "Collection '%s' is missing (likely reset by another process); reconnecting and retrying.",
                settings.chroma_collection,
            )

            self._reconnect()

            return operation()

    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Add pre-embedded documents into the vector database.
        """

        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        self._call_with_reconnect(
            lambda: self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
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
        Retrieve semantically similar documents by precomputed embedding.
        """

        return self._call_with_reconnect(
            lambda: self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k or settings.top_k_results,
            )
        )

    async def delete_document(self, document_id: str) -> None:
        """
        Delete vectorized document.
        """

        self._call_with_reconnect(lambda: self.collection.delete(ids=[document_id]))

    async def reset_collection(self) -> None:
        """
        Remove all stored embeddings.
        """

        self._reset_collection_sync()

        marker_path = self._marker_path()
        marker_path.write_text(
            json.dumps(self._current_marker()),
            encoding="utf-8",
        )

        logger.warning("Vector store reset completed.")

    async def count(self) -> int:
        """
        Total vector count.
        """

        return self._call_with_reconnect(lambda: self.collection.count())

    async def get_embeddings(
        self,
        ids: list[str],
    ) -> dict[str, list[float]]:
        """
        Fetch stored embeddings for a set of chunk ids.

        Used by the retriever to score/rerank candidates against the
        query without paying for a redundant re-embedding call.
        """

        if not ids:
            return {}

        result = self._call_with_reconnect(
            lambda: self.collection.get(
                ids=ids,
                include=["embeddings"],
            )
        )

        returned_ids = result.get("ids", [])
        returned_embeddings = result.get("embeddings", [])

        return {
            doc_id: list(vector) for doc_id, vector in zip(returned_ids, returned_embeddings, strict=True)
        }
