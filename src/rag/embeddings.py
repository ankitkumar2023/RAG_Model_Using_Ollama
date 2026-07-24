from __future__ import annotations

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class EmbeddingGenerator:
    """
    Async embedding generation pipeline, backed by Gemini embeddings.
    """

    def __init__(self) -> None:
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key.get_secret_value(),
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (e.g. a query).
        """

        return await self.embedding_function.aembed_query(text)

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """
        Batch embedding generation for document ingestion.
        """

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            batch_embeddings = await self.embedding_function.aembed_documents(batch)

            embeddings.extend(batch_embeddings)

            logger.info(
                "Embedded batch %s/%s",
                i + len(batch),
                len(texts),
            )

        return embeddings
