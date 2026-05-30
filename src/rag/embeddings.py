from __future__ import annotations

import asyncio
import logging

from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Async embedding generation pipeline.
    """

    def __init__(
        self,
        client: OllamaClient,
    ) -> None:
        self.client = client

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate embedding for single text.
        """

        return await self.client.embeddings(text)

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """
        Batch embedding generation.
        """

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            tasks = [
                self.embed_text(text)
                for text in batch
            ]

            batch_embeddings = await asyncio.gather(
                *tasks
            )

            embeddings.extend(batch_embeddings)

            logger.info(
                "Embedded batch %s/%s",
                i + len(batch),
                len(texts),
            )

        return embeddings