from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from config.settings import get_settings
from src.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class TextChunk:
    """
    Represents a semantic text chunk.
    """

    chunk_id: str
    text: str
    start_index: int
    end_index: int
    token_count: int
    metadata: dict


class TextChunker:
    """
    Production-grade text chunking utility.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

        self.tokenizer = Tokenizer()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize whitespace and formatting.
        """

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def split_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        """
        Split text into overlapping semantic chunks.
        """

        metadata = metadata or {}

        text = self.normalize_text(text)

        tokens = self.tokenizer.encode(text)

        chunks: list[TextChunk] = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk = TextChunk(
                chunk_id=f"chunk_{chunk_index}",
                text=chunk_text,
                start_index=start,
                end_index=end,
                token_count=len(chunk_tokens),
                metadata=metadata,
            )

            chunks.append(chunk)

            start += self.chunk_size - self.overlap
            chunk_index += 1

        logger.info(
            "Generated %s chunks.",
            len(chunks),
        )

        return chunks

    def batch_chunk_documents(
        self,
        documents: Iterable[tuple[str, dict]],
    ) -> list[TextChunk]:
        """
        Chunk multiple documents.
        """

        all_chunks: list[TextChunk] = []

        for text, metadata in documents:
            chunks = self.split_text(
                text=text,
                metadata=metadata,
            )

            all_chunks.extend(chunks)

        return all_chunks
