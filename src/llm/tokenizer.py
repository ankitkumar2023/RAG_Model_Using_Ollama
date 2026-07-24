from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Token utility helper.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def truncate_text(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Safely truncate text to token limit.
        """

        tokens = self.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated = tokens[:max_tokens]

        return self.decode(truncated)
