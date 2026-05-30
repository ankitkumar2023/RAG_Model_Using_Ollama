from __future__ import annotations

import logging
from dataclasses import dataclass

from src.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenCounter:
    """
    Production token accounting utility.
    """

    def __init__(self) -> None:
        self.tokenizer = Tokenizer()

    def count_text_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count tokens in text.
        """

        return self.tokenizer.count_tokens(
            text
        )

    def calculate_stats(
        self,
        prompt: str,
        completion: str,
    ) -> TokenStats:
        """
        Calculate prompt/completion stats.
        """

        prompt_tokens = (
            self.count_text_tokens(prompt)
        )

        completion_tokens = (
            self.count_text_tokens(completion)
        )

        total_tokens = (
            prompt_tokens
            + completion_tokens
        )

        logger.info(
            "Token stats calculated | "
            "Prompt=%s | "
            "Completion=%s | "
            "Total=%s",
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        return TokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Safely truncate content.
        """

        return self.tokenizer.truncate_text(
            text,
            max_tokens,
        )