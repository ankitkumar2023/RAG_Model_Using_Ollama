from __future__ import annotations

import logging
from collections import deque

from config.settings import get_settings
from src.llm.conversation import Message

logger = logging.getLogger(__name__)

settings = get_settings()


class ConversationMemory:
    """
    Sliding-window conversational memory.
    """

    def __init__(
        self,
        max_history: int | None = None,
    ) -> None:
        self.max_history = max_history or settings.max_chat_history

        self._messages: deque[Message] = deque(maxlen=self.max_history)

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        for msg in messages:
            self._messages.append(msg)

    def clear(self) -> None:
        self._messages.clear()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def format_for_prompt(self) -> str:
        """
        Convert memory into structured prompt.
        """

        return "\n".join([f"{m.role.upper()}: {m.content}" for m in self._messages])

    def size(self) -> int:
        return len(self._messages)
