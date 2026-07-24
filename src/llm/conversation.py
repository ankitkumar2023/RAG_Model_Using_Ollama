from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

RoleType = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: RoleType
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Conversation:
    """
    Maintains structured chat history.
    """

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add_message(
        self,
        role: RoleType,
        content: str,
    ) -> None:
        self.messages.append(
            Message(
                role=role,
                content=content,
            )
        )

    def get_messages(self) -> list[Message]:
        return self.messages

    def clear(self) -> None:
        self.messages.clear()

    def last_message(self) -> Message | None:
        if not self.messages:
            return None

        return self.messages[-1]

    def to_prompt(self) -> str:
        """
        Convert structured history into LLM prompt format.
        """

        lines: list[str] = []

        for msg in self.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")

        return "\n".join(lines)

    def trim(self, max_messages: int) -> None:
        """
        Prevent unbounded memory growth.
        """

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
