from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Structured response parsing utilities.
    """

    TOOL_PATTERN = re.compile(
        r"<tool>(.*?)</tool>",
        re.DOTALL,
    )

    JSON_PATTERN = re.compile(
        r"\{.*\}",
        re.DOTALL,
    )

    @classmethod
    def extract_tool_calls(
        cls,
        response: str,
    ) -> list[dict[str, Any]]:
        """
        Extract tool calls embedded in XML tags.

        Example:

        <tool>
        {
            "tool": "calculator",
            "input": "2+2"
        }
        </tool>
        """

        tool_calls: list[dict[str, Any]] = []

        matches = cls.TOOL_PATTERN.findall(response)

        for match in matches:
            try:
                parsed = json.loads(match.strip())
                tool_calls.append(parsed)

            except json.JSONDecodeError:
                logger.warning("Invalid tool call JSON detected.")

        return tool_calls

    @classmethod
    def extract_json(
        cls,
        text: str,
    ) -> dict[str, Any] | None:
        """
        Extract first JSON object from response.
        """

        match = cls.JSON_PATTERN.search(text)

        if not match:
            return None

        try:
            return json.loads(match.group())

        except json.JSONDecodeError:
            return None

    @classmethod
    def clean_response(
        cls,
        text: str,
    ) -> str:
        """
        Clean assistant output.
        """

        text = text.strip()

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL,
        )

        return text.strip()
