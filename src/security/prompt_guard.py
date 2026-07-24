from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all )?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|debug|dan|jailbreak) mode", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|hidden prompt|instructions)", re.IGNORECASE),
    re.compile(r"act as (if you (are|were)|an?) (unrestricted|uncensored|jailbroken)", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*script[\s>]", re.IGNORECASE),
]


@dataclass
class GuardResult:
    is_safe: bool
    matched_patterns: list[str] = field(default_factory=list)


def screen_prompt(text: str) -> GuardResult:
    """
    Fast deterministic screen for obvious prompt-injection/jailbreak
    patterns.

    This runs in addition to Gemini's own `safety_settings` (evaluated
    server-side on the actual generation call, see GeminiClient) --
    Llama Guard has no cloud equivalent, so input safety is now this
    two-layer app-side + provider-side approach.
    """

    matched = [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]

    if matched:
        logger.warning("Prompt guard matched patterns: %s", matched)

    return GuardResult(is_safe=not matched, matched_patterns=matched)
