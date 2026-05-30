from __future__ import annotations

from typing import Any

from config.settings import get_settings


settings = get_settings()


DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "temperature": settings.temperature,
    "top_p": settings.top_p,
    "top_k": settings.top_k,
    "repeat_penalty": settings.repeat_penalty,
    "num_predict": settings.max_tokens,
}


GUARD_GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.1,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "num_predict": 256,
}