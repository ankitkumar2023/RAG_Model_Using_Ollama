from __future__ import annotations

import time
from collections import defaultdict, deque

from config.settings import get_settings

settings = get_settings()


class RateLimiter:
    """
    In-memory sliding-window rate limiter, keyed per session.

    In-memory and per-process by design -- adequate for a single
    Streamlit instance; a multi-instance deployment would need a shared
    store (e.g. Redis) instead.
    """

    def __init__(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        self.max_requests = max_requests or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]

        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            return False

        window.append(now)
        return True
