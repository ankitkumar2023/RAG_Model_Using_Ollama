from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def timing_decorator(
    name: str | None = None,
) -> Callable:
    """
    Sync execution timing decorator.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            start = time.perf_counter()

            try:
                return func(*args, **kwargs)

            finally:
                duration = (
                    time.perf_counter() - start
                )

                logger.info(
                    "Execution completed | "
                    "Function=%s | "
                    "Duration=%.4fs",
                    name or func.__name__,
                    duration,
                )

        return wrapper

    return decorator


def async_timing_decorator(
    name: str | None = None,
) -> Callable:
    """
    Async execution timing decorator.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            start = time.perf_counter()

            try:
                return await func(
                    *args,
                    **kwargs,
                )

            finally:
                duration = (
                    time.perf_counter() - start
                )

                logger.info(
                    "Async execution completed | "
                    "Function=%s | "
                    "Duration=%.4fs",
                    name or func.__name__,
                    duration,
                )

        return wrapper

    return decorator


class Timer:
    """
    Manual performance timer.
    """

    def __init__(self) -> None:
        self.start_time = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        return (
            time.perf_counter()
            - self.start_time
        )


async def sleep_yield() -> None:
    """
    Cooperative async yielding.
    """

    await asyncio.sleep(0)