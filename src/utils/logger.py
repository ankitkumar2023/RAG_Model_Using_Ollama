from __future__ import annotations

import logging
from typing import Any

from config.logging_config import setup_logging

setup_logging()


class LoggerFactory:
    """
    Centralized logger factory.
    """

    @staticmethod
    def get_logger(
        name: str,
    ) -> logging.Logger:
        return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    message: str,
    exception: Exception,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Standardized exception logging.
    """

    logger.exception(
        "%s | Exception=%s | Extra=%s",
        message,
        str(exception),
        extra or {},
    )


def log_structured(
    logger: logging.Logger,
    level: int,
    message: str,
    **kwargs: Any,
) -> None:
    """
    Structured logging helper.
    """

    structured_data = " | ".join([f"{key}={value}" for key, value in kwargs.items()])

    logger.log(
        level,
        "%s | %s",
        message,
        structured_data,
    )
