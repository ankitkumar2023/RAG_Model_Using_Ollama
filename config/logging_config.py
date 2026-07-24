from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from config.settings import get_settings

settings = get_settings()


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """
    Configure structured logging.
    """

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt=("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=LOG_DIR / "application.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            file_handler,
            console_handler,
        ],
    )

    # Reduce noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized.")
