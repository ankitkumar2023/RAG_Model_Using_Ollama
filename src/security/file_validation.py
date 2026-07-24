from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import get_settings

settings = get_settings()


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""


def validate_upload(filename: str, file_bytes: bytes) -> ValidationResult:
    """
    Validate an uploaded file before it enters the ingestion pipeline:
    extension allow-list, non-empty, and a size ceiling.
    """

    suffix = Path(filename).suffix.lower()

    if suffix not in settings.allowed_upload_extensions:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"'{suffix}' is not an allowed file type ({', '.join(settings.allowed_upload_extensions)})."
            ),
        )

    if not file_bytes:
        return ValidationResult(is_valid=False, reason="File is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > settings.max_upload_size_mb:
        return ValidationResult(
            is_valid=False,
            reason=(f"File is {size_mb:.1f}MB, exceeds the {settings.max_upload_size_mb}MB limit."),
        )

    return ValidationResult(is_valid=True)
