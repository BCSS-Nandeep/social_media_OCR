"""API configuration, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]


@dataclass(frozen=True)
class Settings:
    # --------------------------------------------------------------- engines
    # Default language spec. "auto" detects the script per image.
    default_lang: str = os.environ.get("OCR_DEFAULT_LANG", "auto")

    # Engines held per language spec. PaddleOCR predictors are NOT thread-safe,
    # so concurrency comes from having several engines rather than sharing one.
    # Each engine in auto mode holds a detector, six probe recognisers and one
    # full pipeline -- roughly 350 MB. Raise deliberately.
    pool_size: int = _int("OCR_POOL_SIZE", 2)

    # Distinct language specs kept resident. Each new spec builds its own pool
    # and keeps it forever, so an unbounded registry would be a memory leak
    # driven by request input.
    max_pools: int = _int("OCR_MAX_POOLS", 4)

    # Seconds a request waits for a free engine before giving up with 503.
    acquire_timeout: float = float(os.environ.get("OCR_ACQUIRE_TIMEOUT", "120"))

    use_gpu: bool = _bool("OCR_USE_GPU", False)
    warmup_on_startup: bool = _bool("OCR_WARMUP", True)

    # --------------------------------------------------------------- batching
    max_workers: int = _int("OCR_MAX_WORKERS", 0)        # 0 -> derive from pool
    max_batch_files: int = _int("OCR_MAX_BATCH_FILES", 200)
    max_upload_bytes: int = _int("OCR_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
    max_zip_entries: int = _int("OCR_MAX_ZIP_ENTRIES", 500)
    max_zip_uncompressed: int = _int("OCR_MAX_ZIP_UNCOMPRESSED", 512 * 1024 * 1024)

    # ------------------------------------------------------------ filesystem
    # /ocr/folder reads a server-side path chosen by the caller. Left empty it
    # will read anywhere the service account can reach, which is a directory
    # traversal primitive. Set OCR_ALLOWED_ROOTS in any deployment that is not
    # a single-user machine.
    allowed_roots: list[Path] = field(default_factory=lambda: _paths("OCR_ALLOWED_ROOTS"))

    log_level: str = os.environ.get("OCR_LOG_LEVEL", "INFO").upper()

    @property
    def folder_endpoint_is_restricted(self) -> bool:
        return bool(self.allowed_roots)


settings = Settings()
