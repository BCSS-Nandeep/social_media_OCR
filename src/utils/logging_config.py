"""Logging setup. Records filename, language, timing, confidence and errors."""

from __future__ import annotations

import logging
import sys

from ..config import settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def configure(level: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)

    # PaddleOCR/PaddleX log a banner per model construction. At pool sizes
    # above one that repeats for every engine and buries our own lines.
    for noisy in ("ppocr", "paddle", "paddlex"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
