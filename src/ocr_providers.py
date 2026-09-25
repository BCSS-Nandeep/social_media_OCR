"""Two ways to turn an image into text, behind the same small interface.

QwenOCRProvider is the default: it costs nothing until the first request
(it only holds an HTTP client reference) and every call goes to the
already-running vLLM process -- no model is loaded here.

IndicOCRProvider exists for one reason: if Qwen ever fails and
INDICOCR_FALLBACK_ENABLED is set, something still answers the request. It is
built lazily -- the pool of OCR_POOL_SIZE engines (see engine_pool-style
warmup in the old api.py) is only constructed the first time `.extract()` is
actually called, so a normal deployment where Qwen never fails never puts a
single byte of IndicOCR on the GPU. `OCREngine()` itself is free (see
ocr_engine.py's `_models` property) -- `warmup()`/`run()` are what actually
build the model, and this provider is the only thing left that calls them.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from .ocr_engine import OCREngine
from .pipeline import process_image
from .preprocess import PreprocessConfig
from .vlm_client import VLMClient, VLMError

log = logging.getLogger("ocr.providers")

_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]


def _sniff_media_type(data: bytes) -> str:
    """vLLM needs a real content type on the data URI, and callers here
    supply arbitrary image bytes -- sniff by magic number rather than
    trusting a filename extension that may not even exist (base64 input has
    none)."""
    for magic, media_type in _MAGIC_SIGNATURES:
        if data.startswith(magic):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"   # reasonable default; vLLM's decoder is lenient


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Best-effort width/height for the response's `image` block -- purely
    informational, so a decode failure here must not fail the request."""
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None, None
        height, width = image.shape[:2]
        return width, height
    except Exception:  # noqa: BLE001 - dimensions are a nice-to-have, not load-bearing
        return None, None


class QwenOCRProvider:
    """Default OCR path: one HTTP call to the already-running vLLM server."""

    def __init__(self, vlm_client: VLMClient):
        self._vlm = vlm_client

    async def extract(self, image_bytes: bytes, filename: str,
                      min_confidence: float = 0.0) -> dict[str, Any]:
        started = time.perf_counter()
        media_type = _sniff_media_type(image_bytes)
        width, height = _image_dimensions(image_bytes)

        try:
            result = await self._vlm.extract_text(image_bytes, media_type)
        except VLMError:
            raise   # let the caller decide whether to fall back

        elapsed = time.perf_counter() - started
        return {
            "engine": "qwen",
            "image": {"path": None, "name": filename, "width": width, "height": height},
            "settings": {"preprocessing": ["none"]},
            "summary": {
                "text_blocks_detected": len(result.blocks),
                # Qwen has no per-block recognition confidence to report --
                # null, not a fabricated number. See DEPLOYMENT.md.
                "mean_confidence": None,
                "min_confidence": None,
                # min_confidence filtering doesn't apply without a real score
                # to filter on; nothing is ever dropped on this path.
                "blocks_dropped_below_threshold": 0,
                "total_processing_time_sec": round(elapsed, 3),
                "stage_times_sec": {"ocr": round(elapsed, 3), "total": round(elapsed, 3)},
            },
            "full_text": result.full_text,
            "blocks": [
                {
                    "index": i,
                    "order": i,
                    "label": block["type"],
                    "type": block["type"],
                    "text": block["text"],
                    "confidence": None,
                    "bbox_xyxy": None,
                }
                for i, block in enumerate(result.blocks)
            ],
            "error": None,
        }


class IndicOCRProvider:
    """Fallback OCR path, built lazily. Nothing GPU-side happens until the
    first `.extract()` call actually runs -- see module docstring."""

    def __init__(self, pool_size: int, warmup: bool = True):
        self._pool_size = max(1, pool_size)
        self._warmup = warmup
        self._pool: asyncio.Queue[OCREngine] | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._pool is not None

    async def _ensure_pool(self) -> asyncio.Queue[OCREngine]:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                log.warning("Qwen OCR fallback triggered -- building %d IndicOCR "
                           "worker(s) now (first use only)", self._pool_size)
                pool: asyncio.Queue[OCREngine] = asyncio.Queue()
                loop = asyncio.get_event_loop()
                for i in range(self._pool_size):
                    engine = OCREngine()
                    if self._warmup:
                        await loop.run_in_executor(None, engine.warmup)
                    pool.put_nowait(engine)
                    log.info("IndicOCR fallback worker %d/%d ready", i + 1, self._pool_size)
                self._pool = pool
        return self._pool

    async def extract(self, image_bytes: bytes, filename: str,
                      min_confidence: float = 0.0) -> dict[str, Any]:
        pool = await self._ensure_pool()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        engine = await pool.get()
        try:
            result, _ = await asyncio.get_event_loop().run_in_executor(
                None, process_image, tmp_path, engine, PreprocessConfig(), min_confidence)
        finally:
            tmp_path.unlink(missing_ok=True)
            pool.put_nowait(engine)

        if result.error:
            raise RuntimeError(result.error)

        data = result.to_dict()
        data["engine"] = "indicocr"
        return data
