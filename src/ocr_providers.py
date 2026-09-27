"""Two ways to turn an image into text, behind the same small interface --
only one of which is actually used.

QwenOCRProvider is the only engine `src/api.py` calls: it costs nothing
until the first request (it only holds an HTTP client reference) and every
call goes to the already-running vLLM process -- no model is loaded here.

IndicOCRProvider itself is DEAD CODE, kept as a reference implementation
only. `src/api.py` does not import it, construct it, or call it anywhere,
under any condition -- there is no fallback in this service. It's left in
place as a worked example of "how do I plug in another OCR engine behind
this same interface," not because anything in production depends on it.
Running it would still work exactly as described below (lazy pool build,
zero GPU cost until `.extract()` is actually called), but nothing calls
`.extract()` on this class anymore.

The IndicOCR chain it wraps (ocr_engine.py, pipeline.py, preprocess.py,
exporter.py) is a separate matter -- those modules are dead to the API
service too, but NOT dead code overall: run_ocr.py, the standalone CLI
tool, still calls them directly and works exactly as documented. Only
IndicOCRProvider (the API-facing wrapper) has zero remaining callers.
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
from .vlm_image_prep import downscale_for_vlm

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


def _prepare_for_vlm(data: bytes) -> tuple[bytes, str, int | None, int | None]:
    """Decode once: downscale if needed (see vlm_image_prep -- an unresized
    high-res photo can cost the model's entire context budget on its own,
    same reasoning as the video frame path), and report the dimensions of
    whatever is actually sent to the model. A decode failure here must not
    fail the request -- fall back to the original bytes/sniffed media type
    with unknown dimensions, and let the VLM itself reject it if it truly
    can't be read."""
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            return data, _sniff_media_type(data), None, None
        original_height, original_width = image.shape[:2]
        resized = downscale_for_vlm(image)
        if resized is image:
            return data, _sniff_media_type(data), original_width, original_height
        ok, buf = cv2.imencode(".jpg", resized)
        if not ok:
            return data, _sniff_media_type(data), original_width, original_height
        height, width = resized.shape[:2]
        return buf.tobytes(), "image/jpeg", width, height
    except Exception:  # noqa: BLE001 - resizing is a best-effort optimization
        return data, _sniff_media_type(data), None, None


class QwenOCRProvider:
    """Default OCR path: one HTTP call to the already-running vLLM server."""

    def __init__(self, vlm_client: VLMClient):
        self._vlm = vlm_client

    async def extract(self, image_bytes: bytes, filename: str,
                      min_confidence: float = 0.0) -> dict[str, Any]:
        started = time.perf_counter()
        image_bytes, media_type, width, height = _prepare_for_vlm(image_bytes)

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
    """DEAD CODE -- reference only, not called from src/api.py. See the
    module docstring. Left functional (lazy pool build, nothing GPU-side
    until `.extract()` actually runs) as a worked example, not because
    production depends on it."""

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
