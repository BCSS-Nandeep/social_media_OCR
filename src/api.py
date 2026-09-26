"""HTTP API for social-media image OCR and video description.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Both image and video requests go to the same, already-running vLLM process
serving Qwen2.5-VL-7B-Instruct (VLLM_BASE_URL) -- this process never loads a
model itself. Nothing here (image or video) touches a GPU directly; every
inference call is an HTTP request to vLLM, which owns the model and does its
own scheduling/batching.

IndicOCR (src/ocr_engine.py, src/pipeline.py) is kept in the codebase as a
fallback, not a default: `src/ocr_providers.py`'s IndicOCRProvider builds
and warms its worker pool lazily, on the FIRST call to it, which only
happens if Qwen's OCR call raises AND INDICOCR_FALLBACK_ENABLED is set. A
normal deployment where Qwen never fails never puts a byte of IndicOCR on
the GPU -- `nvidia-smi` after startup should show only vLLM's allocation.
See DEPLOYMENT.md for the measured before/after.

Video requests are bounded by MAX_CONCURRENT_VIDEO_JOBS -- vLLM does its own
request scheduling/batching on the GPU it owns, so this only bounds local
CPU work (download, ffprobe, frame extraction), not GPU concurrency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import media_downloader
from .ocr_providers import IndicOCRProvider, QwenOCRProvider
from .video_service import VideoProcessingError, describe_video
from .vlm_client import VLMClient


def _configure_logging() -> None:
    """Without this, every log.info()/log.warning() call in this app
    (including the ones that already existed) is silently dropped -- no
    handler was ever configured anywhere in the hierarchy, so Python's
    logging module falls back to its "last resort" handler, which only
    prints WARNING and above. That meant startup messages never appeared,
    and the only thing PM2's logs ever showed for a failure was a raw
    traceback from log.exception (ERROR level clears the WARNING bar) with
    no surrounding request context. This makes INFO-level request tracing
    (see extract()) actually visible in `pm2 logs social-media-ocr-api`."""
    if logging.getLogger().handlers:
        return   # already configured (e.g. a second import under --reload)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()   # stdout -> PM2's own -out.log
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)


_configure_logging()
log = logging.getLogger("ocr.api")


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MAX_FETCH_BYTES = 25 * 1024 * 1024   # 25 MB, matches the CLI's own sanity range
FETCH_TIMEOUT_S = 15

# --------------------------------------------------------------- OCR engine
# OCR_ENGINE is informational/reserved for now: Qwen is always the default
# and only engine tried first. It exists so /health and logs can say which
# engine is configured without a second env var to keep in sync.
OCR_ENGINE = os.environ.get("OCR_ENGINE", "qwen")
INDICOCR_FALLBACK_ENABLED = _bool("INDICOCR_FALLBACK_ENABLED", False)
OCR_POOL_SIZE = int(os.environ.get("OCR_POOL_SIZE", "1"))   # fallback pool size, if ever built

# --------------------------------------------------------------- video / VLM
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
VLLM_TIMEOUT_SECONDS = float(os.environ.get("VLLM_TIMEOUT_SECONDS", "120"))

MAX_VIDEO_SIZE_MB = float(os.environ.get("MAX_VIDEO_SIZE_MB", "500"))
MAX_VIDEO_SIZE_BYTES = int(MAX_VIDEO_SIZE_MB * 1024 * 1024)
MAX_VIDEO_DURATION_SECONDS = float(os.environ.get("MAX_VIDEO_DURATION_SECONDS", "3600"))
VIDEO_DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("VIDEO_DOWNLOAD_TIMEOUT_SECONDS", "60"))
FRAME_EXTRACTION_TIMEOUT_SECONDS = float(os.environ.get("FRAME_EXTRACTION_TIMEOUT_SECONDS", "120"))
MAX_CONCURRENT_VIDEO_JOBS = int(os.environ.get("MAX_CONCURRENT_VIDEO_JOBS", "1"))

_video_semaphore: asyncio.Semaphore | None = None
_vlm_client: VLMClient | None = None
_qwen_provider: QwenOCRProvider | None = None
# Constructed eagerly (cheap: no GPU work, see ocr_providers.py), but its
# internal pool stays None -- and IndicOCR stays off the GPU -- until the
# first fallback call actually happens.
_indicocr_provider = IndicOCRProvider(OCR_POOL_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _video_semaphore, _vlm_client, _qwen_provider
    _video_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VIDEO_JOBS)
    _vlm_client = VLMClient(VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY, VLLM_TIMEOUT_SECONDS)
    _qwen_provider = QwenOCRProvider(_vlm_client)

    log.info("OCR engine: qwen (default) via %s (model=%s); IndicOCR fallback %s",
             VLLM_BASE_URL, VLLM_MODEL,
             "enabled" if INDICOCR_FALLBACK_ENABLED else "disabled")
    log.info("no GPU model loaded in this process -- vLLM owns the model, "
             "IndicOCR (if ever used) builds lazily on first fallback")
    yield


app = FastAPI(
    title="Social Media OCR API",
    description="Extracts text from social-media images and chronological "
                "descriptions from videos, both via a self-hosted Qwen2.5-VL "
                "model served through vLLM. IndicOCR is retained as an "
                "optional, lazily-loaded fallback for images.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    image_url: str | None = Field(None, description="http(s) URL of the image to fetch.")
    image_base64: str | None = Field(None, description="Raw image bytes, base64-encoded.")
    video_url: str | None = Field(None, description="http(s) URL of the video to fetch and describe.")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0,
                                  description="IndicOCR-fallback only -- Qwen reports no "
                                              "per-block confidence to filter on.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ExtractRequest":
        sources = (self.image_url, self.image_base64, self.video_url)
        if sum(bool(s) for s in sources) != 1:
            raise ValueError(
                "Provide exactly one of image_url, image_base64 or video_url.")
        return self


class ExtractResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool          # default engine (Qwen/vLLM) reachable
    pool_size: int              # IndicOCR fallback pool's configured size
    workers_available: int      # 0 until the fallback pool is actually built
    vlm_available: bool
    vllm_model: str
    ocr_engine: str
    indicocr_loaded: bool
    video_available: bool


def _fetch_url(url: str) -> bytes:
    try:
        return media_downloader.fetch_url(url, max_bytes=MAX_FETCH_BYTES,
                                          timeout=FETCH_TIMEOUT_S).data
    except media_downloader.UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except media_downloader.PayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except media_downloader.FetchFailed as exc:
        raise HTTPException(status_code=422,
                            detail=f"Could not fetch image_url: {exc}") from exc


def _decode_base64(image_base64: str) -> bytes:
    try:
        return media_downloader.decode_base64(image_base64, max_bytes=MAX_FETCH_BYTES)
    except media_downloader.InvalidBase64 as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    except media_downloader.PayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


def _fetch_video_url(url: str) -> bytes:
    try:
        return media_downloader.fetch_url(url, max_bytes=MAX_VIDEO_SIZE_BYTES,
                                          timeout=VIDEO_DOWNLOAD_TIMEOUT_SECONDS).data
    except media_downloader.UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except media_downloader.PayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except media_downloader.FetchFailed as exc:
        raise HTTPException(status_code=422,
                            detail=f"Could not fetch video_url: {exc}") from exc


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    vlm_ok = await _vlm_client.health() if _vlm_client else False
    # DEBUG, not INFO -- monitoring polls this frequently; set LOG_LEVEL=DEBUG
    # to see these when actually chasing a health-check-specific issue.
    log.debug("health check vlm_available=%s indicocr_loaded=%s", vlm_ok, _indicocr_provider.loaded)
    return HealthResponse(
        status="ok", model_loaded=vlm_ok,
        pool_size=OCR_POOL_SIZE, workers_available=0,
        vlm_available=vlm_ok, vllm_model=VLLM_MODEL,
        ocr_engine=OCR_ENGINE, indicocr_loaded=_indicocr_provider.loaded,
        video_available=vlm_ok,
    )


@app.post("/extract", response_model=ExtractResponse, tags=["ocr"],
         summary="Extract text from an image, or a description from a video")
async def extract(request: ExtractRequest, http_request: Request) -> ExtractResponse:
    request_id = uuid.uuid4().hex[:8]
    client_ip = http_request.client.host if http_request.client else "unknown"
    if request.video_url:
        media_kind, media_ref = "video_url", request.video_url
    elif request.image_url:
        media_kind, media_ref = "image_url", request.image_url
    else:
        # Never log the actual base64 payload -- only its length.
        media_kind = "image_base64"
        media_ref = f"<{len(request.image_base64 or '')} chars>"

    log.info("[%s] received client=%s media=%s ref=%s",
             request_id, client_ip, media_kind, media_ref)
    t0 = time.perf_counter()
    try:
        if request.video_url:
            response = await _extract_video(request.video_url, request_id)
        else:
            response = await _extract_image(request, request_id)
    except HTTPException as exc:
        elapsed = time.perf_counter() - t0
        log.warning("[%s] failed status=%d duration=%.3fs error=%s",
                   request_id, exc.status_code, elapsed, exc.detail)
        raise

    elapsed = time.perf_counter() - t0
    engine = (response.data or {}).get("engine") or (response.data or {}).get("media_type", "unknown")
    log.info("[%s] completed success=%s engine=%s duration=%.3fs",
             request_id, response.success, engine, elapsed)
    return response


async def _extract_image(request: ExtractRequest, request_id: str = "") -> ExtractResponse:
    image_bytes = _fetch_url(request.image_url) if request.image_url \
        else _decode_base64(request.image_base64)
    filename = "image_url" if request.image_url else "image_base64"

    try:
        data = await _qwen_provider.extract(image_bytes, filename, request.min_confidence)
        return ExtractResponse(success=True, data=data)
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured error, not a 500 trace
        if not INDICOCR_FALLBACK_ENABLED:
            log.exception("[%s] Qwen OCR failed (fallback disabled)", request_id)
            raise HTTPException(status_code=422, detail=f"Qwen OCR failed: {exc}") from exc

        log.warning("[%s] Qwen OCR failed, falling back to IndicOCR: %s", request_id, exc)
        try:
            data = await _indicocr_provider.extract(image_bytes, filename, request.min_confidence)
            return ExtractResponse(success=True, data=data)
        except Exception as fallback_exc:  # noqa: BLE001
            log.exception("[%s] IndicOCR fallback also failed", request_id)
            raise HTTPException(
                status_code=422,
                detail=f"Qwen OCR failed ({exc}) and IndicOCR fallback also failed: "
                       f"{fallback_exc}") from fallback_exc


async def _extract_video(video_url: str, request_id: str = "") -> ExtractResponse:
    t0 = time.perf_counter()
    video_bytes = _fetch_video_url(video_url)
    download_time = time.perf_counter() - t0

    async with _video_semaphore:
        try:
            data = await describe_video(
                video_bytes,
                vlm_client=_vlm_client,
                max_duration_seconds=MAX_VIDEO_DURATION_SECONDS,
                frame_extraction_timeout_seconds=FRAME_EXTRACTION_TIMEOUT_SECONDS,
                download_time_seconds=download_time,
            )
        except VideoProcessingError as exc:
            log.exception("[%s] video processing failed", request_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExtractResponse(success=True, data=data)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content=ExtractResponse(success=False, error=str(exc.detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Keep the same {success, error} envelope for request-shape errors too
    (e.g. no source given, or more than one), not FastAPI's default
    {"detail": [...]} shape -- callers shouldn't need two error formats."""
    message = "; ".join(e["msg"] for e in exc.errors())
    log.warning("request validation failed client=%s path=%s error=%s",
               request.client.host if request.client else "unknown", request.url.path, message)
    return JSONResponse(status_code=422,
                        content=ExtractResponse(success=False, error=message).model_dump())


PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
