"""Minimal HTTP API around the existing IndicOCR pipeline, plus video
description via a self-hosted Qwen2.5-VL model served through vLLM.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

One route does the whole job: image or video in, extracted content out.
The image path is untouched from before -- image_url/image_base64 still go
straight through process_image() against the resident IndicOCR pool exactly
as before (see git history if you need the pre-video diff). video_url is
new: frames are sampled from the video and sent to a separate vLLM server
(its own process, its own GPU allocation) for a chronological description.
The two paths share only this transport layer and the media_downloader's
safe-fetch logic -- IndicOCR's worker pool and vLLM's model are two
independent GPU-resident resources, never mixed, never loaded into this
process.

Requests are served by a pool of OCR_POOL_SIZE independent IndicOCR
instances (each its own copy of the layout + recognition models, resident
in GPU memory for the life of the process) sharing one asyncio.Queue: a
request checks a worker out, uses it, and returns it -- no worker is ever
shared between two in-flight requests, so the crop-config mutation
ocr_engine.py does internally per call stays safe. A request never gets
rejected for capacity -- if every worker is busy, it waits in the queue
rather than failing.

DEFAULT IS 1, DELIBERATELY, ON A SINGLE-GPU BOX. It's tempting to assume
more worker instances means more throughput; measured against this A10G it
is the opposite. Firing 4 concurrent requests at a 4-worker pool took 75s
total (each request individually took ~75s, i.e. ~6.5x its solo time);
the identical 4 requests against a 1-worker pool (pure FIFO queueing) took
47s total, each finishing in its own ~12s turn. A single GPU without MPS
does not give independent CUDA contexts real parallelism for compute-bound
work -- they context-switch and contend for the same SMs, so concurrent
"workers" here bought nothing but overhead. Raise OCR_POOL_SIZE above 1
only if this ever runs across multiple GPUs (one worker per GPU) or the
recogniser gains a real batched-inference path; on one GPU it will make
things slower, not faster. See DEPLOYMENT.md's Concurrency section for the
measurements this is based on.

Video requests are bounded separately by MAX_CONCURRENT_VIDEO_JOBS. vLLM
does its own request scheduling/batching on the GPU it owns, so this only
bounds local CPU work (download, ffprobe, frame extraction) -- it is not a
GPU concurrency control the way OCR_POOL_SIZE is.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
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
from .ocr_engine import OCREngine
from .pipeline import process_image
from .preprocess import PreprocessConfig
from .video_service import VideoProcessingError, describe_video
from .vlm_client import VLMClient

log = logging.getLogger("ocr.api")

MAX_FETCH_BYTES = 25 * 1024 * 1024   # 25 MB, matches the CLI's own sanity range
FETCH_TIMEOUT_S = 15
POOL_SIZE = int(os.environ.get("OCR_POOL_SIZE", "1"))

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

_pool: asyncio.Queue[OCREngine] | None = None
_video_semaphore: asyncio.Semaphore | None = None
_vlm_client: VLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _video_semaphore, _vlm_client
    _pool = asyncio.Queue()
    _video_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VIDEO_JOBS)
    _vlm_client = VLMClient(VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY, VLLM_TIMEOUT_SECONDS)

    loop = asyncio.get_event_loop()
    log.info("loading %d IndicOCR worker(s) (first run also downloads the "
             "gated model weights)...", POOL_SIZE)
    for i in range(POOL_SIZE):
        engine = OCREngine()
        await loop.run_in_executor(None, engine.warmup)
        await _pool.put(engine)
        log.info("worker %d/%d ready", i + 1, POOL_SIZE)
    log.info("all %d IndicOCR workers ready", POOL_SIZE)
    log.info("VLM client configured for %s (model=%s) -- vLLM runs as its "
             "own process, not loaded here", VLLM_BASE_URL, VLLM_MODEL)
    yield


app = FastAPI(
    title="Social Media OCR API",
    description="Extracts text from social-media images via IndicOCR, and "
                "chronological descriptions from videos via a self-hosted "
                "Qwen2.5-VL model served through vLLM.",
    version="1.1.0",
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
                                  description="Drop layout blocks scoring below this (images only).")

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
    model_loaded: bool
    pool_size: int
    workers_available: int
    vlm_available: bool
    vllm_model: str


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
    return HealthResponse(status="ok", model_loaded=_pool is not None,
                          pool_size=POOL_SIZE,
                          workers_available=_pool.qsize() if _pool else 0,
                          vlm_available=vlm_ok, vllm_model=VLLM_MODEL)


@app.post("/extract", response_model=ExtractResponse, tags=["ocr"],
         summary="Extract text from an image, or a description from a video")
async def extract(request: ExtractRequest) -> ExtractResponse:
    if request.video_url:
        return await _extract_video(request.video_url)
    return await _extract_image(request)


async def _extract_image(request: ExtractRequest) -> ExtractResponse:
    image_bytes = _fetch_url(request.image_url) if request.image_url \
        else _decode_base64(request.image_base64)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    engine = await _pool.get()  # waits here if every worker is busy -- never rejects
    try:
        result, _ = await asyncio.get_event_loop().run_in_executor(
            None, process_image, tmp_path, engine, PreprocessConfig(),
            request.min_confidence)
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured error, not a 500 trace
        log.exception("extraction failed")
        raise HTTPException(status_code=422,
                            detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        _pool.put_nowait(engine)  # back in rotation regardless of success/failure

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return ExtractResponse(success=True, data=result.to_dict())


async def _extract_video(video_url: str) -> ExtractResponse:
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
            log.exception("video processing failed")
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
    return JSONResponse(status_code=422,
                        content=ExtractResponse(success=False, error=message).model_dump())


PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
