"""Minimal HTTP API around the existing IndicOCR pipeline.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

One route does the whole job: image in (by URL or base64), extracted content
out. The OCR pipeline itself (ocr_engine.py / pipeline.py) is untouched --
this module is pure transport glue, matching the same `process_image()` call
`run_ocr.py` makes.

IndicOCR is one GPU-resident model, not a pool of per-language engines like
the old PaddleOCR API had -- there is nothing to pool. Requests are
serialised through a single lock instead: concurrent calls into the same
loaded model haven't been verified safe, so this errs toward correctness
over throughput. See README/DEPLOYMENT.md for the scaling note.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from .ocr_engine import OCREngine
from .pipeline import process_image
from .preprocess import PreprocessConfig

log = logging.getLogger("ocr.api")

MAX_FETCH_BYTES = 25 * 1024 * 1024   # 25 MB, matches the CLI's own sanity range
FETCH_TIMEOUT_S = 15

_engine: OCREngine | None = None
_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    log.info("loading IndicOCR (first run downloads the gated model weights)...")
    _engine = OCREngine()
    await asyncio.get_event_loop().run_in_executor(None, _engine.warmup)
    log.info("IndicOCR ready")
    yield


app = FastAPI(
    title="Social Media OCR API",
    description="Extracts text from social-media poster images via IndicOCR.",
    version="1.0.0",
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    image_url: str | None = Field(None, description="http(s) URL of the image to fetch.")
    image_base64: str | None = Field(None, description="Raw image bytes, base64-encoded.")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0,
                                  description="Drop layout blocks scoring below this.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ExtractRequest":
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("Provide exactly one of image_url or image_base64.")
        return self


class ExtractResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _fetch_url(url: str) -> bytes:
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {scheme!r}")
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as resp:
            data = resp.read(MAX_FETCH_BYTES + 1)
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=422,
                            detail=f"Could not fetch image_url: {exc}") from exc
    if len(data) > MAX_FETCH_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"image_url exceeds {MAX_FETCH_BYTES // (1024*1024)} MB.")
    if not data:
        raise HTTPException(status_code=422, detail="image_url returned no content.")
    return data


def _decode_base64(image_base64: str) -> bytes:
    try:
        data = base64.b64decode(image_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    if len(data) > MAX_FETCH_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"image_base64 exceeds {MAX_FETCH_BYTES // (1024*1024)} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="image_base64 decoded to no content.")
    return data


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_engine is not None)


@app.post("/extract", response_model=ExtractResponse, tags=["ocr"],
         summary="Extract text from a social-media image")
async def extract(request: ExtractRequest) -> ExtractResponse:
    image_bytes = _fetch_url(request.image_url) if request.image_url \
        else _decode_base64(request.image_base64)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        async with _lock:  # one inference at a time -- see module docstring
            result, _ = await asyncio.get_event_loop().run_in_executor(
                None, process_image, tmp_path, _engine, PreprocessConfig(),
                request.min_confidence)
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured error, not a 500 trace
        log.exception("extraction failed")
        raise HTTPException(status_code=422,
                            detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return ExtractResponse(success=True, data=result.to_dict())


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content=ExtractResponse(success=False, error=str(exc.detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Keep the same {success, error} envelope for request-shape errors too
    (e.g. neither image_url nor image_base64 given), not FastAPI's default
    {"detail": [...]} shape -- callers shouldn't need two error formats."""
    message = "; ".join(e["msg"] for e in exc.errors())
    return JSONResponse(status_code=422,
                        content=ExtractResponse(success=False, error=message).model_dump())
