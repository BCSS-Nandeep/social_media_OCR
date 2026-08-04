"""FastAPI application exposing the existing PaddleOCR pipeline.

    uvicorn src.api:app --reload

The OCR itself is untouched — this layer only handles transport, concurrency
and lifecycle. Models are built during startup and reused for every request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .models.response_models import ErrorResponse, HealthResponse
from .routes.ocr import router as ocr_router
from .services.engine_pool import EngineBusy, registry
from .utils.logging_config import configure

log = logging.getLogger("ocr.api")

DESCRIPTION = """
REST wrapper around a PaddleOCR pipeline for Indian-language images.

* **Automatic script detection** — `lang=auto` identifies the script per image
  and reports the probe scores, so the choice is auditable.
* **Thirteen languages across six scripts** — Devanagari (Hindi, Marathi,
  Nepali, Sanskrit, …), Telugu, Tamil, Kannada, Perso-Arabic (Urdu, Kashmiri,
  Sindhi) and Latin. See `GET /languages`.
* **Parallel batches** — a fixed pool of pre-loaded engines; models are never
  rebuilt per request.

Bengali, Assamese, Malayalam, Gujarati, Punjabi, Odia, Santali and Manipuri
have **no PaddleOCR recognition model at any version** and are rejected
explicitly rather than silently read with the wrong script.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure()
    log.info("starting OCR API")
    registry.startup()          # models load here, once
    try:
        yield
    finally:
        log.info("shutting down")
        registry.shutdown()


app = FastAPI(
    title="Social Media OCR API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "Payload too large"},
        422: {"model": ErrorResponse, "description": "Image could not be processed"},
        503: {"model": ErrorResponse, "description": "All OCR engines busy"},
    },
)

app.include_router(ocr_router)


@app.exception_handler(EngineBusy)
async def engine_busy_handler(request: Request, exc: EngineBusy) -> JSONResponse:
    """Saturation is a 503 with Retry-After, not a 500: the request was fine,
    the server is simply at capacity."""
    return JSONResponse(status_code=503, headers={"Retry-After": "30"},
                        content=ErrorResponse(error=str(exc)).model_dump())


@app.get("/health", response_model=HealthResponse, tags=["ops"],
         summary="Liveness and pool state")
async def health() -> HealthResponse:
    status = registry.status()
    return HealthResponse(
        status="ok", gpu=settings.use_gpu, default_lang=settings.default_lang,
        pool_size=settings.pool_size, pools_loaded=registry.loaded,
        engines_ready=status,
    )
