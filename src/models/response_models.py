"""Response schemas. These are what Swagger documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Block(BaseModel):
    text: str = Field(..., description="Recognised text for this region.")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Recogniser score. High confidence is "
                                          "not proof of a correct read.")
    bbox: list[list[float]] = Field(..., description="Four corner points, "
                                                     "clockwise from top-left, in "
                                                     "original image coordinates.")
    bbox_xyxy: list[float] = Field(..., description="[x_min, y_min, x_max, y_max].")
    line: int = Field(..., description="1-based visual line, in reading order.")

    model_config = {"json_schema_extra": {"example": {
        "text": "जिल्हा न्यायालय", "confidence": 0.9712,
        "bbox": [[120, 88], [960, 88], [960, 190], [120, 190]],
        "bbox_xyxy": [120, 88, 960, 190], "line": 1,
    }}}


class OCRResponse(BaseModel):
    """One successfully processed image."""

    success: bool = True
    filename: str
    language: str = Field(..., description="Script actually used, e.g. 'devanagari'.")
    languages: list[str] = Field(default_factory=list,
                                 description="Every recognition pass applied.")
    detected: bool = Field(False, description="True when the script was auto-detected "
                                              "rather than supplied by the caller.")
    script_scores: dict[str, float] | None = Field(
        None, description="Per-script probe scores when auto-detecting.")
    detection_margin: float | None = Field(
        None, description="Gap between the best and second-best script.")
    detection_confident: bool | None = Field(
        None, description="False when the top two scripts are within "
                          "CONFIDENT_MARGIN of each other — the detection was "
                          "close to a coin flip and the text may be read with "
                          "the wrong script. Re-run with an explicit `lang`.")
    warnings: list[str] = Field(default_factory=list,
                                description="Non-fatal problems worth surfacing.")
    processing_time: float = Field(..., description="Seconds for this image.")
    mean_confidence: float
    text_blocks: int = 0
    lines: int = 0
    text: str
    blocks: list[Block] = Field(default_factory=list)

    model_config = {"json_schema_extra": {"example": {
        "success": True, "filename": "poster.jpg", "language": "telugu",
        "languages": ["telugu"], "detected": True,
        "script_scores": {"telugu": 0.959, "kannada": 0.769},
        "processing_time": 4.75, "mean_confidence": 0.9218,
        "text_blocks": 16, "lines": 12,
        "text": "తెలంగాణ ఉద్యమ చరిత్ర", "blocks": [],
    }}}


class FailedItem(BaseModel):
    """One image that could not be processed. The batch continues regardless."""

    filename: str
    success: bool = False
    error: str

    model_config = {"json_schema_extra": {"example": {
        "filename": "bad.jpg", "success": False, "error": "Cannot decode image.",
    }}}


class BatchResponse(BaseModel):
    success: bool = True
    total: int
    processed: int
    failed: int
    total_processing_time: float = Field(..., description="Wall-clock seconds for "
                                                          "the whole batch.")
    workers: int
    results: list[OCRResponse] = Field(default_factory=list)
    failures: list[FailedItem] = Field(default_factory=list)


class LanguageInfo(BaseModel):
    scripts: dict[str, list[str]] = Field(..., description="Script -> language codes.")
    weak: list[str] = Field(default_factory=list,
                            description="Scripts whose only model is a generation "
                                        "behind; expect lower accuracy.")
    unsupported: dict[str, str] = Field(
        default_factory=dict,
        description="Indian languages with no PaddleOCR model at any version.")


class HealthResponse(BaseModel):
    status: str
    gpu: bool
    default_lang: str
    pool_size: int
    pools_loaded: list[str]
    engines_ready: dict[str, int]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Any | None = None
