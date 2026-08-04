"""Request schemas and the shared OCR options."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..config import settings
from ..scripts import UnsupportedLanguage, resolve_models


class OCROptions(BaseModel):
    """Knobs shared by every endpoint.

    Defaults mirror the CLI's measured defaults, so the API and
    `run_ocr.py` produce identical output for the same image.
    """

    lang: str = Field(default=settings.default_lang,
                      description="'auto' to detect the script per image, or "
                                  "'+'-separated languages such as 'te', 'hi+en'. "
                                  "Languages sharing a script share one pass.")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0,
                                  description="Drop blocks below this score. 0 keeps "
                                              "everything, including reads PaddleOCR "
                                              "would normally hide.")
    include_blocks: bool = Field(True, description="Set false for text only; useful "
                                                   "for large batches.")

    @field_validator("lang")
    @classmethod
    def _known_language(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("lang must not be empty")
        if value.lower() == "auto":
            return "auto"
        codes = [c for c in value.split("+") if c.strip()]
        if not codes:
            raise ValueError("lang must name at least one language")
        try:
            resolve_models(codes)          # rejects Bengali, Malayalam, typos, ...
        except UnsupportedLanguage as exc:
            raise ValueError(str(exc)) from exc
        return value


class FolderRequest(OCROptions):
    path: str = Field(..., description="Server-side directory to read images from.")
    recursive: bool = Field(True, description="Search sub-directories too.")

    model_config = {"json_schema_extra": {"example": {
        "path": "D:/images", "lang": "auto", "recursive": True,
    }}}
