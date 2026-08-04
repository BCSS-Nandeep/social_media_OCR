"""End-to-end: image in -> ordered text blocks with confidences out."""

from __future__ import annotations

import time
from pathlib import Path

from .exporter import OCRResult
from .ocr_engine import OCREngine
from .preprocess import PreprocessConfig, load_image, preprocess
from .reading_order import order_blocks, render_text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"No such file or directory: {target}")
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def process_image(path: Path, engine: OCREngine, pre_config: PreprocessConfig,
                  min_confidence: float = 0.0) -> tuple[OCRResult, "object"]:
    """Process one image. Returns the result and the (preprocessed) image array."""
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    original = load_image(str(path))
    timings["load"] = time.perf_counter() - t0
    height, width = original.shape[:2]

    t0 = time.perf_counter()
    image, scale, applied = preprocess(original, pre_config)
    timings["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    blocks, per_lang = engine.run(image)
    timings["ocr"] = time.perf_counter() - t0
    timings.update({f"ocr:{lang}": secs for lang, secs in per_lang.items()})

    # Boxes come back in preprocessed-image space; report original coordinates.
    blocks = [b.scaled(scale) for b in blocks]

    kept = [b for b in blocks if b.confidence >= min_confidence]
    dropped = len(blocks) - len(kept)

    t0 = time.perf_counter()
    ordered, line_numbers = order_blocks(kept)
    text = render_text(ordered, line_numbers)
    timings["ordering"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - started

    result = OCRResult(
        image_path=str(path),
        image_size=(width, height),
        blocks=ordered,
        line_numbers=line_numbers,
        text=text,
        langs=engine.detected_langs or engine.langs,
        preprocessing=applied,
        timings=timings,
        dropped_low_confidence=dropped,
    )
    return result, image
