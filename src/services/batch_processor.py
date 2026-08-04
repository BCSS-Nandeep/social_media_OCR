"""Run the existing OCR pipeline over one or many images.

This is a wrapper, not a reimplementation: every image goes through the same
``process_image`` the CLI uses, so accuracy, script detection, language routing
and reading order are untouched.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path

from ..config import settings
from ..detect_script import CONFIDENT_MARGIN, margin
from ..exporter import OCRResult
from ..models.response_models import Block, FailedItem, OCRResponse
from ..pipeline import process_image
from ..preprocess import PreprocessConfig
from .engine_pool import EnginePool, registry

log = logging.getLogger("ocr.batch")

# Preprocessing stays off, matching the CLI. Sixteen configurations were
# measured and none moved accuracy outside +/-3%, so enabling any of it here
# would cost time for nothing.
PREPROCESS = PreprocessConfig()


def worker_count(n_images: int) -> int:
    """Never more threads than engines: extra threads would only queue.

    PaddleOCR predictors cannot be shared, so real concurrency is capped by the
    pool regardless of how many workers are spawned.
    """
    if settings.max_workers > 0:
        ceiling = settings.max_workers
    else:
        ceiling = min(cpu_count() or 1, settings.pool_size)
    return max(1, min(ceiling, max(1, n_images)))


def to_response(result: OCRResult, *, include_blocks: bool = True,
                filename: str | None = None) -> OCRResponse:
    """Map an OCRResult onto the API schema."""
    scores = result.extra.get("script_scores")
    detected = bool(result.extra.get("detected_script"))
    language = result.extra.get("detected_script") or (
        result.langs[0] if result.langs else "unknown")

    # A near-tie between the top two scripts means the detection was close to a
    # coin flip. Returning that silently would hand callers text read with the
    # wrong script and no way to know, so it is surfaced explicitly.
    gap = margin(scores) if scores else None
    confident = None if gap is None else gap >= CONFIDENT_MARGIN
    warnings: list[str] = []
    if confident is False:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        warnings.append(
            f"Script detection was not decisive: {ranked[0][0]} ({ranked[0][1]:.3f}) "
            f"barely beat {ranked[1][0]} ({ranked[1][1]:.3f}), margin {gap:.3f}. "
            f"The text may be read with the wrong script — re-run with an "
            f"explicit lang, or split a multi-script image."
        )

    blocks: list[Block] = []
    if include_blocks:
        for index, (block, line_no) in enumerate(zip(result.blocks,
                                                     result.line_numbers)):
            data = block.to_dict(index, line_no)
            blocks.append(Block(text=data["text"], confidence=data["confidence"],
                                bbox=data["bbox"], bbox_xyxy=data["bbox_xyxy"],
                                line=line_no))

    return OCRResponse(
        filename=filename or Path(result.image_path).name,
        language=language,
        languages=list(result.langs),
        detected=detected,
        script_scores=scores,
        detection_margin=None if gap is None else round(gap, 4),
        detection_confident=confident,
        warnings=warnings,
        processing_time=round(result.total_time, 3),
        mean_confidence=round(result.mean_confidence, 4),
        text_blocks=result.block_count,
        lines=len(set(result.line_numbers)),
        text=result.text,
        blocks=blocks,
    )


def run_one(path: Path, *, lang: str, min_confidence: float = 0.0,
            include_blocks: bool = True, display_name: str | None = None,
            pool: EnginePool | None = None) -> OCRResponse:
    """OCR a single image. Raises on failure; callers decide how to report it."""
    pool = pool or registry.pool(lang)
    name = display_name or path.name

    with pool.acquire() as engine:
        result, _ = process_image(path, engine, PREPROCESS, min_confidence)
        # Auto mode records its verdict on the engine, not the result.
        if engine.auto_mode and engine.detected_langs:
            result.extra["detected_script"] = engine.detected_langs[0]
            result.extra["script_scores"] = {k: round(v, 4)
                                             for k, v in engine.auto_scores.items()}

    response = to_response(result, include_blocks=include_blocks, filename=name)
    log.info("ok file=%s lang=%s blocks=%d conf=%.4f time=%.2fs",
             name, response.language, response.text_blocks,
             response.mean_confidence, response.processing_time)
    return response


def run_many(items: list[tuple[Path, str]], *, lang: str, min_confidence: float = 0.0,
             include_blocks: bool = True) -> tuple[list[OCRResponse],
                                                   list[FailedItem], float, int]:
    """OCR many images concurrently.

    One bad image never stops the batch — failures are collected and returned
    alongside the successes.

    Returns (results, failures, wall_seconds, workers). Results follow input
    order so callers can line them up with what they submitted.
    """
    started = time.perf_counter()
    workers = worker_count(len(items))
    pool = registry.pool(lang)

    ordered: list[OCRResponse | None] = [None] * len(items)
    failures: list[FailedItem] = []

    log.info("batch start images=%d workers=%d lang=%s", len(items), workers, lang)

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="ocr") as executor:
        futures = {
            executor.submit(run_one, path, lang=lang,
                            min_confidence=min_confidence,
                            include_blocks=include_blocks,
                            display_name=name, pool=pool): (index, name)
            for index, (path, name) in enumerate(items)
        }
        for future in as_completed(futures):
            index, name = futures[future]
            try:
                ordered[index] = future.result()
            except Exception as exc:                    # noqa: BLE001
                log.exception("failed file=%s", name)
                failures.append(FailedItem(
                    filename=name, error=f"{type(exc).__name__}: {exc}"))

    elapsed = time.perf_counter() - started
    results = [r for r in ordered if r is not None]
    log.info("batch done images=%d ok=%d failed=%d %.2fs",
             len(items), len(results), len(failures), elapsed)
    return results, failures, elapsed, workers
