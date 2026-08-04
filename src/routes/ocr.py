"""OCR endpoints. Every one delegates to the existing pipeline unchanged."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import settings
from ..models.request_models import FolderRequest, OCROptions
from ..models.response_models import (BatchResponse, FailedItem, LanguageInfo,
                                      OCRResponse)
from ..scripts import (KNOWN_WEAK, LANGUAGE_SCRIPTS, UNSUPPORTED_SCRIPTS,
                       UnsupportedLanguage, resolve_models)
from ..services.batch_processor import run_many, run_one, worker_count
from ..services.engine_pool import EngineBusy, registry
from ..utils.files import UnsafePath, extract_zip, list_images, resolve_folder, safe_name

log = logging.getLogger("ocr.api")
router = APIRouter(tags=["ocr"])


# --------------------------------------------------------------------- helpers

def _validate_lang(lang: str) -> str:
    lang = (lang or settings.default_lang).strip()
    if lang.lower() == "auto":
        return "auto"
    try:
        resolve_models([c for c in lang.split("+") if c.strip()])
    except UnsupportedLanguage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return lang


def _save_upload(upload: UploadFile, folder: Path) -> Path:
    """Stream an upload to disk, enforcing the size cap as we go.

    Size is checked while copying rather than from any client-supplied header,
    which cannot be trusted.
    """
    target = folder / safe_name(upload.filename)
    written = 0
    with target.open("wb") as sink:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename} exceeds "
                           f"{settings.max_upload_bytes / 1e6:.0f} MB.")
            sink.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return target


def _busy(exc: EngineBusy) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------- routes

@router.post("/ocr", response_model=OCRResponse, summary="OCR a single image")
async def ocr_single(
    file: UploadFile = File(..., description="Image file (jpg, png, webp, bmp, tif)."),
    lang: str = Form(settings.default_lang,
                     description="'auto', or '+'-separated languages e.g. 'hi+en'."),
    min_confidence: float = Form(0.0, ge=0.0, le=1.0),
    include_blocks: bool = Form(True),
) -> OCRResponse:
    """Extract text from one image.

    With `lang=auto` the script is detected from the image itself and reported
    in `language`, alongside the per-script probe scores.
    """
    lang = _validate_lang(lang)
    with tempfile.TemporaryDirectory(prefix="ocr-single-") as tmp:
        path = _save_upload(file, Path(tmp))
        try:
            return run_one(path, lang=lang, min_confidence=min_confidence,
                           include_blocks=include_blocks,
                           display_name=safe_name(file.filename))
        except EngineBusy as exc:
            raise _busy(exc) from exc
        except Exception as exc:                        # noqa: BLE001
            log.exception("single image failed: %s", file.filename)
            raise HTTPException(status_code=422,
                                detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/ocr/batch", response_model=BatchResponse,
             summary="OCR many uploaded images in parallel")
async def ocr_batch(
    files: list[UploadFile] = File(..., description="Two or more image files."),
    lang: str = Form(settings.default_lang),
    min_confidence: float = Form(0.0, ge=0.0, le=1.0),
    include_blocks: bool = Form(True),
) -> BatchResponse:
    """Process an upload set concurrently.

    A file that fails is reported in `failures` and the rest still run.
    """
    lang = _validate_lang(lang)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=413,
            detail=f"{len(files)} files exceeds the limit of "
                   f"{settings.max_batch_files}. Use /ocr/zip or /ocr/folder.")

    with tempfile.TemporaryDirectory(prefix="ocr-batch-") as tmp:
        staged: list[tuple[Path, str]] = []
        failures: list[FailedItem] = []
        for upload in files:
            name = safe_name(upload.filename)
            try:
                staged.append((_save_upload(upload, Path(tmp)), name))
            except HTTPException as exc:
                failures.append(FailedItem(filename=name, error=str(exc.detail)))

        results, run_failures, elapsed, workers = run_many(
            staged, lang=lang, min_confidence=min_confidence,
            include_blocks=include_blocks) if staged else ([], [], 0.0,
                                                           worker_count(1))
        failures.extend(run_failures)

    return BatchResponse(
        success=not failures, total=len(files), processed=len(results),
        failed=len(failures), total_processing_time=round(elapsed, 3),
        workers=workers, results=results, failures=failures,
    )


@router.post("/ocr/folder", response_model=BatchResponse,
             summary="OCR every image in a server-side folder")
async def ocr_folder(request: FolderRequest) -> BatchResponse:
    """Read a directory **on the server** and OCR everything in it.

    The path is resolved on the machine running this service, not the caller's.
    Restrict it with `OCR_ALLOWED_ROOTS` outside a single-user setup.
    """
    try:
        folder = resolve_folder(request.path)
    except UnsafePath as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    images = list_images(folder, recursive=request.recursive)
    if not images:
        raise HTTPException(status_code=404,
                            detail=f"No images found under {folder}.")
    if len(images) > settings.max_batch_files:
        raise HTTPException(
            status_code=413,
            detail=f"{len(images)} images exceeds the limit of "
                   f"{settings.max_batch_files}. Raise OCR_MAX_BATCH_FILES.")

    results, failures, elapsed, workers = run_many(
        [(p, p.name) for p in images], lang=request.lang,
        min_confidence=request.min_confidence,
        include_blocks=request.include_blocks)

    return BatchResponse(
        success=not failures, total=len(images), processed=len(results),
        failed=len(failures), total_processing_time=round(elapsed, 3),
        workers=workers, results=results, failures=failures,
    )


@router.post("/ocr/zip", response_class=FileResponse,
             summary="OCR a ZIP of images, get a ZIP of results back")
async def ocr_zip(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP archive containing images."),
    lang: str = Form(settings.default_lang),
    min_confidence: float = Form(0.0, ge=0.0, le=1.0),
) -> FileResponse:
    """Returns a ZIP containing `results.json` and one `.txt` per image.

    Response is a file download, not JSON.
    """
    lang = _validate_lang(lang)
    workspace = Path(tempfile.mkdtemp(prefix="ocr-zip-"))
    # Registered now so the directory is removed even if this request fails.
    background.add_task(shutil.rmtree, workspace, ignore_errors=True)

    archive = workspace / safe_name(file.filename or "upload.zip")
    written = 0
    with archive.open("wb") as sink:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"ZIP exceeds {settings.max_upload_bytes / 1e6:.0f} MB.")
            sink.write(chunk)

    try:
        images = extract_zip(archive, workspace / "images")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not images:
        raise HTTPException(status_code=400,
                            detail="ZIP contained no supported image files.")

    results, failures, elapsed, workers = run_many(
        [(p, p.name) for p in images], lang=lang,
        min_confidence=min_confidence, include_blocks=True)

    payload = BatchResponse(
        success=not failures, total=len(images), processed=len(results),
        failed=len(failures), total_processing_time=round(elapsed, 3),
        workers=workers, results=results, failures=failures,
    )

    out_dir = workspace / "out"
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        payload.model_dump_json(indent=2), encoding="utf-8")
    for item in results:
        (out_dir / "text" / f"{Path(item.filename).stem}.txt").write_text(
            item.text, encoding="utf-8")

    bundle = workspace / "ocr_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))

    return FileResponse(bundle, media_type="application/zip",
                        filename="ocr_results.zip")


@router.get("/languages", response_model=LanguageInfo,
            summary="Supported languages and scripts")
async def languages() -> LanguageInfo:
    """Which languages this service can read, and which it cannot.

    The unsupported list is not a to-do: PaddleOCR ships no recognition model
    for those scripts at any version.
    """
    by_script: dict[str, list[str]] = {}
    for code, script in sorted(LANGUAGE_SCRIPTS.items()):
        by_script.setdefault(script, []).append(code)
    return LanguageInfo(scripts=by_script, weak=sorted(KNOWN_WEAK),
                        unsupported=dict(sorted(UNSUPPORTED_SCRIPTS.items())))
