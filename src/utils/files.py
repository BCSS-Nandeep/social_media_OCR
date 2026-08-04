"""Filesystem helpers for the API: uploads, folder access, ZIP extraction."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from ..config import settings
from ..pipeline import IMAGE_SUFFIXES

log = logging.getLogger(__name__)


class UnsafePath(ValueError):
    """A caller-supplied path resolved somewhere it is not allowed to reach."""


def safe_name(name: str) -> str:
    """Strip any directory component from a client-supplied filename."""
    cleaned = Path(str(name or "upload")).name
    return cleaned or "upload"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def resolve_folder(raw: str) -> Path:
    """Validate a server-side directory supplied by the caller.

    ``/ocr/folder`` reads whatever path the request names, which is a directory
    traversal primitive unless it is fenced in. ``OCR_ALLOWED_ROOTS`` fences it;
    with the variable unset the endpoint can read anywhere the service account
    can, which is only acceptable on a single-user machine.
    """
    if not raw or not raw.strip():
        raise UnsafePath("path must not be empty")

    try:
        folder = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafePath(f"Cannot resolve path: {raw}") from exc

    if not folder.is_dir():
        raise UnsafePath(f"Not a directory: {folder}")

    roots = settings.allowed_roots
    if roots and not any(folder == r or r in folder.parents for r in roots):
        raise UnsafePath(
            f"{folder} is outside the allowed roots "
            f"({', '.join(str(r) for r in roots)})."
        )
    if not roots:
        log.warning("OCR_ALLOWED_ROOTS is unset; /ocr/folder read %s", folder)
    return folder


def list_images(folder: Path, recursive: bool = True) -> list[Path]:
    walker = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walker if p.is_file() and is_image(p))


def extract_zip(archive: Path, destination: Path) -> list[Path]:
    """Extract image members of a ZIP, safely.

    Guards three ways an archive can attack the host:

    * **Zip slip** — a member named ``../../etc/passwd`` writing outside the
      destination. Every member is resolved and checked to stay inside.
    * **Zip bomb** — a small archive expanding to fill the disk. Capped by
      declared uncompressed size before anything is written.
    * **Entry flood** — hundreds of thousands of members. Capped by count.

    Non-image members, directories, symlinks and device entries are skipped
    rather than trusted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        bundle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a readable ZIP archive: {exc}") from exc

    with bundle:
        entries = bundle.infolist()
        if len(entries) > settings.max_zip_entries:
            raise ValueError(f"ZIP has {len(entries)} entries; limit is "
                             f"{settings.max_zip_entries}.")

        declared = sum(e.file_size for e in entries)
        if declared > settings.max_zip_uncompressed:
            raise ValueError(f"ZIP expands to {declared / 1e6:.0f} MB; limit is "
                             f"{settings.max_zip_uncompressed / 1e6:.0f} MB.")

        root = destination.resolve()
        for entry in entries:
            if entry.is_dir():
                continue
            # 0xA000 marks a symlink in the external attributes' Unix mode.
            if (entry.external_attr >> 16) & 0xA000 == 0xA000:
                log.warning("skipping symlink in zip: %s", entry.filename)
                continue

            name = safe_name(entry.filename)
            target = (root / name).resolve()
            if root not in target.parents and target.parent != root:
                log.warning("skipping zip member escaping destination: %s",
                            entry.filename)
                continue
            if not is_image(target):
                continue

            # Collisions after flattening (a/x.jpg and b/x.jpg both become
            # x.jpg) would silently drop an image, so disambiguate.
            if target.exists():
                stem, suffix = target.stem, target.suffix
                n = 1
                while target.exists():
                    target = root / f"{stem}_{n}{suffix}"
                    n += 1

            with bundle.open(entry) as source, target.open("wb") as sink:
                sink.write(source.read())
            extracted.append(target)

    return sorted(extracted)
