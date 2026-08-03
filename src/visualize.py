"""Debug overlay: draw detected boxes, reading-order index and confidence.

Text is drawn with PIL rather than cv2.putText because cv2's Hershey fonts
cannot render Telugu glyphs at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .exporter import OCRResult

# Fonts with Telugu coverage, best first. Nirmala UI ships with Windows.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Nirmala.ttc",   # Windows ships this as a collection, not a .ttf
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\gautami.ttf",
    "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.90:
        return (46, 160, 67)     # green  - trustworthy
    if confidence >= 0.70:
        return (219, 154, 4)     # amber  - review
    return (218, 54, 51)         # red    - likely wrong


def save_overlay(image_bgr: np.ndarray, result: OCRResult, path: Path) -> Path:
    canvas = Image.fromarray(image_bgr[:, :, ::-1]).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(12, canvas.height // 60))

    for i, block in enumerate(result.blocks):
        color = _color(block.confidence)
        draw.polygon([tuple(p) for p in block.box], outline=color, width=2)
        label = f"{i}:{block.confidence:.2f}"
        x, y = block.x_min, max(0, block.y_min - font.size - 2)
        box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(box, fill=color)
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
