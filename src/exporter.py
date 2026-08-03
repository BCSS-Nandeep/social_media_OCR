"""Result container and TXT / JSON / CSV writers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ocr_engine import TextBlock


@dataclass
class OCRResult:
    image_path: str
    image_size: tuple[int, int]          # (width, height) of the original image
    blocks: list[TextBlock]
    line_numbers: list[int]
    text: str
    langs: list[str]
    preprocessing: list[str]
    timings: dict[str, float]            # seconds, per stage
    dropped_low_confidence: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def total_time(self) -> float:
        return self.timings.get("total", 0.0)

    @property
    def mean_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)

    @property
    def min_confidence(self) -> float:
        return min((b.confidence for b in self.blocks), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": {
                "path": self.image_path,
                "name": Path(self.image_path).name,
                "width": self.image_size[0],
                "height": self.image_size[1],
            },
            "settings": {
                "languages": self.langs,
                "preprocessing": self.preprocessing or ["none"],
            },
            "summary": {
                "text_blocks_detected": self.block_count,
                "lines_detected": len(set(self.line_numbers)),
                "mean_confidence": round(self.mean_confidence, 4),
                "min_confidence": round(self.min_confidence, 4),
                "blocks_dropped_below_threshold": self.dropped_low_confidence,
                "total_processing_time_sec": round(self.total_time, 3),
                "stage_times_sec": {k: round(v, 3) for k, v in self.timings.items()},
            },
            "full_text": self.text,
            "blocks": [b.to_dict(i, ln) for i, (b, ln)
                       in enumerate(zip(self.blocks, self.line_numbers))],
            "error": self.error,
            **self.extra,
        }


# ------------------------------------------------------------------- writers

def write_txt(result: OCRResult, path: Path) -> Path:
    header = [
        f"Image             : {Path(result.image_path).name}",
        f"Size              : {result.image_size[0]}x{result.image_size[1]}",
        f"Languages         : {', '.join(result.langs)}",
        f"Preprocessing     : {', '.join(result.preprocessing) or 'none'}",
        f"Text blocks       : {result.block_count}",
        f"Mean confidence   : {result.mean_confidence:.4f}",
        f"Processing time   : {result.total_time:.3f} s",
        "=" * 68,
        "EXTRACTED TEXT (reading order)",
        "=" * 68,
    ]
    body = [result.text, "", "=" * 68, "PER-BLOCK CONFIDENCE", "=" * 68,
            f"{'#':>4}  {'line':>4}  {'conf':>6}  {'model':>6}  text"]
    for i, (block, line_no) in enumerate(zip(result.blocks, result.line_numbers)):
        body.append(f"{i:>4}  {line_no:>4}  {block.confidence:>6.4f}  "
                    f"{block.lang:>6}  {block.text}")

    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return path


def write_json(result: OCRResult, path: Path) -> Path:
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_csv(result: OCRResult, path: Path) -> Path:
    # utf-8-sig so Excel renders Telugu instead of mojibake.
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "line", "text", "confidence", "lang_model",
                         "x_min", "y_min", "x_max", "y_max"])
        for i, (block, line_no) in enumerate(zip(result.blocks, result.line_numbers)):
            writer.writerow([i, line_no, block.text, f"{block.confidence:.4f}", block.lang,
                             round(block.x_min, 1), round(block.y_min, 1),
                             round(block.x_max, 1), round(block.y_max, 1)])
    return path


WRITERS = {"txt": write_txt, "json": write_json, "csv": write_csv}


def export(result: OCRResult, output_dir: Path, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    written = []
    for fmt in formats:
        writer = WRITERS.get(fmt)
        if writer is None:
            raise ValueError(f"Unsupported export format: {fmt}")
        written.append(writer(result, output_dir / f"{stem}.{fmt}"))
    return written


def write_batch_summary(results: list[OCRResult], path: Path) -> Path:
    payload = {
        "images_processed": len(results),
        "images_failed": sum(1 for r in results if r.error),
        "total_text_blocks": sum(r.block_count for r in results),
        "total_processing_time_sec": round(sum(r.total_time for r in results), 3),
        "mean_time_per_image_sec": round(
            sum(r.total_time for r in results) / len(results), 3) if results else 0.0,
        "images": [
            {
                "name": Path(r.image_path).name,
                "text_blocks": r.block_count,
                "mean_confidence": round(r.mean_confidence, 4),
                "processing_time_sec": round(r.total_time, 3),
                "error": r.error,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
