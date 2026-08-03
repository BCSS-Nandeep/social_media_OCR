#!/usr/bin/env python
"""PaddleOCR text extraction for Instagram / social-media poster images.

Examples
--------
    python run_ocr.py data/input/poster.jpg
    python run_ocr.py data/input --lang te+en --formats txt json csv
    python run_ocr.py poster.png --enhance-contrast --resize --visualize
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src.exporter import OCRResult, export, write_batch_summary
from src.ocr_engine import OCREngine
from src.pipeline import find_images, process_image
from src.preprocess import PreprocessConfig

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract text from poster images with PaddleOCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=Path,
                   help="Image file, or a directory of images (searched recursively).")
    p.add_argument("-o", "--output-dir", type=Path, default=ROOT / "outputs",
                   help="Where to write results (default: ./outputs).")
    p.add_argument("--lang", default="te+en",
                   help="Recognition language(s), '+'-separated. "
                        "'te'=Telugu, 'en'=English, 'te+en'=both passes merged (default).")
    p.add_argument("--formats", nargs="+", default=["txt", "json"],
                   choices=["txt", "json", "csv"],
                   help="Export formats (default: txt json).")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Drop text blocks scoring below this (0-1). Default 0 = keep all, "
                        "including PaddleOCR's own internally-filtered low scores.")
    p.add_argument("--det-box-thresh", type=float, default=0.5,
                   help="Detector box threshold; lower finds more/fainter text. Default 0.5.")
    p.add_argument("--det-limit", type=int, default=960,
                   help="Max side length the detector sees; larger images are "
                        "downscaled to this before detection. Default 960 (PaddleOCR's own).")
    p.add_argument("--unclip", type=float, default=1.5,
                   help="Box inflation before recognition cropping. Default 1.5.")
    p.add_argument("--no-angle-cls", action="store_true",
                   help="Disable the 180-degree text-line angle classifier.")
    p.add_argument("--gpu", action="store_true", help="Use GPU (needs paddlepaddle-gpu).")
    p.add_argument("--visualize", action="store_true",
                   help="Also write an annotated image with boxes and confidences.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-block console output.")

    pre = p.add_argument_group("preprocessing (all optional, all off by default)")
    pre.add_argument("--resize", action="store_true", help="Scale into the OCR-friendly range.")
    pre.add_argument("--min-side", type=int, default=960)
    pre.add_argument("--max-side", type=int, default=2560)
    pre.add_argument("--denoise", action="store_true", help="Edge-preserving denoise.")
    pre.add_argument("--enhance-contrast", action="store_true", help="CLAHE contrast lift.")
    pre.add_argument("--sharpen", action="store_true", help="Unsharp mask (mild).")
    pre.add_argument("--fix-orientation", action="store_true", help="Deskew small rotations.")
    return p


def report(result: OCRResult, quiet: bool) -> None:
    name = Path(result.image_path).name
    if result.error:
        print(f"  [FAILED] {name}: {result.error}")
        return

    print(f"  blocks={result.block_count}  lines={len(set(result.line_numbers))}  "
          f"mean_conf={result.mean_confidence:.4f}  time={result.total_time:.3f}s")
    if result.dropped_low_confidence:
        print(f"  dropped below threshold: {result.dropped_low_confidence}")
    if quiet:
        return
    for i, (block, line_no) in enumerate(zip(result.blocks, result.line_numbers)):
        print(f"    [{i:>3}] L{line_no:<3} {block.confidence:.4f}  {block.text}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    langs = [x.strip() for x in args.lang.split("+") if x.strip()]
    if not langs:
        print("error: --lang must name at least one language", file=sys.stderr)
        return 2

    try:
        images = find_images(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not images:
        print(f"error: no images found under {args.input}", file=sys.stderr)
        return 2

    pre_config = PreprocessConfig(
        resize=args.resize, min_side=args.min_side, max_side=args.max_side,
        denoise=args.denoise, enhance_contrast=args.enhance_contrast,
        sharpen=args.sharpen, fix_orientation=args.fix_orientation,
    )

    print(f"Images       : {len(images)}")
    print(f"Languages    : {' + '.join(langs)}")
    print(f"Preprocessing: {'enabled' if pre_config.any_enabled else 'none'}")
    print("Loading PaddleOCR models (first run downloads them)...")

    engine = OCREngine(langs, use_gpu=args.gpu,
                       use_angle_cls=not args.no_angle_cls,
                       det_db_box_thresh=args.det_box_thresh,
                       det_limit_side_len=args.det_limit,
                       det_db_unclip_ratio=args.unclip,
                       drop_score=0.0)  # filter once, in process_image, so the
                                        # "dropped" count reflects reality
    engine.warmup()

    results: list[OCRResult] = []
    for n, image_path in enumerate(images, start=1):
        print(f"\n[{n}/{len(images)}] {image_path.name}")
        try:
            result, processed = process_image(image_path, engine, pre_config,
                                              args.min_confidence)
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill a batch
            traceback.print_exc(limit=2)
            results.append(OCRResult(str(image_path), (0, 0), [], [], "", langs, [],
                                     {"total": 0.0}, error=f"{type(exc).__name__}: {exc}"))
            report(results[-1], args.quiet)
            continue

        report(result, args.quiet)
        for path in export(result, args.output_dir, args.formats):
            print(f"  -> {path}")

        if args.visualize:
            from src.visualize import save_overlay
            overlay = save_overlay(processed, result,
                                   args.output_dir / "overlays" / f"{image_path.stem}_boxes.png")
            print(f"  -> {overlay}")

        results.append(result)

    if len(results) > 1:
        summary = write_batch_summary(results, args.output_dir / "_batch_summary.json")
        total = sum(r.total_time for r in results)
        print(f"\nBatch: {len(results)} images, {sum(r.block_count for r in results)} blocks, "
              f"{total:.2f}s total ({total / len(results):.2f}s/image)")
        print(f"  -> {summary}")

    return 1 if all(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
