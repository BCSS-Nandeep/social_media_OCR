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
import io
import sys
import traceback
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode Devanagari, Telugu,
# etc.  Force UTF-8 so print() never raises UnicodeEncodeError.
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.detect_script import CONFIDENT_MARGIN, margin
from src.exporter import OCRResult, export, write_batch_summary
from src.ocr_engine import OCREngine
from src.pipeline import find_images, process_image
from src.preprocess import PreprocessConfig
from src.scripts import (AUTO_DETECT_SCRIPTS, KNOWN_WEAK, LANGUAGE_SCRIPTS,
                         UNSUPPORTED_SCRIPTS, UnsupportedLanguage,
                         resolve_models)

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract text from poster images with PaddleOCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Optional so --list-languages can run on its own; checked in main().
    p.add_argument("input", type=Path, nargs="?",
                   help="Image file, or a directory of images (searched recursively).")
    p.add_argument("-o", "--output-dir", type=Path, default=ROOT / "outputs",
                   help="Where to write results (default: ./outputs).")
    p.add_argument("--lang", default="auto",
                   help="Language(s), '+'-separated: te, hi, ta, kn, mr, ur, en, "
                        "sa, ne, gom, ks, sd ... or 'auto' to detect the script "
                        "automatically per image (default). Languages sharing a "
                        "script share one pass, so 'hi+mr+ne' costs the same as "
                        "'hi'. Each additional *script* roughly doubles runtime. "
                        "See --list-languages.")
    p.add_argument("--list-languages", action="store_true",
                   help="Print supported languages and their scripts, then exit.")
    p.add_argument("--auto-merge-latin", action="store_true",
                   help="In auto mode, also run a Latin pass and merge it. "
                        "Doubles runtime; the Indic recognisers already read "
                        "embedded English, so this measured ~0.1 points on "
                        "Telugu posters. Off by default.")
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
    p.add_argument("--det-model", default=None, metavar="NAME",
                   help="Detection model name. Default PP-OCRv6_medium_det "
                        "(+8.5 accuracy points and faster than the stock v5 "
                        "detector).")
    # A dedicated flag rather than `--det-model ""`: PowerShell drops empty
    # string arguments before the process sees them, so the quoted-empty form
    # silently turns into a parse error on the shell most users are on.
    p.add_argument("--legacy-det", action="store_true",
                   help="Use PaddleOCR's own lang-default detector (PP-OCRv5) "
                        "instead of PP-OCRv6.")
    # Default OFF. The textline orientation classifier is trained on document
    # scans; on poster art it misfires badly and rotates upright lines 180
    # degrees, after which recognition returns garbage. Measured on the
    # ground-truth set it cost 17 points of document accuracy overall and 35
    # on the worst image. Detection is unaffected -- block counts are identical
    # either way, so the damage is purely to the crops fed to recognition.
    p.add_argument("--angle-cls", action="store_true",
                   help="Enable the 180-degree text-line orientation classifier. "
                        "Off by default: it misfires on poster text and cost 17 "
                        "points of accuracy on the sample set. Turn on only for "
                        "images with genuinely upside-down text.")
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
    # In auto mode, show which script was detected for this image.
    detected = getattr(result, 'extra', {}).get('detected_script')
    if detected:
        print(f"  detected script: {detected}")
    if quiet:
        return
    for i, (block, line_no) in enumerate(zip(result.blocks, result.line_numbers)):
        print(f"    [{i:>3}] L{line_no:<3} {block.confidence:.4f}  {block.text}")


def print_languages() -> None:
    by_script: dict[str, list[str]] = {}
    for code, script in sorted(LANGUAGE_SCRIPTS.items()):
        by_script.setdefault(script, []).append(code)

    print("Supported — languages sharing a script share one recognition pass:\n")
    for script, codes in sorted(by_script.items()):
        weak = "  (v3 model, weaker than the rest)" if script in KNOWN_WEAK else ""
        print(f"  {script:<12} {', '.join(codes)}{weak}")

    print("\nNot supported — PaddleOCR ships no recognition model for these "
          "scripts,\nat any version:\n")
    for code, name in sorted(UNSUPPORTED_SCRIPTS.items(), key=lambda kv: kv[1]):
        print(f"  {code:<5} {name}")
    print("\nCombine with '+', e.g. --lang hi+en. Each additional *script* "
          "roughly\ndoubles runtime; extra languages on the same script are free.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_languages:
        print_languages()
        return 0
    if args.input is None:
        print("error: an input image or directory is required", file=sys.stderr)
        return 2

    requested = [x.strip() for x in args.lang.split("+") if x.strip()]
    if not requested:
        print("error: --lang must name at least one language", file=sys.stderr)
        return 2

    # 'auto' enables per-image script detection across all supported scripts.
    auto_mode = len(requested) == 1 and requested[0].lower() == "auto"

    if auto_mode:
        langs = list(AUTO_DETECT_SCRIPTS)
    else:
        # Collapse languages to the scripts that actually drive recognition passes.
        try:
            pairs = resolve_models(requested)
        except UnsupportedLanguage as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        langs = [script for script, _ in pairs]
        if len(langs) < len(requested):
            print(f"Note: {len(requested)} languages share {len(langs)} script(s) — "
                  f"running {len(langs)} pass(es).")

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
    if auto_mode:
        print(f"Languages    : auto — detecting the script of each image "
              f"({len(langs)} candidates)")
    else:
        print(f"Languages    : {' + '.join(langs)}")
    print(f"Preprocessing: {'enabled' if pre_config.any_enabled else 'none'}")
    print("Loading PaddleOCR models (first run downloads them)...")

    engine = OCREngine(langs, use_gpu=args.gpu,
                       use_angle_cls=args.angle_cls,
                       det_db_box_thresh=args.det_box_thresh,
                       det_limit_side_len=args.det_limit,
                       det_db_unclip_ratio=args.unclip,
                       det_model="" if args.legacy_det else args.det_model,
                       drop_score=0.0,  # filter once, in process_image, so the
                                        # "dropped" count reflects reality
                       auto_mode=auto_mode,
                       auto_merge_latin=args.auto_merge_latin)
    engine.warmup()

    results: list[OCRResult] = []
    reported_scores = False
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

        # Stash the detected script name so report() can display it.
        if auto_mode and engine.detected_langs:
            result.extra["detected_script"] = engine.detected_langs[0]
            result.extra["script_scores"] = {
                k: round(v, 4) for k, v in engine.auto_scores.items()}

        # Show the probe scores per image, so each choice is auditable rather
        # than asserted. A narrow margin means the detection itself is a coin
        # flip, which the user needs to see.
        if auto_mode and engine.auto_scores:
            ranked = sorted(engine.auto_scores.items(), key=lambda kv: -kv[1])
            gap = margin(engine.auto_scores)
            if not args.quiet:
                print("  script probe: "
                      + ", ".join(f"{s}={v:.3f}" for s, v in ranked[:4]))
            if gap < CONFIDENT_MARGIN and len(ranked) > 1:
                print(f"  WARNING: {ranked[0][0]} beat {ranked[1][0]} by only "
                      f"{gap:.3f} — detection unreliable here, prefer --lang.")

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
