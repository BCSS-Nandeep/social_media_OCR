#!/usr/bin/env python
"""Score OCR output against hand-written ground truth.

    python evaluate.py                      # all images with a ground-truth file
    python evaluate.py --json report.json   # also dump machine-readable results

Ground truth lives in data/ground_truth/<image stem>.txt; OCR results are read
from outputs/<image stem>.json (run run_ocr.py first).

Three numbers are reported, because one number would hide the interesting part:

* **Document accuracy** (1 - CER over the whole page, in reading order) is the
  strictest view. It charges for recognition errors *and* for ordering
  differences, so a page read perfectly but assembled in a different order
  still scores badly.
* **Line accuracy** matches each ground-truth line to its best OCR line before
  scoring. This isolates *recognition* quality from layout ordering.
* **Per-script accuracy** splits the character stream into Telugu and
  ASCII/digits and scores each separately -- the split that matters here,
  since the two recognition models are not equally good.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TELUGU = range(0x0C00, 0x0C80)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (ca != cb)))  # substitution
        previous = current
    return previous[-1]


def normalise(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    NFC matters for Telugu: the same syllable can be encoded with differently
    ordered combining marks, and comparing raw code points would count an
    identical-looking string as wrong.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("‌", "").replace("‍", "")
    return " ".join(text.split())


def accuracy(reference: str, hypothesis: str) -> float:
    reference, hypothesis = normalise(reference), normalise(hypothesis)
    if not reference:
        return 1.0 if not hypothesis else 0.0
    return max(0.0, 1 - levenshtein(reference, hypothesis) / len(reference))


def word_accuracy(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 1.0 if not hyp else 0.0
    # Levenshtein over word tokens, reusing the character routine via a mapping
    # of each distinct word to a single private-use code point.
    vocab: dict[str, str] = {}

    def encode(tokens: list[str]) -> str:
        return "".join(vocab.setdefault(t, chr(0xE000 + len(vocab))) for t in tokens)

    return max(0.0, 1 - levenshtein(encode(ref), encode(hyp)) / len(ref))


def keep_script(text: str, script: str) -> str:
    if script == "telugu":
        chars = [c for c in text if ord(c) in TELUGU]
    else:
        chars = [c for c in text if c.isascii() and c.isalnum()]
    return "".join(chars)


def line_level(reference_lines: list[str], hypothesis_lines: list[str]) -> tuple[float, int]:
    """Greedily pair each reference line with its closest unused OCR line.

    Longest reference lines are matched first: they carry the most signal, so
    they should get first pick of the candidates rather than losing a good
    match to an earlier short line.
    """
    unused = list(hypothesis_lines)
    scored: list[tuple[float, int]] = []
    matched = 0

    for ref in sorted(reference_lines, key=len, reverse=True):
        if not ref.strip():
            continue
        best, best_index = 0.0, None
        for index, hyp in enumerate(unused):
            score = accuracy(ref, hyp)
            if score > best:
                best, best_index = score, index
        if best_index is not None and best > 0:
            unused.pop(best_index)
            matched += 1
        scored.append((best, len(normalise(ref))))

    if not scored:
        return 0.0, 0
    total_weight = sum(w for _, w in scored)
    weighted = sum(s * w for s, w in scored) / total_weight if total_weight else 0.0
    return weighted, matched


def evaluate(stem: str, gt_path: Path, ocr_path: Path) -> dict:
    reference = gt_path.read_text(encoding="utf-8")
    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    hypothesis = payload.get("full_text", "")

    ref_lines = [l for l in reference.splitlines() if l.strip()]
    hyp_lines = [l for l in hypothesis.splitlines() if l.strip()]
    line_acc, matched = line_level(ref_lines, hyp_lines)

    return {
        "image": stem,
        "reference_chars": len(normalise(reference)),
        "blocks_detected": payload["summary"]["text_blocks_detected"],
        "mean_confidence": payload["summary"]["mean_confidence"],
        "document_accuracy": accuracy(reference, hypothesis),
        "line_accuracy": line_acc,
        "word_accuracy": word_accuracy(reference, hypothesis),
        "telugu_accuracy": accuracy(keep_script(reference, "telugu"),
                                    keep_script(hypothesis, "telugu")),
        "latin_accuracy": accuracy(keep_script(reference, "latin"),
                                   keep_script(hypothesis, "latin")),
        "reference_lines": len(ref_lines),
        "lines_matched": matched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "data" / "ground_truth")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--json", type=Path, help="Write the full report here.")
    args = parser.parse_args()

    results = []
    for gt_path in sorted(args.ground_truth.glob("*.txt")):
        ocr_path = args.outputs / f"{gt_path.stem}.json"
        if not ocr_path.exists():
            print(f"  (skipped {gt_path.stem}: no OCR result -- run run_ocr.py first)")
            continue
        results.append(evaluate(gt_path.stem, gt_path, ocr_path))

    if not results:
        print("No image had both a ground-truth file and an OCR result.")
        return 1

    name_width = max(len(r["image"]) for r in results)
    header = (f"{'image':<{name_width}}  {'chars':>6}  {'doc':>7}  {'line':>7}  "
              f"{'word':>7}  {'telugu':>7}  {'latin':>7}  {'conf':>6}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['image']:<{name_width}}  {r['reference_chars']:>6}  "
              f"{r['document_accuracy']:>6.1%}  {r['line_accuracy']:>6.1%}  "
              f"{r['word_accuracy']:>6.1%}  {r['telugu_accuracy']:>6.1%}  "
              f"{r['latin_accuracy']:>6.1%}  {r['mean_confidence']:>6.3f}")

    # Character-weighted totals: a 600-character page should count for more
    # than a 90-character one.
    weight = sum(r["reference_chars"] for r in results)
    def overall(key: str) -> float:
        return sum(r[key] * r["reference_chars"] for r in results) / weight

    print("-" * len(header))
    print(f"{'OVERALL (char-weighted)':<{name_width}}  {weight:>6}  "
          f"{overall('document_accuracy'):>6.1%}  {overall('line_accuracy'):>6.1%}  "
          f"{overall('word_accuracy'):>6.1%}  {overall('telugu_accuracy'):>6.1%}  "
          f"{overall('latin_accuracy'):>6.1%}")

    if args.json:
        args.json.write_text(json.dumps(
            {"images": results,
             "overall": {k: overall(k) for k in
                         ("document_accuracy", "line_accuracy", "word_accuracy",
                          "telugu_accuracy", "latin_accuracy")},
             "reference_chars_total": weight},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
