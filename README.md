# Social Media OCR — PaddleOCR evaluation pipeline

Extracts text from Instagram / social-media poster images (**Telugu + English**),
preserves reading order, and reports a confidence score per text region.

```
image ──► preprocessing (optional) ──► PaddleOCR ──► reading order ──► export
          resize / denoise            detection      line grouping     txt
          CLAHE / sharpen / deskew    angle cls      L→R, T→B          json
                                      recognition                      csv
```

Outputs per image: extracted text, per-block confidence, bounding boxes, total
processing time, and number of detected text blocks.

---

## Quick start

```bash
# 1. install (Python 3.12 required — see below)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. drop poster images into data/input/, then run
.\.venv\Scripts\python.exe run_ocr.py data\input --formats txt json csv

# 3. score against ground truth (optional)
.\.venv\Scripts\python.exe evaluate.py
```

Results land in `outputs/`. First run downloads ~100 MB of models.

---

## How to run — step by step

> **Python version matters.** PaddlePaddle publishes wheels for **CPython
> 3.8–3.12 only**. If your default `python` is 3.13+ it *cannot* install
> PaddlePaddle at all — hence the dedicated 3.12 environment below.

### Step 1 — Open a shell in the project folder

```powershell
cd path\to\social_media_OCR
```

### Step 2 — Make sure Python 3.12 exists

```powershell
py -0p
```

You should see a `-V:3.12` line. If not, install it once:

```powershell
winget install --id Python.Python.3.12 -e
```

### Step 3 — Create the environment and install dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Takes a few minutes — `paddlepaddle` is a ~100 MB download. The version pins in
[requirements.txt](requirements.txt) are **not arbitrary**; see
[Version pins](#version-pins-and-why-they-are-exact) before changing them.

### Step 4 — Put your poster images in `data/input/`

Any `.jpg .jpeg .png .bmp .webp .tif`. Subfolders are searched too.

### Step 5 — Run it

```powershell
.\.venv\Scripts\python.exe run_ocr.py data\input
```

**The first run also downloads the OCR models** (detection + angle
classification + one recogniser per language) into `%USERPROFILE%\.paddlex`.
Later runs skip this.

### Step 6 — Read the results

Everything lands in `outputs/`:

```
outputs/poster.txt                 extracted text + per-block confidence table
outputs/poster.json                full structured record (text, scores, boxes, timings)
outputs/poster.csv                 one row per text block
outputs/_batch_summary.json        per-image totals (only when >1 image)
outputs/overlays/poster_boxes.png  annotated image (only with --visualize)
```

The console prints the same summary as it goes: block count, mean confidence and
processing time per image.

### Common variations

```powershell
# one specific image
.\.venv\Scripts\python.exe run_ocr.py data\input\poster.jpg

# Telugu only — 3x faster than te+en, and only ~1 accuracy point worse
.\.venv\Scripts\python.exe run_ocr.py data\input --lang te

# also write CSV, and an annotated image showing every detected box
.\.venv\Scripts\python.exe run_ocr.py data\input --formats txt json csv --visualize

# faint or low-contrast text
.\.venv\Scripts\python.exe run_ocr.py data\input --enhance-contrast --det-box-thresh 0.3

# hide the per-block console dump, keep the summary
.\.venv\Scripts\python.exe run_ocr.py data\input --quiet

# every option, explained
.\.venv\Scripts\python.exe run_ocr.py --help
```

### If something goes wrong

| Symptom | Cause / fix |
|---|---|
| `No matching distribution found for paddlepaddle` | Your `python` is 3.13+. Use `.\.venv\Scripts\python.exe` (3.12). |
| `paddleocr is not installed` | Step 3 was skipped or failed — rerun `pip install -r requirements.txt`. |
| `Type of attribute: strides is not right` | `paddlepaddle` is too old for PP-OCRv5 models. Needs ≥ 3.2. |
| `ConvertPirAttribute2RuntimeAttribute not support` | `paddlepaddle` 3.3.x PIR bug on Windows CPU. Pin to 3.2.0. |
| `DLL load failed ... Application Control policy` | A locked-down Windows host blocking `pandas` 3.x / `matplotlib` binaries. `pip install "pandas<3"` and uninstall `matplotlib`. |
| First run hangs at "Loading PaddleOCR models" | It is downloading models. Needs internet; give it a few minutes. |
| Telugu shows as `?????` in the console | Cosmetic — terminal codepage. Files in `outputs/` are correct UTF-8. Fix with `chcp 65001`. |
| Telugu is mojibake in Excel | Open the `.csv` via Data → From Text/CSV and pick UTF-8, or use the `.json`. |
| `error: no images found under data\input` | Unsupported extension, or images are elsewhere. |

---

## All options

| Flag | Default | Purpose |
|---|---|---|
| `--lang` | `te+en` | `'+'`-separated recognition passes. `te`, `en`, or both. |
| `--formats` | `txt json` | Any of `txt`, `json`, `csv`. |
| `--min-confidence` | `0.0` | Drop blocks scoring below this; the count is reported. Default keeps everything, including reads PaddleOCR would normally hide (see [Confidence reporting](#confidence-reporting)). |
| `--det-box-thresh` | `0.5` | Lower ⇒ detects more/fainter text, more false positives. |
| `--det-limit` | `960` | Max side the detector sees (larger images downscaled first). |
| `--unclip` | `1.5` | Detected-box inflation before recognition cropping. |
| `--no-angle-cls` | off | Skip the 180° text-line orientation classifier. |
| `--gpu` | off | Requires `paddlepaddle-gpu` instead of `paddlepaddle`. |
| `--visualize` | off | Write `outputs/overlays/<name>_boxes.png` — boxes coloured green ≥0.90, amber ≥0.70, red below. |
| `--resize` | off | Scale short side to ≥`--min-side` (960), long side ≤`--max-side` (2560). |
| `--denoise` | off | Edge-preserving denoise. |
| `--enhance-contrast` | off | CLAHE on the LAB L channel. |
| `--sharpen` | off | Unsharp mask (mild). |
| `--fix-orientation` | off | Deskew rotations under ~15°. |

### Preprocessing is off by default — and that is a measured decision

A grid search against the six-poster ground-truth set covered **16
configurations**: detector resolution 960/1600/2400, box unclip 1.5/2.0/2.5,
2× upscaling, CLAHE, unsharp mask, grayscale, `det_db_score_mode="slow"`,
`use_dilation`, and combinations.

**Every one landed within ±3% of the plain baseline, and none won consistently
across images.** CLAHE and grayscale were mild net losses.

The flags remain for per-image rescue attempts, but do not expect preprocessing
to move batch accuracy. The ceiling is set by the **recognition model**, which
is why the upgrade below mattered and the filters did not.

---

## Accuracy

Measured with [evaluate.py](evaluate.py) against hand-written ground truth in
[data/ground_truth/](data/ground_truth/) — six real Instagram posters, 2,826
reference characters, CPU, no preprocessing.

### Headline numbers (PaddleOCR 3.7 + PP-OCRv5 Telugu)

| Metric | Score |
|---|---|
| **Document character accuracy** | **60.9%** |
| Line accuracy (order-independent) | 40.7% |

### The model upgrade that produced it

The pipeline originally used PaddleOCR 2.9.1. Moving to **PaddleOCR 3.7 with
the `te_PP-OCRv5_mobile_rec` model** was the single change that actually moved
accuracy — nearly **+10 points** where 16 preprocessing configurations moved
nothing:

| Poster | 2.9.1 (old) | 3.7 PP-OCRv5 (now) | Δ |
|---|---|---|---|
| Instagram quiz (1) | 56.3% | **80.7%** | +24.4 |
| Genius cover | 48.2% | **73.0%** | +24.8 |
| Quiz meme | 60.0% | **70.1%** | +10.1 |
| Hand-lettered songs poster | 28.0% | **54.3%** | +26.3 |
| TSLPRB police notice | 48.8% | **56.5%** | +7.7 |
| Genius notes (2/7) | 50.0% | **28.6%** | **−21.4** |
| **Overall (char-weighted)** | **51.2%** | **60.9%** | **+9.7** |

**One poster regressed and it is worth knowing why.** On the "Genius notes"
page the v5 model hallucinates Latin letters across the decorative headline
(`C A SOINE NROUN PUBLICATIONS UXIUJHO`) where the old model produced
wrong-but-Telugu output. Five of six posters improved, most by 10–26 points;
one got materially worse.

### `--lang te` is nearly as good and 3× faster

| Mode | Document accuracy | Time / image |
|---|---|---|
| `te+en` (default) | 60.9% | ~14.6 s |
| `te` only | 59.5% | ~5.2 s |

For 1.4 accuracy points, the second recognition pass costs roughly 3× the
runtime. **Use `--lang te` for batch work** unless a poster is English-heavy.

### What still fails

Conjunct consonants (ఒత్తులు) remain the dominant error class — they collapse
to a single glyph:

| Ground truth | Recognised |
|---|---|
| ప్రశ్నలు | పశృలు |
| అడుగుతున్నారు | అడుగుతునృవ |
| కష్టమైన | కషెఘన |

Digits, years and embedded English (`1605`, `1611`, `Easy`, `Kick`,
`www.tgprb.in`) come through cleanly.

### Conclusions for the stated objective

1. **Detection and reading order are solid** — regions are found and ordered
   correctly even on busy poster art.
2. **English and numeric content is production-quality.**
3. **Telugu is usable for plain body text, unreliable for stylised display
   type.** At ~61% character accuracy, output needs human review before use.
4. **Confidence tracks quality at the image level** and is a usable triage
   signal — but *not* per block: `Contact:` was read as `Contact.` at 0.997.
   Low confidence reliably flags trouble; high confidence guarantees nothing.
5. Further gains require a **fine-tuned Telugu recognition model**, not
   configuration changes.

> **Caveat on method.** Ground truth is a human transcription of the images.
> For hand-lettered posters that transcription is itself uncertain, so those
> per-image figures carry real error bars.

---

## Version pins, and why they are exact

[requirements.txt](requirements.txt) pins narrowly because three separate
version combinations fail on Windows CPU:

| Package | Pin | Reason |
|---|---|---|
| `paddlepaddle` | `==3.2.0` | 3.0.0 cannot load PP-OCRv5 models (`Type of attribute: strides is not right`). 3.3.1 hits a PIR executor bug (`ConvertPirAttribute2RuntimeAttribute not support`). 3.2.0 is the working middle. |
| `paddleocr` | `>=3.7,<4` | Ships `te_PP-OCRv5_mobile_rec`, worth ~+10 accuracy points over the 2.x Telugu model. |
| `pandas` | `<3` | `paddlex` imports pandas; pandas 3.x ships DLLs blocked by some corporate Application Control policies. |

`matplotlib` is deliberately **not** installed — `paddlex` imports it only for
dataset-checking code paths this pipeline never uses, and its `_image` DLL is
blocked on locked-down Windows hosts.

---

## Why `te+en` runs two passes

No single PaddleOCR recognition model handles Telugu and Latin equally well.
The `te` model's dictionary includes ASCII, so it reads English — just less
accurately than the `en` model. With `--lang te+en` both run and the results are
reconciled by `merge_by_overlap` in [src/ocr_engine.py](src/ocr_engine.py).

The merge is **not** a simple IoU dedup, because the two passes use different
detectors that segment differently: `en` emits whole lines, `te` emits
individual words. On a test poster the `en` pass produced one box for
`GRAND OPENING` while the `te` pass produced `GRAND` and `OPENING` separately —
and `GRAND` inside `GRAND OPENING` scores IoU ≈ 0.45, under any sane threshold,
so an IoU rule keeps all three and the text appears twice.

So instead:

1. Overlap is measured as **intersection over the smaller box**, which reports
   ~1.0 for containment. Blocks are clustered by that at ≥ 0.6.
2. Within each cluster, **one pass's segmentation is kept whole** — the one with
   the higher character-weighted mean confidence. Picking per-cluster rather
   than per-block is what prevents emitting a line and its own words side by side.
3. Character weighting stops a short, high-scoring fragment from dragging its
   whole segmentation past a correct full-line read.

A region only one pass detected is kept as-is.

---

## Output format

**`<name>.json`** — the complete record:

```json
{
  "image": { "name": "poster.jpg", "width": 1080, "height": 1350 },
  "settings": { "languages": ["te", "en"], "preprocessing": ["none"] },
  "summary": {
    "text_blocks_detected": 17,
    "lines_detected": 12,
    "mean_confidence": 0.8341,
    "min_confidence": 0.3896,
    "blocks_dropped_below_threshold": 0,
    "total_processing_time_sec": 5.198,
    "stage_times_sec": { "load": 0.02, "preprocess": 0.0, "ocr": 5.17,
                         "ocr:te": 2.6, "ocr:en": 2.57, "ordering": 0.001 }
  },
  "full_text": "…reading-order text, one line per visual line…",
  "blocks": [
    { "index": 0, "line": 1, "text": "GRAND OPENING", "confidence": 0.9871,
      "lang_model": "en", "bbox": [[120,88],[960,88],[960,190],[120,190]],
      "bbox_xyxy": [120, 88, 960, 190] }
  ]
}
```

**`<name>.txt`** — header (image, languages, block count, mean confidence,
processing time), the reading-order text, then a per-block confidence table.

**`<name>.csv`** — one row per block with coordinates. Written as `utf-8-sig`
so Excel renders Telugu instead of mojibake.

**`_batch_summary.json`** — per-image block counts, confidences and timings when
processing more than one image.

Bounding boxes are always in **original image coordinates**, even when
`--resize` changed the image the detector saw.

---

## Reading order

PaddleOCR emits boxes in detector order, which is only roughly top-to-bottom.
[src/reading_order.py](src/reading_order.py) regroups them: two blocks share a
visual line when their vertical spans overlap by ≥50% **of the shorter block's
height**. The relative threshold matters on posters, where a 20 px caption can
sit inside a 200 px headline's vertical span — a fixed pixel tolerance would
swallow it into the headline.

Within a line, blocks sort left→right; lines sort top→bottom. This is a
single-flow model: genuinely multi-column layouts will interleave.

---

## Confidence reporting

PaddleOCR applies its own recognition filter (`drop_score`, default **0.5**)
*inside* the OCR call, discarding weak reads before the caller sees them. That
would make both the block count and the confidence statistics silently
incomplete — the pipeline would report "9 blocks, mean 0.98" while quietly
hiding the one region it got wrong.

This pipeline sets `drop_score=0.0` and does all filtering itself, so:

- every detected region surfaces with its real score, however bad;
- `--min-confidence` is the single filter, and
  `blocks_dropped_below_threshold` reports what it actually removed.

Observed on a test poster: with PaddleOCR's default filter the Telugu word
**హైదరాబాద్** vanished entirely. It was detected fine — recognition returned
`9&997` at 0.3896 and the internal filter dropped it. Now it appears in the
output at 0.39, which is the honest answer.

---

## Project layout

| Path | Role |
|---|---|
| [run_ocr.py](run_ocr.py) | CLI, batch loop, console reporting |
| [evaluate.py](evaluate.py) | CER/WER scoring against ground truth |
| [src/pipeline.py](src/pipeline.py) | One image end-to-end, stage timing |
| [src/preprocess.py](src/preprocess.py) | Optional image stages, non-ASCII-safe loading |
| [src/ocr_engine.py](src/ocr_engine.py) | PaddleOCR wrapper, `TextBlock`, multi-pass merge |
| [src/reading_order.py](src/reading_order.py) | Line grouping and text rendering |
| [src/exporter.py](src/exporter.py) | `OCRResult` + TXT/JSON/CSV writers |
| [src/visualize.py](src/visualize.py) | Annotated debug overlay |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Offline tests (PaddleOCR stubbed) |
| [data/ground_truth/](data/ground_truth/) | Hand-written reference text for scoring |

The engine wrapper accepts **both** PaddleOCR APIs — 2.x `.ocr(img, cls=True)`
returning nested lists, and 3.x `.predict(img)` returning result dicts. It
dispatches on the installed **version**, not on which keyword arguments are
accepted: PaddleOCR 2.x builds its config through `argparse` and silently
swallows unknown keywords, so a try/except probe "succeeds" with 3.x names
while quietly discarding the settings they carry.

---

## Tests

23 tests covering preprocessing, coordinate mapping, reading order, the
multi-pass merge (including the word-inside-line case), both result-parser
formats, and all three exporters. PaddleOCR is stubbed, so they run without it:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

---

## Known limits

- **Telugu accuracy (~61%).** Conjunct consonants (ఒత్తులు) fail systematically
  and hand-lettered display type is unreliable. Treat blocks below ~0.7 as
  needing review — that is what `--min-confidence` and the overlay colours are
  for. Headline-quality Telugu needs a fine-tuned recognition model.
- **Confidence is not accuracy.** High per-block confidence does not guarantee a
  correct read.
- **Stylised type.** Heavy outlines, gradients, arced/curved text and text baked
  into photographs degrade detection. `--det-box-thresh 0.3` recovers some.
- **Multi-column posters** interleave (see Reading order).
- **Emoji and pictographs** are not in any recognition dictionary and are dropped.
