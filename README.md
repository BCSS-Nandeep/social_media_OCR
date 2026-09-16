# Social Media OCR — IndicOCR pipeline

Extracts text from Instagram / social-media poster images (**Telugu + English**,
plus every other constitutionally recognised Indian language), preserves
reading order, and reports a layout-detection confidence per block.

```
image ──► preprocessing (optional) ──► IndicOCR ──► export
          resize / denoise            layout detect    txt
          CLAHE / sharpen / deskew    + reading order   json
                                      + transcription   csv
```

IndicOCR ([bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr),
Bodhan AI + AI4Bharat) is a local, two-stage model: **IndicDocLayout** detects
the page's blocks and their reading order, **IndicBlockOCR** transcribes
them. Script is inferred automatically — there is no per-language flag or
recognition-model pairing to manage.

Outputs per image: reading-ordered Markdown, per-block layout label and
confidence, bounding boxes, and total processing time.

---

## Quick start

```bash
# 1. install (Python 3.12; a CUDA GPU helps but is not required)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. request access to the gated model (one-time, on huggingface.co)
#    -> https://huggingface.co/bodhan-ai/indic-ocr, click "Request access"
.venv/bin/hf auth login          # or: export HF_TOKEN=hf_...

# 3. drop poster images into data/input/, then run
.venv/bin/python run_ocr.py data/input --formats txt json csv
```

Results land in `outputs/`. First run downloads ~1.8GB of weights, then
IndicOCR's own installer picks matching torch/CUDA wheels for your machine.

---

## How to run — step by step

### Step 1 — Request access to the model

`bodhan-ai/indic-ocr` is a **gated** Hugging Face repo. Visit
[the model page](https://huggingface.co/bodhan-ai/indic-ocr), accept the
license, and click "Request access". Approval is usually near-instant for
the stated terms (research/commercial use is broadly allowed — see the
license summary on that page).

### Step 2 — Authenticate locally

```bash
hf auth login
```

or set `HF_TOKEN` in the environment. Either way, `huggingface_hub` picks it
up automatically — nothing else in this repo needs the token directly.

### Step 3 — Create the environment and install dependencies

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

[requirements.txt](requirements.txt) intentionally does **not** pin
`torch`/`transformers`. IndicOCR ships its own `install.sh`, which reads your
GPU driver and selects the matching wheels; this pipeline runs it
automatically the first time it needs the model (see
[src/ocr_engine.py](src/ocr_engine.py)).

### Step 4 — Put your poster images in `data/input/`

Any `.jpg .jpeg .png .bmp .webp .tif`. Subfolders are searched too.

### Step 5 — Run it

```bash
.venv/bin/python run_ocr.py data/input
```

**The first run downloads the model weights** (~1.8GB) and then runs the
vendor installer once. Later runs skip both.

### Step 6 — Read the results

Everything lands in `outputs/`:

```
outputs/poster.txt                 extracted Markdown + per-block detail table
outputs/poster.json                full structured record (text, boxes, timings)
outputs/poster.csv                 one row per layout block
outputs/_batch_summary.json        per-image totals (only when >1 image)
outputs/overlays/poster_boxes.png  annotated image (only with --visualize)
```

The console prints the same summary as it goes: block count, mean
confidence and processing time per image.

### Common variations

```bash
# one specific image
.venv/bin/python run_ocr.py data/input/poster.jpg

# also write CSV, and an annotated image showing every detected box
.venv/bin/python run_ocr.py data/input --formats txt json csv --visualize

# a page with a large or merged-cell table
.venv/bin/python run_ocr.py data/input --table-format markdown

# hide the per-block console dump, keep the summary
.venv/bin/python run_ocr.py data/input --quiet

# every option, explained
.venv/bin/python run_ocr.py --help
```

### If something goes wrong

| Symptom | Cause / fix |
|---|---|
| `Could not download bodhan-ai/indic-ocr` | You haven't requested access, or aren't logged in. See Steps 1–2. |
| `401` from Hugging Face | Access request still pending, or `hf auth login` used the wrong account. |
| First run hangs at "Loading IndicOCR" | Downloading ~1.8GB of weights, then running the vendor installer. Needs internet; give it several minutes on a slow link. |
| `CUDA out of memory` | The model needs a few GB of VRAM. On a small GPU, set `CUDA_VISIBLE_DEVICES=""` to force CPU (slower, but the 0.8B model does run on CPU). |
| Telugu shows as `?????` in the console | Cosmetic — terminal codepage. Files in `outputs/` are correct UTF-8. |
| Telugu is mojibake in Excel | Open the `.csv` via Data → From Text/CSV and pick UTF-8, or use the `.json`. |
| `error: no images found under data/input` | Unsupported extension, or images are elsewhere. |

---

## All options

| Flag | Default | Purpose |
|---|---|---|
| `--formats` | `txt json` | Any of `txt`, `json`, `csv`. |
| `--table-format` | `html` | `html` preserves merged/multi-level table cells; `markdown` is smaller and reads well as plain text for simple tables. |
| `--min-confidence` | `0.0` | Drop layout blocks scoring below this. This is **IndicDocLayout's detection confidence**, not a transcription score — see [Confidence reporting](#confidence-reporting). |
| `--visualize` | off | Write `outputs/overlays/<name>_boxes.png` — boxes coloured green ≥0.90, amber ≥0.70, red below. |
| `--resize` | off | Scale short side to ≥`--min-side` (960), long side ≤`--max-side` (2560). |
| `--denoise` | off | Edge-preserving denoise. |
| `--enhance-contrast` | off | CLAHE on the LAB L channel. |
| `--sharpen` | off | Unsharp mask (mild). |
| `--fix-orientation` | off | Deskew rotations under ~15°. |

### Preprocessing is off by default

IndicOCR's layout detector is trained on 15M+ real documents, including
noisy in-the-wild imagery — the preprocessing stages here predate the
IndicOCR switch and were tuned against a different engine. They're kept as
per-image rescue options (faint text, a rotated photo of a poster) rather
than something to enable by default; re-validate them against IndicOCR
before trusting them to help systematically.

---

## Accuracy

This repo's own ground-truth harness was removed (the six-poster
transcription set it scored against is no longer maintained here — see git
history). The numbers below are **Bodhan AI's own published benchmarks**
for the underlying model, not a local re-measurement:

| Benchmark | IndicOCR | Notes |
|---|---:|---|
| OmniDocBench 1.6 (English subset, overall) | 92.76 | vs. PaddleOCR-VL 96.36, Gemini 3.1 Pro 91.15 |
| olmOCR-Bench (English subset, overall) | 82.2 | vs. Sarvam Vision 84.3, Gemini 3.1 Pro 82.6 |
| IndicOCR-PR, printed word accuracy — **Telugu** | **82.3%** | 100×(1−WER); Sarvam Vision 84.3%, Gemini 3.1 Pro 85.5% |
| IndicOCR-PR, printed word accuracy — English | 97.0% | |
| IndicOCR-HW, handwriting word accuracy — **Telugu** | **53.5%** | handwriting quality is explicitly WIP per the model card |
| IndicOCR-HW, handwriting word accuracy — English | 80.7% | |

Full per-language tables are on the
[model card](https://huggingface.co/bodhan-ai/indic-ocr#indicocr-pr-printed-accuracy-by-language-higher-is-better).

**Re-measuring on this project's own poster set is the natural next step**
once weights are downloaded — there is no harness left in this repo to do
it automatically; see [Known limits](#known-limits).

---

## Output format

**`<name>.json`** — the complete record:

```json
{
  "image": { "name": "poster.jpg", "width": 1080, "height": 1350 },
  "settings": { "preprocessing": ["none"] },
  "summary": {
    "text_blocks_detected": 6,
    "mean_confidence": 0.8341,
    "min_confidence": 0.3896,
    "blocks_dropped_below_threshold": 0,
    "total_processing_time_sec": 4.2,
    "stage_times_sec": { "load": 0.02, "preprocess": 0.0, "ocr": 4.15 }
  },
  "full_text": "...reading-order Markdown for the whole page...",
  "blocks": [
    { "index": 0, "order": 0, "label": "Title", "type": "Title",
      "text": "GRAND OPENING", "confidence": 0.987,
      "bbox_xyxy": [120, 88, 960, 190] }
  ]
}
```

**`<name>.txt`** — header (image size, preprocessing, block count, mean
confidence, processing time), the reading-ordered Markdown, then a
per-block detail table.

**`<name>.csv`** — one row per block with coordinates. Written as
`utf-8-sig` so Excel renders Telugu instead of mojibake.

**`_batch_summary.json`** — per-image block counts, confidences and timings
when processing more than one image.

Bounding boxes are always in **original image coordinates**, even when
`--resize` changed the image IndicOCR saw.

---

## Reading order

IndicDocLayout detects blocks and orders them itself — `order` on each
block is 0-based and gap-free. There is no separate line-grouping step in
this repo any more: a "block" here is already IndicOCR's own paragraph/line
unit (one of 37 layout labels — `Paragraph`, `Title`, `Table`, ...), not a
raw text fragment that needs regrouping.

Pictorial and margin blocks (`Image`, `Header`, `Footer`, `Advertisement`,
...) are detected and keep their place in reading order, but carry
`text: ""` — they were never sent to the recogniser.

Reading order on genuinely complex multi-column layouts is IndicOCR's own
stated limitation (per the model card), not something this pipeline adds or
fixes.

---

## Confidence reporting

`block.confidence` is **IndicDocLayout's detection confidence** — how sure
the layout model is that a box of that type exists there. IndicOCR does not
expose a separate per-block transcription score, unlike the previous
PaddleOCR pipeline (which reported recognition confidence).

Practically: a `Paragraph` block at 0.95 means the layout model is 95% sure
it found a paragraph there — it says nothing about whether the transcribed
text inside is correct. Treat confidence as a **detection** triage signal,
not a proofreading one.

---

## Project layout

| Path | Role |
|---|---|
| [run_ocr.py](run_ocr.py) | CLI, batch loop, console reporting |
| [src/pipeline.py](src/pipeline.py) | One image end-to-end, stage timing |
| [src/preprocess.py](src/preprocess.py) | Optional image stages, non-ASCII-safe loading |
| [src/ocr_engine.py](src/ocr_engine.py) | IndicOCR wrapper (lazy HF download + `TextBlock`) |
| [src/exporter.py](src/exporter.py) | `OCRResult` + TXT/JSON/CSV writers |
| [src/visualize.py](src/visualize.py) | Annotated debug overlay |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Offline tests (IndicOCR stubbed) |
| `data/input/` | Your images (git-ignored) |
| `outputs/` | Generated results (git-ignored) |

---

## Tests

Offline tests covering preprocessing, coordinate mapping, the engine
wrapper's block parsing, and all three exporters. IndicOCR is stubbed, so
they run without the model weights or `huggingface_hub` installed:

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

These do **not** exercise real IndicOCR inference — that needs the gated
weights and a live run against `data/input/`.

---

## Known limits

- **No local accuracy harness.** The ground-truth transcription set this
  repo used to score PaddleOCR was removed before the IndicOCR switch. The
  [Accuracy](#accuracy) numbers above are the vendor's own benchmarks, not
  measured against this project's own posters.
- **Confidence is layout-detection confidence, not a transcription score**
  — see [Confidence reporting](#confidence-reporting).
- **Handwriting is explicitly weaker than print** (53.5% vs 82.3% Telugu
  word accuracy per the vendor's own numbers) and "still a work in
  progress" per the model card.
- **Reading order on complex multi-column layouts** is a stated model
  limitation, inherited as-is.
- **GPU memory.** The OCR stage is a 0.8B-parameter model in bf16
  (~1.6GB weights); on a small GPU (≤4GB) this can be tight alongside
  other processes. CPU fallback works but is slower.
- **Emoji and pictographs** are pictorial content, not text — IndicOCR
  won't transcribe them.
