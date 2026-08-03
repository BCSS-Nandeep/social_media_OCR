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

# both scripts — only for English-dominant images (3.7x slower, no measured gain)
.\.venv\Scripts\python.exe run_ocr.py data\input --lang te+en

# fall back to the older v5 detector
.\.venv\Scripts\python.exe run_ocr.py data\input --legacy-det

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
| `--lang` | `te` | `'+'`-separated recognition passes. `te`, `en`, or both. `te+en` measured no better than `te` alone and takes 3.7× longer. |
| `--det-model` | `PP-OCRv6_medium_det` | Detection model name. |
| `--legacy-det` | off | Use PaddleOCR's lang-default v5 detector instead of v6. |
| `--formats` | `txt json` | Any of `txt`, `json`, `csv`. |
| `--min-confidence` | `0.0` | Drop blocks scoring below this; the count is reported. Default keeps everything, including reads PaddleOCR would normally hide (see [Confidence reporting](#confidence-reporting)). |
| `--det-box-thresh` | `0.5` | Lower ⇒ detects more/fainter text, more false positives. |
| `--det-limit` | `960` | Max side the detector sees (larger images downscaled first). |
| `--unclip` | `1.5` | Detected-box inflation before recognition cropping. |
| `--angle-cls` | off | Enable the 180° orientation classifier. **Off by default — it cost 17 points of accuracy on the sample set.** Turn on only for genuinely rotated text. |
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

### Headline numbers

Current pipeline: **PP-OCRv6 detector + PP-OCRv5 Telugu recogniser**,
`--lang te`, orientation classifier off.

| Metric | Score |
|---|---|
| **Telugu character accuracy** | **92.2%** |
| **Document character accuracy** | **84.9%** |
| Line accuracy (order-independent) | 75.8% |
| **Word accuracy** (whole word exactly right) | **65.5%** |
| Speed | ~5 s / image (CPU) |

### How it got there — three changes, all measured

| Stage | Doc accuracy | Telugu | Word |
|---|---|---|---|
| PaddleOCR 2.9.1, PP-OCRv3-era Telugu model | 51.2% | 45.3% | 16.4% |
| PaddleOCR 3.7 + `te_PP-OCRv5_mobile_rec` | 59.5% | — | — |
| + PP-OCRv6 detector | 68.0% | 69.4% | 51.1% |
| **+ orientation classifier disabled** | **84.9%** | **92.2%** | **65.5%** |

Word accuracy — the number that decides whether output is usable — went from
16.4% to 65.5%, a **4× improvement**.

Note what did *not* work: 16 preprocessing configurations moved accuracy by
less than 3% each. Every real gain came from **model choice and pipeline
configuration**, not from filtering pixels.

### The orientation classifier was destroying text

The single largest win, and the least obvious. PaddleOCR's textline
orientation classifier (`PP-LCNet_x1_0_textline_ori`) decides whether each
detected line is upside-down and rotates it 180° if it thinks so. It is
trained on document scans. On poster art — decorative fonts, coloured
backgrounds, text over photographs — **it misfires and flips upright lines**,
after which recognition returns garbage.

The tell was in the digits. A quiz line reading `1605 … 1611` came out as:

```
QeSe g  O gా 119L ' g gూ G091 L
```

`1611` → `119L`, `1605` → `G091` — those are rotated digits, not
misrecognised ones.

Disabling it, with everything else unchanged:

| Poster | With classifier | Without | Δ |
|---|---:|---:|---:|
| Quiz meme | 61.7% | **96.9%** | **+35.2** |
| Genius notes (2/7) | 34.7% | **79.8%** | **+45.1** |
| Hand-lettered songs poster | 72.0% | **81.7%** | +9.7 |
| Instagram G-7 quiz | 94.1% | **96.7%** | +2.6 |
| TSLPRB police notice | 72.8% | **74.6%** | +1.8 |
| Genius cover | 75.0% | 75.0% | 0.0 |
| **Overall** | **68.0%** | **84.9%** | **+16.9** |

**Block counts are identical either way** — detection was never the problem on
these two posters. The classifier was handing the recogniser upside-down crops.

Six of six images improved or held, so it is off by default. `--angle-cls`
turns it back on for images with genuinely rotated text.

### The PP-OCRv6 detector swap

PP-OCRv6 (PaddleOCR 3.7, June 2026) has **no Telugu recogniser** — it covers
Chinese, English, Japanese and Latin-script languages only:

```python
_PPOCRV6_LANGS = frozenset({"ch", "chinese_cht", "en", "japan"}) | LATIN_LANGS
```

But **detection is script-agnostic**, so the v6 detector can front the v5
Telugu recogniser. Detector comparison, everything else held constant:

| Detector | Doc accuracy | Speed |
|---|---|---|
| `PP-OCRv5_server_det` (stock) | 59.5% | 5.9 s/img |
| **`PP-OCRv6_medium_det`** | **68.0%** | **3.8 s/img** |
| `PP-OCRv6_small_det` | 58.3% | 1.8 s/img |

Medium wins on accuracy *and* speed. The mechanism is visible in the box
counts — it stops shattering text into fragments, so the recogniser sees whole
lines instead of shards:

| Poster | Boxes before | Boxes after | Doc accuracy |
|---|---:|---:|---|
| TSLPRB police notice | 57 | **28** | 55.3% → **72.8%** |
| Genius notes (2/7) | 58 | **43** | 25.6% → **34.7%** |
| Hand-lettered songs poster | 22 | **15** | 54.3% → **72.0%** |
| Instagram G-7 quiz | 33 | **23** | 82.1% → **94.1%** |
| Genius cover | 27 | **22** | 67.8% → **75.0%** |
| Quiz meme | 17 | 16 | 70.1% → **61.7%** |

Five of six improved, by 7–18 points. The quiz meme lost 8: the v6 detector
actually reads *more* of its lines correctly (`రాష్ట్రపతి`, `పార్లమెంట్`,
`అడిగినప్పుడే` are all fixed) but drops one answer line entirely — a localised
detection dropout, not a systematic regression.

> **Naming a detector makes PaddleOCR ignore `lang`.** The recogniser must be
> named in the same call or it silently falls back to the Chinese default and
> returns confident nonsense for Telugu. `RECOGNITION_MODELS` in
> [src/ocr_engine.py](src/ocr_engine.py) keeps the pair together, and tests
> guard it — the failure mode is silent, not loud.

### `--lang te` is the default now

| Mode | Doc accuracy | Time / image |
|---|---|---|
| **`te` (default)** | **68.0%** | **3.8 s** |
| `te+en` | 68.1% | 13.9 s |

With the v6 detector the second English pass buys **0.1 points for 3.7× the
runtime** — the Telugu model already reads embedded English. Use `te+en` only
for English-dominant images.

### Per-image results

| Poster | Doc | Telugu | Word | Confidence |
|---|---:|---:|---:|---:|
| Quiz meme | **96.9%** | 99.1% | 86.8% | 0.903 |
| Instagram G-7 quiz | **96.7%** | 98.7% | 81.0% | 0.931 |
| Hand-lettered songs poster | 81.7% | 82.8% | 59.4% | 0.937 |
| Genius notes (2/7) | 79.8% | 86.7% | 50.0% | 0.880 |
| Genius cover | 75.0% | 81.6% | 47.8% | 0.890 |
| TSLPRB police notice | 74.6% | 96.9% | 62.1% | 0.897 |

Telugu character accuracy is now above 96% on three of six posters. The
remaining document-level gap on the police notice is layout, not recognition —
its Telugu is 96.9% but two-column label/value rows interleave in reading
order.

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
   correctly even on busy poster art, and the v6 detector made this markedly
   better.
2. **English and numeric content is production-quality.**
3. **Telugu recognition is now strong — 92.2% at character level**, above 96%
   on half the sample. The remaining document-level gap is mostly *layout*
   (reading order on multi-column cards), not character recognition.
4. **Confidence tracks quality at the image level** and is a usable triage
   signal — but *not* per block: `Contact:` was read as `Contact.` at 0.997.
   Low confidence reliably flags trouble; high confidence guarantees nothing.
5. Preprocessing was exhausted as a lever early and never paid. **Every gain
   came from model choice and from switching off a stage that was actively
   corrupting input.** Worth remembering: the largest single win came from
   *disabling* a feature, not adding one.
6. The next bottleneck is **reading order on multi-column layouts**, not
   recognition — see Known limits.

> **Caveat on method.** Ground truth is a human transcription of the images.
> For hand-lettered posters that transcription is itself uncertain, so those
> per-image figures carry real error bars.
>
> **The reference files are not in this repo**, and neither are the poster
> images they describe — both were removed as the working image set changed.
> The figures above are therefore reported results, not something a clone can
> re-derive as-is. To reproduce or extend them, put images in `data/input/` and
> write a matching `data/ground_truth/<image stem>.txt` for each, then run
> `evaluate.py`. The scorer itself is in the repo and unchanged.

---

## Version pins, and why they are exact

[requirements.txt](requirements.txt) pins narrowly because three separate
version combinations fail on Windows CPU:

| Package | Pin | Reason |
|---|---|---|
| `paddlepaddle` | `==3.2.0` | 3.0.0 cannot load PP-OCRv5 models (`Type of attribute: strides is not right`). 3.3.1 hits a PIR executor bug (`ConvertPirAttribute2RuntimeAttribute not support`). 3.2.0 is the working middle. |
| `paddleocr` | `>=3.7,<4` | Ships both `te_PP-OCRv5_mobile_rec` (+8 points over the 2.x Telugu model) and the PP-OCRv6 detectors (+8.5 more). |
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
| `data/input/` | Your images (git-ignored) |
| `data/ground_truth/` | Optional `<image stem>.txt` references for `evaluate.py`; create your own |

The engine wrapper accepts **both** PaddleOCR APIs — 2.x `.ocr(img, cls=True)`
returning nested lists, and 3.x `.predict(img)` returning result dicts. It
dispatches on the installed **version**, not on which keyword arguments are
accepted: PaddleOCR 2.x builds its config through `argparse` and silently
swallows unknown keywords, so a try/except probe "succeeds" with 3.x names
while quietly discarding the settings they carry.

---

## Tests

28 tests covering preprocessing, coordinate mapping, reading order, the
multi-pass merge (including the word-inside-line case), detector/recogniser
pairing, both result-parser formats, and all three exporters. PaddleOCR is
stubbed, so they run without it:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

---

## Known limits

- **Telugu accuracy (92% character, 66% word).** Conjunct consonants (ఒత్తులు) still fail sometimes
  and hand-lettered display type is unreliable. Treat blocks below ~0.7 as
  needing review — that is what `--min-confidence` and the overlay colours are
  for. Headline-quality Telugu needs a fine-tuned recognition model.
- **Confidence is not accuracy.** High per-block confidence does not guarantee a
  correct read.
- **Stylised type.** Heavy outlines, gradients, arced/curved text and text baked
  into photographs degrade detection. `--det-box-thresh 0.3` recovers some.
- **Multi-column posters interleave — this is now the main bottleneck.** The
  TSLPRB notice recognises Telugu at 96.9% but scores only 74.6% at document
  level, because its label/value rows are read straight across instead of as
  columns. Fixing reading order is worth more than further recognition work.
- **Emoji and pictographs** are not in any recognition dictionary and are dropped.
