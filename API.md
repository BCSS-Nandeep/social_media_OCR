# OCR REST API

FastAPI wrapper around the existing PaddleOCR pipeline. The OCR itself is
untouched — this layer handles transport, concurrency and lifecycle only, so
the API and `run_ocr.py` return identical text for the same image.

## Run

```powershell
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Swagger UI: <http://127.0.0.1:8000/docs> · ReDoc: `/redoc`

Models load **once during startup**, not per request. The first boot takes
~20 s while the detector, six probe recognisers and the recognition pipeline
load; `/health` returns 200 only once they are ready.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ocr` | One image |
| POST | `/ocr/batch` | Many uploads, processed in parallel |
| POST | `/ocr/folder` | Every image in a server-side directory |
| POST | `/ocr/zip` | ZIP in, ZIP of results out |
| GET | `/languages` | Supported and unsupported languages |
| GET | `/health` | Liveness and pool state |

All OCR endpoints accept `lang` (default `auto`), `min_confidence` (default
`0.0`) and `include_blocks` (default `true`).

---

### POST /ocr

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -F "file=@poster.jpg" \
  -F "lang=auto"
```

```json
{
  "success": true,
  "filename": "poster.jpg",
  "language": "devanagari",
  "languages": ["devanagari"],
  "detected": true,
  "script_scores": {"devanagari": 0.968, "kannada": 0.667, "latin": 0.661},
  "detection_margin": 0.301,
  "detection_confident": true,
  "warnings": [],
  "processing_time": 28.88,
  "mean_confidence": 0.9286,
  "text_blocks": 54,
  "lines": 25,
  "text": "जिल्हा न्यायालय ...",
  "blocks": [
    {
      "text": "जिल्हा न्यायालय",
      "confidence": 0.9712,
      "bbox": [[120, 88], [960, 88], [960, 190], [120, 190]],
      "bbox_xyxy": [120, 88, 960, 190],
      "line": 1
    }
  ]
}
```

**Read `detection_confident` before trusting `text`.** When the top two scripts
score within 0.05 the detection is close to a coin flip and the text may have
been read with the wrong script. A real example — a Hindi + Kannada + English
teaching poster:

```json
{
  "language": "latin",
  "detection_margin": 0.0013,
  "detection_confident": false,
  "warnings": ["Script detection was not decisive: latin (0.992) barely beat
                arabic (0.991), margin 0.001. ..."]
}
```

That image genuinely contains three scripts, which one recognition pass cannot
represent. Re-run it with an explicit `lang`.

---

### POST /ocr/batch

The field name is **`files`** (repeated), not `file`.

```bash
curl -X POST http://127.0.0.1:8000/ocr/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@bad.jpg" \
  -F "include_blocks=false"
```

```json
{
  "success": false,
  "total": 3,
  "processed": 2,
  "failed": 1,
  "total_processing_time": 28.1,
  "workers": 2,
  "results": [
    {"filename": "img1.jpg", "language": "latin",   "mean_confidence": 0.878, "...": "..."},
    {"filename": "img2.jpg", "language": "kannada", "mean_confidence": 0.867, "...": "..."}
  ],
  "failures": [
    {"filename": "bad.jpg", "success": false, "error": "ValueError: Not a decodable image: ..."}
  ]
}
```

One bad image never stops the batch. `success` is `false` when anything failed;
check `results` and `failures` rather than `success` alone.

---

### POST /ocr/folder

Reads a directory **on the server**, not on the client.

```bash
curl -X POST http://127.0.0.1:8000/ocr/folder \
  -H "Content-Type: application/json" \
  -d '{"path": "D:/images", "lang": "auto", "recursive": true}'
```

Same response shape as `/ocr/batch`.

> **This endpoint reads arbitrary server paths.** With `OCR_ALLOWED_ROOTS`
> unset it can read anywhere the service account can, which is a directory
> traversal primitive. Set it in any deployment that is not a single-user
> machine:
>
> ```powershell
> $env:OCR_ALLOWED_ROOTS = "D:\images;E:\incoming"
> ```

---

### POST /ocr/zip

Returns a **ZIP download**, not JSON.

```bash
curl -X POST http://127.0.0.1:8000/ocr/zip \
  -F "file=@images.zip" \
  -F "lang=auto" \
  -o results.zip
```

```
results.json      full BatchResponse
text/img1.txt     extracted text, one file per image
text/img3.txt
```

Extraction rejects zip-slip members (`../../evil.txt`), symlinks, archives over
`OCR_MAX_ZIP_ENTRIES` members or `OCR_MAX_ZIP_UNCOMPRESSED` expanded bytes.
Non-image members are skipped.

---

### GET /languages

```json
{
  "scripts": {
    "devanagari": ["bho","brx","doi","gom","hi","mai","mr","ne","sa"],
    "telugu": ["te"], "tamil": ["ta"], "kannada": ["kn"],
    "arabic": ["ks","sd","ur"], "latin": ["en"]
  },
  "weak": ["kannada"],
  "unsupported": {"bn": "Bengali", "ml": "Malayalam", "...": "..."}
}
```

Requesting an unsupported language returns **400**, never a silent wrong-script
read:

```json
{"detail": "Bengali (bn) has no PaddleOCR recognition model at any version, ..."}
```

---

## Concurrency and why it is a pool

PaddleOCR predictors are **not thread-safe**, and `OCREngine` in auto mode
mutates its own state per image. Sharing one engine across a ThreadPoolExecutor
would corrupt inference. Building one per request is also out: construction
costs seconds and too many resident pipelines segfault the process.

So a fixed pool of engines is built at startup and checked out one thread at a
time. Models load once, concurrency equals `OCR_POOL_SIZE`, and memory stays
bounded no matter how many requests arrive. Worker count is
`min(cpu_count, pool_size, n_images)` — more threads than engines would only
queue.

Each auto-mode engine holds a detector, six probe recognisers and one pipeline:
roughly **350 MB**. Raise `OCR_POOL_SIZE` deliberately.

When every engine is busy for longer than `OCR_ACQUIRE_TIMEOUT`, the request
gets **503** with `Retry-After`, rather than queueing forever.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OCR_DEFAULT_LANG` | `auto` | Language when the request omits one |
| `OCR_POOL_SIZE` | `2` | Engines per language — the concurrency limit |
| `OCR_MAX_POOLS` | `4` | Distinct language specs kept resident |
| `OCR_MAX_WORKERS` | `0` | `0` derives from pool size |
| `OCR_ACQUIRE_TIMEOUT` | `120` | Seconds to wait for a free engine |
| `OCR_USE_GPU` | `false` | Needs `paddlepaddle-gpu` |
| `OCR_ALLOWED_ROOTS` | *(unset)* | Restrict `/ocr/folder`. **Set in production** |
| `OCR_MAX_BATCH_FILES` | `200` | Images per batch/folder request |
| `OCR_MAX_UPLOAD_BYTES` | `26214400` | Per-file upload cap |
| `OCR_MAX_ZIP_ENTRIES` | `500` | Zip-bomb guard |
| `OCR_MAX_ZIP_UNCOMPRESSED` | `536870912` | Zip-bomb guard |
| `OCR_WARMUP` | `true` | Load models at startup |
| `OCR_LOG_LEVEL` | `INFO` | |

## Layout

```
src/
  api.py                      FastAPI app, lifespan, health
  config.py                   env-driven settings
  routes/ocr.py               endpoints
  services/engine_pool.py     pre-built engines, checked out per thread
  services/batch_processor.py ThreadPoolExecutor + OCRResult -> schema
  models/request_models.py    request schemas
  models/response_models.py   response schemas
  utils/files.py              uploads, safe ZIP extraction, path fencing
  utils/logging_config.py     logging

  ocr_engine.py     ─┐
  detect_script.py   │ existing pipeline — unmodified
  scripts.py         │
  pipeline.py        │
  preprocess.py      │
  reading_order.py   │
  exporter.py       ─┘
```

## Logging

```
16:37:36 INFO  ocr.batch  batch start images=3 workers=2 lang=auto
16:37:64 INFO  ocr.batch  ok file=img1.jpg lang=latin blocks=45 conf=0.8776 time=25.58s
16:38:04 INFO  ocr.batch  batch done images=3 ok=2 failed=1 28.10s
```

## Verified behaviour

Exercised against a running server on real posters:

- `/ocr` — auto-detected script, returned text, blocks, confidence
- `/ocr/batch` — 2 images in parallel (28.1 s wall for two ~28 s images),
  corrupt file isolated to `failures`
- `/ocr/folder` — 6 files, **four different scripts detected in one folder**
  (latin, kannada, tamil, devanagari), 1 failure isolated
- `/ocr/zip` — results ZIP with `results.json` + per-image text; a `../../`
  member was rejected and did not escape
- `/languages`, `/health` — as documented
- `lang=bn` → 400; missing folder → 400
- 51 existing tests still pass; `run_ocr.py` output unchanged
