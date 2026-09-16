# API Reference — Social Media OCR extraction service

REST wrapper around the IndicOCR pipeline (`src/pipeline.py`). One route:
give it an image, get back the text on it, plus per-region layout detail.
Deployment/ops (PM2, systemd, network access) are in `DEPLOYMENT.md` — this
document is the integration contract for a *consumer* of the API, who
shouldn't need to know or care that IndicOCR, PM2, or any of this repo's
internals exist.

---

## Scope — read this before wiring anything up

This service extracts text **from an image**. It does not fetch or parse a
social-media *post* (an Instagram/Facebook/Twitter permalink, a post ID,
platform API data) — it has no scraping or platform-API logic at all. If
your input is a post URL, your application resolves that to the actual
**image bytes or a direct image URL** first (e.g. the CDN URL the post's
media actually lives at, or bytes you already downloaded via Cloudinary/S3/
wherever you store fetched media) and hands *that* to this service.

## Base URL

```
http://<host>:8000
```

Currently deployed at `http://98.86.63.69:8000` on the `acb` GPU box (see
`DEPLOYMENT.md` §5 — the security group is **not yet open** to external
callers; that needs to be done in AWS before this is reachable from
anywhere but the box itself). No path prefix, no API version segment yet.

## Authentication

**None currently.** Any caller that can reach the port can call `/extract`.
See `DEPLOYMENT.md` §7 for why that matters and what to do about it before
exposing this beyond a trusted network.

---

## Endpoints

### `GET /health`

Liveness + whether the model has finished loading. No auth, no parameters.

```bash
curl http://98.86.63.69:8000/health
```

```json
{ "status": "ok", "model_loaded": true, "pool_size": 4, "workers_available": 3 }
```

`pool_size` is the number of independent OCR workers held in memory
(`OCR_POOL_SIZE` env var, default 4 — see `DEPLOYMENT.md`). `workers_available`
is how many are idle right now; if it's `0`, incoming requests are queuing
rather than failing (see Processing behavior below). Always `200` if the
process is up at all.

### `POST /extract`

Extract text from one image.

**Headers**

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `image_url` | string | exactly one of `image_url` / `image_base64` | `http://` or `https://` only. Fetched server-side, max 25 MB, 15s timeout. |
| `image_base64` | string | exactly one of `image_url` / `image_base64` | Raw image bytes, base64-encoded (no `data:image/...;base64,` prefix — just the encoded bytes). Max 25 MB decoded. |
| `min_confidence` | number | no (default `0.0`) | Drop layout blocks scoring below this, `0.0`-`1.0`. This is layout-**detection** confidence, not a transcription score — see the note in the response section below. |

Providing both or neither of `image_url`/`image_base64` is a `422`.

**Supported image types**: JPEG, PNG, BMP, WebP, TIFF — anything OpenCV's
`imdecode` handles. Not PDFs, not video, not a raw social-media post link
(see Scope above).

---

## Response structure

Every response — success or failure — is:

```json
{ "success": true | false, "data": {...} | null, "error": string | null }
```

### Success (`200`)

`data` is the pipeline's own result record verbatim (`OCRResult.to_dict()`
in `src/exporter.py` — nothing reshaped for this API):

```json
{
  "success": true,
  "data": {
    "image": { "path": "/tmp/tmpXXXX.jpg", "name": "tmpXXXX.jpg", "width": 568, "height": 712 },
    "settings": { "preprocessing": ["none"] },
    "summary": {
      "text_blocks_detected": 7,
      "mean_confidence": 0.6969,
      "min_confidence": 0.535,
      "blocks_dropped_below_threshold": 0,
      "total_processing_time_sec": 11.6,
      "stage_times_sec": { "load": 0.006, "preprocess": 0.0, "ocr": 11.6, "total": 11.6 }
    },
    "full_text": "BANQUBTS     FULLY AIR-CONDITIONED HALLS\n...",
    "blocks": [
      {
        "index": 0,
        "order": 0,
        "label": "Header",
        "type": "PageHeader",
        "text": "BANQUBTS     FULLY AIR-CONDITIONED HALLS\n9603610361     QUALIFIED TRAINED MANAGEMENT",
        "confidence": 0.89,
        "bbox_xyxy": [6.9, 4.5, 567.2, 78.2]
      }
    ],
    "error": null
  },
  "error": null
}
```

Field notes:

- **`full_text`** — the whole page's text, reading-order, blocks joined by
  a blank line. This is almost always what you want if you just need "the
  text on this poster" and don't care about layout.
- **`blocks[].confidence`** — IndicDocLayout's **detection** confidence
  (how sure the model is a region of that type exists there), not a
  transcription/OCR confidence score — IndicOCR doesn't expose one
  separately. A `Paragraph` at `0.95` confidence can still contain a
  misread word; low confidence is a real signal to review, high confidence
  is not a correctness guarantee.
- **`blocks[].label`** / **`type`** — IndicDocLayout's raw 37-class layout
  label (`Paragraph`, `Title`, `Table`, `Image`, ...) and a coarser type
  grouping. An `Image`/`Header`/`Footer`/`Chart`/`Diagram`/`Advertisement`
  block can still carry real transcribed `text` — this service forces
  those through the recognizer rather than leaving them blank, since
  social-media posters routinely bake real content (a masthead, a price,
  a headline) into exactly that kind of region. See `src/ocr_engine.py`
  for why and how (dedup + hallucination filtering included).
- **`blocks[].bbox_xyxy`** — `[x_min, y_min, x_max, y_max]` in pixels of
  the image as submitted.
- **`image.path`/`image.name`** — a server-side temp filename, not
  meaningful to the caller; ignore it.

### Error responses

| Status | Meaning | Example `error` |
|---|---|---|
| `400` | Bad request — malformed input | `"Invalid image_base64: Invalid base64-encoded string..."` / `"Unsupported URL scheme: 'file'"` |
| `413` | Payload too large | `"image_url exceeds 25 MB."` |
| `422` | Request was well-formed but couldn't be processed | neither/both of `image_url`/`image_base64` given; `image_url` unreachable/timed out; the file wasn't a decodable image; an internal pipeline error |
| `503` | *(reserved — not currently returned; a request whose turn hasn't come up yet in the worker pool queues instead of getting rejected, see Processing behavior)* | |

```json
{ "success": false, "data": null, "error": "Provide exactly one of image_url or image_base64." }
```

---

## Processing behavior

- **A pool of workers, not one.** `OCR_POOL_SIZE` (default 4) independent
  IndicOCR instances share the GPU, each fully capable of handling a
  request end to end. A request checks a worker out of an internal queue,
  uses it, and returns it — if all workers are busy, the request **waits in
  the queue, it does not get rejected**. There is no capacity-based error
  response; the only failures are per-request (bad image, unreachable URL).
- Plan for roughly **4-12 seconds per image** per worker (measured on an
  A10G; more if the image has several
  Image/Header/Footer/Chart/Diagram/Advertisement-labeled regions, since
  each of those is read twice — see `src/ocr_engine.py`). With the default
  pool of 4, up to 4 images process genuinely concurrently before a 5th
  request starts queuing.
- **No request timeout is enforced by this service** beyond what your HTTP
  client sets. Set a client-side timeout that accounts for queue depth
  under load, not just one image's processing time — e.g. 60s covers a
  single image comfortably, but a burst of requests beyond the pool size
  should budget for queuing time on top of that.
- **Cold start**: the very first request after the process starts (or
  restarts) pays the model-load cost inline if `/health` hasn't already
  triggered warmup — normally seconds, but see `DEPLOYMENT.md`'s note on
  variable cold-start latency after a crash.

---

## Integration examples

### curl

```bash
# By URL
curl -X POST http://98.86.63.69:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/poster.jpg"}'

# By base64
curl -X POST http://98.86.63.69:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 poster.jpg)\"}"
```

### Python

```python
import requests

resp = requests.post(
    "http://98.86.63.69:8000/extract",
    json={"image_url": "https://example.com/poster.jpg"},
    timeout=60,
)
resp.raise_for_status()
result = resp.json()

if result["success"]:
    print(result["data"]["full_text"])
else:
    print("extraction failed:", result["error"])
```

Sending bytes you already have instead of a URL:

```python
import base64
import requests

with open("poster.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://98.86.63.69:8000/extract",
    json={"image_base64": encoded},
    timeout=60,
)
print(resp.json())
```

### Node.js (axios) — matches `saga-police`'s own stack

```javascript
const axios = require("axios");

async function extractText(imageUrl) {
  const { data: result } = await axios.post(
    "http://98.86.63.69:8000/extract",
    { image_url: imageUrl },
    { timeout: 60000 }
  );

  if (!result.success) {
    throw new Error(`OCR extraction failed: ${result.error}`);
  }
  return result.data.full_text;
}

extractText("https://example.com/poster.jpg")
  .then((text) => console.log(text))
  .catch((err) => console.error(err.message));
```

Sending bytes already fetched (e.g. from Cloudinary, or a Buffer from
`agent-twitter-client`) instead of a URL:

```javascript
const axios = require("axios");

async function extractTextFromBuffer(imageBuffer) {
  const { data: result } = await axios.post(
    "http://98.86.63.69:8000/extract",
    { image_base64: imageBuffer.toString("base64") },
    { timeout: 60000 }
  );
  if (!result.success) throw new Error(result.error);
  return result.data.full_text;
}
```

---

## OpenAPI / interactive docs

FastAPI generates these automatically from `src/api.py` — no separate
maintenance:

- Swagger UI: `http://98.86.63.69:8000/docs`
- ReDoc: `http://98.86.63.69:8000/redoc`
- Raw schema: `http://98.86.63.69:8000/openapi.json`
