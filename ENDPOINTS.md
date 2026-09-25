# External Endpoints Reference

What's actually running on the GPU box right now, one section per service:
base URL, every usable endpoint, and whether it's meant to be called by an
external application at all. For the full request/response contract of the
main API, see `API.md` (currently stale — pre-dates video and the
Qwen-default engine switch; this file is the accurate one for "what can I
call today").

## Overview

| Service | Port | External-facing? | Purpose |
|---|---|---|---|
| Social Media OCR + Video API | `8000` | **Yes** — this is the one external apps should call | Image OCR, video description |
| Qwen2.5-VL vLLM server | `8001` | No — internal implementation detail | Backs the API's OCR + video paths |
| Qwen3-14B-AWQ vLLM server | `8002` | No — internal, and unrelated to the OCR/video project | General-purpose text LLM, deployed separately on this same box |

Current host: `101.53.140.97`. **None of these ports are reachable from the
public internet right now** — confirmed via an external request to
`101.53.140.97:8000` returning `ECONNREFUSED`. This GPU pod is NAT'd; the
SSH-inbound IP and the pod's actual network position differ, similar to how
Jupyter (port 8888) needed its own provider-side proxy/endpoint
configuration. Until that's set up for these ports too, an external
application can only reach them via:

```bash
ssh -i "path/to/bcss-saga.pem" -L 8000:localhost:8000 root@101.53.140.97
# then call http://localhost:8000 from your own machine
```

or by whatever port-forwarding/"Endpoint" mechanism your GPU provider's
dashboard offers (see `DEPLOYMENT.md` §5-equivalent discussion) — same
tunnel pattern works for 8001/8002 if you ever need direct access to those.

No authentication exists on any of these three services. Do not expose them
publicly without adding one.

---

## 1. Social Media OCR + Video API — the one to integrate with

**Base URL**: `http://101.53.140.97:8000` (or `http://localhost:8000` through
the SSH tunnel above)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + which engine is active, whether IndicOCR fallback has ever loaded |
| `POST` | `/extract` | The one real endpoint — image OCR or video description |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc UI (auto-generated) |
| `GET` | `/openapi.json` | Raw OpenAPI schema |
| `GET` | `/` | Web frontend (`public/index.html`) for manually trying the API |

### `GET /health`

```bash
curl http://101.53.140.97:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "pool_size": 1,
  "workers_available": 0,
  "vlm_available": true,
  "vllm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "ocr_engine": "qwen",
  "indicocr_loaded": false,
  "video_available": true
}
```

`ocr_engine`/`vlm_available`/`video_available` describe the default path
(Qwen via vLLM). `indicocr_loaded` is `false` on a normal deployment —
IndicOCR only loads if Qwen fails and `INDICOCR_FALLBACK_ENABLED=true` on
the server; `pool_size`/`workers_available` describe that fallback pool
specifically, not the default path.

### `POST /extract`

**Headers**: `Content-Type: application/json`

**Request body** — exactly one of these three:

| Field | Type | Notes |
|---|---|---|
| `image_url` | string | `http(s)` only. Fetched server-side, max 25 MB, SSRF-guarded (private/internal IPs rejected). |
| `image_base64` | string | Raw image bytes, base64-encoded, no `data:...;base64,` prefix. Max 25 MB decoded. |
| `video_url` | string | `http(s)` only, same SSRF guard. Max 500 MB, max 3600s (60 min) duration. No `video_base64` — deliberately not implemented, to avoid huge JSON request bodies. |
| `min_confidence` | number, optional | `0.0`-`1.0`. Only affects the IndicOCR **fallback** path — Qwen reports no per-block confidence to filter on. |

Providing zero or more than one of the three media fields is a `422`.

**Response envelope** (always this shape, success or failure):

```json
{ "success": true | false, "data": {...} | null, "error": string | null }
```

**Image response** (`data`, when the request had `image_url`/`image_base64`):

```json
{
  "engine": "qwen",
  "image": { "path": null, "name": "image_base64", "width": 700, "height": 300 },
  "settings": { "preprocessing": ["none"] },
  "summary": {
    "text_blocks_detected": 3,
    "mean_confidence": null,
    "min_confidence": null,
    "blocks_dropped_below_threshold": 0,
    "total_processing_time_sec": 2.327,
    "stage_times_sec": { "ocr": 2.327, "total": 2.327 }
  },
  "full_text": "FARMERS SCHEME\nApply before 30th June\nCall 1800-111-222",
  "blocks": [
    { "index": 0, "order": 0, "label": "headline", "type": "headline",
      "text": "FARMERS SCHEME", "confidence": null, "bbox_xyxy": null }
  ],
  "error": null
}
```

`confidence`/`bbox_xyxy` are `null` on the default (Qwen) path — Qwen
doesn't produce a real detection confidence or bounding box, and this
service never fabricates one. `engine` says which engine actually produced
the response (`"qwen"` normally, `"indicocr"` only if the fallback fired —
see `DEPLOYMENT.md` §10). If `engine: "indicocr"` ever appears,
`confidence`/`bbox_xyxy` are real numbers from IndicDocLayout, not null.

**Video response** (`data`, when the request had `video_url`):

```json
{
  "media_type": "video",
  "duration_seconds": 90.0,
  "frames_processed": 10,
  "chunks_processed": 1,
  "description": "...",
  "summary": "...",
  "events": [ { "timestamp": 3.75, "description": "..." } ],
  "visible_text": ["..."],
  "video_meta": { "width": 320, "height": 240, "fps": 10.0 },
  "timing": { "download": 0.0, "metadata": 0.03, "frame_extraction": 0.05, "vlm": 12.6, "total": 12.7 }
}
```

Frame count follows duration: 4 frames ≤10s, 6 ≤30s, 8 ≤60s, 10 ≤2min,
12 ≤5min, 16 ≤10min; above 10 minutes the video is split into 10-minute
chunks (16 frames each, sequential), combined into one chronological
`description`/`summary`, up to the 60-minute hard cap.

**Errors** — same envelope, `success: false`, `data: null`:

| Status | Meaning |
|---|---|
| `400` | Bad request — unsupported scheme, blocked/private IP (SSRF), invalid base64 |
| `413` | Payload too large |
| `422` | Well-formed but unprocessable — wrong number of media fields, unreachable URL, undecodable media, duration over the cap, Qwen failure with fallback disabled |

### curl examples

```bash
curl -X POST http://101.53.140.97:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/poster.jpg"}'

curl -X POST http://101.53.140.97:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 poster.jpg)\"}"

curl -X POST http://101.53.140.97:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/clip.mp4"}'
```

### Python

```python
import requests

resp = requests.post(
    "http://101.53.140.97:8000/extract",
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

---

## 2. Qwen2.5-VL vLLM server — internal, backs the API above

**Base URL**: `http://101.53.140.97:8001` — the OCR/video API talks to this
at `http://127.0.0.1:8001/v1` internally. **Not meant to be called directly
by an external application** — call `/extract` on the main API instead,
which adds SSRF protection, validation, response-shape compatibility, and
(for video) frame sampling that talking to vLLM raw doesn't give you.
Documented here for completeness/debugging, not as an integration target.

OpenAI-compatible API (standard vLLM surface):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/models` | Lists the served model (`Qwen/Qwen2.5-VL-7B-Instruct`, `max_model_len: 16384`) |
| `POST` | `/v1/chat/completions` | Raw chat-completions call, accepts image content parts |

```bash
curl http://101.53.140.97:8001/v1/models
```

---

## 3. Qwen3-14B-AWQ vLLM server — internal, unrelated to the OCR project

**Base URL**: `http://101.53.140.97:8002`. This is a **separate**
general-purpose text LLM deployed on the same box (not part of the
OCR/video service, not called by `src/api.py` at all) — included here only
because it's a running service on the same host that an external
application could reach if the port were opened.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/models` | Lists served model names: `qwen3:14b-awq`, `Qwen3-14B-AWQ`, `qwen3-14b` |
| `POST` | `/v1/chat/completions` | Standard OpenAI-compatible chat completions (text only, 4096-token context, 4 concurrent sequences max) |

```bash
curl http://101.53.140.97:8002/v1/models
```

---

## GPU sharing note

All three services share one L40S GPU (46 GB). As of the last check:
Qwen2.5-VL ≈ 27.8 GB, Qwen3-14B-AWQ ≈ 13.1 GB, ~4.7 GB free, IndicOCR 0 GB
(loads only if the fallback ever fires). There is no headroom for a fourth
GPU-resident service or for raising any of these models' memory budgets
without first freeing something — check `nvidia-smi` before changing any of
`OCR_POOL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, or
`QWEN3_VLLM_GPU_MEMORY_UTILIZATION`.
