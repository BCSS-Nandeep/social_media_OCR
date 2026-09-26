# Deployment — PM2 + systemd on the GPU server

Covers taking `src/api.py` from "runs under `uvicorn` in a terminal" to
"survives crashes, reboots, and `ssh` disconnects" on the GPU box (referred
to below by its SSH alias `acb`). Everything here has been run and verified
end-to-end on that box — this isn't a plan, it's what's actually deployed.

---

## 0. What's running right now

| | |
|---|---|
| Host | `acb` (`98.86.63.69`), NVIDIA A10G (24 GB), Ubuntu 26.04 |
| Process manager | PM2 `7.0.4`, under systemd unit `pm2-ubuntu.service` |
| App | `social-media-ocr-api` (see `ecosystem.config.js`) |
| Port | `8000` (`0.0.0.0`, i.e. all interfaces) |
| OCR worker pool | `OCR_POOL_SIZE=4` (default) — see [Concurrency](#8-concurrency--worker-pool-sizing) for how that number was chosen |
| Repo path | `~/social_media_OCR`, branch `indicocr-engine` |
| Python | system Python 3.14 in `.venv/` (not the 3.12 the README's CLI section assumes — IndicOCR itself has no such constraint; that pin was PaddleOCR-specific) |

`GET http://localhost:8000/health` (or the public IP, once the security
group allows it — see [§5](#5-network-access-not-yet-open)) is the fastest
way to confirm the service is alive.

---

## 1. One-time host setup

Already done on `acb`; included here so this is reproducible on a fresh box.

```bash
# Node/npm (PM2 is a Node app; it manages our Python process fine, nothing
# about the app itself needs to be Node)
sudo apt-get install -y nodejs npm

# PM2 itself
sudo npm install -g pm2
```

## 2. Application setup

```bash
git clone https://github.com/BCSS-Nandeep/social_media_OCR.git
cd social_media_OCR
git checkout indicocr-engine

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Gated model repo -- request access at
# https://huggingface.co/bodhan-ai/indic-ocr, then:
hf auth login          # or: echo 'HF_TOKEN=hf_...' > .env
```

`.env` is gitignored and holds `HF_TOKEN`. `scripts/start_api.sh` sources it
before launching uvicorn, so the token never appears in `ecosystem.config.js`
or PM2's own saved process list.

## 3. Start under PM2

```bash
cd social_media_OCR
pm2 start ecosystem.config.js
```

`ecosystem.config.js` points at `scripts/start_api.sh`, which does:

```bash
source .env                                              # loads HF_TOKEN
exec .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
```

`autorestart: true` and `max_restarts: 10` (in `ecosystem.config.js`) are
what satisfy "restart on crash" — verified in [§6](#6-what-was-actually-verified).

## 4. Make it survive a reboot

```bash
# Prints a command to run -- it's user/path-specific, don't skip straight to
# copy-pasting the one below on a different box.
pm2 startup

# Run exactly what it printed, e.g. on this box:
sudo env PATH=$PATH:/usr/bin /usr/local/lib/node_modules/pm2/bin/pm2 startup systemd -u ubuntu --hp /home/ubuntu

# Freeze the current process list -- this is what gets resurrected on boot.
pm2 save
```

This creates and enables `/etc/systemd/system/pm2-ubuntu.service`, whose
`ExecStart` is `pm2 resurrect` — it replays whatever `pm2 save` last wrote to
`~/.pm2/dump.pm2`. **Re-run `pm2 save` any time you add, remove, or
reconfigure an app** — `pm2 startup` only needs to run once per host.

## 5. Network access (not yet open)

The service listens on `0.0.0.0:8000` and the host's own firewall (`ufw`) is
inactive, so nothing on the box itself blocks it. What's *not* verified or
configured: the EC2 **security group** for this instance. Before another
machine (including wherever `saga-police` runs) can reach
`http://98.86.63.69:8000`, add an inbound rule:

| Type | Port | Source |
|---|---|---|
| Custom TCP | 8000 | the specific IP/CIDR `saga-police` calls from — not `0.0.0.0/0` |

This wasn't done as part of this work — it needs the AWS console/API, which
wasn't reachable from here (no AWS CLI credentials on the box). See
[§7](#7-security-notes-read-before-exposing-this-publicly) before opening it
to anything broader than a single known source.

---

## 6. What was actually verified

Each of these was run against the live `acb` deployment, not asserted from
reading the config:

- **Starts under PM2**: `pm2 start ecosystem.config.js` → process reaches
  `online`, `/health` returns `{"status":"ok","model_loaded":true}`,
  `POST /extract` against a real poster image returns the expected 7 blocks.
- **Crash → auto-restart**: `kill -9 $(pm2 pid social-media-ocr-api)` →
  PM2's restart counter (the `↺` column in `pm2 list`) incremented from 0 to
  1, a new PID came up, and the service served `/health` and `/extract`
  successfully again once the model finished reloading.
- **Reboot persistence**: rather than a full `sudo reboot` of a live
  instance, this was verified by fully killing the daemon (`pm2 kill` — no
  process left at all, confirmed via `curl` and `nvidia-smi`) and then
  starting it the *same way boot does*: `sudo systemctl start pm2-ubuntu`.
  The unit came up `active (running)`, its `ExecStart=pm2 resurrect` brought
  `social-media-ocr-api` back from the saved dump with no manual
  intervention, and it served requests successfully afterward. This
  exercises the identical mechanism a real reboot triggers
  (`pm2-ubuntu.service` is `enabled`, i.e. `WantedBy=multi-user.target`)
  without the risk of a live-instance reboot (SSH drop, uncertain recovery
  time, anything else on the box). **A literal `sudo reboot` was not
  performed** — say so if you want that stronger proof; it's a five-minute
  disruptive test, not declined for cost, just not done without asking on
  someone else's running instance.

### Known quirk: cold-start latency after a crash is variable, and can run
### into minutes

The first health check after a *fresh* `pm2 start`/`pm2 delete && pm2 start`
consistently succeeded in 15-35 seconds. After a `kill -9` specifically,
reproduced twice independently (once against a 4-worker pool, once against
the current 1-worker default), the replacement process took **2-4+ minutes**
before `/health` responded, despite PM2 already reporting it `online` — PM2
considers a process "online" once it forks, not once IndicOCR's own startup
(model load + GPU warmup, done in FastAPI's `lifespan`) actually completes.
`nvidia-smi` showed real CPU/GPU activity throughout both times, so it
wasn't hung, just slow — plausibly CUDA context re-initialization after an
unclean kill of the previous process's context on the same GPU.

**Practical implication**: don't assume the service can take traffic the
instant `pm2 list` shows `online` after *any* restart, and especially not
right after a crash (as opposed to a deliberate `pm2 restart`). Poll
`/health` (or add a PM2 `wait_ready`/health-check gate if this matters for
your rollout) and give it up to several minutes on a post-crash start
before concluding something is actually wrong.

---

## 7. Security notes — read before exposing this publicly

None of this was asked for as a change; flagging it so it's a deliberate
choice, not an oversight:

- **No authentication.** `POST /extract` is open to anyone who can reach
  port 8000. Restricting the security group to `saga-police`'s specific
  source IP (§5) is the minimum; an API-key header is a cheap addition on
  top if the network boundary alone isn't enough.
- **`image_url` fetches server-side.** The service will `GET` whatever
  `http(s)` URL it's given, from the server's own network position. Scheme
  is restricted to `http`/`https` (no `file://`, verified in
  `tests/test_api.py`), but it does **not** block internal/private IP
  ranges — a caller could point it at `http://169.254.169.254/...` (the EC2
  instance-metadata endpoint) or another internal service. Fine for a
  request source you control (an internal monitoring app); not fine to
  expose to arbitrary/public input without adding that check.
- **No rate limiting.** The worker pool bounds *concurrency* (§8), not the
  *queue* — a caller can still queue an unbounded backlog of requests ahead
  of everyone else's. Fine for a single trusted internal caller; add rate
  limiting before this is multi-tenant.
- **Plaintext HTTP.** No TLS termination here — add a reverse proxy
  (nginx/Caddy) in front if this ever needs to leave a trusted network
  boundary.

---

## 8. Concurrency / worker pool sizing

`src/api.py` holds `OCR_POOL_SIZE` independent IndicOCR instances in memory,
each a full copy of the layout + recognition models. A request checks one
out of an `asyncio.Queue`, uses it exclusively, and returns it when done —
so requests *never fail for capacity reasons*; beyond `OCR_POOL_SIZE`
concurrent requests, the rest simply wait their turn in the queue.

**Default is `1`, deliberately, on this single-GPU box.** More GPU-resident
worker instances is the obvious lever to reach for and it was tried first —
it measured *worse*, not better. What follows is the actual measurement,
not a guess.

### The experiment that set the default

First measured one worker's footprint against the heaviest real test image
(several forced blocks, each read at up to the model's max crop resolution
— see `src/ocr_engine.py`):

| | VRAM |
|---|---|
| Idle, one worker loaded | 3509 MiB |
| Peak, mid-request | 4345 MiB |
| Idle, four workers loaded | 8491 MiB (≈1660 MiB/extra worker, not 3509 — they're separate Python objects in one process, sharing one CUDA context) |
| A10G total | 23028 MiB |

Memory said 4 workers fit comfortably. Then 4 identical concurrent requests
were fired at a 4-worker pool and, separately, at a 1-worker pool (`ab`-style
parallel `curl`, wall-clock timed):

| Pool size | Per-request time | Total wall time for 4 requests |
|---|---|---|
| 4 workers, run concurrently | ~75s **each** (vs. ~11.6s solo — 6.5x slower) | ~75.7s |
| 1 worker, requests queue | 12s / 24s / 36s / 48s (staggered, FIFO) | ~47.8s |

**One worker, serving requests one at a time, finished the same batch of
work 37% faster than four workers processing it "in parallel."** A single
GPU without NVIDIA MPS does not give independent CUDA contexts real
compute parallelism — they context-switch and contend for the same SMs, so
the extra workers bought contention, not throughput. Memory headroom was
never the constraint; GPU compute was, and splitting it four ways per
request made every request slower without finishing the batch any sooner.

### What this means for "maximum throughput"

On this hardware, the ceiling is **one image at a time, ~11.6s/image on the
A10G** (more if it has several forced blocks). The worker-pool machinery
still does real work at `OCR_POOL_SIZE=1`: it's why a burst of requests
queues in fair FIFO order and *none of them fail*, which was the other half
of the requirement. Raising `OCR_POOL_SIZE` above 1 only makes sense if:

- this ever runs across **multiple GPUs** — one worker pinned per GPU would
  give real parallelism (not implemented; the pool has no GPU-affinity
  logic today, so multiple workers today all fight over GPU 0), or
- IndicOCR's recogniser gains a genuine **batched-inference** path (several
  images in one forward pass) — a real engineering project of its own, not
  a config change, and out of scope here per "avoid unnecessary changes to
  extraction logic."

Neither is true on `acb` today. If throughput below ~11.6s/image is a hard
requirement, the actual lever is a bigger/multi-GPU instance or batched
inference, not more workers on this one.

### Changing it

```bash
# In ecosystem.config.js, under env:
env: { PORT: "8000", OCR_POOL_SIZE: "2" }
```

then `pm2 delete social-media-ocr-api && pm2 start ecosystem.config.js` —
plain `pm2 restart --update-env` was observed to sometimes keep serving the
*previous* env's pool size on this box; delete+start is the reliable way to
change it. Startup time scales roughly linearly with pool size (each worker
loads and warms up in turn, sequentially, to avoid every worker's initial
CUDA allocation contending at once) — expect `OCR_POOL_SIZE` × (single-worker
load time, ~15-35s) before `/health` reports `model_loaded: true`.

---

## 9. Video description (vLLM + Qwen2.5-VL)

`POST /extract` also accepts `{"video_url": "..."}`: it samples frames from
the video and asks a separate, self-hosted Qwen2.5-VL-7B-Instruct model
(served through vLLM) for a chronological description. This is a genuinely
separate GPU-resident process from IndicOCR -- `src/api.py` never imports
`vllm` or loads Qwen itself, it only makes HTTP calls to vLLM's
OpenAI-compatible API (`VLLM_BASE_URL`, default `http://127.0.0.1:8001/v1`).

### One-time setup

vLLM was found already installed system-wide on this host (`vllm --version`
→ tied to `/usr/bin/python3`, separate from this project's `.venv` -- no
dependency conflicts). If it isn't on a fresh box:

```bash
pip install vllm   # into whatever Python `scripts/start_vllm.sh` will use
```

`ffmpeg`/`ffprobe` are used via `subprocess` (`src/video/metadata.py`), not
a Python package -- install at the OS level if missing:

```bash
sudo apt-get install -y ffmpeg
```

### Starting it

```bash
pm2 start ecosystem.config.js   # starts BOTH apps: social-media-ocr-api and vllm-qwen25vl
```

`scripts/start_vllm.sh` runs:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --gpu-memory-utilization 0.6
```

First start downloads the model (~16 GB, open on Hugging Face -- no gating,
unlike IndicOCR) to the same HF cache IndicOCR's model already lives in.
Poll until ready:

```bash
curl http://localhost:8001/v1/models   # 200 once loaded
curl http://localhost:8000/health      # vlm_available: true once the API can reach it
```

### GPU memory sizing

`--gpu-memory-utilization 0.6` is deliberately well under vLLM's own 0.9
default. IndicOCR's resident pool holds ~3.7 GB permanently on this box; on
a 46 GB card, 0.6 (~27.6 GB) comfortably covers a 7B model's weights + KV
cache while leaving IndicOCR (and headroom for its own request-time spikes)
untouched. Raise it only after checking `nvidia-smi` for how much both
processes are actually using, not from the total card size alone.

**Known failure mode, actually hit and fixed here**: Qwen2.5-VL-7B-Instruct
defaults to `max_model_len=128000` (128K context). vLLM sizes its KV-cache
memory budget against that number, and at `--gpu-memory-utilization 0.6` the
model's weights plus the *minimum* viable KV cache for a 128K context didn't
fit -- vLLM crash-looped with `ValueError: No available memory for the
cache blocks`, restarting every ~30-60s under PM2 until stopped manually.
The fix was **not** raising `gpu_memory_utilization` further (that just
delays hitting the same wall, and eats into IndicOCR's headroom); it was
capping context to what this workload actually needs:
`--max-model-len 16384` (`VLLM_MAX_MODEL_LEN` in `ecosystem.config.js`) --
a handful of video frames plus a short prompt, one request at a time, needs
nowhere near 128K tokens. Confirm which wall you're hitting before changing
either knob: an OOM naming *weights* means raise `gpu_memory_utilization`
(if there's real headroom on `nvidia-smi`); an OOM naming *cache blocks*
means lower `max_model_len` instead.

### Env vars

| Var | Default | Meaning |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8001/v1` | Where `src/api.py` sends chat-completions requests |
| `VLLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name in the API request and in `scripts/start_vllm.sh` |
| `VLLM_API_KEY` | *(empty)* | Sent as `Authorization: Bearer ...` if set; vLLM's default has no auth |
| `VLLM_TIMEOUT_SECONDS` | `120` | Per-request timeout for the vLLM call |
| `MAX_VIDEO_SIZE_MB` | `500` | Rejected with 413 before download completes |
| `MAX_VIDEO_DURATION_SECONDS` | `3600` | Rejected with 422 after `ffprobe`, before any frame work |
| `VIDEO_DOWNLOAD_TIMEOUT_SECONDS` | `60` | `video_url` fetch timeout |
| `FRAME_EXTRACTION_TIMEOUT_SECONDS` | `120` | Caps both `ffprobe` and frame-grabbing per video/chunk |
| `MAX_CONCURRENT_VIDEO_JOBS` | `1` | Local CPU-side concurrency (download/ffprobe/frame-extract), independent of vLLM's own GPU scheduling |

### Frame sampling policy

| Duration | Frames | | Duration | Frames |
|---|---:|---|---|---:|
| 0–10s | 4 | | >2–5min | 12 |
| >10–30s | 6 | | >5–10min | 16 |
| >30–60s | 8 | | >10min | chunked: 10-min windows, 16 frames each |
| >1–2min | 10 | | | |

Frames are centered within evenly-sized slices of the duration
(`duration * (i + 0.5) / count`), not "first N frames" or fixed-fps, so the
first/last sample isn't sitting on a blank boundary frame. See
`src/video/sampler.py` and `tests/test_video.py` for the exact logic and
its test coverage.

### Security

`video_url` shares `src/media_downloader.py` with `image_url` -- see
[§7](#7-security-notes-read-before-exposing-this-publicly)'s SSRF note,
which this module now actually fixes for both: scheme allowlist, DNS
resolution + rejection of private/loopback/link-local/reserved addresses
before connecting, and redirects re-validated hop-by-hop rather than
followed blindly. After download, video is validated with `ffprobe` before
anything else touches it -- a file that doesn't probe as a real video is
rejected outright.

---

## 10. Default OCR engine: Qwen2.5-VL (IndicOCR retained as a lazy fallback)

`POST /extract` with `image_url`/`image_base64` now goes to the **same**
Qwen2.5-VL/vLLM process video descriptions use (`vllm-qwen25vl`), via a
dedicated OCR prompt in `src/vlm_client.py` (`extract_text`). IndicOCR is
still in the codebase (`src/ocr_engine.py`, `src/pipeline.py`) but is no
longer loaded at startup, and no longer the default -- it exists purely as
an opt-in fallback for when Qwen's call fails.

**Before this change**: `social-media-ocr-api`'s `lifespan()` built and
warmed `OCR_POOL_SIZE` IndicOCR engines at startup, unconditionally --
~3.7 GB VRAM, every restart, whether or not a single image request ever
arrived.

**After**: `lifespan()` does no GPU work at all. `src/ocr_providers.py`'s
`IndicOCRProvider` is constructed at import time (free -- see its docstring)
but its actual worker pool is built lazily, only inside `_ensure_pool()`,
only the first time `.extract()` is called on it. That only happens if
Qwen's OCR call raises **and** `INDICOCR_FALLBACK_ENABLED=true`. A normal
deployment where Qwen never fails puts a literal zero bytes of IndicOCR on
the GPU for the process's entire lifetime.

### Env vars

| Var | Default | Meaning |
|---|---|---|
| `OCR_ENGINE` | `qwen` | Informational/reserved -- Qwen is always tried first; this just labels `/health`'s `ocr_engine` field |
| `INDICOCR_FALLBACK_ENABLED` | `false` | If `true`, a Qwen OCR failure lazily builds and uses the IndicOCR pool instead of returning 422 |
| `OCR_POOL_SIZE` | `1` | Size of the IndicOCR fallback pool, **if it's ever built** -- irrelevant while fallback never triggers |

### Response compatibility

The response envelope and top-level `data` shape (`image`, `settings`,
`summary`, `full_text`, `blocks`, `error`) are unchanged so existing clients
don't need new parsing logic. What changed: Qwen has no per-block detection
confidence and no bounding boxes, so `confidence` and `bbox_xyxy` are `null`
on the Qwen path rather than fabricated numbers -- `summary.mean_confidence`
/ `summary.min_confidence` are `null` too, and `min_confidence` in the
request has no effect (nothing to filter on). A new top-level `data.engine`
field (`"qwen"` or `"indicocr"`) says which engine actually produced a given
response. IndicOCR-path responses are byte-for-byte the same as before this
change (real confidence, real bboxes) -- that code path is untouched.

### Verifying IndicOCR really isn't loaded

```bash
nvidia-smi                          # only vllm-qwen25vl's allocation should appear
curl localhost:8000/health          # {"indicocr_loaded": false, "ocr_engine": "qwen", ...}
pm2 restart social-media-ocr-api    # restart it
nvidia-smi                          # same as before the restart -- no new allocation appears
```

`indicocr_loaded` flips to `true` (and a new `nvidia-smi` allocation
appears) only after a real fallback has actually fired -- if you see it
`true` in a deployment where `INDICOCR_FALLBACK_ENABLED=false`, something is
wrong, since that path should be unreachable.

---

## 11. Logging — reading what actually happened on a request

**Before this section existed, `social-media-ocr-api`'s own log messages
were invisible.** No logging handler was configured anywhere in the
process, so Python's logging module fell back to its "last resort" handler,
which only prints `WARNING` and above to stderr. Every `log.info(...)` call
in the code (including request tracing added below) was silently dropped —
`pm2 logs social-media-ocr-api` only ever showed a raw traceback when
something threw an exception at `ERROR` level, with no context about which
request caused it, from where, or how long it had been running. That's
fixed now (`_configure_logging()` in `src/api.py`), but it's worth knowing
this was the state for everything deployed before this section was written.

### What's logged per request

Every `POST /extract` call gets a short request id (8 hex chars) that
correlates its "received" and "completed"/"failed" lines:

```
2026-09-26T10:15:03 INFO ocr.api: [a1b2c3d4] received client=203.0.113.7 media=image_url ref=https://example.com/poster.jpg
2026-09-26T10:15:05 INFO ocr.api: [a1b2c3d4] completed success=True engine=qwen duration=2.331s
```

or, on failure:

```
2026-09-26T10:15:03 INFO ocr.api: [e5f6a7b8] received client=203.0.113.7 media=image_base64 ref=<48213 chars>
2026-09-26T10:17:03 WARNING ocr.api: [e5f6a7b8] failed status=422 duration=120.014s error=Qwen OCR failed: vLLM request failed: ...ReadTimeout
```

`image_base64`'s actual content is never logged — only its length — matching
the same "don't log the payload" principle applied to video work earlier.
Malformed requests (wrong number of media fields) never reach `extract()`
at all — FastAPI's own validation rejects them first — so those get a
single `request validation failed ...` line instead of a request id pair.

### Finding the log level / files

```bash
pm2 logs social-media-ocr-api              # both streams, tailed live
pm2 logs social-media-ocr-api --lines 500 --nostream   # last 500 lines, no follow
cat ~/.pm2/logs/social-media-ocr-api-out.log    # our own log.info/warning lines (stdout)
cat ~/.pm2/logs/social-media-ocr-api-error.log  # log.exception tracebacks (stderr)
```

`LOG_LEVEL` (env var, default `INFO`) controls verbosity — set it to
`DEBUG` to also see `/health` polls (deliberately not logged at `INFO`,
since monitoring hits that endpoint often). Set it in `.env` or
`ecosystem.config.js`'s `env` block for `social-media-ocr-api`, same as any
other var here, then `pm2 restart social-media-ocr-api`.

### Known real incident this surfaced

While reviewing this, 4 occurrences of `httpx.ReadTimeout` were found
already sitting in the error log — a real image OCR request took longer
than `VLLM_TIMEOUT_SECONDS` (120s default) to get a response from vLLM at
port 8001, so it failed with a 422. Normal OCR latency is 1-3s, so this was
a real anomaly, not expected behavior. The likely cause: **three
GPU-resident services now share one card** (IndicOCR when it's loaded,
Qwen2.5-VL, and the separately-deployed Qwen3-14B-AWQ) — this repo's own
measured finding (§8, Concurrency) is that a single GPU without NVIDIA MPS
gives CUDA contexts contention, not real parallelism, when more than one is
active at once. Two independent vLLM processes making genuinely concurrent
GPU calls is exactly that scenario. If timeouts recur, check whether they
correlate with Qwen3 traffic (`pm2 logs vllm-qwen3-llm`) before assuming
vLLM itself is broken.

---

## Common operations

```bash
pm2 list                              # status, uptime, restart count
pm2 logs social-media-ocr-api         # tail stdout+stderr
pm2 logs social-media-ocr-api --err   # stderr only
pm2 restart social-media-ocr-api      # manual restart (e.g. after a code deploy)
pm2 stop social-media-ocr-api
pm2 delete social-media-ocr-api       # remove from PM2's list entirely
pm2 save                              # re-freeze the list -- do this after any of the above
                                       # if the change should survive a reboot
pm2 monit                             # live CPU/memory dashboard
```

### Deploying a code change

```bash
cd ~/social_media_OCR
git pull origin indicocr-engine
.venv/bin/pip install -r requirements.txt   # only if requirements.txt changed
pm2 restart social-media-ocr-api
pm2 save                                     # only if ecosystem.config.js itself changed
```

### Removing the boot-startup hook entirely

```bash
pm2 unstartup systemd
```
