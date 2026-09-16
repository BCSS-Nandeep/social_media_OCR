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

### Known quirk: cold-start latency after a crash is variable

The first health check after a *fresh* process start succeeded in under 10
seconds. After the `kill -9` test specifically, the replacement process took
noticeably longer (~1-2 minutes) before `/health` responded, despite PM2
already reporting it `online` — PM2 considers a process "online" once it
forks, not once IndicOCR's own startup (model load + GPU warmup, done in
FastAPI's `lifespan`) actually completes. `nvidia-smi` showed real GPU
utilization throughout, so it wasn't hung, just slow — plausibly CUDA
context re-initialization after an unclean kill of the previous process.

**Practical implication**: don't assume the service can take traffic the
instant `pm2 list` shows `online` after any restart. Poll `/health` (or add
a PM2 `wait_ready`/health-check gate if this matters for your rollout) and
give it up to a couple of minutes on a cold or post-crash start.

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

`src/api.py` holds `OCR_POOL_SIZE` (default `4`, env-overridable) independent
IndicOCR instances in memory, each a full copy of the layout + recognition
models. A request checks one out of an `asyncio.Queue`, uses it exclusively,
and returns it when done — so requests *never fail for capacity reasons*;
beyond `OCR_POOL_SIZE` concurrent requests, the rest simply wait their turn
in the queue.

### How 4 was chosen, not assumed

Measured live on `acb`, not estimated from spec sheets:

```bash
# GPU memory watcher, running throughout
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits; sleep 1; done ) > /tmp/gpu_watch.log &

# Then fired the heaviest real test image against a single running worker
curl -X POST http://localhost:8000/extract -d @heaviest_test_payload.json ...
```

| | VRAM |
|---|---|
| Idle, one worker loaded | 3509 MiB |
| Peak, one worker mid-request (heaviest test image — several forced blocks, each read at up to the model's max crop resolution) | 4345 MiB |
| A10G total | 23028 MiB |

All `OCR_POOL_SIZE` workers are separate Python objects **inside the same
process**, not separate OS processes, so they share one CUDA context
instead of paying its ~1 GB-ish overhead per worker again — actual total
usage for 4 workers should land below a naive `4 × 4345 MiB`, though this
wasn't independently re-measured at full pool size beyond confirming the
service starts, stays under the 23 GB ceiling, and serves correctly (§6).
Re-run the same `nvidia-smi` watch during a burst of `OCR_POOL_SIZE`+
concurrent requests before raising `OCR_POOL_SIZE` further on this box, or
before deploying to a smaller GPU.

### Changing it

```bash
# In ecosystem.config.js, under env:
env: { PORT: "8000", OCR_POOL_SIZE: "6" }
```

Startup time scales roughly linearly with pool size — each worker loads and
warms up in turn (sequential by design, to avoid every worker's initial CUDA
allocation contending at once). Expect `OCR_POOL_SIZE` × (single-worker load
time) before `/health` reports `model_loaded: true` on a cold start.

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
