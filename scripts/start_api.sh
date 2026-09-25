#!/usr/bin/env bash
# Entry point PM2 runs. Kept as a shell script rather than baking the command
# into ecosystem.config.js so HF_TOKEN and friends load from .env at process
# start -- .env is gitignored, so nothing secret ends up in the repo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Pinned to the NUMA node the GPU is actually attached to (see
# 'nvidia-smi topo -m' -- GPU0's CPU Affinity row) so CUDA host-side work
# (pinned memory transfers, IndicOCR pre/post-processing) isn't scheduled
# onto the far socket and paying cross-NUMA memory latency on every
# request. Override GPU_NUMA_CPUS if this runs on different hardware.
exec taskset -c "${GPU_NUMA_CPUS:-0-63,128-191}" .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
