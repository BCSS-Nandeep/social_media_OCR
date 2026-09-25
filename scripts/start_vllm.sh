#!/usr/bin/env bash
# Entry point PM2 runs for the vLLM server. Separate script (not inlined in
# ecosystem.config.js) for the same reason as start_api.sh: .env is
# gitignored, so HF_TOKEN and any future overrides load at process start
# without landing in a committed file or PM2's saved process list.
#
# vLLM runs as its OWN process, using the system Python it's installed
# into -- NOT this project's .venv. src/api.py never imports vllm or loads
# Qwen itself; it only talks to this server over HTTP (VLLM_BASE_URL).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# --gpu-memory-utilization is capped well below vLLM's 0.9 default so this
# coexists safely with the IndicOCR process's own resident GPU memory on the
# same card -- see DEPLOYMENT.md's vLLM section for the measurements this is
# based on.
#
# --max-model-len overrides Qwen2.5-VL-7B-Instruct's own default of 128000
# (128K context). vLLM sizes its KV-cache memory budget off this number --
# left at the model's default, weights + the minimum viable KV cache for a
# 128K context didn't fit even at 0.6 utilization ("No available memory for
# the cache blocks"). This workload (a handful of video frames + a short
# prompt, one request at a time) needs nowhere near 128K tokens; 16384 is
# generous headroom over the actual usage and leaves real room for the KV
# cache within the memory budget. Raise it only if a real request needs
# more context AND `nvidia-smi` shows the room to do it.
#
# Pinned to the GPU-local NUMA node (see start_api.sh's comment and
# 'nvidia-smi topo -m') -- same rationale, applies equally to this process.
exec taskset -c "${GPU_NUMA_CPUS:-0-63,128-191}" vllm serve "${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --port "${VLLM_PORT:-8001}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}"
