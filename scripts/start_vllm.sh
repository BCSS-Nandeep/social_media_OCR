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
exec vllm serve "${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --port "${VLLM_PORT:-8001}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
