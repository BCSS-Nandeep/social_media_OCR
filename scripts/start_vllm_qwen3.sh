#!/usr/bin/env bash
# Entry point PM2 runs for the text-only Qwen3-14B-AWQ vLLM server --
# separate GPU-resident process from both IndicOCR (start_api.sh) and the
# Qwen2.5-VL video-description model (start_vllm.sh); all three share one
# GPU. Mirrors start_vllm.sh's structure; see that file's comments for why
# .env is sourced here rather than baked into ecosystem.config.js.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Budget is tight: IndicOCR (~3.7-4.3 GB resident) + Qwen2.5-VL at
# --gpu-memory-utilization 0.6 (~27.6 GB of the 46 GB card) already leaves
# only ~14.6 GB free. --gpu-memory-utilization here is a fraction of the
# FULL card (not of the free remainder), and vLLM's own pre-flight check
# requires strictly less than the memory actually free at startup -- 0.30
# (13.36 GiB) failed against 13.28 GiB free by ~80 MB; 0.28 (~12.47 GiB)
# is the verified-working value with a real margin. Don't raise this
# without checking 'nvidia-smi' for actual headroom first.
#
# --max-model-len/--max-num-seqs/--max-num-batched-tokens are deliberately
# small (4K context, 4 concurrent sequences) to fit the KV cache inside
# that same budget alongside the AWQ weights (~8.76 GiB) -- verified to
# yield a 30,608-token KV cache pool, comfortably covering 4x4096 worst
# case. Do not raise without re-verifying free VRAM and retesting.
#
# Pinned to the GPU-local NUMA node -- see start_api.sh's comment and
# 'nvidia-smi topo -m'.
exec taskset -c "${GPU_NUMA_CPUS:-0-63,128-191}" vllm serve "${QWEN3_VLLM_MODEL:-Qwen/Qwen3-14B-AWQ}" \
  --served-model-name ${QWEN3_VLLM_SERVED_NAMES:-qwen3:14b-awq Qwen3-14B-AWQ qwen3-14b} \
  --host 0.0.0.0 --port "${QWEN3_VLLM_PORT:-8002}" \
  --dtype auto --quantization awq \
  --gpu-memory-utilization "${QWEN3_VLLM_GPU_MEMORY_UTILIZATION:-0.28}" \
  --max-model-len "${QWEN3_VLLM_MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${QWEN3_VLLM_MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${QWEN3_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}" \
  --enable-prefix-caching --kv-cache-dtype fp8 \
  --trust-remote-code --enforce-eager
