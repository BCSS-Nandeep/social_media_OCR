// PM2 process definition. Usage: pm2 start ecosystem.config.js
// See DEPLOYMENT.md for the full setup (Node/PM2 install, startup, save).
//
// GPU_NUMA_CPUS pins all three GPU-resident processes below to the CPU
// cores on the NUMA node the GPU is actually attached to (check with
// 'nvidia-smi topo -m' -- the GPU0 row's 'CPU Affinity' column). This
// avoids cross-socket memory latency on host<->GPU transfers; it does NOT
// change worker/pool counts -- see the OCR_POOL_SIZE comment below for why
// more GPU-resident workers made things slower, not faster, on this
// single-GPU-no-MPS setup.
//
// Redeployed 2026-10-01 onto a fresh pod with a much bigger card (NVIDIA
// A100 80GB, vs the previous L40S 46GB) and a different CPU topology
// (0-63, not 0-63,128-191 -- verified via 'nvidia-smi topo -m' on THIS
// host; don't copy the old range blind onto different hardware). GPU
// memory fractions below were recalibrated for the extra headroom -- see
// each app's own comment for the reasoning. IndicOCR is dead code (see
// src/api.py's module docstring) and never loads regardless of hardware.
module.exports = {
  apps: [
    {
      name: "social-media-ocr-api",
      script: "scripts/start_api.sh",
      interpreter: "none",   // the script has its own #!/usr/bin/env bash shebang
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      // OCR_POOL_SIZE is vestigial -- IndicOCR (the only thing that read it)
      // has no fallback wiring left in src/api.py at all. Kept here only
      // because removing it isn't worth a redeploy of its own; it does
      // nothing.
      env: { PORT: "8000", OCR_POOL_SIZE: "1", GPU_NUMA_CPUS: "0-63" },
    },
    {
      // Separate process serving Qwen2.5-VL-7B-Instruct over vLLM's
      // OpenAI-compatible API. social-media-ocr-api talks to this over HTTP
      // (VLLM_BASE_URL) -- it never loads the model itself.
      //
      // On the 80GB A100: 0.45 util (~36.9 GB) comfortably covers the ~15GB
      // weights plus a much bigger KV cache than the old 46GB card could
      // afford -- max-model-len doubled to 32768 accordingly. Still well
      // under half the card, leaving real room for vllm-qwen3-llm below.
      name: "vllm-qwen25vl",
      script: "scripts/start_vllm.sh",
      interpreter: "none",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: { VLLM_PORT: "8001", VLLM_GPU_MEMORY_UTILIZATION: "0.45", VLLM_MAX_MODEL_LEN: "32768", GPU_NUMA_CPUS: "0-63" },
    },
    {
      // Third GPU-resident process: general-purpose text LLM, Qwen3-14B-AWQ.
      //
      // On the 80GB A100: 0.34 util (~27.9 GB) alongside Qwen2.5-VL's 36.9 GB
      // leaves ~17 GB genuinely free as a safety margin -- both numbers are
      // calculated, not yet empirically verified on this specific card;
      // confirm with the real vLLM preflight check (it reports the exact
      // free-vs-requested numbers on failure, same as every prior tuning
      // pass this service has gone through) and adjust if it doesn't fit
      // first try. max_model_len doubled to 32768, max_num_seqs raised
      // 4 -> 8 since there's room for more concurrent long-context requests
      // now, not just a bigger single one.
      name: "vllm-qwen3-llm",
      script: "scripts/start_vllm_qwen3.sh",
      interpreter: "none",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        QWEN3_VLLM_PORT: "8002",
        QWEN3_VLLM_MODEL: "Qwen/Qwen3-14B-AWQ",
        QWEN3_VLLM_SERVED_NAMES: "qwen3:14b-awq Qwen3-14B-AWQ qwen3-14b",
        QWEN3_VLLM_GPU_MEMORY_UTILIZATION: "0.34",
        QWEN3_VLLM_MAX_MODEL_LEN: "32768",
        QWEN3_VLLM_MAX_NUM_SEQS: "8",
        QWEN3_VLLM_MAX_NUM_BATCHED_TOKENS: "16384",
        GPU_NUMA_CPUS: "0-63",
      },
    },
  ],
};
