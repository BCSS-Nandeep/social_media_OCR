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
      // OCR_POOL_SIZE=1 deliberately: measured slower, not faster, with more
      // workers on this single GPU (real compute contention, not a config
      // issue) -- see DEPLOYMENT.md's Concurrency section before raising it.
      env: { PORT: "8000", OCR_POOL_SIZE: "1", GPU_NUMA_CPUS: "0-63,128-191" },
    },
    {
      // Separate process serving Qwen2.5-VL-7B-Instruct over vLLM's
      // OpenAI-compatible API. social-media-ocr-api talks to this over HTTP
      // (VLLM_BASE_URL) -- it never loads the model itself. See
      // DEPLOYMENT.md's vLLM section for GPU memory sizing.
      name: "vllm-qwen25vl",
      script: "scripts/start_vllm.sh",
      interpreter: "none",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      // VLLM_MAX_MODEL_LEN caps context well below the model's 128K default --
      // at 128K, vLLM couldn't fit even one KV-cache block in this memory
      // budget ("No available memory for the cache blocks"). See
      // scripts/start_vllm.sh and DEPLOYMENT.md's vLLM section.
      env: { VLLM_PORT: "8001", VLLM_GPU_MEMORY_UTILIZATION: "0.6", VLLM_MAX_MODEL_LEN: "16384", GPU_NUMA_CPUS: "0-63,128-191" },
    },
    {
      // Third GPU-resident process: general-purpose text LLM. Switched from
      // Qwen3-14B-AWQ to Qwen3-8B-AWQ -- same family/quantization, much
      // smaller weights, which is what actually buys the bigger context (see
      // the 1930 deployment: 14B at 0.97 util supported 16K context; 8B at
      // 0.85 util -- a SMALLER absolute budget -- supported 32K, because
      // smaller weights leave proportionally more room for KV cache).
      //
      // 0.30 (not 1930's 0.85) is deliberate: this GPU is shared with
      // Qwen2.5-VL (~27.8 GB) and a dormant IndicOCR fallback (~3.7 GB if it
      // ever loads) -- gpu-memory-utilization is a fraction of the WHOLE
      // card, not of what's free, so 0.85 here would ask for ~39 GB against
      // ~18 GB actually free and fail the same way the Qwen2.5-VL max-model-
      // len issue did. 0.30 (~13.8 GB) is a measured, conservative fit --
      // verify with nvidia-smi before raising further.
      name: "vllm-qwen3-llm",
      script: "scripts/start_vllm_qwen3.sh",
      interpreter: "none",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        QWEN3_VLLM_PORT: "8002",
        QWEN3_VLLM_MODEL: "Qwen/Qwen3-8B-AWQ",
        QWEN3_VLLM_SERVED_NAMES: "qwen3:8b-awq Qwen3-8B-AWQ qwen3-8b",
        QWEN3_VLLM_GPU_MEMORY_UTILIZATION: "0.30",
        QWEN3_VLLM_MAX_MODEL_LEN: "32768",
        QWEN3_VLLM_MAX_NUM_SEQS: "16",
        QWEN3_VLLM_MAX_NUM_BATCHED_TOKENS: "16384",
        GPU_NUMA_CPUS: "0-63,128-191",
      },
    },
  ],
};
