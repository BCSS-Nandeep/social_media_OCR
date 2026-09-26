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
      // Third GPU-resident process: general-purpose text LLM, back on
      // Qwen3-14B-AWQ per explicit request (briefly ran Qwen3-8B-AWQ --
      // reverted). 16384 context (up from the original 4096) needs a bigger
      // absolute budget than 8B did for the same context, since 14B's
      // larger weights leave less of the budget for KV cache. 0.38 was
      // tried first and failed by ~60MB (vLLM's own preflight check said:
      // "Free memory ... 16.86 GiB ... less than desired ... 16.92 GiB") --
      // 0.36 (~16.0 GiB) leaves real margin against that measured ceiling.
      // This does NOT touch social-media-ocr-api or vllm-qwen25vl -- if
      // this budget is wrong, only this process fails to start.
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
        QWEN3_VLLM_GPU_MEMORY_UTILIZATION: "0.36",
        QWEN3_VLLM_MAX_MODEL_LEN: "16384",
        QWEN3_VLLM_MAX_NUM_SEQS: "4",
        QWEN3_VLLM_MAX_NUM_BATCHED_TOKENS: "16384",
        GPU_NUMA_CPUS: "0-63,128-191",
      },
    },
  ],
};
