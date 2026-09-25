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
      // Third GPU-resident process: general-purpose text LLM (Qwen3-14B-AWQ),
      // separate from the video-description model above. Sized to fit in
      // whatever's left after IndicOCR + Qwen2.5-VL -- see
      // scripts/start_vllm_qwen3.sh for the exact memory-fit rationale
      // (0.28 utilization, 4K context, 4 concurrent sequences; verified
      // working, do not raise without checking nvidia-smi headroom first).
      name: "vllm-qwen3-llm",
      script: "scripts/start_vllm_qwen3.sh",
      interpreter: "none",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        QWEN3_VLLM_PORT: "8002",
        QWEN3_VLLM_GPU_MEMORY_UTILIZATION: "0.28",
        QWEN3_VLLM_MAX_MODEL_LEN: "4096",
        QWEN3_VLLM_MAX_NUM_SEQS: "4",
        QWEN3_VLLM_MAX_NUM_BATCHED_TOKENS: "4096",
        GPU_NUMA_CPUS: "0-63,128-191",
      },
    },
  ],
};
