// PM2 process definition. Usage: pm2 start ecosystem.config.js
// See DEPLOYMENT.md for the full setup (Node/PM2 install, startup, save).
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
      env: { PORT: "8000", OCR_POOL_SIZE: "1" },
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
      env: { VLLM_PORT: "8001", VLLM_GPU_MEMORY_UTILIZATION: "0.6", VLLM_MAX_MODEL_LEN: "16384" },
    },
  ],
};
