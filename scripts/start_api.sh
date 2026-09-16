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

exec .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
