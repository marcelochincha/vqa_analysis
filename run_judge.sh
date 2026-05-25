#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

ENV_PIPELINE="${ENV_PIPELINE:-vqa-pipeline}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/r2_cleaned.csv}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
API_KEY="${API_KEY:-EMPTY}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
CONCURRENCY="${CONCURRENCY:-16}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

log() {
  echo "[run_judge] $*"
}

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found at ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

log "Running judge stage"
conda activate "${ENV_PIPELINE}"
python -m pipeline judge \
  --data "${DATA_PATH}" \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --batch-size "${BATCH_SIZE}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  "$@"

log "Done"
