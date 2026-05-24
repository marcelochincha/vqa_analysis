#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

ENV_EMBED="${ENV_EMBED:-vqa-embed}"
ENV_PIPELINE="${ENV_PIPELINE:-vqa-pipeline}"

HUMAN_CSV="${HUMAN_CSV:-$ROOT_DIR/data/raw/humans/answers_raw_human.csv}"
VLM_DIR="${VLM_DIR:-$ROOT_DIR/data/raw/vlms}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/r2_cleaned.csv}"

RUN_SETUP="${RUN_SETUP:-true}"

log() {
  echo "[run_all] $*"
}

if [ "${RUN_SETUP}" = "true" ]; then
  log "Running setup (Miniconda + envs)"
  bash "${ROOT_DIR}/scripts/setup.sh"
fi

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found at ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

log "Preprocess: build cleaned CSV"
conda activate "${ENV_PIPELINE}"
python -m pipeline preprocess --human-csv "${HUMAN_CSV}" --vlm-dir "${VLM_DIR}"

log "Embeddings + pipeline"
DATA_PATH="${DATA_PATH}" bash "${ROOT_DIR}/scripts/run_all.sh"

log "Done"
