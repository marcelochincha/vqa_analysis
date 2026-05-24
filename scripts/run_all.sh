#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

ENV_EMBED="${ENV_EMBED:-vqa-embed}"
ENV_PIPELINE="${ENV_PIPELINE:-vqa-pipeline}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/r2_cleaned.csv}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
EMBED_OUTPUT="${EMBED_OUTPUT:-$ROOT_DIR/external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
EMBED_DEVICE="${EMBED_DEVICE:-}"
EMBED_TRUST_REMOTE_CODE="${EMBED_TRUST_REMOTE_CODE:-false}"
EMBED_NPZ="${EMBED_NPZ:-}"
EMBED_NORMALIZE="${EMBED_NORMALIZE:-false}"

PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-false}"
PIPELINE_OUTDIR="${PIPELINE_OUTDIR:-}"

log() {
  echo "[run_all] $*"
}

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found at ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

log "Generating embeddings"
conda activate "${ENV_EMBED}"
embed_cmd=(
  python "${ROOT_DIR}/scripts/generate_embeddings.py"
  --model "${EMBED_MODEL}"
  --data "${DATA_PATH}"
  --output "${EMBED_OUTPUT}"
  --batch-size "${EMBED_BATCH_SIZE}"
  --resume
)
if [ -n "${EMBED_DEVICE}" ]; then
  embed_cmd+=(--device "${EMBED_DEVICE}")
fi
if [ "${EMBED_TRUST_REMOTE_CODE}" = "true" ]; then
  embed_cmd+=(--trust-remote-code)
fi
if [ -n "${EMBED_NPZ}" ]; then
  embed_cmd+=(--npz "${EMBED_NPZ}")
fi
if [ "${EMBED_NORMALIZE}" = "true" ]; then
  embed_cmd+=(--normalize)
fi
"${embed_cmd[@]}"

log "Running pipeline stages"
conda activate "${ENV_PIPELINE}"
pipeline_cmd=(
  python -m pipeline
  --all
  --data "${DATA_PATH}"
  --embeddings "${EMBED_OUTPUT}"
)
if [ "${PIPELINE_PROGRESS}" = "true" ]; then
  pipeline_cmd+=(--progress)
fi
if [ -n "${PIPELINE_OUTDIR}" ]; then
  pipeline_cmd+=(--outdir "${PIPELINE_OUTDIR}")
fi
"${pipeline_cmd[@]}"

log "Done"
