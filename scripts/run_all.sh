#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

ENV_EMBED="${ENV_EMBED:-vqa-embed}"
ENV_PIPELINE="${ENV_PIPELINE:-vqa-pipeline}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/r2_cleaned.csv}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
# EMBED_OUTPUT is optional — when empty, generate_embeddings.py derives a name
# from the model slug + batch size (e.g. qwen3emb4b_batch1_r2_embeddings_cache_keyed.pkl).
EMBED_OUTPUT="${EMBED_OUTPUT:-}"
# Batch size 1 keeps embeddings reproducible across runs — large models can
# vary in low-precision matmul accumulation order when batched.
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-1}"
EMBED_DEVICE="${EMBED_DEVICE:-}"
EMBED_TRUST_REMOTE_CODE="${EMBED_TRUST_REMOTE_CODE:-false}"
EMBED_NPZ="${EMBED_NPZ:-}"
EMBED_NORMALIZE="${EMBED_NORMALIZE:-false}"
EMBED_DTYPE="${EMBED_DTYPE:-}"
EMBED_PADDING_SIDE="${EMBED_PADDING_SIDE:-}"
EMBED_INSTRUCTION="${EMBED_INSTRUCTION:-}"
EMBED_MAX_SEQ_LENGTH="${EMBED_MAX_SEQ_LENGTH:-}"

PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-false}"
PIPELINE_OUTDIR="${PIPELINE_OUTDIR:-}"

# Optional toggles for partial runs:
#   RUN_PREPROCESS=true   -> also run `python -m pipeline preprocess` before embed
#   SKIP_EMBED=true       -> skip the embedding generation step
#   SKIP_PIPELINE=true    -> skip the `python -m pipeline --all` step
RUN_PREPROCESS="${RUN_PREPROCESS:-false}"
SKIP_EMBED="${SKIP_EMBED:-false}"
SKIP_PIPELINE="${SKIP_PIPELINE:-false}"

# Inputs to the optional preprocess step
HUMAN_CSV="${HUMAN_CSV:-$ROOT_DIR/data/raw/humans/answers_raw_human.csv}"
VLM_DIR="${VLM_DIR:-$ROOT_DIR/data/raw/vlms}"

log() {
  echo "[run_all] $*"
}

# Mirror the slug logic in generate_embeddings.py so that both the embed
# step and the pipeline stages point at the same .pkl when EMBED_OUTPUT
# is not provided explicitly.
derive_embed_path() {
  local model="$1"
  local batch="$2"
  local lc_model
  lc_model="$(echo "$model" | tr '[:upper:]' '[:lower:]')"
  local slug
  case "$lc_model" in
    sentence-transformers/all-mpnet-base-v2) slug="allmpnet" ;;
    sentence-transformers/all-minilm-l6-v2)  slug="miniLM" ;;
    qwen/qwen3-embedding-4b)                 slug="qwen3emb4b" ;;
    qwen/qwen3-embedding-0.6b)               slug="qwen3emb0p6b" ;;
    qwen/qwen3-embedding-8b)                 slug="qwen3emb8b" ;;
    *)
      local tail="${model##*/}"
      slug="$(echo "$tail" | tr -cd 'A-Za-z0-9')"
      ;;
  esac
  echo "${ROOT_DIR}/external_embeds/${slug}_batch${batch}_r2_embeddings_cache_keyed.pkl"
}

if [ -z "${EMBED_OUTPUT}" ]; then
  EMBED_OUTPUT="$(derive_embed_path "${EMBED_MODEL}" "${EMBED_BATCH_SIZE}")"
  log "EMBED_OUTPUT not set; derived: ${EMBED_OUTPUT}"
fi

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found at ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [ "${RUN_PREPROCESS}" = "true" ]; then
  log "Running preprocess"
  conda activate "${ENV_PIPELINE}"
  python -m pipeline preprocess \
    --human-csv "${HUMAN_CSV}" \
    --vlm-dir "${VLM_DIR}"
fi

if [ "${SKIP_EMBED}" = "true" ]; then
  log "SKIP_EMBED=true — skipping embedding generation"
else

log "Generating embeddings"
conda activate "${ENV_EMBED}"
embed_cmd=(
  python "${ROOT_DIR}/scripts/generate_embeddings.py"
  --model "${EMBED_MODEL}"
  --data "${DATA_PATH}"
  --batch-size "${EMBED_BATCH_SIZE}"
  --resume
)
if [ -n "${EMBED_OUTPUT}" ]; then
  embed_cmd+=(--output "${EMBED_OUTPUT}")
fi
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
if [ -n "${EMBED_DTYPE}" ]; then
  embed_cmd+=(--dtype "${EMBED_DTYPE}")
fi
if [ -n "${EMBED_PADDING_SIDE}" ]; then
  embed_cmd+=(--padding-side "${EMBED_PADDING_SIDE}")
fi
if [ -n "${EMBED_INSTRUCTION}" ]; then
  embed_cmd+=(--instruction "${EMBED_INSTRUCTION}")
fi
if [ -n "${EMBED_MAX_SEQ_LENGTH}" ]; then
  embed_cmd+=(--max-seq-length "${EMBED_MAX_SEQ_LENGTH}")
fi
"${embed_cmd[@]}"

fi  # end SKIP_EMBED

if [ "${SKIP_PIPELINE}" = "true" ]; then
  log "SKIP_PIPELINE=true — skipping 'python -m pipeline --all'"
else

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

fi  # end SKIP_PIPELINE

log "Done"
