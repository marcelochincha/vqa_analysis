#!/usr/bin/env bash
# Run pipeline stages across all pre-computed embedding variants.
#
# Embed-dependent stages (embed, cosine, rsa) run once per embedding config
# and write to outputs/pipeline_<slug>/.
#
# Non-embed stages (bias) run once and write to outputs/pipeline_non_embed/.
#
# Preprocess is intentionally skipped — assumes data/r2_cleaned.csv exists.
# Judge is also skipped by default (needs a live vLLM server); set RUN_JUDGE=true
# to include it, or run scripts/run_judge.sh separately.
#
# Usage:
#   bash scripts/run_multi_embed.sh
#
#   # Re-run only specific embed slugs (space-separated)
#   EMBED_SLUGS="allmpnet_batch1 qwen3emb4b_batch1" bash scripts/run_multi_embed.sh
#
#   # Skip non-embed stages (bias)
#   SKIP_NON_EMBED=true bash scripts/run_multi_embed.sh
#
#   # Also run the judge stage (requires a running vLLM server)
#   RUN_JUDGE=true bash scripts/run_multi_embed.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/r2_cleaned.csv}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-false}"

# Which embed slugs to process. Override to restrict to a subset.
EMBED_SLUGS="${EMBED_SLUGS:-allmpnet_batch1 allmpnet_batch32 qwen3emb4b_batch1}"

SKIP_NON_EMBED="${SKIP_NON_EMBED:-true}"
RUN_JUDGE="${RUN_JUDGE:-false}"

# Judge settings (only used when RUN_JUDGE=true)
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-4B}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-128}"
JUDGE_CHECKPOINT_EVERY="${JUDGE_CHECKPOINT_EVERY:-10}"
JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-3}"

log() { echo "[run_multi_embed] $*"; }

# Map slug -> pkl path
embed_pkl_for_slug() {
  local slug="$1"
  echo "${ROOT_DIR}/external_embeds/${slug}_r2_embeddings_cache_keyed.pkl"
}

# Activate venv (Windows: Scripts/activate; Linux/Mac: bin/activate)
if [ -f "${VENV_DIR}/Scripts/activate" ]; then
  source "${VENV_DIR}/Scripts/activate"
elif [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Error: venv not found at ${VENV_DIR}" >&2
  exit 1
fi

# ── Embed-dependent stages ────────────────────────────────────────────────────
for slug in ${EMBED_SLUGS}; do
  pkl="$(embed_pkl_for_slug "${slug}")"
  outdir="${ROOT_DIR}/outputs/pipeline_${slug}"

  if [ ! -f "${pkl}" ]; then
    log "WARNING: embeddings not found for slug '${slug}' at ${pkl} — skipping"
    continue
  fi

  log "=== embed / cosine / rsa  [${slug}] -> ${outdir} ==="

  pipeline_cmd=(
    python -m pipeline
    embed cosine rsa
    --data "${DATA_PATH}"
    --embeddings "${pkl}"
    --outdir "${outdir}"
  )
  [ "${PIPELINE_PROGRESS}" = "true" ] && pipeline_cmd+=(--progress)

  "${pipeline_cmd[@]}"
  log "Done: ${slug}"
done

# ── Non-embed stages ──────────────────────────────────────────────────────────
if [ "${SKIP_NON_EMBED}" = "true" ]; then
  log "SKIP_NON_EMBED=true — skipping bias (and judge)"
else
  non_embed_outdir="${ROOT_DIR}/outputs/pipeline_non_embed"

  log "=== bias -> ${non_embed_outdir} ==="
  python -m pipeline bias \
    --data "${DATA_PATH}" \
    --outdir "${non_embed_outdir}"

  judge_checkpoint="${non_embed_outdir}/judge/llm_agreement_scores.parquet"

  if [ -f "${judge_checkpoint}" ]; then
    # Checkpoint exists — resume/re-plot without needing a live vLLM server.
    # score_dataframe_async will find 0 pending rows and skip all LLM calls.
    log "=== judge (from checkpoint, no vLLM needed) -> ${non_embed_outdir} ==="
    python -m pipeline judge \
      --data "${DATA_PATH}" \
      --outdir "${non_embed_outdir}" \
      --model "${JUDGE_MODEL}" \
      --base-url "${JUDGE_BASE_URL}" \
      --api-key "${JUDGE_API_KEY}"
  elif [ "${RUN_JUDGE}" = "true" ]; then
    # No checkpoint — full run, vLLM must be up.
    log "=== judge (full run, vLLM required) -> ${non_embed_outdir} ==="
    python -m pipeline judge \
      --data "${DATA_PATH}" \
      --outdir "${non_embed_outdir}" \
      --model "${JUDGE_MODEL}" \
      --base-url "${JUDGE_BASE_URL}" \
      --api-key "${JUDGE_API_KEY}" \
      --temperature "${JUDGE_TEMPERATURE}" \
      --max-tokens "${JUDGE_MAX_TOKENS}" \
      --concurrency "${JUDGE_CONCURRENCY}" \
      --batch-size "${JUDGE_BATCH_SIZE}" \
      --checkpoint-every "${JUDGE_CHECKPOINT_EVERY}" \
      --max-retries "${JUDGE_MAX_RETRIES}"
  else
    log "No judge checkpoint found and RUN_JUDGE=false — skipping judge."
    log "  To run: start vLLM (bash scripts/bash.sh) then set RUN_JUDGE=true"
  fi
fi

log "All done."
