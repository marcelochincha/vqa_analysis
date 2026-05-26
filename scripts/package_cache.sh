#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Package the inputs + cached artifacts needed to replicate
# the pipeline locally without re-running the slow stages
# (embeddings, LLM judge).
#
# Usage:
#   ./scripts/package_cache.sh                       # default output
#   ./scripts/package_cache.sh ./robusto_cache.tar.gz
#
# What's included:
#   - data/r2_cleaned.csv          (preprocess output, main pipeline input)
#   - data/r2.csv                  (raw block-2 snapshot, if present)
#   - data/raw/                    (humans CSV + VLMs JSON, source data)
#   - data/processed/              (intermediate cleaned CSVs, if present)
#   - external_embeds/*.pkl        (embedding caches at the top level only)
#   - outputs/pipeline/**/*.parquet (all stage checkpoints — judge included)
#   - final_questions_v3.yaml      (questions map used by the judge)
#
# What's EXCLUDED (regenerable or unrelated):
#   - external_embeds/old/         (legacy caches)
#   - data/old/                    (legacy CSVs)
#   - outputs/pipeline/**/*.png    (plots, regenerable)
#   - outputs/paper_frames/        (paper figure frames)
#   - venv/, __pycache__/, .git/
# =========================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT_DIR/robusto_cache_$(date +%Y%m%d_%H%M%S).tar.gz}"

cd "$ROOT_DIR"

log() {
  echo "[package_cache] $*"
}

# --------------------------------------------------------
# Collect target paths (only the ones that actually exist)
# --------------------------------------------------------
PATHS=()

add_if_exists() {
  if [ -e "$1" ]; then
    PATHS+=("$1")
    log "  + $1"
  else
    log "  - skip (missing): $1"
  fi
}

log "Selecting files to package..."

add_if_exists "data/r2_cleaned.csv"
add_if_exists "data/r2.csv"
add_if_exists "data/preprocess.log"
add_if_exists "data/preprocess_block2_audit.csv"
add_if_exists "data/raw"
add_if_exists "data/processed"
add_if_exists "final_questions_v3.yaml"

# Embedding caches at top level only (skip external_embeds/old/)
shopt -s nullglob
for pkl in external_embeds/*.pkl; do
  PATHS+=("$pkl")
  log "  + $pkl"
done
shopt -u nullglob

# All pipeline-stage parquet checkpoints (the judge one is the expensive one)
if [ -d "outputs/pipeline" ]; then
  while IFS= read -r f; do
    PATHS+=("$f")
    log "  + $f"
  done < <(find outputs/pipeline -type f -name "*.parquet")
fi

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "[package_cache] ERROR: nothing to package — none of the expected paths exist." >&2
  exit 1
fi

# --------------------------------------------------------
# Build the tarball
# --------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT")"

log "Building archive: $OUTPUT"
tar -czf "$OUTPUT" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "${PATHS[@]}"

SIZE=$(du -h "$OUTPUT" | cut -f1)
log "Done — $OUTPUT ($SIZE)"

# --------------------------------------------------------
# Print restore instructions
# --------------------------------------------------------
cat <<EOF

To replicate locally:
  1. Copy $OUTPUT to the target machine's repo root.
  2. tar -xzf $(basename "$OUTPUT")
  3. Run the pipeline; it will reuse the embeddings cache and
     judge checkpoint and skip rows already scored:
       python -m pipeline judge        # resumes from llm_agreement_scores.parquet
       python -m pipeline cosine rsa   # uses external_embeds/*.pkl
EOF
