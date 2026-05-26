#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Package every artifact needed to fully reproduce the
# pipeline on another machine — inputs, embeddings, AND all
# rendered outputs (plots, parquets, CSVs, JSONs). The
# recipient should be able to extract and have a working
# copy without re-running anything.
#
# Usage:
#   ./scripts/package_cache.sh                       # default output
#   ./scripts/package_cache.sh ./robusto_cache.tar.gz
#
# What's included:
#   - data/                        (raw + cleaned + audits + logs)
#   - external_embeds/             (every embedding cache — .pkl, .npz, etc.)
#   - outputs/                     (every stage's parquet, PNG, SVG, CSV, JSON
#                                  — bias_agent_order, judge summaries, all
#                                  experiment subdirs e.g. pipeline_q9q10_binary)
#   - final_questions_v3.yaml      (questions map used by the judge)
#
# What's EXCLUDED:
#   - external_embeds/old/         (legacy caches — opt out via --exclude below)
#   - data/old/                    (legacy CSVs)
#   - __pycache__/, *.pyc, venv/, .git/
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

add_if_exists "data"
add_if_exists "external_embeds"
add_if_exists "outputs"
add_if_exists "final_questions_v3.yaml"

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
  --exclude='external_embeds/old' \
  --exclude='data/old' \
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
  3. Nothing else is required — every plot and parquet is already
     in outputs/. To re-render with new code:
       python -m pipeline cosine rsa bias embed   # plots only, reuses parquets
       python -m pipeline judge                   # resumes if checkpoint present
EOF
