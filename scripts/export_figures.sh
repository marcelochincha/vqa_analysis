#!/usr/bin/env bash
# Bundle all SVG and PDF figures from outputs/ (excluding olds/) into a zip.
#
# Usage:
#   bash scripts/export_figures.sh                        # writes figures_<timestamp>.zip at repo root
#   bash scripts/export_figures.sh /tmp/figures.zip       # custom output path

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ZIP="${1:-${ROOT_DIR}/figures_${TIMESTAMP}.zip}"

log() { echo "[export_figures] $*"; }

OUTPUTS_DIR="${ROOT_DIR}/outputs"
# Space-separated list of extensions to include. Default: svg only.
# Override with FORMATS="svg pdf" to include both.
FORMATS="${FORMATS:-svg}"

# SPLIT=true  → one zip per pipeline_* subdirectory (easier to share via Discord)
# SPLIT=false → single zip (default when an explicit output path is given)
SPLIT="${SPLIT:-true}"

if [ ! -d "${OUTPUTS_DIR}" ]; then
  echo "Error: outputs/ directory not found at ${OUTPUTS_DIR}" >&2
  exit 1
fi

# Build find -name expression from FORMATS
name_expr=()
for ext in ${FORMATS}; do
  name_expr+=(-o -name "*.${ext}")
done
name_expr=("${name_expr[@]:1}")

zip_dir() {
  local subdir="$1"
  local label
  label="$(basename "${subdir}")"
  local zip_path="${ROOT_DIR}/figures_${label}_${TIMESTAMP}.zip"

  mapfile -t files < <(
    find "${subdir}" -type f \( "${name_expr[@]}" \) | sort
  )

  if [ ${#files[@]} -eq 0 ]; then
    log "  no files in ${label} — skipping"
    return
  fi

  rm -f "${zip_path}"
  (
    cd "${ROOT_DIR}"
    relative_files=()
    for f in "${files[@]}"; do
      relative_files+=("${f#${ROOT_DIR}/}")
    done
    zip -9 -q "${zip_path}" "${relative_files[@]}"
  )
  local size
  size="$(du -sh "${zip_path}" | cut -f1)"
  log "  ${zip_path}  (${#files[@]} files, ${size})"
}

if [ "${SPLIT}" = "true" ] && [ $# -eq 0 ]; then
  log "Splitting into one zip per pipeline variant (set SPLIT=false for a single zip)"
  for subdir in "${OUTPUTS_DIR}"/pipeline_*/; do
    [ -d "${subdir}" ] || continue
    zip_dir "${subdir}"
  done
else
  # Single-zip mode
  mapfile -t files < <(
    find "${OUTPUTS_DIR}" -type f \( "${name_expr[@]}" \) \
      | grep -v '/olds/' \
      | sort
  )

  if [ ${#files[@]} -eq 0 ]; then
    log "No files found in ${OUTPUTS_DIR} (excluding olds/)"
    exit 0
  fi

  log "Found ${#files[@]} files — writing to ${OUT_ZIP}"
  rm -f "${OUT_ZIP}"
  (
    cd "${ROOT_DIR}"
    relative_files=()
    for f in "${files[@]}"; do
      relative_files+=("${f#${ROOT_DIR}/}")
    done
    zip -9 -q "${OUT_ZIP}" "${relative_files[@]}"
  )
  size="$(du -sh "${OUT_ZIP}" | cut -f1)"
  log "Done: ${OUT_ZIP}  (${size})"
fi
