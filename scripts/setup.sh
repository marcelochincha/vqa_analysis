#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  echo "Error: do not source this script. Run: bash scripts/setup.sh" >&2
  return 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

ENV_EMBED="${ENV_EMBED:-vqa-embed}"
ENV_PIPELINE="${ENV_PIPELINE:-vqa-pipeline}"
ENV_VLLM="${ENV_VLLM:-vqa-vllm}"

CUDA_VERSION="${CUDA_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.1.0}"

log() {
  echo "[setup] $*"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Please install Miniconda/Anaconda and ensure 'conda' is on PATH." >&2
  exit 1
fi

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found at ${CONDA_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda config --set auto_activate_base false >/dev/null

if [ -z "${CUDA_VERSION}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -n1 || true)"
fi

if [ -z "${TORCH_INDEX_URL}" ]; then
  case "${CUDA_VERSION}" in
    12.*)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
      ;;
    11.8*|11.7*|11.6*)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
      ;;
    *)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
      ;;
  esac
fi

log "CUDA_VERSION=${CUDA_VERSION:-unknown}"
log "TORCH_INDEX_URL=${TORCH_INDEX_URL}"

ensure_env() {
  local env_name
  env_name="$1"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    log "Conda env '${env_name}' already exists"
  else
    log "Creating conda env '${env_name}' (python=${PYTHON_VERSION})"
    conda create -y -n "${env_name}" python="${PYTHON_VERSION}" >/dev/null
  fi
}

pip_install() {
  local env_name
  env_name="$1"
  shift
  conda run -n "${env_name}" python -m pip "$@"
}

ensure_env "${ENV_EMBED}"
ensure_env "${ENV_PIPELINE}"
ensure_env "${ENV_VLLM}"

log "Installing embeddings dependencies"
pip_install "${ENV_EMBED}" install --upgrade pip
pip_install "${ENV_EMBED}" install --index-url "${TORCH_INDEX_URL}" --upgrade torch torchvision torchaudio
pip_install "${ENV_EMBED}" install sentence-transformers transformers tqdm numpy pandas

log "Installing pipeline dependencies"
if [ ! -f "${ROOT_DIR}/requirements.txt" ]; then
  echo "Error: requirements.txt not found at ${ROOT_DIR}/requirements.txt" >&2
  exit 1
fi
pipeline_tmp="$(mktemp)"
grep -v -i '^vllm' "${ROOT_DIR}/requirements.txt" > "${pipeline_tmp}"
pip_install "${ENV_PIPELINE}" install --upgrade pip
pip_install "${ENV_PIPELINE}" install -r "${pipeline_tmp}"
pip_install "${ENV_PIPELINE}" install httpx
rm -f "${pipeline_tmp}"

log "Installing vLLM dependencies"
pip_install "${ENV_VLLM}" install --upgrade pip
pip_install "${ENV_VLLM}" install --index-url "${TORCH_INDEX_URL}" --upgrade torch torchvision torchaudio
pip_install "${ENV_VLLM}" install "${VLLM_SPEC}"

log "Setup complete."
log "Activate envs with: conda activate ${ENV_EMBED} | ${ENV_PIPELINE} | ${ENV_VLLM}"
