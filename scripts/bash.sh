#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen3-4B"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-"1"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"32768"}
DTYPE=${DTYPE:-"auto"}

# Set TRUST_REMOTE_CODE=true if your model requires it.
EXTRA_ARGS=()
if [ "${TRUST_REMOTE_CODE:-false}" = "true" ]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" \
  "${EXTRA_ARGS[@]}"
