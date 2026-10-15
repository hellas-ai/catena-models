#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: run-llama-cpp.sh owner/repository[:variant] [prompt] [llama arguments]}"
shift
token_count="${QWEN3_TOKENS:-20}"
gpu_layers="${QWEN3_GPU_LAYERS:-all}"
prompt="${1:-The capital of France is}"
if (( $# > 0 )); then
  shift
fi

exec llama-completion \
  --hf-repo "$model" \
  --prompt "$prompt" \
  --n-predict "$token_count" \
  --ctx-size 8192 \
  --temp 0 \
  --gpu-layers "$gpu_layers" \
  --fit off \
  --no-conversation \
  "$@"
