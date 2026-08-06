#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model_dir="${QWEN3_MODEL_DIR:-$script_dir/models/Qwen3-30B-A3B-Base}"
gguf_path="${QWEN3_GGUF:-$model_dir/model-bf16.gguf}"
token_count="${QWEN3_TOKENS:-20}"
gpu_layers="${QWEN3_GPU_LAYERS:-all}"
prompt="${1:-The capital of France is}"
if (( $# > 0 )); then
  shift
fi

if [[ ! -f "$gguf_path" ]]; then
  echo "error: missing $gguf_path" >&2
  echo "run ./setup-llama-cpp.sh first" >&2
  exit 1
fi

exec llama-completion \
  --model "$gguf_path" \
  --prompt "$prompt" \
  --n-predict "$token_count" \
  --ctx-size 8192 \
  --temp 0 \
  --gpu-layers "$gpu_layers" \
  --fit off \
  --no-conversation \
  --simple-io \
  "$@"
