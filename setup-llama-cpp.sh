#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model_dir="${QWEN3_MODEL_DIR:-$script_dir/models/Qwen3-30B-A3B-Base}"
gguf_path="${QWEN3_GGUF:-$model_dir/model-bf16.gguf}"

if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "error: missing $model_dir/model.safetensors.index.json" >&2
  echo "download Qwen/Qwen3-30B-A3B-Base first; see README.md" >&2
  exit 1
fi

if [[ -f "$gguf_path" ]]; then
  echo "llama.cpp model already exists at $gguf_path" >&2
  exit 0
fi

if [[ -z "${LLAMA_CPP_CONVERT:-}" ]]; then
  echo "error: LLAMA_CPP_CONVERT is not set; run this inside 'nix develop'" >&2
  exit 1
fi

echo "Converting the Hugging Face checkpoint to unquantized BF16 GGUF..." >&2
python "$LLAMA_CPP_CONVERT" \
  "$model_dir" \
  --outfile "$gguf_path" \
  --outtype bf16

