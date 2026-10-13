#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model_name="${1:?usage: setup-llama-cpp.sh owner/model output.gguf}"
gguf_path="${2:?provide the output GGUF filename}"
model_dir="$(python - "$model_name" <<'PYHF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1]))
PYHF
)"

if [[ -z "${LLAMA_CPP_CONVERT:-}" ]]; then
  echo "error: LLAMA_CPP_CONVERT is not set; run this inside 'nix develop'" >&2
  exit 1
fi

echo "Converting the Hugging Face checkpoint to unquantized BF16 GGUF..." >&2
python "$LLAMA_CPP_CONVERT" \
  "$model_dir" \
  --outfile "$gguf_path" \
  --outtype bf16

