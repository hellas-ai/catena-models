#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gguf_path="${1:-$script_dir/models/Qwen3-30B-A3B-Base/model-bf16.gguf}"

if [[ ! -f "$gguf_path" ]]; then
  echo "error: GGUF does not exist: $gguf_path" >&2
  exit 1
fi

echo "Before eviction:"
fincore --bytes "$gguf_path"

python_args=(
  -c
  'import os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)'
  "$gguf_path"
)

if command -v python >/dev/null 2>&1; then
  python "${python_args[@]}"
else
  nix develop "$script_dir" --command python "${python_args[@]}"
fi

echo "After eviction:"
fincore --bytes "$gguf_path"

resident_bytes="$(fincore --bytes --noheadings --output RES "$gguf_path" | tr -d '[:space:]')"
if [[ "$resident_bytes" != "0" ]]; then
  echo "warning: $resident_bytes bytes remain resident; stop processes that still map the GGUF and retry" >&2
fi
