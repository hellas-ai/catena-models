#!/usr/bin/env bash
set -euo pipefail
model="${1:?usage: evict-gguf-cache.sh owner/repo[:variant]}"
python - "$model" <<'PY'
import os,sys
from pathlib import Path
from huggingface_hub import snapshot_download
selector=sys.argv[1]
repo,separator,variant=selector.partition(':')
root=Path(snapshot_download(repo_id=repo,local_files_only=True))
files=[p for p in root.rglob('*.gguf') if not p.name.lower().startswith('mmproj') and (not separator or variant.lower() in str(p.relative_to(root)).lower())]
if not files:raise SystemExit('No matching GGUF files in the HF cache')
for path in files:
    fd=os.open(path,os.O_RDONLY)
    try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    print(path)
PY
