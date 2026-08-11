#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

token_count="${QWEN3_PROFILE_TOKENS:-${2:-20}}"
prompt="${1:-The capital of France is}"
profile_dir="$(mktemp -d /tmp/catena-qwen3-profile.XXXXXX)"
profiler_log="$profile_dir/rocprofv3.log"
module_cache="${QWEN3_PROFILE_MODULE_CACHE:-$script_dir/target/qwen3-profile-module.so}"

echo "building release runner..." >&2
nix develop --command cargo build --release

profiler_root="$(
  nix build --no-link --print-out-paths nixpkgs#rocmPackages.rocprofiler-sdk \
    | tail -n 1
)"
register_root="$(
  nix build --no-link --print-out-paths nixpkgs#rocmPackages.rocprofiler-register \
    | tail -n 1
)"
profiler="$profiler_root/bin/rocprofv3"

if [[ ! -f "$module_cache" ]]; then
  echo "compiling and caching the generated GPU module..." >&2
  nix develop --command env \
    CATENA_GPU_MODULE_SAVE="$module_cache" \
    target/release/catena-qwen3 "$prompt" \
    --max-new-tokens 0 >/dev/null
else
  echo "using cached generated GPU module at $module_cache" >&2
fi

echo "profiling cached BF16 Qwen3 module..." >&2
profiler_status=0
env LD_LIBRARY_PATH="$register_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$profiler" \
  --kernel-trace \
  --output-format csv \
  --minimum-output-data 0 \
  --stats \
  --summary \
  --summary-output-file stdout \
  --summary-units msec \
  --output-directory "$profile_dir" \
  --output-file catena \
  -- \
  nix develop --command env CATENA_GPU_MODULE_LOAD="$module_cache" \
  target/release/catena-qwen3 "$prompt" \
  --max-new-tokens "$token_count" --ignore-eos \
  >"$profiler_log" 2>&1 || profiler_status="$?"

cat "$profiler_log"
echo "profile output: $profile_dir" >&2
exit "$profiler_status"
