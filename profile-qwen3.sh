#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

usage() {
  cat >&2 <<'EOF'
usage: ./profile-qwen3.sh HF_MODEL [PROFILE_TOKENS]

Profile native-BF16 decode from an HF name owner/repository[:variant].
PROFILE_TOKENS defaults to 32. Environment overrides:
  QWEN3_PROFILE_TOKENS          tokens in each profiled pass
  QWEN3_PROFILE_BASELINE_TOKENS tokens in the unprofiled baseline (default 128)
  QWEN3_PROFILE_BASELINE_RUNS   baseline repetitions (default 3)
  QWEN3_PROFILE_MODULE_CACHE    generated GPU module path
  QWEN3_PROFILE_OUTPUT_DIR      output directory (default: a directory in /tmp)
EOF
}

model="${1:-}"
profile_tokens="${QWEN3_PROFILE_TOKENS:-${2:-32}}"
baseline_tokens="${QWEN3_PROFILE_BASELINE_TOKENS:-128}"
baseline_runs="${QWEN3_PROFILE_BASELINE_RUNS:-3}"

if [[ -z "$model" ]]; then
  usage
  exit 2
fi
for value_name in profile_tokens baseline_tokens baseline_runs; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done

if [[ -n "${QWEN3_PROFILE_OUTPUT_DIR:-}" ]]; then
  profile_dir="$QWEN3_PROFILE_OUTPUT_DIR"
  mkdir -p "$profile_dir"
else
  profile_dir="$(mktemp -d /tmp/catena-qwen3-profile.XXXXXX)"
fi
kernel_dir="$profile_dir/kernel"
runtime_dir="$profile_dir/runtime"
mkdir -p "$kernel_dir" "$runtime_dir"
module_cache="${QWEN3_PROFILE_MODULE_CACHE:-$profile_dir/qwen-bf16-profile-module.so}"

echo "building release runner..." >&2
cargo build --release

profiler="${QWEN3_ROCPROFV3:-$(command -v rocprofv3 || true)}"
if [[ -z "$profiler" ]]; then
  echo "rocprofv3 was not found in PATH; set QWEN3_ROCPROFV3" >&2
  exit 2
fi
profiler_real="$(readlink -f "$profiler")"
profiler_root="$(cd -- "$(dirname -- "$profiler_real")/.." && pwd)"
profiler_library_path="$profiler_root/lib"

runner=(
  target/release/catena-qwen3
  --model "$model"
  --bf16
  --bench
  --pp 0
)

echo "compiling and caching the exact BF16 GPU module..." >&2
env \
  CATENA_GPU_MODULE_SAVE="$module_cache" \
  "${runner[@]}" --tg 1 --bench-runs 1 \
  >"$profile_dir/module-build.log" 2>&1

echo "warming the module and checkpoint mapping..." >&2
env \
  CATENA_GPU_MODULE_LOAD="$module_cache" \
  "${runner[@]}" --tg 2 --bench-runs 1 \
  >"$profile_dir/warmup.log" 2>&1

echo "recording ${baseline_runs} independent tg${baseline_tokens} unprofiled baselines..." >&2
: >"$profile_dir/baseline.log"
for ((run = 1; run <= baseline_runs; ++run)); do
  echo "baseline run $run/$baseline_runs" >>"$profile_dir/baseline.log"
  env \
    CATENA_GPU_MODULE_LOAD="$module_cache" \
    "${runner[@]}" --tg "$baseline_tokens" --bench-runs 1 \
    >>"$profile_dir/baseline.log" 2>&1
done
cat "$profile_dir/baseline.log"

profile_status=0

echo "recording tg${profile_tokens} kernel trace..." >&2
env LD_LIBRARY_PATH="$profiler_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$profiler" \
  --kernel-trace \
  --output-format csv \
  --minimum-output-data 0 \
  --stats \
  --summary \
  --summary-per-domain \
  --summary-output-file stdout \
  --summary-units msec \
  --output-directory "$kernel_dir" \
  --output-file catena \
  -- \
  env CATENA_GPU_MODULE_LOAD="$module_cache" \
  "${runner[@]}" --tg "$profile_tokens" --bench-runs 1 \
  >"$kernel_dir/rocprofv3.log" 2>&1 || profile_status="$?"

echo "recording tg${profile_tokens} HIP runtime/allocation trace..." >&2
runtime_status=0
env LD_LIBRARY_PATH="$profiler_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$profiler" \
  --hip-runtime-trace \
  --memory-allocation-trace \
  --output-format csv \
  --minimum-output-data 0 \
  --stats \
  --summary \
  --summary-per-domain \
  --summary-output-file stdout \
  --summary-units msec \
  --output-directory "$runtime_dir" \
  --output-file catena \
  -- \
  env CATENA_GPU_MODULE_LOAD="$module_cache" \
  "${runner[@]}" --tg "$profile_tokens" --bench-runs 1 \
  >"$runtime_dir/rocprofv3.log" 2>&1 || runtime_status="$?"

if [[ "$profile_status" -eq 0 && "$runtime_status" -eq 0 ]]; then
  python3 "$script_dir/summarize-profile.py" "$profile_dir" "$profile_tokens" \
    | tee "$profile_dir/grouped-summary.txt"
else
  echo "profiling failed; kernel status=$profile_status runtime status=$runtime_status" >&2
  cat "$kernel_dir/rocprofv3.log" >&2
  cat "$runtime_dir/rocprofv3.log" >&2
fi

echo "profile output: $profile_dir" >&2
if [[ "$profile_status" -ne 0 ]]; then
  exit "$profile_status"
fi
exit "$runtime_status"
