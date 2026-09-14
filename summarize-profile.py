#!/usr/bin/env python3
"""Summarize rocprofv3 Catena decode statistics by subsystem."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "gated_delta_net" in lowered:
        return "DeltaNet"
    if "moe_down" in lowered or "routed_native_bf16_gemv_residual" in lowered:
        return "routed expert down"
    if "expert_linear_pair" in lowered or "routed_native_bf16_gemv_pair" in lowered:
        return "routed expert gate/up"
    if "shared" in lowered and ("linear" in lowered or "gemv" in lowered):
        return "shared expert GEMV"
    if "final_logits" in lowered or "argmax" in lowered:
        return "vocabulary head / argmax"
    if any(part in lowered for part in ("cached_context", "cached_staged", "cache_append")):
        return "full attention / KV cache"
    if any(part in lowered for part in ("router", "topk", "routing_weight")):
        return "MoE routing"
    if any(part in lowered for part in ("rms", "norm")):
        return "normalization"
    if any(part in lowered for part in ("linear", "gemv", "embedding")):
        return "ordinary projection GEMV"
    return "other kernels"


def read_stats(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def duration(row: dict[str, str]) -> int:
    for key in ("TotalDurationNs", "TotalDuration", "Total Duration (ns)"):
        if key in row and row[key]:
            return int(float(row[key]))
    return 0


def calls(row: dict[str, str]) -> int:
    for key in ("Calls", "Count"):
        if key in row and row[key]:
            return int(float(row[key]))
    return 0


def print_kernel_groups(profile_dir: Path) -> None:
    paths = sorted(profile_dir.glob("kernel/*kernel_stats.csv"))
    if not paths:
        print("No kernel statistics CSV was produced.")
        return
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in read_stats(paths[0]):
        group = classify_kernel(row.get("Name", ""))
        groups[group][0] += calls(row)
        groups[group][1] += duration(row)
    total = sum(item[1] for item in groups.values())
    print("Kernel groups")
    print("-------------")
    for group, (count, elapsed) in sorted(groups.items(), key=lambda item: item[1][1], reverse=True):
        share = 100.0 * elapsed / total if total else 0.0
        print(f"{group:28} {count:8d} calls {elapsed / 1e6:11.3f} ms {share:7.2f}%")
    print(f"{'total kernel time':28} {'':8}       {total / 1e6:11.3f} ms")


def print_runtime_api(profile_dir: Path, token_count: int) -> None:
    paths = sorted(profile_dir.glob("runtime/*hip*_stats.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_stats(path))
    rows = [row for row in rows if duration(row)]
    if not rows:
        print("\nNo HIP runtime statistics CSV was produced.")
        return
    rows.sort(key=duration, reverse=True)
    total = sum(duration(row) for row in rows)
    print("\nTop HIP runtime calls")
    print("---------------------")
    for row in rows[:15]:
        elapsed = duration(row)
        share = 100.0 * elapsed / total if total else 0.0
        name = row.get("Name", "unknown")
        print(f"{name[:42]:42} {calls(row):8d} calls {elapsed / 1e6:11.3f} ms {share:7.2f}%")
    print(f"{'total HIP API time':42} {'':8}       {total / 1e6:11.3f} ms")

    call_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        call_counts[row.get("Name", "")] += calls(row)
    allocations = call_counts.get("hipMallocAsync", 0)
    frees = call_counts.get("hipFreeAsync", 0)
    if allocations or frees:
        imbalance = allocations - frees
        print(
            f"async allocation/free imbalance: {imbalance:+d} "
            f"({imbalance / token_count:+.2f}/token; "
            f"{allocations} allocations, {frees} frees)"
        )
    total_allocations = allocations + call_counts.get("hipMalloc", 0)
    total_frees = frees + call_counts.get("hipFree", 0)
    if total_allocations or total_frees:
        imbalance = total_allocations - total_frees
        print(
            f"total device allocation/free imbalance: {imbalance:+d} "
            f"({imbalance / token_count:+.2f}/token; "
            f"{total_allocations} allocations, {total_frees} frees)"
        )


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: summarize-profile.py PROFILE_DIR [TOKEN_COUNT]", file=sys.stderr)
        return 2
    profile_dir = Path(sys.argv[1])
    try:
        token_count = int(sys.argv[2]) if len(sys.argv) == 3 else 1
    except ValueError:
        print("TOKEN_COUNT must be a positive integer", file=sys.stderr)
        return 2
    if token_count < 1:
        print("TOKEN_COUNT must be a positive integer", file=sys.stderr)
        return 2
    print_kernel_groups(profile_dir)
    print_runtime_api(profile_dir, token_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
