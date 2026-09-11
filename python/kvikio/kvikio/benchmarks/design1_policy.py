# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Design 1 with workload-shaped reads and report the selected policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cupy

import kvikio


CASES: dict[str, dict[str, Any]] = {
    "sequential_large": {
        "io_size": 128 * 1024,
        "expected": ("SEQUENTIAL_SCAN", "GPU_DIRECT", "BYPASS", "DIRECT"),
    },
    "random_cold_small": {
        "io_size": 4 * 1024,
        "expected": ("FINE_GRAINED", "HOST_MEDIATED", "ADMIT", "DIRECT"),
    },
    "random_hot_small": {
        "io_size": 4 * 1024,
        "expected": ("REUSE_DOMINATED", "HOST_MEDIATED", "ADMIT", "DIRECT"),
    },
    "adjacent_unaligned_small": {
        "io_size": 4 * 1024,
        "expected": ("FINE_GRAINED", "GPU_DIRECT", "BYPASS", "SHAPED"),
    },
}


def _offsets(case: str, count: int, io_size: int, file_size: int) -> list[int]:
    usable = file_size - io_size - 4096
    if usable <= 0:
        raise ValueError("benchmark file is too small")
    if case == "sequential_large":
        return [(i * io_size) % usable for i in range(count)]
    if case == "random_hot_small":
        hot = [0, 64 * 1024]
        return [hot[i % len(hot)] for i in range(count)]
    if case == "adjacent_unaligned_small":
        # Each pair straddles a 1 MiB profiling-region boundary. Requests in a
        # pair are mergeable, while their starting offsets remain in distinct
        # regions so the pattern is not mislabeled as reuse-dominated.
        region_size = 1024 * 1024
        pair_count = (count + 1) // 2
        offsets: list[int] = []
        for pair in range(pair_count):
            first = (2 * pair + 1) * region_size - io_size + 3
            offsets.extend((first, first + io_size))
        return [offset % usable for offset in offsets[:count]]
    # Visit distinct 1 MiB profiling regions in a deterministic permutation.
    # This makes random_cold_small cold at exactly the granularity used by
    # IOContext instead of relying on probabilistic random samples.
    region_size = 1024 * 1024
    regions = usable // region_size
    if regions < 64:
        raise ValueError("random_cold_small requires at least 64 usable MiB")
    return [((i * 37) % regions) * region_size for i in range(count)]


def _submit_batched(
    handle: kvikio.CuFile,
    offsets: list[int],
    io_size: int,
    batch_size: int,
) -> tuple[int, list[float]]:
    buffers = [cupy.empty(io_size, dtype=cupy.uint8) for _ in range(batch_size)]
    completed = 0
    batch_latency_us: list[float] = []
    for begin in range(0, len(offsets), batch_size):
        group = offsets[begin : begin + batch_size]
        start = time.perf_counter_ns()
        futures = [
            handle.pread(
                buffers[i],
                size=io_size,
                file_offset=offset,
                task_size=io_size,
            )
            for i, offset in enumerate(group)
        ]
        completed += sum(future.get() for future in futures)
        batch_latency_us.append((time.perf_counter_ns() - start) / 1000.0)
    cupy.cuda.Stream.null.synchronize()
    return completed, batch_latency_us


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = CASES[args.case]
    io_size = int(case["io_size"])
    file_size = args.file.stat().st_size
    profile_offsets = _offsets(args.case, 64, io_size, file_size)
    measured_offsets = _offsets(args.case, args.requests, io_size, file_size)

    settings = {
        "compat_mode": kvikio.CompatMode.OFF,
        "gds_threshold": 0,
        "task_size": io_size,
        "host_cache_enabled": True,
        "request_shaping_enabled": True,
        "host_cache_capacity": args.cache_bytes,
        "host_cache_line_size": 64 * 1024,
        "host_cache_max_io_size": 64 * 1024,
        "host_cache_region_size": 1024 * 1024,
        "host_cache_admission_threshold": 2,
        "host_cache_max_regions": 4096,
    }
    with kvikio.defaults.set(settings):
        with kvikio.CuFile(args.file, "r") as handle:
            _submit_batched(handle, profile_offsets, io_size, args.batch_size)
            selected = handle.io_context()
            before_cache = handle.host_cache_stats()
            before_shaping = selected["shaping"]
            start = time.perf_counter()
            completed, latencies = _submit_batched(
                handle, measured_offsets, io_size, args.batch_size
            )
            elapsed = time.perf_counter() - start
            context = handle.io_context()
            after_cache = handle.host_cache_stats()

    expected_bytes = args.requests * io_size
    if completed != expected_bytes:
        raise RuntimeError(
            f"{args.case}: completed {completed} bytes, expected {expected_bytes}"
        )

    expected = case["expected"]
    actual = (
        selected["workload"],
        selected["path"],
        selected["cache"],
        selected["submit"],
    )
    if actual != expected:
        raise RuntimeError(f"{args.case}: expected policy {expected}, selected {actual}")

    cache_delta = {
        key: after_cache[key] - before_cache.get(key, 0) for key in after_cache
    }
    shaping_delta = {
        key: context["shaping"][key] - before_shaping.get(key, 0)
        for key in context["shaping"]
    }
    ordered = sorted(latencies)
    percentile = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "case": args.case,
        "requests": args.requests,
        "io_size": io_size,
        "batch_size": args.batch_size,
        "completed_bytes": completed,
        "elapsed_seconds": elapsed,
        "iops": args.requests / elapsed,
        "logical_mib_per_second": completed / elapsed / 1024**2,
        "batch_latency_us": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
        "selected_policy": selected,
        "final_context": context,
        "host_cache_delta": cache_delta,
        "request_shaping_delta": shaping_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = run(args)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
