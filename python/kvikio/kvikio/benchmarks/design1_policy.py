# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Design 1 with workload-shaped reads and report the selected policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import cupy

import kvikio
import kvikio.defaults


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


PROFILE_REQUESTS = 64
REGION_SIZE = 1024 * 1024
CACHE_LINE_SIZE = 64 * 1024

POLICY_MODES = {
    "auto": kvikio.PolicyMode.AUTO,
    "host_direct": kvikio.PolicyMode.HOST_DIRECT,
    "host_cache": kvikio.PolicyMode.HOST_CACHE,
    "gds_direct": kvikio.PolicyMode.GDS_DIRECT,
    "gds_shaped": kvikio.PolicyMode.GDS_SHAPED,
}

FORCED_POLICIES = {
    "host_direct": ("HOST_MEDIATED", "BYPASS", "DIRECT"),
    "host_cache": ("HOST_MEDIATED", "ADMIT", "DIRECT"),
    "gds_direct": ("GPU_DIRECT", "BYPASS", "DIRECT"),
    "gds_shaped": ("GPU_DIRECT", "BYPASS", "SHAPED"),
}


def _profile_offsets(case: str, count: int, io_size: int, file_size: int) -> list[int]:
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
        pair_count = (count + 1) // 2
        offsets: list[int] = []
        for pair in range(pair_count):
            first = (2 * pair + 1) * REGION_SIZE - io_size + 3
            offsets.extend((first, first + io_size))
        return [offset % usable for offset in offsets[:count]]
    # Visit distinct 1 MiB profiling regions in a deterministic permutation.
    # This makes random_cold_small cold at exactly the granularity used by
    # IOContext instead of relying on probabilistic random samples.
    regions = usable // REGION_SIZE
    if regions < PROFILE_REQUESTS:
        raise ValueError("random_cold_small requires at least 64 usable MiB")
    return [((i * 37) % regions) * REGION_SIZE for i in range(count)]


def _measurement_offsets(
    case: str,
    count: int,
    io_size: int,
    file_size: int,
    batch_size: int,
    trace_seed: int = 20260910,
) -> list[int]:
    usable = file_size - io_size - 4096
    if usable <= 0:
        raise ValueError("benchmark working set is too small")
    rng = random.Random(trace_seed)
    if case == "sequential_large":
        span = count * io_size
        if span > usable:
            raise ValueError("sequential_large trace exceeds the working set")
        slots = max(1, (usable - span) // io_size + 1)
        base = rng.randrange(slots) * io_size
        return [base + i * io_size for i in range(count)]
    if case == "random_hot_small":
        lines = usable // CACHE_LINE_SIZE
        if lines < 2:
            raise ValueError("random_hot_small requires two cache lines")
        first = rng.randrange(lines - 1)
        hot = [first * CACHE_LINE_SIZE, (first + 1) * CACHE_LINE_SIZE]
        return [hot[i % len(hot)] for i in range(count)]
    if case == "random_cold_small":
        lines = usable // CACHE_LINE_SIZE
        if count > lines:
            raise ValueError(
                "random_cold_small needs one cache line per measured request: "
                f"requests={count}, available_lines={lines}; enlarge --file or "
                "reduce --requests"
            )
        # Sampling without replacement prevents accidental cache-line reuse.
        return [
            line * CACHE_LINE_SIZE
            for line in rng.sample(range(lines), count)
        ]
    # Build complete adjacent bursts without wrapping in the middle of a batch.
    cluster_span = batch_size * io_size
    stride = max(REGION_SIZE, cluster_span + 4096)
    max_base = file_size - cluster_span - 4096
    if max_base <= 0:
        raise ValueError("benchmark file is too small for one shaping batch")
    slots = max(1, max_base // stride)
    first_slot = rng.randrange(slots)
    offsets: list[int] = []
    for begin in range(0, count, batch_size):
        group_size = min(batch_size, count - begin)
        batch_index = begin // batch_size
        base = ((first_slot + batch_index) % slots) * stride + 3
        offsets.extend(base + i * io_size for i in range(group_size))
    return offsets


def _cached_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1])
    return -1


def _control_page_cache(path: Path, mode: str) -> dict[str, Any]:
    """Apply one explicit cache-control policy immediately before measurement."""
    before = _cached_kib()
    file_evicted = False
    global_dropped = False
    if mode in ("file", "global"):
        os.sync()
        if not hasattr(os, "posix_fadvise"):
            raise RuntimeError("file cache control requires os.posix_fadvise")
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            file_evicted = True
        finally:
            os.close(fd)
    if mode == "global":
        try:
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
            global_dropped = True
        except OSError as error:
            raise PermissionError(
                "--page-cache-mode=global requires root and writable drop_caches"
            ) from error
    return {
        "mode": mode,
        "file_evicted": file_evicted,
        "global_dropped": global_dropped,
        "cached_kib_before": before,
        "cached_kib_after": _cached_kib(),
    }


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
    file_stat = args.file.stat()
    file_size = file_stat.st_size
    working_set_bytes = args.working_set_bytes or file_size
    if working_set_bytes > file_size:
        raise ValueError(
            f"working set {working_set_bytes} exceeds file size {file_size}"
        )
    profile_offsets = _profile_offsets(
        args.case, PROFILE_REQUESTS, io_size, working_set_bytes
    )
    measured_offsets = _measurement_offsets(
        args.case,
        args.requests,
        io_size,
        working_set_bytes,
        args.batch_size,
        args.trace_seed,
    )
    trace_id = hashlib.sha256(
        ",".join(str(offset) for offset in measured_offsets).encode("ascii")
    ).hexdigest()[:16]

    settings = {
        "compat_mode": kvikio.CompatMode.OFF,
        "gds_threshold": 0,
        "task_size": io_size,
        "host_cache_enabled": args.policy in ("auto", "host_cache"),
        "request_shaping_enabled": args.policy in ("auto", "gds_shaped"),
        "policy_mode": POLICY_MODES[args.policy],
        "host_cache_capacity": args.cache_bytes,
        "host_cache_line_size": CACHE_LINE_SIZE,
        "host_cache_max_io_size": CACHE_LINE_SIZE,
        "host_cache_region_size": REGION_SIZE,
        "host_cache_admission_threshold": 2,
        "host_cache_max_regions": 4096,
    }
    with kvikio.defaults.set(settings):
        with kvikio.CuFile(args.file, "r") as handle:
            _submit_batched(handle, profile_offsets, io_size, args.batch_size)
            selected = handle.io_context()
            # Policy is retained, while cache contents and region-admission history
            # from profiling are removed before workload measurement.
            handle.clear_host_cache()
            after_reset_cache = handle.host_cache_stats()
            warmup_requests = 0
            warm_host_cache = args.case == "random_hot_small" and args.policy in (
                "auto",
                "host_cache",
            )
            if warm_host_cache:
                # Admit and populate both hot lines before starting the timer.
                # Keep this trace independent of --requests so even a short
                # smoke test has a complete warm-up phase.
                hot = measured_offsets[:2]
                warmup_offsets = [hot[0], hot[1], hot[0], hot[1]]
                _submit_batched(handle, warmup_offsets, io_size, args.batch_size)
                warmup_requests = len(warmup_offsets)
            before_cache = handle.host_cache_stats()
            warmup_admitted_regions = (
                before_cache["admitted_regions"]
                - after_reset_cache["admitted_regions"]
            )
            warmup_storage_bytes = (
                before_cache["storage_bytes"]
                - after_reset_cache["storage_bytes"]
            )
            cache_entries_before_measurement = before_cache["cache_entries"]
            before_shaping = handle.io_context()["shaping"]
            page_cache = _control_page_cache(args.file, args.page_cache_mode)
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

    expected_workload = case["expected"][0]
    expected_policy = (
        case["expected"][1:]
        if args.policy == "auto"
        else FORCED_POLICIES[args.policy]
    )
    actual = (
        selected["path"],
        selected["cache"],
        selected["submit"],
    )
    if selected["policy_mode"] != args.policy.upper():
        raise RuntimeError(
            f"FileHandle retained {selected['policy_mode']} instead of "
            f"requested mode {args.policy.upper()}"
        )
    if selected["workload"] != expected_workload or actual != expected_policy:
        raise RuntimeError(
            f"{args.case}/{args.policy}: expected workload={expected_workload}, "
            f"policy={expected_policy}; selected workload={selected['workload']}, "
            f"policy={actual}"
        )

    cache_delta = {
        key: after_cache[key] - before_cache.get(key, 0) for key in after_cache
    }
    shaping_delta = {
        key: context["shaping"][key] - before_shaping.get(key, 0)
        for key in context["shaping"]
    }
    if args.case == "random_cold_small" and (
        cache_delta.get("hits", 0) != 0
        or cache_delta.get("admitted_regions", 0) != 0
    ):
        raise RuntimeError(
            "random_cold_small unexpectedly reused or admitted cache data: "
            f"{cache_delta}"
        )
    if warm_host_cache and cache_delta.get("hits", 0) != args.requests:
        raise RuntimeError(
            "random_hot_small did not remain fully cached after warm-up: "
            f"{cache_delta}"
        )
    if warm_host_cache and (
        warmup_admitted_regions != 1
        or warmup_storage_bytes != 2 * CACHE_LINE_SIZE
        or cache_entries_before_measurement != 2
    ):
        raise RuntimeError(
            "random_hot_small warm-up did not prepare the expected two-line cache: "
            f"admitted_regions={warmup_admitted_regions}, "
            f"storage_bytes={warmup_storage_bytes}, "
            f"cache_entries={cache_entries_before_measurement}"
        )
    if not warm_host_cache and (
        warmup_admitted_regions != 0
        or warmup_storage_bytes != 0
        or cache_entries_before_measurement != 0
    ):
        raise RuntimeError(
            f"{args.case} unexpectedly retained cache state before measurement: "
            f"admitted_regions={warmup_admitted_regions}, "
            f"storage_bytes={warmup_storage_bytes}, "
            f"cache_entries={cache_entries_before_measurement}"
        )
    if (
        args.case == "adjacent_unaligned_small"
        and selected["submit"] == "SHAPED"
        and shaping_delta.get("physical_requests", args.requests) >= args.requests
    ):
        raise RuntimeError(
            "adjacent_unaligned_small was not coalesced: " f"{shaping_delta}"
        )
    ordered = sorted(latencies)
    percentile = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "case": args.case,
        "policy_mode": args.policy,
        "repeat_id": args.repeat_id,
        "execution_order": args.execution_order,
        "order_seed": args.order_seed,
        "trace_seed": args.trace_seed,
        "trace_id": trace_id,
        "file": str(args.file),
        "file_size_bytes": file_size,
        "file_allocated_bytes": file_stat.st_blocks * 512,
        "working_set_bytes": working_set_bytes,
        "page_cache": page_cache,
        "num_threads": kvikio.defaults.get("num_threads"),
        "requests": args.requests,
        "profile_requests": PROFILE_REQUESTS,
        "warmup_requests": warmup_requests,
        "warmup_admitted_regions": warmup_admitted_regions,
        "warmup_storage_bytes": warmup_storage_bytes,
        "cache_entries_before_measurement": cache_entries_before_measurement,
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
    parser.add_argument("--policy", choices=POLICY_MODES, default="auto")
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--working-set-bytes", type=int)
    parser.add_argument(
        "--page-cache-mode", choices=("none", "file", "global"), default="file"
    )
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument("--execution-order", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=20260911)
    parser.add_argument("--trace-seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.working_set_bytes is not None and args.working_set_bytes <= 0:
        parser.error("--working-set-bytes must be positive")
    result = run(args)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
