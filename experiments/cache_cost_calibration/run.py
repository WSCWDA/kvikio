#!/usr/bin/env python3
"""Measure end-to-end HostCache hit/fill and host/GDS bypass costs."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# HostCache reads this environment variable when it is constructed.
os.environ.setdefault("KVIKIO_HOST_CACHE_PROFILE", "1")

import cupy as cp
import kvikio
import kvikio.defaults

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from groute_experiment_output import run_experiment


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def stats_delta(before, after):
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in set(before) | set(after)
        if key not in ("cache_entries", "sketch_bytes")
    }


def discard_file_pages(path):
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("os.posix_fadvise is required")
    with path.open("rb", buffering=0) as source:
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def common_settings(args):
    return {
        "groute_enabled": True,
        "request_shaping_enabled": False,
        "auto_direct_io_read": args.direct_io,
        "host_cache_capacity": args.cache_lines * args.line_size,
        "host_cache_line_size": args.line_size,
        "host_cache_max_io_size": args.io_size,
        "host_cache_region_size": 4 * args.line_size,
        "host_cache_admission_threshold": 1,
        "host_cache_line_admission": False,
    }


def mode_settings(args, mode):
    config = common_settings(args)
    if mode in ("hit", "fill"):
        config.update(
            compat_mode=kvikio.CompatMode.ON,
            policy_mode=kvikio.PolicyMode.HOST_CACHE,
            host_cache_enabled=True,
        )
    elif mode == "host_bypass":
        config.update(
            compat_mode=kvikio.CompatMode.ON,
            policy_mode=kvikio.PolicyMode.HOST_DIRECT,
            host_cache_enabled=False,
        )
    elif mode == "gds_bypass":
        config.update(
            compat_mode=kvikio.CompatMode.OFF,
            policy_mode=kvikio.PolicyMode.GDS_DIRECT,
            host_cache_enabled=False,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return config


def offsets_for(args, mode):
    if mode == "hit":
        return [0] * args.hit_requests
    # Identical unique cache-line offsets pair fill and bypass measurements.
    return [line * args.line_size for line in range(args.cold_requests)]


def validate(args, mode, offsets, counters):
    if mode == "hit":
        if counters["hits"] != len(offsets) or counters["misses"] != 0:
            raise RuntimeError(f"hit isolation failed: {counters}")
        if counters["storage_bytes"] != 0:
            raise RuntimeError(f"hit run unexpectedly read storage: {counters}")
    elif mode == "fill":
        expected_storage = len(offsets) * args.line_size
        if counters["hits"] != 0 or counters["misses"] != len(offsets):
            raise RuntimeError(f"fill isolation failed: {counters}")
        if counters["storage_bytes"] != expected_storage:
            raise RuntimeError(
                f"fill storage bytes {counters['storage_bytes']} != {expected_storage}"
            )


def execute(args, mode, repeat):
    if mode != "hit" and args.drop_file_pages:
        discard_file_pages(args.file)
    offsets = offsets_for(args, mode)
    gpu = cp.empty(args.io_size, dtype=cp.uint8)
    latencies = []

    with kvikio.defaults.set(mode_settings(args, mode)):
        with kvikio.CuFile(args.file, "r") as handle:
            direct_fd = False
            try:
                direct_fd = bool(handle.open_flags(o_direct=True) & os.O_DIRECT)
            except (OSError, RuntimeError):
                pass
            if args.direct_io and mode != "gds_bypass" and not direct_fd:
                raise RuntimeError("O_DIRECT requested but unavailable")

            if mode == "hit":
                # First request fills the line; it is deliberately outside the sample.
                if handle.raw_read(gpu, size=args.io_size, file_offset=0) != args.io_size:
                    raise RuntimeError("short warm-up read")
            before = handle.host_cache_stats()
            wall_start = time.perf_counter_ns()
            for offset in offsets:
                start = time.perf_counter_ns()
                count = handle.raw_read(gpu, size=args.io_size, file_offset=offset)
                end = time.perf_counter_ns()
                if count != args.io_size:
                    raise RuntimeError(f"short read at offset {offset}: {count}")
                latencies.append(end - start)
            wall_ns = time.perf_counter_ns() - wall_start
            after = handle.host_cache_stats()

    counters = stats_delta(before, after)
    if mode in ("hit", "fill"):
        validate(args, mode, offsets, counters)
    return {
        "kind": "cache_cost_calibration",
        "status": "completed",
        "mode": mode,
        "repeat": repeat,
        "requests": len(offsets),
        "io_size": args.io_size,
        "line_size": args.line_size,
        "cache_lines": args.cache_lines,
        "direct_io": args.direct_io,
        "drop_file_pages": args.drop_file_pages,
        "o_direct_fd_available": direct_fd,
        "latency_ns": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": round(statistics.fmean(latencies), 2),
        },
        "wall_ns_per_request": round(wall_ns / len(offsets), 2),
        "host_cache_stats": counters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--io-size", type=int, default=4096)
    parser.add_argument("--line-size", type=int, default=64 * 1024)
    parser.add_argument("--cache-lines", type=int, default=4)
    parser.add_argument("--hit-requests", type=int, default=20000)
    parser.add_argument("--cold-requests", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--include-gds", action="store_true")
    parser.add_argument("--direct-io", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drop-file-pages", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if min(args.io_size, args.line_size, args.cache_lines,
           args.hit_requests, args.cold_requests, args.repeats) <= 0:
        parser.error("sizes, counts and repeats must be positive")
    if args.line_size % 4096 or args.io_size % 4096 or args.io_size > args.line_size:
        parser.error("I/O and line sizes must be 4 KiB aligned; io-size <= line-size")
    needed = args.cold_requests * args.line_size
    if not args.file.is_file() or args.file.stat().st_size < needed:
        parser.error(f"test file must contain at least {needed} bytes")

    # Initialize CUDA before any timed request.
    cp.cuda.runtime.free(0)
    cp.cuda.Stream.null.synchronize()
    modes = ["hit", "fill", "host_bypass"]
    if args.include_gds:
        modes.append("gds_bypass")
    for repeat in range(args.repeats):
        # Rotate execution order to reduce systematic thermal/order bias.
        for position in range(len(modes)):
            mode = modes[(position + repeat) % len(modes)]
            try:
                record = execute(args, mode, repeat + 1)
            except Exception as error:
                record = {
                    "kind": "cache_cost_calibration",
                    "status": "failed",
                    "mode": mode,
                    "repeat": repeat + 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    run_experiment(main, "cache_cost_calibration")
