#!/usr/bin/env python3
"""Paired GPU-read performance comparison for four host-path cache policies.

All policies replay identical aligned 4 KiB reads on fresh CuFile handles. The
policy order rotates between repeats. Each output line is one JSON run.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cupy as cp
import kvikio
import kvikio.defaults


LINE = 64 * 1024
IO_SIZE = 4096
MODES = ("host_direct", "cache_all", "region", "line")


def trace_for(pattern, requests, working_set):
    if pattern == "scan":
        return list(range(requests))
    if pattern == "cyclic":
        return [i % working_set for i in range(requests)]
    if pattern == "hot_cold":
        # Two hot lines compete with one fresh cold line every three reads.
        return [i // 3 + 2 if i % 3 == 1 else i % 3 // 2 for i in range(requests)]
    # A -> B -> A; the stage labels are included in the JSON output.
    cuts = requests // 3, 2 * requests // 3
    return [i % 4 if i < cuts[0] else
            4 + (i - cuts[0]) % 4 if i < cuts[1] else
            (i - cuts[1]) % 4 for i in range(requests)]


def settings(args, mode):
    cache_mode = mode != "host_direct"
    threshold = 1 if mode == "cache_all" else args.threshold
    return {
        "groute_enabled": True,
        "compat_mode": kvikio.CompatMode.ON,
        "auto_direct_io_read": args.direct_io,
        "policy_mode": (kvikio.PolicyMode.HOST_DIRECT if not cache_mode
                        else kvikio.PolicyMode.HOST_CACHE),
        "host_cache_enabled": cache_mode,
        "request_shaping_enabled": False,
        "host_cache_line_admission": mode == "line",
        "host_cache_capacity": args.cache_lines * LINE,
        "host_cache_line_size": LINE,
        "host_cache_max_io_size": IO_SIZE,
        "host_cache_region_size": 4 * LINE,
        "host_cache_admission_threshold": threshold,
        "host_cache_sketch_bytes": args.sketch_bytes,
        "host_cache_aging_interval": args.aging_interval,
        "host_cache_hit_ns": args.hit_ns,
        "host_cache_fill_ns": args.fill_ns,
        "host_cache_host_bypass_ns": args.bypass_ns,
        "host_cache_gds_bypass_ns": args.bypass_ns,
    }


def percentile(samples, p):
    return sorted(samples)[min(len(samples) - 1, int(p * len(samples)))] / 1000


def proc_read_bytes():
    # POSIX process-level block I/O; does not cover a direct GPU DMA path.
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return None


def prepare_page_cache(path, mode):
    if mode == "file":
        if not hasattr(os, "posix_fadvise"):
            raise RuntimeError("os.posix_fadvise is unavailable on this platform")
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def execute(args, mode, repeat, offsets):
    prepare_page_cache(args.file, args.page_cache)
    gpu = cp.empty(IO_SIZE, dtype=cp.uint8)
    latencies = []
    with kvikio.defaults.set(settings(args, mode)):
        with kvikio.CuFile(args.file, "r") as handle:
            try:
                direct_fd_available = bool(handle.open_flags(o_direct=True) & os.O_DIRECT)
            except (OSError, RuntimeError):
                direct_fd_available = False
            if args.direct_io and not direct_fd_available:
                raise RuntimeError("O_DIRECT is unavailable for this file; cannot run direct I/O comparison")
            block_before = proc_read_bytes()
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            for line in offsets:
                begin = time.perf_counter_ns()
                count = handle.raw_read(gpu, size=IO_SIZE, file_offset=line * LINE)
                latencies.append(time.perf_counter_ns() - begin)
                if count != IO_SIZE:
                    raise RuntimeError(f"short read at line {line}: {count} bytes")
            elapsed = time.perf_counter_ns() - wall_start
            cpu = time.process_time_ns() - cpu_start
            block_after = proc_read_bytes()
            stats = handle.host_cache_stats()

    output = {
        "mode": mode, "pattern": args.pattern, "repeat": repeat,
        "page_cache_control": args.page_cache, "file": str(args.file),
        "host_posix_direct_io_requested": args.direct_io,
        "host_posix_direct_fd_available": direct_fd_available,
        "requests": len(offsets), "cache_lines": args.cache_lines,
        "sketch_bytes": stats["sketch_bytes"], "aging_interval": args.aging_interval,
        "model_ns": {"hit": args.hit_ns, "fill": args.fill_ns, "bypass": args.bypass_ns},
        "elapsed_ms": round(elapsed / 1e6, 3),
        "iops": round(len(offsets) * 1e9 / elapsed, 2),
        "cpu_ms": round(cpu / 1e6, 3),
        "latency_us": {"p50": round(percentile(latencies, 0.5), 3),
                       "p99": round(percentile(latencies, 0.99), 3)},
        "process_block_read_bytes": (block_after - block_before
                                     if block_before is not None and block_after is not None
                                     else None),
        "host_cache_stats": stats,
    }
    if args.pattern == "phase":
        cuts = (0, len(offsets) // 3, 2 * len(offsets) // 3, len(offsets))
        output["phase_latency_us"] = {
            name: {"p50": round(percentile(latencies[cuts[i]:cuts[i + 1]], 0.5), 3),
                   "p99": round(percentile(latencies[cuts[i]:cuts[i + 1]], 0.99), 3)}
            for i, name in enumerate(("A", "B", "return_A"))
        }
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, type=Path, help="existing, stable SSD data file")
    p.add_argument("--pattern", choices=("scan", "hot_cold", "cyclic", "phase"), required=True)
    p.add_argument("--requests", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cache-lines", type=int, default=4)
    p.add_argument("--working-set-lines", type=int, default=256)
    p.add_argument("--threshold", type=int, default=2)
    p.add_argument("--sketch-bytes", type=int, default=4096)
    p.add_argument("--aging-interval", type=int, default=256)
    p.add_argument("--hit-ns", type=int, default=10000)
    p.add_argument("--fill-ns", type=int, default=50000)
    p.add_argument("--bypass-ns", type=int, default=80000)
    p.add_argument("--page-cache", choices=("none", "file"), default="none",
                   help="file attempts POSIX_FADV_DONTNEED before each run")
    p.add_argument("--direct-io", action="store_true",
                   help="enable POSIX O_DIRECT for both host bypass and host-cache fills; not GDS")
    args = p.parse_args()
    if args.requests < 3 or args.repeats < 1 or args.cache_lines < 1 or args.working_set_lines < 1:
        p.error("requests >= 3; repeats, cache-lines and working-set-lines must be positive")
    if not 1 <= args.threshold <= 15 or args.sketch_bytes < 64 or args.sketch_bytes % 64:
        p.error("threshold must be in [1,15]; sketch-bytes a multiple of 64 >= 64")
    if min(args.aging_interval, args.hit_ns, args.fill_ns, args.bypass_ns) <= 0:
        p.error("aging and modeled costs must be positive")
    offsets = trace_for(args.pattern, args.requests, args.working_set_lines)
    if args.file.stat().st_size < (max(offsets) + 1) * LINE:
        p.error(f"file too small: needs {(max(offsets) + 1) * LINE} bytes")
    for repeat in range(args.repeats):
        for position in range(len(MODES)):
            mode = MODES[(position + repeat) % len(MODES)]
            print(json.dumps(execute(args, mode, repeat + 1, offsets)), flush=True)


if __name__ == "__main__":
    main()
