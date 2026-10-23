#!/usr/bin/env python3
"""Compare hot/cold cache behavior across two capacities and sketch sizes.

The timed pass uses the existing performance benchmark. An independent pass
reads cache statistics after every request to classify hot hits and cold fills;
these expensive statistics calls are excluded from throughput measurements.
All JSON records are saved in one /mnt/gds/results/groute_* directory.
"""

import argparse
import json
import statistics
from pathlib import Path

import cupy as cp
import kvikio
import kvikio.defaults

from bench_line_cache_performance import (
    IO_SIZE,
    LINE,
    MODES,
    execute,
    settings,
    trace_for,
)
from groute_experiment_output import run_experiment


def classify(args, mode, offsets):
    """Replay on a fresh cache without including stats calls in timed runs."""
    counts = {"hot_reads": 0, "hot_hits": 0, "cold_reads": 0,
              "cold_hits": 0, "cold_admits": 0}
    gpu = cp.empty(IO_SIZE, dtype=cp.uint8)
    with kvikio.defaults.set(settings(args, mode)):
        with kvikio.CuFile(args.file, "r") as handle:
            previous = handle.host_cache_stats()
            for index, line in enumerate(offsets):
                count = handle.raw_read(gpu, size=IO_SIZE, file_offset=line * LINE)
                if count != IO_SIZE:
                    raise RuntimeError(f"short classification read at line {line}: {count} bytes")
                current = handle.host_cache_stats()
                hit = current["hits"] - previous["hits"]
                if hit not in (0, 1):
                    raise RuntimeError(f"unexpected cache hit delta: {hit}")
                if index % 3 == 1:  # The cold line appears only once in this trace.
                    counts["cold_reads"] += 1
                    counts["cold_hits"] += hit
                    counts["cold_admits"] += int(
                        current["storage_bytes"] > previous["storage_bytes"])
                else:
                    counts["hot_reads"] += 1
                    counts["hot_hits"] += hit
                previous = current
    if counts["cold_hits"] != 0:
        raise RuntimeError("cold line was hit; check the trace and file offsets")
    return {
        "kind": "classification", "mode": mode, "cache_lines": args.cache_lines,
        "sketch_kib": args.sketch_bytes // 1024,
        "hot_hit_ratio": counts["hot_hits"] / counts["hot_reads"],
        "cold_false_admission_ratio": counts["cold_admits"] / counts["cold_reads"],
        "classification": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True,
                        help="stable SSD file covering every requested 64 KiB line")
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-lines", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--sketch-kib", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--page-cache", choices=("none", "file"), default="file")
    parser.add_argument("--buffered", action="store_true",
                        help="use buffered host I/O instead of POSIX O_DIRECT")
    args = parser.parse_args()
    if args.requests < 3 or args.repeats < 1 or any(n < 1 for n in args.cache_lines):
        parser.error("requests >= 3; repeats and cache-lines must be positive")
    if any(kib < 1 for kib in args.sketch_kib):
        parser.error("sketch-kib must be positive")
    if not args.file.is_file():
        parser.error(f"file does not exist: {args.file}")

    offsets = trace_for("hot_cold", args.requests, 256)
    required = (max(offsets) + 1) * LINE
    if args.file.stat().st_size < required:
        parser.error(f"file too small: needs at least {required} bytes")

    args.pattern = "hot_cold"
    args.working_set_lines = 256
    args.threshold = 2
    args.aging_interval = 256
    args.hit_ns = 10000
    args.fill_ns = 50000
    args.bypass_ns = 80000
    args.direct_io = not args.buffered
    capacities = tuple(args.cache_lines)

    # Rotate the policy order in each repeat. Both capacities are measured at
    # each sketch size; the 4-line runs provide an unpressured control.
    timed = {}
    for sketch_kib in args.sketch_kib:
        args.sketch_bytes = sketch_kib * 1024
        for cache_lines in capacities:
            args.cache_lines = cache_lines
            for repeat in range(args.repeats):
                for position in range(len(MODES)):
                    mode = MODES[(position + repeat) % len(MODES)]
                    result = execute(args, mode, repeat + 1, offsets)
                    result.update(kind="performance", sketch_kib=sketch_kib)
                    print(json.dumps(result), flush=True)
                    timed.setdefault((sketch_kib, cache_lines, mode), []).append(result)

    for sketch_kib in args.sketch_kib:
        args.sketch_bytes = sketch_kib * 1024
        for cache_lines in capacities:
            args.cache_lines = cache_lines
            for mode in ("cache_all", "region", "line"):
                diagnostic = classify(args, mode, offsets)
                print(json.dumps(diagnostic), flush=True)

                records = timed[(sketch_kib, cache_lines, mode)]
                median = lambda field: statistics.median(field(r) for r in records)
                print(json.dumps({
                    "kind": "summary", "mode": mode, "cache_lines": cache_lines,
                    "sketch_kib": sketch_kib, "repeats": args.repeats,
                    "direct_io": args.direct_io,
                    "median_iops": median(lambda r: r["iops"]),
                    "median_p99_us": median(lambda r: r["latency_us"]["p99"]),
                    "median_fill_bytes": median(
                        lambda r: r["host_cache_stats"]["storage_bytes"]),
                    "hot_hit_ratio": diagnostic["hot_hit_ratio"],
                    "cold_false_admission_ratio": diagnostic["cold_false_admission_ratio"],
                }), flush=True)


if __name__ == "__main__":
    run_experiment(main, "hotcold_grid")
