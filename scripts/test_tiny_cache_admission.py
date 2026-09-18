#!/usr/bin/env python3
"""Compare cache-all, line admission, and old region admission in two lines."""

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import kvikio
import kvikio.defaults


LINE = 64 * 1024
SIZE = 4096


def config(mode, sketch_bytes):
    return {
        "groute_enabled": True,
        "compat_mode": kvikio.CompatMode.ON,
        "policy_mode": kvikio.PolicyMode.HOST_CACHE,
        "host_cache_line_admission": mode != "region",
        "host_cache_capacity": 2 * LINE,
        "host_cache_line_size": LINE,
        "host_cache_max_io_size": SIZE,
        "host_cache_region_size": 4 * LINE,
        "host_cache_admission_threshold": 1 if mode == "cache_all" else 2,
        "host_cache_sketch_bytes": sketch_bytes,
        "host_cache_aging_interval": 256,
        "host_cache_hit_ns": 10000,
        "host_cache_fill_ns": 50000,
        "host_cache_host_bypass_ns": 80000,
        "host_cache_gds_bypass_ns": 80000,
    }


def run(path, mode, rounds, sketch_bytes):
    gpu = cp.empty(SIZE, dtype=cp.uint8)
    counters = {"hot_reads": 0, "hot_hits": 0, "cold_reads": 0,
                "cold_admitted": 0, "cold_hits": 0}
    with kvikio.defaults.set(config(mode, sketch_bytes)):
        with kvikio.CuFile(path, "r") as handle:
            begin = handle.host_cache_stats()
            previous = begin
            for i in range(rounds):
                for line, kind in ((0, "hot"), (i + 2, "cold"), (1, "hot")):
                    assert handle.raw_read(gpu, size=SIZE, file_offset=line * LINE) == SIZE
                    current = handle.host_cache_stats()
                    hit = current["hits"] - previous["hits"]
                    miss = current["misses"] - previous["misses"]
                    assert (hit, miss) in ((1, 0), (0, 1)), (mode, line, previous, current)
                    counters[f"{kind}_reads"] += 1
                    counters[f"{kind}_hits"] += hit
                    if kind == "cold":
                        counters["cold_admitted"] += (
                            current["admitted_lines"] - previous["admitted_lines"]
                            if mode != "region" else
                            int(miss == 1 and current["storage_bytes"] > previous["storage_bytes"])
                        )
                    previous = current
            end = handle.host_cache_stats()
    # A fresh handle repeats the same trace without per-request stats snapshots.
    with kvikio.defaults.set(config(mode, sketch_bytes)):
        with kvikio.CuFile(path, "r") as handle:
            start = time.perf_counter_ns()
            for i in range(rounds):
                for line in (0, i + 2, 1):
                    assert handle.raw_read(gpu, size=SIZE, file_offset=line * LINE) == SIZE
            elapsed_ns = time.perf_counter_ns() - start
            with path.open("rb") as check:
                check.seek(LINE)
                assert cp.asnumpy(gpu[:64]).tobytes() == check.read(64)
    diff = {k: end[k] - begin[k] for k in begin if k != "cache_entries"}
    return {
        "experiment": "tiny_cache", "mode": mode, "rounds": rounds,
        "sketch_bytes": sketch_bytes if mode != "region" else 0,
        "elapsed_ms": round(elapsed_ns / 1e6, 3),
        "iops": round(3 * rounds * 1e9 / elapsed_ns, 2),
        "hot_hit_ratio": round(counters["hot_hits"] / counters["hot_reads"], 4),
        "cold_admission_ratio": round(counters["cold_admitted"] / counters["cold_reads"], 4),
        "cache_entries": end["cache_entries"], "classification": counters, "stats": diff,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=64)
    p.add_argument("--sketch-bytes", type=int, default=4096)
    args = p.parse_args()
    if args.rounds < 2 or args.sketch_bytes < 64 or args.sketch_bytes % 64:
        p.error("rounds must be >=2; sketch-bytes must be a positive multiple of 64")
    if args.file.stat().st_size < (args.rounds + 2) * LINE:
        p.error("test file too small for cold lines")
    results = [run(args.file, mode, args.rounds, args.sketch_bytes)
               for mode in ("cache_all", "line", "region")]
    for result in results:
        print(json.dumps(result))
    assert results[1]["classification"]["cold_admitted"] == 0, results[1]
    assert results[1]["hot_hit_ratio"] > results[0]["hot_hit_ratio"], results


if __name__ == "__main__":
    main()
