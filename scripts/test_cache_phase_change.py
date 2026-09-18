#!/usr/bin/env python3
"""Check exact two-line LRU behavior, then compare fast/slow sketch aging."""

import argparse
import json
from pathlib import Path

import cupy as cp
import kvikio
import kvikio.defaults


LINE = 64 * 1024
SIZE = 4096


def config(capacity_lines, threshold, aging):
    return {
        "groute_enabled": True,
        "compat_mode": kvikio.CompatMode.ON,
        "policy_mode": kvikio.PolicyMode.HOST_CACHE,
        "host_cache_line_admission": True,
        "host_cache_capacity": capacity_lines * LINE,
        "host_cache_line_size": LINE,
        "host_cache_max_io_size": SIZE,
        "host_cache_region_size": 4 * LINE,
        "host_cache_admission_threshold": threshold,
        "host_cache_sketch_bytes": 4096,
        "host_cache_aging_interval": aging,
        "host_cache_hit_ns": 10000,
        "host_cache_fill_ns": 50000,
        "host_cache_host_bypass_ns": 80000,
        "host_cache_gds_bypass_ns": 80000,
    }


def diff(before, after):
    return {key: after[key] - before[key] for key in before if key != "cache_entries"}


def read_many(handle, gpu, lines):
    for line in lines:
        assert handle.raw_read(gpu, size=SIZE, file_offset=line * LINE) == SIZE


def lru_case(path):
    gpu = cp.empty(SIZE, dtype=cp.uint8)
    with kvikio.defaults.set(config(2, 1, 256)):
        with kvikio.CuFile(path, "r") as handle:
            before = handle.host_cache_stats()
            read_many(handle, gpu, [0, 1, 0, 2, 1])
            after = handle.host_cache_stats()
    change = diff(before, after)
    result = {"experiment": "lru", "stats": change,
              "cache_entries": after["cache_entries"]}
    print(json.dumps(result))
    assert (change["hits"], change["misses"], change["evictions"],
            after["cache_entries"]) == (1, 4, 2, 2), result


def phase_case(path, aging):
    gpu = cp.empty(SIZE, dtype=cp.uint8)
    # Use separate handle/sketch for each aging setting.
    with kvikio.defaults.set(config(4, 2, aging)):
        with kvikio.CuFile(path, "r") as handle:
            stages = (
                ("phase_a", [i % 4 for i in range(512)]),
                ("phase_b", [4 + i % 4 for i in range(2048)]),
                ("return_a_first_8", [i % 4 for i in range(8)]),
                ("return_a_rest", [i % 4 for i in range(504)]),
            )
            for name, lines in stages:
                before = handle.host_cache_stats()
                read_many(handle, gpu, lines)
                after = handle.host_cache_stats()
                change = diff(before, after)
                print(json.dumps({
                    "experiment": "phase_change", "aging_interval": aging,
                    "stage": name, "requests": len(lines),
                    "hit_ratio": round(change["hits"] / len(lines), 4),
                    "cache_entries": after["cache_entries"], "stats": change,
                }))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, type=Path)
    args = p.parse_args()
    if args.file.stat().st_size < 8 * LINE:
        p.error("test file must contain at least eight 64 KiB lines")
    lru_case(args.file)
    for aging in (4, 256):
        phase_case(args.file, aging)


if __name__ == "__main__":
    main()
