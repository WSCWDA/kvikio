# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Single-case benchmark for the KvikIO GDS host-cache MVP.

This benchmark intentionally measures a narrow workload:

* one existing file,
* one CuFile handle,
* one GPU buffer,
* aligned synchronous reads,
* repeated random reads from a fixed hot range.

It is meant to validate the host-cache MVP and to feed the matrix runner in
``scripts/run_gds_hcache_matrix.sh``.  It prints one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import cupy

import kvikio
import kvikio.defaults


def _parse_size(value: str) -> int:
    suffixes = {
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    value = value.strip().lower()
    for suffix, scale in suffixes.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


def _positive_aligned(name: str, value: int, alignment: int = 4096) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value % alignment != 0:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _make_offsets(
    *,
    file_size: int,
    io_size: int,
    line_size: int,
    hot_bytes: int,
    requests: int,
    seed: int,
) -> list[int]:
    hot_bytes = min(hot_bytes, file_size)
    hot_lines = hot_bytes // line_size
    if hot_lines <= 0:
        raise ValueError("hot_bytes must contain at least one cache line")

    rng = random.Random(seed)
    max_slots_per_line = line_size // io_size
    offsets: list[int] = []
    for _ in range(requests):
        line = rng.randrange(hot_lines)
        slot = rng.randrange(max_slots_per_line)
        offsets.append(line * line_size + slot * io_size)
    return offsets


def _configure(args: argparse.Namespace) -> None:
    kvikio.defaults.set("gds_threshold", 0)
    kvikio.defaults.set("compat_mode", args.path == "posix")
    kvikio.defaults.set("host_cache_enabled", args.path == "hcache")
    kvikio.defaults.set("host_cache_capacity", args.cache_bytes)
    kvikio.defaults.set("host_cache_line_size", args.line_size)
    kvikio.defaults.set("host_cache_max_io_size", args.host_max)


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    file_path = Path(args.file)
    file_size = file_path.stat().st_size

    _positive_aligned("io-size", args.io_size)
    _positive_aligned("line-size", args.line_size)
    _positive_aligned("host-max", args.host_max)
    _positive_aligned("hot-bytes", args.hot_bytes)
    if args.io_size > args.line_size:
        raise ValueError("io-size must be <= line-size")
    if args.host_max > args.line_size:
        raise ValueError("host-max must be <= line-size")
    if args.cache_bytes < args.line_size:
        raise ValueError("cache-bytes must hold at least one cache line")
    if file_size < args.line_size:
        raise ValueError("file must be at least one cache line")

    # Current MVP benchmark intentionally excludes 16 KiB cache lines because
    # smoke tests showed the implementation is only reliable from 64 KiB lines.
    if args.path == "hcache" and args.line_size < 64 * 1024:
        raise ValueError("hcache line-size must be >= 65536 for the MVP benchmark")

    _configure(args)

    offsets = _make_offsets(
        file_size=file_size,
        io_size=args.io_size,
        line_size=args.line_size,
        hot_bytes=args.hot_bytes,
        requests=args.requests,
        seed=args.seed,
    )

    buf = cupy.empty(args.io_size, dtype="uint8")
    f = kvikio.CuFile(file_path, "r")
    try:
        if args.path == "hcache" and args.warmup:
            seen_lines = sorted({offset // args.line_size for offset in offsets})
            warm_buf = cupy.empty(args.io_size, dtype="uint8")
            for line in seen_lines:
                f.pread(warm_buf, size=args.io_size, file_offset=line * args.line_size).get()
            cupy.cuda.Stream.null.synchronize()

        before = f.host_cache_stats()
        t0 = time.perf_counter()
        for offset in offsets:
            n = f.pread(buf, size=args.io_size, file_offset=offset).get()
            if n != args.io_size:
                raise RuntimeError(f"short read: expected {args.io_size}, got {n}")
        cupy.cuda.Stream.null.synchronize()
        seconds = time.perf_counter() - t0
        after = f.host_cache_stats()
    finally:
        f.close()

    stats = _stats_delta(before, after)
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    hit_rate = hits / (hits + misses) if hits + misses else None

    logical_bytes = args.requests * args.io_size
    result: dict[str, Any] = {
        "path": args.path,
        "file": str(file_path),
        "file_size": file_size,
        "io_size": args.io_size,
        "line_size": args.line_size,
        "host_max": args.host_max,
        "cache_bytes": args.cache_bytes,
        "hot_bytes": args.hot_bytes,
        "requests": args.requests,
        "seed": args.seed,
        "warmup": args.warmup,
        "seconds": seconds,
        "iops": args.requests / seconds,
        "mib_per_s": logical_bytes / seconds / 1024 / 1024,
        "avg_us": seconds * 1_000_000 / args.requests,
        "hit_rate": hit_rate,
        "stats": stats,
    }
    if misses:
        result["read_amplification_vs_request"] = stats.get("storage_bytes", 0) / (
            misses * args.io_size
        )
    else:
        result["read_amplification_vs_request"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=os.environ.get("HCACHE_BENCH_FILE"), required=False)
    parser.add_argument("--path", choices=["gds", "posix", "hcache"], required=True)
    parser.add_argument("--io-size", type=_parse_size, default=4096)
    parser.add_argument("--line-size", type=_parse_size, default=64 * 1024)
    parser.add_argument("--host-max", type=_parse_size, default=64 * 1024)
    parser.add_argument("--cache-bytes", type=_parse_size, default=1024**3)
    parser.add_argument("--hot-bytes", type=_parse_size, default=64 * 1024**2)
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.file:
        raise SystemExit("--file or HCACHE_BENCH_FILE is required")

    print(json.dumps(_run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
