# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Replay a BFS/PageRank page-reference trace through KvikIO or G-Route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cupy
import numpy as np

import kvikio
import kvikio.defaults


POLICIES = (
    "kvikio_threshold",
    "auto",
    "host_direct",
    "host_cache",
    "gds_direct",
    "gds_shaped",
)
POLICY_MODES = {
    "auto": kvikio.PolicyMode.AUTO,
    "host_direct": kvikio.PolicyMode.HOST_DIRECT,
    "host_cache": kvikio.PolicyMode.HOST_CACHE,
    "gds_direct": kvikio.PolicyMode.GDS_DIRECT,
    "gds_shaped": kvikio.PolicyMode.GDS_SHAPED,
}


def read_trace(trace: Path) -> tuple[np.memmap, dict[str, Any]]:
    metadata_path = trace.with_suffix(trace.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "groute-graph-trace-v1":
        raise ValueError(f"unsupported trace metadata: {metadata_path}")
    if trace.stat().st_size % 8:
        raise ValueError(f"trace size is not uint64-aligned: {trace}")
    offsets = np.memmap(trace, mode="r", dtype="<u8")
    if len(offsets) != metadata["request_count"]:
        raise ValueError("trace request count differs from its metadata")
    digest = hashlib.sha256()
    with trace.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != metadata["trace_sha256"]:
        raise ValueError("trace checksum differs from its metadata")
    return offsets, metadata


def _cache_control(path: Path, mode: str) -> dict[str, bool | str]:
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
            raise PermissionError("global cache control requires root") from error
    return {
        "mode": mode,
        "file_evicted": file_evicted,
        "global_dropped": global_dropped,
    }


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _settings(args: argparse.Namespace, page_size: int) -> dict[str, Any]:
    if args.policy == "kvikio_threshold":
        return {
            "compat_mode": kvikio.CompatMode.OFF,
            "groute_enabled": False,
            "gds_threshold": args.kvikio_threshold,
            "task_size": page_size,
            "num_threads": args.num_threads,
        }
    return {
        "compat_mode": kvikio.CompatMode.OFF,
        "groute_enabled": True,
        "gds_threshold": 0,
        "task_size": page_size,
        "num_threads": args.num_threads,
        "policy_mode": POLICY_MODES[args.policy],
        "host_cache_enabled": args.policy in ("auto", "host_cache"),
        "request_shaping_enabled": args.policy in ("auto", "gds_shaped"),
        "host_cache_capacity": args.cache_bytes,
        "host_cache_line_size": args.cache_line_size,
        "host_cache_max_io_size": args.cache_line_size,
        "host_cache_region_size": args.cache_region_size,
        "host_cache_admission_threshold": args.cache_admission_threshold,
        "host_cache_max_regions": args.cache_max_regions,
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    offsets, metadata = read_trace(args.trace)
    page_size = int(metadata["page_size"])
    data_file = Path(metadata["data_file"])
    if not data_file.is_file():
        raise FileNotFoundError(f"trace data file is missing: {data_file}")
    if len(offsets) == 0:
        raise ValueError("cannot replay an empty trace")
    data_file_size = data_file.stat().st_size
    max_offset = int(offsets.max())
    if max_offset >= data_file_size:
        raise ValueError("trace contains an offset beyond the data file")
    request_sizes = np.minimum(page_size, data_file_size - offsets)

    cache_control = _cache_control(data_file, args.page_cache_mode)
    buffers = [cupy.empty(page_size, dtype=cupy.uint8) for _ in range(args.batch_size)]
    latencies: list[float] = []
    completed = 0
    start_total = time.perf_counter()
    with kvikio.defaults.set(_settings(args, page_size)):
        with kvikio.CuFile(data_file, "r") as handle:
            start_io = time.perf_counter()
            for begin in range(0, len(offsets), args.batch_size):
                batch = offsets[begin : begin + args.batch_size]
                batch_sizes = request_sizes[begin : begin + args.batch_size]
                start_batch = time.perf_counter_ns()
                futures = [
                    handle.pread(
                        buffers[i],
                        size=int(size),
                        file_offset=int(offset),
                        task_size=page_size,
                    )
                    for i, (offset, size) in enumerate(zip(batch, batch_sizes))
                ]
                completed += sum(int(future.get()) for future in futures)
                latencies.append((time.perf_counter_ns() - start_batch) / 1000.0)
            cupy.cuda.Stream.null.synchronize()
            io_seconds = time.perf_counter() - start_io
            context = handle.io_context()
            cache = handle.host_cache_stats()
    total_seconds = time.perf_counter() - start_total

    expected = int(request_sizes.sum())
    if completed != expected:
        raise RuntimeError(f"completed {completed} bytes, expected {expected}")
    native = args.policy == "kvikio_threshold"
    if bool(context["enabled"]) == native:
        raise RuntimeError("G-Route master-switch state does not match the policy")

    return {
        "algorithm": metadata["algorithm"],
        "policy_mode": args.policy,
        "repeat_id": args.repeat_id,
        "execution_order": args.execution_order,
        "trace": str(args.trace),
        "trace_sha256": metadata["trace_sha256"],
        "trace_requests": len(offsets),
        "page_size": page_size,
        "logical_bytes": expected,
        "data_file": str(data_file),
        "data_file_bytes": data_file_size,
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "kvikio_threshold_bytes": args.kvikio_threshold if native else 0,
        "page_cache": cache_control,
        "io_seconds": io_seconds,
        "total_seconds": total_seconds,
        "iops": len(offsets) / io_seconds,
        "logical_mib_per_second": expected / io_seconds / 1024**2,
        "batch_latency_us": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "selected_policy": context,
        "host_cache": cache,
        "trace_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--kvikio-threshold", type=int, default=16 * 1024)
    parser.add_argument("--cache-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--cache-line-size", type=int, default=64 * 1024)
    parser.add_argument("--cache-region-size", type=int, default=1024 * 1024)
    parser.add_argument("--cache-admission-threshold", type=int, default=2)
    parser.add_argument("--cache-max-regions", type=int, default=8192)
    parser.add_argument(
        "--page-cache-mode", choices=("none", "file", "global"), default="file"
    )
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument("--execution-order", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("batch_size", "num_threads", "cache_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    result = replay(args)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
