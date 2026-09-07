# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Design 2 on identical fine-grained, unaligned read requests.

The three modes differ only in the 64-request profile used to select a stable IOContext policy.
The measured request offsets and buffers are identical in every mode.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cupy
import numpy

import kvikio
import kvikio.defaults

KIB = 1024
MIB = 1024 * KIB
PROFILE_REQUESTS = 64
MEASURE_BASE = 8 * MIB + 3


def layout(
    requests: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
) -> tuple[int, int]:
    requests_per_cluster = batch_size // clusters_per_batch
    cluster_stride = max(64 * KIB, requests_per_cluster * io_size + 4096)
    waves = (requests + batch_size - 1) // batch_size
    return cluster_stride, waves


def measured_offsets(
    wave_begin: int,
    wave_count: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
) -> list[int]:
    requests_per_cluster = batch_size // clusters_per_batch
    cluster_stride, _ = layout(
        wave_count, io_size, batch_size, clusters_per_batch
    )
    wave = wave_begin // batch_size
    wave_stride = clusters_per_batch * cluster_stride
    return [
        MEASURE_BASE
        + wave * wave_stride
        + (i // requests_per_cluster) * cluster_stride
        + (i % requests_per_cluster) * io_size
        for i in range(wave_count)
    ]


def required_file_size(
    requests: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
) -> int:
    cluster_stride, waves = layout(
        requests, io_size, batch_size, clusters_per_batch
    )
    return max(
        MEASURE_BASE + waves * clusters_per_batch * cluster_stride + 8192,
        16 * MIB,
    )


def prepare_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        offset = 0
        chunk_size = MIB
        while offset < size:
            nbytes = min(chunk_size, size - offset)
            chunk = (numpy.arange(nbytes, dtype=numpy.uint64) + offset) % 251
            data = chunk.astype(numpy.uint8, copy=False).tobytes()
            written = os.pwrite(fd, data, offset)
            if written != nbytes:
                raise RuntimeError(f"short preparation write: {written} != {nbytes}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def finish(future) -> int:
    return int(future.get())


def evict_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def warm_profile(handle: kvikio.CuFile, mode: str, small_buffer, large_buffer) -> None:
    futures = []
    if mode == "direct":
        size = 64 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(handle.pread(large_buffer, size, i * size, task_size=size))
    elif mode == "host":
        size = 4 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(
                handle.pread(small_buffer, size, i * 64 * KIB, task_size=size)
            )
    else:
        size = 4 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(
                handle.pread(small_buffer, size, 3 + i * size, task_size=size)
            )
    for future in futures:
        finish(future)


def verify(path: Path, buffers, offsets: list[int], io_size: int) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        for buf, offset in zip(buffers, offsets):
            expected = os.pread(fd, io_size, offset)
            actual = cupy.asnumpy(buf).tobytes()
            if actual != expected:
                raise AssertionError(f"data mismatch at file offset {offset}")
    finally:
        os.close(fd)


def run_mode(
    path: Path,
    mode: str,
    requests: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
    verify_data: bool,
) -> dict:
    buffers = [cupy.empty(io_size, dtype=cupy.uint8) for _ in range(batch_size)]
    large_buffer = cupy.empty(64 * KIB, dtype=cupy.uint8)
    total_span = required_file_size(
        requests, io_size, batch_size, clusters_per_batch
    )

    with kvikio.defaults.set(
        {
            "compat_mode": kvikio.CompatMode.OFF,
            "gds_threshold": 0,
            "host_cache_enabled": False,
            "request_shaping_enabled": True,
        }
    ):
        with kvikio.CuFile(path, "r") as handle:
            file_size = path.stat().st_size
            if file_size < total_span:
                raise ValueError(
                    f"file is too small: need at least {total_span} bytes, got {file_size}"
                )
            warm_profile(handle, mode, buffers[0], large_buffer)
            selected = handle.io_context()
            expected = {
                "direct": ("GPU_DIRECT", "DIRECT"),
                "host": ("HOST_MEDIATED", "DIRECT"),
                "shaped": ("GPU_DIRECT", "SHAPED"),
            }[mode]
            if (selected["path"], selected["submit"]) != expected:
                raise RuntimeError(f"unexpected {mode} policy: {selected}")
            shaping_before = selected["shaping"].copy()

            evict_file_cache(path)
            cupy.cuda.runtime.deviceSynchronize()
            begin = time.perf_counter()
            completed = 0
            last_buffers = []
            last_offsets = []
            for wave_begin in range(0, requests, batch_size):
                wave_count = min(batch_size, requests - wave_begin)
                offsets = measured_offsets(
                    wave_begin,
                    wave_count,
                    io_size,
                    batch_size,
                    clusters_per_batch,
                )
                futures = [
                    handle.pread(buffers[i], io_size, offsets[i], task_size=io_size)
                    for i in range(wave_count)
                ]
                completed += sum(finish(future) for future in futures)
                last_buffers = buffers[:wave_count]
                last_offsets = offsets
            cupy.cuda.runtime.deviceSynchronize()
            elapsed = time.perf_counter() - begin
            if verify_data:
                verify(path, last_buffers, last_offsets, io_size)
            context = handle.io_context()
            shaping_total = context["shaping"]
            cumulative = {
                "logical_requests",
                "physical_requests",
                "logical_bytes",
                "submitted_bytes",
                "shaped_groups",
                "direct_fallbacks",
                "collection_batches",
            }
            context["shaping_total"] = shaping_total
            context["shaping"] = {
                key: (
                    value - shaping_before.get(key, 0)
                    if key in cumulative
                    else value
                )
                for key, value in shaping_total.items()
            }

    return {
        "mode": mode,
        "requests": requests,
        "io_size": io_size,
        "batch_size": batch_size,
        "clusters_per_batch": clusters_per_batch,
        "completed_bytes": completed,
        "elapsed_seconds": elapsed,
        "iops": requests / elapsed,
        "logical_mib_per_second": completed / MIB / elapsed,
        "context": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--io-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clusters-per-batch", type=int, default=1)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.requests < 1 or args.io_size < 1 or args.batch_size < 1:
        parser.error("--requests, --io-size, and --batch-size must be positive")
    if (
        args.clusters_per_batch < 1
        or args.clusters_per_batch > args.batch_size
        or args.batch_size % args.clusters_per_batch != 0
    ):
        parser.error("--clusters-per-batch must evenly divide --batch-size")

    required_size = required_file_size(
        args.requests,
        args.io_size,
        args.batch_size,
        args.clusters_per_batch,
    )
    if args.prepare:
        prepare_file(args.file, required_size)

    results = [
        run_mode(
            args.file,
            mode,
            args.requests,
            args.io_size,
            args.batch_size,
            args.clusters_per_batch,
            args.verify,
        )
        for mode in ("direct", "host", "shaped")
    ]
    by_mode = {result["mode"]: result for result in results}
    shaping = by_mode["shaped"]["context"]["shaping"]
    logical_requests = shaping["logical_requests"]
    payload = {
        "file": str(args.file),
        "results": results,
        "summary": {
            "shaped_vs_direct_iops": (
                by_mode["shaped"]["iops"] / by_mode["direct"]["iops"]
            ),
            "shaped_vs_host_iops": (
                by_mode["shaped"]["iops"] / by_mode["host"]["iops"]
            ),
            "physical_request_reduction": (
                0.0
                if logical_requests == 0
                else 1.0 - shaping["physical_requests"] / logical_requests
            ),
            "submitted_amplification": (
                0.0
                if shaping["logical_bytes"] == 0
                else shaping["submitted_bytes"] / shaping["logical_bytes"]
            ),
            "logical_requests_per_physical": (
                0.0
                if shaping["physical_requests"] == 0
                else shaping["logical_requests"] / shaping["physical_requests"]
            ),
        },
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
