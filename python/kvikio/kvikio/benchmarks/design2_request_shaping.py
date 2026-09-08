# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate four cold-storage paths on identical unaligned read requests.

The benchmark compares buffered Host I/O, O_DIRECT Host I/O, per-request GDS,
and shaped GDS.  A large access domain can be used to keep the working
set above DRAM while preserving identical logical requests in every mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
PATTERN_PERIOD = 251
PATTERN_CHUNK_SIZE = PATTERN_PERIOD * 256 * KIB
PATTERN_MARKER_VERSION = 2
DEFAULT_MODES = ("host_buffered", "host_direct", "gds_direct", "gds_shaped")


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
    working_set_bytes: int | None = None,
) -> list[int]:
    requests_per_cluster = batch_size // clusters_per_batch
    cluster_stride, _ = layout(
        wave_count, io_size, batch_size, clusters_per_batch
    )
    wave = wave_begin // batch_size
    wave_stride = clusters_per_batch * cluster_stride
    if working_set_bytes is None:
        wave_slot = wave
    else:
        usable_bytes = working_set_bytes - MEASURE_BASE - 8192
        slots = usable_bytes // wave_stride
        total_waves = (wave_begin + wave_count + batch_size - 1) // batch_size
        if slots < total_waves or slots < 1:
            raise ValueError(
                "working set is too small for the requested batch layout"
            )
        # A deterministic affine permutation spreads waves across the complete
        # working set without turning requests inside a cluster into random I/O.
        stride = min(104729, slots - 1) if slots > 1 else 1
        while stride > 1 and math.gcd(stride, slots) != 1:
            stride -= 1
        wave_slot = (17 + wave * stride) % slots
    return [
        MEASURE_BASE
        + wave_slot * wave_stride
        + (i // requests_per_cluster) * cluster_stride
        + (i % requests_per_cluster) * io_size
        for i in range(wave_count)
    ]


def required_file_size(
    requests: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
    working_set_bytes: int | None = None,
) -> int:
    cluster_stride, waves = layout(
        requests, io_size, batch_size, clusters_per_batch
    )
    layout_size = max(
        MEASURE_BASE + waves * clusters_per_batch * cluster_stride + 8192,
        16 * MIB,
    )
    return max(layout_size, working_set_bytes or 0)


def prepare_file(path: Path, size: int) -> None:
    """Materialize a deterministic file, reusing a matching prepared file.

    Merely truncating or fallocating a 256+ GiB file would leave unwritten
    extents that filesystems can satisfy as zeros without device reads.  Every
    byte is therefore written once.  The repeated pattern is position-correct
    because ``PATTERN_CHUNK_SIZE`` is divisible by ``PATTERN_PERIOD``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + ".design2-pattern.json")
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (
            path.stat().st_size == size
            and metadata == {
                "version": PATTERN_MARKER_VERSION,
                "size": size,
                "period": PATTERN_PERIOD,
            }
        ):
            return
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        offset = 0
        pattern = bytes(range(PATTERN_PERIOD)) * (
            PATTERN_CHUNK_SIZE // PATTERN_PERIOD
        )
        next_progress = 8 * 1024 * MIB
        while offset < size:
            data = memoryview(pattern)[: min(len(pattern), size - offset)]
            while data:
                written = os.write(fd, data)
                if written <= 0:
                    raise RuntimeError("short preparation write")
                offset += written
                data = data[written:]
            if offset >= next_progress:
                print(
                    f"Prepared {offset / (1024**3):.1f}/{size / (1024**3):.1f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress += 8 * 1024 * MIB
        os.fsync(fd)
    finally:
        os.close(fd)
    marker_tmp = marker.with_suffix(marker.suffix + ".tmp")
    marker_tmp.write_text(
        json.dumps(
            {
                "version": PATTERN_MARKER_VERSION,
                "size": size,
                "period": PATTERN_PERIOD,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker)
    evict_file_cache(path)


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


def cached_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1])
    return -1


def drop_global_page_cache(path: Path) -> dict[str, int | bool]:
    """Evict this file and the global Linux page cache or fail explicitly."""
    before = cached_kib()
    os.sync()
    evict_file_cache(path)
    try:
        with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as file:
            file.write("3\n")
    except OSError as error:
        raise PermissionError(
            "cold-cache mode requires root and writable /proc/sys/vm/drop_caches"
        ) from error
    return {
        "requested": True,
        "succeeded": True,
        "cached_kib_before": before,
        "cached_kib_after": cached_kib(),
    }


def warm_profile(
    handle: kvikio.CuFile,
    mode: str,
    small_buffers: list,
    large_buffers: list,
) -> None:
    futures = []
    if mode == "gds_direct":
        size = 64 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(
                handle.pread(large_buffers[i], size, i * size, task_size=size)
            )
    elif mode in ("host_buffered", "host_direct"):
        size = 4 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(
                handle.pread(
                    small_buffers[i], size, i * 64 * KIB, task_size=size
                )
            )
    else:
        size = 4 * KIB
        for i in range(PROFILE_REQUESTS):
            futures.append(
                handle.pread(
                    small_buffers[i], size, 3 + i * size, task_size=size
                )
            )
    for future in futures:
        finish(future)


def verify_wave(buffers, offsets: list[int], io_size: int) -> None:
    """Verify a wave before its reusable GPU buffers are overwritten.

    ``prepare_file()`` stores byte ``file_offset % 251`` at every position.  Build
    the expected data from that invariant instead of reading the file through
    POSIX, which would populate the page cache and bias the Host baseline.
    """
    relative = numpy.arange(io_size, dtype=numpy.uint64)
    for buf, offset in zip(buffers, offsets):
        expected = ((relative + offset) % 251).astype(numpy.uint8)
        actual = cupy.asnumpy(buf).reshape(-1)
        if not numpy.array_equal(actual, expected):
            mismatch = int(numpy.flatnonzero(actual != expected)[0])
            raise AssertionError(
                f"data mismatch at file offset {offset + mismatch}"
            )


def latency_summary(latency_ns: list[int]) -> dict[str, float | int]:
    """Return compact host-observed, end-to-end logical request latency stats."""
    if not latency_ns:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    latency_us = numpy.asarray(latency_ns, dtype=numpy.float64) / 1_000.0
    p50, p95, p99 = numpy.percentile(latency_us, (50, 95, 99))
    return {
        "count": len(latency_ns),
        "mean": float(latency_us.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(latency_us.max()),
    }


def run_mode(
    path: Path,
    mode: str,
    requests: int,
    io_size: int,
    batch_size: int,
    clusters_per_batch: int,
    verify_data: bool,
    working_set_bytes: int | None = None,
    drop_caches: bool = False,
) -> dict:
    buffers = [cupy.empty(io_size, dtype=cupy.uint8) for _ in range(batch_size)]
    profile_small_buffers = [
        cupy.empty(4 * KIB, dtype=cupy.uint8) for _ in range(PROFILE_REQUESTS)
    ]
    profile_large_buffers = [
        cupy.empty(64 * KIB, dtype=cupy.uint8) for _ in range(PROFILE_REQUESTS)
    ]
    total_span = required_file_size(
        requests,
        io_size,
        batch_size,
        clusters_per_batch,
        working_set_bytes,
    )

    if mode not in DEFAULT_MODES:
        raise ValueError(f"unknown mode: {mode}")
    host_direct = mode == "host_direct"

    with kvikio.defaults.set(
        {
            "compat_mode": kvikio.CompatMode.OFF,
            "gds_threshold": 0,
            "host_cache_enabled": False,
            "request_shaping_enabled": True,
            "auto_direct_io_read": host_direct,
            "auto_direct_io_read_overread": host_direct,
        }
    ):
        print(f"BEGIN mode={mode}", file=sys.stderr, flush=True)
        with kvikio.CuFile(path, "r") as handle:
            file_size = path.stat().st_size
            if file_size < total_span:
                raise ValueError(
                    f"file is too small: need at least {total_span} bytes, got {file_size}"
                )
            direct_fd_flags = None
            if mode == "host_direct":
                direct_fd_flags = handle.open_flags(True)
                if not direct_fd_flags & os.O_DIRECT:
                    raise RuntimeError(
                        "host_direct requested, but KvikIO did not open an O_DIRECT fd"
                    )
            warm_profile(
                handle, mode, profile_small_buffers, profile_large_buffers
            )
            selected = handle.io_context()
            expected = {
                "gds_direct": ("GPU_DIRECT", "DIRECT"),
                "host_buffered": ("HOST_MEDIATED", "DIRECT"),
                "host_direct": ("HOST_MEDIATED", "DIRECT"),
                "gds_shaped": ("GPU_DIRECT", "SHAPED"),
            }[mode]
            if (selected["path"], selected["submit"]) != expected:
                raise RuntimeError(f"unexpected {mode} policy: {selected}")
            shaping_before = selected["shaping"].copy()

            cache_drop = (
                drop_global_page_cache(path)
                if drop_caches
                else {"requested": False, "succeeded": False}
            )
            evict_file_cache(path)
            cupy.cuda.runtime.deviceSynchronize()
            begin = time.perf_counter()
            completed = 0
            verified_requests = 0
            verified_bytes = 0
            verification_seconds = 0.0
            latency_ns = []
            for wave_begin in range(0, requests, batch_size):
                wave_count = min(batch_size, requests - wave_begin)
                offsets = measured_offsets(
                    wave_begin,
                    wave_count,
                    io_size,
                    batch_size,
                    clusters_per_batch,
                    working_set_bytes,
                )
                futures = []
                submitted_ns = []
                for i in range(wave_count):
                    submitted_ns.append(time.perf_counter_ns())
                    futures.append(
                        handle.pread(
                            buffers[i], io_size, offsets[i], task_size=io_size
                        )
                    )
                for future, request_begin_ns in zip(futures, submitted_ns):
                    completed += finish(future)
                    latency_ns.append(time.perf_counter_ns() - request_begin_ns)
            cupy.cuda.runtime.deviceSynchronize()
            elapsed = time.perf_counter() - begin
            expected_bytes = requests * io_size
            if completed != expected_bytes:
                raise RuntimeError(
                    f"short benchmark read: {completed} != {expected_bytes}"
                )
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

            # Replay the complete request stream outside the timed region.  Each
            # wave is checked before its GPU buffers are reused, so --verify now
            # covers every logical request without contaminating performance or
            # latency measurements with D2H validation work.
            if verify_data:
                verify_begin = time.perf_counter()
                for wave_begin in range(0, requests, batch_size):
                    wave_count = min(batch_size, requests - wave_begin)
                    offsets = measured_offsets(
                        wave_begin,
                        wave_count,
                        io_size,
                        batch_size,
                        clusters_per_batch,
                        working_set_bytes,
                    )
                    futures = [
                        handle.pread(
                            buffers[i], io_size, offsets[i], task_size=io_size
                        )
                        for i in range(wave_count)
                    ]
                    for future in futures:
                        finish(future)
                    verify_wave(buffers[:wave_count], offsets, io_size)
                    verified_requests += wave_count
                    verified_bytes += wave_count * io_size
                cupy.cuda.runtime.deviceSynchronize()
                verification_seconds = time.perf_counter() - verify_begin
                if verified_requests != requests:
                    raise RuntimeError(
                        "incomplete verification replay: "
                        f"{verified_requests} != {requests}"
                    )
        print(f"END mode={mode}", file=sys.stderr, flush=True)

    return {
        "mode": mode,
        "requests": requests,
        "io_size": io_size,
        "batch_size": batch_size,
        "clusters_per_batch": clusters_per_batch,
        "working_set_bytes": total_span,
        "backend": {
            "path": "HOST_MEDIATED" if mode.startswith("host_") else "GPU_DIRECT",
            "host_io": (
                "O_DIRECT_OVERREAD"
                if mode == "host_direct"
                else "BUFFERED"
                if mode == "host_buffered"
                else "NONE"
            ),
            "page_cache_cold": bool(drop_caches),
            "direct_fd_open_flags": direct_fd_flags,
        },
        "cache_drop": cache_drop,
        "completed_bytes": completed,
        "elapsed_seconds": elapsed,
        "iops": requests / elapsed,
        "logical_mib_per_second": completed / MIB / elapsed,
        "latency_us": latency_summary(latency_ns),
        "verification_mode": "full_replay" if verify_data else "disabled",
        "verified_requests": verified_requests,
        "verified_bytes": verified_bytes,
        "verification_seconds": verification_seconds,
        "context": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--io-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clusters-per-batch", type=int, default=1)
    parser.add_argument("--working-set-bytes", type=int)
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES)
    )
    parser.add_argument("--drop-caches", action="store_true")
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
        args.working_set_bytes,
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
            args.working_set_bytes,
            args.drop_caches,
        )
        for mode in args.modes
    ]
    by_mode = {result["mode"]: result for result in results}
    if set(by_mode) != set(DEFAULT_MODES):
        parser.error("--modes must contain all four modes for comparable output")
    shaping = by_mode["gds_shaped"]["context"]["shaping"]
    logical_requests = shaping["logical_requests"]
    payload = {
        "file": str(args.file),
        "working_set_bytes": required_size,
        "cache_policy": "global_drop_caches" if args.drop_caches else "file_fadvise_only",
        "num_threads": kvikio.defaults.get("num_threads"),
        "results": results,
        "summary": {
            "shaped_vs_direct_iops": (
                by_mode["gds_shaped"]["iops"] / by_mode["gds_direct"]["iops"]
            ),
            "shaped_vs_host_buffered_iops": (
                by_mode["gds_shaped"]["iops"] / by_mode["host_buffered"]["iops"]
            ),
            "shaped_vs_host_direct_iops": (
                by_mode["gds_shaped"]["iops"] / by_mode["host_direct"]["iops"]
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
