# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled context benchmark for G-Route.

The benchmark is derived from ``single_node_io.py`` but is designed for the
near-oracle routing evaluation. It keeps request size independent of logical
concurrency and generates deterministic low/high-reuse request streams.

Example
-------

    python -m kvikio.benchmarks.groute_context_bench \
      -d /mnt/gds \
      --request-size "512 KiB" \
      --concurrency 1 16 64 \
      --reuse low \
      --policy host_cache gds_direct threshold auto \
      --working-set "8 GiB" \
      --requests-per-run 128 \
      --nruns 1 \
      --csv /mnt/gds/results/groute-context/smoke.csv

By default, Linux page/inode/dentry caches are dropped immediately before each
policy run. This makes low-reuse comparisons start from an equivalent cold
cache state. For ``high`` reuse, caches are dropped before the run, followed by
an untimed warm-up on the same CuFile handle, and then the timed pass. Disable
this behavior with ``--no-drop-caches`` when intentionally studying naturally
warmed cache state.

The benchmark forces KvikIO compatibility mode OFF by default so that a forced
GDS policy actually exercises libcufile. Use ``--compat-mode auto`` only when
that behavior is explicitly desired.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import random
import statistics
from dataclasses import dataclass
from time import perf_counter as clock

import cupy
from dask.utils import format_bytes, parse_bytes

import kvikio
import kvikio.buffer
import kvikio.defaults
from kvikio.benchmarks.utils import parse_directory, pprint_sys_info

ALIGNMENT = 4096
POLICIES = ("host_cache", "gds_direct", "threshold", "auto")
REUSE_MODES = ("low", "high")

# Cython enum setters currently require integer/enum values, not strings.
# Keep these values synchronized with cpp/include/kvikio/defaults.hpp and
# python/kvikio/kvikio/_lib/defaults.pyx.
POLICY_MODE = {
    "auto": 0,
    "host_direct": 1,
    "host_cache": 2,
    "gds_direct": 3,
    "gds_shaped": 4,
}
COMPAT_MODE = {
    "off": 0,
    "on": 1,
    "auto": 2,
}


@dataclass(frozen=True)
class Context:
    request_size: int
    concurrency: int
    reuse: str


@dataclass
class RunResult:
    policy: str
    request_size: int
    concurrency: int
    reuse: str
    run: int
    elapsed_s: float
    logical_bytes: int
    throughput_Bps: float
    unique_offsets: int
    offset_reuse_ratio: float


def _validate_args(args) -> None:
    if args.request_size % ALIGNMENT != 0:
        raise ValueError("--request-size must be 4-KiB aligned")
    if args.working_set % ALIGNMENT != 0:
        raise ValueError("--working-set must be 4-KiB aligned")
    if args.working_set < args.request_size:
        raise ValueError("--working-set must be >= --request-size")
    if args.requests_per_run < 1:
        raise ValueError("--requests-per-run must be >= 1")
    if args.nruns < 1:
        raise ValueError("--nruns must be >= 1")
    if args.hotset_requests < 1:
        raise ValueError("--hotset-requests must be >= 1")
    if args.thread_pool_threads < 1:
        raise ValueError("--thread-pool-threads must be >= 1")
    for c in args.concurrency:
        if c < 1:
            raise ValueError("all --concurrency values must be >= 1")


def _drop_linux_caches() -> None:
    """Drop Linux page, inode, and dentry caches.

    The benchmark intentionally performs this immediately before each policy
    run so all policies begin from an equivalent cache state. Writing to
    ``/proc/sys/vm/drop_caches`` normally requires root privileges.
    """
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except PermissionError as exc:
        raise RuntimeError(
            "Unable to drop Linux caches. Run as root or use --no-drop-caches."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Failed to write /proc/sys/vm/drop_caches. "
            "Use --no-drop-caches to disable cache dropping."
        ) from exc


def _file_is_sparse(path: pathlib.Path, nbytes: int) -> bool:
    st = path.stat()
    allocated = getattr(st, "st_blocks", 0) * 512
    return st.st_size != nbytes or allocated < nbytes


def _write_dense_file(path: pathlib.Path, nbytes: int) -> None:
    """Create a physically backed file rather than a sparse/truncated file.

    A sparse file is invalid for a storage benchmark because reads from holes
    can be satisfied as zeros by the filesystem without real NVMe data reads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = min(64 * 1024 * 1024, nbytes)
    pattern = bytes([0xA5]) * chunk_size
    written = 0
    print(f"Preparing dense backing file: {path} ({format_bytes(nbytes)})")
    with open(path, "wb", buffering=0) as f:
        while written < nbytes:
            n = min(chunk_size, nbytes - written)
            f.write(pattern[:n])
            written += n
        f.flush()
        os.fsync(f.fileno())


def _ensure_file(path: pathlib.Path, nbytes: int, force_prepare: bool) -> None:
    if force_prepare or not path.exists() or _file_is_sparse(path, nbytes):
        _write_dense_file(path, nbytes)
        return
    print(f"Using existing dense backing file: {path}")


def _valid_request_offsets(working_set: int, request_size: int) -> list[int]:
    """Return non-overlapping request-sized slots.

    Using request-sized slots prevents two distinct 'low reuse' logical
    requests from accidentally sharing pages merely because their starts are
    only 4-KiB apart.
    """
    return list(range(0, working_set - request_size + 1, request_size))


def _generate_offsets(
    *,
    reuse: str,
    request_size: int,
    working_set: int,
    requests_per_run: int,
    hotset_requests: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    valid = _valid_request_offsets(working_set, request_size)
    if not valid:
        return [0] * requests_per_run

    if reuse == "low":
        offsets: list[int] = []
        while len(offsets) < requests_per_run:
            cycle = valid.copy()
            rng.shuffle(cycle)
            offsets.extend(cycle[: requests_per_run - len(offsets)])
        return offsets

    hot_count = min(hotset_requests, len(valid))
    hot = rng.sample(valid, hot_count)
    return [rng.choice(hot) for _ in range(requests_per_run)]


def _reuse_ratio(offsets: list[int]) -> float:
    if not offsets:
        return 0.0
    return 1.0 - len(set(offsets)) / len(offsets)


def _policy_config(policy: str, threshold: int, compat_mode: str) -> dict:
    """Build KvikIO settings for a benchmark policy.

    ``threshold`` disables G-Route and restores native KvikIO threshold
    dispatch. Forced Host/GDS policies enable G-Route and set PolicyMode.
    """
    common = {
        "compat_mode": COMPAT_MODE[compat_mode],
        "request_shaping_enabled": False,
    }

    if policy == "threshold":
        return {
            **common,
            "groute_enabled": False,
            "gds_threshold": threshold,
        }

    return {
        **common,
        "groute_enabled": True,
        "policy_mode": POLICY_MODE[policy],
    }


def _execute_batches(f, buffers, offsets, request_size, concurrency) -> int:
    total = 0
    for base in range(0, len(offsets), concurrency):
        batch = offsets[base : base + concurrency]
        futures = [
            f.pread(
                buffers[i],
                size=request_size,
                file_offset=off,
                task_size=request_size,
            )
            for i, off in enumerate(batch)
        ]
        total += sum(fut.get() for fut in futures)
    cupy.cuda.get_current_stream().synchronize()
    return total


def _run_read_batches(
    file_path: pathlib.Path,
    offsets: list[int],
    request_size: int,
    concurrency: int,
    pre_register_buffer: bool,
    warmup: bool,
) -> tuple[float, int]:
    """Execute reads while preserving per-FileHandle cache state.

    For high-reuse runs, the untimed warm-up and timed pass intentionally use
    the same CuFile handle. G-Route host-cache state is FileHandle-local, so
    reopening the file between warm-up and measurement would discard the state
    that the experiment is intended to measure.
    """
    buffers = [cupy.empty(request_size, dtype="uint8") for _ in range(concurrency)]
    if pre_register_buffer:
        for buf in buffers:
            kvikio.buffer.memory_register(buf)

    f = kvikio.CuFile(file_path, flags="r")
    try:
        if warmup:
            _execute_batches(f, buffers, offsets, request_size, concurrency)

        t0 = clock()
        total = _execute_batches(f, buffers, offsets, request_size, concurrency)
        elapsed = clock() - t0
    finally:
        f.close()
        if pre_register_buffer:
            for buf in buffers:
                kvikio.buffer.memory_deregister(buf)

    return elapsed, total


def _write_csv(path: pathlib.Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "request_size",
                "concurrency",
                "reuse",
                "run",
                "elapsed_s",
                "logical_bytes",
                "throughput_Bps",
                "throughput_GiBps",
                "unique_offsets",
                "offset_reuse_ratio",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.policy,
                    r.request_size,
                    r.concurrency,
                    r.reuse,
                    r.run,
                    f"{r.elapsed_s:.9f}",
                    r.logical_bytes,
                    f"{r.throughput_Bps:.3f}",
                    f"{r.throughput_Bps / (1024**3):.6f}",
                    r.unique_offsets,
                    f"{r.offset_reuse_ratio:.6f}",
                ]
            )


def _print_summary(rows: list[RunResult]) -> None:
    grouped: dict[tuple[str, int, int, str], list[float]] = {}
    for r in rows:
        key = (r.policy, r.request_size, r.concurrency, r.reuse)
        grouped.setdefault(key, []).append(r.throughput_Bps)

    print("\nSummary")
    print("=======")
    for (policy, req, conc, reuse), samples in grouped.items():
        mean = statistics.mean(samples)
        spread = ""
        if len(samples) > 1:
            std_pct = statistics.stdev(samples) / mean * 100 if mean else 0.0
            spread = f" +/- {std_pct:.2f}%"
        print(
            f"{policy:12s} req={format_bytes(req):>9s} "
            f"C={conc:>3d} reuse={reuse:>4s}  "
            f"{format_bytes(mean):>10s}/s{spread}"
        )


def main(args) -> None:
    _validate_args(args)
    cupy.cuda.set_allocator(None)
    cupy.arange(10)

    file_path = args.dir / args.file_name
    _ensure_file(file_path, args.working_set, args.prepare_file)

    print("G-Route controlled context benchmark")
    print("------------------------------------")
    pprint_sys_info()
    print("------------------------------------")
    print(f"file              | {file_path}")
    print(f"working set       | {format_bytes(args.working_set)}")
    print(f"request size      | {format_bytes(args.request_size)}")
    print(f"concurrency       | {args.concurrency}")
    print(f"reuse modes       | {args.reuse}")
    print(f"policies          | {args.policy}")
    print(f"requests/run      | {args.requests_per_run}")
    print(f"nruns             | {args.nruns}")
    print(f"thread pool       | {args.thread_pool_threads}")
    print(f"GDS threshold     | {format_bytes(args.gds_threshold)}")
    print(f"compat mode       | {args.compat_mode}")
    print(f"drop caches       | {args.drop_caches}")
    print(f"pre-reg-buf       | {args.pre_register_buffer}")
    print(f"high-reuse warmup | {args.warm_high_reuse}")
    print("====================================")

    rows: list[RunResult] = []

    for reuse in args.reuse:
        for concurrency in args.concurrency:
            context = Context(args.request_size, concurrency, reuse)

            for run_idx in range(1, args.nruns + 1):
                # Paired comparison: every policy sees the exact same request
                # stream for a given (context, run).
                seed = (
                    args.seed
                    + run_idx
                    + 1009 * concurrency
                    + (0 if reuse == "low" else 1000003)
                )
                offsets = _generate_offsets(
                    reuse=reuse,
                    request_size=context.request_size,
                    working_set=args.working_set,
                    requests_per_run=args.requests_per_run,
                    hotset_requests=args.hotset_requests,
                    seed=seed,
                )
                reuse_ratio = _reuse_ratio(offsets)

                for policy in args.policy:
                    if args.drop_caches:
                        _drop_linux_caches()

                    config = _policy_config(
                        policy, args.gds_threshold, args.compat_mode
                    )
                    config["num_threads"] = args.thread_pool_threads

                    with kvikio.defaults.set(config):
                        elapsed, logical_bytes = _run_read_batches(
                            file_path,
                            offsets,
                            context.request_size,
                            context.concurrency,
                            args.pre_register_buffer,
                            warmup=(reuse == "high" and args.warm_high_reuse),
                        )

                    expected = len(offsets) * context.request_size
                    if logical_bytes != expected:
                        raise RuntimeError(
                            f"I/O mismatch: expected {expected} bytes, got {logical_bytes}"
                        )

                    throughput = logical_bytes / elapsed
                    rows.append(
                        RunResult(
                            policy=policy,
                            request_size=context.request_size,
                            concurrency=context.concurrency,
                            reuse=context.reuse,
                            run=run_idx,
                            elapsed_s=elapsed,
                            logical_bytes=logical_bytes,
                            throughput_Bps=throughput,
                            unique_offsets=len(set(offsets)),
                            offset_reuse_ratio=reuse_ratio,
                        )
                    )
                    print(
                        f"run {run_idx:02d} | {policy:12s} | "
                        f"req={format_bytes(context.request_size):>9s} | "
                        f"C={context.concurrency:>3d} | reuse={context.reuse:>4s} | "
                        f"{throughput / (1024**3):.3f} GiB/s | "
                        f"offset-reuse={reuse_ratio:.3f}"
                    )

    _write_csv(args.csv, rows)
    _print_summary(rows)
    print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled G-Route path-selection benchmark"
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        type=parse_directory,
        help="Directory on the target NVMe filesystem.",
    )
    parser.add_argument(
        "--file-name",
        default="groute-context.dat",
        help="Backing-file name inside --dir.",
    )
    parser.add_argument(
        "--request-size",
        default="512 KiB",
        type=parse_bytes,
        help="Logical request size. Must be 4-KiB aligned.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 16, 64],
        help="Concurrent logical requests per batch.",
    )
    parser.add_argument(
        "--reuse",
        nargs="+",
        choices=REUSE_MODES,
        default=list(REUSE_MODES),
        help="Reuse modes to evaluate.",
    )
    parser.add_argument(
        "--policy",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Execution policies to compare.",
    )
    parser.add_argument(
        "--working-set",
        default="8 GiB",
        type=parse_bytes,
        help="Backing-file and request working-set size.",
    )
    parser.add_argument(
        "--requests-per-run",
        default=512,
        type=int,
        help="Logical requests executed per timed run.",
    )
    parser.add_argument(
        "--hotset-requests",
        default=32,
        type=int,
        help="Distinct request-sized offsets used by high-reuse mode.",
    )
    parser.add_argument(
        "--nruns",
        default=5,
        type=int,
        help="Timed repetitions for each context/policy.",
    )
    parser.add_argument(
        "--thread-pool-threads",
        default=64,
        type=int,
        help="KvikIO worker count; held constant while concurrency varies.",
    )
    parser.add_argument(
        "--gds-threshold",
        default="16 KiB",
        type=parse_bytes,
        help="Threshold used only by the KvikIO-threshold baseline.",
    )
    parser.add_argument(
        "--compat-mode",
        choices=tuple(COMPAT_MODE),
        default="off",
        help="KvikIO compatibility mode. Default off is required for real GDS.",
    )
    parser.add_argument(
        "--drop-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop Linux page/inode/dentry caches immediately before every "
            "policy run (default: enabled)."
        ),
    )
    parser.add_argument(
        "--pre-register-buffer",
        action="store_true",
        help="Pre-register GPU buffers with cuFile.",
    )
    parser.add_argument(
        "--warm-high-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm high-reuse offsets on the same CuFile handle before timing.",
    )
    parser.add_argument(
        "--prepare-file",
        action="store_true",
        help="Force recreation of the dense backing file before the experiment.",
    )
    parser.add_argument(
        "--seed",
        default=7,
        type=int,
        help="Base random seed for deterministic paired request streams.",
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("groute_context_results.csv"),
        help="Output CSV path.",
    )

    main(parser.parse_args())
