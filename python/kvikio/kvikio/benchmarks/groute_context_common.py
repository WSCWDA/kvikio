# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared engine for controlled G-Route microbenchmarks.

The three public benchmark entry points intentionally vary only one context
axis at a time (request size, logical concurrency, or data hotness). This
module owns the common mechanics so that all experiments use identical file
preparation, cache control, request generation, policy setup, submission,
CSV output, and statistics.
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

# Cython enum setters currently require integer/enum values rather than names.
# Keep these synchronized with cpp/include/kvikio/defaults.hpp and
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
    hotset_requests: int = 32
    label: str = ""


@dataclass
class RunResult:
    benchmark: str
    context_label: str
    policy: str
    request_size: int
    concurrency: int
    reuse: str
    hotset_requests: int
    run: int
    elapsed_s: float
    logical_bytes: int
    throughput_Bps: float
    unique_offsets: int
    offset_reuse_ratio: float


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all controlled G-Route microbenchmarks."""
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
        help="Dense backing-file name inside --dir.",
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
        default=1024,
        type=int,
        help="Logical requests executed per timed run.",
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
        help="KvikIO worker count; fixed independently of logical concurrency.",
    )
    parser.add_argument(
        "--gds-threshold",
        default="16 KiB",
        type=parse_bytes,
        help="Threshold used only by the KvikIO fixed-threshold baseline.",
    )
    parser.add_argument(
        "--compat-mode",
        choices=tuple(COMPAT_MODE),
        default="off",
        help="KvikIO compatibility mode; off forces real libcufile/GDS use.",
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
        "--drop-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop Linux page/inode/dentry caches immediately before every "
            "policy run (default: enabled)."
        ),
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
        help="Base seed for deterministic paired request streams.",
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        required=True,
        help="Output CSV path.",
    )


def _validate_common(args, contexts: list[Context]) -> None:
    if args.working_set % ALIGNMENT != 0:
        raise ValueError("--working-set must be 4-KiB aligned")
    if args.requests_per_run < 1:
        raise ValueError("--requests-per-run must be >= 1")
    if args.nruns < 1:
        raise ValueError("--nruns must be >= 1")
    if args.thread_pool_threads < 1:
        raise ValueError("--thread-pool-threads must be >= 1")

    for context in contexts:
        if context.request_size % ALIGNMENT != 0:
            raise ValueError("all request sizes must be 4-KiB aligned")
        if context.request_size > args.working_set:
            raise ValueError("request size must not exceed --working-set")
        if context.concurrency < 1:
            raise ValueError("logical concurrency must be >= 1")
        if context.reuse not in REUSE_MODES:
            raise ValueError(f"invalid reuse mode: {context.reuse}")
        if context.hotset_requests < 1:
            raise ValueError("hotset_requests must be >= 1")


def drop_linux_caches() -> None:
    """Drop Linux page, inode, and dentry caches."""
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
    """Create a physically backed file instead of a sparse file."""
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


def ensure_file(path: pathlib.Path, nbytes: int, force_prepare: bool) -> None:
    if force_prepare or not path.exists() or _file_is_sparse(path, nbytes):
        _write_dense_file(path, nbytes)
    else:
        print(f"Using existing dense backing file: {path}")


def _valid_request_offsets(working_set: int, request_size: int) -> list[int]:
    """Return non-overlapping request-sized slots."""
    return list(range(0, working_set - request_size + 1, request_size))


def generate_offsets(
    *,
    reuse: str,
    request_size: int,
    working_set: int,
    requests_per_run: int,
    hotset_requests: int,
    seed: int,
) -> list[int]:
    """Generate a deterministic low- or high-reuse logical request stream."""
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


def reuse_ratio(offsets: list[int]) -> float:
    if not offsets:
        return 0.0
    return 1.0 - len(set(offsets)) / len(offsets)


def policy_config(policy: str, threshold: int, compat_mode: str) -> dict:
    """Build KvikIO settings for a forced or automatic benchmark policy."""
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


def execute_batches(f, buffers, offsets, request_size, concurrency) -> int:
    """Submit exactly ``concurrency`` outstanding logical reads per batch."""
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


def run_read_batches(
    file_path: pathlib.Path,
    offsets: list[int],
    request_size: int,
    concurrency: int,
    pre_register_buffer: bool,
    warmup: bool,
) -> tuple[float, int]:
    """Execute one policy run; high-reuse warmup uses the same CuFile handle."""
    buffers = [cupy.empty(request_size, dtype="uint8") for _ in range(concurrency)]
    if pre_register_buffer:
        for buf in buffers:
            kvikio.buffer.memory_register(buf)

    f = kvikio.CuFile(file_path, flags="r")
    try:
        if warmup:
            execute_batches(f, buffers, offsets, request_size, concurrency)

        t0 = clock()
        total = execute_batches(f, buffers, offsets, request_size, concurrency)
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
                "benchmark",
                "context_label",
                "policy",
                "request_size",
                "concurrency",
                "reuse",
                "hotset_requests",
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
                    r.benchmark,
                    r.context_label,
                    r.policy,
                    r.request_size,
                    r.concurrency,
                    r.reuse,
                    r.hotset_requests,
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
    grouped: dict[tuple[str, int, int, str, int], list[float]] = {}
    reuse_values: dict[tuple[str, int, int, str, int], list[float]] = {}
    for r in rows:
        key = (
            r.policy,
            r.request_size,
            r.concurrency,
            r.reuse,
            r.hotset_requests,
        )
        grouped.setdefault(key, []).append(r.throughput_Bps)
        reuse_values.setdefault(key, []).append(r.offset_reuse_ratio)

    print("\nSummary")
    print("=======")
    for (policy, req, conc, reuse, hotset), samples in grouped.items():
        mean = statistics.mean(samples)
        observed_reuse = statistics.mean(reuse_values[(policy, req, conc, reuse, hotset)])
        spread = ""
        if len(samples) > 1:
            std_pct = statistics.stdev(samples) / mean * 100 if mean else 0.0
            spread = f" +/- {std_pct:.2f}%"
        hot = f" hotset={hotset:>5d}" if reuse == "high" else ""
        print(
            f"{policy:12s} req={format_bytes(req):>9s} C={conc:>3d} "
            f"reuse={observed_reuse:>5.3f}{hot}  "
            f"{format_bytes(mean):>10s}/s{spread}"
        )


def _print_header(
    benchmark_name: str,
    args,
    file_path: pathlib.Path,
    contexts: list[Context],
) -> None:
    print(f"G-Route {benchmark_name} benchmark")
    print("------------------------------------")
    # Apply the requested compatibility mode while printing diagnostics so
    # AUTO's nonzero enum value is not mistaken for compatibility mode.
    with kvikio.defaults.set("compat_mode", COMPAT_MODE[args.compat_mode]):
        pprint_sys_info()
    print("------------------------------------")
    print(f"file              | {file_path}")
    print(f"working set       | {format_bytes(args.working_set)}")
    print(f"contexts          | {len(contexts)}")
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


def run_context_benchmark(
    *,
    benchmark_name: str,
    args,
    contexts: list[Context],
) -> list[RunResult]:
    """Run a list of controlled contexts using paired traces across policies."""
    _validate_common(args, contexts)

    cupy.cuda.set_allocator(None)
    cupy.arange(10)  # initialize CUDA before timing

    file_path = args.dir / args.file_name
    ensure_file(file_path, args.working_set, args.prepare_file)
    _print_header(benchmark_name, args, file_path, contexts)

    rows: list[RunResult] = []

    for context_index, context in enumerate(contexts):
        label = context.label or f"context-{context_index:03d}"
        print(
            f"\n[{label}] req={format_bytes(context.request_size)}, "
            f"C={context.concurrency}, reuse={context.reuse}, "
            f"hotset={context.hotset_requests if context.reuse == 'high' else '-'}"
        )

        for run_idx in range(1, args.nruns + 1):
            # Same (context, run) gets the exact same logical trace for all policies.
            seed = args.seed + 1_000_003 * context_index + run_idx
            offsets = generate_offsets(
                reuse=context.reuse,
                request_size=context.request_size,
                working_set=args.working_set,
                requests_per_run=args.requests_per_run,
                hotset_requests=context.hotset_requests,
                seed=seed,
            )
            observed_reuse = reuse_ratio(offsets)

            for policy in args.policy:
                if args.drop_caches:
                    drop_linux_caches()

                config = policy_config(policy, args.gds_threshold, args.compat_mode)
                config["num_threads"] = args.thread_pool_threads

                with kvikio.defaults.set(config):
                    elapsed, logical_bytes = run_read_batches(
                        file_path=file_path,
                        offsets=offsets,
                        request_size=context.request_size,
                        concurrency=context.concurrency,
                        pre_register_buffer=args.pre_register_buffer,
                        warmup=(context.reuse == "high" and args.warm_high_reuse),
                    )

                expected = len(offsets) * context.request_size
                if logical_bytes != expected:
                    raise RuntimeError(
                        f"I/O mismatch: expected {expected} bytes, got {logical_bytes}"
                    )

                throughput = logical_bytes / elapsed
                rows.append(
                    RunResult(
                        benchmark=benchmark_name,
                        context_label=label,
                        policy=policy,
                        request_size=context.request_size,
                        concurrency=context.concurrency,
                        reuse=context.reuse,
                        hotset_requests=context.hotset_requests,
                        run=run_idx,
                        elapsed_s=elapsed,
                        logical_bytes=logical_bytes,
                        throughput_Bps=throughput,
                        unique_offsets=len(set(offsets)),
                        offset_reuse_ratio=observed_reuse,
                    )
                )
                print(
                    f"run {run_idx:02d} | {policy:12s} | "
                    f"req={format_bytes(context.request_size):>9s} | "
                    f"C={context.concurrency:>3d} | "
                    f"reuse={observed_reuse:>5.3f} | "
                    f"{throughput / (1024**3):.3f} GiB/s"
                )

    _write_csv(args.csv, rows)
    _print_summary(rows)
    print(f"\nCSV written to: {args.csv}")
    return rows
