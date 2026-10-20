# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled benchmark for G-Route path-selection evaluation.

This benchmark is derived from ``single_node_io.py`` but decouples request size
from concurrency and adds controlled reuse/working-set generation. It is aimed
at evaluating the near-oracle routing claim under representative I/O contexts.

Example:

    python -m kvikio.benchmarks.groute_context_bench \
      -d /mnt/gds/groute-context \
      --request-size "512 KiB" \
      --concurrency 1 16 64 \
      --reuse low high \
      --policy host_cache gds_direct threshold auto \
      --working-set "8 GiB" \
      --requests-per-run 512 \
      --nruns 5 \
      --csv /mnt/gds/results/groute-context/results.csv

Notes
-----
* Reads are 4-KiB aligned.
* Request size and concurrency are independent variables.
* ``low`` reuse generates non-repeating offsets within a run whenever the
  working set is large enough.
* ``high`` reuse repeatedly draws from a configurable hot set. A warm-up pass
  is executed before timed high-reuse runs so the host-cache path can exploit
  reuse naturally.
* The benchmark does not drop Linux caches itself. For cold-cache experiments,
  drop caches externally before each process invocation, e.g.::

      sync; echo 3 > /proc/sys/vm/drop_caches

  This keeps cache-control policy explicit and avoids timing perturbation from
  in-process cache flushing.
"""

from __future__ import annotations

import argparse
import csv
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


def _align_down(value: int, alignment: int = ALIGNMENT) -> int:
    return value - value % alignment


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
    for c in args.concurrency:
        if c < 1:
            raise ValueError("all --concurrency values must be >= 1")


def _ensure_file(path: pathlib.Path, nbytes: int) -> None:
    """Create or resize a backing file without including setup in timed runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == nbytes:
        return
    with open(path, "wb") as f:
        f.truncate(nbytes)


def _all_valid_offsets(working_set: int, request_size: int) -> range:
    last = working_set - request_size
    return range(0, last + 1, ALIGNMENT)


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
    valid = _all_valid_offsets(working_set, request_size)
    nslots = len(valid)
    if nslots == 0:
        return [0] * requests_per_run

    if reuse == "low":
        # Prefer non-repeating offsets. If the run contains more requests than
        # available aligned slots, shuffle cycles rather than introducing a
        # strong hot set.
        offsets: list[int] = []
        while len(offsets) < requests_per_run:
            cycle = list(valid)
            rng.shuffle(cycle)
            offsets.extend(cycle[: requests_per_run - len(offsets)])
        return offsets

    hot_count = min(hotset_requests, nslots)
    hot = rng.sample(list(valid), hot_count)
    return [rng.choice(hot) for _ in range(requests_per_run)]


def _reuse_ratio(offsets: list[int]) -> float:
    if not offsets:
        return 0.0
    return 1.0 - len(set(offsets)) / len(offsets)


def _policy_config(policy: str, threshold: int) -> dict:
    """Return KvikIO settings for one benchmark policy.

    ``threshold`` intentionally disables G-Route and restores native KvikIO
    size-threshold dispatch. Forced policies keep G-Route enabled because
    PolicyMode is implemented by the G-Route path selector.
    """
    common = {
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
        "policy_mode": policy,
    }


def _run_read_batches(
    file_path: pathlib.Path,
    offsets: list[int],
    request_size: int,
    concurrency: int,
    pre_register_buffer: bool,
) -> tuple[float, int]:
    """Execute fixed-size reads with at most ``concurrency`` in flight."""
    buffers = [cupy.empty(request_size, dtype="uint8") for _ in range(concurrency)]
    if pre_register_buffer:
        for buf in buffers:
            kvikio.buffer.memory_register(buf)

    total = 0
    f = kvikio.CuFile(file_path, flags="r")
    try:
        t0 = clock()
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
        elapsed = clock() - t0
    finally:
        f.close()
        if pre_register_buffer:
            for buf in buffers:
                kvikio.buffer.memory_deregister(buf)
    return elapsed, total


def _warm_reuse_set(args, file_path: pathlib.Path, offsets: list[int], policy: str) -> None:
    if not args.warm_high_reuse:
        return
    # Warm with the same policy so the state being measured is the state that
    # policy naturally creates. The warm-up is deliberately untimed.
    config = _policy_config(policy, args.gds_threshold)
    config["num_threads"] = args.thread_pool_threads
    with kvikio.defaults.set(config):
        _run_read_batches(
            file_path,
            offsets,
            args.request_size,
            max(1, min(args.warmup_concurrency, max(args.concurrency))),
            args.pre_register_buffer,
        )


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
        if len(samples) > 1:
            std_pct = statistics.stdev(samples) / mean * 100 if mean else 0.0
            spread = f" +/- {std_pct:.2f}%"
        else:
            spread = ""
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
    _ensure_file(file_path, args.working_set)

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
    print(f"pre-reg-buf       | {args.pre_register_buffer}")
    print(f"high-reuse warmup | {args.warm_high_reuse}")
    print("====================================")

    rows: list[RunResult] = []

    for reuse in args.reuse:
        for concurrency in args.concurrency:
            context = Context(args.request_size, concurrency, reuse)

            for run_idx in range(1, args.nruns + 1):
                # Same offsets are reused across policies for a given
                # (context, run), making policy comparisons paired/fair.
                seed = args.seed + run_idx + 1009 * concurrency + (0 if reuse == "low" else 1000003)
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
                    if reuse == "high":
                        _warm_reuse_set(args, file_path, offsets, policy)

                    config = _policy_config(policy, args.gds_threshold)
                    config["num_threads"] = args.thread_pool_threads

                    with kvikio.defaults.set(config):
                        elapsed, logical_bytes = _run_read_batches(
                            file_path,
                            offsets,
                            context.request_size,
                            context.concurrency,
                            args.pre_register_buffer,
                        )

                    expected = len(offsets) * context.request_size
                    if logical_bytes != expected:
                        raise RuntimeError(
                            f"I/O mismatch: expected {expected} bytes, got {logical_bytes}"
                        )
                    throughput = logical_bytes / elapsed
                    result = RunResult(
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
                    rows.append(result)
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
        help="Aligned file/offset working-set size.",
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
        help="Number of distinct aligned offsets used by high-reuse mode.",
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
        help="KvikIO worker-thread count; held constant while concurrency varies.",
    )
    parser.add_argument(
        "--gds-threshold",
        default="16 KiB",
        type=parse_bytes,
        help="Threshold used only by the native KvikIO-threshold baseline.",
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
        help="Warm the high-reuse offset set before each timed run.",
    )
    parser.add_argument(
        "--warmup-concurrency",
        default=16,
        type=int,
        help="Concurrency used by the untimed high-reuse warm-up.",
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
