# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure G-Route routing overhead in a GDS-dominant concurrency region.

This benchmark fixes a cold, low-reuse 64-KiB workload for which the forced
GDS path is expected to dominate on the evaluation platform, then sweeps the
number of outstanding logical requests. It compares only ``gds_direct`` and
``auto`` and reports the throughput ratio and routing penalty:

    ratio   = throughput_auto / throughput_gds
    penalty = max(0, 1 - ratio)

The goal is not to demonstrate a path crossover. Instead, it tests whether
G-Route preserves the scaling behavior of the already-preferred direct path
without introducing material context-tracking/routing overhead.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics

from dask.utils import parse_bytes

from kvikio.benchmarks.groute_context_common import (
    Context,
    add_common_arguments,
    run_context_benchmark,
)


def _summarize_penalty(rows, output_csv):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.concurrency, row.policy), []).append(row.throughput_Bps)

    concurrencies = sorted({row.concurrency for row in rows})
    summary_rows = []

    print("\nConcurrency robustness")
    print("======================")
    print("C    GDS GiB/s    Auto GiB/s   Auto/GDS   Penalty")

    penalties = []
    ratios = []

    for concurrency in concurrencies:
        gds = grouped.get((concurrency, "gds_direct"), [])
        auto = grouped.get((concurrency, "auto"), [])
        if not gds or not auto:
            raise RuntimeError(
                "robustness benchmark requires both gds_direct and auto policies"
            )

        gds_mean = statistics.mean(gds)
        auto_mean = statistics.mean(auto)
        ratio = auto_mean / gds_mean if gds_mean else 0.0
        penalty = max(0.0, 1.0 - ratio)
        ratios.append(ratio)
        penalties.append(penalty)

        gds_std = statistics.stdev(gds) if len(gds) > 1 else 0.0
        auto_std = statistics.stdev(auto) if len(auto) > 1 else 0.0

        summary_rows.append(
            {
                "concurrency": concurrency,
                "gds_mean_GiBps": gds_mean / (1024**3),
                "gds_std_GiBps": gds_std / (1024**3),
                "auto_mean_GiBps": auto_mean / (1024**3),
                "auto_std_GiBps": auto_std / (1024**3),
                "auto_to_gds": ratio,
                "penalty": penalty,
                "penalty_percent": penalty * 100.0,
            }
        )

        print(
            f"{concurrency:<3d}  "
            f"{gds_mean / (1024**3):>9.3f}    "
            f"{auto_mean / (1024**3):>9.3f}    "
            f"{ratio:>8.3f}   "
            f"{penalty * 100:>6.2f}%"
        )

    mean_ratio = statistics.mean(ratios)
    mean_penalty = statistics.mean(penalties)
    worst_penalty = max(penalties)

    print("----------------------")
    print(f"mean Auto/GDS  | {mean_ratio:.3f}")
    print(f"mean penalty   | {mean_penalty * 100:.2f}%")
    print(f"worst penalty  | {worst_penalty * 100:.2f}%")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"penalty CSV    | {output_csv}")
    return mean_penalty, worst_penalty


def main(args) -> None:
    # Prior characterization shows this region is GDS-dominant on the
    # evaluation platform: fixed 64-KiB, cold, low-reuse reads. Logical
    # concurrency is therefore the only changing context dimension.
    contexts = [
        Context(
            request_size=args.request_size,
            concurrency=concurrency,
            reuse="low",
            hotset_requests=1,
            label=f"concurrency-{concurrency}",
        )
        for concurrency in args.concurrency
    ]

    rows = run_context_benchmark(
        benchmark_name="concurrency-robustness",
        args=args,
        contexts=contexts,
    )

    penalty_csv = args.penalty_csv
    if penalty_csv is None:
        penalty_csv = args.csv.with_name(f"{args.csv.stem}_penalty{args.csv.suffix}")

    _, worst_penalty = _summarize_penalty(rows, penalty_csv)

    if args.fail_above_penalty is not None and worst_penalty > args.fail_above_penalty:
        raise RuntimeError(
            f"worst Auto penalty {worst_penalty * 100:.2f}% exceeds "
            f"limit {args.fail_above_penalty * 100:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "G-Route concurrency robustness benchmark. It compares forced "
            "GDS with AUTO while sweeping outstanding logical I/O concurrency "
            "inside a cold, low-reuse, GDS-dominant context region."
        )
    )
    add_common_arguments(parser)
    parser.set_defaults(
        policy=["gds_direct", "auto"],
        requests_per_run=1024,
        nruns=5,
        thread_pool_threads=64,
    )
    parser.add_argument(
        "--request-size",
        type=parse_bytes,
        default=parse_bytes("64 KiB"),
        help="Fixed request size; default 64 KiB.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Outstanding logical requests per batch.",
    )
    parser.add_argument(
        "--penalty-csv",
        type=pathlib.Path,
        default=None,
        help="Optional per-concurrency Auto/GDS penalty summary CSV.",
    )
    parser.add_argument(
        "--fail-above-penalty",
        type=float,
        default=None,
        help=(
            "Optional fractional worst-case penalty limit (e.g., 0.05 for 5%%). "
            "When set, exit with an error if any concurrency exceeds it."
        ),
    )
    main(parser.parse_args())
