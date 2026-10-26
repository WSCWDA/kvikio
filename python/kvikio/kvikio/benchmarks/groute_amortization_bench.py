# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure how quickly G-Route AUTO startup cost is amortized.

The benchmark fixes request size, logical concurrency, and reuse, then sweeps
only the number of requests in a run. It compares forced GDS against AUTO and
reports paired per-run elapsed-time overhead and throughput ratio.

Using the same base seed for every request-count point makes the shorter trace
a prefix of the longer trace for a given repetition (subject to the common
trace generator), while GDS and AUTO always execute exactly the same trace.
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


def _write_combined_raw(path: pathlib.Path, all_rows: list[tuple[int, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "requests_per_run",
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
        for requests_per_run, r in all_rows:
            writer.writerow(
                [
                    requests_per_run,
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


def _summarize(all_rows: list[tuple[int, object]], summary_csv: pathlib.Path) -> None:
    by_request_count: dict[int, list[object]] = {}
    for requests_per_run, row in all_rows:
        by_request_count.setdefault(requests_per_run, []).append(row)

    summary_rows = []

    print("\nStartup amortization")
    print("====================")
    print(
        "Requests    GDS GiB/s   Auto GiB/s   Auto/GDS   "
        "Elapsed overhead"
    )

    for requests_per_run in sorted(by_request_count):
        rows = by_request_count[requests_per_run]
        paired: dict[int, dict[str, object]] = {}
        for row in rows:
            paired.setdefault(row.run, {})[row.policy] = row

        throughput_ratios = []
        elapsed_overheads = []
        gds_throughputs = []
        auto_throughputs = []

        for run_idx in sorted(paired):
            sample = paired[run_idx]
            if "gds_direct" not in sample or "auto" not in sample:
                raise RuntimeError(
                    f"missing paired GDS/AUTO result for requests={requests_per_run}, "
                    f"run={run_idx}"
                )

            gds = sample["gds_direct"]
            auto = sample["auto"]

            gds_throughputs.append(gds.throughput_Bps)
            auto_throughputs.append(auto.throughput_Bps)
            throughput_ratios.append(auto.throughput_Bps / gds.throughput_Bps)
            elapsed_overheads.append((auto.elapsed_s - gds.elapsed_s) / gds.elapsed_s)

        gds_mean = statistics.mean(gds_throughputs)
        auto_mean = statistics.mean(auto_throughputs)
        ratio_mean = statistics.mean(throughput_ratios)
        ratio_std = statistics.stdev(throughput_ratios) if len(throughput_ratios) > 1 else 0.0
        overhead_mean = statistics.mean(elapsed_overheads)
        overhead_std = statistics.stdev(elapsed_overheads) if len(elapsed_overheads) > 1 else 0.0

        summary_rows.append(
            {
                "requests_per_run": requests_per_run,
                "gds_mean_GiBps": gds_mean / (1024**3),
                "auto_mean_GiBps": auto_mean / (1024**3),
                "auto_to_gds_mean": ratio_mean,
                "auto_to_gds_std": ratio_std,
                "elapsed_overhead_mean": overhead_mean,
                "elapsed_overhead_std": overhead_std,
                "elapsed_overhead_percent": overhead_mean * 100.0,
                "nruns": len(throughput_ratios),
            }
        )

        print(
            f"{requests_per_run:>8d}    "
            f"{gds_mean / (1024**3):>8.3f}    "
            f"{auto_mean / (1024**3):>8.3f}    "
            f"{ratio_mean:>8.3f}    "
            f"{overhead_mean * 100:>8.2f}%"
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"summary CSV | {summary_csv}")


def main(args) -> None:
    if args.policy != ["gds_direct", "auto"]:
        raise ValueError(
            "amortization benchmark requires exactly: --policy gds_direct auto"
        )
    if any(n < 1 for n in args.request_counts):
        raise ValueError("all --request-counts values must be >= 1")

    original_csv = args.csv
    original_requests_per_run = args.requests_per_run

    part_dir = original_csv.parent / f"{original_csv.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[int, object]] = []

    try:
        for requests_per_run in args.request_counts:
            print("\n" + "=" * 72)
            print(f"Amortization point: {requests_per_run} requests/run")
            print("=" * 72)

            args.requests_per_run = requests_per_run
            args.csv = part_dir / f"requests_{requests_per_run}.csv"

            context = Context(
                request_size=args.request_size,
                concurrency=args.concurrency,
                reuse="low",
                hotset_requests=1,
                label=f"requests-{requests_per_run}",
            )

            rows = run_context_benchmark(
                benchmark_name="startup-amortization",
                args=args,
                contexts=[context],
            )
            all_rows.extend((requests_per_run, row) for row in rows)
    finally:
        args.csv = original_csv
        args.requests_per_run = original_requests_per_run

    _write_combined_raw(original_csv, all_rows)

    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = original_csv.with_name(
            f"{original_csv.stem}_summary{original_csv.suffix}"
        )
    _summarize(all_rows, summary_csv)

    print(f"raw CSV     | {original_csv}")
    print(f"point CSVs  | {part_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "G-Route startup-cost amortization benchmark. Request count varies "
            "while request size, logical concurrency, and low-reuse context stay fixed."
        )
    )
    add_common_arguments(parser)
    parser.set_defaults(
        policy=["gds_direct", "auto"],
        nruns=10,
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
        type=int,
        default=4,
        help="Fixed outstanding logical requests; default 4.",
    )
    parser.add_argument(
        "--request-counts",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Requests per timed run to sweep.",
    )
    parser.add_argument(
        "--summary-csv",
        type=pathlib.Path,
        default=None,
        help="Optional amortization summary CSV path.",
    )
    main(parser.parse_args())
