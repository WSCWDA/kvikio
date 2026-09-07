#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON files produced by run_design2_thread_sweep.sh."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


NAME_PATTERN = re.compile(r"design2_c(?P<clusters>\d+)_t(?P<threads>\d+)_r(?P<repeat>\d+)")
RAW_FIELDS = [
    "case",
    "repeat",
    "clusters_per_batch",
    "num_threads",
    "direct_iops",
    "host_iops",
    "shaped_iops",
    "direct_mib_per_second",
    "host_mib_per_second",
    "shaped_mib_per_second",
    "shaped_vs_direct_iops",
    "shaped_vs_host_iops",
    "physical_request_reduction",
    "submitted_amplification",
    "logical_requests_per_physical",
    "logical_requests",
    "physical_requests",
    "collection_batches",
    "max_collected_requests",
    "max_inflight_physical",
    "direct_fallbacks",
]


def row_from_data(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    match = NAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"unexpected result filename: {path.name}")
    results = {result["mode"]: result for result in data["results"]}
    if set(results) != {"direct", "host", "shaped"}:
        raise ValueError(f"missing benchmark mode in {path}")
    shaped = results["shaped"]["context"]["shaping"]
    summary = data["summary"]
    parsed_clusters = int(match.group("clusters"))
    configured_clusters = int(results["shaped"]["clusters_per_batch"])
    if configured_clusters != parsed_clusters:
        raise ValueError(
            f"{path}: requested {parsed_clusters} clusters but benchmark used "
            f"{configured_clusters}"
        )
    parsed_threads = int(match.group("threads"))
    configured_threads = int(data.get("num_threads", parsed_threads))
    if configured_threads != parsed_threads:
        raise ValueError(
            f"{path}: requested {parsed_threads} threads but KvikIO used "
            f"{configured_threads}"
        )
    return {
        "case": path.stem,
        "repeat": int(match.group("repeat")),
        "clusters_per_batch": configured_clusters,
        "num_threads": configured_threads,
        "direct_iops": results["direct"]["iops"],
        "host_iops": results["host"]["iops"],
        "shaped_iops": results["shaped"]["iops"],
        "direct_mib_per_second": results["direct"]["logical_mib_per_second"],
        "host_mib_per_second": results["host"]["logical_mib_per_second"],
        "shaped_mib_per_second": results["shaped"]["logical_mib_per_second"],
        "shaped_vs_direct_iops": summary["shaped_vs_direct_iops"],
        "shaped_vs_host_iops": summary["shaped_vs_host_iops"],
        "physical_request_reduction": summary["physical_request_reduction"],
        "submitted_amplification": summary["submitted_amplification"],
        "logical_requests_per_physical": summary[
            "logical_requests_per_physical"
        ],
        "logical_requests": shaped["logical_requests"],
        "physical_requests": shaped["physical_requests"],
        "collection_batches": shaped["collection_batches"],
        "max_collected_requests": shaped["max_collected_requests"],
        "max_inflight_physical": shaped["max_inflight_physical"],
        "direct_fallbacks": shaped["direct_fallbacks"],
    }


def median(group: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in group)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    rows = [
        row_from_data(path, json.loads(path.read_text()))
        for path in sorted(args.result_root.glob("design2_c*_t*_r*.json"))
    ]
    if not rows:
        raise SystemExit(f"no successful result files found in {args.result_root}")

    raw_out = args.raw_out or args.result_root / "raw_results.csv"
    summary_out = args.summary_out or args.result_root / "summary.csv"
    write_csv(raw_out, rows, RAW_FIELDS)

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["clusters_per_batch"], row["num_threads"])].append(row)

    summary_rows = []
    for (clusters, threads), group in sorted(groups.items()):
        summary_rows.append(
            {
                "clusters_per_batch": clusters,
                "num_threads": threads,
                "runs": len(group),
                "direct_iops_median": median(group, "direct_iops"),
                "host_iops_median": median(group, "host_iops"),
                "shaped_iops_median": median(group, "shaped_iops"),
                "shaped_iops_stdev": statistics.stdev(
                    float(row["shaped_iops"]) for row in group
                )
                if len(group) > 1
                else 0.0,
                "shaped_vs_direct_median": median(
                    group, "shaped_vs_direct_iops"
                ),
                "shaped_vs_host_median": median(group, "shaped_vs_host_iops"),
                "physical_reduction_median": median(
                    group, "physical_request_reduction"
                ),
                "amplification_median": median(group, "submitted_amplification"),
                "logical_per_physical_median": median(
                    group, "logical_requests_per_physical"
                ),
                "collection_batches_median": median(group, "collection_batches"),
                "max_collected_requests_max": max(
                    int(row["max_collected_requests"]) for row in group
                ),
                "max_inflight_physical_max": max(
                    int(row["max_inflight_physical"]) for row in group
                ),
                "direct_fallbacks_median": median(group, "direct_fallbacks"),
            }
        )

    summary_fields = list(summary_rows[0])
    write_csv(summary_out, summary_rows, summary_fields)
    print(f"Wrote {raw_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
