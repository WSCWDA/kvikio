#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON files produced by scripts/run_gds_hcache_matrix.sh."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_FIELDS = [
    "case",
    "path",
    "io_size",
    "line_size",
    "cache_bytes",
    "hot_bytes",
    "requests",
    "seconds",
    "iops",
    "mib_per_s",
    "avg_us",
    "hit_rate",
    "hits",
    "misses",
    "evictions",
    "storage_bytes",
    "h2d_bytes",
    "read_amplification_vs_request",
]


def _row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    stats = data.get("stats", {})
    row = {field: data.get(field) for field in RAW_FIELDS}
    row["case"] = path.stem
    for key in ["hits", "misses", "evictions", "storage_bytes", "h2d_bytes"]:
        row[key] = stats.get(key, 0)
    return row


def _summary_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["path"],
        row["io_size"],
        row["line_size"],
        row["cache_bytes"],
        row["hot_bytes"],
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    rows = [_row(path) for path in sorted(args.result_root.glob("*.json"))]
    raw_out = args.raw_out or args.result_root / "raw_results.csv"
    summary_out = args.summary_out or args.result_root / "summary.csv"

    _write_csv(raw_out, rows, RAW_FIELDS)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_summary_key(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        path, io_size, line_size, cache_bytes, hot_bytes = key
        iops_values = [float(row["iops"]) for row in group]
        mib_values = [float(row["mib_per_s"]) for row in group]
        hit_values = [
            float(row["hit_rate"]) for row in group if row["hit_rate"] is not None
        ]
        summary_rows.append(
            {
                "path": path,
                "io_size": io_size,
                "line_size": line_size,
                "cache_bytes": cache_bytes,
                "hot_bytes": hot_bytes,
                "runs": len(group),
                "iops_mean": statistics.mean(iops_values),
                "iops_stdev": statistics.stdev(iops_values)
                if len(iops_values) > 1
                else 0.0,
                "mib_per_s_mean": statistics.mean(mib_values),
                "mib_per_s_stdev": statistics.stdev(mib_values)
                if len(mib_values) > 1
                else 0.0,
                "hit_rate_mean": statistics.mean(hit_values) if hit_values else "",
                "misses_mean": statistics.mean(float(row["misses"]) for row in group),
                "evictions_mean": statistics.mean(
                    float(row["evictions"]) for row in group
                ),
                "storage_bytes_mean": statistics.mean(
                    float(row["storage_bytes"]) for row in group
                ),
            }
        )

    _write_csv(
        summary_out,
        summary_rows,
        [
            "path",
            "io_size",
            "line_size",
            "cache_bytes",
            "hot_bytes",
            "runs",
            "iops_mean",
            "iops_stdev",
            "mib_per_s_mean",
            "mib_per_s_stdev",
            "hit_rate_mean",
            "misses_mean",
            "evictions_mean",
            "storage_bytes_mean",
        ],
    )
    print(f"Wrote {raw_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
