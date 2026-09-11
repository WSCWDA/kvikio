#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.result_root.glob("design1_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = data["selected_policy"]
        cache = data["host_cache_delta"]
        shaping = data["request_shaping_delta"]
        rows.append(
            {
                "case": data["case"],
                "workload": policy["workload"],
                "path": policy["path"],
                "cache": policy["cache"],
                "submit": policy["submit"],
                "iops": data["iops"],
                "mib_per_second": data["logical_mib_per_second"],
                "batch_p99_us": data["batch_latency_us"]["p99"],
                "cache_hits": cache.get("hits", 0),
                "admitted_regions": cache.get("admitted_regions", 0),
                "physical_requests": shaping.get("physical_requests", 0),
            }
        )
    if not rows:
        raise SystemExit("No completed design1 JSON files found")

    raw_path = args.result_root / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    summary: list[dict[str, Any]] = []
    for case, group in sorted(grouped.items()):
        first = group[0]
        summary.append(
            {
                "case": case,
                "policy": f'{first["path"]}/{first["cache"]}/{first["submit"]}',
                "runs": len(group),
                "iops_median": statistics.median(float(x["iops"]) for x in group),
                "iops_stdev": statistics.stdev(float(x["iops"]) for x in group)
                if len(group) > 1
                else 0.0,
                "batch_p99_us_median": statistics.median(
                    float(x["batch_p99_us"]) for x in group
                ),
                "cache_hits_median": statistics.median(
                    int(x["cache_hits"]) for x in group
                ),
                "physical_requests_median": statistics.median(
                    int(x["physical_requests"]) for x in group
                ),
            }
        )
    summary_path = args.result_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
