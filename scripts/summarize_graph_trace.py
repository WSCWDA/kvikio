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
    for path in sorted(args.result_root.glob("graph_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = data["selected_policy"]
        cache = data["host_cache"]
        shaping = policy.get("shaping", {})
        latency = data["batch_latency_us"]
        rows.append(
            {
                "source_json": path.name,
                "algorithm": data["algorithm"],
                "policy_mode": data["policy_mode"],
                "repeat_id": data["repeat_id"],
                "execution_order": data["execution_order"],
                "trace_sha256": data["trace_sha256"],
                "trace_requests": data["trace_requests"],
                "page_size": data["page_size"],
                "batch_size": data["batch_size"],
                "num_threads": data["num_threads"],
                "workload": policy["workload"],
                "path": policy["path"],
                "cache": policy["cache"],
                "submit": policy["submit"],
                "io_seconds": data["io_seconds"],
                "total_seconds": data["total_seconds"],
                "iops": data["iops"],
                "mib_per_second": data["logical_mib_per_second"],
                "batch_p50_us": latency["p50"],
                "batch_p95_us": latency["p95"],
                "batch_p99_us": latency["p99"],
                "cache_hits": cache.get("hits", 0),
                "cache_misses": cache.get("misses", 0),
                "storage_bytes": cache.get("storage_bytes", 0),
                "admitted_regions": cache.get("admitted_regions", 0),
                "admission_bypasses": cache.get("admission_bypasses", 0),
                "physical_requests": shaping.get("physical_requests", 0),
                "submitted_bytes": shaping.get("submitted_bytes", 0),
            }
        )
    if not rows:
        raise SystemExit("No completed graph JSON files found")

    traces: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        traces[(row["algorithm"], int(row["repeat_id"]))].add(
            str(row["trace_sha256"])
        )
    mismatch = {key: value for key, value in traces.items() if len(value) != 1}
    if mismatch:
        raise SystemExit(f"policies did not replay identical traces: {mismatch}")

    raw = args.result_root / "raw_results.csv"
    with raw.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["policy_mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for (algorithm, policy_mode), group in sorted(grouped.items()):
        first = group[0]
        summary.append(
            {
                "algorithm": algorithm,
                "policy_mode": policy_mode,
                "selected_policy": (
                    f'{first["path"]}/{first["cache"]}/{first["submit"]}'
                ),
                "runs": len(group),
                "trace_requests": first["trace_requests"],
                "iops_median": statistics.median(float(x["iops"]) for x in group),
                "iops_stdev": statistics.stdev(float(x["iops"]) for x in group)
                if len(group) > 1
                else 0.0,
                "io_seconds_median": statistics.median(
                    float(x["io_seconds"]) for x in group
                ),
                "batch_p99_us_median": statistics.median(
                    float(x["batch_p99_us"]) for x in group
                ),
                "cache_hit_ratio_median": statistics.median(
                    int(x["cache_hits"])
                    / max(1, int(x["cache_hits"]) + int(x["cache_misses"]))
                    for x in group
                ),
                "storage_bytes_median": statistics.median(
                    int(x["storage_bytes"]) for x in group
                ),
                "physical_requests_median": statistics.median(
                    int(x["physical_requests"]) for x in group
                ),
            }
        )
    threshold = {
        row["algorithm"]: float(row["iops_median"])
        for row in summary
        if row["policy_mode"] == "kvikio_threshold"
    }
    best_forced: dict[str, float] = defaultdict(float)
    for row in summary:
        if row["policy_mode"] not in ("auto", "kvikio_threshold"):
            best_forced[row["algorithm"]] = max(
                best_forced[row["algorithm"]], float(row["iops_median"])
            )
    for row in summary:
        base = threshold.get(row["algorithm"], 0.0)
        best = best_forced.get(row["algorithm"], 0.0)
        row["speedup_vs_kvikio_threshold"] = (
            float(row["iops_median"]) / base if base else ""
        )
        row["fraction_of_best_forced"] = (
            float(row["iops_median"]) / best if best else ""
        )
    summary_path = args.result_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {raw}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
