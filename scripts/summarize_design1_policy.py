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
        page_cache = data.get("page_cache", {})
        latency = data["batch_latency_us"]
        rows.append(
            {
                "source_json": path.name,
                "case": data["case"],
                "policy_mode": data.get("policy_mode", "auto"),
                "repeat_id": data.get("repeat_id", 0),
                "execution_order": data.get("execution_order", 0),
                "order_seed": data.get("order_seed", ""),
                "trace_seed": data.get("trace_seed", ""),
                "trace_id": data.get("trace_id", ""),
                "file": data.get("file", ""),
                "file_size_bytes": data.get("file_size_bytes", ""),
                "file_allocated_bytes": data.get("file_allocated_bytes", ""),
                "working_set_bytes": data.get("working_set_bytes", ""),
                "page_cache_mode": page_cache.get("mode", "unknown"),
                "file_cache_evicted": page_cache.get("file_evicted", False),
                "global_cache_dropped": page_cache.get("global_dropped", False),
                "cached_kib_before": page_cache.get("cached_kib_before", ""),
                "cached_kib_after": page_cache.get("cached_kib_after", ""),
                "num_threads": data.get("num_threads", ""),
                "workload": policy["workload"],
                "path": policy["path"],
                "cache": policy["cache"],
                "submit": policy["submit"],
                "profile_requests": data.get("profile_requests", 64),
                "warmup_requests": data.get("warmup_requests", 0),
                "warmup_admitted_regions": data.get(
                    "warmup_admitted_regions", 0
                ),
                "warmup_storage_bytes": data.get("warmup_storage_bytes", 0),
                "cache_entries_before_measurement": data.get(
                    "cache_entries_before_measurement", 0
                ),
                "elapsed_seconds": data.get("elapsed_seconds", ""),
                "iops": data["iops"],
                "mib_per_second": data["logical_mib_per_second"],
                "batch_p50_us": latency.get("p50", ""),
                "batch_p95_us": latency.get("p95", ""),
                "batch_p99_us": latency["p99"],
                "cache_hits": cache.get("hits", 0),
                "cache_misses": cache.get("misses", 0),
                "admitted_regions": cache.get("admitted_regions", 0),
                "admission_bypasses": cache.get("admission_bypasses", 0),
                "storage_bytes": cache.get("storage_bytes", 0),
                "physical_requests": shaping.get("physical_requests", 0),
            }
        )
    if not rows:
        raise SystemExit("No completed design1 JSON files found")

    # Results are paired by (case, repeat).  Refuse to summarize a matrix in
    # which policies accidentally used different logical request traces.
    traces: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row["trace_id"]:
            traces[(row["case"], int(row["repeat_id"]))].add(row["trace_id"])
    mismatched = {key: ids for key, ids in traces.items() if len(ids) != 1}
    if mismatched:
        raise SystemExit(f"Mismatched trace_id within paired runs: {mismatched}")

    raw_path = args.result_root / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["policy_mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for (case, policy_mode), group in sorted(grouped.items()):
        first = group[0]
        summary.append(
            {
                "case": case,
                "policy_mode": policy_mode,
                "policy": f'{first["path"]}/{first["cache"]}/{first["submit"]}',
                "working_set_bytes": first["working_set_bytes"],
                "page_cache_mode": first["page_cache_mode"],
                "num_threads": first["num_threads"],
                "runs": len(group),
                "iops_median": statistics.median(float(x["iops"]) for x in group),
                "iops_stdev": statistics.stdev(float(x["iops"]) for x in group)
                if len(group) > 1
                else 0.0,
                "batch_p99_us_median": statistics.median(
                    float(x["batch_p99_us"]) for x in group
                ),
                "batch_p50_us_median": statistics.median(
                    float(x["batch_p50_us"] or x["batch_p99_us"]) for x in group
                ),
                "batch_p95_us_median": statistics.median(
                    float(x["batch_p95_us"] or x["batch_p99_us"]) for x in group
                ),
                "cache_hits_median": statistics.median(
                    int(x["cache_hits"]) for x in group
                ),
                "cache_misses_median": statistics.median(
                    int(x["cache_misses"]) for x in group
                ),
                "admitted_regions_median": statistics.median(
                    int(x["admitted_regions"]) for x in group
                ),
                "admission_bypasses_median": statistics.median(
                    int(x["admission_bypasses"]) for x in group
                ),
                "warmup_admitted_regions_median": statistics.median(
                    int(x["warmup_admitted_regions"]) for x in group
                ),
                "warmup_storage_bytes_median": statistics.median(
                    int(x["warmup_storage_bytes"]) for x in group
                ),
                "cache_entries_before_measurement_median": statistics.median(
                    int(x["cache_entries_before_measurement"]) for x in group
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
