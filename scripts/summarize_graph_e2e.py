#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize end-to-end BFS/PageRank policy runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("e2e_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = data["selected_policy"]
        cache = data["cache"]
        shaping = data["shaping"]
        rows.append(
            {
                "source_json": path.name,
                "algorithm": data["algorithm"],
                "policy_mode": data["policy_mode"],
                "repeat_id": data["repeat_id"],
                "execution_order": data["execution_order"],
                "groute_enabled": data["groute_enabled"],
                "dispatch": data["dispatch"],
                "workload": policy["workload"],
                "path": policy["path"],
                "cache_policy": policy["cache"],
                "submit_policy": policy["submit"],
                "vertices": data["vertex_count"],
                "edges": data["edge_count"],
                "iterations": data["iterations"],
                "visited_vertices": data["visited_vertices"],
                "processed_edges": data["processed_edges"],
                "rank_sum": data["rank_sum"],
                "logical_requests": data["logical_requests"],
                "logical_bytes": data["logical_bytes"],
                "logical_trace_hash": data["logical_trace_hash"],
                "result_hash": data["result_hash"],
                "algorithm_seconds": data["algorithm_seconds"],
                "job_seconds": data["job_seconds"],
                "teps": data["teps"],
                "cache_hits": cache["hits"],
                "cache_misses": cache["misses"],
                "cache_storage_bytes": cache["storage_bytes"],
                "admitted_regions": cache["admitted_regions"],
                "physical_requests": shaping["physical_requests"],
                "submitted_bytes": shaping["submitted_bytes"],
            }
        )
    if not rows:
        raise SystemExit("No completed e2e JSON files found")
    return rows


def validate_correctness(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["algorithm"]), int(row["repeat_id"]))].append(row)
    failures: list[str] = []
    for key, group in sorted(groups.items()):
        exact_fields = [
            "iterations",
            "processed_edges",
            "logical_requests",
            "logical_bytes",
            "logical_trace_hash",
        ]
        if key[0] == "bfs":
            exact_fields.extend(("visited_vertices", "result_hash"))
        for field in exact_fields:
            values = {row[field] for row in group}
            if len(values) != 1:
                failures.append(f"{key}: {field} differs: {sorted(values)}")
        if key[0] == "pagerank":
            rank_sums = [float(row["rank_sum"]) for row in group]
            reference = max(1.0, abs(statistics.mean(rank_sums)))
            if max(rank_sums) - min(rank_sums) > reference * 1e-5:
                failures.append(f"{key}: rank_sum differs: {rank_sums}")
    if failures:
        raise SystemExit("cross-policy correctness failure:\n" + "\n".join(failures))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["algorithm"]), str(row["policy_mode"]))].append(row)
    result: list[dict[str, Any]] = []
    for (algorithm, policy_mode), group in sorted(groups.items()):
        first = group[0]
        times = [float(row["algorithm_seconds"]) for row in group]
        result.append(
            {
                "algorithm": algorithm,
                "policy_mode": policy_mode,
                "dispatch": first["dispatch"],
                "selected_policy": (
                    f'{first["path"]}/{first["cache_policy"]}/'
                    f'{first["submit_policy"]}'
                ),
                "runs": len(group),
                "iterations": first["iterations"],
                "processed_edges": first["processed_edges"],
                "logical_requests": first["logical_requests"],
                "algorithm_seconds_median": statistics.median(times),
                "algorithm_seconds_stdev": statistics.stdev(times)
                if len(times) > 1
                else 0.0,
                "job_seconds_median": _median(group, "job_seconds"),
                "teps_median": _median(group, "teps"),
                "cache_hit_ratio_median": statistics.median(
                    int(row["cache_hits"])
                    / max(1, int(row["cache_hits"]) + int(row["cache_misses"]))
                    for row in group
                ),
                "physical_requests_median": _median(group, "physical_requests"),
                "submitted_bytes_median": _median(group, "submitted_bytes"),
                "speedup_vs_kvikio_threshold": "",
                "fraction_of_best_forced": "",
            }
        )

    baseline = {
        row["algorithm"]: float(row["algorithm_seconds_median"])
        for row in result
        if row["policy_mode"] == "kvikio_threshold"
    }
    best_forced: dict[str, float] = {}
    for row in result:
        if row["policy_mode"] in {"auto", "kvikio_threshold"}:
            continue
        algorithm = str(row["algorithm"])
        value = float(row["algorithm_seconds_median"])
        best_forced[algorithm] = min(best_forced.get(algorithm, math.inf), value)
    for row in result:
        algorithm = str(row["algorithm"])
        duration = float(row["algorithm_seconds_median"])
        if algorithm in baseline:
            row["speedup_vs_kvikio_threshold"] = baseline[algorithm] / duration
        if algorithm in best_forced:
            row["fraction_of_best_forced"] = best_forced[algorithm] / duration
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.result_root)
    validate_correctness(rows)
    summary = summarize(rows)
    write_csv(args.result_root / "raw_results.csv", rows)
    write_csv(args.result_root / "summary.csv", summary)
    print(f"Validated {len(rows)} end-to-end runs")
    print(f"Wrote {args.result_root / 'raw_results.csv'}")
    print(f"Wrote {args.result_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
