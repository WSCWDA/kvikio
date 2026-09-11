#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_TIME = re.compile(r"^\[REPORT\] Time ([0-9.eE+-]+)$", re.MULTILINE)
REPORT_IO = re.compile(r"^\[REPORT\] IO ([0-9]+)$", re.MULTILINE)
REPORT_RECALL = re.compile(r"^\[REPORT\] RECALL: ([0-9.eE+-]+)$", re.MULTILINE)
REPORT_LATENCY = re.compile(r"^\[REPORT\] LAT[0-9]+ ([0-9.eE+-]+)$", re.MULTILINE)
GROUTE_STATS = re.compile(r"^\[GROUTE_STATS\] (\{.*\})$", re.MULTILINE)


def query_count(path: Path, data_type: str) -> int:
    with path.open("rb") as stream:
        header = stream.read(4)
    if len(header) != 4:
        raise ValueError(f"query file has no vector header: {path}")
    (dimension,) = struct.unpack("<I", header)
    element_bytes = 4 if data_type == "float" else 1
    record_bytes = 4 + dimension * element_bytes
    size = path.stat().st_size
    if dimension == 0 or size % record_bytes:
        raise ValueError(
            f"{path} is not a valid {data_type} vector file: "
            f"size={size}, dimension={dimension}"
        )
    return size // record_bytes


def one(pattern: re.Pattern[str], text: str, label: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def parse_log(path: Path, total_queries: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed = float(one(REPORT_TIME, text, "time report"))
    latency = [float(value) for value in REPORT_LATENCY.findall(text)]
    if not latency:
        raise ValueError("no per-thread latency reports")
    stats_matches = GROUTE_STATS.findall(text)
    stats = json.loads(stats_matches[-1]) if stats_matches else {}
    mode_match = re.match(r"diskann_(.+)_r[0-9]+\.log$", path.name)
    if not mode_match:
        raise ValueError(f"unexpected log name: {path.name}")
    return {
        "run": path.stem,
        "mode": mode_match.group(1),
        "elapsed_seconds": elapsed,
        "qps": total_queries / elapsed,
        "recall": float(one(REPORT_RECALL, text, "recall report")),
        "page_reads": int(one(REPORT_IO, text, "I/O report")),
        "mean_thread_latency_ms": statistics.mean(latency),
        "workload": stats.get("workload", ""),
        "path": stats.get("path", ""),
        "cache": stats.get("cache", ""),
        "submit": stats.get("submit", ""),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "admitted_regions": stats.get("admitted_regions", 0),
        "physical_requests": stats.get("physical_requests", 0),
        "submitted_bytes": stats.get("submitted_bytes", 0),
        "io_batch_latency_us_p99": stats.get("io_batch_latency_us_p99", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DiskANN/GustANN G-Route runs")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--data-type", choices=("float", "uint8"), required=True)
    parser.add_argument("--query-repeats", type=int, required=True)
    parser.add_argument("--recall-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    total_queries = query_count(args.query_file, args.data_type) * args.query_repeats
    rows = [
        parse_log(path, total_queries)
        for path in sorted(args.result_root.glob("diskann_*.log"))
        if "[REPORT] Time " in path.read_text(encoding="utf-8", errors="replace")
    ]
    if not rows:
        raise SystemExit("No successful DiskANN/GustANN logs found")

    raw_path = args.result_root / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary: list[dict[str, Any]] = []
    for mode, group in sorted(grouped.items()):
        qps = [float(row["qps"]) for row in group]
        policies = sorted(
            {
                "/".join(
                    str(row[key]) for key in ("path", "cache", "submit")
                    if row[key]
                )
                for row in group
            }
            - {""}
        )
        summary.append(
            {
                "mode": mode,
                "runs": len(group),
                "qps_median": statistics.median(qps),
                "qps_stdev": statistics.stdev(qps) if len(qps) > 1 else 0.0,
                "latency_ms_median": statistics.median(
                    float(row["mean_thread_latency_ms"]) for row in group
                ),
                "recall_median": statistics.median(
                    float(row["recall"]) for row in group
                ),
                "page_reads_median": statistics.median(
                    int(row["page_reads"]) for row in group
                ),
                "selected_policy": "|".join(policies),
                "cache_hits_median": statistics.median(
                    int(row["cache_hits"]) for row in group
                ),
                "physical_requests_median": statistics.median(
                    int(row["physical_requests"]) for row in group
                ),
            }
        )
    recall_by_mode = {
        row["mode"]: float(row["recall_median"]) for row in summary
    }
    if len(recall_by_mode) > 1 and (
        max(recall_by_mode.values()) - min(recall_by_mode.values())
        > args.recall_tolerance
    ):
        raise SystemExit(
            "Recall mismatch across backends: "
            f"{recall_by_mode} (tolerance={args.recall_tolerance})"
        )
    aio_qps = next(
        (float(row["qps_median"]) for row in summary if row["mode"] == "aio"),
        None,
    )
    for row in summary:
        row["qps_vs_aio"] = (
            float(row["qps_median"]) / aio_qps if aio_qps else ""
        )
    summary_path = args.result_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
