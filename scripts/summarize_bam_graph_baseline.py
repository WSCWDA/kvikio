#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Extract BaM BFS/PageRank algorithm times into a separate reference table."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


TIME_PATTERNS = (
    re.compile(r"AvgTime\s*[:=]?\s*([0-9.eE+-]+)\s*(ms|s)?", re.IGNORECASE),
    re.compile(r"TotalTime\s*[:=]?\s*([0-9.eE+-]+)\s*(ms|s)?", re.IGNORECASE),
    re.compile(
        r"(?:elapsed|runtime|time)\s*[:=]\s*([0-9.eE+-]+)"
        r"\s*(ms|s|sec|seconds)?",
        re.IGNORECASE,
    ),
)
NAME = re.compile(r"bam_(bfs|pagerank)_i(\d+)_m(\d+)_r(\d+)\.log$")


def extract_seconds(text: str, path: Path) -> float:
    for pattern in TIME_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            value, unit = matches[-1]
            return float(value) / 1000.0 if unit.lower() == "ms" else float(value)
    raise ValueError(f"cannot find algorithm time in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for path in sorted(args.result_root.glob("bam_*.log")):
        match = NAME.match(path.name)
        if not match:
            continue
        algorithm, impl_type, memalloc, repeat = match.groups()
        try:
            seconds = extract_seconds(path.read_text(encoding="utf-8", errors="replace"), path)
        except ValueError as error:
            failures.append(str(error))
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "impl_type": int(impl_type),
                "memalloc": int(memalloc),
                "repeat_id": int(repeat),
                "algorithm_seconds": seconds,
                "source_log": path.name,
            }
        )
    if failures:
        raise SystemExit("\n".join(failures))
    if not rows:
        raise SystemExit("No parseable BaM baseline logs found")
    with (args.result_root / "raw_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["algorithm"]), int(row["impl_type"]), int(row["memalloc"]))].append(
            float(row["algorithm_seconds"])
        )
    summary = [
        {
            "algorithm": key[0],
            "impl_type": key[1],
            "memalloc": key[2],
            "runs": len(values),
            "algorithm_seconds_median": statistics.median(values),
            "algorithm_seconds_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in sorted(groups.items())
    ]
    with (args.result_root / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {args.result_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
