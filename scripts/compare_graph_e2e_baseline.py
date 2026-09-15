#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build a clearly labeled G-Route versus external BaM reference table/figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groute-summary", type=Path, required=True)
    parser.add_argument("--bam-summary", type=Path, required=True)
    parser.add_argument("--groute-policy", default="auto")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    groute = {
        row["algorithm"]: row
        for row in _read(args.groute_summary)
        if row["policy_mode"] == args.groute_policy
    }
    bam_rows = _read(args.bam_summary)
    if not groute or not bam_rows:
        raise SystemExit("missing G-Route policy or BaM baseline rows")
    bam = {row["algorithm"]: row for row in bam_rows}
    rows = []
    for algorithm in ("bfs", "pagerank"):
        if algorithm not in groute or algorithm not in bam:
            continue
        groute_seconds = float(groute[algorithm]["algorithm_seconds_median"])
        bam_seconds = float(bam[algorithm]["algorithm_seconds_median"])
        rows.append(
            {
                "algorithm": algorithm,
                "groute_policy": args.groute_policy,
                "groute_seconds_median": groute_seconds,
                "bam_impl_type": bam[algorithm]["impl_type"],
                "bam_memalloc": bam[algorithm]["memalloc"],
                "bam_seconds_median": bam_seconds,
                "groute_speedup_vs_bam": bam_seconds / groute_seconds,
                "comparison_scope": "different_executors_external_reference",
            }
        )
    if not rows:
        raise SystemExit("summaries have no common BFS/PageRank algorithm")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table = args.output_prefix.with_suffix(".csv")
    with table.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install matplotlib and numpy to create plots") from error
    algorithms = [row["algorithm"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(4.4, 2.75))
    axis.bar(
        positions - width / 2,
        [float(row["bam_seconds_median"]) for row in rows],
        width,
        label="BaM reference",
        color="#9A9A9A",
        edgecolor="#333333",
        linewidth=0.55,
    )
    axis.bar(
        positions + width / 2,
        [float(row["groute_seconds_median"]) for row in rows],
        width,
        label=f"G-Route {args.groute_policy}",
        color="#2878B5",
        edgecolor="#333333",
        linewidth=0.55,
    )
    axis.set_xticks(positions, ["BFS" if name == "bfs" else "PageRank" for name in algorithms])
    axis.set_ylabel("Algorithm time (s)")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=0.6)
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    print(f"Wrote {table}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
