#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Create paper-ready BFS/PageRank policy comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICIES = [
    "kvikio_threshold",
    "auto",
    "host_direct",
    "host_cache",
    "gds_direct",
    "gds_shaped",
]
LABELS = {
    "kvikio_threshold": "KvikIO\nthreshold",
    "auto": "G-Route\nAuto",
    "host_direct": "Host\ndirect",
    "host_cache": "Host\ncache",
    "gds_direct": "GDS\ndirect",
    "gds_shaped": "GDS\nshaped",
}
COLORS = {
    "kvikio_threshold": "#9A9A9A",
    "auto": "#2878B5",
    "host_direct": "#9AC9DB",
    "host_cache": "#55A868",
    "gds_direct": "#E5A84B",
    "gds_shaped": "#D66B45",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install matplotlib and numpy to create plots") from error

    with args.summary.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    algorithms = [
        name
        for name in ("bfs", "pagerank")
        if any(row["algorithm"] == name for row in rows)
    ]
    if not algorithms:
        raise SystemExit("summary contains no BFS/PageRank rows")
    lookup = {(row["algorithm"], row["policy_mode"]): row for row in rows}

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(3.45 * len(algorithms), 2.65),
        squeeze=False,
    )
    for axis, algorithm in zip(axes[0], algorithms):
        available = [policy for policy in POLICIES if (algorithm, policy) in lookup]
        values = [
            float(lookup[(algorithm, policy)]["speedup_vs_kvikio_threshold"])
            for policy in available
        ]
        positions = np.arange(len(available))
        bars = axis.bar(
            positions,
            values,
            color=[COLORS[policy] for policy in available],
            edgecolor="#333333",
            linewidth=0.55,
        )
        axis.axhline(1.0, color="#555555", linewidth=0.75, linestyle="--")
        axis.set_xticks(positions, [LABELS[policy] for policy in available], fontsize=7)
        axis.set_ylabel("Speedup over KvikIO threshold" if axis is axes[0][0] else "")
        axis.set_title("BFS" if algorithm == "bfs" else "PageRank", fontsize=10)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(1.15, max(values) * 1.18))
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + axis.get_ylim()[1] * 0.025,
                f"{value:.2f}×",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.6)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
