#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare every PageRank rank vector to a same-trace, same-repeat reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from array import array
from pathlib import Path


def compare_pair(reference: Path, candidate: Path, vertex_count: int, atol: float, rtol: float):
    expected_size = vertex_count * 4  # Graph executor writes native float32 ranks.
    if reference.stat().st_size != expected_size or candidate.stat().st_size != expected_size:
        raise ValueError(
            f"rank file length mismatch: expected {expected_size} bytes for {vertex_count} "
            f"vertices; {reference}: {reference.stat().st_size}, "
            f"{candidate}: {candidate.stat().st_size}"
        )
    max_abs = max_rel = sum_abs = sum_reference_abs = 0.0
    errors = 0
    with reference.open("rb") as left, candidate.open("rb") as right:
        for start in range(0, vertex_count, 1 << 18):
            count = min(1 << 18, vertex_count - start)
            x, y = array("f"), array("f")
            x.frombytes(left.read(count * 4))
            y.frombytes(right.read(count * 4))
            for a, b in zip(x, y):
                if not math.isfinite(a) or not math.isfinite(b):
                    errors += 1
                    continue
                diff = abs(a - b)
                max_abs = max(max_abs, diff)
                max_rel = max(max_rel, diff / max(abs(a), atol, 1e-30))
                sum_abs += diff
                sum_reference_abs += abs(a)
                if diff > atol + rtol * abs(a):
                    errors += 1
    return {
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "mean_abs_error": sum_abs / vertex_count,
        "relative_l1_error": sum_abs / max(sum_reference_abs, 1e-30),
        "out_of_tolerance_vertices": errors,
        "passed": errors == 0,
    }


def compare_results(root: Path, reference_policy: str, atol: float | str, rtol: float):
    if sys.byteorder != "little":
        raise ValueError("rank files require a little-endian host")
    if atol != "auto" and not isinstance(atol, (int, float)):
        raise ValueError("atol must be a nonnegative float or auto")
    if not all(math.isfinite(v) and v >= 0 for v in ((0.0 if atol == "auto" else atol), rtol)):
        raise ValueError("atol and rtol must be nonnegative finite values")
    samples = {}
    for path in sorted(root.glob("e2e_pagerank_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = (int(data["repeat_id"]), data["policy_mode"])
        if key in samples:
            raise ValueError(f"duplicate PageRank run: {key}")
        ranks = path.with_suffix(".ranks.f32")
        if not ranks.is_file():
            raise ValueError(f"missing rank vector for {path}: {ranks}")
        samples[key] = (data, ranks)
    if not samples:
        raise ValueError("no PageRank JSON results found")
    rows = []
    for (repeat, policy), (data, ranks) in sorted(samples.items()):
        if policy == reference_policy:
            continue
        reference = samples.get((repeat, reference_policy))
        if reference is None:
            raise ValueError(f"missing {reference_policy} rank vector in repeat {repeat}")
        baseline, baseline_ranks = reference
        for key in ("vertex_count", "iterations", "logical_trace_hash", "logical_requests"):
            if data[key] != baseline[key]:
                raise ValueError(f"repeat {repeat}, {policy}: {key} differs from {reference_policy}")
        vertex_count = int(data["vertex_count"])
        if vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        effective_atol = 1e-3 / vertex_count if atol == "auto" else atol
        comparison = compare_pair(
            baseline_ranks, ranks, vertex_count, effective_atol, rtol
        )
        rows.append(
            {
                "repeat_id": repeat,
                "reference_policy": reference_policy,
                "policy_mode": policy,
                "vertices": data["vertex_count"],
                "atol": effective_atol,
                "rtol": rtol,
                **comparison,
            }
        )
    if not rows:
        raise ValueError("at least two PageRank policies are needed for a comparison")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--reference-policy", default="host_direct")
    parser.add_argument("--atol", default="auto", help="float or auto: 1e-3/vertex_count")
    parser.add_argument("--rtol", default=1e-3, type=float)
    args = parser.parse_args()
    try:
        atol = args.atol if args.atol == "auto" else float(args.atol)
        rows = compare_results(args.result_root, args.reference_policy, atol, args.rtol)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    output = args.result_root / "pagerank_rank_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if not row["passed"]]
    print(f"Wrote {output}; {len(rows) - len(failures)}/{len(rows)} comparisons passed")
    if failures:
        raise SystemExit(
            "PageRank per-vertex tolerance exceeded: "
            + ", ".join(
                f"repeat={row['repeat_id']} policy={row['policy_mode']} "
                f"violations={row['out_of_tolerance_vertices']}"
                for row in failures
            )
        )


if __name__ == "__main__":
    main()
