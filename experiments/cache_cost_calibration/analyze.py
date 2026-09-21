#!/usr/bin/env python3
"""Aggregate a cache-cost calibration without printing results to the terminal."""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from groute_experiment_output import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="groute_cache_cost_calibration_* directory or results.jsonl")
    args = parser.parse_args()
    source = args.input / "results.jsonl" if args.input.is_dir() else args.input
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    successful = [row for row in rows if row.get("status") == "completed"]
    grouped = defaultdict(list)
    for row in successful:
        grouped[row["mode"]].append(row)

    required = ("hit", "fill", "host_bypass")
    missing = [mode for mode in required if not grouped[mode]]
    if missing:
        raise RuntimeError(f"missing successful modes: {missing}")

    summary = {}
    for mode, records in sorted(grouped.items()):
        summary[mode] = {
            "successful_repeats": len(records),
            "p50_ns": round(statistics.median(r["latency_ns"]["p50"] for r in records)),
            "p99_ns": round(statistics.median(r["latency_ns"]["p99"] for r in records)),
            "p50_ns_runs": [r["latency_ns"]["p50"] for r in records],
        }

    hit = summary["hit"]["p50_ns"]
    fill = summary["fill"]["p50_ns"]
    break_even = {}
    for mode in ("host_bypass", "gds_bypass"):
        if mode not in summary:
            continue
        bypass = summary[mode]["p50_ns"]
        saving = bypass - hit
        extra_fill = max(fill - bypass, 0)
        break_even[mode] = {
            "hit_ns": hit,
            "fill_ns": fill,
            "bypass_ns": bypass,
            "saving_per_hit_ns": saving,
            "extra_fill_ns": extra_fill,
            "minimum_future_hits": (
                None if saving <= 0 else math.ceil(extra_fill / saving)
            ),
            "cache_hit_is_beneficial": saving > 0,
        }

    print(json.dumps({
        "kind": "cache_cost_analysis",
        "source": str(source),
        "failed_records": len(rows) - len(successful),
        "per_mode": summary,
        "break_even": break_even,
    }), flush=True)


if __name__ == "__main__":
    run_experiment(main, "cache_cost_analysis")
