#!/usr/bin/env python3
"""Select configurations on tuning data and summarize held-out phase-one results."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_records(paths):
    records = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("results.jsonl"))
        for candidate in candidates:
            with candidate.open() as source:
                records.extend(json.loads(line) for line in source if line.strip())
    if not records:
        raise RuntimeError("no results.jsonl records found")
    return records


def mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else 0.0


def select(args):
    records = read_records(args.inputs)
    records = [row for row in records if row.get("split") == args.split]
    if not records:
        raise RuntimeError(f"no records for split {args.split!r}")
    grouped = defaultdict(list)
    for row in records:
        if not row["policy"].startswith("dual_"):
            continue
        key = (
            row["fill_ns"], row["cache_lines"], row["policy"],
            row["frequency_threshold"], row["momentum_threshold"],
            row["momentum_window"],
        )
        grouped[key].append(row)

    summaries = []
    for key, rows in grouped.items():
        fill_ns, cache_lines, policy, f_threshold, m_threshold, window = key
        summaries.append({
            "fill_ns": fill_ns,
            "cache_lines": cache_lines,
            "policy": policy,
            "frequency_threshold": f_threshold,
            "momentum_threshold": m_threshold,
            "momentum_window": window,
            "net": mean(rows, "net_saved_ns_per_request"),
            "pollution": mean(rows, "counterfactual_pollution_miss_rate"),
            "decision": mean(rows, "median_decision_ns_per_request"),
        })
    if not summaries:
        raise RuntimeError("no dual-score configurations available for selection")

    by_cost_capacity = defaultdict(list)
    for item in summaries:
        by_cost_capacity[(item["fill_ns"], item["cache_lines"])].append(item)

    selected = {}
    for key, candidates in by_cost_capacity.items():
        best_net = max(item["net"] for item in candidates)
        tolerance = abs(best_net) * 0.01
        near_best = [item for item in candidates if item["net"] >= best_net - tolerance]
        selected[key] = min(
            near_best,
            key=lambda item: (
                item["pollution"], item["decision"],
                -item["frequency_threshold"], -item["momentum_threshold"], item["policy"],
            ),
        )

    if args.fixed_from_fill is not None:
        capacities = sorted({capacity for _, capacity in selected})
        fills = sorted({fill for fill, _ in selected})
        fixed = {}
        for capacity in capacities:
            source = selected.get((args.fixed_from_fill, capacity))
            if source is None:
                raise RuntimeError(
                    f"no selection for fill={args.fixed_from_fill}, cache={capacity}")
            for fill in fills:
                fixed[(fill, capacity)] = {**source, "fill_ns": fill}
        selected = fixed

    payload = {
        "selected_on": args.split,
        "selection_rule": "within_1pct_best_net_then_min_pollution_then_min_decision_cost",
        "fixed_from_fill": args.fixed_from_fill,
        "selections": [
            {key: value for key, value in item.items()
             if key not in {"net", "pollution", "decision"}}
            for _, item in sorted(selected.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Selection saved to {args.output}")


def report(args):
    records = read_records(args.inputs)
    grouped = defaultdict(list)
    fields = (
        "split", "fill_ns", "cache_lines", "policy",
        "frequency_threshold", "momentum_threshold", "momentum_window",
    )
    for row in records:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    summary = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary.append(dict(zip(fields, key),
            samples=len(rows),
            net_saved_ns_per_request=mean(rows, "net_saved_ns_per_request"),
            counterfactual_pollution_miss_rate=mean(
                rows, "counterfactual_pollution_miss_rate"),
            victim_reaccess_misses=mean(rows, "victim_reaccess_misses"),
            two_ref_second_hit_ratio=mean(rows, "two_ref_second_hit_ratio"),
            scan_hot_evictions=mean(rows, "scan_hot_evictions"),
            hit_ratio=mean(rows, "hit_ratio"),
        ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Summary saved to {csv_path}; matplotlib unavailable, plots skipped")
        return
    plot_rows = [row for row in summary if row["split"] in
                 {"final_in_domain", "final_heldout_trace"}]
    for metric, filename, ylabel in (
        ("net_saved_ns_per_request", "cost_sensitivity_net_saved.pdf",
         "Modeled net saved (ns/request)"),
        ("counterfactual_pollution_miss_rate", "cost_sensitivity_pollution.pdf",
         "Counterfactual pollution miss rate"),
    ):
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        series = defaultdict(list)
        for row in plot_rows:
            series[(row["policy"], row["cache_lines"])].append(row)
        for (policy, capacity), rows in sorted(series.items()):
            rows.sort(key=lambda row: row["fill_ns"])
            axis.plot([row["fill_ns"] for row in rows], [row[metric] for row in rows],
                      marker="o", label=f"{policy}, {capacity} lines")
        axis.set_xlabel("fill_ns")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename)
        plt.close(fig)
    print(f"Report saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    select_parser.add_argument("--split", default="tuning")
    select_parser.add_argument("--fixed-from-fill", type=int)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.set_defaults(function=select)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.set_defaults(function=report)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
