#!/usr/bin/env python3
"""Replay dynamic cache-line traces against independent admission policies."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from groute_experiment_output import run_experiment


LINE = 64 * 1024
REQUEST_CLASS = {"background": 0, "hot": 1, "scan": 2, "two_reference_burst": 3}
SCORE_POLICIES = {
    "max": "dual_max",
    "weighted": "dual_weighted",
    "multiplicative": "dual_multiplicative",
    "lexicographic": "dual_lexicographic",
}


def synthetic_traces(requests, seed):
    rng = random.Random(seed)

    stable = []
    weights = [1.0 / (i + 1) ** 1.1 for i in range(128)]
    for line in rng.choices(list(range(128)), weights=weights, k=requests):
        hot = int(line < 8)
        stable.append((line, 0, hot, REQUEST_CLASS["hot" if hot else "background"]))

    shift = []
    phase_size = requests // 3
    for phase, base in enumerate((0, 8, 0)):
        count = phase_size if phase < 2 else requests - len(shift)
        for index in range(count):
            if rng.random() < 0.85:
                shift.append((base + rng.randrange(8), phase, 1, REQUEST_CLASS["hot"]))
            else:
                shift.append((1000 + phase * requests + index, phase, 0,
                              REQUEST_CLASS["background"]))

    scan_mix = []
    for index in range(requests):
        if index % 3 == 1:
            scan_mix.append((10000 + index, 0, 0, REQUEST_CLASS["scan"]))
        else:
            scan_mix.append((0 if index % 3 == 0 else 1, 0, 1, REQUEST_CLASS["hot"]))

    burst = []
    next_burst = 20000
    while len(burst) < requests:
        if rng.random() < 0.65:
            burst.append((rng.randrange(4), 0, 1, REQUEST_CLASS["hot"]))
        else:
            pair = (next_burst, 0, 0, REQUEST_CLASS["two_reference_burst"])
            burst.extend((pair, pair))
            next_burst += 1
    return {
        "stable_zipf": stable,
        "phase_shift_aba": shift,
        "hot_scan_mix": scan_mix,
        "two_reference_burst": burst[:requests],
    }


def load_real_trace(path, unit, limit, column):
    result = []
    with path.open() as source:
        for line in source:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if column >= len(fields):
                continue
            try:
                value = int(fields[column], 0)
            except ValueError:
                continue
            result.append((value // LINE if unit == "bytes" else value, 0, -1, -1))
            if limit and len(result) >= limit:
                break
    if not result:
        raise ValueError("trace contains no numeric offsets")
    return {path.stem: result}


def write_trace(path, events):
    with path.open("w") as output:
        output.writelines(
            f"{line} {phase} {hot} {request_class}\n"
            for line, phase, hot, request_class in events
        )


def compile_replay(root, binary):
    subprocess.run([
        "g++", "-O3", "-std=c++17", "-pthread",
        "-I" + str(root / "cpp/include"),
        str(root / "cpp/src/line_admission.cpp"),
        str(root / "experiments/frequency_momentum_ablation/replay.cpp"),
        "-o", str(binary),
    ], check=True)


def replay(binary, trace, args, policy, cache_lines, momentum_window,
           fill_ns, frequency_threshold, momentum_threshold):
    command = [
        str(binary), str(trace), policy, str(cache_lines),
        str(args.frequency_bytes), str(args.frequency_window), str(frequency_threshold),
        str(args.momentum_bytes), str(momentum_window), str(momentum_threshold),
        str(LINE), str(args.hit_ns), str(fill_ns), str(args.bypass_ns),
    ]
    runs = []
    for _ in range(args.repeats):
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        runs.append(json.loads(completed.stdout))
    result = runs[0]
    result["decision_ns_per_request_runs"] = [r["decision_ns_per_request"] for r in runs]
    result["median_decision_ns_per_request"] = statistics.median(
        result["decision_ns_per_request_runs"])
    result.update(
        frequency_threshold=frequency_threshold,
        momentum_threshold=momentum_threshold,
        momentum_window=momentum_window,
        cache_lines=cache_lines,
        fill_ns=fill_ns,
    )
    return result


def mark_pareto(records, labeled):
    def dominates(left, right):
        hit_metric = "hot_hit_ratio" if labeled else "hit_ratio"
        common = (
            left["net_saved_ns"] >= right["net_saved_ns"]
            and left["counterfactual_pollution_misses"]
            <= right["counterfactual_pollution_misses"]
            and left[hit_metric] >= right[hit_metric]
        )
        strict = (
            left["net_saved_ns"] > right["net_saved_ns"]
            or left["counterfactual_pollution_misses"]
            < right["counterfactual_pollution_misses"]
            or left[hit_metric] > right[hit_metric]
        )
        return common and strict

    for record in records:
        record["pareto"] = not any(
            other is not record and dominates(other, record) for other in records)


def load_split(args):
    if not args.split:
        return args.seeds, args.traces
    if args.splits_file is None:
        raise ValueError("--split requires --splits-file")
    config = json.loads(args.splits_file.read_text())
    if args.split not in config:
        raise ValueError(f"unknown split {args.split!r}")
    selected = config[args.split]
    return selected["seeds"], selected["traces"]


def load_selections(path):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return {
        (item["fill_ns"], item["cache_lines"]): item
        for item in payload["selections"]
    }


def policy_configs(args, selection=None):
    if selection is not None:
        return [(
            selection["policy"],
            selection["frequency_threshold"],
            selection["momentum_threshold"],
        )]
    configs = []
    requested = set(args.policies)
    first_f = args.frequency_thresholds[0]
    first_m = args.momentum_thresholds[0]
    if "cache_all" in requested:
        configs.append(("cache_all", first_f, first_m))
    if "frequency" in requested:
        configs.extend(("frequency", value, first_m) for value in args.frequency_thresholds)
    if "momentum" in requested:
        configs.extend(("momentum", first_f, value) for value in args.momentum_thresholds)
    if "hybrid" in requested:
        configs.extend(("hybrid", f, m) for f in args.frequency_thresholds
                       for m in args.momentum_thresholds)
    for mode in args.score_modes:
        policy = SCORE_POLICIES[mode]
        if "dual" in requested or policy in requested:
            configs.extend((policy, f, m) for f in args.frequency_thresholds
                           for m in args.momentum_thresholds)
    return configs


def trace_manifest(events):
    digest = hashlib.sha256()
    counts = {name: 0 for name in REQUEST_CLASS}
    reverse = {value: key for key, value in REQUEST_CLASS.items()}
    for event in events:
        digest.update((" ".join(map(str, event)) + "\n").encode())
        if event[3] in reverse:
            counts[reverse[event[3]]] += 1
    return {"requests": len(events), "sha256": digest.hexdigest(), "classes": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=30000)
    parser.add_argument("--cache-lines", type=int, nargs="+", default=[16])
    parser.add_argument("--frequency-bytes", type=int, default=8192)
    parser.add_argument("--momentum-bytes", type=int, default=1024)
    parser.add_argument("--frequency-window", type=int, default=32768)
    parser.add_argument("--momentum-window", type=int, default=512)
    parser.add_argument("--momentum-windows", type=int, nargs="+")
    parser.add_argument("--frequency-thresholds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--momentum-thresholds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--score-modes", nargs="+", choices=tuple(SCORE_POLICIES),
                        default=list(SCORE_POLICIES))
    parser.add_argument("--policies", nargs="+",
                        default=["cache_all", "frequency", "momentum", "hybrid", "dual"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260920, 20260921, 20260922, 20260923, 20260924])
    parser.add_argument("--hit-ns", type=int, default=11915)
    parser.add_argument("--fill-ns", type=int, default=52784)
    parser.add_argument("--fill-ns-values", type=int, nargs="+")
    parser.add_argument("--bypass-ns", type=int, default=65977)
    parser.add_argument("--traces", nargs="+",
                        default=["stable_zipf", "phase_shift_aba",
                                 "hot_scan_mix", "two_reference_burst"])
    parser.add_argument("--splits-file", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--experiment", default="frequency_momentum_ablation")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-unit", choices=("bytes", "lines"), default="bytes")
    parser.add_argument("--trace-column", type=int, default=0)
    parser.add_argument("--trace-limit", type=int, default=0)
    args = parser.parse_args()

    momentum_windows = args.momentum_windows or [args.momentum_window]
    fill_values = args.fill_ns_values or [args.fill_ns]
    seeds, trace_names = load_split(args)
    selections = load_selections(args.selection_file)
    if min(args.requests, *args.cache_lines, args.frequency_window,
           *momentum_windows, args.repeats) <= 0:
        parser.error("request, cache, window and repeat values must be positive")
    if args.frequency_bytes < 64 or args.frequency_bytes % 64:
        parser.error("frequency-bytes must be a multiple of 64")
    if args.momentum_bytes < 64 or args.momentum_bytes % 64:
        parser.error("momentum-bytes must be a multiple of 64")
    if any(value < 1 or value > 15
           for value in args.frequency_thresholds + args.momentum_thresholds):
        parser.error("thresholds must be in [1, 15]")
    if min(args.hit_ns, args.bypass_ns, *fill_values) < 0:
        parser.error("cost-model values must be non-negative")

    manifests = {}
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        binary = temporary / "frequency_momentum_replay"
        compile_replay(ROOT, binary)
        if args.trace:
            seed_runs = [(None, load_real_trace(
                args.trace, args.trace_unit, args.trace_limit, args.trace_column))]
        else:
            seed_runs = [(seed, synthetic_traces(args.requests, seed)) for seed in seeds]
        for seed, all_traces in seed_runs:
            selected_trace_names = list(all_traces) if args.trace else trace_names
            unknown = set(selected_trace_names) - set(all_traces)
            if unknown:
                raise ValueError(f"unknown traces: {sorted(unknown)}")
            for trace_name in selected_trace_names:
                events = all_traces[trace_name]
                trace_path = temporary / f"{trace_name}_{seed}.trace"
                write_trace(trace_path, events)
                manifests[f"{trace_name}:{seed}"] = trace_manifest(events)
                for cache_lines in args.cache_lines:
                    for fill_ns in fill_values:
                        selection = selections.get((fill_ns, cache_lines)) if selections else None
                        if selections is not None and selection is None:
                            raise ValueError(
                                "selection file has no entry for "
                                f"fill_ns={fill_ns}, cache_lines={cache_lines}")
                        windows = [selection["momentum_window"]] if selection else momentum_windows
                        for momentum_window in windows:
                            configs = policy_configs(args, selection)
                            records = [replay(
                                binary, trace_path, args, policy, cache_lines,
                                momentum_window, fill_ns, f_threshold, m_threshold)
                                for policy, f_threshold, m_threshold in configs]
                            mark_pareto(records, labeled=events[0][2] >= 0)
                            for record in records:
                                record.update(
                                    kind=args.experiment,
                                    split=args.split,
                                    trace=trace_name,
                                    frequency_bytes=args.frequency_bytes,
                                    frequency_window=args.frequency_window,
                                    momentum_bytes=args.momentum_bytes,
                                    seed=seed,
                                )
                                print(json.dumps(record), flush=True)

    result_dir = os.environ.get("GROUTE_RESULT_DIR")
    if result_dir:
        (Path(result_dir) / "trace_manifest.json").write_text(
            json.dumps(manifests, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    run_experiment(main, "frequency_momentum_ablation")
