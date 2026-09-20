#!/usr/bin/env python3
"""Replay dynamic traces against admission-policy ablations without GPU or SSD I/O."""

import argparse
import json
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


def synthetic_traces(requests, seed):
    rng = random.Random(seed)

    stable = []
    weights = [1.0 / (i + 1) ** 1.1 for i in range(128)]
    population = list(range(128))
    for line in rng.choices(population, weights=weights, k=requests):
        stable.append((line, 0, int(line < 8)))

    shift = []
    phase_size = requests // 3
    for phase, base in enumerate((0, 8, 0)):
        count = phase_size if phase < 2 else requests - len(shift)
        for i in range(count):
            if rng.random() < 0.85:
                line = base + rng.randrange(8)
                hot = 1
            else:
                line = 1000 + phase * requests + i
                hot = 0
            shift.append((line, phase, hot))

    scan_mix = []
    for i in range(requests):
        if i % 3 == 1:
            scan_mix.append((10000 + i, 0, 0))
        else:
            scan_mix.append((0 if i % 3 == 0 else 1, 0, 1))

    burst = []
    next_burst = 20000
    while len(burst) < requests:
        if rng.random() < 0.65:
            burst.append((rng.randrange(4), 0, 1))
        else:
            # A two-reference burst is cold in the oracle: it never returns
            # after the second adjacent access and should not occupy the cache.
            burst.extend([(next_burst, 0, 0), (next_burst, 0, 0)])
            next_burst += 1
    return {
        "stable_zipf": stable[:requests],
        "phase_shift_aba": shift,
        "hot_scan_mix": scan_mix,
        "short_burst": burst[:requests],
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
            except ValueError:  # Permit one or more textual header lines.
                continue
            result.append((value // LINE if unit == "bytes" else value, 0, -1))
            if limit and len(result) >= limit:
                break
    if not result:
        raise ValueError("trace contains no numeric offsets")
    return {path.stem: result}


def write_trace(path, events):
    with path.open("w") as output:
        output.writelines(f"{line} {phase} {hot}\n" for line, phase, hot in events)


def compile_replay(root, binary):
    subprocess.run([
        "g++", "-O3", "-std=c++17", "-pthread",
        "-I" + str(root / "cpp/include"),
        str(root / "cpp/src/line_admission.cpp"),
        str(root / "experiments/frequency_momentum_ablation/replay.cpp"),
        "-o", str(binary),
    ], check=True)


def replay(binary, trace, args, policy, frequency_threshold, momentum_threshold):
    command = [
        str(binary), str(trace), policy, str(args.cache_lines),
        str(args.frequency_bytes), str(args.frequency_window), str(frequency_threshold),
        str(args.momentum_bytes), str(args.momentum_window), str(momentum_threshold),
        str(LINE), str(args.hit_ns), str(args.fill_ns), str(args.bypass_ns),
    ]
    runs = []
    for _ in range(args.repeats):
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        runs.append(json.loads(completed.stdout))
    result = runs[0]
    result["decision_ns_per_request_runs"] = [r["decision_ns_per_request"] for r in runs]
    result["median_decision_ns_per_request"] = statistics.median(
        result["decision_ns_per_request_runs"])
    result.update(frequency_threshold=frequency_threshold,
                  momentum_threshold=momentum_threshold)
    return result


def mark_pareto(records, labeled):
    def dominates(left, right):
        common = (left["net_saved_ns"] >= right["net_saved_ns"] and
                  left["pollution_misses"] <= right["pollution_misses"] and
                  left["hot_hit_ratio" if labeled else "hit_ratio"] >=
                  right["hot_hit_ratio" if labeled else "hit_ratio"])
        strict = (left["net_saved_ns"] > right["net_saved_ns"] or
                  left["pollution_misses"] < right["pollution_misses"] or
                  left["hot_hit_ratio" if labeled else "hit_ratio"] >
                  right["hot_hit_ratio" if labeled else "hit_ratio"])
        if labeled:
            common = common and (
                left["mean_detection_delay_requests"] <= right["mean_detection_delay_requests"])
            strict = strict or (
                left["mean_detection_delay_requests"] < right["mean_detection_delay_requests"])
        return common and strict

    for record in records:
        record["pareto"] = not any(
            other is not record and dominates(other, record) for other in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=30000)
    parser.add_argument("--cache-lines", type=int, default=16)
    parser.add_argument("--frequency-bytes", type=int, default=8192)
    parser.add_argument("--momentum-bytes", type=int, default=1024)
    parser.add_argument("--frequency-window", type=int, default=32768,
                        help="requests per complete frequency-sketch aging sweep")
    parser.add_argument("--momentum-window", type=int, default=512,
                        help="requests per complete momentum-sketch aging sweep")
    parser.add_argument("--frequency-thresholds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--momentum-thresholds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--repeats", type=int, default=5,
                        help="repeat only for CPU-overhead timing; policy outcomes are deterministic")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260920, 20260921, 20260922, 20260923, 20260924],
                        help="synthetic-trace seeds (default: five independent seeds)")
    parser.add_argument("--seed", type=int,
                        help="deprecated single-seed shortcut; overrides --seeds")
    parser.add_argument("--hit-ns", type=int, default=10000)
    parser.add_argument("--fill-ns", type=int, default=200000)
    parser.add_argument("--bypass-ns", type=int, default=80000)
    parser.add_argument("--trace", type=Path,
                        help="optional real trace: first column is an offset or line number")
    parser.add_argument("--trace-unit", choices=("bytes", "lines"), default="bytes")
    parser.add_argument("--trace-column", type=int, default=0,
                        help="zero-based whitespace/CSV column containing offset or line")
    parser.add_argument("--trace-limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.requests, args.cache_lines, args.frequency_window,
           args.momentum_window, args.repeats) <= 0:
        parser.error("request, cache, window and repeat values must be positive")
    if args.frequency_bytes < 64 or args.frequency_bytes % 64:
        parser.error("frequency-bytes must be a multiple of 64")
    if args.momentum_bytes < 64 or args.momentum_bytes % 64:
        parser.error("momentum-bytes must be a multiple of 64")
    thresholds = args.frequency_thresholds + args.momentum_thresholds
    if any(value < 1 or value > 15 for value in thresholds):
        parser.error("thresholds must be in [1, 15]")
    if args.trace_column < 0:
        parser.error("trace-column must be non-negative")
    if min(args.hit_ns, args.fill_ns, args.bypass_ns) < 0:
        parser.error("cost-model values must be non-negative")

    seeds = [args.seed] if args.seed is not None else args.seeds
    if not seeds:
        parser.error("at least one seed is required")
    root = ROOT
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        binary = temporary / "frequency_momentum_replay"
        compile_replay(root, binary)
        if args.trace:
            seed_runs = [(None, load_real_trace(args.trace, args.trace_unit,
                                                args.trace_limit, args.trace_column))]
        else:
            seed_runs = [(seed, synthetic_traces(args.requests, seed)) for seed in seeds]
        for seed, traces in seed_runs:
            for trace_name, events in traces.items():
                trace_path = temporary / f"{trace_name}_{seed}.trace"
                write_trace(trace_path, events)
                configs = [("cache_all", args.frequency_thresholds[0],
                            args.momentum_thresholds[0])]
                configs += [("frequency", f, args.momentum_thresholds[0])
                            for f in args.frequency_thresholds]
                configs += [("momentum", args.frequency_thresholds[0], m)
                            for m in args.momentum_thresholds]
                configs += [("hybrid", f, m) for f in args.frequency_thresholds
                            for m in args.momentum_thresholds]
                configs += [("dual_score", f, m) for f in args.frequency_thresholds
                            for m in args.momentum_thresholds]
                records = [replay(binary, trace_path, args, *config) for config in configs]
                mark_pareto(records, labeled=events[0][2] >= 0)
                for record in records:
                    record.update(kind="frequency_momentum_ablation", trace=trace_name,
                                  cache_lines=args.cache_lines,
                                  frequency_bytes=args.frequency_bytes,
                                  frequency_window=args.frequency_window,
                                  momentum_bytes=args.momentum_bytes,
                                  momentum_window=args.momentum_window,
                                  seed=seed)
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    run_experiment(main, "frequency_momentum_ablation")
