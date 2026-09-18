#!/usr/bin/env python3
"""Measure line-sketch behavior and end-to-end lookup cost on fixed traces."""

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cupy as cp
import kvikio
import kvikio.defaults


LINE = 64 * 1024
SIZE = 4096

# Single-thread metadata-only measurements. Exact map is an intentionally
# optimistic baseline (it has no synchronization). Both start from empty state.
CPP_BENCH = r'''
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <kvikio/line_admission.hpp>

int main(int argc, char** argv) {
  auto const n = std::stoull(argv[1]);
  auto const sketch_bytes = std::stoull(argv[2]);
  for (unsigned hot : {0u, 1u}) {
    std::uint64_t checksum = 0;
    kvikio::detail::LineAdmission sketch(65536, sketch_bytes, 256, 2, 10000, 50000);
    auto const start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < n; ++i) {
      auto const key = hot ? i % 4 : i % 4096;
      checksum += sketch.should_admit(key * 65536, 80000);
    }
    auto const end = std::chrono::steady_clock::now();
    std::cout << (hot ? "hot" : "scan") << " sketch_ns_per_op "
              << std::chrono::duration<double, std::nano>(end - start).count() / n
              << " decisions " << checksum << '\n';

    std::unordered_map<std::uint64_t, std::uint8_t> exact;
    checksum = 0;
    auto const exact_start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < n; ++i) {
      auto const key = hot ? i % 4 : i % 4096;
      auto& count = exact[key];
      checksum += count > 0;
      if (count < 15) { ++count; }
    }
    auto const exact_end = std::chrono::steady_clock::now();
    std::cout << (hot ? "hot" : "scan") << " exact_map_ns_per_op "
              << std::chrono::duration<double, std::nano>(exact_end - exact_start).count() / n
              << " decisions " << checksum << " keys " << exact.size() << '\n';
  }
}
'''


def settings(line_admission, sketch_bytes=4096, fill_ns=50000):
    return {
        "groute_enabled": True,
        "compat_mode": kvikio.CompatMode.ON,
        "policy_mode": kvikio.PolicyMode.HOST_CACHE,
        "host_cache_line_admission": line_admission,
        "host_cache_capacity": 4 * LINE,
        "host_cache_line_size": LINE,
        "host_cache_max_io_size": SIZE,
        "host_cache_region_size": 4 * LINE,
        "host_cache_admission_threshold": 2,
        "host_cache_sketch_bytes": sketch_bytes,
        "host_cache_aging_interval": 256,
        "host_cache_hit_ns": 10000,
        "host_cache_fill_ns": fill_ns,
        "host_cache_host_bypass_ns": 80000,
        "host_cache_gds_bypass_ns": 80000,
    }


def delta(a, b):
    return {key: b[key] - a[key] for key in b if key != "cache_entries"}


def one_case(path, config, offsets):
    gpu = cp.empty(SIZE, dtype=cp.uint8)
    with kvikio.defaults.set(config):
        with kvikio.CuFile(path, "r") as handle:
            before = handle.host_cache_stats()
            start = time.perf_counter_ns()
            for offset in offsets:
                assert handle.raw_read(gpu, size=SIZE, file_offset=offset) == SIZE
            elapsed = time.perf_counter_ns() - start
            with path.open("rb") as check:
                check.seek(offsets[-1])
                assert cp.asnumpy(gpu[:64]).tobytes() == check.read(64)
            after = handle.host_cache_stats()
    return elapsed, delta(before, after), after["cache_entries"]


def metadata_bench(requests, sketch_bytes):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "line_metadata.cpp"
        binary = Path(directory) / "line_metadata"
        source.write_text(CPP_BENCH)
        subprocess.run([
            "g++", "-O3", "-std=c++17", "-pthread", "-I" + str(root / "cpp/include"),
            str(root / "cpp/src/line_admission.cpp"), str(source), "-o", str(binary),
        ], check=True)
        output = subprocess.run([str(binary), str(requests), str(sketch_bytes)],
                                capture_output=True, text=True, check=True)
        print(json.dumps({"experiment": "metadata_only", "requests": requests,
                          "sketch_bytes": sketch_bytes, "results": output.stdout.splitlines()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, type=Path, help="existing file on test SSD")
    p.add_argument("--requests", type=int, default=10000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sketch-bytes", type=int, default=4096)
    p.add_argument("--no-cpp", action="store_true", help="skip CPU-only metadata benchmark")
    args = p.parse_args()
    if args.requests <= 0 or args.repeats <= 0 or args.sketch_bytes < 64 or args.sketch_bytes % 64:
        p.error("requests/repeats must be positive; sketch-bytes must be a positive multiple of 64")
    if args.file.stat().st_size < 256 * LINE:
        p.error("test file must contain at least 256 lines (16 MiB)")

    # Functional cost gate: previous count r=1 saves 70us, below extra fill 120us;
    # r=2 saves 140us, so the third access fills.
    offsets = [0, 0, 0, 0, LINE]
    _, counts, entries = one_case(args.file, settings(True, args.sketch_bytes, 200000), offsets)
    assert (counts["misses"], counts["hits"], counts["admitted_lines"],
            counts["benefit_bypasses"], entries) == (4, 1, 1, 1, 1), counts
    print(json.dumps({"experiment": "cost_gate", "stats": counts, "cache_entries": entries}))
    if not args.no_cpp:
        metadata_bench(max(1000000, args.requests), args.sketch_bytes)

    # Same offsets for old region admission and the new line sketch.
    traces = {
        "scan": [i % 256 * LINE for i in range(args.requests)],
        "hot": [i % 4 * LINE for i in range(args.requests)],
    }
    for name, trace in traces.items():
        for mode in ("region", "line"):
            times = []
            totals = []
            for _ in range(args.repeats):
                elapsed, totals, entries = one_case(
                    args.file, settings(mode == "line", args.sketch_bytes), trace
                )
                times.append(elapsed)
            median_ns = statistics.median(times)
            operations = totals["hits"] + totals["misses"]
            print(json.dumps({
                "experiment": "overhead", "trace": name, "mode": mode,
                "profile_enabled": os.environ.get("KVIKIO_HOST_CACHE_PROFILE") == "1",
                "requests": len(trace), "repeats": args.repeats,
                "elapsed_ms_runs": [round(t / 1e6, 3) for t in times],
                "median_iops": round(len(trace) * 1e9 / median_ns, 2),
                "lookup_ns_per_request": round(totals["lookup_ns"] / operations, 2),
                "lookup_wait_ns_per_request": round(totals["lookup_wait_ns"] / operations, 2),
                "cache_entries": entries, "stats_last_run": totals,
            }))


if __name__ == "__main__":
    main()
