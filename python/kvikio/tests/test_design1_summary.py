# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_design1_summary_preserves_warmup_boundary_metrics(tmp_path):
    result = {
        "case": "random_hot_small",
        "policy_mode": "host_cache",
        "profile_requests": 64,
        "warmup_requests": 4,
        "warmup_admitted_regions": 1,
        "warmup_storage_bytes": 128 * 1024,
        "cache_entries_before_measurement": 2,
        "iops": 100.0,
        "logical_mib_per_second": 1.0,
        "batch_latency_us": {"p99": 10.0},
        "selected_policy": {
            "workload": "REUSE_DOMINATED",
            "path": "HOST_MEDIATED",
            "cache": "ADMIT",
            "submit": "DIRECT",
        },
        "host_cache_delta": {
            "hits": 1024,
            "misses": 0,
            "admitted_regions": 0,
            "admission_bypasses": 0,
            "storage_bytes": 0,
        },
        "request_shaping_delta": {"physical_requests": 0},
    }
    (tmp_path / "design1_random_hot_small_r1.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    repo_root = Path(__file__).parents[3]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_design1_policy.py"),
            "--result-root",
            str(tmp_path),
        ],
        check=True,
    )

    with (tmp_path / "raw_results.csv").open(newline="", encoding="utf-8") as f:
        raw = next(csv.DictReader(f))
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary = next(csv.DictReader(f))

    assert raw["warmup_admitted_regions"] == "1"
    assert raw["policy_mode"] == "host_cache"
    assert raw["warmup_storage_bytes"] == str(128 * 1024)
    assert raw["cache_entries_before_measurement"] == "2"
    assert summary["warmup_admitted_regions_median"] == "1"
    assert summary["policy_mode"] == "host_cache"
    assert summary["warmup_storage_bytes_median"] == str(128 * 1024)
    assert summary["cache_entries_before_measurement_median"] == "2"
