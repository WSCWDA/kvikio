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
        "repeat_id": 1,
        "execution_order": 3,
        "order_seed": 123,
        "trace_seed": 456,
        "trace_id": "abc123",
        "file": "/mnt/gds/data.bin",
        "file_size_bytes": 1024**3,
        "file_allocated_bytes": 1024**3,
        "working_set_bytes": 512 * 1024**2,
        "page_cache": {
            "mode": "global",
            "file_evicted": True,
            "global_dropped": True,
            "cached_kib_before": 200,
            "cached_kib_after": 100,
        },
        "num_threads": 8,
        "profile_requests": 64,
        "warmup_requests": 4,
        "warmup_admitted_regions": 1,
        "warmup_storage_bytes": 128 * 1024,
        "cache_entries_before_measurement": 2,
        "iops": 100.0,
        "logical_mib_per_second": 1.0,
        "elapsed_seconds": 1.0,
        "batch_latency_us": {"p50": 5.0, "p95": 8.0, "p99": 10.0},
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
    assert raw["repeat_id"] == "1"
    assert raw["execution_order"] == "3"
    assert raw["trace_id"] == "abc123"
    assert raw["page_cache_mode"] == "global"
    assert raw["global_cache_dropped"] == "True"
    assert raw["working_set_bytes"] == str(512 * 1024**2)
    assert raw["warmup_storage_bytes"] == str(128 * 1024)
    assert raw["cache_entries_before_measurement"] == "2"
    assert summary["warmup_admitted_regions_median"] == "1"
    assert summary["policy_mode"] == "host_cache"
    assert summary["page_cache_mode"] == "global"
    assert summary["batch_p50_us_median"] == "5.0"
    assert summary["batch_p95_us_median"] == "8.0"
    assert summary["warmup_storage_bytes_median"] == str(128 * 1024)
    assert summary["cache_entries_before_measurement_median"] == "2"


def test_design1_summary_rejects_unpaired_traces(tmp_path):
    base = {
        "case": "random_cold_small",
        "repeat_id": 1,
        "iops": 100.0,
        "logical_mib_per_second": 1.0,
        "batch_latency_us": {"p50": 5.0, "p95": 8.0, "p99": 10.0},
        "selected_policy": {
            "workload": "FINE_GRAINED",
            "path": "HOST_MEDIATED",
            "cache": "ADMIT",
            "submit": "DIRECT",
        },
        "host_cache_delta": {},
        "request_shaping_delta": {},
    }
    for policy, trace_id in (("auto", "trace-a"), ("host_cache", "trace-b")):
        result = dict(base, policy_mode=policy, trace_id=trace_id)
        (tmp_path / f"design1_{policy}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_design1_policy.py"),
            "--result-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Mismatched trace_id" in completed.stderr
