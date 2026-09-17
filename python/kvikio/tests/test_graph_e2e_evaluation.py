# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


ROOT = Path(__file__).parents[3]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(policy: str, *, algorithm: str = "bfs", result_hash: int = 19):
    return {
        "algorithm": algorithm,
        "policy_mode": policy,
        "repeat_id": 1,
        "execution_order": 0,
        "groute_enabled": policy != "kvikio_threshold",
        "dispatch": "NATIVE_KVIKIO_THRESHOLD"
        if policy == "kvikio_threshold"
        else "GROUTE_POLICY",
        "selected_policy": {
            "workload": "GENERAL",
            "path": "GPU_DIRECT",
            "cache": "BYPASS",
            "submit": "DIRECT",
        },
        "cache": {"hits": 0, "misses": 0, "storage_bytes": 0, "admitted_regions": 0},
        "shaping": {"physical_requests": 0, "submitted_bytes": 0},
        "vertex_count": 8,
        "edge_count": 16,
        "iterations": 3,
        "visited_vertices": 8,
        "processed_edges": 16,
        "rank_sum": 0.0,
        "logical_requests": 8,
        "logical_bytes": 128,
        "logical_trace_hash": 7,
        "result_hash": result_hash,
        "algorithm_seconds": 2.0 if policy == "kvikio_threshold" else 1.0,
        "job_seconds": 2.1,
        "teps": 8.0,
    }


def test_graph_e2e_summary_validates_and_computes_speedup(tmp_path):
    module = _load_script("summarize_graph_e2e.py")
    for policy in ("kvikio_threshold", "auto", "gds_direct"):
        (tmp_path / f"e2e_bfs_{policy}_r1.json").write_text(
            json.dumps(_result(policy)), encoding="utf-8"
        )
    rows = module.load_rows(tmp_path)
    module.validate_correctness(rows)
    summary = module.summarize(rows)
    auto = next(row for row in summary if row["policy_mode"] == "auto")
    assert auto["speedup_vs_kvikio_threshold"] == 2.0
    assert auto["fraction_of_best_forced"] == 1.0


def test_graph_e2e_summary_rejects_different_bfs_result(tmp_path):
    module = _load_script("summarize_graph_e2e.py")
    (tmp_path / "e2e_bfs_kvikio_threshold_r1.json").write_text(
        json.dumps(_result("kvikio_threshold")), encoding="utf-8"
    )
    (tmp_path / "e2e_bfs_auto_r1.json").write_text(
        json.dumps(_result("auto", result_hash=20)), encoding="utf-8"
    )
    rows = module.load_rows(tmp_path)
    try:
        module.validate_correctness(rows)
    except SystemExit as error:
        assert "result_hash differs" in str(error)
    else:
        raise AssertionError("mismatched BFS result was accepted")


def test_phase_auto_is_not_counted_as_a_forced_policy(tmp_path):
    module = _load_script("summarize_graph_e2e.py")
    for policy in ("kvikio_threshold", "auto", "auto_phase", "gds_direct"):
        row = _result(policy)
        if policy == "auto_phase":
            row["algorithm_seconds"] = 0.25
        (tmp_path / f"e2e_bfs_{policy}_r1.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
    summary = module.summarize(module.load_rows(tmp_path))
    auto = next(row for row in summary if row["policy_mode"] == "auto")
    assert auto["fraction_of_best_forced"] == 1.0
    phase = next(row for row in summary if row["policy_mode"] == "auto_phase")
    assert phase["speedup_vs_kvikio_threshold"] == 8.0


def test_graph_executor_uses_long_lived_handle_and_double_buffer():
    source = (ROOT / "scripts" / "graph" / "groute_graph_e2e.cu").read_text(
        encoding="utf-8"
    )
    assert "kvikio::FileHandle file(edge_path, \"r\")" in source
    assert "std::array<Slot*, 2>" in source
    assert "file_header_bytes + request.edge_begin" in source
    assert "std::sort(frontier.begin(), frontier.end())" in source
    assert "std::sort(active.begin(), active.end())" in source
    assert "file.pread_batch(reads, reinterpret_cast<CUstream>(stream_)" in source


def test_graph_e2e_summary_preserves_cache_timing_and_legacy_results(tmp_path):
    module = _load_script("summarize_graph_e2e.py")
    baseline = _result("kvikio_threshold")
    measured = _result("host_cache")
    measured["cache"].update(
        hits=128,
        lookup_wait_ns=100,
        lookup_ns=200,
        storage_read_ns=300,
        copy_submit_ns=400,
        completion_wait_ns=500,
        copy_completions=4,
        batch_calls=4,
        batch_cache_reads=128,
        pinned_bypasses=0,
    )
    for row in (baseline, measured):
        (tmp_path / f'e2e_bfs_{row["policy_mode"]}_r1.json').write_text(
            json.dumps(row), encoding="utf-8"
        )
    rows = module.load_rows(tmp_path)
    module.validate_correctness(rows)
    assert rows[0]["cache_batch_reads"] == 128 or rows[1]["cache_batch_reads"] == 128
    assert sorted(row["cache_lookup_ns"] for row in rows) == [0, 200]
    assert sorted(row["cache_completion_wait_ns"] for row in rows) == [0, 500]


def test_pagerank_vector_comparison_checks_every_vertex(tmp_path):
    module = _load_script("compare_pagerank_ranks.py")
    for policy, ranks in (
        ("host_direct", (0.25, 0.5, 0.75, 1.0)),
        ("host_cache", (0.25, 0.5, 0.75, 1.00001)),
    ):
        row = _result(policy, algorithm="pagerank")
        row["vertex_count"] = 4
        (tmp_path / f"e2e_pagerank_{policy}_r1.json").write_text(json.dumps(row))
        (tmp_path / f"e2e_pagerank_{policy}_r1.ranks.f32").write_bytes(
            struct.pack("<4f", *ranks)
        )
    comparisons = module.compare_results(tmp_path, "host_direct", 1e-5, 1e-4)
    assert len(comparisons) == 1
    assert comparisons[0]["passed"]
    scaled = module.compare_results(tmp_path, "host_direct", "auto", 1e-3)
    assert scaled[0]["atol"] == 1e-3 / 4
    assert scaled[0]["passed"]
    (tmp_path / "e2e_pagerank_host_cache_r1.ranks.f32").write_bytes(
        struct.pack("<4f", 0.25, 0.5, 0.75, 1.5)
    )
    comparisons = module.compare_results(tmp_path, "host_direct", 1e-5, 1e-4)
    assert comparisons[0]["out_of_tolerance_vertices"] == 1
    assert not comparisons[0]["passed"]


def test_pagerank_vector_comparison_rejects_missing_or_wrong_trace(tmp_path):
    module = _load_script("compare_pagerank_ranks.py")
    for policy in ("host_direct", "host_cache"):
        row = _result(policy, algorithm="pagerank")
        row["vertex_count"] = 1
        (tmp_path / f"e2e_pagerank_{policy}_r1.json").write_text(json.dumps(row))
        (tmp_path / f"e2e_pagerank_{policy}_r1.ranks.f32").write_bytes(
            struct.pack("<f", 0.25)
        )
    (tmp_path / "e2e_pagerank_host_cache_r1.ranks.f32").unlink()
    try:
        module.compare_results(tmp_path, "host_direct", 1e-5, 1e-4)
    except ValueError as error:
        assert "missing rank vector" in str(error)
    else:
        raise AssertionError("missing PageRank data was accepted")

    (tmp_path / "e2e_pagerank_host_cache_r1.ranks.f32").write_bytes(
        struct.pack("<f", 0.25)
    )
    path = tmp_path / "e2e_pagerank_host_cache_r1.json"
    row = json.loads(path.read_text())
    row["logical_trace_hash"] = 999
    path.write_text(json.dumps(row))
    try:
        module.compare_results(tmp_path, "host_direct", 1e-5, 1e-4)
    except ValueError as error:
        assert "logical_trace_hash" in str(error)
    else:
        raise AssertionError("mismatched PageRank trace was accepted")
