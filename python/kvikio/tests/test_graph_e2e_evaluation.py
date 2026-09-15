# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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


def test_graph_executor_uses_long_lived_handle_and_double_buffer():
    source = (ROOT / "scripts" / "graph" / "groute_graph_e2e.cu").read_text(
        encoding="utf-8"
    )
    assert "kvikio::FileHandle file(edge_path, \"r\")" in source
    assert "std::array<Slot*, 2>" in source
    assert "file_header_bytes + request.edge_begin" in source
    assert "std::sort(frontier.begin(), frontier.end())" in source
    assert "std::sort(active.begin(), active.end())" in source
