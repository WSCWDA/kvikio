# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "scripts" / "graph" / "generate_graph_trace.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_graph_trace", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_bam(path: Path, values: list[int], header_count: int | None = None):
    with path.open("wb") as output:
        np.asarray(
            [len(values) if header_count is None else header_count, 0], dtype="<u8"
        ).tofile(output)
        np.asarray(values, dtype="<u8").tofile(output)


def test_bfs_trace_follows_real_frontiers(tmp_path):
    generator = _load_generator()
    prefix = tmp_path / "tiny.bel"
    _write_bam(Path(str(prefix) + ".col"), [0, 2, 3, 4, 4])
    _write_bam(Path(str(prefix) + ".dst"), [1, 2, 3, 3])
    trace = tmp_path / "bfs.trace"
    args = generator.build_parser().parse_args(
        [
            "--algorithm",
            "bfs",
            "--graph",
            str(prefix),
            "--output",
            str(trace),
            "--source",
            "0",
        ]
    )
    metadata = generator.generate(args)

    # Level 0 touches one page; vertices 1 and 2 each issue their own logical
    # adjacency request to that page. Vertex 3 has no outgoing edges.
    assert np.fromfile(trace, dtype="<u8").tolist() == [0, 0, 0]
    assert metadata["level_requests"] == [1, 2, 0]
    assert metadata["visited_vertices"] == 4
    assert metadata["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()


def test_pagerank_trace_preserves_cross_iteration_reuse(tmp_path):
    generator = _load_generator()
    prefix = tmp_path / "tiny.bel"
    # Deliberately use the common converter's incorrect .col header. Payload
    # length and CSR offsets remain authoritative.
    _write_bam(Path(str(prefix) + ".col"), [0, 2, 3], header_count=3)
    _write_bam(Path(str(prefix) + ".dst"), [1, 0, 1])
    trace = tmp_path / "pagerank.trace"
    args = generator.build_parser().parse_args(
        [
            "--algorithm",
            "pagerank",
            "--graph",
            str(prefix),
            "--output",
            str(trace),
            "--iterations",
            "3",
        ]
    )
    metadata = generator.generate(args)

    assert np.fromfile(trace, dtype="<u8").tolist() == [0, 0, 0, 0, 0, 0]
    assert metadata["iteration_requests"] == [2, 2, 2]
    assert metadata["active_vertices"] == [2, 2, 2]
    assert metadata["iterations"] == 3


def test_graph_scripts_keep_requested_bam_baseline():
    script = (REPO_ROOT / "scripts" / "run_bam_graph_baseline.sh").read_text()
    assert "--impl_type 4 --memalloc 2" in script
    matrix = (REPO_ROOT / "scripts" / "run_graph_trace_matrix.sh").read_text()
    expected = "kvikio_threshold auto host_direct host_cache gds_direct gds_shaped"
    assert expected in matrix
