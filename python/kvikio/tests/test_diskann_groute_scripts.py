# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import struct
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "summarize_diskann_groute.py"
SPEC = importlib.util.spec_from_file_location("summarize_diskann_groute", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


@pytest.mark.parametrize(("data_type", "element_bytes"), (("float", 4), ("uint8", 1)))
def test_query_count(tmp_path, data_type, element_bytes):
    path = tmp_path / "query.vecs"
    dimension = 8
    record = struct.pack("<I", dimension) + bytes(dimension * element_bytes)
    path.write_bytes(record * 7)

    assert SUMMARY.query_count(path, data_type) == 7


def test_parse_groute_log(tmp_path):
    stats = {
        "workload": "FINE_GRAINED",
        "path": "HOST_MEDIATED",
        "cache": "ADMIT",
        "submit": "DIRECT",
        "cache_hits": 17,
        "physical_requests": 23,
    }
    path = tmp_path / "diskann_groute_r1.log"
    path.write_text(
        "[REPORT] LAT0 1.5\n"
        "[REPORT] LAT1 2.5\n"
        "[REPORT] Time 2.0\n"
        "[REPORT] IO 42\n"
        f"[GROUTE_STATS] {json.dumps(stats)}\n"
        "[REPORT] RECALL: 0.97\n",
        encoding="utf-8",
    )

    row = SUMMARY.parse_log(path, total_queries=100)

    assert row["mode"] == "groute"
    assert row["qps"] == 50
    assert row["mean_thread_latency_ms"] == 2
    assert row["recall"] == pytest.approx(0.97)
    assert row["cache_hits"] == 17
