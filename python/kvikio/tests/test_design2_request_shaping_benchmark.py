# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

cupy = pytest.importorskip("cupy")

from kvikio.benchmarks.design2_request_shaping import (  # noqa: E402
    latency_summary,
    measured_offsets,
    verify_wave,
)


def test_latency_summary():
    result = latency_summary([1_000, 2_000, 3_000, 4_000])

    assert result["count"] == 4
    assert result["mean"] == pytest.approx(2.5)
    assert result["p50"] == pytest.approx(2.5)
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)
    assert result["max"] == pytest.approx(4.0)


def test_latency_summary_empty():
    assert latency_summary([]) == {
        "count": 0,
        "mean": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }


def test_verify_wave_checks_every_buffer():
    offsets = measured_offsets(0, 4, 4096, 4, 1)
    buffers = [
        cupy.asarray(
            ((numpy.arange(4096, dtype=numpy.uint64) + offset) % 251).astype(
                numpy.uint8
            )
        )
        for offset in offsets
    ]

    verify_wave(buffers, offsets, 4096)

    buffers[1][17] ^= 1
    with pytest.raises(AssertionError, match=str(offsets[1] + 17)):
        verify_wave(buffers, offsets, 4096)
