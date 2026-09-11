# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cupy")

from kvikio.benchmarks.design1_policy import _offsets  # noqa: E402


def test_random_cold_profile_uses_distinct_regions():
    offsets = _offsets("random_cold_small", 64, 4096, 128 * 1024**2)

    assert len({offset // (1024**2) for offset in offsets}) == 64


def test_unaligned_profile_is_mergeable_without_region_reuse():
    offsets = _offsets("adjacent_unaligned_small", 64, 4096, 128 * 1024**2)

    assert all(offset % 4096 == 3 for offset in offsets)
    assert len({offset // (1024**2) for offset in offsets}) == 64
    assert sum(right == left + 4096 for left, right in zip(offsets, offsets[1:])) == 32
