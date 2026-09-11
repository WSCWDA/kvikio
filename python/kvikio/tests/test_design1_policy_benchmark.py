# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cupy")

from kvikio.benchmarks.design1_policy import (  # noqa: E402
    FORCED_POLICIES,
    POLICY_MODES,
    _measurement_offsets,
    _profile_offsets,
)


def test_random_cold_profile_uses_distinct_regions():
    offsets = _profile_offsets(
        "random_cold_small", 64, 4096, 128 * 1024**2
    )

    assert len({offset // (1024**2) for offset in offsets}) == 64


def test_unaligned_profile_is_mergeable_without_region_reuse():
    offsets = _profile_offsets(
        "adjacent_unaligned_small", 64, 4096, 128 * 1024**2
    )

    assert all(offset % 4096 == 3 for offset in offsets)
    assert len({offset // (1024**2) for offset in offsets}) == 64
    assert sum(right == left + 4096 for left, right in zip(offsets, offsets[1:])) == 32


def test_random_cold_measurement_never_reuses_a_cache_line():
    offsets = _measurement_offsets(
        "random_cold_small", 1024, 4096, 128 * 1024**2, 32
    )

    assert len({offset // (64 * 1024) for offset in offsets}) == 1024


def test_shaped_measurement_preserves_complete_batches():
    offsets = _measurement_offsets(
        "adjacent_unaligned_small", 64, 4096, 128 * 1024**2, 32
    )

    for begin in (0, 32):
        group = offsets[begin : begin + 32]
        assert all(right == left + 4096 for left, right in zip(group, group[1:]))


def test_forced_policy_modes_cover_all_controlled_baselines():
    assert set(POLICY_MODES) == {
        "auto",
        "host_direct",
        "host_cache",
        "gds_direct",
        "gds_shaped",
    }
    assert FORCED_POLICIES == {
        "host_direct": ("HOST_MEDIATED", "BYPASS", "DIRECT"),
        "host_cache": ("HOST_MEDIATED", "ADMIT", "DIRECT"),
        "gds_direct": ("GPU_DIRECT", "BYPASS", "DIRECT"),
        "gds_shaped": ("GPU_DIRECT", "BYPASS", "SHAPED"),
    }
