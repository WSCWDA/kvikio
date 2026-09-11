# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cupy")

from kvikio.benchmarks.design1_policy import (  # noqa: E402
    FORCED_POLICIES,
    POLICY_MODES,
    _control_page_cache,
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


@pytest.mark.parametrize(
    "case,io_size",
    [
        ("sequential_large", 128 * 1024),
        ("random_cold_small", 4096),
        ("random_hot_small", 4096),
        ("adjacent_unaligned_small", 4096),
    ],
)
def test_measurement_trace_is_reproducible_and_seeded(case, io_size):
    args = (case, 64, io_size, 128 * 1024**2, 32)

    first = _measurement_offsets(*args, trace_seed=17)
    assert first == _measurement_offsets(*args, trace_seed=17)
    assert first != _measurement_offsets(*args, trace_seed=18)


def test_page_cache_none_records_no_eviction(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"data")

    result = _control_page_cache(path, "none")

    assert result["mode"] == "none"
    assert result["file_evicted"] is False
    assert result["global_dropped"] is False


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
