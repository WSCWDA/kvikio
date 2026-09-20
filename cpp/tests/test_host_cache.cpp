/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

#include <kvikio/host_cache.hpp>

namespace {

constexpr std::size_t line_size   = 64 * 1024;
constexpr std::size_t region_size = 4 * line_size;

TEST(RegionAdmissionTest, repeated_line_promotes_entire_region)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 2, 8};

  EXPECT_FALSE(admission.should_admit(0));
  EXPECT_TRUE(admission.should_admit(4096));
  // Promotion is region-wide: a different line in the same region is now admitted.
  EXPECT_TRUE(admission.should_admit(line_size));

  auto const stats = admission.stats();
  EXPECT_EQ(stats.admitted_regions, 1);
  EXPECT_EQ(stats.bypassed_requests, 1);
  EXPECT_EQ(stats.tracked_regions, 1);
}

TEST(RegionAdmissionTest, sequential_lines_do_not_look_reused)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 2, 8};

  for (std::size_t line = 0; line < 4; ++line) {
    EXPECT_FALSE(admission.should_admit(line * line_size));
  }

  auto const stats = admission.stats();
  EXPECT_EQ(stats.admitted_regions, 0);
  EXPECT_EQ(stats.bypassed_requests, 4);
}

TEST(RegionAdmissionTest, threshold_one_is_cache_all_baseline)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 1, 8};

  EXPECT_TRUE(admission.should_admit(0));
  EXPECT_TRUE(admission.should_admit(region_size));

  auto const stats = admission.stats();
  EXPECT_EQ(stats.admitted_regions, 2);
  EXPECT_EQ(stats.bypassed_requests, 0);
}

TEST(RegionAdmissionTest, regions_are_classified_independently)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 2, 8};

  EXPECT_FALSE(admission.should_admit(0));
  EXPECT_TRUE(admission.should_admit(0));
  EXPECT_FALSE(admission.should_admit(region_size));
  EXPECT_TRUE(admission.should_admit(2 * line_size));

  auto const stats = admission.stats();
  EXPECT_EQ(stats.admitted_regions, 1);
  EXPECT_EQ(stats.tracked_regions, 2);
}

TEST(RegionAdmissionTest, metadata_is_bounded_and_lru)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 2, 2};

  EXPECT_FALSE(admission.should_admit(0));
  EXPECT_FALSE(admission.should_admit(region_size));
  // Refresh region zero, making region one the LRU victim.
  EXPECT_TRUE(admission.should_admit(0));
  EXPECT_FALSE(admission.should_admit(2 * region_size));

  auto const stats = admission.stats();
  EXPECT_EQ(stats.tracked_regions, 2);
  EXPECT_EQ(stats.metadata_evictions, 1);
  EXPECT_EQ(stats.admitted_regions, 1);
}

TEST(RegionAdmissionTest, clear_forgets_history_but_keeps_counters)
{
  kvikio::detail::RegionAdmission admission{region_size, line_size, 2, 8};

  EXPECT_FALSE(admission.should_admit(0));
  EXPECT_TRUE(admission.should_admit(0));
  admission.clear();
  EXPECT_FALSE(admission.should_admit(0));

  auto const stats = admission.stats();
  EXPECT_EQ(stats.tracked_regions, 1);
  EXPECT_EQ(stats.admitted_regions, 1);
  EXPECT_EQ(stats.bypassed_requests, 2);
}

TEST(RegionAdmissionTest, validates_configuration)
{
  EXPECT_THROW((kvikio::detail::RegionAdmission{region_size, 0, 2, 8}), std::invalid_argument);
  EXPECT_THROW(
    (kvikio::detail::RegionAdmission{line_size - 1, line_size, 2, 8}), std::invalid_argument);
  EXPECT_NO_THROW((kvikio::detail::RegionAdmission{65 * line_size, line_size, 2, 8}));
  EXPECT_THROW(
    (kvikio::detail::RegionAdmission{region_size, line_size, 0, 8}), std::invalid_argument);
  EXPECT_THROW(
    (kvikio::detail::RegionAdmission{region_size, line_size, 2, 0}), std::invalid_argument);
}

TEST(LineAdmissionTest, tracks_lines_independently_and_uses_benefit)
{
  // Fill costs 40 us more than bypass; each subsequent hit saves 30 us.
  kvikio::detail::LineAdmission admission{line_size, 64 * 1024, 256, 2, 10000, 80000};
  EXPECT_FALSE(admission.should_admit(0, 40000));
  EXPECT_FALSE(admission.should_admit(4096, 40000));
  EXPECT_TRUE(admission.should_admit(0, 40000));
  // A neighboring line in the same region does not inherit admission.
  EXPECT_FALSE(admission.should_admit(line_size, 40000));
  EXPECT_EQ(admission.admissions(), 1);
  EXPECT_EQ(admission.bypasses(), 3);
}

TEST(LineAdmissionTest, respects_fallback_path_and_clear)
{
  kvikio::detail::LineAdmission admission{line_size, 64, 4, 2, 10000, 80000};
  EXPECT_FALSE(admission.should_admit(0, 15000));
  EXPECT_FALSE(admission.should_admit(0, 15000));
  EXPECT_TRUE(admission.should_admit(0, 70000));
  EXPECT_EQ(admission.aging_steps(), 0);
  admission.clear();
  EXPECT_FALSE(admission.should_admit(0, 70000));
}

TEST(LineAdmissionTest, rejects_invalid_configuration)
{
  EXPECT_THROW((kvikio::detail::LineAdmission{line_size, 63, 1, 2, 1, 2}),
               std::invalid_argument);
  EXPECT_THROW((kvikio::detail::LineAdmission{line_size, 64, 0, 2, 1, 2}),
               std::invalid_argument);
}

TEST(LineAdmissionTest, ages_history_without_full_sketch_scan)
{
  kvikio::detail::LineAdmission admission{line_size, 64, 1, 2, 10000, 80000};
  EXPECT_FALSE(admission.should_admit(0, 70000));
  EXPECT_FALSE(admission.should_admit(0, 70000));
  EXPECT_EQ(admission.aging_steps(), 2);
}

TEST(LineAdmissionTest, cache_hits_reinforce_line_history)
{
  kvikio::detail::LineAdmission admission{line_size, 64 * 1024, 256, 2, 10000, 80000};
  EXPECT_FALSE(admission.should_admit(0, 40000));
  admission.observe_hit(0);
  EXPECT_TRUE(admission.should_admit(0, 40000));
  EXPECT_EQ(admission.admissions(), 1);
  EXPECT_EQ(admission.bypasses(), 1);
}

TEST(FrequencyMomentumAdmissionTest, momentum_detects_a_new_hot_line_first)
{
  kvikio::detail::FrequencyMomentumAdmission admission{
    line_size, 4096, 4096, 4, 64, 32, 2};
  auto first = admission.observe(7 * line_size);
  auto second = admission.observe(7 * line_size);
  EXPECT_FALSE(first.admit);
  EXPECT_TRUE(second.admit);
  EXPECT_EQ(second.signal, kvikio::detail::AdmissionSignal::momentum);
  EXPECT_EQ(second.frequency, 2);
  EXPECT_EQ(second.momentum, 2);
}

TEST(FrequencyMomentumAdmissionTest, reports_both_signals)
{
  kvikio::detail::FrequencyMomentumAdmission admission{
    line_size, 4096, 4096, 2, 64, 32, 2};
  EXPECT_FALSE(admission.observe(0).admit);
  auto second = admission.observe(0);
  EXPECT_TRUE(second.admit);
  EXPECT_EQ(second.signal, kvikio::detail::AdmissionSignal::both);
  EXPECT_EQ(admission.metadata_bytes(), 4096 + 64);
}

TEST(FrequencyMomentumAdmissionTest, estimate_does_not_modify_history)
{
  kvikio::detail::FrequencyMomentumAdmission admission{
    line_size, 4096, 4096, 3, 64, 32, 3};
  auto first = admission.observe(5 * line_size);
  auto estimate = admission.estimate(5 * line_size);
  auto repeated_estimate = admission.estimate(5 * line_size);
  auto second = admission.observe(5 * line_size);
  EXPECT_EQ(first.frequency, 1);
  EXPECT_EQ(estimate.frequency, 1);
  EXPECT_EQ(repeated_estimate.frequency, 1);
  EXPECT_EQ(second.frequency, 2);
  EXPECT_EQ(estimate.momentum, 1);
  EXPECT_EQ(repeated_estimate.momentum, 1);
  EXPECT_EQ(second.momentum, 2);
}

TEST(FrequencyMomentumAdmissionTest, window_is_independent_of_sketch_capacity)
{
  kvikio::detail::FrequencyMomentumAdmission admission{
    line_size, 128, 8, 15, 64, 8, 15};
  for (std::size_t i = 0; i < 8; ++i) { (void)admission.observe(i * line_size); }
  // Both trackers complete exactly one distributed sweep in eight observations.
  EXPECT_EQ(admission.frequency_aging_steps(), 2);
  EXPECT_EQ(admission.momentum_aging_steps(), 1);
}

TEST(FrequencyMomentumAdmissionTest, rejects_invalid_configuration)
{
  EXPECT_THROW((kvikio::detail::FrequencyMomentumAdmission{
                 line_size, 63, 8, 2, 64, 8, 2}),
               std::invalid_argument);
  EXPECT_THROW((kvikio::detail::FrequencyMomentumAdmission{
                 line_size, 64, 0, 2, 64, 8, 2}),
               std::invalid_argument);
  EXPECT_THROW((kvikio::detail::FrequencyMomentumAdmission{
                 line_size, 64, 8, 16, 64, 8, 2}),
               std::invalid_argument);
}

}  // namespace
