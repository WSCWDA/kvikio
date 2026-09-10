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

}  // namespace
