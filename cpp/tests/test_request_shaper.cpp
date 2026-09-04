/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <vector>

#include <kvikio/request_shaper.hpp>

#include <gtest/gtest.h>

namespace {

kvikio::detail::LogicalRead request(std::size_t id,
                                    std::size_t file_offset,
                                    std::size_t size,
                                    std::size_t file_size = 1024 * 1024)
{
  return {.id             = id,
          .dev_ptr_base   = reinterpret_cast<void*>(0x100000 + id * 0x10000),
          .dev_ptr_offset = 0,
          .file_offset    = file_offset,
          .size           = size,
          .file_size      = file_size,
          .context        = nullptr};
}

}  // namespace

TEST(RequestPlannerTest, merges_adjacent_unaligned_reads)
{
  kvikio::detail::RequestPlanner planner;
  auto plans =
    planner.plan({request(0, 4100, 2000), request(1, 6200, 2000), request(2, 8300, 2000)});

  ASSERT_EQ(plans.size(), 1);
  EXPECT_TRUE(plans[0].shaped);
  EXPECT_EQ(plans[0].file_offset, 4096);
  EXPECT_EQ(plans[0].size, 8192);
  EXPECT_EQ(plans[0].logical_bytes, 6000);
  ASSERT_EQ(plans[0].slices.size(), 3);
  EXPECT_EQ(plans[0].slices[0].staging_offset, 4);
  EXPECT_EQ(plans[0].slices[1].staging_offset, 2104);
  EXPECT_EQ(plans[0].slices[2].staging_offset, 4204);
}

TEST(RequestPlannerTest, keeps_distant_reads_independent)
{
  kvikio::detail::RequestPlanner planner;
  auto plans = planner.plan({request(0, 3, 4096), request(1, 64 * 1024 + 3, 4096)});

  ASSERT_EQ(plans.size(), 2);
  EXPECT_FALSE(plans[0].shaped);
  EXPECT_FALSE(plans[1].shaped);
}

TEST(RequestPlannerTest, rejects_excessive_amplification)
{
  kvikio::detail::RequestPlanner planner;
  auto plans = planner.plan({request(0, 1, 1), request(1, 2, 1)});

  ASSERT_EQ(plans.size(), 2);
  EXPECT_FALSE(plans[0].shaped);
  EXPECT_FALSE(plans[1].shaped);
}

TEST(RequestPlannerTest, does_not_align_past_end_of_file)
{
  kvikio::detail::RequestPlanner planner;
  auto plans = planner.plan({request(0, 9000, 400, 10000), request(1, 9400, 400, 10000)});

  ASSERT_EQ(plans.size(), 2);
  EXPECT_FALSE(plans[0].shaped);
  EXPECT_FALSE(plans[1].shaped);
}

TEST(RequestPlannerTest, keeps_large_pattern4_style_reads_direct)
{
  kvikio::detail::RequestPlanner planner;
  constexpr std::size_t mib = 1024 * 1024;
  auto plans                = planner.plan(
    {request(0, 3, 100 * mib, 256 * mib), request(1, 100 * mib + 3, 100 * mib, 256 * mib)});

  ASSERT_EQ(plans.size(), 2);
  EXPECT_FALSE(plans[0].shaped);
  EXPECT_FALSE(plans[1].shaped);
}
