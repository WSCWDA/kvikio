/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

#include <kvikio/io_context.hpp>

#include <gtest/gtest.h>

TEST(IOContextTest, sequential_scan_reads_select_gds)
{
  kvikio::IOContext context{true};
  constexpr std::size_t size = 64 * 1024;
  for (std::size_t i = 0; i < kvikio::IOContext::profile_request_limit; ++i) {
    context.observe(size, i * size);
  }

  auto const snapshot = context.snapshot();
  EXPECT_TRUE(snapshot.stats.profile_complete);
  EXPECT_EQ(snapshot.workload, kvikio::WorkloadClass::SEQUENTIAL_SCAN);
  EXPECT_EQ(snapshot.policy.path, kvikio::IOPath::GPU_DIRECT);
  EXPECT_EQ(snapshot.policy.cache, kvikio::CachePolicy::BYPASS);
  EXPECT_EQ(snapshot.policy.submit, kvikio::SubmitPolicy::DIRECT);
}

TEST(IOContextTest, repeated_small_reads_select_host_cache)
{
  kvikio::IOContext context{true};
  constexpr std::size_t size = 4 * 1024;
  for (std::size_t i = 0; i < kvikio::IOContext::profile_request_limit; ++i) {
    context.observe(size, (i % 2) * 64 * 1024);
  }

  auto const snapshot = context.snapshot();
  EXPECT_EQ(snapshot.workload, kvikio::WorkloadClass::REUSE_DOMINATED);
  EXPECT_EQ(snapshot.policy.path, kvikio::IOPath::HOST_MEDIATED);
  EXPECT_EQ(snapshot.policy.cache, kvikio::CachePolicy::ADMIT);
  EXPECT_GT(snapshot.stats.repeated_region_ratio, 0.9);
}

TEST(IOContextTest, cold_small_reads_select_host_without_admission)
{
  kvikio::IOContext context{true};
  constexpr std::size_t size = 4 * 1024;
  for (std::size_t i = 0; i < kvikio::IOContext::profile_request_limit; ++i) {
    context.observe(size, i * 64 * 1024);
  }

  auto const snapshot = context.snapshot();
  EXPECT_EQ(snapshot.workload, kvikio::WorkloadClass::FINE_GRAINED);
  EXPECT_EQ(snapshot.policy.path, kvikio::IOPath::HOST_MEDIATED);
  EXPECT_EQ(snapshot.policy.cache, kvikio::CachePolicy::BYPASS);
}

TEST(IOContextTest, general_large_random_reads_select_gds)
{
  kvikio::IOContext context{true};
  constexpr std::size_t size = 128 * 1024;
  for (std::size_t i = 0; i < kvikio::IOContext::profile_request_limit; ++i) {
    auto const region = (i * 17) % kvikio::IOContext::profile_request_limit;
    context.observe(size, region * 4 * size);
  }

  auto const snapshot = context.snapshot();
  EXPECT_EQ(snapshot.workload, kvikio::WorkloadClass::GENERAL);
  EXPECT_EQ(snapshot.policy.path, kvikio::IOPath::GPU_DIRECT);
  EXPECT_EQ(snapshot.policy.cache, kvikio::CachePolicy::BYPASS);
}

TEST(IOContextTest, policy_is_stable_after_profile)
{
  kvikio::IOContext context{false};
  constexpr std::size_t size = 64 * 1024;
  for (std::size_t i = 0; i < kvikio::IOContext::profile_request_limit; ++i) {
    context.observe(size, i * size);
  }
  auto const selected = context.policy();

  for (std::size_t i = 0; i < 256; ++i) {
    context.observe(4 * 1024, 0);
  }

  EXPECT_EQ(context.policy().path, selected.path);
  EXPECT_EQ(context.policy().cache, selected.cache);
  EXPECT_EQ(context.stats().request_count, kvikio::IOContext::profile_request_limit + 256);
}
