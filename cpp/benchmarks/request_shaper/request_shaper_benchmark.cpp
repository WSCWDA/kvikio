/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <vector>

#include <kvikio/request_shaper.hpp>

#include <benchmark/benchmark.h>

static void BM_plan_adjacent_unaligned(benchmark::State& state)
{
  auto const request_count = static_cast<std::size_t>(state.range(0));
  std::vector<kvikio::detail::LogicalRead> requests;
  requests.reserve(request_count);
  for (std::size_t i = 0; i < request_count; ++i) {
    requests.push_back(
      {.id = i, .file_offset = 3 + i * 4096, .size = 4096, .file_size = 1024 * 1024});
  }
  kvikio::detail::RequestPlanner planner;
  for (auto _ : state) {
    auto plans = planner.plan(requests);
    benchmark::DoNotOptimize(plans);
  }
  auto const plans                    = planner.plan(requests);
  state.counters["logical_requests"]  = static_cast<double>(request_count);
  state.counters["physical_requests"] = static_cast<double>(plans.size());
}
BENCHMARK(BM_plan_adjacent_unaligned)->Arg(2)->Arg(8)->Arg(16)->Arg(32);

BENCHMARK_MAIN();
