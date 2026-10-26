/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>

#include <kvikio/io_context.hpp>

#include <benchmark/benchmark.h>

namespace {

constexpr std::size_t request_size        = 64 * 1024;
constexpr std::size_t request_stride      = 64 * 1024;
constexpr std::size_t baseline_requests   = kvikio::IOContext::profile_request_limit;
constexpr std::size_t tracking_requests   = kvikio::IOContext::profile_request_limit - 1;
constexpr std::size_t profiling_requests  = kvikio::IOContext::profile_request_limit;
constexpr std::size_t steady_requests     = 4096;

using clock_type = std::chrono::steady_clock;

inline std::size_t offset_for(std::size_t i) noexcept { return i * request_stride; }

void set_manual_ns_per_request(benchmark::State& state,
                               clock_type::time_point begin,
                               clock_type::time_point end,
                               std::size_t requests)
{
  auto const seconds = std::chrono::duration<double>(end - begin).count();
  state.SetIterationTime(seconds / static_cast<double>(requests));
}

/*
 * Fixed-path baseline: measure only the policy lookup that the I/O fast path
 * performs after routing has already been decided. No context observation is
 * executed here.
 */
void BM_GRouteBaseline(benchmark::State& state)
{
  kvikio::IOContext context{/*host_cache_available=*/true,
                            /*request_shaping_available=*/false,
                            {},
                            kvikio::PolicyMode::GDS_DIRECT};

  for (auto _ : state) {
    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < baseline_requests; ++i) {
      auto const policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, baseline_requests);
  }

  state.counters["requests_per_sample"] = static_cast<double>(baseline_requests);
}

/*
 * Tracking-only cost: use a forced GDS policy and measure observe()+policy()
 * for the first 63 requests. Stopping before request 64 deliberately excludes
 * IOContext::classify(), isolating bounded context tracking from classification.
 * context.reset() is intentionally outside the manually timed interval.
 */
void BM_GRouteTrackingOnly(benchmark::State& state)
{
  kvikio::IOContext context{/*host_cache_available=*/true,
                            /*request_shaping_available=*/false,
                            {},
                            kvikio::PolicyMode::GDS_DIRECT};

  for (auto _ : state) {
    context.reset();

    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < tracking_requests; ++i) {
      context.observe(request_size, offset_for(i));
      auto const policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, tracking_requests);
  }

  state.counters["requests_per_sample"] = static_cast<double>(tracking_requests);
}

/*
 * AUTO profiling cost: measure the complete 64-request profiling window.
 * The 64th observation invokes classify(), so the reported average includes
 * tracking, policy lookup, and the one-time classification cost amortized over
 * the profiling window. reset() is outside the manually timed interval.
 */
void BM_GRouteAutoProfiling(benchmark::State& state)
{
  kvikio::IOContext context{/*host_cache_available=*/true,
                            /*request_shaping_available=*/false,
                            {},
                            kvikio::PolicyMode::AUTO};

  for (auto _ : state) {
    context.reset();

    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < profiling_requests; ++i) {
      context.observe(request_size, offset_for(i));
      auto const policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, profiling_requests);
    benchmark::DoNotOptimize(context.profile_complete());
  }

  state.counters["requests_per_sample"] = static_cast<double>(profiling_requests);
}

/*
 * AUTO steady-state cost: complete profiling outside the timed region, then
 * measure the normal post-classification fast path. In the current IOContext
 * implementation observe() performs only the request/byte counters after the
 * 64-request window, followed by the stable policy lookup.
 */
void BM_GRouteAutoSteadyState(benchmark::State& state)
{
  kvikio::IOContext context{/*host_cache_available=*/true,
                            /*request_shaping_available=*/false,
                            {},
                            kvikio::PolicyMode::AUTO};

  for (std::size_t i = 0; i < profiling_requests; ++i) {
    context.observe(request_size, offset_for(i));
    benchmark::DoNotOptimize(context.policy());
  }
  if (!context.profile_complete()) {
    state.SkipWithError("AUTO profiling did not complete");
    return;
  }

  std::size_t request_index = profiling_requests;
  for (auto _ : state) {
    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < steady_requests; ++i, ++request_index) {
      context.observe(request_size, offset_for(request_index));
      auto const policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, steady_requests);
  }

  state.counters["requests_per_sample"] = static_cast<double>(steady_requests);
}

BENCHMARK(BM_GRouteBaseline)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteTrackingOnly)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteAutoProfiling)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteAutoSteadyState)->UseManualTime()->Unit(benchmark::kNanosecond);

}  // namespace

BENCHMARK_MAIN();
