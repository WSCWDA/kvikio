/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <kvikio/io_context.hpp>

#include <benchmark/benchmark.h>

namespace {

constexpr std::size_t request_size        = 64 * 1024;
constexpr std::size_t request_stride      = 64 * 1024;
constexpr std::size_t gds_threshold       = 16 * 1024;
constexpr std::size_t baseline_requests   = kvikio::IOContext::profile_request_limit;
constexpr std::size_t threshold_requests  = 4096;
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
 * Build a runtime request-size stream for the static-threshold benchmark.
 * The stream is deliberately generated at runtime (outside the timed region)
 * so the compiler cannot constant-fold the threshold decision. Roughly half
 * of the requests fall below the 16-KiB threshold and half at/above it.
 */
std::vector<std::size_t> make_threshold_sizes()
{
  constexpr std::size_t candidates[] = {
    4 * 1024, 8 * 1024, 12 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024};

  std::vector<std::size_t> sizes(threshold_requests);
  std::uint64_t x = 0x9e3779b97f4a7c15ULL;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    // Small deterministic xorshift sequence; generation is not timed.
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    sizes[i] = candidates[x % (sizeof(candidates) / sizeof(candidates[0]))];
  }
  return sizes;
}

/*
 * Fixed-path lower bound: routing is already decided and the fast path only
 * reads the installed IOPolicy. This is an idealized lower bound, not a
 * realistic adaptive-routing baseline.
 */
void BM_GRouteFixedPathLowerBound(benchmark::State& state)
{
  kvikio::IOContext context{/*host_cache_available=*/true,
                            /*request_shaping_available=*/false,
                            {},
                            kvikio::PolicyMode::GDS_DIRECT};

  for (auto _ : state) {
    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < baseline_requests; ++i) {
      auto policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, baseline_requests);
  }

  state.counters["requests_per_sample"] = static_cast<double>(baseline_requests);
}

/*
 * KvikIO-style static size-threshold routing. This measures only the runtime
 * routing decision itself: compare the request size against gds_threshold and
 * materialize the selected path. No storage I/O is issued.
 */
void BM_GRouteStaticThreshold(benchmark::State& state)
{
  auto const sizes = make_threshold_sizes();

  for (auto _ : state) {
    auto const begin = clock_type::now();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      auto const size = sizes[i];
      auto path = size < gds_threshold ? kvikio::IOPath::HOST_MEDIATED : kvikio::IOPath::GPU_DIRECT;
      benchmark::DoNotOptimize(path);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, sizes.size());
  }

  state.counters["requests_per_sample"] = static_cast<double>(sizes.size());
  state.counters["threshold_KiB"]       = static_cast<double>(gds_threshold / 1024);
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
      auto policy = context.policy();
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
      auto policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, profiling_requests);
    auto profile_complete = context.profile_complete();
    benchmark::DoNotOptimize(profile_complete);
  }

  state.counters["requests_per_sample"] = static_cast<double>(profiling_requests);
}

/*
 * AUTO steady-state cost: complete profiling outside the timed region, then
 * measure the normal post-classification fast path. In the current IOContext
 * implementation observe() updates only the request/byte counters after the
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
    auto policy = context.policy();
    benchmark::DoNotOptimize(policy);
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
      auto policy = context.policy();
      benchmark::DoNotOptimize(policy);
    }
    auto const end = clock_type::now();
    set_manual_ns_per_request(state, begin, end, steady_requests);
  }

  state.counters["requests_per_sample"] = static_cast<double>(steady_requests);
}

BENCHMARK(BM_GRouteFixedPathLowerBound)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteStaticThreshold)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteTrackingOnly)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteAutoProfiling)->UseManualTime()->Unit(benchmark::kNanosecond);
BENCHMARK(BM_GRouteAutoSteadyState)->UseManualTime()->Unit(benchmark::kNanosecond);

}  // namespace

BENCHMARK_MAIN();
