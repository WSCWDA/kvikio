// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "interface.hpp"
#include "../common.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <kvikio/file_handle.hpp>

namespace gustann {
namespace {

char const* workload_name(kvikio::WorkloadClass value)
{
  switch (value) {
    case kvikio::WorkloadClass::UNKNOWN: return "UNKNOWN";
    case kvikio::WorkloadClass::SEQUENTIAL_SCAN: return "SEQUENTIAL_SCAN";
    case kvikio::WorkloadClass::REUSE_DOMINATED: return "REUSE_DOMINATED";
    case kvikio::WorkloadClass::GENERAL: return "GENERAL";
    case kvikio::WorkloadClass::FINE_GRAINED: return "FINE_GRAINED";
  }
  return "UNKNOWN";
}

char const* path_name(kvikio::IOPath value)
{
  return value == kvikio::IOPath::GPU_DIRECT ? "GPU_DIRECT" : "HOST_MEDIATED";
}

char const* cache_name(kvikio::CachePolicy value)
{
  return value == kvikio::CachePolicy::ADMIT ? "ADMIT" : "BYPASS";
}

char const* submit_name(kvikio::SubmitPolicy value)
{
  return value == kvikio::SubmitPolicy::SHAPED ? "SHAPED" : "DIRECT";
}

class GRouteLoader final : public IndexLoader {
  using Clock = std::chrono::steady_clock;

  struct Pending {
    std::vector<std::future<std::size_t>> reads;
    Clock::time_point start{};
    bool active{};
  };

 public:
  GRouteLoader(char const* filename, int ctx_cnt)
    : file_{std::make_unique<kvikio::FileHandle>(filename, "r")}, pending_(ctx_cnt)
  {
    if (ctx_cnt <= 0) { throw std::invalid_argument("GRoute ctx count must be positive"); }
    INFO("GRouteLoader initialized: ctx_cnt={}", ctx_cnt);
  }

  void submit_task(std::vector<IoRequest> const& requests, int, int ctx_id) override
  {
    auto& pending = pending_.at(static_cast<std::size_t>(ctx_id));
    if (pending.active) { throw std::logic_error("GRoute context submitted while active"); }
    pending.reads.clear();
    pending.reads.reserve(requests.size());
    pending.start  = Clock::now();
    pending.active = !requests.empty();
    for (auto const& [block, destination] : requests) {
      auto const offset = (static_cast<std::size_t>(block) + 1) * PAGE_SIZE;
      pending.reads.emplace_back(file_->pread(destination,
                                               PAGE_SIZE,
                                               offset,
                                               PAGE_SIZE,
                                               0,
                                               false));
    }
  }

  bool poll_task(int ctx_id) override
  {
    auto& pending = pending_.at(static_cast<std::size_t>(ctx_id));
    if (!pending.active) { return true; }
    for (auto& read : pending.reads) {
      if (read.wait_for(std::chrono::seconds{0}) != std::future_status::ready) { return false; }
    }
    for (auto& read : pending.reads) {
      if (read.get() != PAGE_SIZE) { throw std::runtime_error("short GRoute index-page read"); }
    }
    auto const elapsed = std::chrono::duration<double, std::micro>(Clock::now() - pending.start);
    {
      std::lock_guard lock{latency_mutex_};
      batch_latency_us_.push_back(elapsed.count());
    }
    pending.active = false;
    return true;
  }

  uint8_t* create_buffer(int64_t size) override
  {
    void* buffer{};
    auto const status = cudaMalloc(&buffer, static_cast<std::size_t>(size));
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string{"cudaMalloc: "} + cudaGetErrorString(status));
    }
    return static_cast<uint8_t*>(buffer);
  }

  void destroy_buffer(uint8_t* buffer) override
  {
    if (buffer != nullptr) { cudaFree(buffer); }
  }

  bool is_device_buffer() const override { return true; }

  void log_latency(std::vector<double> const&) override
  {
    std::vector<double> latency;
    {
      std::lock_guard lock{latency_mutex_};
      latency = batch_latency_us_;
    }
    std::sort(latency.begin(), latency.end());
    auto percentile = [&latency](double q) {
      if (latency.empty()) { return 0.0; }
      auto const index = std::min(latency.size() - 1,
                                  static_cast<std::size_t>(q * latency.size()));
      return latency[index];
    };
    auto const context = file_->io_context_snapshot();
    auto const cache   = file_->host_cache_stats();
    auto const shaping = file_->request_shaper_stats();
    std::cout << "[GROUTE_STATS] {\"workload\":\"" << workload_name(context.workload)
              << "\",\"path\":\"" << path_name(context.policy.path)
              << "\",\"cache\":\"" << cache_name(context.policy.cache)
              << "\",\"submit\":\"" << submit_name(context.policy.submit)
              << "\",\"logical_requests\":" << context.stats.request_count
              << ",\"io_batch_latency_us_p50\":" << percentile(0.50)
              << ",\"io_batch_latency_us_p95\":" << percentile(0.95)
              << ",\"io_batch_latency_us_p99\":" << percentile(0.99)
              << ",\"cache_hits\":" << cache.hits
              << ",\"cache_misses\":" << cache.misses
              << ",\"admitted_regions\":" << cache.admitted_regions
              << ",\"admission_bypasses\":" << cache.admission_bypasses
              << ",\"physical_requests\":" << shaping.physical_requests
              << ",\"submitted_bytes\":" << shaping.submitted_bytes << "}" << std::endl;
  }

 private:
  std::unique_ptr<kvikio::FileHandle> file_;
  std::vector<Pending> pending_;
  std::mutex latency_mutex_;
  std::vector<double> batch_latency_us_;
};

}  // namespace

std::shared_ptr<IndexLoader> create_groute_loader(char const* filename, int ctx_cnt)
{
  return std::make_shared<GRouteLoader>(filename, ctx_cnt);
}

}  // namespace gustann
