/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>

#include <kvikio/io_context.hpp>

namespace kvikio {
namespace {

constexpr std::uint64_t streaming_ratio_percent = 75;
constexpr std::uint64_t reuse_ratio_percent     = 25;
constexpr std::size_t small_io_threshold        = 64 * 1024;

std::uint64_t region_hash(std::size_t region) noexcept
{
  return static_cast<std::uint64_t>(region) * 11400714819323198485ull;
}

}  // namespace

IOContext::IOContext(bool host_cache_available) noexcept
  : _host_cache_available{host_cache_available}
{
  reset();
}

void IOContext::observe(std::size_t size, std::size_t file_offset) noexcept
{
  if (size == 0) { return; }

  auto const request_index = _request_count.fetch_add(1, std::memory_order_relaxed) + 1;
  _requested_bytes.fetch_add(size, std::memory_order_relaxed);
  if (request_index > profile_request_limit) { return; }

  auto const previous_end = _last_request_end.exchange(file_offset + size, std::memory_order_relaxed);
  if (previous_end == file_offset) {
    _sequential_requests.fetch_add(1, std::memory_order_relaxed);
  }

  auto const hash       = region_hash(file_offset / region_size);
  auto const word_index = static_cast<std::size_t>((hash >> 6) % region_filter_len);
  auto const mask       = std::uint64_t{1} << (hash & 63);
  auto const previous   = _seen_regions[word_index].fetch_or(mask, std::memory_order_relaxed);
  if ((previous & mask) != 0) { _repeated_regions.fetch_add(1, std::memory_order_relaxed); }

  _profiled_bytes.fetch_add(size, std::memory_order_relaxed);
  auto const completed = _profiled_requests.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (completed == profile_request_limit) { classify(); }
}

void IOContext::classify() noexcept
{
  auto const requests  = _profiled_requests.load(std::memory_order_acquire);
  auto const bytes     = _profiled_bytes.load(std::memory_order_relaxed);
  auto const sequential = _sequential_requests.load(std::memory_order_relaxed);
  auto const repeated   = _repeated_regions.load(std::memory_order_relaxed);
  if (requests == 0) { return; }

  auto const average_size = bytes / requests;
  if (repeated * 100 >= requests * reuse_ratio_percent && average_size <= small_io_threshold) {
    _workload.store(WorkloadClass::REUSE_DOMINATED, std::memory_order_relaxed);
    _path.store(IOPath::HOST_MEDIATED, std::memory_order_relaxed);
    _cache.store(_host_cache_available ? CachePolicy::ADMIT : CachePolicy::BYPASS,
                 std::memory_order_relaxed);
  } else if (sequential * 100 >= requests * streaming_ratio_percent &&
             average_size >= small_io_threshold) {
    _workload.store(WorkloadClass::STREAMING, std::memory_order_relaxed);
    _path.store(IOPath::GPU_DIRECT, std::memory_order_relaxed);
    _cache.store(CachePolicy::BYPASS, std::memory_order_relaxed);
  } else {
    _workload.store(WorkloadClass::GENERAL, std::memory_order_relaxed);
    _path.store(average_size < small_io_threshold ? IOPath::HOST_MEDIATED : IOPath::GPU_DIRECT,
                std::memory_order_relaxed);
    _cache.store(CachePolicy::BYPASS, std::memory_order_relaxed);
  }
  _profile_complete.store(true, std::memory_order_release);
}

IOPolicy IOContext::policy() const noexcept
{
  return {_path.load(std::memory_order_relaxed),
          _cache.load(std::memory_order_relaxed),
          _submit.load(std::memory_order_relaxed)};
}

WorkloadClass IOContext::workload() const noexcept
{
  return _workload.load(std::memory_order_relaxed);
}

RuntimeStats IOContext::stats() const noexcept
{
  auto const requests   = _profiled_requests.load(std::memory_order_acquire);
  auto const bytes      = _profiled_bytes.load(std::memory_order_relaxed);
  auto const sequential = _sequential_requests.load(std::memory_order_relaxed);
  auto const repeated   = _repeated_regions.load(std::memory_order_relaxed);
  return {_request_count.load(std::memory_order_relaxed),
          _requested_bytes.load(std::memory_order_relaxed),
          requests,
          requests == 0 ? 0.0 : static_cast<double>(bytes) / static_cast<double>(requests),
          requests == 0 ? 0.0 : static_cast<double>(sequential) / static_cast<double>(requests),
          requests == 0 ? 0.0 : static_cast<double>(repeated) / static_cast<double>(requests),
          profile_complete()};
}

IOContextSnapshot IOContext::snapshot() const noexcept
{
  return {workload(), policy(), stats()};
}

bool IOContext::profile_complete() const noexcept
{
  return _profile_complete.load(std::memory_order_acquire);
}

void IOContext::reset() noexcept
{
  _request_count.store(0, std::memory_order_relaxed);
  _requested_bytes.store(0, std::memory_order_relaxed);
  _profiled_requests.store(0, std::memory_order_relaxed);
  _profiled_bytes.store(0, std::memory_order_relaxed);
  _sequential_requests.store(0, std::memory_order_relaxed);
  _repeated_regions.store(0, std::memory_order_relaxed);
  _last_request_end.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
  for (auto& word : _seen_regions) {
    word.store(0, std::memory_order_relaxed);
  }
  _workload.store(WorkloadClass::UNKNOWN, std::memory_order_relaxed);
  _path.store(IOPath::GPU_DIRECT, std::memory_order_relaxed);
  _cache.store(_host_cache_available ? CachePolicy::ADMIT : CachePolicy::BYPASS,
               std::memory_order_relaxed);
  _submit.store(SubmitPolicy::DIRECT, std::memory_order_relaxed);
  _profile_complete.store(false, std::memory_order_release);
}

}  // namespace kvikio
