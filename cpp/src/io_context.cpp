/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include <kvikio/io_context.hpp>

namespace kvikio {
namespace {

constexpr std::uint64_t streaming_ratio_percent = 75;
constexpr std::uint64_t reuse_ratio_percent     = 25;
constexpr std::size_t io_size_threshold         = 64 * 1024;

std::uint64_t region_hash(std::size_t region) noexcept
{
  return static_cast<std::uint64_t>(region) * 11400714819323198485ull;
}

}  // namespace

IOContext::IOContext(bool host_cache_available,
                     bool request_shaping_available,
                     ShapingConfig shaping_config) noexcept
  : _host_cache_available{host_cache_available},
    _request_shaping_available{request_shaping_available},
    _shaping_config{shaping_config}
{
  if (_shaping_config.alignment == 0) { _shaping_config.alignment = 4096; }
  reset();
}

void IOContext::observe(std::size_t size, std::size_t file_offset) noexcept
{
  observe(nullptr, size, file_offset, 0);
}

void IOContext::observe(void const* dev_ptr_base,
                        std::size_t size,
                        std::size_t file_offset,
                        std::size_t dev_ptr_offset) noexcept
{
  if (size == 0) { return; }

  auto const request_index = _request_count.fetch_add(1, std::memory_order_relaxed) + 1;
  _requested_bytes.fetch_add(size, std::memory_order_relaxed);
  if (request_index > profile_request_limit) { return; }

  auto const request_end    = file_offset + size;
  auto const previous_end   = _last_request_end.exchange(request_end, std::memory_order_relaxed);
  auto const previous_begin = _last_request_begin.exchange(file_offset, std::memory_order_relaxed);
  if (previous_end == file_offset) { _sequential_requests.fetch_add(1, std::memory_order_relaxed); }
  if (previous_end != std::numeric_limits<std::size_t>::max()) {
    bool const overlaps = file_offset < previous_end && request_end > previous_begin;
    auto const gap =
      file_offset >= previous_end
        ? file_offset - previous_end
        : (previous_begin >= request_end ? previous_begin - request_end : std::size_t{0});
    if (overlaps || gap <= _shaping_config.max_merge_gap) {
      _mergeable_requests.fetch_add(1, std::memory_order_relaxed);
    }
  }

  auto const alignment = _shaping_config.alignment;
  if (file_offset % alignment != 0) {
    _file_offset_unaligned.fetch_add(1, std::memory_order_relaxed);
  }
  if (size % alignment != 0) { _size_unaligned.fetch_add(1, std::memory_order_relaxed); }
  if (dev_ptr_base != nullptr &&
      (reinterpret_cast<std::uintptr_t>(dev_ptr_base) + dev_ptr_offset) % alignment != 0) {
    _device_address_unaligned.fetch_add(1, std::memory_order_relaxed);
  }
  auto const aligned_begin = file_offset - file_offset % alignment;
  auto const aligned_end =
    request_end % alignment == 0 ? request_end : request_end + alignment - request_end % alignment;
  _aligned_physical_bytes.fetch_add(aligned_end - aligned_begin, std::memory_order_relaxed);

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
  auto const requests   = _profiled_requests.load(std::memory_order_acquire);
  auto const bytes      = _profiled_bytes.load(std::memory_order_relaxed);
  auto const sequential = _sequential_requests.load(std::memory_order_relaxed);
  auto const repeated   = _repeated_regions.load(std::memory_order_relaxed);
  auto const mergeable  = _mergeable_requests.load(std::memory_order_relaxed);
  if (requests == 0) { return; }

  auto const average_size = bytes / requests;
  if (repeated * 100 >= requests * reuse_ratio_percent && average_size <= io_size_threshold) {
    _workload.store(WorkloadClass::REUSE_DOMINATED, std::memory_order_relaxed);
    _path.store(IOPath::HOST_MEDIATED, std::memory_order_relaxed);
    _cache.store(_host_cache_available ? CachePolicy::ADMIT : CachePolicy::BYPASS,
                 std::memory_order_relaxed);
  } else if (sequential * 100 >= requests * streaming_ratio_percent &&
             average_size >= io_size_threshold) {
    _workload.store(WorkloadClass::SEQUENTIAL_SCAN, std::memory_order_relaxed);
    _path.store(IOPath::GPU_DIRECT, std::memory_order_relaxed);
    _cache.store(CachePolicy::BYPASS, std::memory_order_relaxed);
  } else if (average_size < io_size_threshold && _request_shaping_available &&
             static_cast<double>(mergeable) / static_cast<double>(requests) >=
               _shaping_config.min_mergeable_ratio) {
    _workload.store(WorkloadClass::FINE_GRAINED, std::memory_order_relaxed);
    _path.store(IOPath::GPU_DIRECT, std::memory_order_relaxed);
    _cache.store(CachePolicy::BYPASS, std::memory_order_relaxed);
    _submit.store(SubmitPolicy::SHAPED, std::memory_order_relaxed);
  } else if (average_size < io_size_threshold) {
    _workload.store(WorkloadClass::FINE_GRAINED, std::memory_order_relaxed);
    _path.store(IOPath::HOST_MEDIATED, std::memory_order_relaxed);
    _cache.store(CachePolicy::BYPASS, std::memory_order_relaxed);
  } else {
    _workload.store(WorkloadClass::GENERAL, std::memory_order_relaxed);
    _path.store(IOPath::GPU_DIRECT, std::memory_order_relaxed);
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
  auto const requests         = _profiled_requests.load(std::memory_order_acquire);
  auto const bytes            = _profiled_bytes.load(std::memory_order_relaxed);
  auto const sequential       = _sequential_requests.load(std::memory_order_relaxed);
  auto const repeated         = _repeated_regions.load(std::memory_order_relaxed);
  auto const file_unaligned   = _file_offset_unaligned.load(std::memory_order_relaxed);
  auto const size_unaligned   = _size_unaligned.load(std::memory_order_relaxed);
  auto const device_unaligned = _device_address_unaligned.load(std::memory_order_relaxed);
  auto const mergeable        = _mergeable_requests.load(std::memory_order_relaxed);
  auto const aligned_bytes    = _aligned_physical_bytes.load(std::memory_order_relaxed);
  return {
    _request_count.load(std::memory_order_relaxed),
    _requested_bytes.load(std::memory_order_relaxed),
    requests,
    requests == 0 ? 0.0 : static_cast<double>(bytes) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(sequential) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(repeated) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(file_unaligned) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(size_unaligned) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(device_unaligned) / static_cast<double>(requests),
    requests == 0 ? 0.0 : static_cast<double>(mergeable) / static_cast<double>(requests),
    bytes == 0 ? 0.0 : static_cast<double>(aligned_bytes) / static_cast<double>(bytes),
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
  _file_offset_unaligned.store(0, std::memory_order_relaxed);
  _size_unaligned.store(0, std::memory_order_relaxed);
  _device_address_unaligned.store(0, std::memory_order_relaxed);
  _mergeable_requests.store(0, std::memory_order_relaxed);
  _aligned_physical_bytes.store(0, std::memory_order_relaxed);
  _last_request_begin.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
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
