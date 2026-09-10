/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/detail/stream.hpp>
#include <kvikio/host_cache.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

class RegionAdmission::Impl {
 public:
  struct Entry {
    Entry(std::size_t lines_per_region, std::list<std::size_t>::iterator lru)
      : line_accesses(lines_per_region), lru{lru}
    {
    }

    std::vector<std::uint8_t> line_accesses;
    bool admitted{};
    std::list<std::size_t>::iterator lru;
  };

  Impl(std::size_t region_size,
       std::size_t line_size,
       std::size_t admission_threshold,
       std::size_t max_regions)
    : region_size{region_size},
      line_size{line_size},
      lines_per_region{line_size == 0 ? 0 : region_size / line_size},
      admission_threshold{admission_threshold},
      max_regions{max_regions}
  {
    KVIKIO_EXPECT(line_size > 0 && region_size >= line_size && region_size % line_size == 0,
                  "region size must contain a whole number of cache lines",
                  std::invalid_argument);
    KVIKIO_EXPECT(admission_threshold > 0 && admission_threshold <= 255,
                  "region admission threshold must be in [1, 255]",
                  std::invalid_argument);
    KVIKIO_EXPECT(max_regions > 0,
                  "region admission metadata capacity must be positive",
                  std::invalid_argument);
  }

  std::size_t region_size;
  std::size_t line_size;
  std::size_t lines_per_region;
  std::size_t admission_threshold;
  std::size_t max_regions;
  mutable std::mutex mutex;
  std::unordered_map<std::size_t, Entry> entries;
  std::list<std::size_t> lru;
  RegionAdmissionStats counters{};
};

RegionAdmission::RegionAdmission(std::size_t region_size,
                                 std::size_t line_size,
                                 std::size_t admission_threshold,
                                 std::size_t max_regions)
  : _impl{std::make_unique<Impl>(region_size, line_size, admission_threshold, max_regions)}
{
}

RegionAdmission::~RegionAdmission() noexcept = default;

bool RegionAdmission::should_admit(std::size_t file_offset)
{
  std::lock_guard lock{_impl->mutex};
  auto const region = file_offset / _impl->region_size;
  auto found        = _impl->entries.find(region);
  if (found == _impl->entries.end()) {
    if (_impl->entries.size() == _impl->max_regions) {
      auto const victim = _impl->lru.back();
      _impl->entries.erase(victim);
      _impl->lru.pop_back();
      ++_impl->counters.metadata_evictions;
    }
    _impl->lru.push_front(region);
    found = _impl->entries.try_emplace(region, _impl->lines_per_region, _impl->lru.begin()).first;
  } else {
    _impl->lru.splice(_impl->lru.begin(), _impl->lru, found->second.lru);
    found->second.lru = _impl->lru.begin();
  }

  auto& entry = found->second;
  if (!entry.admitted) {
    auto const line = (file_offset % _impl->region_size) / _impl->line_size;
    auto& accesses  = entry.line_accesses[line];
    if (accesses < std::numeric_limits<std::uint8_t>::max()) { ++accesses; }
    if (accesses >= _impl->admission_threshold) {
      entry.admitted = true;
      ++_impl->counters.admitted_regions;
    }
  }

  if (!entry.admitted) { ++_impl->counters.bypassed_requests; }
  return entry.admitted;
}

void RegionAdmission::clear() noexcept
{
  std::lock_guard lock{_impl->mutex};
  _impl->entries.clear();
  _impl->lru.clear();
}

RegionAdmissionStats RegionAdmission::stats() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  auto ret            = _impl->counters;
  ret.tracked_regions = _impl->entries.size();
  return ret;
}

class HostCache::Impl {
 public:
  struct Entry {
    std::size_t slot{};
    std::size_t valid_bytes{};
    std::list<std::size_t>::iterator lru;
  };

  Impl(std::size_t capacity,
       std::size_t line_size,
       std::size_t max_io_size,
       std::size_t region_size,
       std::size_t admission_threshold,
       std::size_t max_regions)
    : capacity{line_size == 0 ? 0 : capacity - capacity % line_size},
      line_size{line_size},
      max_io_size{max_io_size},
      admission{region_size, line_size, admission_threshold, max_regions}
  {
    KVIKIO_EXPECT(this->capacity >= line_size,
                  "host cache capacity must hold at least one cache line",
                  std::invalid_argument);
    free_slots.reserve(this->capacity / line_size);
    for (std::size_t i = 0; i < this->capacity / line_size; ++i) {
      free_slots.push_back(this->capacity / line_size - 1 - i);
    }
  }

  void ensure_storage()
  {
    if (storage != nullptr) { return; }
    allocation_context = get_context_from_pointer(active_device_pointer);
    PushAndPopContext context{allocation_context};
    storage = allocator.allocate(capacity);
  }

  void release_storage() noexcept
  {
    if (storage == nullptr) { return; }
    try {
      PushAndPopContext context{allocation_context};
      allocator.deallocate(storage, capacity);
    } catch (...) {
    }
    storage = nullptr;
  }

  std::size_t capacity;
  std::size_t line_size;
  std::size_t max_io_size;
  RegionAdmission admission;
  mutable std::mutex mutex;
  CudaPageAlignedPinnedAllocator allocator;
  void* storage{};
  void* active_device_pointer{};
  CUcontext allocation_context{};
  std::vector<std::size_t> free_slots;
  std::unordered_map<std::size_t, Entry> entries;
  std::list<std::size_t> lru;
  HostCacheStats counters{};
};

HostCache::HostCache(std::size_t capacity,
                     std::size_t line_size,
                     std::size_t max_io_size,
                     std::size_t region_size,
                     std::size_t admission_threshold,
                     std::size_t max_regions)
  : _impl{std::make_unique<Impl>(
      capacity, line_size, max_io_size, region_size, admission_threshold, max_regions)}
{
}

HostCache::~HostCache() noexcept { _impl->release_storage(); }

bool HostCache::eligible(std::size_t size, std::size_t file_offset) const noexcept
{
  if (size == 0 || size > _impl->max_io_size) { return false; }
  auto const line_offset = file_offset % _impl->line_size;
  return size <= _impl->line_size - line_offset;
}

std::optional<std::size_t> HostCache::read(int fd_direct_off,
                                           int fd_direct_on,
                                           void* dev_ptr_base,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           std::size_t dev_ptr_offset)
{
  if (!eligible(size, file_offset)) { return std::nullopt; }

  std::lock_guard lock{_impl->mutex};
  _impl->active_device_pointer = dev_ptr_base;

  auto const line_offset = file_offset - file_offset % _impl->line_size;
  auto const in_line      = file_offset - line_offset;
  auto found              = _impl->entries.find(line_offset);

  if (found == _impl->entries.end()) {
    ++_impl->counters.misses;
    if (!_impl->admission.should_admit(file_offset)) {
      _impl->counters.admission_bypass_bytes += size;
      return std::nullopt;
    }
    _impl->ensure_storage();
    if (_impl->free_slots.empty()) {
      auto const victim = _impl->lru.back();
      auto const entry  = _impl->entries.find(victim);
      _impl->free_slots.push_back(entry->second.slot);
      _impl->entries.erase(entry);
      _impl->lru.pop_back();
      ++_impl->counters.evictions;
    }

    auto const slot = _impl->free_slots.back();
    _impl->free_slots.pop_back();
    auto* line = static_cast<char*>(_impl->storage) + slot * _impl->line_size;
    ssize_t bytes_read{};
    try {
      bytes_read = posix_host_io<IOOperationType::READ, PartialIO::YES>(
        fd_direct_off, line, _impl->line_size, line_offset, fd_direct_on);
      KVIKIO_EXPECT(bytes_read > 0, "host cache read reached end of file");
    } catch (...) {
      _impl->free_slots.push_back(slot);
      throw;
    }
    _impl->counters.storage_bytes += static_cast<std::uint64_t>(bytes_read);
    _impl->lru.push_front(line_offset);
    found = _impl->entries
              .emplace(line_offset,
                       Impl::Entry{slot, static_cast<std::size_t>(bytes_read), _impl->lru.begin()})
              .first;
  } else {
    ++_impl->counters.hits;
    _impl->lru.splice(_impl->lru.begin(), _impl->lru, found->second.lru);
    found->second.lru = _impl->lru.begin();
  }

  if (in_line >= found->second.valid_bytes) { return std::size_t{0}; }
  auto const bytes_to_copy = std::min(size, found->second.valid_bytes - in_line);
  auto* src = static_cast<char*>(_impl->storage) + found->second.slot * _impl->line_size + in_line;
  CUcontext context = get_context_from_pointer(dev_ptr_base);
  PushAndPopContext context_guard{context};
  auto stream = StreamCachePerThreadAndContext::get();
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::cuda_memcpy_async(
    convert_void2deviceptr(dev_ptr_base) + dev_ptr_offset,
    convert_void2deviceptr(src),
    bytes_to_copy,
    stream));
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
  _impl->counters.h2d_bytes += bytes_to_copy;
  return bytes_to_copy;
}

void HostCache::clear() noexcept
{
  std::lock_guard lock{_impl->mutex};
  _impl->entries.clear();
  _impl->lru.clear();
  _impl->free_slots.clear();
  for (std::size_t i = 0; i < _impl->capacity / _impl->line_size; ++i) {
    _impl->free_slots.push_back(_impl->capacity / _impl->line_size - 1 - i);
  }
  _impl->admission.clear();
}

HostCacheStats HostCache::stats() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  auto ret                 = _impl->counters;
  auto const admission     = _impl->admission.stats();
  ret.admitted_regions     = admission.admitted_regions;
  ret.admission_bypasses   = admission.bypassed_requests;
  ret.metadata_evictions   = admission.metadata_evictions;
  ret.tracked_regions      = admission.tracked_regions;
  return ret;
}

}  // namespace kvikio::detail
