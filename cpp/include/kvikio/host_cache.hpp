/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

/** @brief Snapshot of per-file host-cache counters and current occupancy. */
struct HostCacheStats {
  std::uint64_t hits{};
  std::uint64_t misses{};
  std::uint64_t evictions{};
  std::uint64_t storage_bytes{};
  std::uint64_t h2d_bytes{};
  std::uint64_t admitted_regions{};
  std::uint64_t admission_bypasses{};
  std::uint64_t admission_bypass_bytes{};
  std::uint64_t metadata_evictions{};
  std::uint64_t tracked_regions{};
  std::uint64_t cache_entries{};
  // Timings are accumulated only with KVIKIO_HOST_CACHE_PROFILE=1.
  std::uint64_t lookup_wait_ns{};
  std::uint64_t lookup_ns{};
  std::uint64_t storage_read_ns{};
  std::uint64_t copy_submit_ns{};
  std::uint64_t completion_wait_ns{};
  std::uint64_t copy_completions{};
  std::uint64_t batch_calls{};
  std::uint64_t batch_cache_reads{};
  std::uint64_t pinned_bypasses{};
};

namespace detail {

struct HostCacheReadRequest {
  void* device_ptr{};
  std::size_t size{};
  std::size_t file_offset{};
};

/** @brief Snapshot of the bounded region-admission metadata. */
struct RegionAdmissionStats {
  std::uint64_t admitted_regions{};
  std::uint64_t bypassed_requests{};
  std::uint64_t metadata_evictions{};
  std::uint64_t tracked_regions{};
};

/**
 * @brief Bounded reuse detector that promotes cache regions based on repeated cache-line access.
 *
 * Merely touching several different lines in one region is not evidence of reuse: a sequential
 * scan does exactly that. A region is therefore promoted only after one of its cache lines has
 * been observed `admission_threshold` times. Once promoted, all cache-line misses in that region
 * may enter the host cache. Region metadata is maintained in a bounded LRU table.
 */
class RegionAdmission {
 public:
  RegionAdmission(std::size_t region_size,
                  std::size_t line_size,
                  std::size_t admission_threshold,
                  std::size_t max_regions);
  RegionAdmission(RegionAdmission const&)            = delete;
  RegionAdmission& operator=(RegionAdmission const&) = delete;
  RegionAdmission(RegionAdmission&&)                 = delete;
  RegionAdmission& operator=(RegionAdmission&&)      = delete;
  ~RegionAdmission() noexcept;

  /** @brief Observe an access and return whether its region is admitted. */
  [[nodiscard]] bool should_admit(std::size_t file_offset);

  /** @brief Forget current region history without resetting cumulative counters. */
  void clear() noexcept;

  [[nodiscard]] RegionAdmissionStats stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

/**
 * @brief Small-read cache backed by page-aligned, CUDA-registered host memory.
 *
 * The cache is owned by one FileHandle. Requests must fit within a single cache line.
 * Its implementation is intentionally hidden to keep FileHandle's ABI surface small.
 */
class HostCache {
 public:
  HostCache(std::size_t capacity,
            std::size_t line_size,
            std::size_t max_io_size,
            std::size_t region_size,
            std::size_t admission_threshold,
            std::size_t max_regions);
  HostCache(HostCache const&)            = delete;
  HostCache& operator=(HostCache const&) = delete;
  HostCache(HostCache&&)                 = delete;
  HostCache& operator=(HostCache&&)      = delete;
  ~HostCache() noexcept;

  [[nodiscard]] bool eligible(std::size_t size, std::size_t file_offset) const noexcept;

  std::optional<std::size_t> read(int fd_direct_off,
                                  int fd_direct_on,
                                  void* dev_ptr_base,
                                  std::size_t size,
                                  std::size_t file_offset,
                                  std::size_t dev_ptr_offset);

  /**
   * @brief Copy eligible cached reads to GPU on one stream and synchronize once per batch.
   *
   * An empty optional means that the caller must execute the normal file read. The cache pins
   * every source line until all enqueued GPU copies have completed, including on exceptions.
   */
  std::vector<std::optional<std::size_t>> read_batch(
    int fd_direct_off,
    int fd_direct_on,
    std::vector<HostCacheReadRequest> const& requests,
    CUstream stream);

  void clear() noexcept;
  [[nodiscard]] HostCacheStats stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace detail
}  // namespace kvikio
