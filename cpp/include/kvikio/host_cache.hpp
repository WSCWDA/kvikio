/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <kvikio/error.hpp>

namespace kvikio {

/** @brief Snapshot of per-file host-cache counters. */
struct HostCacheStats {
  std::uint64_t hits{};
  std::uint64_t misses{};
  std::uint64_t evictions{};
  std::uint64_t storage_bytes{};
  std::uint64_t h2d_bytes{};
};

namespace detail {

/**
 * @brief Small-read cache backed by page-aligned, CUDA-registered host memory.
 *
 * The cache is owned by one FileHandle. Requests must fit within a single cache line.
 * Its implementation is intentionally hidden to keep FileHandle's ABI surface small.
 */
class HostCache {
 public:
  HostCache(std::size_t capacity, std::size_t line_size, std::size_t max_io_size);
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

  void clear() noexcept;
  [[nodiscard]] HostCacheStats stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace detail
}  // namespace kvikio
