/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace kvikio {

enum class WorkloadClass : std::uint8_t { UNKNOWN, STREAMING, REUSE_DOMINATED, GENERAL };

enum class IOPath : std::uint8_t { HOST_MEDIATED, GPU_DIRECT };

enum class CachePolicy : std::uint8_t { BYPASS, ADMIT };

enum class SubmitPolicy : std::uint8_t { DIRECT, SHAPED };

struct IOPolicy {
  IOPath path{IOPath::GPU_DIRECT};
  CachePolicy cache{CachePolicy::BYPASS};
  SubmitPolicy submit{SubmitPolicy::DIRECT};
};

struct RuntimeStats {
  std::uint64_t request_count{};
  std::uint64_t requested_bytes{};
  std::uint64_t profiled_requests{};
  double average_io_size{};
  double sequential_ratio{};
  double repeated_region_ratio{};
  bool profile_complete{};
};

struct IOContextSnapshot {
  WorkloadClass workload{WorkloadClass::UNKNOWN};
  IOPolicy policy{};
  RuntimeStats stats{};
};

/**
 * @brief Per-FileHandle workload profile and stable read policy.
 *
 * The first `profile_request_limit` logical device reads form one profiling window. A policy is
 * selected once at the end of that window and reused for the rest of the handle lifetime. This
 * intentionally avoids per-request prediction and policy oscillation in the small-I/O path.
 */
class IOContext {
 public:
  static constexpr std::uint64_t profile_request_limit = 64;

  explicit IOContext(bool host_cache_available = false) noexcept;
  IOContext(IOContext const&)            = delete;
  IOContext& operator=(IOContext const&) = delete;
  IOContext(IOContext&&)                 = delete;
  IOContext& operator=(IOContext&&)      = delete;
  ~IOContext()                           = default;

  void observe(std::size_t size, std::size_t file_offset) noexcept;
  [[nodiscard]] IOPolicy policy() const noexcept;
  [[nodiscard]] WorkloadClass workload() const noexcept;
  [[nodiscard]] RuntimeStats stats() const noexcept;
  [[nodiscard]] IOContextSnapshot snapshot() const noexcept;
  [[nodiscard]] bool profile_complete() const noexcept;
  void reset() noexcept;

 private:
  static constexpr std::size_t region_size       = 64 * 1024;
  static constexpr std::size_t region_filter_len = 4;

  void classify() noexcept;

  bool _host_cache_available{};
  std::atomic<std::uint64_t> _request_count{};
  std::atomic<std::uint64_t> _requested_bytes{};
  std::atomic<std::uint64_t> _profiled_requests{};
  std::atomic<std::uint64_t> _profiled_bytes{};
  std::atomic<std::uint64_t> _sequential_requests{};
  std::atomic<std::uint64_t> _repeated_regions{};
  std::atomic<std::size_t> _last_request_end{};
  std::array<std::atomic<std::uint64_t>, region_filter_len> _seen_regions{};
  std::atomic<WorkloadClass> _workload{WorkloadClass::UNKNOWN};
  std::atomic<IOPath> _path{IOPath::GPU_DIRECT};
  std::atomic<CachePolicy> _cache{CachePolicy::BYPASS};
  std::atomic<SubmitPolicy> _submit{SubmitPolicy::DIRECT};
  std::atomic<bool> _profile_complete{};
};

}  // namespace kvikio
