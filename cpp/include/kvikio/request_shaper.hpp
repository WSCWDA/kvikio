/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include <kvikio/io_context.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/threadpool_wrapper.hpp>

namespace kvikio {

struct RequestShaperStats {
  std::uint64_t logical_requests{};
  std::uint64_t physical_requests{};
  std::uint64_t logical_bytes{};
  std::uint64_t submitted_bytes{};
  std::uint64_t shaped_groups{};
  std::uint64_t direct_fallbacks{};
};

namespace detail {

struct LogicalRead {
  std::size_t id{};
  void* dev_ptr_base{};
  std::size_t dev_ptr_offset{};
  std::size_t file_offset{};
  std::size_t size{};
  std::size_t file_size{};
  CUcontext context{};
};

struct ReadSlice {
  std::size_t logical_id{};
  std::size_t staging_offset{};
  void* destination_base{};
  std::size_t destination_offset{};
  std::size_t size{};
};

struct PhysicalReadPlan {
  std::size_t file_offset{};
  std::size_t size{};
  std::size_t logical_bytes{};
  bool shaped{};
  std::vector<ReadSlice> slices{};
};

/** @brief Pure request planner used independently of CUDA and cuFile execution. */
class RequestPlanner {
 public:
  explicit RequestPlanner(ShapingConfig config = {});
  [[nodiscard]] std::vector<PhysicalReadPlan> plan(std::vector<LogicalRead> const& requests) const;

 private:
  ShapingConfig _config;
};

/**
 * @brief Per-file deferred executor for fine-grained device reads.
 *
 * Logical reads are collected for a bounded interval. Mergeable file ranges are read once into a
 * persistent registered GPU staging buffer and scattered to the original destinations using D2D
 * copies. Completion of each returned future means that its destination contains the requested
 * bytes.
 */
class RequestShaper {
 public:
  explicit RequestShaper(ThreadPool* thread_pool, ShapingConfig config = {});
  RequestShaper(RequestShaper const&)            = delete;
  RequestShaper& operator=(RequestShaper const&) = delete;
  RequestShaper(RequestShaper&&)                 = delete;
  RequestShaper& operator=(RequestShaper&&)      = delete;
  ~RequestShaper() noexcept;

  std::future<std::size_t> submit(CUfileHandle_t file_handle,
                                  void* dev_ptr_base,
                                  std::size_t size,
                                  std::size_t file_offset,
                                  std::size_t dev_ptr_offset,
                                  std::size_t file_size,
                                  CUcontext context,
                                  bool sync_default_stream);

  void close();
  [[nodiscard]] RequestShaperStats stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace detail
}  // namespace kvikio
