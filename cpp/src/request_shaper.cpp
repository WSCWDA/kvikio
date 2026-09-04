/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <kvikio/detail/stream.hpp>
#include <kvikio/error.hpp>
#include <kvikio/request_shaper.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {
namespace {

std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{ return value - value % alignment; }

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  auto const remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

PhysicalReadPlan direct_plan(LogicalRead const& request)
{
  return {.file_offset   = request.file_offset,
          .size          = request.size,
          .logical_bytes = request.size,
          .shaped        = false,
          .slices = {{request.id, 0, request.dev_ptr_base, request.dev_ptr_offset, request.size}}};
}

}  // namespace

RequestPlanner::RequestPlanner(ShapingConfig config) : _config{config}
{
  KVIKIO_EXPECT(_config.alignment > 0, "request shaping alignment must be positive");
  KVIKIO_EXPECT(_config.max_batch_bytes >= _config.alignment,
                "request shaping batch must hold at least one aligned block");
  KVIKIO_EXPECT(_config.max_batch_requests > 0,
                "request shaping batch request limit must be positive");
  KVIKIO_EXPECT(_config.max_amplification >= 1.0,
                "request shaping amplification limit must be at least one");
}

std::vector<PhysicalReadPlan> RequestPlanner::plan(std::vector<LogicalRead> const& requests) const
{
  if (requests.empty()) { return {}; }

  std::vector<LogicalRead const*> sorted;
  sorted.reserve(requests.size());
  for (auto const& request : requests) {
    if (request.size != 0) { sorted.push_back(&request); }
  }
  std::stable_sort(sorted.begin(), sorted.end(), [](auto const* lhs, auto const* rhs) {
    return lhs->file_offset < rhs->file_offset;
  });

  std::vector<PhysicalReadPlan> result;
  std::vector<LogicalRead const*> group;

  auto emit_group = [&] {
    if (group.empty()) { return; }
    auto begin = group.front()->file_offset;
    auto end   = begin + group.front()->size;
    std::size_t logical_bytes{};
    std::size_t file_size = group.front()->file_size;
    for (auto const* request : group) {
      end = std::max(end, request->file_offset + request->size);
      logical_bytes += request->size;
      file_size = std::min(file_size, request->file_size);
    }
    auto const physical_begin = align_down(begin, _config.alignment);
    auto const physical_end   = align_up(end, _config.alignment);
    auto const physical_bytes = physical_end - physical_begin;
    auto const amplification =
      static_cast<double>(physical_bytes) / static_cast<double>(logical_bytes);
    bool const profitable = group.size() >= 2 && physical_end <= file_size &&
                            physical_bytes <= _config.max_batch_bytes &&
                            amplification <= _config.max_amplification;
    if (!profitable) {
      for (auto const* request : group) {
        result.push_back(direct_plan(*request));
      }
      group.clear();
      return;
    }

    PhysicalReadPlan plan{.file_offset   = physical_begin,
                          .size          = physical_bytes,
                          .logical_bytes = logical_bytes,
                          .shaped        = true};
    plan.slices.reserve(group.size());
    for (auto const* request : group) {
      plan.slices.push_back({request->id,
                             request->file_offset - physical_begin,
                             request->dev_ptr_base,
                             request->dev_ptr_offset,
                             request->size});
    }
    result.push_back(std::move(plan));
    group.clear();
  };

  for (auto const* request : sorted) {
    if (request->size > _config.max_request_size) {
      emit_group();
      result.push_back(direct_plan(*request));
      continue;
    }
    if (group.empty()) {
      group.push_back(request);
      continue;
    }
    auto group_end = group.front()->file_offset + group.front()->size;
    for (auto const* existing : group) {
      group_end = std::max(group_end, existing->file_offset + existing->size);
    }
    auto const gap = request->file_offset > group_end ? request->file_offset - group_end : 0;
    auto const candidate_end   = std::max(group_end, request->file_offset + request->size);
    auto const candidate_bytes = align_up(candidate_end, _config.alignment) -
                                 align_down(group.front()->file_offset, _config.alignment);
    if (gap <= _config.max_merge_gap && group.size() < _config.max_batch_requests &&
        candidate_bytes <= _config.max_batch_bytes && request->context == group.front()->context) {
      group.push_back(request);
    } else {
      emit_group();
      group.push_back(request);
    }
  }
  emit_group();
  return result;
}

class RequestShaper::Impl {
 public:
  struct PendingRead {
    CUfileHandle_t file_handle{};
    LogicalRead request{};
    bool sync_default_stream{};
    std::promise<std::size_t> completion{};
  };

  Impl(ThreadPool* thread_pool, ShapingConfig config)
    : thread_pool{thread_pool}, config{config}, planner{config}
  { KVIKIO_EXPECT(thread_pool != nullptr, "request shaper thread pool must not be nullptr"); }

  ~Impl() noexcept
  {
    try {
      close();
    } catch (...) {
    }
  }

  std::future<std::size_t> submit(CUfileHandle_t file_handle,
                                  void* dev_ptr_base,
                                  std::size_t size,
                                  std::size_t file_offset,
                                  std::size_t dev_ptr_offset,
                                  std::size_t file_size,
                                  CUcontext context,
                                  bool sync_default_stream)
  {
    if (size == 0) {
      std::promise<std::size_t> completion;
      auto future = completion.get_future();
      completion.set_value(0);
      return future;
    }
    auto pending                 = std::make_shared<PendingRead>();
    pending->file_handle         = file_handle;
    pending->request             = {.dev_ptr_base   = dev_ptr_base,
                                    .dev_ptr_offset = dev_ptr_offset,
                                    .file_offset    = file_offset,
                                    .size           = size,
                                    .file_size      = file_size,
                                    .context        = context};
    pending->sync_default_stream = sync_default_stream;
    auto future                  = pending->completion.get_future();
    {
      std::lock_guard lock{mutex};
      KVIKIO_EXPECT(!closing, "cannot submit to a closed request shaper");
      pending_reads.push_back(std::move(pending));
      pending_bytes += size;
      if (!worker_active) {
        worker_active = true;
        worker        = thread_pool->submit_task([this] { drain(); });
      }
    }
    condition.notify_all();
    return future;
  }

  void close()
  {
    {
      std::lock_guard lock{mutex};
      if (closing && !worker.valid()) {
        release_staging();
        return;
      }
      closing = true;
    }
    condition.notify_all();
    if (worker.valid()) { worker.get(); }
    release_staging();
  }

  RequestShaperStats get_stats() const noexcept
  {
    return {.logical_requests  = logical_requests.load(std::memory_order_relaxed),
            .physical_requests = physical_requests.load(std::memory_order_relaxed),
            .logical_bytes     = logical_bytes.load(std::memory_order_relaxed),
            .submitted_bytes   = submitted_bytes.load(std::memory_order_relaxed),
            .shaped_groups     = shaped_groups.load(std::memory_order_relaxed),
            .direct_fallbacks  = direct_fallbacks.load(std::memory_order_relaxed)};
  }

 private:
  void drain() noexcept
  {
    while (true) {
      std::vector<std::shared_ptr<PendingRead>> batch;
      {
        std::unique_lock lock{mutex};
        condition.wait_for(lock, std::chrono::microseconds{config.collection_window_us}, [this] {
          return closing || pending_reads.size() >= config.max_batch_requests ||
                 pending_bytes >= config.max_batch_bytes;
        });
        batch.swap(pending_reads);
        pending_bytes = 0;
        if (batch.empty()) {
          worker_active = false;
          condition.notify_all();
          return;
        }
      }
      process(batch);
      std::lock_guard lock{mutex};
      if (pending_reads.empty()) {
        worker_active = false;
        condition.notify_all();
        return;
      }
    }
  }

  void process(std::vector<std::shared_ptr<PendingRead>>& batch) noexcept
  {
    std::vector<LogicalRead> requests;
    requests.reserve(batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
      batch[i]->request.id = i;
      requests.push_back(batch[i]->request);
    }
    auto const plans = planner.plan(requests);
    for (auto const& plan : plans) {
      try {
        auto const context = requests.at(plan.slices.front().logical_id).context;
        PushAndPopContext context_guard{context};
        bool sync_default_stream = false;
        for (auto const& slice : plan.slices) {
          sync_default_stream =
            sync_default_stream || batch.at(slice.logical_id)->sync_default_stream;
        }
        if (sync_default_stream) {
          KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
        }
        if (plan.shaped) {
          ensure_staging(context);
          auto const handle = batch.at(plan.slices.front().logical_id)->file_handle;
          auto const ret    = cuFileAPI::instance().Read(handle,
                                                         reinterpret_cast<void*>(staging),
                                                         plan.size,
                                                         convert_size2off(plan.file_offset),
                                                         0);
          KVIKIO_CUFILE_CHECK_BYTES_DONE(ret);
          KVIKIO_EXPECT(static_cast<std::size_t>(ret) == plan.size,
                        "short physical read while executing shaped request");
          for (auto const& slice : plan.slices) {
            KVIKIO_CUDA_DRIVER_TRY(cudaAPI::cuda_memcpy_async(
              convert_void2deviceptr(slice.destination_base) + slice.destination_offset,
              staging + slice.staging_offset,
              slice.size,
              stream));
          }
          KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
          for (auto const& slice : plan.slices) {
            batch.at(slice.logical_id)->completion.set_value(slice.size);
          }
          shaped_groups.fetch_add(1, std::memory_order_relaxed);
        } else {
          auto const& slice = plan.slices.front();
          auto const handle = batch.at(slice.logical_id)->file_handle;
          auto const ret = cuFileAPI::instance().Read(handle,
                                                      slice.destination_base,
                                                      slice.size,
                                                      convert_size2off(plan.file_offset),
                                                      convert_size2off(slice.destination_offset));
          KVIKIO_CUFILE_CHECK_BYTES_DONE(ret);
          batch.at(slice.logical_id)->completion.set_value(static_cast<std::size_t>(ret));
          direct_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        logical_requests.fetch_add(plan.slices.size(), std::memory_order_relaxed);
        physical_requests.fetch_add(1, std::memory_order_relaxed);
        logical_bytes.fetch_add(plan.logical_bytes, std::memory_order_relaxed);
        submitted_bytes.fetch_add(plan.size, std::memory_order_relaxed);
      } catch (...) {
        auto const error = std::current_exception();
        for (auto const& slice : plan.slices) {
          try {
            batch.at(slice.logical_id)->completion.set_exception(error);
          } catch (...) {
          }
        }
      }
    }
  }

  void ensure_staging(CUcontext context)
  {
    if (staging != 0 && staging_context == context) { return; }
    release_staging();
    staging_context = context;
    PushAndPopContext context_guard{context};
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemAlloc(&staging, config.max_batch_bytes));
    bool registered = false;
    try {
      KVIKIO_CUFILE_TRY(cuFileAPI::instance().BufRegister(
        reinterpret_cast<void*>(staging), config.max_batch_bytes, 0));
      registered = true;
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    } catch (...) {
      if (registered) {
        try {
          cuFileAPI::instance().BufDeregister(reinterpret_cast<void*>(staging));
        } catch (...) {
        }
      }
      cudaAPI::instance().MemFree(staging);
      staging         = 0;
      staging_context = nullptr;
      throw;
    }
  }

  void release_staging() noexcept
  {
    if (staging == 0) { return; }
    try {
      PushAndPopContext context_guard{staging_context};
      if (stream != nullptr) {
        cudaAPI::instance().StreamSynchronize(stream);
        cudaAPI::instance().StreamDestroy(stream);
      }
      cuFileAPI::instance().BufDeregister(reinterpret_cast<void*>(staging));
      cudaAPI::instance().MemFree(staging);
    } catch (...) {
    }
    staging         = 0;
    stream          = nullptr;
    staging_context = nullptr;
  }

  ThreadPool* thread_pool;
  ShapingConfig config;
  RequestPlanner planner;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::vector<std::shared_ptr<PendingRead>> pending_reads;
  std::size_t pending_bytes{};
  bool worker_active{};
  bool closing{};
  std::future<void> worker;

  CUdeviceptr staging{};
  CUstream stream{};
  CUcontext staging_context{};

  std::atomic<std::uint64_t> logical_requests{};
  std::atomic<std::uint64_t> physical_requests{};
  std::atomic<std::uint64_t> logical_bytes{};
  std::atomic<std::uint64_t> submitted_bytes{};
  std::atomic<std::uint64_t> shaped_groups{};
  std::atomic<std::uint64_t> direct_fallbacks{};
};

RequestShaper::RequestShaper(ThreadPool* thread_pool, ShapingConfig config)
  : _impl{std::make_unique<Impl>(thread_pool, config)}
{
}

RequestShaper::~RequestShaper() noexcept = default;

std::future<std::size_t> RequestShaper::submit(CUfileHandle_t file_handle,
                                               void* dev_ptr_base,
                                               std::size_t size,
                                               std::size_t file_offset,
                                               std::size_t dev_ptr_offset,
                                               std::size_t file_size,
                                               CUcontext context,
                                               bool sync_default_stream)
{
  return _impl->submit(file_handle,
                       dev_ptr_base,
                       size,
                       file_offset,
                       dev_ptr_offset,
                       file_size,
                       context,
                       sync_default_stream);
}

void RequestShaper::close() { _impl->close(); }

RequestShaperStats RequestShaper::stats() const noexcept { return _impl->get_stats(); }

}  // namespace kvikio::detail
