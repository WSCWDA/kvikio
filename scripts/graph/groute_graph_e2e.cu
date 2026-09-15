// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t file_header_bytes = 16;

void cuda_check(cudaError_t status, char const* expression)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{expression} + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

struct Options {
  std::string algorithm;
  std::string graph;
  std::string policy{"auto"};
  std::string output;
  std::uint32_t source{1};
  std::uint32_t max_iterations{100};
  std::size_t staging_bytes{256ULL << 20};
  std::size_t batch_requests{32};
  std::size_t max_segment_bytes{256ULL << 10};
  std::uint32_t threads{256};
  int gpu{0};
  float alpha{0.85F};
  float tolerance{0.001F};
  std::uint32_t repeat_id{0};
  std::uint32_t execution_order{0};
};

std::uint64_t parse_u64(char const* value, char const* name)
{
  std::size_t consumed{};
  auto const parsed = std::stoull(value, &consumed);
  if (consumed != std::strlen(value)) { throw std::invalid_argument(std::string{name}); }
  return parsed;
}

Options parse_options(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    auto need_value = [&](char const* name) {
      if (++i == argc) { throw std::invalid_argument(std::string{"missing value for "} + name); }
      return argv[i];
    };
    std::string_view argument{argv[i]};
    if (argument == "--algorithm") {
      options.algorithm = need_value("--algorithm");
    } else if (argument == "--graph") {
      options.graph = need_value("--graph");
    } else if (argument == "--policy") {
      options.policy = need_value("--policy");
    } else if (argument == "--output") {
      options.output = need_value("--output");
    } else if (argument == "--source") {
      options.source = parse_u64(need_value("--source"), "--source");
    } else if (argument == "--max-iterations") {
      options.max_iterations = parse_u64(need_value("--max-iterations"), "--max-iterations");
    } else if (argument == "--staging-bytes") {
      options.staging_bytes = parse_u64(need_value("--staging-bytes"), "--staging-bytes");
    } else if (argument == "--batch-requests") {
      options.batch_requests = parse_u64(need_value("--batch-requests"), "--batch-requests");
    } else if (argument == "--max-segment-bytes") {
      options.max_segment_bytes =
        parse_u64(need_value("--max-segment-bytes"), "--max-segment-bytes");
    } else if (argument == "--threads") {
      options.threads = parse_u64(need_value("--threads"), "--threads");
    } else if (argument == "--gpu") {
      options.gpu = static_cast<int>(parse_u64(need_value("--gpu"), "--gpu"));
    } else if (argument == "--alpha") {
      options.alpha = std::stof(need_value("--alpha"));
    } else if (argument == "--tolerance") {
      options.tolerance = std::stof(need_value("--tolerance"));
    } else if (argument == "--repeat-id") {
      options.repeat_id = parse_u64(need_value("--repeat-id"), "--repeat-id");
    } else if (argument == "--execution-order") {
      options.execution_order =
        parse_u64(need_value("--execution-order"), "--execution-order");
    } else {
      throw std::invalid_argument("unknown option: " + std::string{argument});
    }
  }
  if (options.algorithm != "bfs" && options.algorithm != "pagerank") {
    throw std::invalid_argument("--algorithm must be bfs or pagerank");
  }
  if (options.graph.empty()) { throw std::invalid_argument("--graph is required"); }
  if (options.output.empty()) { throw std::invalid_argument("--output is required"); }
  if (options.max_iterations == 0 || options.staging_bytes == 0 ||
      options.batch_requests == 0 || options.max_segment_bytes == 0 || options.threads == 0) {
    throw std::invalid_argument("numeric controls must be positive");
  }
  if (options.max_segment_bytes > options.staging_bytes) {
    throw std::invalid_argument("max segment must fit in one staging slot");
  }
  if (options.max_segment_bytes / sizeof(std::uint64_t) >
      std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("max segment contains too many edges");
  }
  if (!(options.alpha > 0.0F && options.alpha < 1.0F) || options.tolerance <= 0.0F) {
    throw std::invalid_argument("invalid PageRank alpha or tolerance");
  }
  return options;
}

template <typename T>
void device_allocate(T** pointer, std::size_t count)
{
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)));
}

std::uint64_t fnv1a(void const* data, std::size_t size, std::uint64_t hash = 1469598103934665603ULL)
{
  auto const* bytes = static_cast<unsigned char const*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct BamArray {
  std::uint64_t header_count{};
  std::vector<std::uint64_t> values;
};

BamArray read_bam_array(std::string const& path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) { throw std::runtime_error("cannot open " + path); }
  auto const bytes = static_cast<std::uint64_t>(input.tellg());
  if (bytes < file_header_bytes || (bytes - file_header_bytes) % sizeof(std::uint64_t)) {
    throw std::runtime_error("invalid BaM array: " + path);
  }
  input.seekg(0);
  BamArray result;
  std::uint64_t reserved{};
  input.read(reinterpret_cast<char*>(&result.header_count), sizeof(result.header_count));
  input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
  result.values.resize((bytes - file_header_bytes) / sizeof(std::uint64_t));
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(std::uint64_t)));
  if (!input) { throw std::runtime_error("short read from " + path); }
  return result;
}

std::uint64_t read_edge_count(std::string const& path)
{
  std::ifstream input(path, std::ios::binary);
  std::uint64_t count{};
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!input) { throw std::runtime_error("cannot read edge count from " + path); }
  return count;
}

struct DeviceRequest {
  std::uint32_t vertex{};
  std::uint32_t edge_count{};
  std::uint64_t buffer_offset{};
};

struct LogicalRequest {
  DeviceRequest device{};
  std::uint64_t edge_begin{};
};

using Batch = std::vector<LogicalRequest>;

std::vector<Batch> make_batches(std::vector<std::uint32_t> const& vertices,
                                std::vector<std::uint64_t> const& col,
                                Options const& options)
{
  std::vector<Batch> batches;
  Batch current;
  current.reserve(options.batch_requests);
  std::size_t used{};
  auto flush = [&] {
    if (!current.empty()) {
      batches.push_back(std::move(current));
      current = Batch{};
      current.reserve(options.batch_requests);
      used = 0;
    }
  };
  for (auto const vertex : vertices) {
    auto begin = col.at(vertex);
    auto const end = col.at(static_cast<std::size_t>(vertex) + 1);
    while (begin < end) {
      auto const remaining_edges = end - begin;
      auto const segment_edges = std::min<std::uint64_t>(
        remaining_edges, options.max_segment_bytes / sizeof(std::uint64_t));
      auto const bytes = static_cast<std::size_t>(segment_edges * sizeof(std::uint64_t));
      if (!current.empty() &&
          (current.size() == options.batch_requests || used + bytes > options.staging_bytes)) {
        flush();
      }
      current.push_back(LogicalRequest{DeviceRequest{vertex,
                                                     static_cast<std::uint32_t>(segment_edges),
                                                     static_cast<std::uint64_t>(used)},
                                       begin});
      used += bytes;
      begin += segment_edges;
    }
  }
  flush();
  return batches;
}

class Slot {
 public:
  Slot(std::size_t staging_bytes, std::size_t max_requests)
  {
    CUDA_CHECK(cudaMalloc(&edges_, staging_bytes));
    device_allocate(&requests_, max_requests);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  }
  Slot(Slot const&)            = delete;
  Slot& operator=(Slot const&) = delete;
  ~Slot()
  {
    if (active_) { cudaEventSynchronize(done_); }
    if (done_) { cudaEventDestroy(done_); }
    if (stream_) { cudaStreamDestroy(stream_); }
    if (requests_) { cudaFree(requests_); }
    if (edges_) { cudaFree(edges_); }
  }

  void wait_reusable()
  {
    if (active_) {
      CUDA_CHECK(cudaEventSynchronize(done_));
      active_ = false;
    }
  }

  void submit(kvikio::FileHandle& file, Batch const& batch)
  {
    wait_reusable();
    host_requests_.clear();
    futures_.clear();
    expected_bytes_.clear();
    host_requests_.reserve(batch.size());
    futures_.reserve(batch.size());
    expected_bytes_.reserve(batch.size());
    for (auto const& request : batch) {
      host_requests_.push_back(request.device);
      auto const bytes =
        static_cast<std::size_t>(request.device.edge_count) * sizeof(std::uint64_t);
      auto* destination = static_cast<std::byte*>(edges_) + request.device.buffer_offset;
      futures_.push_back(file.pread(destination,
                                    bytes,
                                    file_header_bytes + request.edge_begin * sizeof(std::uint64_t),
                                    bytes,
                                    kvikio::defaults::gds_threshold(),
                                    false));
      expected_bytes_.push_back(bytes);
      logical_bytes_ += bytes;
      logical_requests_++;
    }
  }

  void finish_io()
  {
    for (std::size_t i = 0; i < futures_.size(); ++i) {
      auto const bytes = futures_[i].get();
      if (bytes != expected_bytes_[i]) { throw std::runtime_error("short graph adjacency read"); }
    }
    CUDA_CHECK(cudaMemcpyAsync(requests_,
                               host_requests_.data(),
                               host_requests_.size() * sizeof(DeviceRequest),
                               cudaMemcpyHostToDevice,
                               stream_));
  }

  void mark_active()
  {
    CUDA_CHECK(cudaEventRecord(done_, stream_));
    active_ = true;
  }

  void wait_for(Slot const& previous)
  {
    CUDA_CHECK(cudaStreamWaitEvent(stream_, previous.done_, 0));
  }

  [[nodiscard]] std::uint64_t* edges() const { return static_cast<std::uint64_t*>(edges_); }
  [[nodiscard]] DeviceRequest* requests() const { return requests_; }
  [[nodiscard]] std::size_t request_count() const { return host_requests_.size(); }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] std::uint64_t logical_requests() const { return logical_requests_; }
  [[nodiscard]] std::uint64_t logical_bytes() const { return logical_bytes_; }

 private:
  void* edges_{};
  DeviceRequest* requests_{};
  cudaStream_t stream_{};
  cudaEvent_t done_{};
  bool active_{};
  std::vector<DeviceRequest> host_requests_;
  std::vector<std::future<std::size_t>> futures_;
  std::vector<std::size_t> expected_bytes_;
  std::uint64_t logical_requests_{};
  std::uint64_t logical_bytes_{};
};

__global__ void bfs_expand(std::uint64_t const* edges,
                           DeviceRequest const* requests,
                           std::uint32_t request_count,
                           std::uint32_t vertex_count,
                           std::uint32_t* visited,
                           std::uint32_t* next_frontier,
                           std::uint32_t* next_count)
{
  auto const request_id = blockIdx.x;
  if (request_id >= request_count) { return; }
  auto const request = requests[request_id];
  auto const* neighbors = reinterpret_cast<std::uint64_t const*>(
    reinterpret_cast<std::byte const*>(edges) + request.buffer_offset);
  for (std::uint32_t i = threadIdx.x; i < request.edge_count; i += blockDim.x) {
    auto const neighbor = static_cast<std::uint32_t>(neighbors[i]);
    if (neighbor < vertex_count && atomicCAS(visited + neighbor, 0U, 1U) == 0U) {
      auto const position = atomicAdd(next_count, 1U);
      next_frontier[position] = neighbor;
    }
  }
}

__global__ void pagerank_initialize(std::uint64_t const* col,
                                    std::uint32_t vertex_count,
                                    float alpha,
                                    float* rank,
                                    float* delta,
                                    float* residual)
{
  auto const vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= vertex_count) { return; }
  auto const degree = col[vertex + 1] - col[vertex];
  rank[vertex] = 1.0F - alpha;
  delta[vertex] = degree == 0 ? 0.0F : (1.0F - alpha) * alpha / degree;
  residual[vertex] = 0.0F;
}

__global__ void pagerank_push(std::uint64_t const* edges,
                              DeviceRequest const* requests,
                              std::uint32_t request_count,
                              std::uint32_t vertex_count,
                              float const* delta,
                              float* residual)
{
  auto const request_id = blockIdx.x;
  if (request_id >= request_count) { return; }
  auto const request = requests[request_id];
  auto const contribution = delta[request.vertex];
  auto const* neighbors = reinterpret_cast<std::uint64_t const*>(
    reinterpret_cast<std::byte const*>(edges) + request.buffer_offset);
  for (std::uint32_t i = threadIdx.x; i < request.edge_count; i += blockDim.x) {
    auto const neighbor = static_cast<std::uint32_t>(neighbors[i]);
    if (neighbor < vertex_count) { atomicAdd(residual + neighbor, contribution); }
  }
}

__global__ void pagerank_update(std::uint64_t const* col,
                                std::uint32_t vertex_count,
                                float alpha,
                                float tolerance,
                                float* rank,
                                float* delta,
                                float* residual,
                                std::uint32_t* next_active,
                                std::uint32_t* next_count)
{
  auto const vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= vertex_count) { return; }
  auto const value = residual[vertex];
  auto const degree = col[vertex + 1] - col[vertex];
  if (value > tolerance && degree > 0) {
    rank[vertex] += value;
    delta[vertex] = value * alpha / degree;
    next_active[atomicAdd(next_count, 1U)] = vertex;
  } else {
    delta[vertex] = 0.0F;
  }
  residual[vertex] = 0.0F;
}

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

char const* policy_mode_name(kvikio::PolicyMode value)
{
  switch (value) {
    case kvikio::PolicyMode::AUTO: return "AUTO";
    case kvikio::PolicyMode::HOST_DIRECT: return "HOST_DIRECT";
    case kvikio::PolicyMode::HOST_CACHE: return "HOST_CACHE";
    case kvikio::PolicyMode::GDS_DIRECT: return "GDS_DIRECT";
    case kvikio::PolicyMode::GDS_SHAPED: return "GDS_SHAPED";
  }
  return "AUTO";
}

struct Result {
  std::uint32_t iterations{};
  std::uint64_t visited{};
  std::uint64_t processed_edges{};
  double rank_sum{};
  double algorithm_seconds{};
  double job_seconds{};
  std::uint64_t trace_hash{1469598103934665603ULL};
  std::uint64_t result_hash{};
};

void update_trace_hash(Result& result, std::vector<Batch> const& batches)
{
  for (auto const& batch : batches) {
    for (auto const& request : batch) {
      result.trace_hash =
        fnv1a(&request.device.vertex, sizeof(request.device.vertex), result.trace_hash);
      result.trace_hash = fnv1a(&request.edge_begin, sizeof(request.edge_begin), result.trace_hash);
      result.trace_hash = fnv1a(&request.device.edge_count,
                                sizeof(request.device.edge_count),
                                result.trace_hash);
    }
  }
}

template <typename Launch>
void execute_batches(std::vector<Batch> const& batches,
                     kvikio::FileHandle& file,
                     std::array<Slot*, 2> slots,
                     Launch launch)
{
  if (batches.empty()) { return; }
  slots[0]->submit(file, batches[0]);
  Slot* previous_compute{};
  for (std::size_t i = 0; i < batches.size(); ++i) {
    auto& current = *slots[i % slots.size()];
    if (i + 1 < batches.size()) {
      auto& next = *slots[(i + 1) % slots.size()];
      next.wait_reusable();
      next.submit(file, batches[i + 1]);
    }
    current.finish_io();
    if (previous_compute != nullptr) { current.wait_for(*previous_compute); }
    launch(current);
    current.mark_active();
    previous_compute = &current;
  }
  for (auto* slot : slots) { slot->wait_reusable(); }
}

Result run_bfs(Options const& options,
               std::vector<std::uint64_t> const& col,
               std::string const& edge_path,
               Clock::time_point job_start,
               kvikio::IOContextSnapshot& context,
               kvikio::HostCacheStats& cache,
               kvikio::RequestShaperStats& shaping,
               std::uint64_t& logical_requests,
               std::uint64_t& logical_bytes)
{
  auto const vertex_count = static_cast<std::uint32_t>(col.size() - 1);
  if (options.source >= vertex_count) { throw std::invalid_argument("BFS source out of range"); }
  kvikio::FileHandle file(edge_path, "r");
  Slot slot0(options.staging_bytes, options.batch_requests);
  Slot slot1(options.staging_bytes, options.batch_requests);
  std::array<Slot*, 2> slots{&slot0, &slot1};
  std::uint32_t *visited{}, *frontier_a{}, *frontier_b{}, *next_count{};
  device_allocate(&visited, vertex_count);
  device_allocate(&frontier_a, vertex_count);
  device_allocate(&frontier_b, vertex_count);
  device_allocate(&next_count, 1);
  CUDA_CHECK(cudaMemset(visited, 0, vertex_count * sizeof(std::uint32_t)));
  std::uint32_t const one{1};
  CUDA_CHECK(cudaMemcpy(visited + options.source,
                        &one,
                        sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(frontier_a,
                        &options.source,
                        sizeof(options.source),
                        cudaMemcpyHostToDevice));

  std::vector<std::uint32_t> frontier{options.source};
  Result result;
  result.visited = 1;
  auto const algorithm_start = Clock::now();
  while (!frontier.empty() && result.iterations < options.max_iterations) {
    std::sort(frontier.begin(), frontier.end());
    auto const batches = make_batches(frontier, col, options);
    update_trace_hash(result, batches);
    result.processed_edges += std::accumulate(
      batches.begin(), batches.end(), std::uint64_t{}, [](auto total, Batch const& batch) {
        for (auto const& request : batch) { total += request.device.edge_count; }
        return total;
      });
    CUDA_CHECK(cudaMemset(next_count, 0, sizeof(std::uint32_t)));
    execute_batches(batches, file, slots, [&](Slot& slot) {
      bfs_expand<<<slot.request_count(), options.threads, 0, slot.stream()>>>(slot.edges(),
                                                                             slot.requests(),
                                                                             slot.request_count(),
                                                                             vertex_count,
                                                                             visited,
                                                                             frontier_b,
                                                                             next_count);
      CUDA_CHECK(cudaGetLastError());
    });
    std::uint32_t count{};
    CUDA_CHECK(cudaMemcpy(&count, next_count, sizeof(count), cudaMemcpyDeviceToHost));
    frontier.resize(count);
    if (count != 0) {
      CUDA_CHECK(cudaMemcpy(frontier.data(),
                            frontier_b,
                            count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost));
    }
    result.visited += count;
    std::swap(frontier_a, frontier_b);
    result.iterations++;
  }
  result.algorithm_seconds = std::chrono::duration<double>(Clock::now() - algorithm_start).count();
  std::vector<std::uint32_t> host_visited(vertex_count);
  CUDA_CHECK(cudaMemcpy(host_visited.data(),
                        visited,
                        host_visited.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost));
  result.result_hash = fnv1a(host_visited.data(), host_visited.size() * sizeof(std::uint32_t));
  context = file.io_context_snapshot();
  cache = file.host_cache_stats();
  shaping = file.request_shaper_stats();
  logical_requests = slot0.logical_requests() + slot1.logical_requests();
  logical_bytes = slot0.logical_bytes() + slot1.logical_bytes();
  result.job_seconds = std::chrono::duration<double>(Clock::now() - job_start).count();
  cudaFree(next_count);
  cudaFree(frontier_b);
  cudaFree(frontier_a);
  cudaFree(visited);
  return result;
}

Result run_pagerank(Options const& options,
                    std::vector<std::uint64_t> const& col,
                    std::string const& edge_path,
                    Clock::time_point job_start,
                    kvikio::IOContextSnapshot& context,
                    kvikio::HostCacheStats& cache,
                    kvikio::RequestShaperStats& shaping,
                    std::uint64_t& logical_requests,
                    std::uint64_t& logical_bytes)
{
  auto const vertex_count = static_cast<std::uint32_t>(col.size() - 1);
  kvikio::FileHandle file(edge_path, "r");
  Slot slot0(options.staging_bytes, options.batch_requests);
  Slot slot1(options.staging_bytes, options.batch_requests);
  std::array<Slot*, 2> slots{&slot0, &slot1};
  std::uint64_t* col_device{};
  float *rank{}, *delta{}, *residual{};
  std::uint32_t *next_active{}, *next_count{};
  device_allocate(&col_device, col.size());
  CUDA_CHECK(cudaMemcpy(col_device,
                        col.data(),
                        col.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice));
  device_allocate(&rank, vertex_count);
  device_allocate(&delta, vertex_count);
  device_allocate(&residual, vertex_count);
  device_allocate(&next_active, vertex_count);
  device_allocate(&next_count, 1);
  auto const blocks = (vertex_count + options.threads - 1) / options.threads;
  pagerank_initialize<<<blocks, options.threads>>>(
    col_device, vertex_count, options.alpha, rank, delta, residual);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint32_t> active(vertex_count);
  std::iota(active.begin(), active.end(), 0U);
  Result result;
  auto const algorithm_start = Clock::now();
  while (!active.empty() && result.iterations < options.max_iterations) {
    std::sort(active.begin(), active.end());
    auto const batches = make_batches(active, col, options);
    update_trace_hash(result, batches);
    result.processed_edges += std::accumulate(
      batches.begin(), batches.end(), std::uint64_t{}, [](auto total, Batch const& batch) {
        for (auto const& request : batch) { total += request.device.edge_count; }
        return total;
      });
    execute_batches(batches, file, slots, [&](Slot& slot) {
      pagerank_push<<<slot.request_count(), options.threads, 0, slot.stream()>>>(slot.edges(),
                                                                               slot.requests(),
                                                                               slot.request_count(),
                                                                               vertex_count,
                                                                               delta,
                                                                               residual);
      CUDA_CHECK(cudaGetLastError());
    });
    CUDA_CHECK(cudaMemset(next_count, 0, sizeof(std::uint32_t)));
    pagerank_update<<<blocks, options.threads>>>(col_device,
                                                 vertex_count,
                                                 options.alpha,
                                                 options.tolerance,
                                                 rank,
                                                 delta,
                                                 residual,
                                                 next_active,
                                                 next_count);
    CUDA_CHECK(cudaGetLastError());
    std::uint32_t count{};
    CUDA_CHECK(cudaMemcpy(&count, next_count, sizeof(count), cudaMemcpyDeviceToHost));
    active.resize(count);
    if (count != 0) {
      CUDA_CHECK(cudaMemcpy(active.data(),
                            next_active,
                            count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost));
    }
    result.iterations++;
  }
  result.algorithm_seconds = std::chrono::duration<double>(Clock::now() - algorithm_start).count();
  std::vector<float> host_rank(vertex_count);
  CUDA_CHECK(cudaMemcpy(
    host_rank.data(), rank, vertex_count * sizeof(float), cudaMemcpyDeviceToHost));
  result.rank_sum = std::accumulate(host_rank.begin(), host_rank.end(), 0.0);
  std::vector<std::int64_t> quantized_rank(vertex_count);
  std::transform(host_rank.begin(), host_rank.end(), quantized_rank.begin(), [](float value) {
    return static_cast<std::int64_t>(std::llround(static_cast<double>(value) * 1'000'000.0));
  });
  result.result_hash =
    fnv1a(quantized_rank.data(), quantized_rank.size() * sizeof(std::int64_t));
  context = file.io_context_snapshot();
  cache = file.host_cache_stats();
  shaping = file.request_shaper_stats();
  logical_requests = slot0.logical_requests() + slot1.logical_requests();
  logical_bytes = slot0.logical_bytes() + slot1.logical_bytes();
  result.job_seconds = std::chrono::duration<double>(Clock::now() - job_start).count();
  cudaFree(next_count);
  cudaFree(next_active);
  cudaFree(residual);
  cudaFree(delta);
  cudaFree(rank);
  cudaFree(col_device);
  return result;
}

void write_json(Options const& options,
                Result const& result,
                std::uint64_t vertex_count,
                std::uint64_t edge_count,
                kvikio::IOContextSnapshot const& context,
                kvikio::HostCacheStats const& cache,
                kvikio::RequestShaperStats const& shaping,
                std::uint64_t logical_requests,
                std::uint64_t logical_bytes)
{
  std::ofstream out(options.output);
  if (!out) { throw std::runtime_error("cannot create " + options.output); }
  out << std::setprecision(17);
  out << "{\n"
      << "  \"algorithm\": \"" << options.algorithm << "\",\n"
      << "  \"policy_mode\": \"" << options.policy << "\",\n"
      << "  \"repeat_id\": " << options.repeat_id << ",\n"
      << "  \"execution_order\": " << options.execution_order << ",\n"
      << "  \"graph\": \"" << options.graph << "\",\n"
      << "  \"vertex_count\": " << vertex_count << ",\n"
      << "  \"edge_count\": " << edge_count << ",\n"
      << "  \"iterations\": " << result.iterations << ",\n"
      << "  \"visited_vertices\": " << result.visited << ",\n"
      << "  \"processed_edges\": " << result.processed_edges << ",\n"
      << "  \"rank_sum\": " << result.rank_sum << ",\n"
      << "  \"algorithm_seconds\": " << result.algorithm_seconds << ",\n"
      << "  \"job_seconds\": " << result.job_seconds << ",\n"
      << "  \"logical_requests\": " << logical_requests << ",\n"
      << "  \"logical_bytes\": " << logical_bytes << ",\n"
      << "  \"logical_trace_hash\": " << result.trace_hash << ",\n"
      << "  \"result_hash\": " << result.result_hash << ",\n"
      << "  \"groute_enabled\": "
      << (kvikio::defaults::groute_enabled() ? "true" : "false") << ",\n"
      << "  \"dispatch\": \""
      << (options.policy == "kvikio_threshold" ? "NATIVE_KVIKIO_THRESHOLD" : "GROUTE_POLICY")
      << "\",\n"
      << "  \"teps\": "
      << (result.processed_edges / std::max(result.algorithm_seconds, 1e-12)) << ",\n"
      << "  \"selected_policy\": {\"mode\": \"" << policy_mode_name(context.policy_mode)
      << "\", \"workload\": \"" << workload_name(context.workload)
      << "\", \"path\": \"" << path_name(context.policy.path) << "\", \"cache\": \""
      << cache_name(context.policy.cache) << "\", \"submit\": \""
      << submit_name(context.policy.submit) << "\"},\n"
      << "  \"profile\": {\"requests\": " << context.stats.request_count
      << ", \"profiled_requests\": " << context.stats.profiled_requests
      << ", \"average_io_size\": " << context.stats.average_io_size
      << ", \"sequential_ratio\": " << context.stats.sequential_ratio
      << ", \"reuse_ratio\": " << context.stats.repeated_region_ratio
      << ", \"unaligned_ratio\": " << context.stats.file_offset_unaligned_ratio
      << ", \"mergeable_ratio\": " << context.stats.mergeable_ratio << "},\n"
      << "  \"cache\": {\"hits\": " << cache.hits << ", \"misses\": " << cache.misses
      << ", \"storage_bytes\": " << cache.storage_bytes
      << ", \"admitted_regions\": " << cache.admitted_regions
      << ", \"cache_entries\": " << cache.cache_entries << "},\n"
      << "  \"shaping\": {\"logical_requests\": " << shaping.logical_requests
      << ", \"physical_requests\": " << shaping.physical_requests
      << ", \"submitted_bytes\": " << shaping.submitted_bytes
      << ", \"direct_fallbacks\": " << shaping.direct_fallbacks << "}\n"
      << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
  auto const job_start = Clock::now();
  try {
    auto const options = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.gpu));
    auto const col_file = read_bam_array(options.graph + ".col");
    if (col_file.values.size() < 2) { throw std::runtime_error("empty CSR offset array"); }
    auto const edge_path = options.graph + ".dst";
    auto const edge_count = read_edge_count(edge_path);
    if (col_file.values.back() > edge_count) {
      throw std::runtime_error("CSR offsets exceed .dst edge count");
    }
    if (col_file.values.size() - 1 > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("current executor requires at most 2^32-1 vertices");
    }
    kvikio::IOContextSnapshot context;
    kvikio::HostCacheStats cache;
    kvikio::RequestShaperStats shaping;
    std::uint64_t logical_requests{};
    std::uint64_t logical_bytes{};
    Result result;
    if (options.algorithm == "bfs") {
      result = run_bfs(options,
                       col_file.values,
                       edge_path,
                       job_start,
                       context,
                       cache,
                       shaping,
                       logical_requests,
                       logical_bytes);
    } else {
      result = run_pagerank(options,
                            col_file.values,
                            edge_path,
                            job_start,
                            context,
                            cache,
                            shaping,
                            logical_requests,
                            logical_bytes);
    }
    write_json(options,
               result,
               col_file.values.size() - 1,
               edge_count,
               context,
               cache,
               shaping,
               logical_requests,
               logical_bytes);
    std::cout << "Wrote " << options.output << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "groute_graph_e2e: " << error.what() << '\n';
    return 1;
  }
}
