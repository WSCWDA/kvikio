// BFS-only diagnostic instrumentation; included before page_cache.h.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
struct BfsPageEvent { unsigned long long page, edges; };
struct BfsTraceState {
    BfsPageEvent* events;
    unsigned long long capacity, count, outstanding, peak, submissions, submission_sum, page_bytes;
};
__device__ __managed__ BfsTraceState bfs_trace_state = {};
static inline void bfs_trace_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__device__ inline void bfs_trace_page(unsigned long long edge) {
    if (!bfs_trace_state.events) return;
    unsigned long long page = edge * 8 / bfs_trace_state.page_bytes;
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, page);
    if ((threadIdx.x & 31) == __ffs(peers)-1) {
        auto pos = atomicAdd(&bfs_trace_state.count, 1ULL);
        if (pos < bfs_trace_state.capacity)
            bfs_trace_state.events[pos] = {page, (unsigned long long)__popc(peers)};
    }
}
// Before sq_enqueue through cq_poll: includes software queue backpressure.
__device__ inline void bfs_trace_io_begin() {
    if (!bfs_trace_state.events) return;
    auto n = atomicAdd(&bfs_trace_state.outstanding, 1ULL) + 1;
    atomicMax(&bfs_trace_state.peak, n);
    atomicAdd(&bfs_trace_state.submissions, 1ULL);
    atomicAdd(&bfs_trace_state.submission_sum, n);
}
__device__ inline void bfs_trace_io_end() {
    if (bfs_trace_state.events) atomicAdd(&bfs_trace_state.outstanding, ~0ULL);
}
class BfsTrace {
    std::string prefix, stem;
    std::ofstream pages, levels;
    std::set<unsigned long long> seen;
public:
    BfsTrace(unsigned long long page_bytes, int impl, int mem, int controllers) {
        const char* p = std::getenv("BFS_TRACE_PREFIX");
        if (!p) return;
        if (impl != 9 || mem != 6 || controllers != 1 || page_bytes % 4096 || (page_bytes & (page_bytes-1)))
            throw std::runtime_error("trace requires impl=9, memalloc=6, n_ctrls=1, page_size multiple of 4096");
        prefix = p;
        unsigned long long capacity = 4*1024*1024;
        if (const char* c = std::getenv("BFS_TRACE_EVENTS")) {
            size_t used = 0;
            capacity = std::stoull(c, &used);
            if (used != std::string(c).size() || c[0] == '-') throw std::runtime_error("invalid BFS_TRACE_EVENTS");
        }
        if (!capacity || capacity > SIZE_MAX/sizeof(BfsPageEvent)) throw std::runtime_error("invalid trace capacity");
        BfsPageEvent* ptr;
        bfs_trace_check(cudaMalloc((void**)&ptr, capacity * sizeof(BfsPageEvent)));
        bfs_trace_state.events = ptr;
        bfs_trace_state.capacity = capacity;
        bfs_trace_state.page_bytes = page_bytes;
    }
    void run(int r) {
        if (prefix.empty()) return;
        seen.clear(); pages.close(); levels.close();
        stem = prefix + "_run_" + std::to_string(r);
        if (std::ifstream(stem + "_pages.csv").good() || std::ifstream(stem + "_complete").good())
            throw std::runtime_error("trace output exists; use next run prefix");
        pages.open(stem + "_pages.csv", std::ios::trunc);
        levels.open(stem + "_levels.csv", std::ios::trunc);
        if (!pages || !levels) throw std::runtime_error("cannot open trace output");
        pages << "level,sequence,page_id,page_bytes,edge_accesses\n";
        levels << "level,frontier_size,edge_accesses,unique_pages,page_reuse_ratio,cross_level_page_reuse_ratio,outstanding_page_requests_peak,outstanding_at_submission_mean,page_submissions\n";
    }
    void begin() {
        if (prefix.empty()) return;
        bfs_trace_check(cudaDeviceSynchronize());
        bfs_trace_state.count = bfs_trace_state.outstanding = bfs_trace_state.peak = 0;
        bfs_trace_state.submissions = bfs_trace_state.submission_sum = 0;
    }
    void finish(unsigned level, unsigned long long frontier) {
        if (prefix.empty()) return;
        bfs_trace_check(cudaDeviceSynchronize());
        if (bfs_trace_state.count > bfs_trace_state.capacity) throw std::runtime_error("BFS trace overflow: discard this run and increase BFS_TRACE_EVENTS");
        if (bfs_trace_state.outstanding) throw std::runtime_error("unbalanced outstanding counter");
        std::vector<BfsPageEvent> events(bfs_trace_state.count);
        bfs_trace_check(cudaMemcpy(events.data(), bfs_trace_state.events, events.size()*sizeof(BfsPageEvent), cudaMemcpyDeviceToHost));
        std::set<unsigned long long> unique;
        unsigned long long edges = 0, reused = 0;
        for (size_t i=0; i<events.size(); ++i) {
            const auto& e=events[i]; edges += e.edges; unique.insert(e.page);
            pages << level << ',' << i << ',' << e.page << ',' << bfs_trace_state.page_bytes << ',' << e.edges << '\n';
        }
        for (auto p:unique) if (seen.count(p)) ++reused;
        seen.insert(unique.begin(), unique.end());
        levels << level << ',' << frontier << ',' << edges << ',' << unique.size() << ','
               << (events.empty()?0.0:1.0-double(unique.size())/events.size()) << ','
               << (unique.empty()?0.0:double(reused)/unique.size()) << ',' << bfs_trace_state.peak << ','
               << (bfs_trace_state.submissions?double(bfs_trace_state.submission_sum)/bfs_trace_state.submissions:0.0) << ',' << bfs_trace_state.submissions << '\n';
        pages.flush(); levels.flush();
        if (!pages || !levels) throw std::runtime_error("trace output write failed");
    }
    void complete() {
        if (prefix.empty()) return;
        std::ofstream done(stem + "_complete");
        done << "complete\n";
        done.flush();
        if (!done) throw std::runtime_error("cannot write completion marker");
    }
    ~BfsTrace() { if (!prefix.empty()) cudaFree(bfs_trace_state.events); }
};
#define BFS_TRACE_IO_BEGIN() bfs_trace_io_begin()
#define BFS_TRACE_IO_END() bfs_trace_io_end()
