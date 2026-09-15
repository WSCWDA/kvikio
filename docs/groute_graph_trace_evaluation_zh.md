# G-Route BFS/PageRank 真实 Trace 评估

本评估借鉴 BaM `benchmarks/bfs` 与 `benchmarks/pagerank` 的 CSR 数据格式和
4 KiB 页访问粒度，但不把 G-Route 伪装成 BaM 的 GPU-side page cache。
BaM 的 `BAFS_DIRECT` 在 CUDA kernel 内按需访问 SSD；G-Route 是 KvikIO 内的
host runtime，二者的提交位置不同。为了隔离 I/O policy 的影响，本评估先由
算法产生真实访问序列，再让所有 KvikIO/G-Route 策略回放同一个 trace。

## 1. 数据与实验口径

BaM 图由两个文件组成：

- `PREFIX.col`：带 16 B header 的 CSR row offsets；
- `PREFIX.dst`：带 16 B header 的 uint64 destination vertices。

现有数据对应的 prefix 是：

```bash
export DATASET_DIR=/path/to/bafsdata/bel
export GRAPH_PREFIX="$DATASET_DIR/GAP-kron-s20.bel"
```

即脚本实际读取 `GAP-kron-s20.bel.col` 和 `GAP-kron-s20.bel.dst`。
目录中的 32 GiB `GAP-kron.bel` 和 `GAP-urand.bel` 不能直接用于本脚本，除非
同时存在对应的 `.col` 和 `.dst`。另外，s20 的 `.dst` 只有约 240 MiB，适合
功能验证和真实 trace 对比，但不能证明 out-of-core working set 大于 DRAM；
论文最终结果应补充完整规模 CSR 文件。

### BFS trace

BFS 从指定 source 出发，按真实 frontier 查 CSR 邻接表。每个邻接表覆盖的
4 KiB 文件页按原顺序写入 trace；下一层 frontier 由实际 `.dst` 内容决定。
因此，BFS 的随机性、局部性和跨层复用来自图本身，不是合成 offset。

### PageRank trace

PageRank 使用与 BaM 相同的 residual-driven push 过程（`alpha=0.85`、
`tolerance=0.001`）：首轮激活所有顶点，之后仅由 residual 超过阈值的顶点
访问邻接表。trace 因而同时保存每轮顺序性、active set 的变化以及跨 iteration
重复引用，可用于验证 G-Route 能否识别“单轮扫描、跨轮复用”的相变。

## 2. 生成 trace

```bash
export RESULT_DIR=/path/to/results
mkdir -p "$RESULT_DIR/groute-graph-traces"

python scripts/graph/generate_graph_trace.py \
  --algorithm bfs \
  --graph "$GRAPH_PREFIX" \
  --source 1 \
  --page-size 4096 \
  --output "$RESULT_DIR/groute-graph-traces/kron-s20-bfs.trace"

python scripts/graph/generate_graph_trace.py \
  --algorithm pagerank \
  --graph "$GRAPH_PREFIX" \
  --iterations 10 \
  --page-size 4096 \
  --output "$RESULT_DIR/groute-graph-traces/kron-s20-pagerank.trace"
```

每个 trace 同时生成 `.trace.json`。其中包含 SHA-256、请求数、BFS 层数和访问
每层请求数，或 PageRank 每轮请求数。矩阵汇总器会拒绝同一算法内 hash 不同
的结果，防止策略间误用不同 trace。

## 3. 冒烟测试与完整矩阵

冒烟测试先限制 trace 长度：

```bash
GRAPH_PREFIX="$GRAPH_PREFIX" \
TRACE_ROOT="$RESULT_DIR/groute-graph-smoke-traces" \
RESULT_ROOT="$RESULT_DIR/groute-graph-smoke" \
MAX_TRACE_REQUESTS=1024 \
REPEATS=1 \
POLICIES="kvikio_threshold auto" \
bash scripts/run_graph_trace_matrix.sh
```

完整 s20 矩阵：

```bash
GRAPH_PREFIX="$GRAPH_PREFIX" \
TRACE_ROOT="$RESULT_DIR/groute-graph-traces" \
RESULT_ROOT="$RESULT_DIR/groute-graph-matrix" \
REGENERATE_TRACES=1 \
REPEATS=10 \
BATCH_SIZE=32 \
KVIKIO_NTHREADS=8 \
PAGE_CACHE_MODE=global \
POLICIES="kvikio_threshold auto host_direct host_cache gds_direct gds_shaped" \
bash scripts/run_graph_trace_matrix.sh
```

`PAGE_CACHE_MODE=global` 需要 root。矩阵在不同 repeat 中轮转 policy 顺序，且
每个 policy 使用新进程、新 `FileHandle` 和相同 trace。结果位于：

- `raw_results.csv`：逐次运行结果；
- `summary.csv`：中位数、标准差、P99 batch latency、cache hit ratio、
  storage bytes 和 shaped physical requests；
- `failed_runs.txt`：仅在失败时非空。

原始 KvikIO 固定阈值基线由 `kvikio_threshold` 提供：它在同一个环境中设置
`KVIKIO_GROUTE_ENABLED=0` 的等效配置，并保留真实 `gds_threshold=16 KiB`。
由于 trace 请求为 4 KiB，该基线会走原始 KvikIO 的 host path。

## 4. BaM UVM baseline

按指定配置运行 BaM 的 `COALESCE_PC=4`、`UVM_DIRECT=2`：

```bash
BAM_BUILD_DIR=/path/to/bam/build \
GRAPH_PREFIX="$GRAPH_PREFIX" \
BFS_SOURCE=1 \
REPEATS=10 \
RESULT_ROOT="$RESULT_DIR/bam-uvm-baseline" \
bash scripts/run_bam_graph_baseline.sh
```

该结果用于报告 BaM 程序端到端运行时间；不要用它除以 trace replay 时间生成
speedup，因为前者包含图计算，后者只测同 trace 的 I/O service time。论文中
应分别报告：

1. BaM UVM end-to-end 时间，作为既有系统参考；
2. 同 trace 下 KvikIO threshold 与 G-Route 的 I/O 时间，证明 runtime policy；
3. 后续将 G-Route loader 接入同一图执行器后的真正端到端时间。

## 5. 论文应报告的结论链

不能只报告 IOPS。至少同时展示：

1. `trace_sha256` 相同，证明工作量一致；
2. Auto 最终选择的 `path/cache/submit`；
3. cache hit ratio、storage bytes 与 physical requests；
4. I/O service time、P99 batch latency；
5. BFS 不同 source、PageRank 不同 iteration 数下的敏感性。

合理的因果结论应是：真实图 trace 的局部性或跨轮复用改变 G-Route 的策略，
进而减少 SSD 请求/传输字节，最终降低 I/O service time。若 Auto 对 PageRank
一直保持 `SEQUENTIAL_SCAN/GPU_DIRECT`，这应作为当前仅用前 64 请求 profiling
无法识别跨 iteration phase change 的边界，而不是隐藏该结果。
