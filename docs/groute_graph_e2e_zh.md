# G-Route BFS/PageRank 端到端评估

## 1. 评估目标与公平边界

`scripts/graph/groute_graph_e2e.cu` 是同一个 CUDA 图执行器。BFS 与 PageRank 在所有实验中使用相同的 CSR、GPU kernel、staging 容量、批大小和线程配置；矩阵中唯一变化的是 KvikIO 的执行策略。因此主结果回答的是：在图算法及其数据依赖不变时，G-Route 相对原生 KvikIO 固定阈值能否降低邻接表加载时间并改善端到端性能。

执行器保留一个贯穿完整算法生命周期的 `kvikio::FileHandle`。每一层 BFS frontier 或每轮 PageRank active set 被排序后，按照 CSR 中的真实邻接区间生成逻辑请求：

```text
file_offset = 16 + col[v] * 8
size        = (col[v + 1] - col[v]) * 8
```

两个 GPU staging slot 构成双缓冲。slot A 上的邻接数据参与 GPU 计算时，slot B 可以提交下一批 KvikIO 请求。计算完成后不把边数据复制回 CPU；BFS frontier 和 PageRank active set 只在每轮边界返回主机，以形成下一轮数据依赖。

### Host Cache 命中路径分段计时和批量读取

图执行器对每个 staging slot 调用一次 `FileHandle::pread_batch`。完成前 64 次请求的原有 profiling 后，Host Cache 对同一批请求的缓存行执行查找和准入；缓存行在持锁期间固定，锁释放后将命中数据排入该 slot 的 CUDA stream，整批只等待一次拷贝完成。未命中请求沿用原 `pread` 后端；一条逻辑请求的准入/策略观察只发生一次。单请求 `pread` 也复用相同的锁外拷贝逻辑。

使用 `KVIKIO_HOST_CACHE_PROFILE=1` 启用计时；时间字段为文件生命周期累计纳秒：`lookup_wait_ns`（等锁）、`lookup_ns`（查找、元数据、缓存分配，剔除存储读时间）、`storage_read_ns`（miss 填充）、`copy_submit_ns`（H2D 提交）、`completion_wait_ns`（等待 CUDA stream）。`batch_calls` 统计多请求批次，`batch_cache_reads` 统计该批次已完成的缓存拷贝请求数，`copy_completions` 统计同步次数；未开启开关时计时字段为零。时间包含不同线程可能重叠的阶段，不能直接相加当作算法耗时。

构建并安装本分支 `libkvikio`，重新编译图执行器后，在小图上先验证正确性和计时：

```bash
bash scripts/build_graph_e2e.sh
KVIKIO_HOST_CACHE_PROFILE=1 \
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/bel/GAP-kron-s20.bel \
RESULT_ROOT=/mnt/gds/results/groute-cache-batch-smoke \
ALGORITHMS="bfs pagerank" POLICIES="kvikio_threshold host_direct host_cache" \
REPEATS=1 BFS_MAX_LEVELS=4 PAGERANK_ITERATIONS=2 \
PAGE_CACHE_MODE=none PLOT=0 bash scripts/run_graph_e2e_matrix.sh
```

矩阵脚本验证同算法不同策略的 trace/result 后，查看 `raw_results.csv` 中的 `cache_*_ns`、`cache_batch_calls` 和 `cache_batch_reads`。在 `host_cache` 策略中，当小请求复用并达到准入阈值时，预期 `cache_batch_reads > 0` 且 `cache_copy_completions < cache_batch_reads`；若该图/这些迭代尚无命中，增大 `BFS_MAX_LEVELS` 或 `PAGERANK_ITERATIONS`。计时开关会引入额外取时钟开销，论文性能对比应关闭计时，单独使用上述计时实验做开销归因。若需要测量冷 SSD，请使用 `PAGE_CACHE_MODE=global`（root）和固定预热/冷态边界，避免把 Page Cache 命中误判成 Host Cache 收益。

主表包含六个策略：

| 名称 | 控制方式 | 含义 |
|---|---|---|
| `kvikio_threshold` | `KVIKIO_GROUTE_ENABLED=0` | 未修改的 KvikIO 阈值派发基线 |
| `auto` | `KVIKIO_POLICY_MODE=auto` | G-Route 自动选择 |
| `host_direct` | 强制策略 | Host-mediated，无 G-Route cache |
| `host_cache` | 强制策略 | region-level host cache |
| `gds_direct` | 强制策略 | 每请求 GDS |
| `gds_shaped` | 强制策略 | GDS 内部 bounded shaping |

BaM 的 `impl_type=4, memalloc=2` 使用另一套执行器和计时边界，输出到独立的参考表，不能直接充当主表中 G-Route 的归一化分母。

## 2. 构建

先构建并安装当前 G-Route 分支的 libkvikio，然后构建图执行器：

```bash
conda activate kvikio-dev
cd /home/cwd/gds/kvikio_opt/kvikio

bash scripts/build_graph_e2e.sh
```

如果 KvikIO 安装在非 conda 前缀：

```bash
CMAKE_PREFIX_PATH=/path/to/kvikio/install \
GRAPH_BUILD_DIR=/tmp/groute-graph-build \
bash scripts/build_graph_e2e.sh
```

## 3. 冒烟测试

本地数据的图前缀为：

```bash
export GRAPH_PREFIX=/home/cwd/dataset/bafsdata/bel/GAP-kron-s20.bel
```

先分别执行一次短 BFS 和短 PageRank：

```bash
KVIKIO_GROUTE_ENABLED=1 \
KVIKIO_POLICY_MODE=auto \
KVIKIO_HOST_CACHE=1 \
KVIKIO_REQUEST_SHAPING=1 \
build/graph-e2e/groute_graph_e2e \
  --algorithm bfs --graph "$GRAPH_PREFIX" --policy auto \
  --source 1 --max-iterations 3 \
  --output /tmp/groute-bfs-smoke.json

KVIKIO_GROUTE_ENABLED=1 \
KVIKIO_POLICY_MODE=auto \
KVIKIO_HOST_CACHE=1 \
KVIKIO_REQUEST_SHAPING=1 \
build/graph-e2e/groute_graph_e2e \
  --algorithm pagerank --graph "$GRAPH_PREFIX" --policy auto \
  --max-iterations 2 \
  --output /tmp/groute-pagerank-smoke.json
```

检查 JSON 中 `processed_edges > 0`、BFS 的 `visited_vertices > 1`，以及 PageRank 的 `rank_sum` 为有限正数。

## 4. 完整六策略矩阵

建议先用 3 次重复确认稳定，再生成论文结果时使用至少 10 次重复：

```bash
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/bel/GAP-kron-s20.bel \
RESULT_ROOT=/mnt/gds/results/groute-graph-e2e \
REPEATS=10 \
BFS_MAX_LEVELS=100 \
PAGERANK_ITERATIONS=10 \
KVIKIO_NTHREADS=4 \
KVIKIO_THRESHOLD_BYTES=16384 \
PAGE_CACHE_MODE=global \
bash scripts/run_graph_e2e_matrix.sh
```

`PAGE_CACHE_MODE=global` 需要 root，并在每个策略进程前清除 Linux page cache。没有 root 时使用 `PAGE_CACHE_MODE=file`；论文必须报告所用模式。

### 按 BFS 层检查策略是否被启动阶段误导

更新并重新构建图执行器后，可先只运行三个关键策略；使用新的结果目录，不混入旧版 JSON：

```bash
./build.sh libkvikio kvikio --pydevelop
bash scripts/build_graph_e2e.sh
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/mtx_all/GAP-kron.bel \
RESULT_ROOT=/mnt/gds/results/groute-kron-bfs-levels \
ALGORITHMS=bfs \
POLICIES="kvikio_threshold auto host_direct" \
REPEATS=1 BFS_SOURCE=1 BFS_MAX_LEVELS=100 \
BATCH_REQUESTS=1024 PAGE_CACHE_MODE=global PLOT=0 \
bash scripts/run_graph_e2e_matrix.sh
```

每完成一层，`e2e_bfs_<policy>_r1.log` 会立即打印一行 `BFS layer=...`。完成后，
每个 JSON 的 `bfs_levels` 数组记录该层的 `logical_requests`、
`average_io_size`、精确的 `p50_io_size`、`io_seconds`、`layer_seconds`、
`policy_start` 和 `policy_end`。零邻接请求的层将平均值、中位数及 I/O 时间记录为零。
原生 KvikIO 阈值基线的策略显示为 `KVIKIO_THRESHOLD`，因为它按请求大小派发，
没有 FileHandle 级别的固定路径。

`io_seconds` 是主机线程调用 `pread` 提交请求及等待 futures 完成时的累计时间；
同步 Host Cache 读也在其中。它不包含等待 GPU kernel 完成的时间，I/O 与 kernel
可能重叠，不能用作独立的 SSD 读取时间。`layer_seconds` 是从开始排序 frontier
到下一层 frontier 拷回主机的墙钟时间，包含请求规划、I/O 和 GPU 计算。

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/gds/results/groute-kron-bfs-levels')
for path in sorted(root.glob('e2e_bfs_*.json')):
    result = json.loads(path.read_text())
    print('\n', result['policy_mode'])
    print('layer  requests      mean_B   p50_B   io_s     layer_s  policy_start -> policy_end')
    for level in result['bfs_levels']:
        print(f"{level['layer']:>5}  {level['logical_requests']:>12} "
              f"{level['average_io_size']:>8.1f} "
              f"{level['p50_io_size']:>7.1f} "
              f"{level['io_seconds']:>8.2f} "
              f"{level['layer_seconds']:>8.2f}  "
              f"{level['policy_start']} -> {level['policy_end']}")
PY
```

比较 Auto 第一层及后续各层的请求大小和策略即可确认早期采样是否代表主体工作量；
比较同一层三个策略的 `layer_seconds` 才能判断路径选择带来的端到端影响。

### 仅用于验证的阶段切换原型

`auto_phase` 和 `auto` 使用相同的前 64 次请求画像。区别是 BFS 每层结束、两组 staging
slot 的 futures 与 CUDA event 都完成之后，`auto_phase` 检查刚结束的这一层：若至少
`PHASE_MIN_REQUESTS` 个有效邻接读取，且该层请求大小的 p50 小于 `PHASE_P50_BYTES`，
就把**同一个** `FileHandle` 后续层的执行策略切成
`HOST_MEDIATED/BYPASS/DIRECT`。默认门槛分别为 100,000 次与 16 KiB。
它只切换一次，不影响其他策略或 PageRank。`switched_after_layer` 记录发生切换的层号，
JSON 每层的 `policy_end` 表示下一层将使用的策略。

先用相同的全量 BFS 数据、同一份执行器进行 3 次重复：

```bash
./build.sh libkvikio kvikio --pydevelop
bash scripts/build_graph_e2e.sh
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/mtx_all/GAP-kron.bel \
RESULT_ROOT=/mnt/gds/results/groute-kron-bfs-phase-r3 \
ALGORITHMS=bfs \
POLICIES="kvikio_threshold auto auto_phase host_direct" \
REPEATS=3 BFS_SOURCE=1 BFS_MAX_LEVELS=100 \
BATCH_REQUESTS=1024 KVIKIO_NTHREADS=4 \
PHASE_MIN_REQUESTS=100000 PHASE_P50_BYTES=16384 \
PAGE_CACHE_MODE=global PLOT=1 \
bash scripts/run_graph_e2e_matrix.sh
```

检查 `failed_runs.txt` 为空且相同 repeat 中四个策略的 `logical_trace_hash`、
`result_hash` 一致；汇总器会自动执行这些检查。查看 `summary.csv` 的时间中位数，
以及 `e2e_bfs_auto_phase_r*.json` 中的 `switched_after_layer`、`bfs_levels`。
完整 GAP-kron 预期第 3 层结束后切换；若实际换到其他层，应按 JSON 的 p50 和请求数解释。
图中的 `auto_phase` 明确标为实验性的阶段策略，不代表 G-Route 当前默认 Auto 行为。

该原型的门槛来自上述单条 BFS trace，意在验证安全切换能否追回第 4、5 层的时间。
在其他源点、GAP-urand 和不同缓存状态下验证之前，不能作为最终的通用在线算法。

### 新版批量缓存：完整图 BFS 对照

使用完整 GAP-kron 的 `.col/.dst`（不要用 s20），保持图、源点、执行器及预热/清缓存
方式一致。性能运行不打开缓存分段计时，使用全新结果目录；每组运行前脚本会按
`PAGE_CACHE_MODE=global` 清除 Linux page cache（需要 root）。

```bash
./build.sh libkvikio kvikio --pydevelop
bash scripts/build_graph_e2e.sh
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/mtx_all/GAP-kron.bel \
RESULT_ROOT=/mnt/gds/results/groute-kron-bfs-batch-full \
ALGORITHMS=bfs \
POLICIES="kvikio_threshold auto auto_phase host_direct host_cache" \
REPEATS=3 BFS_SOURCE=1 BFS_MAX_LEVELS=100 \
BATCH_REQUESTS=1024 STAGING_BYTES=268435456 \
KVIKIO_NTHREADS=4 HOST_CACHE_BYTES=1073741824 \
KVIKIO_HOST_CACHE_PROFILE=0 PAGE_CACHE_MODE=global PLOT=0 \
bash scripts/run_graph_e2e_matrix.sh
```

检查 `failed_runs.txt` 为空；脚本汇总时要求同一次 repeat 的 BFS
`logical_trace_hash`、`logical_requests` 和 `result_hash` 一致。分别比较 `summary.csv`
的 `algorithm_seconds_median`，以及 JSON `bfs_levels` 中第 4/5 层的
`layer_seconds`。重点检查 `host_cache` 的 `cache.hits`、`cache.batch_calls`、
`cache.batch_cache_reads`、`cache.copy_completions`，确认缓存命中确实使用批量路径。
报告 `host_cache` 对 `host_direct` 的性能和分层差值；若仍较慢，单独重复一次
`KVIKIO_HOST_CACHE_PROFILE=1` 的诊断运行，查看 `cache.lookup_wait_ns`、
`cache.lookup_ns`、`cache.copy_submit_ns`、`cache.completion_wait_ns` 和
`cache.storage_read_ns`。这些是各请求累计计时，可能与线程并发重叠，不能直接
相加为端到端耗时。`host_direct` 指绕过 G-Route Host Cache，不等于强制 POSIX
`O_DIRECT`。完成 GAP-kron 后换成 `GAP-urand.bel` 并使用新目录复现。

### PageRank：逐顶点数值正确性

旧版汇总器比较 rank sum 和粗量化的 result hash；这无法发现少数顶点的数值差异。
新版执行器的可选 `--rank-output` 把最终 float32 rank 向量写到 `.ranks.f32`，
矩阵脚本在相同 repeat 的逻辑 trace、顶点数和迭代轮数一致后，与 `host_direct`
逐顶点比较。先使用 s20 做完整迭代验证：

```bash
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/bel/GAP-kron-s20.bel \
RESULT_ROOT=/mnt/gds/results/groute-kron-s20-pagerank-ranks \
ALGORITHMS=pagerank \
POLICIES="host_direct host_cache auto kvikio_threshold" \
REPEATS=1 PAGERANK_ITERATIONS=10 BATCH_REQUESTS=1024 \
VERIFY_PAGERANK_RANKS=1 RANK_REFERENCE_POLICY=host_direct \
PAGE_CACHE_MODE=global PLOT=0 \
bash scripts/run_graph_e2e_matrix.sh
cat /mnt/gds/results/groute-kron-s20-pagerank-ranks/pagerank_rank_comparison.csv
```

判定为每个顶点 `|rank - reference| <= atol + rtol * |reference|`；默认
`atol=1e-3/vertex_count`、`rtol=1e-3`，在 CSV 中输出实际阈值、最大绝对误差、
相对 L1 误差和超限顶点数。出现超限或缺少向量文件时脚本退出非零；应检查数值
差异与 trace，不要靠增大容差掩盖错误。完整 GAP-kron 可在小图通过后沿用
`GRAPH_PREFIX=/home/cwd/dataset/bafsdata/mtx_all/GAP-kron.bel`，先设
`PAGERANK_ITERATIONS=1` 验证资源消耗，再用 10 轮与完整图性能实验的设置一致。
完整图每个策略/重复的 rank 向量约 512 MiB；预留结果目录空间。
逐顶点检查单独运行：向量写盘不计入程序报告的计时，但会影响随后实验的
SSD/page cache 状态；性能矩阵应在另一个新目录用 `VERIFY_PAGERANK_RANKS=0` 重跑。

脚本生成：

- `raw_results.csv`：每次运行的原始数据；
- `summary.csv`：中位数、标准差、相对 KvikIO threshold 的 speedup；
- `graph_e2e.pdf` 和 `graph_e2e.png`：BFS/PageRank 对比图；
- `failed_runs.txt`：失败运行（全通过时为空或不存在）。

汇总器会验证同一算法、同一次重复下各策略具有相同的逻辑 trace hash、请求数和处理边数。BFS 还要求 visited bitmap hash 完全一致；PageRank 汇总器要求 `rank_sum` 在 `1e-5` 相对误差内一致。逐顶点 rank 检查需要另外打开 `VERIFY_PAGERANK_RANKS=1`；不能只凭 rank sum 宣称逐顶点正确。

## 5. BaM 外部参考

按指定的 `COALESCE_PC=4, UVM_DIRECT=2` 运行：

```bash
BAM_BUILD_DIR=/home/cwd/src/bam/build \
GRAPH_PREFIX=/home/cwd/dataset/bafsdata/bel/GAP-kron-s20.bel \
RESULT_ROOT=/mnt/gds/results/bam-graph-reference \
BAM_IMPL_TYPE=4 BAM_MEMALLOC=2 REPEATS=10 \
bash scripts/run_bam_graph_baseline.sh
```

这会生成独立的 `raw_results.csv` 与 `summary.csv`。若要测量干净的 UVM coalescing 组合，可另设 `BAM_IMPL_TYPE=1 BAM_MEMALLOC=2`，并在论文中明确区分两组。

生成与指定 BaM baseline 的独立对比表和绝对时间图：

```bash
python scripts/compare_graph_e2e_baseline.py \
  --groute-summary /mnt/gds/results/groute-graph-e2e/summary.csv \
  --bam-summary /mnt/gds/results/bam-graph-reference/summary.csv \
  --groute-policy auto \
  --output-prefix /mnt/gds/results/graph-groute-vs-bam
```

输出为 `graph-groute-vs-bam.csv/.pdf/.png`。该图必须标注为“external reference”：BaM 是 GPU-initiated page-cache 执行器，而 G-Route 是 host-driven KvikIO loader，不能据此单独归因某一个 G-Route design。

## 6. 论文表格与结论规则

论文主表建议报告每个算法的 `algorithm_seconds_median`、`teps_median`、标准差和 `speedup_vs_kvikio_threshold`。图使用生成的归一化 speedup，原始秒数保留在表格中。

只有以下条件同时满足，才能声称端到端收益：

1. `auto` 的 `algorithm_seconds_median` 小于 `kvikio_threshold`；
2. 计算结果与逻辑请求校验通过；
3. `auto` 接近最优强制策略，结合 `fraction_of_best_forced` 报告差距；
4. 在至少一个超过 DRAM 或明显冷缓存的数据集上复现，而不只使用约 240 MiB 的 `GAP-kron-s20.bel.dst`。

当前 s20 数据适合功能正确性和策略消融，但其 edge payload 可进入 DRAM，不能单独证明 out-of-core 优势。最终论文还应加入完整 GAP-kron/GAP-urand 的 `.col/.dst`，并报告 SSD、GPU、DRAM 容量以及 cache-control 方法。
