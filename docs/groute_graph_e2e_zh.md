# G-Route BFS/PageRank 端到端评估

## 1. 评估目标与公平边界

`scripts/graph/groute_graph_e2e.cu` 是同一个 CUDA 图执行器。BFS 与 PageRank 在所有实验中使用相同的 CSR、GPU kernel、staging 容量、批大小和线程配置；矩阵中唯一变化的是 KvikIO 的执行策略。因此主结果回答的是：在图算法及其数据依赖不变时，G-Route 相对原生 KvikIO 固定阈值能否降低邻接表加载时间并改善端到端性能。

执行器保留一个贯穿完整算法生命周期的 `kvikio::FileHandle`。每一层 BFS frontier 或每轮 PageRank active set 被排序后，按照 CSR 中的真实邻接区间生成逻辑请求：

```text
file_offset = 16 + col[v] * 8
size        = (col[v + 1] - col[v]) * 8
```

两个 GPU staging slot 构成双缓冲。slot A 上的邻接数据参与 GPU 计算时，slot B 可以提交下一批 KvikIO 请求。计算完成后不把边数据复制回 CPU；BFS frontier 和 PageRank active set 只在每轮边界返回主机，以形成下一轮数据依赖。

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

脚本生成：

- `raw_results.csv`：每次运行的原始数据；
- `summary.csv`：中位数、标准差、相对 KvikIO threshold 的 speedup；
- `graph_e2e.pdf` 和 `graph_e2e.png`：BFS/PageRank 对比图；
- `failed_runs.txt`：失败运行（全通过时为空或不存在）。

汇总器会验证同一算法、同一次重复下各策略具有相同的逻辑 trace hash、请求数和处理边数。BFS 还要求 visited bitmap hash 完全一致；PageRank 要求 `rank_sum` 在 `1e-5` 相对误差内一致。验证失败时不会产生可用于论文的结果。

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
