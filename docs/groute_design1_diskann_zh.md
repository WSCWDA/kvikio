# G-Route Design 1 与 DiskANN 端到端评估

本文档给出两组互补实验。`run_design1_policy.sh` 验证 IOContext 能否把明确的
访问特征映射到预期策略；`run_diskann_groute_e2e.sh` 使用 DiskANN 生成的 SSD
索引，在 GustANN Hybrid GPU 搜索中比较原生 AIO 与 G-Route。后者测量的是完整
ANN 查询，而不是脱离应用的 4 KiB I/O 微基准。

## 1. Design 1 策略验证

准备一个至少 128 MiB、位于 GDS 文件系统上的真实文件（不能只创建稀疏文件）：

```bash
dd if=/dev/urandom of=/mnt/gds/groute-design1.bin bs=1M count=128 status=progress
```

运行四类访问模式，每类重复五次：

```bash
DESIGN1_FILE=/mnt/gds/groute-design1.bin \
RESULT_ROOT=/mnt/gds/results/groute-design1 \
REPEATS=5 REQUESTS=4096 BATCH_SIZE=32 \
bash scripts/run_design1_policy.sh
```

四种 pattern 的前 64 个请求是有意构造的，不依赖随机碰撞：

| Pattern | 预期 IOContext 策略 |
|---|---|
| 大块顺序扫描 | `GPU_DIRECT/BYPASS/DIRECT` |
| 小块、跨 region 的冷随机访问 | `HOST_MEDIATED/ADMIT/DIRECT`；region admission 仍可拒绝一次性数据 |
| 小块、热点 region 复用 | `HOST_MEDIATED/ADMIT/DIRECT` |
| 小块、非对齐且可合并 | `GPU_DIRECT/BYPASS/SHAPED` |

若策略不符，单次运行直接失败。结果位于 `raw_results.csv` 与 `summary.csv`；策略
正确性是本实验的首要结论，IOPS 只用于衡量选择开销，不应把四种不同 workload
之间的性能直接互比。

## 2. DiskANN/GustANN 端到端实验

### 2.1 为什么采用这一集成点

Microsoft DiskANN 的 `build_disk_index` 负责生成标准 SSD index、PQ 数据等文件；
GustANN 能直接解析这些文件，并把图搜索中的 4 KiB page 请求交给可替换的
`IndexLoader`。因此补丁新增 `groute` loader，让页面直接进入 GPU buffer，同时
保留 `aio` loader 作为 Host-mediated 基线。

这与 GustANN 的 BaM 路径不同：BaM 从 GPU kernel 发起对裸设备的访问，需要定制
内核模块，不能由 host-side KvikIO API 透明拦截。这里复用的是 GustANN 的 Hybrid
GPU 搜索计算路径，以便在普通文件系统上的同一 DiskANN index 上做等价对比。

### 2.2 构建

准备 GustANN 源码。若要严格复现实验，可固定到已验证提交
`dd8e70b2b1b5d22acb2373685fdc416511e73b6d`：

```bash
git clone https://github.com/thustorage/GustANN.git /opt/GustANN
git -C /opt/GustANN checkout dd8e70b2b1b5d22acb2373685fdc416511e73b6d

GUSTANN_DIR=/opt/GustANN \
CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
bash scripts/setup_gustann_groute.sh
```

该脚本只给外部 GustANN checkout 应用
`scripts/diskann/gustann-groute.patch`，并复制 `groute_loader.cpp`；不会把 GustANN
或 DiskANN vendor 到 KvikIO 仓库。生成程序为
`/opt/GustANN/build-groute/bin/search_disk_hybrid`。

### 2.3 索引与导航图

使用 WSCWDA/DiskANN `cpp_main` 或 GustANN 的 DiskANN submodule 生成索引：

```bash
build/apps/build_disk_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix /mnt/gds/diskann/index_R128_L200 \
  --data_path /data/base.fbin \
  -B 32 -M 64 -R 128 -L 200
```

端到端运行需要以下匹配的文件：

- `/mnt/gds/diskann/index_R128_L200_disk.index`；
- PQ prefix `/mnt/gds/diskann/index_R128_L200_pq`；
- query `.fvecs`（`float`）或 `.bvecs`（`uint8`）；
- ground truth `.ivecs`；
- GustANN navigation graph 目录。

navigation graph 可按 GustANN 的 `scripts/gen_pivot_graph.sh` 生成：先从原始 `.fbin`
或 `.bbin` 采样，再调用 DiskANN `build_memory_index`，最终目录包含
`nav_index`、`nav_index.tags` 等文件。

### 2.4 运行

```bash
GUSTANN_DIR=/opt/GustANN \
INDEX_FILE=/mnt/gds/diskann/index_R128_L200_disk.index \
PQ_PREFIX=/mnt/gds/diskann/index_R128_L200_pq \
NAV_GRAPH=/mnt/gds/diskann/nav \
QUERY_FILE=/data/query.fvecs \
GT_FILE=/data/groundtruth.ivecs \
DATA_TYPE=float TOPK=10 EF_SEARCH=100 \
MINIBATCH=32 SEARCH_THREADS=4 CTX_PER_THREAD=4 \
QUERY_REPEATS=1 REPEATS=5 \
RESULT_ROOT=/mnt/gds/results/groute-diskann \
bash scripts/run_diskann_groute_e2e.sh
```

脚本交替 `aio→groute` 与 `groute→aio` 的执行顺序，减少热机和设备状态漂移。
需要冷缓存实验时以 root 设置 `DROP_CACHES=1`。`summary.csv` 汇总中位 QPS、线程
平均查询延迟、Recall、page reads，以及 G-Route 最终选择的
`path/cache/submit`。有效结论必须同时满足：

1. 两条路径的 Recall 一致；
2. G-Route 日志存在 `[GROUTE_STATS]`，证明不是静默回退到 AIO；
3. 至少报告 5 次运行的中位数和标准差；
4. 固定数据集、index、`TOPK`、`EF_SEARCH`、minibatch、线程与 context 数，仅改变 backend。

建议论文主图画 Recall--QPS 曲线：扫描多个 `EF_SEARCH`，每个点分别运行上述矩阵；
另用一张分解表报告所选 IOContext policy、cache hit、physical requests 与 p99 I/O
batch latency，以解释性能来自路径选择、region cache 还是 request shaping。
