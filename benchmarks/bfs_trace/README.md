# BFS trace → Host Cached / GDS replay（实验教程）

目的：先检验 BFS 的真实页需求是否存在路径选择空间，再决定是否移植端到端 BFS。
实现位于 KvikIO 测试分支，不改变 KvikIO 核心代码。BaM 修改由补丁工具应用到独立 checkout。

**这不是 G-Route 路由器，也不是端到端 BFS 对比。** trace replay 不执行 edge relaxation，
不保留 GPU warp 的时序、依赖或计算/I/O重叠。它用于筛查路径偏好，不能直接报告 BFS speedup。

## 1. 当前支持范围与度量

固定参考 BaM commit：`315fadfc5c5c018a64596157bfac94ecbb7d87a2`。
仅支持单 SSD、`--impl_type 9 --memalloc 6`（FRONTIER_COALESCE_PC），页大小为 >=4 KiB 的 2 的幂。
初始实验使用 4096 B。其它 BFS 实现及多 SSD 布局会报错，而非静默产生错位 trace。

| 输出 | 定义 |
|---|---|
| frontier_size | 该层传给 frontier kernel 的 `active`，不是 NVMe queue depth |
| edge_accesses | 该层实际 `seq_read(i)` 调用次数，warp 同页事件的 multiplicity 之和 |
| unique_pages | 该层逻辑页 ID 的去重数量 |
| page_reuse_ratio | `1 - unique_pages / warp-coalesced page demand events`，层内重复页需求比例 |
| cross_level_page_reuse_ratio | 本层 unique pages 中，在该 run 更早层出现过的比例 |
| outstanding_page_requests_peak | 数据页 `read_data()` 中，`sq_enqueue` 前至 `cq_poll` 返回后的软件请求计数峰值 |
| outstanding_at_submission_mean | 每次数据页提交入口观察到的 outstanding 计数均值；不是时间加权均值 |
| page_submissions | 该层实际进入 `read_data()` 的数据页读取次数（GPU cache miss 后） |

逻辑页事件为 `(level, sequence, page_id, page_bytes, edge_accesses)`，不是 8 B 的物理 SSD I/O。
同一 warp 在同一次访问中触及同一页的 lanes 合并为一个事件；保留重复事件，以便 replay 共同的 GPU cache。
序号通过全局 atomic 分配，只代表 instrumentation 的线性化顺序，不等同于真实 I/O 到达时间。
不采集 wall-clock timestamp。

**outstanding 包含等待 SQ 空位的时间，不能称为硬件 queue depth。**
BaM `enqueue_second()` 发出的额外单 block 队列推进读没有计入数据页指标，也没有 replay。
因此页需求与数据页 outstanding 可以对应，但不等于 BaM 全部 NVMe 命令的统计。

层内重复需求及跨层复用均不等于 Host cache hit ratio；页已留在 GPU cache 时不会访问 Host。
采集增加 atomic、GPU 内存写和层间同步，可能改变调度和 outstanding；不要将 traced BFS 的耗时当作性能结果。
每次 run 独立输出，只有 `_complete` 标记存在才允许 replay；溢出或写入失败的 trace 必须整次丢弃。

## 2. 获取测试分支

```bash
git clone --branch codex/bfs-trace-replay https://github.com/WSCWDA/kvikio.git kvikio-bfs
cd kvikio-bfs
export KVIKIO_BFS_ROOT="$PWD"
python3 -m unittest discover -s benchmarks/bfs_trace/tests -v
```

Replay 工具独立于 core，可直接使用现有 KvikIO 26.6 + CuPy 环境，不需要为了该测试重新编译 KvikIO。
先检查：

```bash
python3 -c 'import cupy, kvikio; print(cupy.__version__, kvikio.__version__, kvikio.__file__)'
nvidia-smi
command -v nvcc
```

若需要编译本分支 KvikIO core，注意 base main 的 Python 包要求 **Python >=3.11**，默认依赖指向 CUDA 13。
不要直接把它覆盖到现有 Python 3.10 / CUDA 12 的实验环境。
使用匹配 main 的独立开发环境，先按根目录 `CONTRIBUTING.md` / `dependencies.yaml` 安装构建依赖，再运行：

```bash
./build.sh libkvikio kvikio --pydevelop
```

以上是 core 的官方构建入口；本测试的必需 CUDA 编译发生在下面的 BaM 项目中。

## 3. 给 BaM BFS 加 trace 并编译

```bash
cd "$(dirname "$KVIKIO_BFS_ROOT")"
git clone https://github.com/ZaidQureshi/bam.git bam-bfs-trace
cd bam-bfs-trace
git checkout 315fadfc5c5c018a64596157bfac94ecbb7d87a2
git submodule update --init --recursive
python3 "$KVIKIO_BFS_ROOT/benchmarks/bfs_trace/apply_bam_trace.py" "$PWD"
git diff --check
cmake -S . -B build
cmake --build build --target libnvm -j 8
cmake --build build --target benchmarks -j 8
```

如自动检测 NVIDIA driver 源码失败，在 cmake 配置时加
`-DNVIDIA=/usr/src/nvidia-<实际驱动版本>/`，遵循 BaM README 的要求。
运行前先查看 `build/bin/nvm-bfs-bench --help`。
该采集模式依赖已配置好的 BaM P2P/libnvm 控制器，不能仅靠安装 CUDA 运行。
本教程不自动解绑 SSD、加载驱动或向裸设备写图；沿用已验证的 BaM 数据部署流程。
文件系统 replay 的 SSD 需要绑定 Linux NVMe 驱动，两种访问方式通常不能同时用于同一 SSD。
先采集，再在文件系统设备上 replay；也可使用另一块同型号 SSD并在报告中说明。

## 4. 采集真实 BFS

BaM 输入使用 `<graph>.bel.col` 和 `<graph>.bel.dst`。
`.col` 为 vertex offsets，`.dst` 包含 16 B header + uint64 edges。
采集前确保 NVMe 上 `--loffset` 指向同一 `.dst` 的**去掉 header 的 edge payload**；
BaM readwrite 部署时的 input offset 应为 16 B。禁止使用包含 header 的 raw layout。
该布局必须与原来可正确运行的 BaM BFS一致。

```bash
cd /path/to/bam-bfs-trace
mkdir -p results/bfs_path_opportunity/run_01
sudo env \
  BFS_TRACE_PREFIX="$PWD/results/bfs_path_opportunity/run_01/bfs" \
  BFS_TRACE_EVENTS=4194304 \
  ./build/bin/nvm-bfs-bench \
  -f /path/to/graph.bel --loffset 0 \
  --impl_type 9 --memalloc 6 --src 12345 \
  --n_ctrls 1 --page_size 4096 --gpu 0 --threads 128 \
  --maxPCSize 4194304 \
  > results/bfs_path_opportunity/run_01/bfs.log 2>&1
```

将 src 换成有效且能遍历较多顶点的根。**上游代码中 src != 0 时 total_run 固定为 2，
不受 repeat 控制**，所以会得到 run_0 和 run_1；初步用 run_0，run_1 含 BaM GPU cache 的预热影响。
这两个 run 是同一源节点；不要把它们当成不同根节点样本。
记录无 trace 的同配置 BFS 作为采集扰动参考，不把它与 replay 耗时混算。

输出：`bfs_run_0_pages.csv`、`bfs_run_0_levels.csv`、`bfs_run_0_complete`。
默认事件缓冲约 64 MiB。若出现 overflow，增大 BFS_TRACE_EVENTS并使用新的 run_02 目录重采；
不能用采样/截断 trace 做“同一完整 BFS”的路径比较。
GPU cache 在采集中为 4 MiB（1024 个 4 KiB 页），便于与下面 replay 容量一致。

## 5. 准备文件系统上的相同 edge 数据

```bash
cd "$KVIKIO_BFS_ROOT"
python3 benchmarks/bfs_trace/prepare_edges.py \
  /path/to/graph.bel.dst /mnt/gds/bfs_edges_4k.bin --page-bytes 4096
```

该程序去掉两个 uint64 header，只复制 `edge_count*8` B payload，并在末尾补零到 page 边界；
输出存在时拒绝覆盖。BaM不解释 header 的 typeT字段，本工具也不猜测其枚举含义。
源数据必须确实为 uint64 edge IDs。`page_id=p` 对应新文件偏移 `p*4096`，不使用裸设备 loffset。
只做读；没有 trace 的页不影响 replay。

## 6. 禁止 GDS 静默退回 Host

首先运行节点上安装的 `gdscheck -p`（常见路径 `/usr/local/cuda/gds/tools/gdscheck`）。
基于当前节点的 cufile.json 复制一份实验配置，保留其文件系统、设备及其它设置，仅关闭兼容回退：

```bash
mkdir -p results/bfs_path_opportunity/config
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('/etc/cufile.json').read_text())
cfg.setdefault('properties', {})['allow_compat_mode'] = False
Path('results/bfs_path_opportunity/config/cufile_direct.json').write_text(json.dumps(cfg, indent=2))
PY
```

若节点配置包含注释，先转换为严格 JSON再执行上述脚本。
Replay 的 GDS 模式要求提供该配置，强制 KvikIO compat_mode=False，并调用 raw_read 跳过 size threshold。
文件需要 O_DIRECT 支持，GPU buffers / file offsets / request sizes 均按 4 KiB 对齐，并注册 GPU buffers。
仍需用节点的 cuFile 统计/日志确认数据走 direct path；`is_direct_io_supported` 本身只证明 O_DIRECT 可用。
不能将 compatibility read 标为 GDS。现有节点若只有 compat mode 能运行，此步骤失败就是环境限制，
应先解决 GDS支持问题，不能改回 AUTO 后继续声称在比较原生 GDS。

## 7. 校验读取内容，再做性能 replay

把 trace 三个文件复制到 KvikIO 下的 `results/bfs_path_opportunity/trace/`，目录命名按实验目的，不用时间戳。

```bash
cd "$KVIKIO_BFS_ROOT"
export BFS_PAGES="$PWD/results/bfs_path_opportunity/trace/bfs_run_0_pages.csv"
export BFS_LEVELS="$PWD/results/bfs_path_opportunity/trace/bfs_run_0_levels.csv"
export BFS_DIRECT_CONFIG="$PWD/results/bfs_path_opportunity/config/cufile_direct.json"

python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file /mnt/gds/bfs_edges_4k.bin \
  --mode host-cached --gpu-cache-pages 1024 --qd trace --max-qd 64 --verify \
  --output results/bfs_path_opportunity/correctness/host.csv
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file /mnt/gds/bfs_edges_4k.bin \
  --mode gds --cufile-config "$BFS_DIRECT_CONFIG" \
  --gpu-cache-pages 1024 --qd trace --max-qd 64 --verify \
  --output results/bfs_path_opportunity/correctness/gds.csv
```

`--verify` 对每个 cache miss 逐字节比较 GPU 内容与文件。这会预热 Linux page cache、增加同步，
所以 correctness 输出不可做性能比较，compare.py 会拒绝它。

性能测试使用相同命令但去掉 --verify，将输出分别写为：

```bash
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file /mnt/gds/bfs_edges_4k.bin \
  --mode host-cached --gpu-cache-pages 1024 --qd trace --max-qd 64 \
  --output results/bfs_path_opportunity/run_01/host.csv
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file /mnt/gds/bfs_edges_4k.bin \
  --mode gds --cufile-config "$BFS_DIRECT_CONFIG" \
  --gpu-cache-pages 1024 --qd trace --max-qd 64 \
  --output results/bfs_path_opportunity/run_01/gds.csv
python3 benchmarks/bfs_trace/compare.py \
  results/bfs_path_opportunity/run_01/host.csv \
  results/bfs_path_opportunity/run_01/gds.csv \
  --output results/bfs_path_opportunity/run_01/comparison.csv
```

Host Cached 的实现为 **buffered preadv → pinned Host buffer → GPU**，复用来自 Linux page cache，
不是新增的用户态 Host cache / 准入策略。Host staging buffers / CUDA streams与GDS注册均在计时前预分配，避免逐请求分配干扰。
两路径共用 bounded GPU LRU（跨层保留）、相同在途重复页合并和确定性批次 drain/eviction。
这个 LRU 模型不是 BaM 原缓存策略的逐指令复现；两路径的请求列表可公平比较，不能声称复现 BaM 全部调度。

`--qd trace` 取每层采集的软件 outstanding 峰值，裁剪到 [1, max-qd]；这是一种**并发上限 replay**，
不是完整时序 replay。另跑固定 --qd 1、4、16、64，区分页集合变化和受控并发的影响。
`replay_pending_peak` 是 scheduler 持有的 futures 数；可能含已完成未回收的 future，**不是实测硬件 outstanding**。
受 CPU/Python调度、GIL、批次屏障影响，replay吞吐不能直接等同GPU发起I/O的吞吐。

为了评估真实 Host reuse，至少分别测：

- **cold-start**：在专用实验节点控制 page cache 初态，两路径配对前采用同一预定协议；
  如实验允许全局清缓存，由操作者执行 `sync` 与 `echo 3 > /proc/sys/vm/drop_caches`。
  工具不会自动执行；只能在整个 run 前清理，不能每层清理，否则破坏跨层 reuse。
- **warm-start**：按明确定义的预热流程读取该工作集，再从 level 0 replay，标明是否能放进 Host DRAM。
- **capacity pressure**：固定GPU cache容量，改变Host可用内存的受控配置，避免把可缓存全图的情况泛化到更大图。

上述性能命令若紧接 correctness 执行且未控制初态，属于 warm/未知状态，不能标为 cold。
不同顺序交替（Host/GDS 与 GDS/Host），每种条件至少 5 个独立进程，run_01…run_05，
记录page cache协议、显存/Host容量、SSD和拓扑。compare 检查trace校验和、QD、缓存容量和逐层miss数一致。

## 8. 判断 BFS 是否值得继续

检查 levels.csv 的 frontier、unique pages、cross-level reuse 与真实 page_submissions，
不要把 frontier大误认为实际 I/O并发必然高：GPU resident warps、degree、页重叠和cache hits都会限制它。

只有当多次实验中同页大小的层出现稳定的 Host/GDS 性能顺序反转，且优势层占用足够总时间，
才有理由继续做应用集成。比较固定 Host、固定 GDS总耗时与逐层min之和可以描述潜在空间；
逐层min使用不同执行中的cache状态，**不是有状态动态路由的真实 oracle**，也不含路由开销。
少数廉价层的反转未必产生可测的应用收益。

建议上传：每个source的完整trace三件套、BaM日志、每种QD/缓存条件至少5次Host/GDS CSV+JSON、
比较CSV、gdscheck及cuFile direct证据、page cache控制协议。不需要先上传完整graph payload。

## 9. 已验证与待验证

本次开发环境只完成 Python语法检查、CPU调度/读取布局/trace完整性测试，以及针对固定BaM源码的补丁应用检查。
当前环境没有 nvcc、GPU、CuPy或KvikIO：**BaM CUDA编译、GPU读取正确性、原生GDS和性能均未运行**。
不要把提交成功误认为已经验证了Host/GDS crossover。
