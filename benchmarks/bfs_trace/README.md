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
export BAM_BFS_ROOT="$PWD"
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

## 3.1 集中配置项目、数据集和结果路径

**数据集路径由命令行指定，不需要修改 KvikIO 的源码或 settings.h。**
建议在执行采集、数据准备和 replay 的同一个 shell 中设置以下变量；切换终端后重新设置。
这些 export 只是教程的命令组织方式，程序不会自动读取 GRAPH_PREFIX、BFS_EDGE_FILE 等变量，
必须像后面的示例一样把它们传入 `-f`、`--edge-file` 等参数。

```bash
# 修改这三项为你的实际目录；项目路径使用 clone 后的绝对路径。
export KVIKIO_BFS_ROOT=/path/to/kvikio-bfs
export BAM_BFS_ROOT=/path/to/bam-bfs-trace
export GRAPH_PREFIX=/mnt/dataset/graph/uk-2007-05.bel

# BaM 使用的两个现有数据文件，不是目录。
export GRAPH_COL="${GRAPH_PREFIX}.col"
export GRAPH_DST="${GRAPH_PREFIX}.dst"

# replay 文件放在支持原生 GDS 的已挂载文件系统中。
export BFS_EDGE_FILE=/mnt/gds/uk-2007-05_edges_4k.bin

# 采集参数：根节点必须属于该图，且建议为有出边的非零顶点。
export BFS_SOURCE=12345
export BFS_PAGE_BYTES=4096
export BFS_GPU=0
export BFS_GPU_CACHE_PAGES=1024
export BFS_RAW_OFFSET=0

# 按实验目的和轮次组织结果；不要重复使用已有输出目录。
export BFS_RESULT_ROOT="$KVIKIO_BFS_ROOT/results/bfs_path_opportunity"
export BFS_CAPTURE_DIR="$BFS_RESULT_ROOT/capture/run_01"
export BFS_REPLAY_DIR="$BFS_RESULT_ROOT/replay/run_01"
export BFS_TRACE_PREFIX="$BFS_CAPTURE_DIR/bfs"
export BFS_PAGES="${BFS_TRACE_PREFIX}_run_0_pages.csv"
export BFS_LEVELS="${BFS_TRACE_PREFIX}_run_0_levels.csv"
export BFS_COMPLETE="${BFS_TRACE_PREFIX}_run_0_complete"
export BFS_DIRECT_CONFIG="$BFS_RESULT_ROOT/config/cufile_direct.json"

mkdir -p "$BFS_CAPTURE_DIR" "$BFS_REPLAY_DIR" "$BFS_RESULT_ROOT/config"
```

| 变量 / 参数 | 含义 | 示例 |
|---|---|---|
| GRAPH_PREFIX → `-f` | 图路径前缀，程序自动追加 `.col` / `.dst` | `/mnt/dataset/graph/uk-2007-05.bel` |
| GRAPH_DST | 原始带 header 的 edge 文件 | `uk-2007-05.bel.dst` |
| BFS_EDGE_FILE → `--edge-file` | 去 header、补齐后的文件系统 edge payload | `/mnt/gds/uk-2007-05_edges_4k.bin` |
| BFS_RAW_OFFSET → `--loffset` | edge payload 在 BaM 裸设备中的字节偏移 | `0` 或部署时指定的对齐偏移 |
| BFS_TRACE_PREFIX | 采集输出文件名前缀 | `capture/run_01/bfs` |
| BFS_PAGES / BFS_LEVELS | 同一次 BFS run 的页事件和层指标 | `bfs_run_0_pages.csv` / `bfs_run_0_levels.csv` |

`-f` 后面不能填目录，也不能填 `.col` 或 `.dst` 完整文件名。
例如 `-f /mnt/dataset/graph/uk-2007-05.bel.dst` 会导致程序寻找
`uk-2007-05.bel.dst.col` 和 `uk-2007-05.bel.dst.dst`，从而打开失败。
`uk-2007-05.bel` 本身不一定要存在；必须存在的是加后缀后的两个文件。

## 3.2 先检查数据格式、根节点与容量

下面只检查现有文件，不构建图、不访问裸设备、不扫描完整 edge 数组。
`.col` 的首个 uint64 是 offsets 数量，BaM 用它减 1 得到顶点数；
`.dst` 的首个 uint64 是 edge 数量，随后均有一个 typeT字段。
两文件的实际数组均从第 16 字节开始。该教程适用于 BaM 的 uint64 CSR 数据，
不能直接输入文本 edge list、SIFT 向量或其它 ANN 数据。

```bash
python3 - <<'PY'
import os
import struct
from pathlib import Path
import numpy as np

col, dst = Path(os.environ['GRAPH_COL']), Path(os.environ['GRAPH_DST'])
for path in (col, dst):
    if not path.is_file():
        raise SystemExit(f'找不到数据文件: {path}')

def header(path):
    with path.open('rb') as f:
        raw = f.read(16)
    if len(raw) != 16:
        raise SystemExit(f'文件头不足16字节: {path}')
    return struct.unpack('<QQ', raw)

n_offsets, col_type = header(col)
n_edges, dst_type = header(dst)
if n_offsets < 2 or col.stat().st_size < 16 + n_offsets*8:
    raise SystemExit('col offsets 数量/文件长度不匹配')
if dst.stat().st_size < 16 + n_edges*8:
    raise SystemExit('dst edge 数量/文件长度不匹配')
vertices = n_offsets-1
root = int(os.environ['BFS_SOURCE'])
page = int(os.environ['BFS_PAGE_BYTES'])
offset = int(os.environ['BFS_RAW_OFFSET'])
if page < 4096 or page & (page-1):
    raise SystemExit('page_size 必须为 >=4096 的2的幂')
if offset < 0 or offset % page:
    raise SystemExit('裸设备 offset 必须非负且按 page_size 对齐')
if not 0 < root < vertices:
    raise SystemExit(f'请选非零合法根节点: 0 < src < {vertices}')
offsets = np.memmap(col, dtype='<u8', mode='r', offset=16, shape=(n_offsets,))
if int(offsets[0]) != 0 or int(offsets[-1]) != n_edges:
    raise SystemExit('CSR 首末 offset 与 edge 数量不一致')
# 分块检查，避免一次性创建全图大小的差分数组。
for begin in range(0, n_offsets-1, 1_000_000):
    part = offsets[begin:min(n_offsets, begin+1_000_001)]
    if np.any(part[1:] < part[:-1]):
        raise SystemExit(f'CSR offsets 非单调，检查位置附近: {begin}')
degree = int(offsets[root+1])-int(offsets[root])
if degree == 0:
    raise SystemExit('源节点没有出边；换一个源节点进行路径机会测试')
payload = n_edges*8
padded = ((payload+page-1)//page)*page
print(f'vertices={vertices}, edges={n_edges}, src={root}, degree={degree}')
print(f'header typeT: col={col_type}, dst={dst_type}（不推断其枚举含义）')
print(f'edge payload={payload} B; replay padded={padded} B')
print(f'裸设备至少需覆盖字节区间 [{offset}, {offset+padded})')
print(f'GPU edge cache={int(os.environ["BFS_GPU_CACHE_PAGES"])*page} B')
print('以上为格式检查；未验证所有 edge IDs 和裸设备中的实际内容。')
PY

df -h "$(dirname "$BFS_EDGE_FILE")"
findmnt -T "$(dirname "$BFS_EDGE_FILE")"
```

确认 replay 输出所在目录已经存在、空间足够，且确实位于目标 SSD 的文件系统上。
不要把原始 `.dst` 删除：BaM 在初始化时仍需读取其 header，prepare_edges.py 也用它生成相同 payload。
真正的大图中 vertex offsets、labels、frontier 和 trace buffer也需要 GPU 内存；
4 MiB edge cache 不代表整个 BFS 仅需4 MiB显存。

## 3.3 区分 BaM 裸设备布局与文件系统布局

**仅设置 `-f` 或生成 BFS_EDGE_FILE，都不会部署 BaM 裸设备数据。**
BaM 初始化从文件读取 CSR offsets及edge计数，运行中的 `seq_read()` 从 libnvm控制器读取 edges。
Host/GDS replay 则对 Linux 文件系统中的 BFS_EDGE_FILE 做读取。

同一逻辑页 p 应满足：

| 数据副本 | 页 p 对应的字节位置 |
|---|---|
| 原始 GRAPH_DST | `16 + p * BFS_PAGE_BYTES`，末页只含部分有效edges时不足一页 |
| BaM 裸设备上的 edge payload | `BFS_RAW_OFFSET + p * BFS_PAGE_BYTES` |
| 文件系统 BFS_EDGE_FILE | `p * BFS_PAGE_BYTES`，末页由 prepare_edges.py 补零 |

BaM readwrite程序涉及的参数如下；先用其 `--help` 核对本地构建。

| readwrite 参数 | 数据部署时的含义 |
|---|---|
| `--input` / `-f` | 实际要写入裸设备的源文件 |
| `--ioffset` / `-i` | 源文件起始字节偏移；用原始 `.dst` 时为16，用去header后的payload时为0 |
| `--loffset` / `-l` | 目标裸设备字节偏移，必须与 BFS 的 BFS_RAW_OFFSET 一致 |
| `--access_type` | 上游定义0为读、1为写；部署会写入裸设备 |
| `--n_ctrls` | 本测试只使用一个控制器 |

不要把已有数据或已挂载文件系统所在 SSD 当作裸设备部署目标；裸设备写入会覆盖指定区域。
本教程不执行数据部署或设备解绑；沿用你已验证的 BaM部署流程。
尤其要检查最后一批的 padding 和实际写入范围，不能仅依据 payload大小判断上游writer不会越过它。
如果还没有可正确运行的 BaM数据布局，先完成 BaM自己的部署与正确性验证，再加trace。

控制器路径也不是通过 `-f` 指定的：固定上游 BFS 的 `main.cu` 中有
`sam_ctrls_paths` / `intel_ctrls_paths`，单控制器时两数组第0项均为 `/dev/libnvm0`；
`--ssd 0/1` 选择对应路径数组，不会按 SSD型号自动发现设备。
部署程序与BFS必须指向同一控制器/namespace，并确认已有控制器映射。
如需修改路径数组，应在 BaM checkout 中完成并重新编译；不需要改 KvikIO数据路径。

## 4. 采集真实 BFS

BaM 输入使用 `<graph>.bel.col` 和 `<graph>.bel.dst`。
`.col` 为 vertex offsets，`.dst` 包含 16 B header + uint64 edges。
采集前确保 NVMe 上 `--loffset` 指向同一 `.dst` 的**去掉 header 的 edge payload**；
用原始 .dst 作为 BaM readwrite输入时，input offset应为16 B；用去header后的payload时为0。
禁止使用包含 header 的 raw layout。
该布局必须与原来可正确运行的 BaM BFS一致。

```bash
cd "$BAM_BFS_ROOT"
mkdir -p "$BFS_CAPTURE_DIR"
sudo env \
  BFS_TRACE_PREFIX="$BFS_TRACE_PREFIX" \
  BFS_TRACE_EVENTS=4194304 \
  ./build/bin/nvm-bfs-bench \
  -f "$GRAPH_PREFIX" --loffset "$BFS_RAW_OFFSET" \
  --impl_type 9 --memalloc 6 --src "$BFS_SOURCE" \
  --n_ctrls 1 --page_size "$BFS_PAGE_BYTES" --gpu "$BFS_GPU" --threads 128 \
  --maxPCSize "$((BFS_GPU_CACHE_PAGES * BFS_PAGE_BYTES))" \
  > "$BFS_CAPTURE_DIR/bfs.log" 2>&1
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
  "$GRAPH_DST" "$BFS_EDGE_FILE" --page-bytes "$BFS_PAGE_BYTES"
```

该程序去掉两个 uint64 header，只复制 `edge_count*8` B payload，并在末尾补零到 page 边界；
输出存在时拒绝覆盖。BaM不解释 header 的 typeT字段，本工具也不猜测其枚举含义。
源数据必须确实为 uint64 edge IDs。`page_id=p` 对应新文件偏移 `p*BFS_PAGE_BYTES`（默认 p*4096），不使用裸设备 loffset。
只做读；没有 trace 的页不影响 replay。

## 6. 禁止 GDS 静默退回 Host

首先运行节点上安装的 `gdscheck -p`（常见路径 `/usr/local/cuda/gds/tools/gdscheck`）。
基于当前节点的 cufile.json 复制一份实验配置，保留其文件系统、设备及其它设置，仅关闭兼容回退：

```bash
mkdir -p "$BFS_RESULT_ROOT/config"
python3 - <<'PY'
import json
import os
from pathlib import Path
cfg = json.loads(Path('/etc/cufile.json').read_text())
cfg.setdefault('properties', {})['allow_compat_mode'] = False
Path(os.environ['BFS_DIRECT_CONFIG']).write_text(json.dumps(cfg, indent=2))
PY
```

若节点配置包含注释，先转换为严格 JSON再执行上述脚本。
Replay 的 GDS 模式要求提供该配置，强制 KvikIO compat_mode=False，并调用 raw_read 跳过 size threshold。
文件需要 O_DIRECT 支持，GPU buffers / file offsets / request sizes 均按 4 KiB 对齐，并注册 GPU buffers。
仍需用节点的 cuFile 统计/日志确认数据走 direct path；`is_direct_io_supported` 本身只证明 O_DIRECT 可用。
不能将 compatibility read 标为 GDS。现有节点若只有 compat mode 能运行，此步骤失败就是环境限制，
应先解决 GDS支持问题，不能改回 AUTO 后继续声称在比较原生 GDS。

## 7. 校验读取内容，再做性能 replay

如果采集和 replay 在同一节点，上面 BFS_PAGES / BFS_LEVELS可直接指向采集输出，无需复制。
如果换节点，完整复制pages.csv、levels.csv和同名前缀的complete标记，然后重新设置这三个路径；
两种replay使用同一次run，不能分别重新采集。目录命名按实验目的，不用时间戳。

```bash
cd "$KVIKIO_BFS_ROOT"
test -s "$BFS_PAGES"
test -s "$BFS_LEVELS"
test -f "$BFS_COMPLETE"
test -s "$BFS_EDGE_FILE"
test -s "$BFS_DIRECT_CONFIG"
mkdir -p "$BFS_RESULT_ROOT/correctness/run_01"

python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file "$BFS_EDGE_FILE" \
  --mode host-cached --gpu "$BFS_GPU" --gpu-cache-pages "$BFS_GPU_CACHE_PAGES" --qd trace --max-qd 64 --verify \
  --output "$BFS_RESULT_ROOT/correctness/run_01/host.csv"
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file "$BFS_EDGE_FILE" \
  --mode gds --cufile-config "$BFS_DIRECT_CONFIG" \
  --gpu "$BFS_GPU" --gpu-cache-pages "$BFS_GPU_CACHE_PAGES" --qd trace --max-qd 64 --verify \
  --output "$BFS_RESULT_ROOT/correctness/run_01/gds.csv"
```

`--verify` 对每个 cache miss 逐字节比较 GPU 内容与文件。这会预热 Linux page cache、增加同步，
所以 correctness 输出不可做性能比较，compare.py 会拒绝它。

性能测试使用相同命令但去掉 --verify，将输出分别写为：

```bash
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file "$BFS_EDGE_FILE" \
  --mode host-cached --gpu "$BFS_GPU" --gpu-cache-pages "$BFS_GPU_CACHE_PAGES" --qd trace --max-qd 64 \
  --output "$BFS_REPLAY_DIR/host.csv"
python3 benchmarks/bfs_trace/replay.py \
  --pages "$BFS_PAGES" --levels "$BFS_LEVELS" --edge-file "$BFS_EDGE_FILE" \
  --mode gds --cufile-config "$BFS_DIRECT_CONFIG" \
  --gpu "$BFS_GPU" --gpu-cache-pages "$BFS_GPU_CACHE_PAGES" --qd trace --max-qd 64 \
  --output "$BFS_REPLAY_DIR/gds.csv"
python3 benchmarks/bfs_trace/compare.py \
  "$BFS_REPLAY_DIR/host.csv" \
  "$BFS_REPLAY_DIR/gds.csv" \
  --output "$BFS_REPLAY_DIR/comparison.csv"
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

## 7.1 下一轮实验和常见路径错误

更换run编号时，明确区分重采trace和重复replay：

```bash
# 仅重复 replay：仍使用原 BFS_PAGES / BFS_LEVELS / BFS_COMPLETE。
export BFS_REPLAY_DIR="$BFS_RESULT_ROOT/replay/run_02"
mkdir -p "$BFS_REPLAY_DIR"
# 重跑第7节两条性能命令与compare命令。
```

若要更换图或源节点重新采集，先修改GRAPH_PREFIX/GRAPH_COL/GRAPH_DST或BFS_SOURCE，
再改BFS_CAPTURE_DIR/BFS_TRACE_PREFIX及三项trace输入变量，重复数据检查与采集。
更换图还需要准备对应BFS_EDGE_FILE并重新验证BaM裸设备布局；不能用新trace读取旧图payload。
同一图仅换源节点时不必重新准备edge文件。

| 现象 | 优先检查 |
|---|---|
| Vertex/Edge file open failed | `-f` 是否为前缀；`${GRAPH_PREFIX}.col` / `.dst` 是否存在 |
| `/dev/libnvm0` 打不开 | BaM控制器/驱动映射，非GRAPH_PREFIX路径问题 |
| BFS输出异常、访问非法顶点 | 裸设备是否写了同图payload、是否跳过16 B header、loffset是否正确；也需检查原始图 |
| 找不到complete标记 | BFS是否正常完成；发生overflow时整次重采，不要手动创建标记 |
| replay提示edge file too short | `--edge-file` 是否为正确图的去header且按本次page size补齐的文件 |
| replay存在output already exists | 换下一轮结果目录；工具不会覆盖已有replay结果 |
| GDS报错但Host可运行 | direct配置、文件系统、GPU/SSD拓扑与GDS支持；不要改为AUTO后标为原生GDS |
| compare提示unpaired replay | 两条路径是否共用相同trace、edge文件、QD和GPU cache容量 |

`BFS_COMPLETE` 是核对用变量；replay程序根据 `*_pages.csv` 的文件名前缀自行定位 `*_complete`，
没有单独的 `--complete` 参数。复制或重命名时必须保持三件套的统一前缀。

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
