# G-Route DiskANN/GustANN端到端评估

`run_diskann_groute_e2e.sh`使用DiskANN生成的标准SSD index，在GustANN Hybrid GPU
图搜索中比较原生AIO loader与KvikIO G-Route loader。两组实验保持ANN算法、index、
查询和搜索参数不变，只替换页面加载路径。

## 集成边界

DiskANN的`build_disk_index`负责生成`*_disk.index`和PQ数据。GustANN解析该格式，
并将搜索产生的4 KiB page请求交给`IndexLoader`。G-Route补丁增加`groute` backend：

```text
GustANN GPU graph search
        └── IndexLoader
              ├── aio: SSD → pinned Host buffer → GPU kernel
              └── groute: SSD/Host Cache → KvikIO policy → GPU buffer
```

这不是BaM backend的封装。BaM由GPU kernel访问裸NVMe设备并依赖定制内核模块，
host-side KvikIO无法透明拦截。当前方案复用GustANN Hybrid计算路径，使两个backend
能在普通GDS文件系统上的同一DiskANN index上进行等价对比。

## 构建

补丁已针对GustANN提交`dd8e70b2b1b5d22acb2373685fdc416511e73b6d`验证：

```bash
git clone https://github.com/thustorage/GustANN.git /opt/GustANN
git -C /opt/GustANN checkout dd8e70b2b1b5d22acb2373685fdc416511e73b6d

GUSTANN_DIR=/opt/GustANN \
CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
bash scripts/setup_gustann_groute.sh
```

脚本向外部GustANN checkout应用`scripts/diskann/gustann-groute.patch`并安装
`groute_loader.cpp`，不在KvikIO仓库中vendor DiskANN或GustANN。输出程序为：

```text
/opt/GustANN/build-groute/bin/search_disk_hybrid
```

## 准备DiskANN索引

可以使用WSCWDA/DiskANN的`cpp_main`分支或GustANN的DiskANN submodule：

```bash
build/apps/build_disk_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix /mnt/gds/diskann/index_R128_L200 \
  --data_path /data/base.fbin \
  -B 32 -M 64 -R 128 -L 200
```

运行需要匹配的以下输入：

- `index_R128_L200_disk.index`；
- PQ prefix `index_R128_L200_pq`；
- `.fvecs`或`.bvecs`查询；
- `.ivecs` ground truth；
- GustANN navigation graph目录。

navigation graph按照GustANN的`scripts/gen_pivot_graph.sh`生成：从原始`.fbin`或
`.bbin`采样，调用DiskANN `build_memory_index`，并生成`nav_index`及tags文件。

## 运行

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

脚本在奇偶重复中交替`aio→groute`和`groute→aio`顺序。冷缓存实验可由root设置
`DROP_CACHES=1`。结果包括中位QPS、线程平均查询延迟、Recall、page reads，以及
G-Route最终选择的policy、cache和shaping统计。

## 有效性要求

1. AIO与G-Route Recall差异不超过配置容差；
2. G-Route日志包含`[GROUTE_STATS]`，排除静默回退；
3. 每个配置至少重复5次并报告中位数和标准差；
4. 固定数据集、index、`TOPK`、`EF_SEARCH`、minibatch、线程和context数量；
5. 扫描多个`EF_SEARCH`绘制Recall--QPS曲线；
6. 同时报告所选policy、cache hit、physical requests与p99 I/O batch latency，解释
   性能来自路径选择、region cache还是request shaping。
