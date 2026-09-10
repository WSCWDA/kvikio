# KvikIO GDS-HCache MVP 使用说明

这版改动给 `CuFile` 增加了一个**每文件句柄独立**的小读缓存和有界区域准入器。准入器以
1 MiB region 聚合多个 cache line，但只把同一 cache line 的重复访问视为复用证据，避免把顺序
扫描同一区域内的不同 cache line 误判为热点。默认第二次访问某个 cache line 时提升其所在
region；此后该 region 中符合条件的 cache-line miss 才能插入 Host Cache。

## 构建安装

按 KvikIO 原有方式构建即可，没有新增 `liburing` 依赖：

```bash
./build.sh libkvikio kvikio
```

如果只构建 C++ 库：

```bash
cmake -S cpp -B build -DKvikIO_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 启用缓存

缓存默认关闭，并且配置在创建 `CuFile` 时生效：

```bash
export KVIKIO_HOST_CACHE=ON
export KVIKIO_HOST_CACHE_CAPACITY=$((1024 * 1024 * 1024))
export KVIKIO_HOST_CACHE_LINE_SIZE=$((64 * 1024))
export KVIKIO_HOST_CACHE_MAX_IO_SIZE=$((64 * 1024))
export KVIKIO_HOST_CACHE_REGION_SIZE=$((1024 * 1024))
export KVIKIO_HOST_CACHE_ADMISSION_THRESHOLD=2
export KVIKIO_HOST_CACHE_MAX_REGIONS=4096
```

也可以在 Python 中配置，然后再打开文件：

```python
import kvikio

kvikio.defaults.set({
    "host_cache_enabled": True,
    "host_cache_capacity": 1024 * 1024 * 1024,
    "host_cache_line_size": 64 * 1024,
    "host_cache_max_io_size": 64 * 1024,
    "host_cache_region_size": 1024 * 1024,
    "host_cache_admission_threshold": 2,
    "host_cache_max_regions": 4096,
})

f = kvikio.CuFile("/mnt/gds/test.bin", "r")
# 重复调用 f.pread(...) 或 f.raw_read(...) 后查看统计
print(f.host_cache_stats())
```

统计字段含义：`hits` 和 `misses` 为 cache-line 查找结果；`admitted_regions` 为得到重复
证据并被提升的 region 数；`admission_bypasses` 和 `admission_bypass_bytes` 为准入前绕过缓存
的请求数和字节数；`metadata_evictions` 为有界 region metadata 的 LRU 淘汰次数；
`storage_bytes` 为实际从文件读入缓存的字节数，`h2d_bytes` 为缓存拷贝到 GPU 的字节数。

## Region-level Admission

一次请求首先查找目标 cache line。命中时直接 H2D；未命中时，准入器按文件偏移计算 region
和 region 内的 cache-line 编号，并递增该 line 的饱和访问计数。计数未达到阈值时返回
`BYPASS`，由 `FileHandle` 当前选定的主路径完成本次读取，不分配缓存存储或 cache slot。计数达到
阈值后提升整个 region，后续该 region 的 cache-line miss 可以插入缓存。

Region metadata 使用独立 LRU，默认最多跟踪 4096 个 region。它与数据缓存 LRU 分离：前者
回答“这个 region 是否值得缓存”，后者回答“缓存满时淘汰哪一个 cache line”。写入或显式调用
`clear_host_cache()` 时，两类当前状态同时失效，但累计统计计数保留。

## MVP 边界

- 仅缓存目标为 GPU 内存、大小不超过上限且不跨缓存行的读取。
- Host 内存读取、cuFile 原生异步流接口保持原路径。
- KvikIO 通过同一 `CuFile` 写入前会清空缓存。
- 其他进程或其他文件句柄写同一个文件时不会自动通知本缓存；这种场景应调用
  `clear_host_cache()`，或关闭缓存。
- 当前每个缓存句柄独占容量，首次合格读取时才分配页锁定内存。大量同时打开的文件应减小容量。

## IOContext workload 分类

每个 `FileHandle` 的 `IOContext` 使用前 64 个逻辑 GPU 读取请求统计平均 I/O 大小、顺序访问
比例和重复区域比例，并生成一次长期复用的策略。分类标签描述主导性能特征，而不是把文件或
应用强制归入互斥的静态类型：

- `SEQUENTIAL_SCAN`：顺序比例不低于 75%，且平均 I/O 不小于 64 KiB；使用 GPU-direct，
  绕过 Host Cache。
- `REUSE_DOMINATED`：重复区域比例不低于 25%，且平均 I/O 不大于 64 KiB；使用
  Host-mediated 路径，并在 HCache 可用时启用 region admission。
- `FINE_GRAINED`：不满足复用条件的平均小于 64 KiB 的请求；使用 Host-mediated 路径，
  在 HCache 可用时仍交给 region admission 做最终判断。这里 `ADMIT` 表示允许检查 region，
  并不表示无条件缓存；一次性扫描仍会被拒绝。
- `GENERAL`：其余大粒度或混合请求；默认使用 GPU-direct，并绕过 HCache。

文件大小没有作为 `SMALL_FILE` 或 `LARGE_FILE` 枚举值，因为它与访问顺序、请求粒度和复用
特征相互独立。例如，大文件也可能只访问一个很小的热点区域。若后续需要利用文件大小，应将
其作为独立的 `FileTraits` 输入，而不是加入 `WorkloadClass`。

## 如何确认 SSD 是否真的被缓存挡住

先清理文件页缓存并观察块设备，而不是只看程序的 `storage_bytes`：

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
iostat -x 1
```

预期第一次遍历出现 SSD 读取，第二次读取相同热点时 `hits` 和 `h2d_bytes` 增长，而
`storage_bytes` 基本不再增长。若 `iostat` 始终为零，先用 `findmnt -T 文件路径` 确认监控的
设备，并检查文件是否位于 RAID、LVM、容器 overlay 或网络文件系统上。

## 正式 Benchmark 矩阵

仓库内提供了一个面向 MVP 的固定矩阵脚本：

```bash
HCACHE_BENCH_FILE=/mnt/gds2/cwd_test/kvikio_hcache/bench-1g.bin \
REQUESTS=100000 \
REPEATS=5 \
RESULT_ROOT=/mnt/gds2/cwd_test/kvikio_hcache/results-formal \
scripts/run_gds_hcache_matrix.sh \
  2>&1 | tee /tmp/hcache_matrix_formal.log
```

当前正式矩阵只包含 `64 KiB` 和 `256 KiB` 两种 cache line，故意去掉了冒烟测试中失败的
`16 KiB` cache line。矩阵如下：

- I/O size：`4 KiB`、`16 KiB`、`64 KiB`
- HCache line size：`64 KiB`、`256 KiB`
- HCache capacity：`16 MiB`、`32 MiB`、`64 MiB`、`128 MiB`
- Admission threshold：`1`（cache-all 基线）、`2`（region admission）
- Hot set：默认 `64 MiB`
- Baseline：GDS no-cache、POSIX no-cache

汇总结果：

```bash
python scripts/summarize_gds_hcache_matrix.py \
  /mnt/gds2/cwd_test/kvikio_hcache/results-formal
```

输出文件：

- `/mnt/gds2/cwd_test/kvikio_hcache/results-formal/raw_results.csv`
- `/mnt/gds2/cwd_test/kvikio_hcache/results-formal/summary.csv`
