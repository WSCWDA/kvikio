# KvikIO GDS-HCache MVP 使用说明

这版改动给 `CuFile` 增加了一个**每文件句柄独立**的小读缓存。第一次读取某个 64 KiB
缓存行时，KvikIO 通过现有 POSIX/O_DIRECT 路径把整行读入页对齐的 CUDA 页锁定内存；后续命中
同一行时只做 Host-to-Device 拷贝，不再访问 SSD。

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
```

也可以在 Python 中配置，然后再打开文件：

```python
import kvikio

kvikio.defaults.set({
    "host_cache_enabled": True,
    "host_cache_capacity": 1024 * 1024 * 1024,
    "host_cache_line_size": 64 * 1024,
    "host_cache_max_io_size": 64 * 1024,
})

f = kvikio.CuFile("/mnt/gds/test.bin", "r")
# 重复调用 f.pread(...) 或 f.raw_read(...) 后查看统计
print(f.host_cache_stats())
```

统计字段含义：`hits` 为缓存命中次数，`misses` 为需要读存储的次数，`evictions` 为 LRU
淘汰次数，`storage_bytes` 为实际从文件读入缓存的字节数，`h2d_bytes` 为缓存拷贝到 GPU 的
字节数。

## MVP 边界

- 仅缓存目标为 GPU 内存、大小不超过上限且不跨缓存行的读取。
- Host 内存读取、cuFile 原生异步流接口保持原路径。
- KvikIO 通过同一 `CuFile` 写入前会清空缓存。
- 其他进程或其他文件句柄写同一个文件时不会自动通知本缓存；这种场景应调用
  `clear_host_cache()`，或关闭缓存。
- 当前每个缓存句柄独占容量，首次合格读取时才分配页锁定内存。大量同时打开的文件应减小容量。

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
