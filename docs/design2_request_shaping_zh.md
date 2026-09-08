# Design 2：放大感知的请求整形

Design 2 只对满足以下条件的设备读取启用：`IOContext` 已完成 64 个逻辑请求的画像、
稳定策略为 `GPU_DIRECT + SHAPED`、文件没有进入 compatibility mode，并且请求通过
线程池 `pread()` 提交。同步 `read()` 和原生 `cuFileReadAsync()` 保持原语义。

## 机制

每个 `FileHandle` 拥有一个短窗口请求队列。收集器等待队列达到 32 个请求，或者首个请求
等待 200 微秒后形成批次。队列按照文件偏移排序，将重叠或间隔不超过 4 KiB 的请求组成
候选组。只有候选组包含至少两个请求、对齐后的物理范围不超过 256 KiB、不会越过文件
末尾，并且 `physical_bytes / logical_bytes <= 1.5` 时才整形。

整形器维护 4 个延迟分配、长期复用的 GPU staging buffer slot。每个物理 plan 作为独立任务
提交到 per-FileHandle 有界 executor，因此不同 plan 可以并行执行。executor 线程数为
`min(KVIKIO_NTHREADS, staging_buffer_pool_size + 1)`：一个线程可运行 collector，其余线程执行
物理 plan；当前默认上限为 5。每个 slot 拥有独立 CUDA Stream 和 Event；D2D 分发后记录
Event，并在 Event 完成后兑现逻辑请求的 `std::future`。不能获益的请求仍作为独立物理任务
提交。

staging slot 不显式调用 `cuFileBufRegister/cuFileBufDeregister`。这保留 Runtime 自己的 GPU
staging pool 和请求合并，但把底层所需的临时注册交给 cuFile 管理，隔离 Pattern 4/5
非对齐路径上并发 shaped worker 的显式注册生命周期。代价是 cuFile 可能增加一次内部
staging copy，因此必须同时通过稳定性矩阵和 IOPS 对照验证。

关闭文件时，整形器先等待收集器退出，再等待所有已提交 physical task 的
`std::future` 完成，然后销毁 per-handle executor 的 worker，最后才释放 staging slots。
这样 CUDA/cuFile 线程局部状态会在 CUDA context 和 cuFile handle 仍有效时回收，而不会延迟到
进程级 `kvikio::defaults` 静态析构阶段。仅等待活动任务计数归零是不够的：
worker 可能已经递减计数，但其 lambda 尚未完全退出，此时提前析构整形器会形成生命周期
竞态。

设置 `KVIKIO_REQUEST_SHAPING=1` 或：

```python
kvikio.defaults.set("request_shaping_enabled", True)
```

## Benchmark

下面的测试使用完全相同的 4 KiB、非对齐连续文件偏移请求，只通过预热阶段固定三种
`IOContext` 策略：GDS Direct、Host-mediated 和 Shaped GDS。计时不包含 64 个画像请求。
实际计时区域从 8 MiB+3 开始，与画像区域分离；每种模式计时前还会调用
`POSIX_FADV_DONTNEED`，避免文件创建和画像阶段残留的 Page Cache 直接影响 Host 基线。

```bash
python -m kvikio.benchmarks.design2_request_shaping \
  --file /mnt/gds2/cwd_test/design2-64m.bin \
  --prepare \
  --requests 8192 \
  --io-size 4096 \
  --batch-size 32 \
  --verify \
  --output /tmp/design2-results.json
```

上面每批形成一个合并 plan，用于比较端到端收益。下面每批构造 4 个相距 64 KiB 的连续
请求簇，预期形成 4 个可并发的物理 plan，用于验证 staging pool：

```bash
python -m kvikio.benchmarks.design2_request_shaping \
  --file /mnt/gds2/cwd_test/design2-96m.bin \
  --prepare \
  --requests 8192 \
  --io-size 4096 \
  --batch-size 32 \
  --clusters-per-batch 4 \
  --verify \
  --output /tmp/design2-pool-results.json
```

在设备线程池至少有 4 个线程时，pool 场景应观察到 `max_inflight_physical > 1`；否则说明
物理 plan 虽已独立提交，但实际执行仍被设备或线程配置串行化。

运行前应确认：

```bash
sudo sh -c 'echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled'
grep -E 'IO stats|Ops' /proc/driver/nvidia-fs/stats
```

Benchmark 输出以下关键指标：

- `iops` 和 `logical_mib_per_second`：端到端性能；
- `context.shaping.logical_requests`：进入整形器的逻辑请求数；
- `context.shaping.physical_requests`：实际调用 cuFile 的请求数；
- `context.shaping.shaped_groups`：成功合并的请求组数；
- `context.shaping.collection_batches`：计时阶段形成的收集批次数；
- `context.shaping.max_collected_requests`：单批实际收集到的最大请求数；
- `context.shaping.max_inflight_physical`：同时执行的 cuFile 物理调用峰值；
- `context.shaping.submitted_bytes / logical_bytes`：Runtime 可见的提交放大率。

机制生效至少需要同时满足：`submit == SHAPED`、`shaped_groups > 0` 且
`physical_requests < logical_requests`。性能有效还要求 Shaped 的 IOPS 或逻辑带宽高于 Direct；
若只有请求数下降而性能没有提高，说明物理 I/O、Event 等待或 D2D 分发成本仍抵消了收益。
Benchmark 的 `context.shaping` 是计时前后差值，`context.shaping_total` 保留包含画像请求的累计值。

测试完成后关闭统计：

```bash
sudo sh -c 'echo 0 > /sys/module/nvidia_fs/parameters/rw_stats_enabled'
```
