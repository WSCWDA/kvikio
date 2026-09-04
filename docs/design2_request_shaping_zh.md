# Design 2：放大感知的请求整形

Design 2 只对满足以下条件的设备读取启用：`IOContext` 已完成 64 个逻辑请求的画像、
稳定策略为 `GPU_DIRECT + SHAPED`、文件没有进入 compatibility mode，并且请求通过
线程池 `pread()` 提交。同步 `read()` 和原生 `cuFileReadAsync()` 保持原语义。

## 机制

每个 `FileHandle` 拥有一个短窗口请求队列。队列按照文件偏移排序，将重叠或间隔不超过
4 KiB 的请求组成候选组。只有候选组包含至少两个请求、对齐后的物理范围不超过 256 KiB、
不会越过文件末尾，并且 `physical_bytes / logical_bytes <= 1.5` 时才整形。

整形请求读取到一个延迟分配、长期注册的 GPU staging buffer，再通过同一内部 CUDA Stream
执行 D2D 分发。`std::future` 只在对应 D2D 完成后完成。不能获益的请求继续逐请求调用
`cuFileRead`。

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
- `context.shaping.submitted_bytes / logical_bytes`：Runtime 可见的提交放大率。

机制生效至少需要同时满足：`submit == SHAPED`、`shaped_groups > 0` 且
`physical_requests < logical_requests`。性能有效还要求 Shaped 的 IOPS 或逻辑带宽高于 Direct；
若只有请求数下降而性能没有提高，说明 20 微秒收集窗口或 D2D 分发成本抵消了收益。

测试完成后关闭统计：

```bash
sudo sh -c 'echo 0 > /sys/module/nvidia_fs/parameters/rw_stats_enabled'
```
