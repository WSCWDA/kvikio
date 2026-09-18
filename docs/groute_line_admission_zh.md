# G-Route cache-line 复用与收益准入

这是现有每 `FileHandle` 独立的 Host Cache 的可选策略，不是 GPU 显存缓存，也不修改 Linux Page Cache。数据条目继续采用原有的 LRU 淘汰。默认沿用原来的 region 准入策略，以便直接进行同 trace 对照。

启用新策略（应在打开文件之前设置）：

```python
import kvikio
import kvikio.defaults

kvikio.defaults.set({
    "host_cache_enabled": True,
    "host_cache_line_admission": True,
    "host_cache_sketch_bytes": 64 * 1024,
    "host_cache_aging_interval": 256,
    "host_cache_admission_threshold": 2,
    "host_cache_hit_ns": 10000,
    "host_cache_fill_ns": 120000,
    "host_cache_host_bypass_ns": 80000,
    "host_cache_gds_bypass_ns": 80000,
})
```

同名环境变量均以 `KVIKIO_` 大写形式配置，例如 `KVIKIO_HOST_CACHE_LINE_ADMISSION=ON`。上述纳秒成本仅为初始模型值，**不是该设备实测值**；正式实验必须在目标机器上分别测量 Host Cache hit、Host Cache fill、POSIX fallback、GDS fallback 的同粒度请求延迟，然后配置相应值。兼容模式按 Host fallback 建模。`HOST_CACHE` 强制策略进入准入器；`AUTO` 模式即使文件级策略初期绕过 cache，也允许符合大小限制的请求查询新准入器；强制 `GDS_DIRECT` 保持绕过 cache，用于基线实验。

每个合格请求（包括缓存命中）按 `(file handle, offset / cache_line_size)` 更新固定容量的 blocked sketch；只有 miss 才进行准入决策。每个键的四个 4-bit counter 位于同一 64-B CPU cache line；查询取最小值，更新只递增最小值。每 `aging_interval` 个合格请求衰减一个 64-B block，避免一次扫描整个 sketch。每个 handle 的 sketch 空间为 `sketch_bytes`，与文件大小无关。碰撞会高估复用；衰减在各 block 间错开，可能在短时间内高估或低估热点，应在变化 trace 中验证。

令 `r` 为更新前的复用估计、`B` 为当前 fallback 成本、`H` 为命中成本、`F` 为一次 cache fill 加向 GPU 拷贝的成本，准入条件为：

```text
r + 1 >= admission_threshold
r > 0
B > H
r * (B - H) >= max(F - B, 0)
```

阈值为 1 时保留 cache-all 基线行为。这里 `r` 被当作未来复用次数的低开销代理，不是未来访问次数的无偏预测。实际填充仍可能因锁竞争、Linux Page Cache、GDS 兼容模式、读放大和并发发生偏差。无论成本模型如何，cache hit 总是直接返回；容量满且所有条目都在 GPU copy 中被 pin 时仍回退原路径。

`host_cache_stats()` 新增 `sketch_bytes`、`sketch_aging_steps`、`benefit_bypasses`、`admitted_lines`。后两项分别统计已达到访问门槛但收益不足的 miss，以及通过准入判断的 miss；不是不同 line 的精确去重计数。原有 region 元数据统计在新策略下为零。

推荐对比：原 region 策略、新 line 策略、阈值 1 的 cache-all、无缓存；使用相同 DiskANN/GustANN trace，报告端到端 IOPS/p99、实际块设备读取、cache 命中、缓存填充字节数、准入判断开销和 CPU 锁等待。其他进程或句柄写同一文件仍需显式清空缓存。
