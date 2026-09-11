# G-Route Design 1：IOContext策略选择评估

`run_design1_policy.sh`验证IOContext能否从有限profiling窗口识别访问特征，并将
同一FileHandle后续请求映射到预期的I/O path、cache和submit策略。该实验首先验证
策略正确性，不把不同workload之间的IOPS差异解释为性能提升。

## 实验阶段

每次运行严格分为以下阶段：

1. **Profiling**：前64个请求构造明确的顺序性、复用性、粒度、对齐和可合并特征；
2. **Policy selection**：读取IOContext并核对预期策略；
3. **Cache reset**：保留IOContext policy，清空profiling产生的cache line与region
   admission历史；
4. **Warm-up**：仅`random_hot_small`执行4次请求，使两个热点cache line达到准入阈值；
5. **Measurement**：统计正式请求的IOPS、带宽、batch p99、cache和shaping增量。

在warm-up与measurement边界额外记录三个状态：

- `warmup_admitted_regions`：仅由warm-up新准入的region数量；
- `warmup_storage_bytes`：仅由warm-up触发的storage读取字节数；
- `cache_entries_before_measurement`：计时开始前已经存在的cache line数量。

对当前两条热点line的trace，三者必须分别为`1`、`128 KiB`和`2`。正式测量阶段
应为1024次hit、0次miss和0 storage bytes，从而区分“预热填充成本”和“稳态命中成本”。

每个逻辑`pread()`只允许调用一次region admission。cache miss进入thread pool后不会
在`read_impl()`中再次增加访问计数。因此，准入阈值2表示两个逻辑请求，而不是一个
请求的两次内部检查。

## Workload定义

| Workload | 测量访问模式 | 预期策略 | 关键正确性条件 |
|---|---|---|---|
| `sequential_large` | 128 KiB顺序读取 | `GPU_DIRECT/BYPASS/DIRECT` | 不使用Host Cache和shaping |
| `random_cold_small` | 每次访问不同的64 KiB cache line | `HOST_MEDIATED/ADMIT/DIRECT` | 0 hit、0 admitted region |
| `random_hot_small` | 两条热点cache line交替访问 | `HOST_MEDIATED/ADMIT/DIRECT` | warm-up后测量请求全部hit |
| `adjacent_unaligned_small` | 每批最多32个连续、非对齐4 KiB请求 | `GPU_DIRECT/BYPASS/SHAPED` | 物理请求数小于逻辑请求数 |

`ADMIT`表示请求进入region admission判断，不表示所有小请求都会缓存。冷访问应被
拒绝，热点访问达到阈值后才分配Host Cache line。

## 文件大小与运行方法

默认1024个测量请求需要至少128 MiB真实文件：

```bash
dd if=/dev/urandom of=/mnt/gds/groute-design1.bin bs=1M count=128 status=progress
```

```bash
DESIGN1_FILE=/mnt/gds/groute-design1.bin \
RESULT_ROOT=/mnt/gds/results/groute-design1 \
REPEATS=5 REQUESTS=1024 BATCH_SIZE=32 \
bash scripts/run_design1_policy.sh
```

冷随机请求要求每个请求对应不同的64 KiB cache line。所需文件大小近似为：

```text
max(128 MiB, REQUESTS × 64 KiB + 8 KiB)
```

因此，如果保持`REQUESTS=4096`，文件至少需要约256 MiB。脚本会在运行前检查，
不会通过offset回绕把冷访问静默转换成重复访问。

## 结果解释

单次策略不符合预期、cold workload出现cache hit/admission、hot workload未全部命中，
或shaped workload没有减少物理请求时，benchmark直接失败。输出包括：

- 每次运行的JSON与日志；
- `raw_results.csv`：逐次policy、性能、cache和shaping统计；
- `summary.csv`：按workload汇总的中位数与标准差。

该实验能回答“IOContext是否作出预期决策”和“决策机制是否真正生效”。若要证明
所选策略接近最优，还需要在完全相同offset trace上强制运行Host、GDS Direct和
GDS Shaped基线。

## 论文最终评估仍需补充的实验

当前脚本是机制正确性实验，不应单独承担完整性能结论。最终Evaluation至少还需要：

1. **同trace强制策略对比**：每条trace分别运行Host Direct、Host Cache、GDS Direct
   与GDS Shaped，证明IOContext所选策略接近该trace下的最优策略；
2. **决策开销**：报告前64次profiling的额外延迟、分类时间和稳态每请求开销；
3. **边界与敏感性**：扫描I/O size、reuse ratio、mergeable ratio、alignment
   amplification、collector timeout、cache capacity和admission threshold；
4. **混合及相变trace**：顺序、随机、复用阶段在同一FileHandle中切换，验证是否需要
   coarse-grained re-profiling；当前policy在首次64请求后保持不变，该实验必须明确
   其适用边界；
5. **并发扩展性**：扫描KvikIO线程数、提交线程数和并发FileHandle，报告IOPS、带宽、
   CPU利用率、p50/p95/p99及失败率。

性能实验应使用至少8192个请求和10次重复。1024请求仅产生32个batch latency样本，
其p99接近单次最大值，不适合作为论文尾延迟结论。
