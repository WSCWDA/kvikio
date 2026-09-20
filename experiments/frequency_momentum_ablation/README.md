# Frequency–Momentum 独立准入消融

本实验不执行 SSD、CUDA 或 H2D I/O，只重放 cache-line trace。五种策略共享相同容量的
LRU 数据缓存，因此差异来自准入器，而不是淘汰器或 G-Route 路径选择。

## 策略

- `cache_all`：每个 miss 都插入，作为 LRU 基线；
- `frequency`：仅使用长期 blocked CBF；
- `momentum`：仅使用短期 blocked CBF；
- `hybrid`：Frequency 或 Momentum 达到阈值即可插入。
- `dual_score`：先满足 Hybrid 复用门槛；缓存满时比较 candidate 与当前 LRU victim 的
  `max(frequency/F_threshold, momentum/M_threshold)`，仅在 candidate 严格更高时换入。
  victim 仍然完全由 LRU 选择，因此这是准入消融，不是新的淘汰算法。

两个 tracker 都使用 4-bit counter、四次 hash、conservative update，并将一个 key 的四个
counter 放在同一个 64 B block。`frequency_window` 和 `momentum_window` 表示完成一次全
sketch 衰减所需的请求数，因此改变 sketch 容量不会同时改变历史窗口。victim 分数通过
只读 `estimate()` 获取，不更新 counter，也不推进 aging。

## 合成动态 trace

在仓库根目录执行：

```bash
python experiments/frequency_momentum_ablation/run.py
```

默认配置为 30,000 requests、16-line cache、8 KiB Frequency sketch、1 KiB Momentum
sketch、32,768/512 requests 的长短窗口，并扫描 Frequency 阈值 3/4 和 Momentum 阈值
2/3/4。默认独立运行 5 个 seed（20260920–20260924），每项 CPU 时间重复五次。脚本生成
四类 trace：稳定 Zipf、A→B→A 热点切换、热点混合一次性扫描、短暂 burst。

指定 5 个以上 seed：

```bash
python experiments/frequency_momentum_ablation/run.py \
  --seeds 11 22 33 44 55 66
```

结果只写入：

```text
/mnt/gds/results/groute_frequency_momentum_ablation_<UTC>_<ID>/
  run.json
  results.jsonl
```

## 实际 GPU I/O trace

若第一列是字节 offset：

```bash
python experiments/frequency_momentum_ablation/run.py \
  --trace /path/to/gpu-read-offsets.txt \
  --trace-unit bytes \
  --trace-column 0 \
  --trace-limit 1000000
```

若输入已经是 64 KiB line ID，使用 `--trace-unit lines`。空白和 CSV 均可；文本表头会被
跳过。未标注真实热点的 trace 仍能计算 hit ratio、false admission 和 hits/admission，
但热点识别延迟、hot hit ratio 和 hot eviction 为 `null` 或零。要评价阶段识别，应先按
query batch、epoch 或时间窗口切分 trace 并补充阶段热点标签。

## 主要字段

- `mean_detection_delay_requests`：阶段内热点首次出现到首次进入缓存的请求数；
- `false_admission_rate`：插入后直到淘汰或 trace 结束都未产生 hit 的比例；
- `low_value_admission_rate`（兼容别名 `low_value_admission`）：一次驻留期间的收益
  `hits × (bypass_ns-hit_ns) - (fill_ns-bypass_ns)` 小于等于零的准入比例；
- `net_saved_ns`：相对所有请求均 bypass 的模型化净节省：
  `hits × max(bypass-hit, 0) - admissions × max(fill-bypass, 0)`；
- `pollution_misses`：低价值准入所淘汰的 victim，在重新驻留前再次被请求而形成的 miss。
  这是基于真实重放事件的因果归属指标，不是 shadow-cache 的反事实 miss 差；
- `score_rejections`：通过复用阈值、但因 candidate 分数不高于 LRU victim 而被拒绝的次数；
- `hot_evictions`：插入新 line 时淘汰当前阶段热点的次数；
- `hot_hit_ratio` / `hit_ratio`：热点请求和全部请求的有效命中率；
- `hits_per_admission`：每次缓存填充带来的命中数；
- `median_decision_ns_per_request`：包含 tracker 与统一 LRU 模拟的 CPU replay 开销；
- `pareto`：在 `net_saved_ns`、pollution miss、热点命中率和检测延迟上未被其他配置支配。

默认成本模型为 `hit=10 us`、`fill=200 us`、`bypass=80 us`，可用 `--hit-ns`、
`--fill-ns`、`--bypass-ns` 替换为目标机器的实测中位数。低价值和净收益结论依赖这三个值。

先从 `phase_shift_aba` 中筛选比 Frequency-only 检测更快的 Hybrid 配置，再检查它在
`hot_scan_mix` 和 `short_burst` 中的 false admission、hot eviction 是否可接受。只有同一
配置在多类 trace 上保持 Pareto 或接近 Pareto，才进入 HostCache 和 G-Route 集成阶段。
