# Frequency–Momentum 独立准入消融

该实验只重放 cache-line trace，不执行 SSD、CUDA 或 H2D I/O。它用于在接入 G-Route
之前隔离验证 reuse estimator、candidate–victim admission 和成本模型。

## 策略

- `cache_all`：每个 miss 都插入，LRU 淘汰；
- `frequency`：只使用长期 Blocked CBF；
- `momentum`：只使用短期 Blocked CBF；
- `hybrid`：Frequency 或 Momentum 达阈值即准入；
- `dual_max`：使用 `max(F/Ft, M/Mt)` 比较 candidate 和 LRU victim；
- `dual_weighted`：使用 `0.5 F/Ft + 0.5 M/Mt`；
- `dual_multiplicative`：使用 `(1+F/Ft)(1+M/Mt)-1`；
- `dual_lexicographic`：依次比较 momentum-ready、frequency-ready、M/Mt、F/Ft。

四种 dual policy 只改变 replacement admission；victim 始终由 LRU 选择。candidate 与
victim 分数相同则拒绝 candidate。

## 收益与污染

相对所有请求都走 bypass：

```text
net_saved_ns = hits * (bypass_ns - hit_ns)
             - admissions * (fill_ns - bypass_ns)
```

`victim_reaccess_misses` 只表示被淘汰对象后来再次出现。实验同时维护
`reject_on_full` shadow cache：它与 primary 共享 trace、阈值和初始填充规则，但缓存满后
拒绝 replacement。`primary miss && shadow hit` 才计为
`counterfactual_pollution_misses`。该指标是相对于明确 shadow policy 的反事实，不是与基线
无关的绝对污染真值。

合成 trace 带有 `hot`、`scan`、`two_reference_burst` 标签，因此可以分别报告 scan
淘汰热点和 two-reference 第二次请求命中率。`momentum_threshold=1` 允许首次出现的候选
参与准入；它无法预先区分一次性 scan 与未来 burst，scan 保护由 candidate–victim 比较
承担。

## 无数据泄漏的数据划分

`splits.json` 固定以下集合：

- `tuning`：在 `stable_zipf`、`phase_shift_aba` 和训练 seed 上选择参数；
- `validation`：相同 trace 的不同 seed，只确认方向；
- `final_in_domain`：相同 trace 的最终保留 seed；
- `final_heldout_trace`：未参与选参的 `hot_scan_mix`、`two_reference_burst`。

`analyze_phase1.py select` 先最大化 tuning 平均 `net_saved_ns_per_request`，保留与最优值
相差不超过 1% 的配置，再依次最小化反事实污染和 CPU 决策开销。最终 replay 只能读取生成
的 `selection.json`，不能再次搜索阈值。

## 正确性测试

```bash
python experiments/frequency_momentum_ablation/test_phase1.py -v
```

测试覆盖 signed benefit、M=1 two-reference hit、scan 热点保护、四种 score 以及 shadow
pollution 与 victim reaccess 的分离。

## 完整阶段一实验

```bash
cd /home/cwd/gds/kvikio_opt/kvikio-line-admission
bash experiments/frequency_momentum_ablation/run_phase1.sh run_01
```

再次复现时使用：

```bash
bash experiments/frequency_momentum_ablation/run_phase1.sh run_02
```

结果目录不使用时间戳：

```text
/mnt/gds/results/groute_phase1_cost_sensitivity/run_01/
/mnt/gds/results/groute_phase1_momentum_threshold1/run_01/
/mnt/gds/results/groute_phase1_score_ablation/run_01/
/mnt/gds/results/groute_phase1_shadow_pollution/run_01/
/mnt/gds/results/groute_phase1_holdout_evaluation/run_01/
```

每次 `run.py` 调用还可显式指定：

```bash
python experiments/frequency_momentum_ablation/run.py \
  --requests 30000 \
  --cache-lines 2 4 \
  --momentum-windows 128 512 \
  --frequency-thresholds 3 4 \
  --momentum-thresholds 1 2 3 4 \
  --score-modes max weighted multiplicative lexicographic \
  --fill-ns-values 52784 58000 62000 72000 90000 \
  --splits-file experiments/frequency_momentum_ablation/splits.json \
  --split tuning \
  --output-dir /mnt/gds/results/groute_phase1_manual/run_01
```

若省略 `--output-dir`，结果工具自动使用：

```text
/mnt/gds/results/groute_frequency_momentum_ablation/run_01
/mnt/gds/results/groute_frequency_momentum_ablation/run_02
```

不再生成带 UTC 时间戳和随机后缀的目录。
