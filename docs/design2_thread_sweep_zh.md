# Design2 线程扫描实验教程

本实验扫描 `KVIKIO_NTHREADS=1/2/4/8/16`，回答两个独立问题：增加 KvikIO 设备线程
是否让多个 physical plan 真正并行，以及 Shaped GDS 在相同线程配置下是否仍优于 Direct
GDS。每个测试点启动全新的 Python 进程，避免设备级静态线程池沿用上一个配置。

## 1. 更新和编译

```bash
git pull origin codex/gds-hcache-v26.06.00
./build.sh libkvikio kvikio --pydevelop
```

确认当前环境和 GPU：

```bash
python - <<'PY'
import kvikio
import kvikio.defaults

print("kvikio:", kvikio.__file__)
print("default num_threads:", kvikio.defaults.get("num_threads"))
PY
nvidia-smi -L
```

默认线程数为 1。不要在同一个 Python 进程内依次修改线程数，因为 KvikIO 的设备级线程池
在第一次打开文件时创建并缓存。

## 2. 运行完整矩阵

```bash
cd /home/cwd/gds/kvikio_opt/kvikio

DESIGN2_BENCH_FILE=/mnt/gds/cwd_test/design2-thread-sweep.bin \
RESULT_ROOT=/mnt/gds/cwd_test/design2-thread-sweep-results \
REPEATS=5 \
NTHREADS_LIST="1 2 4 8 16" \
CLUSTERS_LIST="1 4" \
bash scripts/run_design2_thread_sweep.sh
```

默认参数为 8192 个 4 KiB 请求、逻辑 batch size 32，并执行数据正确性校验。脚本只在开始
时创建一次测试文件。之后每个配置均通过独立命令：

```bash
KVIKIO_NTHREADS=<线程数> python -m kvikio.benchmarks.design2_request_shaping ...
```

其中 `clusters=1` 测试每批合并为一个大 plan 的端到端收益；`clusters=4` 测试每批产生
4 个独立 plan，用于验证 staging buffer pool 和设备线程池并发。

快速冒烟可使用：

```bash
REPEATS=1 NTHREADS_LIST="1 4 8" CLUSTERS_LIST="4" \
DESIGN2_BENCH_FILE=/mnt/gds/cwd_test/design2-thread-sweep.bin \
bash scripts/run_design2_thread_sweep.sh
```

如果暂时不需要逐请求数据校验，可以设置 `VERIFY=0`；正式论文实验应保留默认校验。

## 3. 输出文件

结果目录包含：

- `design2_c4_t8_r3.json`：四簇、8线程、第3次的完整结果；
- 同名 `.log`：CuPy、CUDA、GDS错误和标准输出；
- `raw_results.csv`：每次独立运行一行；
- `summary.csv`：按簇数和线程数汇总的中位数；
- `failed_runs.txt`：失败配置、退出码及对应日志；
- `metadata.txt`：参数、Git提交和GPU信息。

每次运行前会删除该测试点可能残留的旧JSON。因此失败测试不会被同名历史结果计入
`summary.csv`。

查看核心结果：

```bash
column -s, -t /mnt/gds/cwd_test/design2-thread-sweep-results/summary.csv | less -S
```

也可以重新汇总已有JSON：

```bash
python scripts/summarize_design2_thread_sweep.py \
  /mnt/gds/cwd_test/design2-thread-sweep-results
```

## 4. 判读规则

首先检查 `clusters=4`：

1. `max_inflight_physical_max` 应从线程1时的1增加到2至4；否则没有证据证明 physical plan
   并发执行。
2. `max_collected_requests_max` 理想为32；若持续低于32，瓶颈仍包括200微秒收集窗口。
3. `logical_per_physical_median` 理想接近8，`physical_reduction_median` 理想接近0.875。
4. `amplification_median` 理想接近1.125，明显低于逐请求对齐的2.0。

然后比较性能：

- `shaped_vs_direct_median > 1`：Design2在相同线程数下优于逐请求GDS；
- `shaped_vs_host_median > 1`：Shaped GDS超过当前Host路径；
- 如果Direct随线程增加而增长、Shaped没有增长，瓶颈在收集/D2D/Event阶段；
- 如果二者在相同线程点同时饱和，瓶颈更可能位于SSD、NVFS或PCIe路径。

论文结论必须采用同一线程数下的Direct和Shaped结果，不应使用单线程Direct与多线程Shaped
交叉比较。建议报告5次中位数和标准差，并保留所有原始JSON。

## 5. 可选的nvidia-fs验证

```bash
sudo sh -c 'echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled'
grep -E 'IO stats|Ops' /proc/driver/nvidia-fs/stats
```

扫描完成后：

```bash
sudo sh -c 'echo 0 > /sys/module/nvidia_fs/parameters/rw_stats_enabled'
```

CuPy多包安装警告不会改变线程扫描逻辑，但正式实验前仍建议只保留与CUDA版本匹配的一套
CuPy，避免把环境不稳定性带入论文数据。

修复或修改并发执行器后，建议先运行高并发析构压力测试：

```bash
RESULT_ROOT=/mnt/gds/cwd_test/design2-thread-sweep-soak \
REPEATS=20 \
NTHREADS_LIST="8 16" \
CLUSTERS_LIST="1 4" \
bash scripts/run_design2_thread_sweep.sh

test ! -s /mnt/gds/cwd_test/design2-thread-sweep-soak/failed_runs.txt
```

只有全部80个独立进程均正常退出，才继续使用该版本生成论文性能数据。
