# Design2 线程扫描实验教程

本实验扫描 `KVIKIO_NTHREADS=1/2/4/8/16`，在大于DRAM的272 GiB数据文件上比较四条路径：

- `host_buffered`：冷Page Cache下的POSIX buffered read，再H2D复制；
- `host_direct`：POSIX `O_DIRECT`对齐扩读，再H2D复制；
- `gds_direct`：每个逻辑请求独立通过GDS提交；
- `gds_shaped`：跨请求收集、对齐和合并后通过GDS提交。

每种路径计时前都执行`sync + POSIX_FADV_DONTNEED + drop_caches=3`，并使用分散在完整文件
范围内且不重复的测量offset。这里消除的是Page Cache命中收益；`host_buffered`仍然经过Linux
Page Cache，只是每个测量页在本轮第一次访问时处于cold状态。

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

sudo -E env \
DESIGN2_BENCH_FILE=/mnt/gds/cwd_test/design2-cold-272g.bin \
RESULT_ROOT=/mnt/gds/cwd_test/design2-four-path-results \
WORKING_SET_GIB=272 \
REPEATS=5 \
NTHREADS_LIST="1 2 4 8 16" \
CLUSTERS_LIST="1 4" \
bash scripts/run_design2_thread_sweep.sh
```

脚本必须以root运行，因为非root进程不能写`/proc/sys/vm/drop_caches`。默认会检查数据文件
大于`/proc/meminfo`中的`MemTotal`，并在剩余磁盘空间不足时提前终止。

第一次运行会完整写入272 GiB确定性数据，不能使用`truncate`或仅`fallocate`：未写入extent
可能由文件系统直接返回零，无法证明访问了SSD。后续运行若文件大小和
`.design2-pattern.json`标记匹配，则直接复用，不再重写。

默认参数为8192个4 KiB请求、逻辑batch size 32，并执行全请求数据正确性校验。请求offset
分散到272 GiB文件范围内，但单轮实际读取量仍是`requests × io_size = 32 MiB`；因此论文中应
称其为“272 GiB dataset/address domain上的cold random sampling”，不能声称每轮扫描了272 GiB。
每个模式
先完成不含校验开销的计时阶段，再完整重放相同的 8192 个请求；重放时每个 wave 都在 GPU
buffer 被复用前逐字节校验。预期数据由测试文件的确定性内容直接生成，不通过 POSIX 再读文件，
避免校验过程污染计时阶段。四条路径的顺序会随repeat轮转，降低固定顺序造成的SSD温度或
控制器缓存偏差。脚本只在开始时创建一次测试文件。之后每个配置均通过独立命令：

```bash
sudo KVIKIO_NTHREADS=<线程数> python -m kvikio.benchmarks.design2_request_shaping \
  --working-set-bytes 292057776128 --drop-caches ...
```

其中 `clusters=1` 测试每批合并为一个大 plan 的端到端收益；`clusters=4` 测试每批产生
4 个独立 plan，用于验证 staging buffer pool 和设备线程池并发。

快速冒烟可使用：

```bash
sudo -E env REPEATS=1 NTHREADS_LIST="1 4 8" CLUSTERS_LIST="4" \
WORKING_SET_GIB=272 \
DESIGN2_BENCH_FILE=/mnt/gds/cwd_test/design2-cold-272g.bin \
bash scripts/run_design2_thread_sweep.sh
```

如果暂时不需要全请求校验，可以设置 `VERIFY=0`；正式论文实验应保留默认校验，并确认
`fully_verified_runs == runs`。

只补测此前稳定性矩阵中缺少的 1/2/4 线程性能点，可运行：

```bash
DESIGN2_BENCH_FILE=/mnt/gds/cwd_test/design2-thread-sweep.bin \
RESULT_ROOT=/mnt/gds/cwd_test/design2-thread-sweep-low-threads \
REPEATS=10 \
NTHREADS_LIST="1 2 4" \
CLUSTERS_LIST="1 4" \
bash scripts/run_design2_thread_sweep.sh
```

## 3. 输出文件

结果目录包含：

- `design2_c4_t8_r3.json`：四簇、8线程、第3次的完整结果；
- 同名 `.log`：CuPy、CUDA、GDS错误和标准输出；
- `raw_results.csv`：每次独立运行一行；
- `summary.csv`：按簇数和线程数汇总的中位数；
- `failed_runs.txt`：失败配置、退出码及对应日志；
- `metadata.txt`：参数、Git提交和GPU信息。

每个模式的 JSON 还包含：

- `latency_us.{mean,p50,p95,p99,max}`：从调用 `pread()` 前到按提交顺序观察到 Future
  完成为止的逻辑请求端到端延迟；
- `verified_requests` 和 `verified_bytes`：计时后完整重放并校验的数据量；
- `verification_seconds`：完整校验重放耗时，不计入 `elapsed_seconds`、IOPS和延迟。

这里的延迟是应用按提交顺序调用 `Future.get()` 时可观察的完成延迟，包含 collector 等待、物理
I/O和D2D分发，但不是设备内部完成时间；因此可用于比较四种模式的API可见尾延迟，不应解释为
单次 `cuFileRead` 的纯设备延迟。

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
5. `fully_verified_runs` 必须等于 `runs`；否则该配置不能进入论文性能结果。

然后比较性能：

- `shaped_vs_direct_median > 1`：Design2在相同线程数下优于逐请求GDS；
- `shaped_vs_host_buffered_median > 1`：Shaped GDS超过cold buffered Host路径；
- `shaped_vs_host_direct_median > 1`：Shaped GDS超过Host `O_DIRECT`路径；
- 如果Direct随线程增加而增长、Shaped没有增长，瓶颈在收集/D2D/Event阶段；
- 如果二者在相同线程点同时饱和，瓶颈更可能位于SSD、NVFS或PCIe路径。
- `gds_shaped_latency_p95_us_median` 和 `gds_shaped_latency_p99_us_median` 用于判断合并收益是否以明显
  增加尾延迟为代价。

`cold_cache_runs`和`fully_verified_runs`都必须等于`runs`，否则该配置不得进入论文图表。
论文结论必须采用同一线程数下的四路径结果，不应跨线程比较。建议报告至少10次中位数、
标准差和95%置信区间，并保留所有原始JSON。

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

Request Shaper 使用由 `KVIKIO_NTHREADS` 限定的 per-handle executor，但物理并发同时受 4 个
staging slots 限制。因此 8 与 16 线程仍用于退出稳定性压力测试，不应预期 shaped 吞吐随线程数
从 8 到 16 继续线性增长。
