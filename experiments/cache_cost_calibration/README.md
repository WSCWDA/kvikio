# HostCache cost calibration

This experiment isolates the three end-to-end costs used by line admission:

- `hit`: an already resident HostCache line copied to the GPU;
- `fill`: a unique line read from storage into HostCache and then copied to the GPU;
- `host_bypass` / `gds_bypass`: the same logical request without HostCache.

All results are stored below `/mnt/gds/results/groute_*`; individual measurements are never
printed directly to the terminal.

## Data

For the default 8,192 cold requests and 64 KiB lines, the input must be a regular, fully
allocated file of at least 512 MiB. A 2 GiB file leaves room for larger follow-up runs:

```bash
bash experiments/cache_cost_calibration/prepare_data.sh \
  /mnt/gds/groute-cache-cost.bin 2
```

Do not use `truncate` or `fallocate` alone: sparse/unwritten extents can return zeros without
performing representative SSD reads. The preparation script uses direct, synchronous writes and
saves allocation and mount metadata.

## Run

Host path only:

```bash
python experiments/cache_cost_calibration/run.py \
  --file /mnt/gds/groute-cache-cost.bin \
  --io-size 4096 --line-size 65536 --cache-lines 4 \
  --hit-requests 20000 --cold-requests 8192 --repeats 7 \
  --direct-io --drop-file-pages
```

Include forced GDS if the mount and driver support it:

```bash
python experiments/cache_cost_calibration/run.py \
  --file /mnt/gds/groute-cache-cost.bin \
  --io-size 4096 --line-size 65536 --cache-lines 4 \
  --hit-requests 20000 --cold-requests 8192 --repeats 7 \
  --direct-io --drop-file-pages --include-gds
```

Each mode is executed once per repeat and the order rotates. The script validates that a hit run
has no misses/storage reads and that a fill run has only misses and exactly one line of storage I/O
per request. A failed GDS mode is recorded rather than deleting the successful host measurements.

For the buffered POSIX cold-file model, replace `--direct-io` with `--no-direct-io` and retain
`--drop-file-pages`. Do not mix direct and buffered runs in one cost tuple.

## Analyze

Pass the calibration directory printed by `run.py`:

```bash
python experiments/cache_cost_calibration/analyze.py \
  --input /mnt/gds/results/groute_cache_cost_calibration_<timestamp>_<id>
```

The analysis is saved to a new `groute_cache_cost_analysis_*` directory. It reports the median of
the seven per-run p50 values and computes:

```text
minimum_future_hits = ceil(max(fill_ns - bypass_ns, 0) / (bypass_ns - hit_ns))
```

If `bypass_ns <= hit_ns`, caching cannot save time under that measured configuration and the
minimum is reported as `null`.

Use the median p50 tuple for admission decisions. Retain p99 as a tail-latency guardrail; do not
substitute internal `storage_read_ns` for `fill_ns`, because it excludes lookup, replacement, H2D
submission, and completion.
