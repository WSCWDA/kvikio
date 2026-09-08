#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -u
set -o pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DESIGN2_BENCH_FILE="${DESIGN2_BENCH_FILE:-/mnt/gds/cwd_test/design2-thread-sweep.bin}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/kvikio-design2-thread-sweep-$(date +%Y%m%d-%H%M%S)}"
WORKING_SET_GIB="${WORKING_SET_GIB:-272}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
REQUESTS="${REQUESTS:-8192}"
IO_SIZE="${IO_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPEATS="${REPEATS:-5}"
NTHREADS_LIST="${NTHREADS_LIST:-1 2 4 8 16}"
CLUSTERS_LIST="${CLUSTERS_LIST:-1 4}"
VERIFY="${VERIFY:-1}"
DROP_CACHES="${DROP_CACHES:-1}"
ALLOW_WORKING_SET_LE_RAM="${ALLOW_WORKING_SET_LE_RAM:-0}"

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

for value in "${WORKING_SET_GIB}" "${MIN_FREE_GIB}" "${REQUESTS}" "${IO_SIZE}" "${BATCH_SIZE}" "${REPEATS}"; do
  if ! is_positive_integer "${value}"; then
    echo "Expected a positive integer, got: ${value}" >&2
    exit 2
  fi
done
for flag_name in VERIFY DROP_CACHES ALLOW_WORKING_SET_LE_RAM; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "${flag_name} must be 0 or 1, got: ${flag_value}" >&2
    exit 2
  fi
done
WORKING_SET_BYTES=$((WORKING_SET_GIB * 1024 * 1024 * 1024))
MEM_TOTAL_BYTES=$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo | cut -d. -f1)
if (( WORKING_SET_BYTES <= MEM_TOTAL_BYTES )) && [[ "${ALLOW_WORKING_SET_LE_RAM}" != "1" ]]; then
  echo "WORKING_SET_GIB=${WORKING_SET_GIB} is not larger than MemTotal ($((MEM_TOTAL_BYTES / 1024 / 1024 / 1024)) GiB)" >&2
  echo "Increase WORKING_SET_GIB; do not label this run as larger-than-DRAM." >&2
  exit 2
fi
if [[ "${DROP_CACHES}" == "1" ]]; then
  if [[ "$(id -u)" -ne 0 || ! -w /proc/sys/vm/drop_caches ]]; then
    echo "DROP_CACHES=1 requires running the complete script as root." >&2
    echo "Use: sudo -E bash scripts/run_design2_thread_sweep.sh" >&2
    exit 2
  fi
  cache_args=(--drop-caches)
else
  cache_args=()
fi
verify_args=()
if [[ "${VERIFY}" == "1" ]]; then
  verify_args+=(--verify)
fi

read -r -a nthreads_values <<< "${NTHREADS_LIST}"
read -r -a cluster_values <<< "${CLUSTERS_LIST}"
if [[ ${#nthreads_values[@]} -eq 0 || ${#cluster_values[@]} -eq 0 ]]; then
  echo "NTHREADS_LIST and CLUSTERS_LIST must not be empty" >&2
  exit 2
fi

max_clusters=1
for threads in "${nthreads_values[@]}"; do
  if ! is_positive_integer "${threads}"; then
    echo "Invalid thread count: ${threads}" >&2
    exit 2
  fi
done
for clusters in "${cluster_values[@]}"; do
  if ! is_positive_integer "${clusters}" || (( BATCH_SIZE % clusters != 0 )); then
    echo "Cluster count must be positive and divide BATCH_SIZE: ${clusters}" >&2
    exit 2
  fi
  if (( clusters > max_clusters )); then
    max_clusters="${clusters}"
  fi
done

mkdir -p "${RESULT_ROOT}"
failures_path="${RESULT_ROOT}/failed_runs.txt"
: > "${failures_path}"

DESIGN2_BENCH_FILE="${DESIGN2_BENCH_FILE}" \
WORKING_SET_BYTES="${WORKING_SET_BYTES}" \
MIN_FREE_GIB="${MIN_FREE_GIB}" \
"${PYTHON_BIN}" - <<'PY'
import os
import shutil
from pathlib import Path

path = Path(os.environ["DESIGN2_BENCH_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
target = int(os.environ["WORKING_SET_BYTES"])
reserve = int(os.environ["MIN_FREE_GIB"]) * 1024**3
allocated = path.stat().st_blocks * 512 if path.exists() else 0
additional = max(0, target - allocated)
free = shutil.disk_usage(path.parent).free
if free < additional + reserve:
    raise SystemExit(
        f"insufficient disk space: need {additional + reserve} bytes including "
        f"reserve, have {free} bytes"
    )
print(
    f"Preflight: target={target / 1024**3:.1f} GiB, "
    f"allocated={allocated / 1024**3:.1f} GiB, free={free / 1024**3:.1f} GiB"
)
PY
preflight_status=$?
if [[ ${preflight_status} -ne 0 ]]; then
  exit "${preflight_status}"
fi

echo "Preparing a fully materialized ${WORKING_SET_GIB} GiB file: ${DESIGN2_BENCH_FILE}"
echo "A matching file and .design2-pattern.json marker are reused without rewriting."
DESIGN2_BENCH_FILE="${DESIGN2_BENCH_FILE}" \
REQUESTS="${REQUESTS}" \
IO_SIZE="${IO_SIZE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_CLUSTERS="${max_clusters}" \
WORKING_SET_BYTES="${WORKING_SET_BYTES}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from kvikio.benchmarks.design2_request_shaping import prepare_file, required_file_size

path = Path(os.environ["DESIGN2_BENCH_FILE"])
size = required_file_size(
    int(os.environ["REQUESTS"]),
    int(os.environ["IO_SIZE"]),
    int(os.environ["BATCH_SIZE"]),
    int(os.environ["MAX_CLUSTERS"]),
    int(os.environ["WORKING_SET_BYTES"]),
)
prepare_file(path, size)
print(f"Prepared {path} ({size} bytes)")
PY
prepare_status=$?
if [[ ${prepare_status} -ne 0 ]]; then
  echo "Failed to prepare benchmark file" >&2
  exit "${prepare_status}"
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "benchmark_file=${DESIGN2_BENCH_FILE}"
  echo "working_set_gib=${WORKING_SET_GIB}"
  echo "working_set_bytes=${WORKING_SET_BYTES}"
  echo "mem_total_bytes=${MEM_TOTAL_BYTES}"
  echo "drop_caches=${DROP_CACHES}"
  echo "requests=${REQUESTS}"
  echo "io_size=${IO_SIZE}"
  echo "batch_size=${BATCH_SIZE}"
  echo "repeats=${REPEATS}"
  echo "nthreads=${NTHREADS_LIST}"
  echo "clusters=${CLUSTERS_LIST}"
  echo "verify=${VERIFY}"
  git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
  nvidia-smi -L 2>/dev/null || true
} > "${RESULT_ROOT}/metadata.txt"

run_status=0
for clusters in "${cluster_values[@]}"; do
  for threads in "${nthreads_values[@]}"; do
    for repeat in $(seq 1 "${REPEATS}"); do
      name="design2_c${clusters}_t${threads}_r${repeat}"
      json_path="${RESULT_ROOT}/${name}.json"
      log_path="${RESULT_ROOT}/${name}.log"
      tmp_json="${json_path}.tmp"
      rm -f "${json_path}" "${tmp_json}"

      # Rotate the four paths so repeated runs do not systematically favor a
      # later path through SSD-controller temperature or device-side caching.
      rotation=$(((repeat + threads + clusters) % 4))
      case "${rotation}" in
        0) mode_order="host_buffered host_direct gds_direct gds_shaped" ;;
        1) mode_order="host_direct gds_direct gds_shaped host_buffered" ;;
        2) mode_order="gds_direct gds_shaped host_buffered host_direct" ;;
        3) mode_order="gds_shaped host_buffered host_direct gds_direct" ;;
      esac
      read -r -a mode_values <<< "${mode_order}"

      echo "Running clusters=${clusters}, threads=${threads}, repeat=${repeat}, modes=${mode_order}"
      if KVIKIO_NTHREADS="${threads}" \
        "${PYTHON_BIN}" -m kvikio.benchmarks.design2_request_shaping \
          --file "${DESIGN2_BENCH_FILE}" \
          --requests "${REQUESTS}" \
          --io-size "${IO_SIZE}" \
          --batch-size "${BATCH_SIZE}" \
          --clusters-per-batch "${clusters}" \
          --working-set-bytes "${WORKING_SET_BYTES}" \
          --modes "${mode_values[@]}" \
          "${cache_args[@]}" \
          "${verify_args[@]}" \
          --output "${tmp_json}" > "${log_path}" 2>&1; then
        mv "${tmp_json}" "${json_path}"
        echo "PASS: ${json_path}"
      else
        status=$?
        rm -f "${tmp_json}"
        echo "FAIL: ${name}, exit=${status}; see ${log_path}" >&2
        echo "${name},exit=${status},log=${log_path}" >> "${failures_path}"
        run_status=1
      fi
    done
  done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_design2_thread_sweep.py" "${RESULT_ROOT}"
summary_status=$?
echo "Results: ${RESULT_ROOT}"
echo "Raw CSV: ${RESULT_ROOT}/raw_results.csv"
echo "Summary: ${RESULT_ROOT}/summary.csv"
if [[ -s "${failures_path}" ]]; then
  echo "Failures: ${failures_path}"
fi

if [[ ${run_status} -ne 0 || ${summary_status} -ne 0 ]]; then
  exit 1
fi
