#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Compare GustANN's host AIO loader with the KvikIO G-Route loader on the
# same DiskANN SSD index and query workload.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
GUSTANN_DIR="${GUSTANN_DIR:?Set GUSTANN_DIR to the patched GustANN checkout}"
GUSTANN_BUILD_DIR="${GUSTANN_BUILD_DIR:-${GUSTANN_DIR}/build-groute}"
GUSTANN_BIN="${GUSTANN_BIN:-${GUSTANN_BUILD_DIR}/bin/search_disk_hybrid}"
INDEX_FILE="${INDEX_FILE:?Set INDEX_FILE to DiskANN's *_disk.index file}"
QUERY_FILE="${QUERY_FILE:?Set QUERY_FILE to a .fvecs or .bvecs query file}"
GT_FILE="${GT_FILE:?Set GT_FILE to a matching .ivecs ground-truth file}"
PQ_PREFIX="${PQ_PREFIX:?Set PQ_PREFIX to DiskANN's PQ-data prefix}"
NAV_GRAPH="${NAV_GRAPH:?Set NAV_GRAPH to GustANN's navigation-graph prefix}"
DATA_TYPE="${DATA_TYPE:-float}"
TOPK="${TOPK:-10}"
EF_SEARCH="${EF_SEARCH:-100}"
MINIBATCH="${MINIBATCH:-32}"
SEARCH_THREADS="${SEARCH_THREADS:-4}"
CTX_PER_THREAD="${CTX_PER_THREAD:-4}"
QUERY_REPEATS="${QUERY_REPEATS:-1}"
REPEATS="${REPEATS:-5}"
MODES="${MODES:-aio groute}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/groute-diskann-e2e}"
DROP_CACHES="${DROP_CACHES:-0}"
KVIKIO_THREADS="${KVIKIO_THREADS:-${SEARCH_THREADS}}"

for path in "${GUSTANN_BIN}" "${INDEX_FILE}" "${QUERY_FILE}" "${GT_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${GUSTANN_BIN}" ]]; then
  echo "GustANN binary is not executable: ${GUSTANN_BIN}" >&2
  exit 2
fi
if [[ "${DATA_TYPE}" != "float" && "${DATA_TYPE}" != "uint8" ]]; then
  echo "DATA_TYPE must be float or uint8" >&2
  exit 2
fi
for integer in "${TOPK}" "${EF_SEARCH}" "${MINIBATCH}" "${SEARCH_THREADS}" \
               "${CTX_PER_THREAD}" "${QUERY_REPEATS}" "${REPEATS}" "${KVIKIO_THREADS}"; do
  if ! [[ "${integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "All numeric controls must be positive integers; got ${integer}" >&2
    exit 2
  fi
done
if (( EF_SEARCH < TOPK )); then
  echo "EF_SEARCH must be at least TOPK" >&2
  exit 2
fi
read -r -a mode_array <<<"${MODES}"
if (( ${#mode_array[@]} == 0 )); then
  echo "MODES must contain aio, groute, or both" >&2
  exit 2
fi
for mode in "${mode_array[@]}"; do
  if [[ "${mode}" != "aio" && "${mode}" != "groute" ]]; then
    echo "Unsupported mode: ${mode}; expected aio or groute" >&2
    exit 2
  fi
done

mkdir -p "${RESULT_ROOT}"
rm -f "${RESULT_ROOT}/failed_runs.txt"
status=0

drop_linux_caches() {
  if [[ "${DROP_CACHES}" != "1" ]]; then
    return
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "DROP_CACHES=1 requires root" >&2
    exit 2
  fi
  sync
  echo 3 > /proc/sys/vm/drop_caches
}

run_one() {
  local mode=$1
  local repeat=$2
  local name="diskann_${mode}_r${repeat}"
  local log="${RESULT_ROOT}/${name}.log"
  echo "Running backend=${mode}, repeat=${repeat}"
  drop_linux_caches

  if env \
      KVIKIO_COMPAT_MODE=OFF \
      KVIKIO_GDS_THRESHOLD=0 \
      KVIKIO_TASK_SIZE=4096 \
      KVIKIO_NTHREADS="${KVIKIO_THREADS}" \
      KVIKIO_HOST_CACHE=1 \
      KVIKIO_HOST_CACHE_CAPACITY="${KVIKIO_HOST_CACHE_CAPACITY:-1073741824}" \
      KVIKIO_HOST_CACHE_LINE_SIZE="${KVIKIO_HOST_CACHE_LINE_SIZE:-65536}" \
      KVIKIO_HOST_CACHE_MAX_IO_SIZE="${KVIKIO_HOST_CACHE_MAX_IO_SIZE:-65536}" \
      KVIKIO_HOST_CACHE_REGION_SIZE="${KVIKIO_HOST_CACHE_REGION_SIZE:-1048576}" \
      KVIKIO_HOST_CACHE_ADMISSION_THRESHOLD="${KVIKIO_HOST_CACHE_ADMISSION_THRESHOLD:-2}" \
      KVIKIO_HOST_CACHE_MAX_REGIONS="${KVIKIO_HOST_CACHE_MAX_REGIONS:-4096}" \
      KVIKIO_REQUEST_SHAPING=1 \
      "${GUSTANN_BIN}" \
        --query "${QUERY_FILE}" \
        --index "${INDEX_FILE}" \
        --ground_truth "${GT_FILE}" \
        --data_type "${DATA_TYPE}" \
        --topk "${TOPK}" \
        --ef_search "${EF_SEARCH}" \
        --pq_data "${PQ_PREFIX}" \
        --nav_graph "${NAV_GRAPH}" \
        --repeat "${QUERY_REPEATS}" \
        --minibatch "${MINIBATCH}" \
        --thread "${SEARCH_THREADS}" \
        --ctx_per_thread "${CTX_PER_THREAD}" \
        --io_backend "${mode}" >"${log}" 2>&1; then
    if ! grep -q '^\[REPORT\] Time ' "${log}" || \
       ! grep -q '^\[REPORT\] RECALL:' "${log}"; then
      echo "${name},exit=missing_report,log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
      status=1
    elif [[ "${mode}" == "groute" ]] && ! grep -q '^\[GROUTE_STATS\] ' "${log}"; then
      echo "${name},exit=missing_groute_stats,log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
      status=1
    else
      echo "PASS: ${log}"
    fi
  else
    local rc=$?
    echo "${name},exit=${rc},log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
    status=1
  fi
}

for repeat in $(seq 1 "${REPEATS}"); do
  if (( repeat % 2 == 0 )); then
    for ((i=${#mode_array[@]} - 1; i>=0; --i)); do
      run_one "${mode_array[i]}" "${repeat}"
    done
  else
    for mode in "${mode_array[@]}"; do
      run_one "${mode}" "${repeat}"
    done
  fi
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_diskann_groute.py" \
  --result-root "${RESULT_ROOT}" \
  --query-file "${QUERY_FILE}" \
  --data-type "${DATA_TYPE}" \
  --query-repeats "${QUERY_REPEATS}" || status=1

echo "Results: ${RESULT_ROOT}"
exit "${status}"
