#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end BFS/PageRank: the graph executor is fixed and only the KvikIO policy changes.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GRAPH_PREFIX="${GRAPH_PREFIX:?Set GRAPH_PREFIX to the BaM-format graph prefix}"
GRAPH_E2E_BIN="${GRAPH_E2E_BIN:-${REPO_ROOT}/build/graph-e2e/groute_graph_e2e}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/groute-graph-e2e}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALGORITHMS="${ALGORITHMS:-bfs pagerank}"
POLICIES="${POLICIES:-kvikio_threshold auto host_direct host_cache gds_direct gds_shaped}"
REPEATS="${REPEATS:-5}"
GPU="${GPU:-0}"
BFS_SOURCE="${BFS_SOURCE:-1}"
BFS_MAX_LEVELS="${BFS_MAX_LEVELS:-100}"
PAGERANK_ITERATIONS="${PAGERANK_ITERATIONS:-10}"
STAGING_BYTES="${STAGING_BYTES:-268435456}"
BATCH_REQUESTS="${BATCH_REQUESTS:-32}"
MAX_SEGMENT_BYTES="${MAX_SEGMENT_BYTES:-262144}"
CUDA_THREADS="${CUDA_THREADS:-256}"
KVIKIO_NTHREADS="${KVIKIO_NTHREADS:-4}"
KVIKIO_THRESHOLD_BYTES="${KVIKIO_THRESHOLD_BYTES:-16384}"
PAGE_CACHE_MODE="${PAGE_CACHE_MODE:-file}"
HOST_CACHE_BYTES="${HOST_CACHE_BYTES:-1073741824}"
PHASE_MIN_REQUESTS="${PHASE_MIN_REQUESTS:-100000}"
PHASE_P50_BYTES="${PHASE_P50_BYTES:-16384}"

for path in "${GRAPH_E2E_BIN}" "${GRAPH_PREFIX}.col" "${GRAPH_PREFIX}.dst"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
[[ -x "${GRAPH_E2E_BIN}" ]] || { echo "Not executable: ${GRAPH_E2E_BIN}" >&2; exit 2; }
for integer in "${REPEATS}" "${BFS_MAX_LEVELS}" "${PAGERANK_ITERATIONS}" \
               "${STAGING_BYTES}" "${BATCH_REQUESTS}" "${MAX_SEGMENT_BYTES}" \
               "${CUDA_THREADS}" "${KVIKIO_NTHREADS}" \
               "${PHASE_MIN_REQUESTS}" "${PHASE_P50_BYTES}"; do
  [[ "${integer}" =~ ^[1-9][0-9]*$ ]] || { echo "Positive integer expected: ${integer}" >&2; exit 2; }
done
[[ "${PAGE_CACHE_MODE}" =~ ^(none|file|global)$ ]] || {
  echo "PAGE_CACHE_MODE must be none, file, or global" >&2; exit 2;
}
if [[ "${PAGE_CACHE_MODE}" == "global" && "$(id -u)" -ne 0 ]]; then
  echo "PAGE_CACHE_MODE=global requires root" >&2
  exit 2
fi

read -r -a algorithm_array <<<"${ALGORITHMS}"
read -r -a policy_array <<<"${POLICIES}"
valid_policies=" kvikio_threshold auto auto_phase host_direct host_cache gds_direct gds_shaped "
for policy in "${policy_array[@]}"; do
  [[ "${valid_policies}" == *" ${policy} "* ]] || { echo "Unsupported policy: ${policy}" >&2; exit 2; }
done
for algorithm in "${algorithm_array[@]}"; do
  [[ "${algorithm}" == "bfs" || "${algorithm}" == "pagerank" ]] || {
    echo "Unsupported algorithm: ${algorithm}" >&2; exit 2;
  }
  for policy in "${policy_array[@]}"; do
    if [[ "${algorithm}" != "bfs" && "${policy}" == "auto_phase" ]]; then
      echo "auto_phase is supported only for BFS; set ALGORITHMS=bfs" >&2
      exit 2
    fi
  done
done

mkdir -p "${RESULT_ROOT}"
rm -f "${RESULT_ROOT}/failed_runs.txt"
status=0

evict_page_cache() {
  if [[ "${PAGE_CACHE_MODE}" == "none" ]]; then return; fi
  sync
  if command -v vmtouch >/dev/null 2>&1; then
    vmtouch -e "${GRAPH_PREFIX}.dst" >/dev/null 2>&1 || true
  else
    "${PYTHON_BIN}" - "${GRAPH_PREFIX}.dst" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
PY
  fi
  if [[ "${PAGE_CACHE_MODE}" == "global" ]]; then echo 3 > /proc/sys/vm/drop_caches; fi
}

run_one() {
  local algorithm=$1 policy=$2 repeat=$3 order=$4
  local name="e2e_${algorithm}_${policy}_r${repeat}"
  local output="${RESULT_ROOT}/${name}.json"
  local temporary="${output}.tmp"
  local log="${RESULT_ROOT}/${name}.log"
  local groute=1 threshold=0 effective="${policy}"
  if [[ "${policy}" == "kvikio_threshold" ]]; then
    groute=0
    threshold="${KVIKIO_THRESHOLD_BYTES}"
    effective=auto
  fi
  local phase_switch=0
  if [[ "${policy}" == "auto_phase" ]]; then
    effective=auto
    phase_switch=1
  fi
  local host_cache=0 shaping=0
  [[ "${policy}" == "auto" || "${policy}" == "auto_phase" ||
     "${policy}" == "host_cache" ]] && host_cache=1
  [[ "${policy}" == "auto" || "${policy}" == "auto_phase" ||
     "${policy}" == "gds_shaped" ]] && shaping=1
  local iterations="${BFS_MAX_LEVELS}"
  [[ "${algorithm}" == "pagerank" ]] && iterations="${PAGERANK_ITERATIONS}"

  evict_page_cache
  rm -f "${temporary}"
  echo "Running algorithm=${algorithm}, policy=${policy}, repeat=${repeat}"
  if env \
      KVIKIO_COMPAT_MODE=OFF \
      KVIKIO_GROUTE_ENABLED="${groute}" \
      KVIKIO_GDS_THRESHOLD="${threshold}" \
      KVIKIO_TASK_SIZE=4096 \
      KVIKIO_NTHREADS="${KVIKIO_NTHREADS}" \
      KVIKIO_POLICY_MODE="${effective}" \
      KVIKIO_HOST_CACHE="${host_cache}" \
      KVIKIO_HOST_CACHE_CAPACITY="${HOST_CACHE_BYTES}" \
      KVIKIO_REQUEST_SHAPING="${shaping}" \
      "${GRAPH_E2E_BIN}" \
        --algorithm "${algorithm}" --graph "${GRAPH_PREFIX}" --policy "${policy}" \
        --output "${temporary}" --source "${BFS_SOURCE}" --max-iterations "${iterations}" \
        --staging-bytes "${STAGING_BYTES}" --batch-requests "${BATCH_REQUESTS}" \
        --max-segment-bytes "${MAX_SEGMENT_BYTES}" --threads "${CUDA_THREADS}" \
        --phase-switch "${phase_switch}" --phase-min-requests "${PHASE_MIN_REQUESTS}" \
        --phase-p50-bytes "${PHASE_P50_BYTES}" \
        --gpu "${GPU}" --repeat-id "${repeat}" --execution-order "${order}" \
        >"${log}" 2>&1; then
    mv "${temporary}" "${output}"
    echo "PASS: ${output}"
  else
    local rc=$?
    rm -f "${temporary}"
    echo "${name},exit=${rc},log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
    status=1
  fi
}

for repeat in $(seq 1 "${REPEATS}"); do
  for algorithm in "${algorithm_array[@]}"; do
    count=${#policy_array[@]}
    for ((order=0; order<count; ++order)); do
      index=$(((order + repeat - 1) % count))
      run_one "${algorithm}" "${policy_array[index]}" "${repeat}" "${order}"
    done
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_graph_e2e.py" \
  --result-root "${RESULT_ROOT}" || status=1
if [[ "${PLOT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/plot_graph_e2e.py" \
    --summary "${RESULT_ROOT}/summary.csv" --output-prefix "${RESULT_ROOT}/graph_e2e" || status=1
fi
echo "Results: ${RESULT_ROOT}"
exit "${status}"
