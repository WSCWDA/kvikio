#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Extract BFS/PageRank page-reference traces from a BaM-format CSR graph and
# replay exactly the same traces through native KvikIO and G-Route policies.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
GRAPH_PREFIX="${GRAPH_PREFIX:?Set GRAPH_PREFIX to a BaM graph prefix}"
TRACE_ROOT="${TRACE_ROOT:-/tmp/groute-graph-traces}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/groute-graph-matrix}"
ALGORITHMS="${ALGORITHMS:-bfs pagerank}"
POLICIES="${POLICIES:-kvikio_threshold auto host_direct host_cache gds_direct gds_shaped}"
REPEATS="${REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
KVIKIO_NTHREADS="${KVIKIO_NTHREADS:-4}"
KVIKIO_THRESHOLD_BYTES="${KVIKIO_THRESHOLD_BYTES:-16384}"
PAGE_CACHE_MODE="${PAGE_CACHE_MODE:-file}"
BFS_SOURCE="${BFS_SOURCE:-1}"
BFS_MAX_LEVELS="${BFS_MAX_LEVELS:-100}"
PAGERANK_ITERATIONS="${PAGERANK_ITERATIONS:-10}"
MAX_TRACE_REQUESTS="${MAX_TRACE_REQUESTS:-0}"

for path in "${GRAPH_PREFIX}.col" "${GRAPH_PREFIX}.dst"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing BaM graph file: ${path}" >&2
    exit 2
  fi
done
for integer in "${REPEATS}" "${BATCH_SIZE}" "${KVIKIO_NTHREADS}" \
               "${BFS_MAX_LEVELS}" "${PAGERANK_ITERATIONS}"; do
  if ! [[ "${integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Positive integer expected, got: ${integer}" >&2
    exit 2
  fi
done
if [[ "${PAGE_CACHE_MODE}" == "global" && "$(id -u)" -ne 0 ]]; then
  echo "PAGE_CACHE_MODE=global requires root" >&2
  exit 2
fi

mkdir -p "${TRACE_ROOT}" "${RESULT_ROOT}"
rm -f "${RESULT_ROOT}/failed_runs.txt"
read -r -a algorithm_array <<<"${ALGORITHMS}"
read -r -a policy_array <<<"${POLICIES}"
status=0

trace_path() {
  echo "${TRACE_ROOT}/$(basename "${GRAPH_PREFIX}")-$1-p4096.trace"
}

prepare_trace() {
  local algorithm=$1
  local trace
  trace=$(trace_path "${algorithm}")
  local args=(
    --algorithm "${algorithm}"
    --graph "${GRAPH_PREFIX}"
    --output "${trace}"
    --page-size 4096
  )
  if [[ "${algorithm}" == "bfs" ]]; then
    args+=(--source "${BFS_SOURCE}" --max-levels "${BFS_MAX_LEVELS}")
  else
    args+=(--iterations "${PAGERANK_ITERATIONS}")
  fi
  if (( MAX_TRACE_REQUESTS > 0 )); then
    args+=(--max-requests "${MAX_TRACE_REQUESTS}")
  fi
  echo "Generating ${algorithm} trace: ${trace}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/graph/generate_graph_trace.py" \
    "${args[@]}" >"${trace}.generate.log"
}

for algorithm in "${algorithm_array[@]}"; do
  if [[ "${algorithm}" != "bfs" && "${algorithm}" != "pagerank" ]]; then
    echo "Unsupported algorithm: ${algorithm}" >&2
    exit 2
  fi
  trace=$(trace_path "${algorithm}")
  if [[ ! -f "${trace}" || ! -f "${trace}.json" || "${REGENERATE_TRACES:-0}" == "1" ]]; then
    prepare_trace "${algorithm}"
  fi
done

for repeat in $(seq 1 "${REPEATS}"); do
  for algorithm in "${algorithm_array[@]}"; do
    trace=$(trace_path "${algorithm}")
    count=${#policy_array[@]}
    for ((position=0; position<count; ++position)); do
      # Rotate policy order each repeat to avoid a systematic first-run bias.
      index=$(((position + repeat - 1) % count))
      policy=${policy_array[index]}
      name="graph_${algorithm}_${policy}_r${repeat}"
      output="${RESULT_ROOT}/${name}.json"
      log="${RESULT_ROOT}/${name}.log"
      temporary="${output}.tmp"
      rm -f "${temporary}"
      echo "Running algorithm=${algorithm}, policy=${policy}, repeat=${repeat}"
      if "${PYTHON_BIN}" -m kvikio.benchmarks.graph_trace_replay \
          --trace "${trace}" \
          --policy "${policy}" \
          --batch-size "${BATCH_SIZE}" \
          --num-threads "${KVIKIO_NTHREADS}" \
          --kvikio-threshold "${KVIKIO_THRESHOLD_BYTES}" \
          --page-cache-mode "${PAGE_CACHE_MODE}" \
          --repeat-id "${repeat}" \
          --execution-order "${position}" \
          --output "${temporary}" >"${log}" 2>&1; then
        mv "${temporary}" "${output}"
        echo "PASS: ${output}"
      else
        rc=$?
        rm -f "${temporary}"
        echo "${name},exit=${rc},log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
        status=1
      fi
    done
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_graph_trace.py" \
  --result-root "${RESULT_ROOT}" || status=1
echo "Results: ${RESULT_ROOT}"
exit "${status}"
