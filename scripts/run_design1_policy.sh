#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DESIGN1_FILE="${DESIGN1_FILE:?Set DESIGN1_FILE to an existing, non-sparse SSD file}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/groute-design1-policy}"
REQUESTS="${REQUESTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPEATS="${REPEATS:-5}"
CASES="${CASES:-sequential_large random_cold_small random_hot_small adjacent_unaligned_small}"
POLICIES="${POLICIES:-auto}"
PAGE_CACHE_MODE="${PAGE_CACHE_MODE:-file}"
WORKING_SET_BYTES="${WORKING_SET_BYTES:-}"
ORDER_SEED="${ORDER_SEED:-20260911}"
TRACE_SEED="${TRACE_SEED:-20260910}"

for integer in "${REQUESTS}" "${BATCH_SIZE}" "${REPEATS}"; do
  if ! [[ "${integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REQUESTS, BATCH_SIZE, and REPEATS must be positive integers" >&2
    exit 2
  fi
done

for integer in "${ORDER_SEED}" "${TRACE_SEED}"; do
  if ! [[ "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ORDER_SEED and TRACE_SEED must be non-negative integers" >&2
    exit 2
  fi
done

if [[ -n "${WORKING_SET_BYTES}" ]] && ! [[ "${WORKING_SET_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKING_SET_BYTES must be a positive integer when set" >&2
  exit 2
fi

case "${PAGE_CACHE_MODE}" in
  none | file | global) ;;
  *)
    echo "PAGE_CACHE_MODE must be none, file, or global" >&2
    exit 2
    ;;
esac
if [[ "${PAGE_CACHE_MODE}" == "global" && "$(id -u)" -ne 0 ]]; then
  echo "PAGE_CACHE_MODE=global requires root" >&2
  exit 2
fi

for policy in ${POLICIES}; do
  case "${policy}" in
    auto | host_direct | host_cache | gds_direct | gds_shaped) ;;
    *)
      echo "Unsupported policy '${policy}'." >&2
      echo "Expected: auto host_direct host_cache gds_direct gds_shaped" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${DESIGN1_FILE}" ]]; then
  echo "Missing benchmark file: ${DESIGN1_FILE}" >&2
  exit 2
fi
file_size=$(stat -c %s "${DESIGN1_FILE}")
allocated_bytes=$(( $(stat -c %b "${DESIGN1_FILE}") * 512 ))
if (( allocated_bytes * 100 < file_size * 95 )); then
  echo "DESIGN1_FILE appears sparse: allocated=${allocated_bytes}, size=${file_size}" >&2
  exit 2
fi
if [[ -n "${WORKING_SET_BYTES}" ]] && (( WORKING_SET_BYTES > file_size )); then
  echo "WORKING_SET_BYTES exceeds DESIGN1_FILE size" >&2
  exit 2
fi
cold_required=$((REQUESTS * 65536 + 8192))
minimum_size=$((128 * 1024 * 1024))
if (( cold_required > minimum_size )); then
  minimum_size=${cold_required}
fi
effective_working_set=${WORKING_SET_BYTES:-${file_size}}
if (( effective_working_set < minimum_size )); then
  echo "Design 1 working set is too small: need at least ${minimum_size} bytes" >&2
  echo "for ${REQUESTS} cold requests" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
# Do not let JSON files from an older matrix contaminate the new summary.
rm -f "${RESULT_ROOT}"/design1_*.json \
      "${RESULT_ROOT}"/design1_*.json.tmp \
      "${RESULT_ROOT}"/design1_*.log \
      "${RESULT_ROOT}/raw_results.csv" \
      "${RESULT_ROOT}/summary.csv" \
      "${RESULT_ROOT}/experiment_metadata.txt" \
      "${RESULT_ROOT}/failed_runs.txt"
status=0
read -r -a policy_array <<<"${POLICIES}"
working_set_args=()
if [[ -n "${WORKING_SET_BYTES}" ]]; then
  working_set_args=(--working-set-bytes "${WORKING_SET_BYTES}")
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "benchmark_file=${DESIGN1_FILE}"
  echo "file_size_bytes=${file_size}"
  echo "file_allocated_bytes=${allocated_bytes}"
  echo "working_set_bytes=${effective_working_set}"
  echo "mem_total_bytes=$(( $(awk '/MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))"
  echo "page_cache_mode=${PAGE_CACHE_MODE}"
  echo "requests=${REQUESTS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "repeats=${REPEATS}"
  echo "cases=${CASES}"
  echo "policies=${POLICIES}"
  echo "order_seed=${ORDER_SEED}"
  echo "trace_seed=${TRACE_SEED}"
  echo "kvikio_nthreads=${KVIKIO_NTHREADS:-default}"
  git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
  uname -a | sed 's/^/uname=/'
  nvidia-smi -L 2>/dev/null || true
} >"${RESULT_ROOT}/experiment_metadata.txt"

for repeat in $(seq 1 "${REPEATS}"); do
  for case_name in ${CASES}; do
    policy_order=$("${PYTHON_BIN}" -c \
      'import random,sys; p=sys.argv[3:]; random.Random(f"{sys.argv[1]}:{sys.argv[2]}").shuffle(p); print(" ".join(p))' \
      "${ORDER_SEED}" "${repeat}:${case_name}" "${policy_array[@]}")
    read -r -a ordered_policies <<<"${policy_order}"
    execution_order=0
    for policy in "${ordered_policies[@]}"; do
      name="design1_${case_name}_p${policy}_r${repeat}"
      output="${RESULT_ROOT}/${name}.json"
      log="${RESULT_ROOT}/${name}.log"
      echo "Running ${case_name}, policy=${policy}, repeat=${repeat}, order=${execution_order}"
      if "${PYTHON_BIN}" -m kvikio.benchmarks.design1_policy \
        --file "${DESIGN1_FILE}" \
        --case "${case_name}" \
        --policy "${policy}" \
        --requests "${REQUESTS}" \
        --batch-size "${BATCH_SIZE}" \
        --page-cache-mode "${PAGE_CACHE_MODE}" \
        --repeat-id "${repeat}" \
        --execution-order "${execution_order}" \
        --order-seed "${ORDER_SEED}" \
        --trace-seed "$((TRACE_SEED + repeat))" \
        "${working_set_args[@]}" \
        --output "${output}.tmp" >"${log}" 2>&1; then
        mv "${output}.tmp" "${output}"
        echo "PASS: ${output}"
      else
        rc=$?
        rm -f "${output}.tmp"
        echo "${name},exit=${rc},log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
        status=1
      fi
      execution_order=$((execution_order + 1))
    done
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_design1_policy.py" \
  --result-root "${RESULT_ROOT}" || status=1
exit "${status}"
