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

for integer in "${REQUESTS}" "${BATCH_SIZE}" "${REPEATS}"; do
  if ! [[ "${integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REQUESTS, BATCH_SIZE, and REPEATS must be positive integers" >&2
    exit 2
  fi
done

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
cold_required=$((REQUESTS * 65536 + 8192))
minimum_size=$((128 * 1024 * 1024))
if (( cold_required > minimum_size )); then
  minimum_size=${cold_required}
fi
if (( file_size < minimum_size )); then
  echo "DESIGN1_FILE is too small: need at least ${minimum_size} bytes" >&2
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
      "${RESULT_ROOT}/failed_runs.txt"
status=0

for repeat in $(seq 1 "${REPEATS}"); do
  for case_name in ${CASES}; do
    for policy in ${POLICIES}; do
      name="design1_${case_name}_p${policy}_r${repeat}"
      output="${RESULT_ROOT}/${name}.json"
      log="${RESULT_ROOT}/${name}.log"
      echo "Running ${case_name}, policy=${policy}, repeat=${repeat}"
      if "${PYTHON_BIN}" -m kvikio.benchmarks.design1_policy \
        --file "${DESIGN1_FILE}" \
        --case "${case_name}" \
        --policy "${policy}" \
        --requests "${REQUESTS}" \
        --batch-size "${BATCH_SIZE}" \
        --output "${output}.tmp" >"${log}" 2>&1; then
        mv "${output}.tmp" "${output}"
        echo "PASS: ${output}"
      else
        rc=$?
        rm -f "${output}.tmp"
        echo "${name},exit=${rc},log=${log}" | tee -a "${RESULT_ROOT}/failed_runs.txt"
        status=1
      fi
    done
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_design1_policy.py" \
  --result-root "${RESULT_ROOT}" || status=1
exit "${status}"
