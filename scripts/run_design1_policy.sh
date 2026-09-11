#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DESIGN1_FILE="${DESIGN1_FILE:?Set DESIGN1_FILE to an existing, non-sparse SSD file}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/groute-design1-policy}"
REQUESTS="${REQUESTS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPEATS="${REPEATS:-5}"
CASES="${CASES:-sequential_large random_cold_small random_hot_small adjacent_unaligned_small}"

for integer in "${REQUESTS}" "${BATCH_SIZE}" "${REPEATS}"; do
  if ! [[ "${integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REQUESTS, BATCH_SIZE, and REPEATS must be positive integers" >&2
    exit 2
  fi
done

if [[ ! -f "${DESIGN1_FILE}" ]]; then
  echo "Missing benchmark file: ${DESIGN1_FILE}" >&2
  exit 2
fi
if (( $(stat -c %s "${DESIGN1_FILE}") < 134217728 )); then
  echo "DESIGN1_FILE must be at least 128 MiB" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
rm -f "${RESULT_ROOT}/failed_runs.txt"
status=0

for repeat in $(seq 1 "${REPEATS}"); do
  for case_name in ${CASES}; do
    name="design1_${case_name}_r${repeat}"
    output="${RESULT_ROOT}/${name}.json"
    log="${RESULT_ROOT}/${name}.log"
    echo "Running ${case_name}, repeat=${repeat}"
    if "${PYTHON_BIN}" -m kvikio.benchmarks.design1_policy \
      --file "${DESIGN1_FILE}" \
      --case "${case_name}" \
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

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_design1_policy.py" \
  --result-root "${RESULT_ROOT}" || status=1
exit "${status}"
