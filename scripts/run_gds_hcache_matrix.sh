#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -u
set -o pipefail

if [[ -z "${HCACHE_BENCH_FILE:-}" ]]; then
  echo "HCACHE_BENCH_FILE is required" >&2
  exit 2
fi

REQUESTS="${REQUESTS:-100000}"
REPEATS="${REPEATS:-5}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/kvikio-hcache-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

IO_SIZES=(4096 16384 65536)
# HCache line=16KiB is intentionally excluded from the formal matrix.
# The current MVP was validated from 64KiB cache lines upward.
LINE_SIZES=(65536 262144)
CAPS=(16777216 33554432 67108864 134217728)
HOT_BYTES="${HOT_BYTES:-67108864}"

mkdir -p "${RESULT_ROOT}"

drop_caches() {
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches
  fi
}

run_case() {
  local name="$1"
  local path="$2"
  local io_size="$3"
  local line_size="$4"
  local cache_bytes="$5"
  local repeat="$6"
  local json_path="${RESULT_ROOT}/${name}_r${repeat}.json"
  local log_path="${RESULT_ROOT}/${name}_r${repeat}.log"
  local tmp_json="${json_path}.tmp"

  echo "Running: ${name}, repeat=${repeat}"
  drop_caches

  KVIKIO_HOST_CACHE="$([[ "${path}" == "hcache" ]] && echo ON || echo OFF)" \
  KVIKIO_HOST_CACHE_CAPACITY="${cache_bytes}" \
  KVIKIO_HOST_CACHE_LINE_SIZE="${line_size}" \
  KVIKIO_HOST_CACHE_MAX_IO_SIZE="${io_size}" \
  "${PYTHON_BIN}" -m kvikio.benchmarks.gds_hcache \
    --file "${HCACHE_BENCH_FILE}" \
    --path "${path}" \
    --io-size "${io_size}" \
    --line-size "${line_size}" \
    --host-max "${io_size}" \
    --cache-bytes "${cache_bytes}" \
    --hot-bytes "${HOT_BYTES}" \
    --requests "${REQUESTS}" \
    --seed "${repeat}" \
    > "${tmp_json}" 2> "${log_path}"

  local status=$?
  if [[ ${status} -eq 0 ]]; then
    mv "${tmp_json}" "${json_path}"
    echo "PASS: ${json_path}"
  else
    rm -f "${tmp_json}"
    echo "FAIL: ${name}, repeat=${repeat}, exit=${status}" >&2
    echo "See: ${log_path}" >&2
  fi
  return "${status}"
}

status=0
for repeat in $(seq 1 "${REPEATS}"); do
  for io_size in "${IO_SIZES[@]}"; do
    run_case "gds_io${io_size}" "gds" "${io_size}" 65536 67108864 "${repeat}" || status=1
    run_case "posix_io${io_size}" "posix" "${io_size}" 65536 67108864 "${repeat}" || status=1
  done

  for io_size in "${IO_SIZES[@]}"; do
    for line_size in "${LINE_SIZES[@]}"; do
      if (( io_size > line_size )); then
        continue
      fi
      for cache_bytes in "${CAPS[@]}"; do
        if (( cache_bytes < line_size )); then
          continue
        fi
        cap_mib=$((cache_bytes / 1024 / 1024))
        run_case \
          "hcache_io${io_size}_line${line_size}_cap${cap_mib}m" \
          "hcache" \
          "${io_size}" \
          "${line_size}" \
          "${cache_bytes}" \
          "${repeat}" || status=1
      done
    done
  done
done

exit "${status}"
