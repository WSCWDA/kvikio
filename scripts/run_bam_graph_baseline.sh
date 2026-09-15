#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Run the requested BaM implementation/memory-allocation reference.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
BAM_BUILD_DIR="${BAM_BUILD_DIR:?Set BAM_BUILD_DIR to the BaM build directory}"
GRAPH_PREFIX="${GRAPH_PREFIX:?Set GRAPH_PREFIX to a BaM graph prefix}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bam-graph-baseline}"
REPEATS="${REPEATS:-5}"
GPU="${GPU:-0}"
BFS_SOURCE="${BFS_SOURCE:-1}"
BAM_THREADS="${BAM_THREADS:-128}"
BAM_PAGE_SIZE="${BAM_PAGE_SIZE:-4096}"
BAM_IMPL_TYPE="${BAM_IMPL_TYPE:-4}"
BAM_MEMALLOC="${BAM_MEMALLOC:-2}"

BFS_BIN="${BFS_BIN:-${BAM_BUILD_DIR}/bin/nvm-bfs-bench}"
PAGERANK_BIN="${PAGERANK_BIN:-${BAM_BUILD_DIR}/bin/nvm-pagerank-bench}"
for path in "${BFS_BIN}" "${PAGERANK_BIN}" "${GRAPH_PREFIX}.col" "${GRAPH_PREFIX}.dst"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${RESULT_ROOT}"

for repeat in $(seq 1 "${REPEATS}"); do
  # User-requested default: COALESCE_PC=4 and UVM_DIRECT=2.
  "${BFS_BIN}" \
    --input "${GRAPH_PREFIX}" --impl_type "${BAM_IMPL_TYPE}" --memalloc "${BAM_MEMALLOC}" \
    --src "${BFS_SOURCE}" --page_size "${BAM_PAGE_SIZE}" \
    --gpu "${GPU}" --threads "${BAM_THREADS}" \
    >"${RESULT_ROOT}/bam_bfs_i${BAM_IMPL_TYPE}_m${BAM_MEMALLOC}_r${repeat}.log" 2>&1
  "${PAGERANK_BIN}" \
    --input "${GRAPH_PREFIX}" --impl_type "${BAM_IMPL_TYPE}" --memalloc "${BAM_MEMALLOC}" \
    --page_size "${BAM_PAGE_SIZE}" --gpu "${GPU}" --threads "${BAM_THREADS}" \
    >"${RESULT_ROOT}/bam_pagerank_i${BAM_IMPL_TYPE}_m${BAM_MEMALLOC}_r${repeat}.log" 2>&1
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_bam_graph_baseline.py" \
  --result-root "${RESULT_ROOT}"
echo "BaM baseline logs: ${RESULT_ROOT}"
