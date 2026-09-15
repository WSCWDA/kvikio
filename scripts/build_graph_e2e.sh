#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${GRAPH_BUILD_DIR:-${REPO_ROOT}/build/graph-e2e}"
PREFIX="${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-}}"

cmake -S "${REPO_ROOT}/scripts/graph" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PREFIX}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_PARALLEL_LEVEL:-4}"
echo "Built ${BUILD_DIR}/groute_graph_e2e"
