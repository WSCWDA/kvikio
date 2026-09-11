#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GUSTANN_DIR="${GUSTANN_DIR:?Set GUSTANN_DIR to a clean thustorage/GustANN checkout}"
BUILD_DIR="${GUSTANN_BUILD_DIR:-${GUSTANN_DIR}/build-groute}"

if [[ ! -d "${GUSTANN_DIR}/.git" ]]; then
  echo "GUSTANN_DIR is not a Git checkout: ${GUSTANN_DIR}" >&2
  exit 2
fi

patch_file="${REPO_ROOT}/scripts/diskann/gustann-groute.patch"
if git -C "${GUSTANN_DIR}" apply --reverse --check "${patch_file}" 2>/dev/null; then
  echo "GustANN G-Route patch already applied"
elif git -C "${GUSTANN_DIR}" apply --check "${patch_file}"; then
  git -C "${GUSTANN_DIR}" apply "${patch_file}"
else
  echo "Patch does not apply cleanly; use the GustANN main revision documented by this repository" >&2
  exit 2
fi

install -m 0644 \
  "${REPO_ROOT}/scripts/diskann/groute_loader.cpp" \
  "${GUSTANN_DIR}/src/io/groute.cpp"

cmake -S "${GUSTANN_DIR}" -B "${BUILD_DIR}" \
  -DGUSTANN_USE_AIO=ON \
  -DGUSTANN_USE_GROUTE=ON \
  -DGUSTANN_USE_BAM=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-/usr/local}}"
cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS:-$(nproc)}"
echo "Built ${BUILD_DIR}/bin/search_disk_hybrid"
