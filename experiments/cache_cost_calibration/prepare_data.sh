#!/usr/bin/env bash
set -euo pipefail

FILE=${1:-/mnt/gds/groute-cache-cost.bin}
SIZE_GIB=${2:-2}
RESULT_ROOT=/mnt/gds/results
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR="${RESULT_ROOT}/groute_cache_cost_data_${STAMP}_$$"

if (( SIZE_GIB < 1 )); then
  echo "SIZE_GIB must be at least 1" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
mkdir -p "$(dirname "${FILE}")"

# This is an allocated file, not a sparse truncate. oflag=direct keeps the
# preparation write from filling the Linux page cache.
dd if=/dev/zero of="${FILE}" bs=16M count=$((SIZE_GIB * 64)) \
  oflag=direct conv=fsync status=progress \
  >"${RESULT_DIR}/dd.stdout" 2>"${RESULT_DIR}/dd.stderr"

FILE_BYTES=$(stat -c %s "${FILE}")
ALLOCATED_BYTES=$(du -B1 "${FILE}" | awk '{print $1}')
MOUNT=$(findmnt -T "${FILE}" -no TARGET,FSTYPE,SOURCE,OPTIONS)

python - "${RESULT_DIR}/data.json" "${FILE}" "${FILE_BYTES}" \
  "${ALLOCATED_BYTES}" "${SIZE_GIB}" "${MOUNT}" <<'PY'
import json
import sys
from pathlib import Path

output, file, file_bytes, allocated_bytes, size_gib, mount = sys.argv[1:]
record = {
    "kind": "cache_cost_test_data",
    "file": file,
    "requested_gib": int(size_gib),
    "file_bytes": int(file_bytes),
    "allocated_bytes": int(allocated_bytes),
    "fully_allocated": int(allocated_bytes) >= int(file_bytes),
    "mount": mount,
}
Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY

echo "Data file: ${FILE}"
echo "Preparation metadata saved to ${RESULT_DIR}"
