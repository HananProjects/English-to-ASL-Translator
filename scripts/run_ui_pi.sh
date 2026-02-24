#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAMERA_INDEX="${ASL_CAMERA_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if ! command -v libcamerify >/dev/null 2>&1; then
  echo "Error: libcamerify not found. Install Raspberry Pi camera apps/tools first." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
ASL_CAMERA_INDEX="${CAMERA_INDEX}" libcamerify "${PYTHON_BIN}" -m ui.main
