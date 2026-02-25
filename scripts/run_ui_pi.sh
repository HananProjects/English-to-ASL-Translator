#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAMERA_INDEX="${ASL_CAMERA_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
# Optional camera tuning knobs (override when launching):
# ASL_CAMERA_WIDTH / ASL_CAMERA_HEIGHT / ASL_CAMERA_FPS
# ASL_PROCESS_WIDTH / ASL_PROCESS_HEIGHT / ASL_PROCESS_EVERY_N
# ASL_CAMERA_ZOOM (1.0 = no crop, 1.2 = mild zoom, 1.5+ strong zoom)
# ASL_CAMERA_BUFFER_SIZE (1 recommended for lower latency)
CAMERA_WIDTH="${ASL_CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${ASL_CAMERA_HEIGHT:-720}"
CAMERA_FPS="${ASL_CAMERA_FPS:-30}"
PROCESS_WIDTH="${ASL_PROCESS_WIDTH:-640}"
PROCESS_HEIGHT="${ASL_PROCESS_HEIGHT:-360}"
PROCESS_EVERY_N="${ASL_PROCESS_EVERY_N:-1}"
CAMERA_ZOOM="${ASL_CAMERA_ZOOM:-1.0}"
CAMERA_BUFFER_SIZE="${ASL_CAMERA_BUFFER_SIZE:-1}"
TOUCH_UI="${ASL_TOUCH_UI:-1}"
TOUCH_FULLSCREEN="${ASL_TOUCH_FULLSCREEN:-1}"

if ! command -v libcamerify >/dev/null 2>&1; then
  echo "Error: libcamerify not found. Install Raspberry Pi camera apps/tools first." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
ASL_CAMERA_INDEX="${CAMERA_INDEX}" \
ASL_CAMERA_WIDTH="${CAMERA_WIDTH}" \
ASL_CAMERA_HEIGHT="${CAMERA_HEIGHT}" \
ASL_CAMERA_FPS="${CAMERA_FPS}" \
ASL_PROCESS_WIDTH="${PROCESS_WIDTH}" \
ASL_PROCESS_HEIGHT="${PROCESS_HEIGHT}" \
ASL_PROCESS_EVERY_N="${PROCESS_EVERY_N}" \
ASL_CAMERA_ZOOM="${CAMERA_ZOOM}" \
ASL_CAMERA_BUFFER_SIZE="${CAMERA_BUFFER_SIZE}" \
ASL_TOUCH_UI="${TOUCH_UI}" \
ASL_TOUCH_FULLSCREEN="${TOUCH_FULLSCREEN}" \
libcamerify "${PYTHON_BIN}" -m ui.main
