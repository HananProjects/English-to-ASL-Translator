#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

# Camera + UI defaults for touchscreen presentation use.
export ASL_TOUCH_UI="${ASL_TOUCH_UI:-1}"
export ASL_TOUCH_FULLSCREEN="${ASL_TOUCH_FULLSCREEN:-1}"
export ASL_DEMO_MODE="${ASL_DEMO_MODE:-1}"
export ASL_DEMO_PHRASE_SNAP="${ASL_DEMO_PHRASE_SNAP:-0}"
export ASL_DEMO_VOCAB="${ASL_DEMO_VOCAB:-ME,GOOD,WE,GO,SCHOOL}"
export ASL_DEMO_TOKEN_REMAP="${ASL_DEMO_TOKEN_REMAP:-PLEASE:ME,NAME:ME,MY:ME,THANK_YOU:GOOD,ABOUT:ME,WHICH:ME}"
export ASL_STABLE_FRAMES="${ASL_STABLE_FRAMES:-3}"
export ASL_MIN_CONFIDENCE="${ASL_MIN_CONFIDENCE:-0.52}"
export ASL_COMMIT_MIN_CONFIDENCE="${ASL_COMMIT_MIN_CONFIDENCE:-0.60}"
export ASL_EMIT_COOLDOWN_FRAMES="${ASL_EMIT_COOLDOWN_FRAMES:-4}"
export ASL_CAMERA_WIDTH="${ASL_CAMERA_WIDTH:-1280}"
export ASL_CAMERA_HEIGHT="${ASL_CAMERA_HEIGHT:-720}"
export ASL_CAMERA_FPS="${ASL_CAMERA_FPS:-30}"
export ASL_PROCESS_WIDTH="${ASL_PROCESS_WIDTH:-640}"
export ASL_PROCESS_HEIGHT="${ASL_PROCESS_HEIGHT:-360}"
export ASL_PROCESS_EVERY_N="${ASL_PROCESS_EVERY_N:-1}"
export ASL_CAMERA_BUFFER_SIZE="${ASL_CAMERA_BUFFER_SIZE:-1}"
export ASL_CAMERA_ZOOM="${ASL_CAMERA_ZOOM:-1.0}"
export ASL_TTS_ALSA_DEVICE="${ASL_TTS_ALSA_DEVICE:-default:CARD=wm8960soundcard}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

cd "${REPO_ROOT}"

if command -v libcamerify >/dev/null 2>&1; then
  exec libcamerify "${PYTHON_BIN}" -m ui.main
fi

echo "[launcher] libcamerify not found; running UI directly."
exec "${PYTHON_BIN}" -m ui.main
