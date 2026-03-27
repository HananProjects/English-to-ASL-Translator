#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# macOS does not use libcamerify. Prevent auto re-exec path.
export ASL_DISABLE_LIBCAMERIFY="${ASL_DISABLE_LIBCAMERIFY:-1}"

# Desktop defaults.
export ASL_TOUCH_UI="${ASL_TOUCH_UI:-0}"
export ASL_TOUCH_FULLSCREEN="${ASL_TOUCH_FULLSCREEN:-0}"

# Keep live recognition honest by default (no demo token remaps).
export ASL_DEMO_MODE="${ASL_DEMO_MODE:-0}"

# Camera defaults tuned for stable desktop throughput.
export ASL_CAMERA_INDEX="${ASL_CAMERA_INDEX:-0}"
export ASL_CAMERA_PROBE_COUNT="${ASL_CAMERA_PROBE_COUNT:-6}"
export ASL_CAMERA_WIDTH="${ASL_CAMERA_WIDTH:-1280}"
export ASL_CAMERA_HEIGHT="${ASL_CAMERA_HEIGHT:-720}"
export ASL_CAMERA_FPS="${ASL_CAMERA_FPS:-30}"
export ASL_CAMERA_BUFFER_SIZE="${ASL_CAMERA_BUFFER_SIZE:-1}"
export ASL_CAMERA_ZOOM="${ASL_CAMERA_ZOOM:-1.0}"
export ASL_CAMERA_FAST_MODE="${ASL_CAMERA_FAST_MODE:-0}"
export ASL_PROCESS_WIDTH="${ASL_PROCESS_WIDTH:-640}"
export ASL_PROCESS_HEIGHT="${ASL_PROCESS_HEIGHT:-360}"
export ASL_PROCESS_EVERY_N="${ASL_PROCESS_EVERY_N:-1}"

# Smoothing / hold balance for steadier tokens without excessive lag.
export ASL_POSE_SMOOTHING="${ASL_POSE_SMOOTHING:-0.45}"
export ASL_HAND_HOLD_FRAMES="${ASL_HAND_HOLD_FRAMES:-2}"
export ASL_BODY_HOLD_FRAMES="${ASL_BODY_HOLD_FRAMES:-1}"

# Recognition commit gates: balanced for live signing phrases.
export ASL_SINGLE_SIGN_CAPTURE="${ASL_SINGLE_SIGN_CAPTURE:-0}"
export ASL_SUPPRESS_IDLE_HAND="${ASL_SUPPRESS_IDLE_HAND:-0}"
export ASL_MODEL_REJECT_CONFIDENCE="${ASL_MODEL_REJECT_CONFIDENCE:-0.30}"
export ASL_MIN_CONFIDENCE="${ASL_MIN_CONFIDENCE:-0.55}"
export ASL_COMMIT_MIN_CONFIDENCE="${ASL_COMMIT_MIN_CONFIDENCE:-0.65}"
export ASL_STABLE_FRAMES="${ASL_STABLE_FRAMES:-4}"
export ASL_EMIT_COOLDOWN_FRAMES="${ASL_EMIT_COOLDOWN_FRAMES:-6}"
export ASL_PAUSE_FRAMES="${ASL_PAUSE_FRAMES:-10}"
export ASL_TRANSITION_MOTION_THRESHOLD="${ASL_TRANSITION_MOTION_THRESHOLD:-0.024}"
export ASL_COMMIT_MOTION_THRESHOLD="${ASL_COMMIT_MOTION_THRESHOLD:-0.016}"
export ASL_MOTION_SETTLE_FRAMES="${ASL_MOTION_SETTLE_FRAMES:-2}"
export ASL_AUTO_STOP_NO_HAND_FRAMES="${ASL_AUTO_STOP_NO_HAND_FRAMES:-8}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m ui.main
