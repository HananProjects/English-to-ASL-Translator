#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${ASL_MODEL_PATH:-${REPO_ROOT}/models/asl_landmark_classifier_v2.npz}"
LABELS_FILE="${ASL_LABELS_FILE:-${REPO_ROOT}/data/labels_100.txt}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Error: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "[tune] model=${MODEL_PATH}"
echo "[tune] labels=${LABELS_FILE}"

"${PYTHON_BIN}" scripts/eval_asl_sequence_model.py \
  --model "${MODEL_PATH}" \
  --labels-file "${LABELS_FILE}" \
  --reject-below-confidence 0.40 \
  --report data/eval/asl_eval_val_r040.json

"${PYTHON_BIN}" scripts/eval_asl_sequence_model.py \
  --model "${MODEL_PATH}" \
  --labels-file "${LABELS_FILE}" \
  --reject-below-confidence 0.55 \
  --report data/eval/asl_eval_val_r055.json

"${PYTHON_BIN}" scripts/eval_asl_sequence_model.py \
  --model "${MODEL_PATH}" \
  --labels-file "${LABELS_FILE}" \
  --reject-below-confidence 0.65 \
  --report data/eval/asl_eval_val_r065.json

echo "[tune] complete"
