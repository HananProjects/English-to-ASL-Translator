#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LAUNCHER="/home/capstone/capstone/English-to-ASL-Translator/English-to-ASL-Translator/scripts/launch_touchscreen_assisted.sh"

if [[ -x "${SCRIPT_DIR}/launch_touchscreen_assisted.sh" ]]; then
  exec "${SCRIPT_DIR}/launch_touchscreen_assisted.sh"
fi

exec "${REPO_LAUNCHER}"
