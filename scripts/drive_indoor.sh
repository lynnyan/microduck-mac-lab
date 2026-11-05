#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ ! -f scenes/indoor_lab.xml ]]; then
  .venv/bin/python scripts/build_indoor_scene.py
fi
bash scripts/view.sh --keyboard --scene scenes/indoor_lab.xml --seconds 600 --episodes 100 "$@"
