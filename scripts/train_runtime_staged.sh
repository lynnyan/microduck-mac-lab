#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-outputs/runtime_staged_$(date +%Y%m%d_%H%M%S)}"
TASK="${MICRODUCK_TASK:-stand}"
ENVS="${MICRODUCK_ENVS:-4}"
# Explicit stages retain each model + its matching normalizer for evaluation.
.venv/bin/python test_runtime_contract.py
bash scripts/train_runtime.sh --task "$TASK" --envs "$ENVS" --dr-profile nominal \
  --timesteps "${MICRODUCK_NOMINAL_STEPS:-200000}" --output "$OUTPUT/nominal"
bash scripts/train_runtime.sh --task "$TASK" --envs "$ENVS" --dr-profile sensors \
  --resume "$OUTPUT/nominal" --timesteps "${MICRODUCK_SENSOR_STEPS:-400000}" --output "$OUTPUT/sensors"
bash scripts/train_runtime.sh --task "$TASK" --envs "$ENVS" --dr-profile dynamics \
  --resume "$OUTPUT/sensors" --timesteps "${MICRODUCK_DYNAMICS_STEPS:-400000}" --output "$OUTPUT/dynamics"
bash scripts/export_runtime.sh --run "$OUTPUT/dynamics" --output "$OUTPUT/export"
echo "Local bundle: $OUTPUT/export. Compare all stages before selecting a candidate."
