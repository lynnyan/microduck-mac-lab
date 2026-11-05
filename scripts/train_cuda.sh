#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
  echo "This script needs a Linux CUDA host. Use train_macos.sh for the local CPU experiment." >&2
  exit 2
fi
if [[ ! -x "$ROOT/upstream/.venv/bin/python" ]]; then
  echo "Run bash scripts/setup_cuda.sh on the CUDA host first." >&2
  exit 1
fi
cd "$ROOT/upstream"
TASK="${MICRODUCK_TASK:-Mjlab-Velocity-Flat-MicroDuck}"
ENVS="${MICRODUCK_ENVS:-4096}"
ITERS="${MICRODUCK_ITERS:-5000}"
# Always validate the actual task first, before an expensive full run.
"$ROOT/.tools/uv" run --locked train "$TASK" --env.scene.num-envs 64 \
  --agent.max-iterations 5 --agent.logger tensorboard --agent.run-name cuda_smoke
exec "$ROOT/.tools/uv" run --locked train "$TASK" --env.scene.num-envs "$ENVS" \
  --agent.max-iterations "$ITERS" --agent.logger tensorboard \
  --agent.run-name cuda_baseline "$@"
