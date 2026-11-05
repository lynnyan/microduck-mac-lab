#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
  echo "Official mjlab/MuJoCo Warp training requires a Linux NVIDIA CUDA host." >&2
  echo "On this Mac use scripts/setup.sh and scripts/train_macos.sh." >&2
  exit 2
fi
bash "$ROOT/scripts/bootstrap_uv.sh"
bash "$ROOT/scripts/fetch_upstream.sh"
cd "$ROOT/upstream"
export UV_HTTP_TIMEOUT=600
"$ROOT/.tools/uv" sync --locked --python 3.12
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA Torch unavailable"; print(torch.cuda.get_device_name(0))'
"$ROOT/.tools/uv" run --locked list-envs
