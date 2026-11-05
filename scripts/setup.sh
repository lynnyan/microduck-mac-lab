#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/bootstrap_uv.sh"
bash "$ROOT/scripts/fetch_upstream.sh"
cd "$ROOT"
"$ROOT/.tools/uv" sync --locked --python 3.12
"$ROOT/.venv/bin/python" verify.py
"$ROOT/.venv/bin/python" test_runtime_contract.py
echo "Ready. Local training: bash scripts/train_macos.sh --task stand --timesteps 200000"
