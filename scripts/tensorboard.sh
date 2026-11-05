#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/tensorboard" --logdir "$ROOT/outputs" --host 127.0.0.1 --port 6006 "$@"
