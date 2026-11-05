#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -eq 0 ]]; then
  set -- --onnx models/alpha_walking.onnx --command 0.3 0 0 --episodes 1 --seconds 30
fi
# The driving window owns GLFW on Python's main thread; the standard passive
# viewer still needs mjpython on macOS.
for arg in "$@"; do
  if [[ "$arg" == "--keyboard" ]]; then
    exec .venv/bin/python play.py "$@"
  fi
done
if [[ "$(uname -s)" == Darwin ]]; then
  exec .venv/bin/mjpython play.py "$@"
else
  exec .venv/bin/python play.py "$@"
fi
