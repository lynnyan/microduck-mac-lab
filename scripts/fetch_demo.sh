#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/models"
curl -fL --retry 2 \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448/alpha_walking.onnx \
  -o "$ROOT/models/alpha_walking.onnx.download"
"$ROOT/.venv/bin/python" - "$ROOT/models/alpha_walking.onnx.download" <<'PY'
import onnxruntime as ort
import sys
import hashlib
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
assert session.get_inputs()[0].shape == [1,61]
assert session.get_outputs()[0].shape == [1,14]
PY
mv "$ROOT/models/alpha_walking.onnx.download" "$ROOT/models/alpha_walking.onnx"
