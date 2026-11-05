#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$ROOT/.tools/uv"
if [[ -x "$UV" ]] && "$UV" --version >/dev/null 2>&1; then
  exit 0
fi
mkdir -p "$ROOT/.tools"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) UV_TARGET=aarch64-apple-darwin ;;
  Linux-x86_64) UV_TARGET=x86_64-unknown-linux-gnu ;;
  Linux-aarch64) UV_TARGET=aarch64-unknown-linux-gnu ;;
  *) echo "Unsupported platform; install uv from https://docs.astral.sh/uv/" >&2; exit 1 ;;
esac
curl -fL --retry 2 "https://github.com/astral-sh/uv/releases/download/0.13.7/uv-${UV_TARGET}.tar.gz" \
  -o "$ROOT/.tools/uv.tar.gz"
tar -xzf "$ROOT/.tools/uv.tar.gz" -C "$ROOT/.tools" --strip-components=1
"$UV" --version
