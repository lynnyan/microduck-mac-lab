#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REV=bae39f5900b318f7e87fc485c30a31213cfa14eb
if [[ ! -d "$ROOT/upstream/.git" ]]; then
  git clone --no-checkout --filter=blob:none https://github.com/pollen-robotics/microduck_rl.git "$ROOT/upstream"
  git -C "$ROOT/upstream" checkout "$REV"
fi
if [[ "$(git -C "$ROOT/upstream" rev-parse HEAD)" != "$REV" ]]; then
  echo "upstream revision differs; preserve your changes and select $REV manually." >&2
  exit 1
fi
if [[ -n "$(git -C "$ROOT/upstream" status --porcelain --untracked-files=no)" ]]; then
  echo "upstream has tracked changes; review them before running this pinned recipe." >&2
  exit 1
fi
