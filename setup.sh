#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python 3.11 .venv --allow-existing
uv pip install --python .venv/bin/python -e '.[test]' zstandard==0.25.0
.venv/bin/python scripts/install_macos_runtime.py
.venv/bin/python -m pickparts_agent.app --smoke
