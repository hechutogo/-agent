#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash setup.sh first." >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  set -- --demo A B --view
fi
exec .venv/bin/python -m pickparts_agent.app "$@"
