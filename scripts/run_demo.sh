#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -eq 0 ]]; then
  set -- all
fi
exec "$PYTHON_BIN" scripts/demo.py "$@"
