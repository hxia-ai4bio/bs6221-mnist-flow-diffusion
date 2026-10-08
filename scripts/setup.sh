#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -s -m pip install -r requirements.txt
"$PYTHON_BIN" -s -m pip check
"$PYTHON_BIN" scripts/download_weights.py
"$PYTHON_BIN" scripts/restore_shared_data.py
