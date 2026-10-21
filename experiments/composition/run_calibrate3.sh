#!/usr/bin/env bash
# Study 3 calibration round 3 (16 runs, ~35 min on 10 cores). From the repo root on macOS, after Study 2 finishes:
#     bash experiments/composition/run_calibrate3.sh     (resumable)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"; RESULTS="$ROOT/results/composition"; mkdir -p "$RESULTS"; LOG="$RESULTS/calibrate3.log"
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"; WORKERS="${WORKERS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}"
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then export CAFFEINATED=1; exec caffeinate -dims bash "$0" "$@"; fi
PY="$ROOT/.venv/bin/python"; export PYTHONPATH="$ROOT/src"
"$PY" -m pytest -q tests/test_composition.py 2>&1 | tail -1 | tee -a "$LOG"
"$PY" experiments/grokking/run_sweep.py --module composition --grid experiments/composition/grids/calibrate3.json \
  --out "$RESULTS/calibrate3" --workers "$WORKERS" 2>&1 | tee -a "$LOG"
echo "done: $RESULTS/calibrate3" | tee -a "$LOG"
