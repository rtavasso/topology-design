#!/usr/bin/env bash
# Study 2 pipeline (stability ceiling vs composition depth). From the repo root on macOS:
#     bash experiments/stability/run.sh        (resumable)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"; RESULTS="$ROOT/results/stability"; mkdir -p "$RESULTS"; LOG="$RESULTS/run.log"
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"; WORKERS="${WORKERS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then export CAFFEINATED=1; exec caffeinate -dims bash "$0" "$@"; fi
PY="$ROOT/.venv/bin/python"; export PYTHONPATH="$ROOT/src"
log "start study 2: $WORKERS workers, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
"$PY" -m pytest -q tests/test_grokking.py 2>&1 | tail -1 | tee -a "$LOG"
"$PY" experiments/grokking/run_sweep.py --grid experiments/stability/grid.json --out "$RESULTS/runs" \
  --workers "$WORKERS" 2>&1 | tee -a "$LOG"
"$PY" experiments/stability/analyze.py --runs "$RESULTS/runs" --out "$RESULTS/report" 2>&1 | tail -30 | tee -a "$LOG"
log "done. report: $RESULTS/report/summary.md"
