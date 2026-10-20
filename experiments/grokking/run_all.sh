#!/usr/bin/env bash
# One-command, unattended, resumable pipeline for the grokking x topology study.
# Run from the repo root on macOS (native, not a VM):
#     bash experiments/grokking/run_all.sh            # uses all but 2 cores
#     WORKERS=6 bash experiments/grokking/run_all.sh  # or choose
# Safe to re-run after an interruption: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
RESULTS="$ROOT/results/grokking"
mkdir -p "$RESULTS"
LOG="$RESULTS/run_all.log"
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"
WORKERS="${WORKERS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# keep the Mac awake while this script runs
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then
  export CAFFEINATED=1
  exec caffeinate -dims bash "$0" "$@"
fi

log "start: $NCPU cpus, $WORKERS workers, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
if [ ! -x .venv/bin/python ]; then
  log "creating .venv"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  # torch 2.14.1 matches the version the test suite was developed and run with (macOS 14+ arm64 wheel on PyPI);
  # fall back to the newest compatible version on older macOS.
  .venv/bin/pip install --quiet "torch==2.14.1" || .venv/bin/pip install --quiet torch
  .venv/bin/pip install --quiet numpy matplotlib pytest
fi
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"
"$PY" -c "import torch; print('torch', torch.__version__)" | tee -a "$LOG"

log "stage 0: tests"
"$PY" -m pytest -q tests/test_grokking.py 2>&1 | tail -1 | tee -a "$LOG"

log "stage 1: per-arm LR tuning"
"$PY" experiments/grokking/run_sweep.py --grid experiments/grokking/grids/lr_tune.json \
  --out "$RESULTS/lr_tune" --workers "$WORKERS" 2>&1 | tee -a "$LOG"

log "stage 2: select LRs"
"$PY" experiments/grokking/select_lr.py --runs "$RESULTS/lr_tune" \
  --out "$RESULTS/main_tuned.json" 2>&1 | tee -a "$LOG"

log "stage 3: main study"
"$PY" experiments/grokking/run_sweep.py --grid "$RESULTS/main_tuned.json" \
  --out "$RESULTS/main" --workers "$WORKERS" 2>&1 | tee -a "$LOG"

log "stage 4: analysis"
"$PY" experiments/grokking/analyze.py --runs "$RESULTS/main" --out "$RESULTS/report" > /dev/null 2>&1 || true
"$PY" experiments/grokking/analyze.py --runs "$RESULTS/lr_tune" --out "$RESULTS/report_lr_tune" > /dev/null 2>&1 || true
log "done. report: $RESULTS/report/summary.md"
