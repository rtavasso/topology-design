#!/usr/bin/env bash
# Study 4 pipeline (what each path does). From the repo root on macOS:
#     bash experiments/paths/run.sh        (resumable; ~2-3 h on 10 cores)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"; RESULTS="$ROOT/results/paths"; mkdir -p "$RESULTS"; LOG="$RESULTS/run.log"
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"; WORKERS="${WORKERS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then export CAFFEINATED=1; exec caffeinate -dims bash "$0" "$@"; fi
PY="$ROOT/.venv/bin/python"; export PYTHONPATH="$ROOT/src"
log "start study 4: $WORKERS workers, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
"$PY" -m pytest -q tests/test_grokking.py tests/test_paths.py 2>&1 | tail -1 | tee -a "$LOG"
log "stage 0: per-graph lr selection (90 runs)"
"$PY" experiments/grokking/run_sweep.py --grid experiments/paths/grids/stage0.json --out "$RESULTS/stage0" \
  --workers "$WORKERS" 2>&1 | tee -a "$LOG"
"$PY" experiments/paths/select_lr.py --runs "$RESULTS/stage0" --out "$RESULTS" 2>&1 | tee -a "$LOG"
log "stage 1: main runs (60 runs)"
"$PY" experiments/grokking/run_sweep.py --grid "$RESULTS/stage1_grid.json" --out "$RESULTS/stage1" \
  --workers "$WORKERS" 2>&1 | tee -a "$LOG"
"$PY" experiments/paths/analyze.py --runs "$RESULTS/stage1" --selection "$RESULTS/selection.json" \
  --out "$RESULTS/report" 2>&1 | tail -40 | tee -a "$LOG"
log "done. report: $RESULTS/report/summary.md (compact data: $RESULTS/report/compact.json)"
