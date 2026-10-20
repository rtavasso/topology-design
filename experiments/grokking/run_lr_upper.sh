#!/usr/bin/env bash
# Amendment A3 pipeline: upper-LR tuning -> per-arm selection -> 10-seed main study at the new lrs -> analysis.
# Run from the repo root on macOS:  bash experiments/grokking/run_lr_upper.sh   (resumable)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"; RESULTS="$ROOT/results/grokking"; mkdir -p "$RESULTS"; LOG="$RESULTS/run_lr_upper.log"
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"; WORKERS="${WORKERS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then export CAFFEINATED=1; exec caffeinate -dims bash "$0" "$@"; fi
PY="$ROOT/.venv/bin/python"; export PYTHONPATH="$ROOT/src"
log "start A3: $WORKERS workers, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
"$PY" -m pytest -q tests/test_grokking.py 2>&1 | tail -1 | tee -a "$LOG"
log "stage 1: upper-LR tuning"
"$PY" experiments/grokking/run_sweep.py --grid experiments/grokking/grids/lr_tune_upper.json \
  --out "$RESULTS/lr_tune_upper" --workers "$WORKERS" 2>&1 | tee -a "$LOG"
log "stage 2: select LRs"
"$PY" experiments/grokking/select_lr.py --runs "$RESULTS/lr_tune_upper" --out "$RESULTS/main_v2.json" \
  --tag main_v2 --no-shared --no-depth 2>&1 | tee -a "$LOG"
log "stage 3: 10-seed main study at the A3-tuned lrs"
"$PY" experiments/grokking/run_sweep.py --grid "$RESULTS/main_v2.json" --out "$RESULTS/main_v2" \
  --workers "$WORKERS" 2>&1 | tee -a "$LOG"
log "stage 4: analysis"
"$PY" experiments/grokking/analyze.py --runs "$RESULTS/main_v2" --out "$RESULTS/report_v2" > /dev/null 2>&1 || true
"$PY" experiments/grokking/analyze.py --runs "$RESULTS/lr_tune_upper" --out "$RESULTS/report_lr_tune_upper" > /dev/null 2>&1 || true
log "done. report: $RESULTS/report_v2/summary.md"
