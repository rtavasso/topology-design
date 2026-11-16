"""Stage 0 of experiments/theory/worlds/DESIGN.md: does the monolith fail on held-out worlds? EXPLORATORY.
Predictions P0a-P0d are in DESIGN.md section 7, committed before this was run."""
import json
import os
import statistics
import sys
from worlds import run

HERE = os.path.dirname(os.path.abspath(__file__))
med = statistics.median
RUNS = [(t, s) for t in (10, 11, 12) for s in (0, 1)]


def cell(v):
    return f"{med(v):.1e} [{min(v):.1e}, {max(v):.1e}]"


if __name__ == "__main__":
    mode, task = sys.argv[1], sys.argv[2]
    if mode == "rates":
        print(f"| task | rate | seen | held out | held out / seen |\n|---|---|---|---|---|", flush=True)
        best = {}
        for lr in [float(x) for x in sys.argv[3:]] or (0.003, 0.01, 0.03, 0.1):
            rs = [run(dict(task=task, cover=9, lr=lr, tseed=t), 100) for t in (10, 11, 12)]
            best[lr] = med([r["seen"] for r in rs])
            print(f"| {task} | {lr:g} | {cell([r['seen'] for r in rs])} | {cell([r['held'] for r in rs])} | {med([r['held'] / r['seen'] for r in rs]):.1f} |", flush=True)
        json.dump({task: min(best, key=best.get)}, open(os.path.join(HERE, f"stage0_lr_{task}.json"), "w"))
    else:
        lr = json.load(open(os.path.join(HERE, f"stage0_lr_{task}.json")))[task]
        store = {}
        print(f"| task | coverage | width | seen | held out | held out / seen (median of per-run ratios) | reference: blend | reference: one |\n|---|---|---|---|---|---|---|---|", flush=True)
        for cover, W in [(7, 40), (9, 40), (11, 40), (13, 40), (15, 40), (7, 80)]:
            rs = [run(dict(task=task, cover=cover, lr=lr, tseed=t, W=W), s) for t, s in RUNS]
            store[f"{cover}_{W}"] = rs
            if cover < 15:
                ratio = f"{med([r['held'] / r['seen'] for r in rs]):.1f}"
                refs = f"{med([r['ref']['blend'] for r in rs]):.2f} | {med([r['ref']['one'] for r in rs]):.2f}"
                held = cell([r["held"] for r in rs])
            else:
                ratio, refs, held = "-", "- | -", "-"
            print(f"| {task} | {cover} | {W} | {cell([r['seen'] for r in rs])} | {held} | {ratio} | {refs} |", flush=True)
            json.dump(store, open(os.path.join(HERE, f"stage0_{task}.json"), "w"))
