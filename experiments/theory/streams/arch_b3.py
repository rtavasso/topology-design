"""Round B3: how short the dense start can be. EXPLORATORY.

Round B2: a dense start of 10 % of training followed by top-1 solved 5/5 with four streams (predicted at most 3/5), and
a dense first half solved 5/5 with eight streams. Softmax scores, register reg0.

Predictions, written before the run (15 Nov 2026):
  F1  eight streams, dense 10 % then top-1    solved 5/5.
  F2  eight streams, dense 3 % then top-1     solved at most 2/5 (90 steps do not separate eight streams).
  F3  four streams, dense 3 % then top-1      solved at most 3/5.
  F4  eight streams, dense until the gate's mean top probability passes 0.8, then top-1   solved 5/5, switching before
      20 % of training.
  F5  four streams, the same rule             solved 5/5, switching before 20 % of training.
"""
import sys
from loopmoe import run

R = dict(progress="reg0", score="softmax", topk=1)
BIG = dict(R, K=8, P=5, ntrain=80, B=8)
ROWS = [("F1 8 streams: dense 10 % then top-1", dict(BIG, anneal=0.1)),
        ("F2 8 streams: dense 3 % then top-1", dict(BIG, anneal=0.03)),
        ("F3 4 streams: dense 3 % then top-1", dict(R, anneal=0.03)),
        ("F4 8 streams: dense until confident, then top-1", dict(BIG, auto=0.8)),
        ("F5 4 streams: dense until confident, then top-1", dict(R, auto=0.8))]
if __name__ == "__main__":
    med = lambda v: sorted(v)[len(v) // 2]
    print("| variant | training error | held-out error | solved (<1e-2) | purity | dense steps of 3,000 |")
    print("|---|---|---|---|---|---|", flush=True)
    for i in sys.argv[1:]:
        label, over = ROWS[int(i)]
        rs = [run(over, s) for s in range(5)]
        te = [r["test"] for r in rs]
        sw = [r["switch"] for r in rs if r["switch"] is not None]
        dense = f"{med(sw)} [{min(sw)}, {max(sw)}]" if sw else f"{int(over['anneal'] * 3000)}"
        print(f"| {label} | {med([r['train'] for r in rs]):.1e} | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/5 | "
              f"{med([r['purity'] for r in rs]):.2f} | {dense} |", flush=True)
