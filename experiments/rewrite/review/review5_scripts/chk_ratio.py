"""Q4c: the ladder's matched ratio (sizes 6-7 over size <= 5, (world, k, binding) cells, reweighted to eval7), own code."""
import json
from eng import *
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
E5 = {w: read("eval5", w) for w in TRAIN}
E7 = {w: read("eval7", w) for w in TRAIN}
def ratio(ws, d):
    num = den = n = 0.0; unmatched = 0
    for w in ws:
        c5, c7 = {}, {}
        for r in E5[w]:
            if r["k"] <= 4: c5.setdefault((r["k"], bool(r["binding"])), []).append(bool(r["ladder"][str(d)]))
        for r in E7[w]:
            c7.setdefault((r["k"], bool(r["binding"])), []).append(bool(r["ladder"][str(d)]))
        for key, v7 in c7.items():
            if key not in c5: unmatched += len(v7); continue
            num += sum(v7); den += len(v7) * sum(c5[key]) / len(c5[key]); n += len(v7)
    return num / den, den / n, num / n, int(n), unmatched
for name, ws in (("training worlds", TRAIN), ("single-axiom", TRAIN[:4]), ("training pairs", TRAIN[4:])):
    for d in (1, 2, 3):
        R, base5, rate7, n, um = ratio(ws, d)
        print(f"{name:16s} d={d} ratio {R:.3f} (derived {D['p3_ladder_matched_ratio'][name]['d'+str(d)]:.3f})  size<=5 base (reweighted) {base5:.3f}  eval7 rate {rate7:.3f}  n {n} unmatched {um}")
print("eval7 k max per world", {w: max(r["k"] for r in E7[w]) for w in TRAIN}, "caps", {r["cap"] for w in TRAIN for r in E7[w]})
# per-world ratios at d=3 (how uneven is rho?)
for w in TRAIN:
    print(w, [round(ratio([w], d)[0], 3) for d in (1, 2, 3)], [round(ratio([w], d)[1], 3) for d in (1, 2, 3)])
