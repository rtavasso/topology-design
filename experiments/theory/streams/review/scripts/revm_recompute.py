"""Recompute every Round M figure from the per-seed JSON files (no training)."""
import json, statistics as st, sys
H = "/home/claude/topology-design/experiments/theory/streams/"
d = {}
for f in ("arch_m_results_a.json", "arch_m_results_b.json"):
    x = json.load(open(H + f)); print(f, list(x.keys())); d.update(x)
med = st.median
CAND = {"M3": "M1", "M3a": "M1", "M3b": "M1", "M16": "M1", "M4": "M1", "M4a": "M1", "M7": "M1",
        "M9": "M8", "M17": "M8", "M10": "M8", "M13": "M12", "M13b": "M12", "M15": "M12"}
print("row n nparams | train4 med [min,max] | te4 med [min max] | te4/train4 | te8/te4 med [min,max] n>10 | x2/te4 | c1;c2 | addr | switch | solved te4,te8,x2")
for k, s in d.items():
    f = lambda n: [r[n] for r in s]
    r84 = [r["te8"] / r["te4"] for r in s]
    print(f"{k:5s} {len(s)} {s[0]['nparams']} | {med(f('train4')):.2e} [{min(f('train4')):.1e},{max(f('train4')):.1e}] | {med(f('te4')):.2e} [{min(f('te4')):.1e},{max(f('te4')):.1e}] | "
          f"{med([r['te4']/r['train4'] for r in s]):.3f} | {med(r84):.3f} [{min(r84):.2f},{max(r84):.1f}] n>10={sum(x>10 for x in r84)} | {med([r['x2']/r['te4'] for r in s]):.2f} | "
          f"{med(f('c1')):.1e};{med(f('c2')):.1e} | {med(f('addr')):.2f} min {min(f('addr')):.2f} | sw {sorted(x for x in f('switch') if x is not None)} | "
          f"{sum(x<1e-2 for x in f('te4'))},{sum(x<1e-2 for x in f('te8'))},{sum(x<1e-2 for x in f('x2'))}")
print()
print("paired: row | te4 ratio med [min,max] cand better | te8 ratio | x2 ratio | ratio of medians | geo-mean ratio | te6 ratio")
import math
for k, c in CAND.items():
    rr = lambda n: [a[n] / b[n] for a, b in zip(d[k], d[c])]
    r4 = rr("te4")
    gm = math.exp(sum(math.log(x) for x in r4) / len(r4))
    print(f"{k:5s} | {med(r4):.2f} [{min(r4):.2f},{max(r4):.1f}] {sum(x>1 for x in r4)}/10 | {med(rr('te8')):.2f} | {med(rr('x2')):.2f} | {med([r['te4'] for r in d[k]])/med([r['te4'] for r in d[c]]):.2f} | {gm:.2f} | train ratio {med(rr('train4')):.2f}")
print()
for k in ("M1", "M4", "M8", "M10", "M4a", "M12"):
    print(k, "te4", " ".join(f"{r['te4']:.1e}" for r in d[k]))
    print(k, "te8/te4", " ".join(f"{r['te8']/r['te4']:.1f}" for r in d[k]))
print("all rows te8/te4 per seed >10:")
for k, s in d.items():
    big = [(i, round(r["te8"] / r["te4"], 1)) for i, r in enumerate(s) if r["te8"] / r["te4"] > 10]
    print(k, big, "max", round(max(r["te8"] / r["te4"] for r in s), 1))
