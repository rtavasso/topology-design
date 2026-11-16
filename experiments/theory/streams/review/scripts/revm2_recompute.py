"""Review of rounds M / M2 / M2 extension: recompute every figure in the note's section from the per-seed JSON (no training)."""
import json, math, statistics as st
H = "/home/claude/topology-design/experiments/theory/streams/"
med = st.median
def load(pre):
    d = {}
    for t in "ab":
        d.update(json.load(open(H + f"{pre}_results_{t}.json")))
    return d
M, S, X = load("arch_m"), load("arch_m2"), load("arch_m2x")
CM = {"M3": "M1", "M3a": "M1", "M3b": "M1", "M16": "M1", "M4": "M1", "M4a": "M1", "M7": "M1",
      "M9": "M8", "M17": "M8", "M10": "M8", "M13": "M12", "M13b": "M12", "M15": "M12"}
CN = {"N4": "N1", "N3": "N1", "N4a": "N1s", "N10": "N8", "N14": "N12", "N13": "N12"}
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))

for name, d, C in (("ROUND M", M, CM), ("M2 AS DESIGNED", S, CN), ("M2 EXTENSION", X, CN)):
    print("=" * 20, name)
    print("row | np | train4 med | te4 med [min,max] | te4/train4 med [min,max] n>1 | te8/te4 med, n>3, n>10 | x2/te4 med | c1 med; c2 med | addr med,min | switch med[min,max] | train_ran")
    for k, s in d.items():
        f = lambda n: [r[n] for r in s]
        g = [r["te4"] / r["train4"] for r in s]
        r84 = [r["te8"] / r["te4"] for r in s]
        sw = [x for x in f("switch") if x is not None]
        tr = [r.get("train_ran") for r in s]
        print(f"{k:4s} | {s[0]['nparams']} | {med(f('train4')):.2e} | {med(f('te4')):.2e} [{min(f('te4')):.1e},{max(f('te4')):.1e}] | {med(g):.3f} [{min(g):.2f},{max(g):.2f}] {sum(x>1 for x in g)} | "
              f"{med(r84):.2f} n>3={sum(x>3 for x in r84)} n>10={sum(x>10 for x in r84)} | {med([r['x2']/r['te4'] for r in s]):.1f} (x2 med {med(f('x2')):.1e}) | {med(f('c1')):.1e}; {med(f('c2')):.1e} | "
              f"{med(f('addr')):.2f},{min(f('addr')):.2f} | {(med(sw), min(sw), max(sw), len(sw)) if sw else None} | {tr if tr[0] is not None else ''}")
    print("paired: row vs cand | te4 ratio med [min,max] n>1 | geo | c1 ratio med n>1 | c2 ratio med n>1 | te8 ratio med n>1 | x2 ratio med n>1")
    for k, c in C.items():
        rr = lambda n: [a[n] / b[n] for a, b in zip(d[k], d[c])]
        r4, r1, r2, r8, rx = rr("te4"), rr("c1"), rr("c2"), rr("te8"), rr("x2")
        print(f"{k:4s} vs {c:4s} | {med(r4):.3f} [{min(r4):.2f},{max(r4):.1f}] {sum(x>1 for x in r4)}/10 | {gm(r4):.2f} | {med(r1):.2f} {sum(x>1 for x in r1)}/10 | {med(r2):.2f} {sum(x>1 for x in r2)}/10 | "
              f"{med(r8):.2f} {sum(x>1 for x in r8)}/10 | {med(rx):.2f} {sum(x>1 for x in rx)}/10")
    print()

print("per-seed te4 and ratios for the bold rows")
for nm, d, a, b in (("M", M, "M4", "M1"), ("M", M, "M10", "M8"), ("S", S, "N4", "N1"), ("S", S, "N10", "N8"), ("X", X, "N4", "N1"), ("X", X, "N10", "N8"),
                    ("S", S, "N4a", "N1s"), ("X", X, "N4a", "N1s"), ("S", S, "N14", "N12"), ("X", X, "N14", "N12")):
    print(nm, a, "/", b, "te4:", " ".join(f"{x[ 'te4']/y['te4']:.2f}" for x, y in zip(d[a], d[b])))
    print(nm, a, "/", b, "c1 :", " ".join(f"{x[ 'c1']/y['c1']:.2f}" for x, y in zip(d[a], d[b])))
for nm, d, k in (("S", S, "N1s"), ("X", X, "N1s"), ("S", S, "N4a"), ("X", X, "N4a")):
    print(nm, k, "te4", " ".join(f"{r['te4']:.1e}" for r in d[k]), "| addr", [round(r["addr"], 2) for r in d[k]])

print("\nheld-out/training range over attention rows (medians), per regime")
for name, d in (("M", M), ("S", S), ("X", X)):
    ms = {k: med([r["te4"] / r["train4"] for r in s]) for k, s in d.items() if k != "M7"}
    allr = [r["te4"] / r["train4"] for k, s in d.items() if k != "M7" for r in s]
    print(name, f"medians {min(ms.values()):.3f} ({min(ms, key=ms.get)}) .. {max(ms.values()):.3f} ({max(ms, key=ms.get)}); per seed {min(allr):.3f} .. {max(allr):.3f}")
g7 = [r["te4"] / r["train4"] for r in M["M7"]]
print("M7 gap per seed", [round(x, 2) for x in g7], "train med", med([r["train4"] for r in M["M7"]]))

print("\ncross-regime, same seeds (unpaired in optimiser but same data): candidate te4 by regime; te8/te4 > 3 counts")
for d, n in ((M, "M"), (S, "S"), (X, "X")):
    for k, s in d.items():
        r = [x["te8"] / x["te4"] for x in s]
        print(n, k, "n(te8/te4>3) =", sum(x > 3 for x in r), " te8 med", f"{med([x['te8'] for x in s]):.1e}", "te6/te4 n>3", sum(x["te6"] / x["te4"] > 3 for x in s), "solved te8 (<1e-2):", sum(x["te8"] < 1e-2 for x in s))

print("\nx2: absolute medians and x2/te4")
for d, n in ((M, "M"), (S, "S"), (X, "X")):
    for k, s in d.items():
        print(n, k, f"x2 med {med([r['x2'] for r in s]):.1e}  x2/te4 med {med([r['x2']/r['te4'] for r in s]):.1f}")

print("\ntrain_ran -> mean block madds in training (streams rows): w*d*2 per stream")
cfg = {"N1": (4, 12, 0), "N8": (8, 12, 0), "N1s": (3, 6, 0), "N12": (4, 12, 1)}
for d, n in ((S, "S"), (X, "X")):
    for k, (K, w, gh) in cfg.items():
        tr = [r["train_ran"] for r in d[k]]
        gate = K * 8 + (24 * (8 + 6) + K * 24 if gh else 0)
        cost = [gate + t * 2 * w * 6 for t in tr]
        sw = [r["switch"] for r in d[k]]
        print(n, k, "train_ran med", round(med(tr), 2), [round(t, 2) for t in tr], "mean madds med", round(med(cost)), "range", round(min(cost)), round(max(cost)), "switch", sw)
for k in ("M1", "M8", "M12"):
    print("M", k, "switch", [r["switch"] for r in M[k]])

# ---- second part: tests the note does not report
def signflip_p(logs):
    """Exact two-sided sign-flip (permutation) test on the mean of log ratios."""
    n = len(logs); obs = abs(sum(logs)); c = 0
    for m in range(1 << n):
        s = sum(x if (m >> i) & 1 else -x for i, x in enumerate(logs))
        c += abs(s) >= obs - 1e-12
    return c / (1 << n)
from math import comb, log
def sign_p(k, n):
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
print("\nIs 0.63 / 0.62 distinguishable from 1?  (log ratio per-unit gate / candidate)")
pool = []
for nm, d, a, b in (("M", M, "M4", "M1"), ("M", M, "M10", "M8"), ("X", X, "N4", "N1"), ("X", X, "N10", "N8")):
    l = [log(x["te4"] / y["te4"]) for x, y in zip(d[a], d[b])]
    k = sum(x < 0 for x in l)
    print(nm, a, b, f"gate ahead {k}/10 sign p={sign_p(k, 10):.2f}  geo {math.exp(sum(l)/10):.2f}  sign-flip p={signflip_p(l):.3f}")
    if nm == "X": pool += l
k = sum(x < 0 for x in pool)
print("X pooled 20:", f"gate ahead {k}/20 sign p={sign_p(k, 20):.3f} geo {math.exp(sum(pool)/20):.2f} sign-flip p={signflip_p(pool):.3f}")

print("\nInputs doubled: per-seed x2 error ratio block/candidate, and whose error rises more (x2/te4)")
for nm, d, a, b in (("M", M, "M4", "M1"), ("M", M, "M10", "M8"), ("S", S, "N4", "N1"), ("S", S, "N10", "N8"), ("X", X, "N4", "N1"), ("X", X, "N10", "N8"),
                    ("S", S, "N14", "N12"), ("X", X, "N14", "N12")):
    rx = [x["x2"] / y["x2"] for x, y in zip(d[a], d[b])]
    rise_c = [y["x2"] / y["te4"] for y in d[b]]; rise_g = [x["x2"] / x["te4"] for x in d[a]]
    l = [log(v) for v in rx]
    print(nm, a, b, f"x2 ratio med {med(rx):.2f} [{min(rx):.2f},{max(rx):.2f}] cand better {sum(v>1 for v in rx)}/10 sign-flip p={signflip_p(l):.3f} | rise cand {med(rise_c):.1f} gate {med(rise_g):.1f}; cand rises more in {sum(c>g for c,g in zip(rise_c,rise_g))}/10")

print("\nLength 8 in the extension: per-seed te8, and te6")
for k in ("N1", "N4", "N8", "N10", "N12", "N14"):
    print("X", k, "te4", " ".join(f"{r['te4']:.0e}" for r in X[k]))
    print("X", k, "te6", " ".join(f"{r['te6']:.0e}" for r in X[k]))
    print("X", k, "te8", " ".join(f"{r['te8']:.0e}" for r in X[k]), "| <1e-2:", sum(r["te8"] < 1e-2 for r in X[k]), " te8/te4>3:", sum(r["te8"]/r["te4"] > 3 for r in X[k]), " te8/te4>100:", sum(r["te8"]/r["te4"] > 100 for r in X[k]))
for k in ("M1", "N1"):
    d = M if k == "M1" else S
    print(k, "te8/te4", " ".join(f"{r['te8']/r['te4']:.1f}" for r in d[k]))
print("\nS4 second clause N13/N14:", f"{med([a['te4']/b['te4'] for a,b in zip(S['N13'],S['N14'])]):.1f}", " X:", f"{med([a['te4']/b['te4'] for a,b in zip(X['N13'],X['N14'])]):.1f}")
print("\nSame-seed comparison across regimes (candidate at 0.01 decaying vs per-unit gate at its best 0.1): gate X / cand S")
for a, b in (("N4", "N1"), ("N10", "N8")):
    r = [x["te4"] / y["te4"] for x, y in zip(X[a], S[b])]
    print(a, b, f"med {med(r):.2f}")
print("M vs S step-1000 overlap: c1 of M1 vs N1 per seed", [f"{a['c1']/b['c1']:.2f}" for a, b in zip(M["M1"], S["N1"])], "M4 vs N4", [f"{a['c1']/b['c1']:.2f}" for a, b in zip(M["M4"], S["N4"])])

print("\nSeeds whose register has lost the address IN RANGE (addr < 1 at length 4) and their te4")
for nm, d in (("M", M), ("S", S), ("X", X)):
    for k, s in d.items():
        bad = [(i, round(r["addr"], 2), f"{r['te4']:.1e}") for i, r in enumerate(s) if r["addr"] < 1]
        if bad: print(nm, k, bad, "| seeds with te4 >= 1e-2:", [i for i, r in enumerate(s) if r["te4"] >= 1e-2])
print("\nSwitch forced by the cap (step 1501 = half of training):")
for nm, d in (("M", M), ("S", S), ("X", X)):
    for k, s in d.items():
        if s[0]["switch"] is not None: print(nm, k, "capped seeds:", [i for i, r in enumerate(s) if r["switch"] >= 1501], " c1 of those:", [f"{r['c1']:.0e}" for r in s if r["switch"] >= 1001])
