"""The author's recount of review 1's two decisive numbers, before acting on them (own traversal; reuses power.Instance).
  (1) beta of a rule-identity lookup: belief frozen at P(W | u, s), every action of a rejected rule removed.
  (2) the share of gamma's denominator that sits on decisions where no training world agrees with the evidence
      (where the memorising reference is defined only by a fallback).
Usage: PYTHONPATH=src python3 experiments/belief/review1_checks.py > experiments/belief/review1_checks.txt"""
import collections, random, sys
import numpy as np
sys.path.insert(0, "experiments/belief")
import power as P

def walk(ins, trains=None):
    v, b, Pm, A, calls = 0, ins.B, 0, 0, 0
    rejected = collections.defaultdict(set)
    out = []
    while v != ins.s and b > 0 and calls < ins.B + 40:
        live = [w for w in ins.cons if ins.dw(v, w) <= b]
        upd = [w for w in live if (w & Pm) == Pm and (w & A) == 0]
        q = ins.scores(v, upd)
        if not q:
            break
        acts = ins.out[v]
        frozen = ins.scores(v, live)
        def norm(d):
            z = sum(d.values())
            return np.array([d.get(a, 0.0) / z for a in acts]) if z > 0 else None
        m = norm({a: x for a, x in frozen.items() if a not in rejected[v]})
        rm = norm({a: x for a, x in frozen.items() if not A >> ins.unit_of[a[0]] & 1})
        rec = dict(after=calls > 0, q=norm(q), m=m, rm=rm)
        if trains:
            for N, T in trains.items():
                base = [w for w in ins.cons if w in T] or ins.cons
                tu = [w for w in base if ins.dw(v, w) <= b and (w & Pm) == Pm and (w & A) == 0]
                rec[("fb", N)] = not tu
        out.append(rec)
        a = max(sorted(q), key=lambda t: q[t])
        calls += 1
        j = ins.unit_of[a[0]]
        if ins.true >> j & 1:
            v, b, Pm = a[1], b - 1, Pm | (1 << j)
        else:
            rejected[v].add(a); A |= 1 << j
    return out

rng = random.Random(0)
num = den = 0.0
n = 0
for _ in range(600):
    ins = P.Instance(rng, "H8")
    for r in walk(ins):
        if r["after"] and r["m"] is not None and r["rm"] is not None:
            d = r["q"] - r["m"]
            num += float((r["rm"] - r["m"]) @ d); den += float(d @ d); n += 1
print(f"(1) rule-identity lookup on a frozen belief: beta = {num / den:.3f} over {n} decisions after a verdict (600 instances, seed 0)")

order = list(range(1, 256)); random.Random(7).shuffle(order)
heldout, pool = set(order[:64]), order[64:]
trains = {N: set(pool[:N]) for N in (8, 16, 32, 64)}
rng = random.Random(1)
rows = []
while len(rows) < 400:
    ins = P.Instance(rng, "H8")
    if ins.true not in heldout:
        continue
    full = walk(ins, trains)
    per = {N: ins.trace(train=T) for N, T in trains.items()}
    rows.append((full, per))
for N in trains:
    tot = fb = 0.0; nfb = nd = 0
    for full, per in rows:
        for r0, r in zip(full, per[N]):
            e = float(((r["q"] - r["qt"]) ** 2).sum())
            tot += e; nd += 1
            if r0[("fb", N)]:
                fb += e; nfb += 1
    print(f"(2) N = {N}: no training world agrees with the evidence on {nfb / nd:.0%} of held-out decisions; they carry {fb / tot:.0%} of gamma's denominator")
