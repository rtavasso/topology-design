"""The author's recount of design review 2's decisive numbers (my own traversal, refs3.py machinery; not the reviewer's code).

B1: are classes G and O earned by the wrong kind of model?
    - exact posterior with a prior of 0.9 per rule (a posterior; the review says it is classed O)
    - version space: uniform over actions optimal in at least one fitting world (no weights; the review says G out of support)
    - what-if planner: O averaged with O minus one more untested rule (no weights; the review says G out of support)
    - uniform over valid actions of rules not rejected (ignores the goal; the review says G on levels)
B2: at 4 training worlds, which rules are in every training world or in none (verdicts that never occur in training).

Usage: PYTHONPATH=src python3 experiments/belief/review2_checks.py [instances per draw] > experiments/belief/review2_checks.txt
"""
import collections, math, random, sys
import numpy as np
sys.path.insert(0, "experiments/belief")
import power as P
from refs3 import Reg, vec, rm, wscores, dchange, walk, draw_sets, GRID, DRAWS

RULES = ("Af", "Ab", "C", "If", "IbR", "IbL", "Df", "Db")
W09 = {w: 0.9 ** bin(w).count("1") * 0.1 ** (8 - bin(w).count("1")) for w in range(1, 256)}
ONE = collections.defaultdict(lambda: 1.0)


def support_uniform(p):
    x = (p > 1e-12).astype(float)
    return x / x.sum() if x.sum() > 0 else None


def whatif(ins, v, b, Pm, A, units_present):
    worlds = [255 & ~A]
    for j in range(8):
        if not (Pm | A) >> j & 1:
            w = 255 & ~(A | (1 << j))
            if w and ins.dw(v, w) <= b:
                worlds.append(w)
    return worlds


if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    g = np.random.default_rng(5)
    print("B2. Rules present in every training world / in none, at 4 and 8 training worlds:")
    for draw in DRAWS:
        _h, trains, _m = draw_sets(draw)
        for N in (4, 8):
            T = trains[N]
            allin = [RULES[j] for j in range(8) if all(w >> j & 1 for w in T)]
            none = [RULES[j] for j in range(8) if not any(w >> j & 1 for w in T)]
            print(f"  draw {draw}, N = {N}: in every training world {allin or '-'}; in none {none or '-'}")
    print("\nB1. Coefficients with 95 % bootstrap bounds over held-out worlds; levels on (G - M, O - M), out-of-support changes on (dG, dO | g, h).")
    names = ("post09", "vspace", "whatif", "uniform")
    for draw in DRAWS[:2]:
        heldout, trains, margs = draw_sets(draw)
        rng = random.Random(100 + draw)
        RL = {N: Reg(2) for N in GRID}
        RO = {N: Reg(4) for N in GRID}
        n = 0
        while n < n_inst:
            ins = P.Instance(rng, "H8")
            if ins.true not in heldout:
                continue
            n += 1

            def dec(v, alive, live, Pm, A, calls):
                acts = ins.out[v]
                units = np.array([ins.unit_of[c] for c, _x, _p in acts])
                q = vec(ins.scores(v, live), acts)
                o = vec(ins.scores(v, [255 & ~A]), acts)
                wi = [w for w in whatif(ins, v, 98, Pm, A, None) if w in live]
                lev = {"post09": vec(wscores(ins, v, live, W09), acts), "vspace": support_uniform(q),
                       "whatif": vec(wscores(ins, v, wi, ONE), acts),
                       "uniform": (lambda x: x / x.sum())(np.array([0.0 if A >> int(j) & 1 else 1.0 for j in units]))}
                probes = []
                for r in sorted(set(units.tolist())):
                    if (Pm | A) >> r & 1:
                        continue
                    wr = [w for w in live if not w >> r & 1]
                    base = rm(q, units, r)
                    if not wr or len(wr) == len(live) or base is None:
                        continue
                    q1 = vec(ins.scores(v, wr), acts)
                    o1 = vec(ins.scores(v, [255 & ~(A | (1 << r))]), acts)
                    wi1 = [w for w in ([255 & ~(A | (1 << r))] + [255 & ~(A | (1 << r) | (1 << j)) for j in range(8) if not (Pm | A | (1 << r)) >> j & 1]) if w in wr]
                    ch = {"post09": dchange(lev["post09"], vec(wscores(ins, v, wr, W09), acts), units, r),
                          "vspace": dchange(lev["vspace"], support_uniform(q1), units, r),
                          "whatif": dchange(lev["whatif"], vec(wscores(ins, v, wi1, ONE), acts), units, r),
                          "uniform": None}
                    probes.append((r, wr, q1, base, dchange(o, o1, units, r), ch))
                for N, T in trains.items():
                    liveT = [w for w in live if w in T]
                    qT = vec(ins.scores(v, liveT), acts) if liveT else None
                    if liveT:
                        RL[N].add(ins.true, (q - qT, o - qT), {k: x - qT for k, x in lev.items() if x is not None})
                    for r, wr, q1, base, do, ch in probes:
                        if any(w in T for w in wr):
                            continue
                        bT = rm(qT, units, r) if liveT else None
                        ob = rm(o, units, r)
                        RO[N].add(ins.true, (q1 - base, do, None if bT is None else base - bT, None if bT is None or ob is None else ob - bT), ch)

            walk(ins, None, dec)
            ins._d = ins.R = ins.out = None

        def ci(R, m, i):
            W = list(R.G)
            xs = [R._solve([W[k] for k in g.integers(0, len(W), len(W))], m)[i] for _ in range(300)]
            return f"{R.coef(m)[i]:.2f} [{np.nanpercentile(xs, 2.5):.2f}, {np.nanpercentile(xs, 97.5):.2f}]"

        for N in (4, 8, 16, 32):
            print(f"\n  draw {draw}, N = {N}  ({n_inst} held-out-world instances)")
            for m in names:
                lv = f"levels c_G {ci(RL[N], m, 0)}, c_O {ci(RL[N], m, 1)}"
                oc = f"out of support b_G {ci(RO[N], m, 0)}, b_O {ci(RO[N], m, 1)}" if m != "uniform" else "out of support: 0 by construction"
                print(f"    {m:8s} {lv};  {oc}")
