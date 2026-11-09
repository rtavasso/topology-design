"""Token-blind rank-then-mask policy that can plan only inside the eight TRAINING worlds (no composed planning, no token
reading), scored masked on the held-out binding strata. Same ranking as MINW+mask with candidates = training worlds;
fallback = smallest distance-to-s of the successor in any training world containing the action's axiom; then lex."""
import sys, json, time
from common import *
from s2_cueblind import acts, dist_from, RI
from topology_design.rewrite.rules import RULE_AX, world_from_name
from topology_design.rewrite.term import parse
e5 = load("eval5", PAIRS + TRIPLES)
def pol(t, s, W, cap, T, D):
    u = t
    for i in range(T):
        if u == s: return True
        cand = [a for a in acts(u, cap) if RULE_AX[a[0]] in W]
        if not cand: return False
        def key(a):
            best = None; fb = None
            for cw, d in D.items():
                if RULE_AX[a[0]] not in cw: continue
                if u in d and d.get(a[2]) == d[u] - 1:
                    k_ = (len(cw), d[u], cw)
                    if best is None or k_ < best: best = k_
                if a[2] in d:
                    f_ = (d[a[2]], len(cw), cw)
                    if fb is None or f_ < fb: fb = f_
            if best is not None: return (0, best, RI[a[0]], a[1])
            if fb is not None: return (1, fb, RI[a[0]], a[1])
            return (2, (0, 0, ""), RI[a[0]], a[1])
        u = min(cand, key=key)[2]
    return u == s
tot = {}
for w in PAIRS + TRIPLES:
    W = world_from_name(w); ok = n = 0; okg = 0
    for r in e5[w]:
        if not r["binding"]: continue
        t, s, cap, T = parse(r["t"]), parse(r["s"]), r["cap"], 2 * r["k"] + 2
        D = {cw: dist_from(s, cap, world_from_name(cw)) for cw in TRAIN_WORLDS}
        ok += pol(t, s, W, cap, T, D); n += 1; okg += bool(r["ladder"]["1"])
    tot[w] = (ok, n, okg)
    print(w, n, f"TRAINW+mask {ok/n:.3f} greedy {okg/n:.3f}", flush=True)
for nm, ws in (("pairs", PAIRS), ("triples", TRIPLES)):
    o = sum(tot[w][0] for w in ws); n = sum(tot[w][1] for w in ws); g = sum(tot[w][2] for w in ws)
    print(nm, n, f"TRAINW+mask {o/n:.3f} greedy {g/n:.3f}")
