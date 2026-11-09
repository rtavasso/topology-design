"""Cross-check of the reference (0.531 / 0.212) and of the greedy-fallback variant with the reviewer's own engine
(own BFS distances and move generator; only the tree-edit distance is the library's)."""
import json, sys
from collections import deque
sys.path.insert(0, "REPO_ROOT/src")
from topology_design.rewrite.proxy import ted
from eng import *
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
fl = D["p2_comb"]
RI = {r: i for i, r in enumerate(RULES)}
def dists(s, U, cap):
    d = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        for rule, i, v in moves(u, cap):
            if AX[rule] in U and v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d
def policy(u, s, W, cap, dist, mode):
    cand = [(r, i, v) for r, i, v in moves(u, cap) if AX[r] in W]
    if not cand: return None
    def tiers(a):
        r, i, v = a; best = fb = None
        for name, d in dist.items():
            if AX[r] not in name: continue
            if u in d and d.get(v) == d[u] - 1:
                k_ = (len(name), d[u], name); best = k_ if best is None or k_ < best else best
            if v in d:
                f_ = (d[v], len(name), name); fb = f_ if fb is None or f_ < fb else fb
        return best, fb
    T = {a: tiers(a) for a in cand}
    t0 = [a for a in cand if T[a][0] is not None]
    if t0: return min(t0, key=lambda a: (T[a][0], RI[a[0]], a[1]))
    if mode == "glue": return min(cand, key=lambda a: (ted(a[2], s), RI[a[0]], a[1]))
    t1 = [a for a in cand if T[a][1] is not None]
    if t1: return min(t1, key=lambda a: (T[a][1], RI[a[0]], a[1]))
    return min(cand, key=lambda a: (RI[a[0]], a[1]))
def roll(t, s, W, cap, Tn, dist, mode, trace=None):
    u = t
    for _ in range(Tn):
        if u == s: return True
        a = policy(u, s, W, cap, dist, mode)
        if a is None: return False
        if trace is not None: trace.append(a[0])
        u = a[2]
    return u == s
tot = {}
ex = []
for w in HELD_B:
    rs = [r for r in read("eval5", w) if r["binding"] and fl[w][f"{r['t']}|{r['s']}"]["comb"]]
    a = b = mism = 0; nax = 0
    for r in rs:
        t, s, Tn = parse(r["t"]), parse(r["s"]), 2 * r["k"] + 2
        dist = {u: dists(s, set(u), r["cap"]) for u in TRAIN}
        x = roll(t, s, set(w), r["cap"], Tn, dist, "ref"); tr = []
        y = roll(t, s, set(w), r["cap"], Tn, dist, "glue", tr)
        a += x; b += y; mism += x != fl[w][f"{r['t']}|{r['s']}"]["trainw"]
        if y:
            nax += len({AX[q] for q in tr}) >= 2
            if len(ex) < 3 and w == "CI": ex.append((r["t"], r["s"], r["k"], tr))
    tot[w] = (len(rs), a, b, mism, nax)
    print(w, "n", len(rs), "reference", a, "glue", b, "reference vs derived mismatches", mism, "| glue successes whose route uses >= 2 axioms:", nax)
for cls, ws in (("pairs", ["AI", "CI"]), ("triples", ["ACI", "ACD", "AID", "CID"])):
    n = sum(tot[w][0] for w in ws)
    print(cls, n, "reference %.3f glue %.3f" % (sum(tot[w][1] for w in ws) / n, sum(tot[w][2] for w in ws) / n))
print(ex)
