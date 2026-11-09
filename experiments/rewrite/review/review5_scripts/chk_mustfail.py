"""Q4b: must-fail flags recomputed with the reviewer's own engine and a bidirectional search; and the same set under the
positions the categorical head is actually labelled on at cap 5 (positions <= 6; A-rules at positions <= 4)."""
import json, sys, time, random
from eng import *
import eng
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
mf, un = D["eval7_mustfail"], D["eval7_unaddressable"]
VALID = {tuple(x) for x in json.load(open("valid_slots_cap5.json"))}
def allow8(rule, i): return i <= 8
def allow6(rule, i): return i <= 6
def allowS(rule, i): return (i, rule) in VALID
def ball(start, W, cap, radius, allow, stop=None):
    dist = {start: 0}; fr = [start]
    for d in range(1, radius + 1):
        nx = []
        for u in fr:
            for rule, i, v in moves(u, cap):
                if AX[rule] not in W or not allow(rule, i) or v in dist: continue
                dist[v] = d; nx.append(v)
        fr = nx
        if not fr: break
        if len(dist) > 600000: return None
    return dist
def route_within(t, s, W, cap, T, allow):
    """is there a route of <= T allowed steps? the allowed graph is symmetric (a rewrite and its inverse sit at the same
    prefix index, and every allow() used here admits both directions of an axiom at a position)."""
    a = (T + 1) // 2; b = T // 2
    F = ball(t, W, cap, a, allow)
    if s in F: return True
    B = ball(s, W, cap, b, allow)
    if F is None or B is None: return None
    small, big = (F, B) if len(F) < len(B) else (B, F)
    return any(v in big and small[v] + big[v] <= T for v in small)
t0 = time.time()
rows = []
tot = {"flag": 0, "mf8": 0, "mf8_derived": 0, "mism": 0, "mf6": 0, "mfS": 0, "n": 0, "short6": 0}
ownflags = {}
for w in TRAIN:
    recs = read("eval7", w)
    W = set(w)
    c = {"flag": 0, "mf8": 0, "mf8_derived": 0, "mism": 0, "mf6": 0, "mfS": 0, "n": len(recs)}
    ownflags[w] = {}
    for r in recs:
        key = f"{r['t']}|{r['s']}"
        t, s, T = parse(r["t"]), parse(r["s"]), 2 * r["k"] + 2
        flagged = bool(un[w].get(key))
        r6 = route_within(t, s, W, r["cap"], T, allow6)
        rS = r6 and route_within(t, s, W, r["cap"], T, allowS)
        assert r6 is not None and rS is not None
        m8 = None
        if flagged or not r6:
            r8 = route_within(t, s, W, r["cap"], T, allow8)
            assert r8 is not None
            m8 = not r8
        if flagged:
            c["flag"] += 1
            dv = mf[w].get(key)
            c["mf8_derived"] += dv is True
            c["mism"] += (dv is True) != m8
        else:
            assert not m8, "an unflagged instance has an addressable shortest proof"
        c["mf8"] += bool(m8); c["mf6"] += not r6; c["mfS"] += not rS
        ownflags[w][key] = {"mf8": bool(m8), "mf6": not r6, "mfS": not rS, "flag": flagged}
    eng._mv.clear()
    print(w, c, f"{time.time()-t0:.0f}s", flush=True)
    for k_ in tot: tot[k_] += c.get(k_, 0)
print("ALL", tot)
json.dump(ownflags, open("own_mustfail.json", "w"))
