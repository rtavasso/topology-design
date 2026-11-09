"""Token-blind symbolic policies on eval5, scored as the masked / unmasked readings would score a router.
 FW      : the cue-blind planner the ceiling assumes (ranks actions by full-world distance of the successor to s; lex tie-break)
           - unmasked: argmax over all actions (full-world valid); an out-of-world action ends the episode
           - masked  : argmax over the actions valid in the instance's world
 FWstrict: masked, but only actions on a full-world shortest path count (fail if none is valid)
 MINW    : token-blind, mask-free: picks the candidate world with the fewest axioms in which s is reachable from t
           (ties: shorter proof, then name), follows a shortest proof there. Scored unmasked.
"""
import sys, json, time
from collections import deque
from common import *
from topology_design.rewrite.rules import RULES, RULE_AX, valid_actions, world_from_name
from topology_design.rewrite.term import parse
RI = {r: i for i, r in enumerate(RULES)}
FULLW = frozenset("ACID")
_adj = {}
def acts(u, cap):                       # all full-world state-changing actions at u: [(rule, path, new)] in lex order
    key = (u, cap)
    a = _adj.get(key)
    if a is None:
        a = sorted(valid_actions(u, FULLW, cap), key=lambda x: (RI[x[0]], x[1]))
        _adj[key] = a
    return a
def dist_from(s, cap, W=FULLW, radius=None):      # symmetric graph: distances to s = distances from s
    d = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        if radius is not None and d[u] >= radius: continue
        for r, p, v in acts(u, cap):
            if RULE_AX[r] in W and v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d
def rollout_rank(t, s, W, cap, T, dfull, masked, strict=False):
    u = t
    for i in range(T):
        if u == s: return True
        cand = acts(u, cap)
        if masked:
            cand = [a for a in cand if RULE_AX[a[0]] in W]
        if not cand: return False
        best = min(cand, key=lambda a: (dfull.get(a[2], 99), RI[a[0]], a[1]))
        if strict and dfull.get(best[2], 99) != dfull[u] - 1: return False
        if RULE_AX[best[0]] not in W: return False          # unmasked: out of world ends the episode
        u = best[2]
    return u == s
def run(worlds, cand_worlds_fn):
    e5 = load("eval5", worlds)
    out = {}
    for w in worlds:
        W = world_from_name(w); t0 = time.time()
        rows = []
        for r in e5[w]:
            t, s, cap, k = parse(r["t"]), parse(r["s"]), r["cap"], r["k"]
            T = 2 * k + 2
            dfull = dist_from(s, cap)
            assert dfull[t] == (r["d_full"] if r["d_full"] is not None else k), (r["t"], r["s"], dfull[t], r["d_full"], k)
            res = {"binding": bool(r["binding"]), "k": k, "GP1": bool(r["first"]["GP1"]),
                   "fw_u": rollout_rank(t, s, W, cap, T, dfull, masked=False),
                   "fw_m": rollout_rank(t, s, W, cap, T, dfull, masked=True),
                   "fw_ms": rollout_rank(t, s, W, cap, T, dfull, masked=True, strict=True),
                   "d1": bool(r["ladder"]["1"]), "d3": bool(r["ladder"]["3"])}
            # MINW
            best = None
            for cw in cand_worlds_fn(w):
                CW = world_from_name(cw)
                dd = dist_from(t, cap, CW, radius=T).get(s)
                if dd is None: continue
                key = (len(cw), dd, cw)
                if best is None or key < best: best = key
            res["minw"] = bool(best is not None and set(best[2]) <= set(w) and best[1] <= T)
            res["minw_world"] = best[2] if best else None
            rows.append(res)
        out[w] = rows
        print(w, f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return out
if __name__ == "__main__":
    res = run(ALL, lambda w: TRAIN_WORLDS if w in TRAIN_WORLDS else ALL)
    json.dump(res, open("s2_out.json", "w"))
