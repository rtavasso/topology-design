"""Q1/Q5: symbolic policies on P2's stratum (209 / 509), the training-world reference re-run in a different order, and
variants of the non-composing reference."""
import json, sys, random, time
sys.path.insert(0, "REPO_ROOT/src")
from topology_design.rewrite.proxy import goal_distances, trainworld_policy, trainworld_rollout, ted, RULE_INDEX
from topology_design.rewrite.rules import RULE_AX, valid_actions, world_from_name
from topology_design.rewrite.term import parse
from eng import read, TRAIN, HELD_B
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
flags = D["p2_comb"]
full = world_from_name("ACID")
cache = {}
def acts(u):
    a = cache.get(u)
    if a is None:
        a = valid_actions(u, full, 5); cache[u] = a
    return a

def variant_policy(u, s, W, cap, dist, mode, rng=None):
    cand = valid_actions(u, W, cap)
    if not cand: return None
    keys = {}
    for a in cand:
        rule, path, v = a
        best = fb = None
        for name, d in dist.items():
            if RULE_AX[rule] not in name: continue
            if u in d and d.get(v) == d[u] - 1:
                k_ = (d[u], len(name), name) if mode == "shorter-first" else (len(name), d[u], name)
                if best is None or k_ < best: best = k_
            if v in d:
                f_ = (d[v], len(name), name)
                if fb is None or f_ < fb: fb = f_
        keys[a] = (best, fb)
    t0 = [a for a in cand if keys[a][0] is not None]
    if t0:
        if mode == "random-tier0": return rng.choice(t0)
        if mode == "sigma-tiebreak": return min(t0, key=lambda a: (ted(a[2], s), RULE_INDEX[a[0]], a[1]))
        if mode == "any-world-then-lex": return min(t0, key=lambda a: (RULE_INDEX[a[0]], a[1]))
        return min(t0, key=lambda a: (keys[a][0], RULE_INDEX[a[0]], a[1]))
    if mode == "greedy-fallback":
        return min(cand, key=lambda a: (ted(a[2], s), RULE_INDEX[a[0]], a[1]))
    t1 = [a for a in cand if keys[a][1] is not None]
    if t1:
        return min(t1, key=lambda a: (keys[a][1], RULE_INDEX[a[0]], a[1]))
    if mode == "greedy-last":
        return min(cand, key=lambda a: (ted(a[2], s), RULE_INDEX[a[0]], a[1]))
    return min(cand, key=lambda a: (RULE_INDEX[a[0]], a[1]))

def roll(t, s, W, cap, T, dist, mode, rng=None):
    u = t
    for _ in range(T):
        if u == s: return True
        a = variant_policy(u, s, W, cap, dist, mode, rng)
        if a is None: return False
        u = a[2]
    return u == s

MODES = ["reference(reimpl)", "subs-only", "shorter-first", "sigma-tiebreak", "any-world-then-lex", "greedy-fallback", "greedy-last", "random-tier0"]
res = {}
t0 = time.time()
tiers = {"pairs": [0, 0, 0], "triples": [0, 0, 0]}
for w in HELD_B:
    W = world_from_name(w)
    recs = [r for r in read("eval5", w) if r["binding"] and flags[w][f"{r['t']}|{r['s']}"]["comb"]]
    recs = recs[::-1]                                  # reversed order: the reference must not depend on evaluation order
    subs = [u for u in TRAIN if set(u) < set(w)]
    out = {m: 0 for m in MODES}; out.update({"n": len(recs), "d1": 0, "d2": 0, "d3": 0, "trainw_derived": 0, "trainw_rerun": 0, "rerun_mismatch": 0, "random-tier0-draws": []})
    rr = [0, 0, 0]
    for r in recs:
        t, s, T = parse(r["t"]), parse(r["s"]), 2 * r["k"] + 2
        dist = {u: goal_distances(s, world_from_name(u), r["cap"], actions=acts) for u in TRAIN}
        dsub = {u: dist[u] for u in subs}
        f = flags[w][f"{r['t']}|{r['s']}"]
        a = bool(trainworld_rollout(t, s, W, r["cap"], T, dist))
        b = bool(trainworld_rollout(t, s, W, r["cap"], T, dict(reversed(list(dist.items())))))   # dict order must not matter
        out["trainw_derived"] += f["trainw"]; out["trainw_rerun"] += a; out["rerun_mismatch"] += (a != f["trainw"]) or (a != b)
        for d in (1, 2, 3): out[f"d{d}"] += bool(r["ladder"][str(d)])
        out["reference(reimpl)"] += roll(t, s, W, r["cap"], T, dist, "ref")
        out["subs-only"] += roll(t, s, W, r["cap"], T, dsub, "ref")
        for m in MODES[2:7]:
            out[m] += roll(t, s, W, r["cap"], T, dist, m)
        for i in range(3):
            rr[i] += roll(t, s, W, r["cap"], T, dist, "random-tier0", random.Random(1000 * i + hash(r["t"]) % 997))
    out["random-tier0"] = sum(rr) / 3; out["random-tier0-draws"] = rr
    res[w] = out
    print(w, out, f"{time.time()-t0:.0f}s", flush=True)
for cls, ws in (("pairs", ["AI", "CI"]), ("triples", ["ACI", "ACD", "AID", "CID"])):
    n = sum(res[w]["n"] for w in ws)
    print(cls, "n", n, {k: round(sum(res[w][k] for w in ws) / n, 3) for k in ["d1", "d2", "d3", "trainw_derived", "trainw_rerun"] + MODES}, "mismatch", sum(res[w]["rerun_mismatch"] for w in ws),
          "random draws", [round(sum(res[w]["random-tier0-draws"][i] for w in ws) / n, 3) for i in range(3)])
json.dump(res, open("p2_policies.json", "w"))
