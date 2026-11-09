"""(a) MINW+mask: a token-blind rank-then-mask policy on the binding strata (masked scoring).
   score(a) = best (|W'|, d_W'(u,s), W') over candidate worlds W' containing a's axiom in which a is on a W'-shortest path;
   fallback: full-world distance of the successor. The mask (true world) is applied after ranking, as for a router.
(b) P7's ceiling: instance-draws the ceiling counts as failures (canonical axioms not within the two examples' axioms) —
   share solved by a planner restricted to the examples' axioms U (any proof within T), and by example-blind policies."""
import sys, json, random, time
from itertools import combinations
from common import *
from s2_cueblind import acts, dist_from, RI, FULLW
from topology_design.rewrite.rules import RULE_AX, world_from_name
from topology_design.rewrite.term import parse
from topology_design.rewrite.data import dec_action
s2 = json.load(open("s2_out.json"))
e5 = load("eval5")
def subsets(w):
    return ["".join(c) for n in range(1, len(w) + 1) for c in combinations(w, n)]
def minw_mask(t, s, W, cap, T, D):
    u = t
    for i in range(T):
        if u == s: return True
        cand = [a for a in acts(u, cap) if RULE_AX[a[0]] in W]
        if not cand: return False
        def key(a):
            best = None
            for cw, d in D.items():
                if RULE_AX[a[0]] in cw and u in d and d.get(a[2]) == d[u] - 1:
                    k_ = (len(cw), d[u], cw)
                    if best is None or k_ < best: best = k_
            return (0, best, RI[a[0]], a[1]) if best is not None else (1, (D["ACID"].get(a[2], 99), 0, ""), RI[a[0]], a[1])
        u = min(cand, key=key)[2]
    return u == s
out = {}
for w in ALL:
    if w == "ACID": continue
    W = world_from_name(w); t0 = time.time(); rows = []
    cands = TRAIN_WORLDS if w in TRAIN_WORLDS else ALL
    for r, r2 in zip(e5[w], s2[w]):
        if not r["binding"]: continue
        t, s, cap, T = parse(r["t"]), parse(r["s"]), r["cap"], 2 * r["k"] + 2
        D = {cw: dist_from(s, cap, world_from_name(cw)) for cw in set(cands) | {"ACID"}}
        mm = minw_mask(t, s, W, cap, T, {cw: D[cw] for cw in set(cands) | {"ACID"}})
        dU = {U: dist_from(s, cap, world_from_name(U), radius=T).get(t) for U in subsets(w)}
        rows.append({"minw_mask": mm, "fw_m": r2["fw_m"], "minw": r2["minw"], "d1": r2["d1"], "d3": r2["d3"], "k": r["k"],
                     "canon_ax": "".join(sorted({RULE_AX[dec_action(a)[0]] for a in r["canon"][:-1]})), "dU": dU})
    out[w] = rows
    print(w, len(rows), f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
json.dump(out, open("s5_out.json", "w"))
def rt(rows, key): return sum(r[key] for r in rows) / len(rows)
for nm, ws in (("pairs", PAIRS), ("triples", TRIPLES), ("train", TRAIN_WORLDS)):
    b = [r for w in ws for r in out[w]]
    print(f"{nm} binding n={len(b)}: greedy {rt(b,'d1'):.3f} d3 {rt(b,'d3'):.3f} | token-blind masked: FW+mask {rt(b,'fw_m'):.3f} MINW+mask {rt(b,'minw_mask'):.3f} | MINW unmasked {rt(b,'minw'):.3f}")
for w in ALL:
    if w != "ACID" and out[w]:
        print("  ", w, len(out[w]), f"MINW+mask {rt(out[w],'minw_mask'):.3f} FW+mask {rt(out[w],'fw_m'):.3f} d1 {rt(out[w],'d1'):.3f}")
# (b) examples ceiling on the binding stratum: draws as in derive (two examples from the same eval set, canonical axioms)
rng = random.Random(1)
for nm, ws in (("train", TRAIN_WORLDS), ("pairs", PAIRS), ("triples", TRIPLES)):
    tot = cok = uok = fail_n = fail_U = fail_fw = fail_minw = fail_mm = 0
    for w in ws:
        allax = ["".join(sorted({RULE_AX[dec_action(a)[0]] for a in r["canon"][:-1]})) for r in e5[w]]
        for r in out[w]:
            for _ in range(5):
                ex = rng.sample(range(len(allax)), 2)
                U = "".join(a for a in "ACID" if a in set(allax[ex[0]]) | set(allax[ex[1]]))
                c = set(r["canon_ax"]) <= set(U)
                reach = r["dU"].get(U) is not None
                tot += 1; cok += c; uok += reach
                if not c:
                    fail_n += 1; fail_U += reach; fail_fw += r["fw_m"]; fail_minw += r["minw"]; fail_mm += r["minw_mask"]
    print(f"{nm} binding: ceiling (canon axioms within examples) {cok/tot:.3f}; planner restricted to the examples' axioms, any proof within T: {uok/tot:.3f}; "
          f"on the {fail_n/tot:.3f} the ceiling counts as failures: U-planner {fail_U/max(1,fail_n):.2f}, FW+mask {fail_fw/max(1,fail_n):.2f}, MINW(unmasked) {fail_minw/max(1,fail_n):.2f}, MINW+mask {fail_mm/max(1,fail_n):.2f}")
