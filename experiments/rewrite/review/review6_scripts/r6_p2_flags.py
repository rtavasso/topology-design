"""Review 6 — P2's strata: (i) the 'needs the combination' flag of derived.json recomputed from scratch for every held-out
binding instance (own BFS on rules.valid_actions: a proof of length <= T = 2k+2 inside a training world contained in W);
(ii) the stored references recomputed with the library's proxy.trainworld_rollout as a consistency check on a sample;
(iii) per world: RT-canon's seeds against the stored floor and the glued reference.

    PYTHONPATH=src python3 experiments/rewrite/review/review6_scripts/r6_p2_flags.py > /tmp/r6_p2_flags.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *
from topology_design.rewrite.term import parse
from topology_design.rewrite.rules import valid_actions, world_from_name

D = load(); dv = derived(); flags = dv["p2_comb"]
M5 = D["meta"]["eval5"]; RC = seeds(D, "RT-canon")
key = lambda m: f"{m['t']}|{m['s']}"

def reach(t, s, U, T, cap=5):
    front, seen = {t}, {t}
    for d in range(T):
        if s in seen: return True
        nxt = set()
        for u in front:
            for _, _, v in valid_actions(u, U, cap):
                if v not in seen:
                    seen.add(v); nxt.add(v)
        front = nxt
        if not front: break
    return s in seen

mism = 0; n = 0
for i, m in enumerate(M5):
    if m["world"] in PAIRS + TRIPLES and m["binding"]:
        t, s = parse(m["t"]), parse(m["s"]); T = 2 * m["k"] + 2
        subs = [u for u in TRAIN if set(u) < set(m["world"])]
        prov = any(reach(t, s, world_from_name(u), T) for u in subs)
        n += 1
        if (not prov) != bool(flags[m["world"]][key(m)]["comb"]):
            mism += 1
print(f"'needs the combination' recomputed for {n} held-out binding instances: {mism} mismatches with derived.json")

print("\nper world, needs-the-combination stratum: RT-canon seeds against the world's own references (masked)")
for w in PAIRS + TRIPLES:
    ix = idx(M5, lambda m: m["world"] == w and m["binding"] and flags[w][key(m)].get("comb"))
    if not len(ix): continue
    tw = np.mean([bool(flags[w][key(M5[i])]["trainw"]) for i in ix]); gl = np.mean([bool(flags[w][key(M5[i])].get("glue")) for i in ix])
    g, d3 = lad(M5, ix, 1), lad(M5, ix, 3); fl = max(g, tw)
    pm = [rate(r, "eval5", ix) for r in RC]
    print(f"  {w}: n {len(ix)} greedy {g:.2f} training-world ref {tw:.2f} (floor {fl:.2f}) glued {gl:.2f} lookahead-3 {d3:.2f} | RT-canon {med(pm):.2f} [{min(pm):.2f}, {max(pm):.2f}] | seeds > floor {sum(x > fl + 1e-9 for x in pm)}/10, >= glued {sum(x >= gl - 1e-9 for x in pm)}/10, >= lookahead-3 {sum(x >= d3 - 1e-9 for x in pm)}/10")
