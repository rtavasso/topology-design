"""Held-out binding instances (eval5): does a shortest in-world proof already live inside a TRAINING world contained in W
(no unseen axiom combination needed)? dU[U] = d_U(t,s) within T for every non-empty subset U of W (from s5)."""
import json
from collections import Counter
from common import *
out = json.load(open("s5_out.json"))
e5 = load("eval5", PAIRS + TRIPLES)
TW = set(TRAIN_WORLDS)
agg = {}
for w in PAIRS + TRIPLES:
    rows = out[w]; n = len(rows)
    subs = [U for U in rows[0]["dU"] if U != w and U in TW]
    c = Counter()
    for r in rows:
        k = r["k"]
        short = any(r["dU"][U] == k for U in subs)                       # a shortest W-proof lies in a training sub-world
        within = any(r["dU"][U] is not None for U in subs)              # some proof within T = 2k+2 lies in a training sub-world
        canon_in_train = r["canon_ax"] in TW
        c["shortest in a training sub-world"] += short; c["some proof within T in a training sub-world"] += within
        c["canonical proof's axioms form a training world"] += canon_in_train
        c["needs the unseen combination (no proof within T in any training sub-world)"] += (not within)
        c["greedy ok & needs combination"] += (not within and r["d1"]); c["d3 ok & needs combination"] += (not within and r["d3"])
        c["minw_mask ok & needs combination"] += (not within and r["minw_mask"])
    agg[w] = (n, c)
    print(w, n, {k_: f"{v}/{n} = {v/n:.3f}" for k_, v in c.items()}, "training sub-worlds:", subs)
for nm, ws in (("pairs", PAIRS), ("triples", TRIPLES)):
    n = sum(agg[w][0] for w in ws); c = Counter()
    for w in ws: c.update(agg[w][1])
    print(nm, n, {k_: f"{v/n:.3f}" for k_, v in c.items()})
    need = c["needs the unseen combination (no proof within T in any training sub-world)"]
    print(f"   on the {need} instances that need the combination: greedy {c['greedy ok & needs combination']/need:.3f}, d3 {c['d3 ok & needs combination']/need:.3f}, MINW+mask {c['minw_mask ok & needs combination']/need:.3f}")
