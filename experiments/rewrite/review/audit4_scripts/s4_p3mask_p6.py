from collections import Counter
from common import *
from topology_design.rewrite.rules import valid_actions, world_from_name
from topology_design.rewrite.term import parse, nodes
from topology_design.rewrite.data import dec_action
e5 = load("eval5", TRAIN_WORLDS); e7 = load("eval7")
un = DER["eval7_unaddressable"]
print("== P3: flagged-unaddressable eval7 instances: valid-action set at t under the mask")
tot = Counter()
for w in ALL:
    W = world_from_name(w)
    fl = [r for r in e7[w] if un[w][f"{r['t']}|{r['s']}"]]
    c = Counter()
    for r in fl:
        t = parse(r["t"]); idx = {p: i for i, (p, _, _) in enumerate(nodes(t))}
        va = valid_actions(t, W, r["cap"])
        hi = [a for a in va if idx[a[1]] > 8]
        c["all valid actions at pos>=9 (mask leaves only untrained rows + HALT)"] += (len(hi) == len(va))
        c["<=2 valid actions"] += (len(va) <= 2)
        c["n"] += 1
    tot.update(c)
    print(w, dict(c))
print("total", dict(tot))
print("== P6: eval5 training worlds, instances with >1 trace class and >1 shortest-path first action")
def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))
allsel = []
for w in TRAIN_WORLDS:
    sel = []
    for r in e5[w]:
        tf = next(p["target"] for p in r["post"] if p["state"] == r["t"])
        if r["n_trace_classes"] > 1 and len(tf) > 1:
            sel.append((r, tf))
    allsel += sel
    if sel:
        c1 = sum(tv({r["canon"][0]: 1.0}, tf) for r, tf in sel) / len(sel)
        cu = sum(tv({a: 1 / len(tf) for a in tf}, tf) for r, tf in sel) / len(sel)
        mode = sum(tf[r["canon"][0]] >= max(tf.values()) - 1e-12 for r, tf in sel) / len(sel)
        print(f"{w}: n={len(sel)} of {len(e5[w])}; TV(canon one-hot, target)={c1:.3f}; TV(uniform over first actions, target)={cu:.3f}; canon is a target mode: {mode:.2f}; mean #first actions {sum(len(tf) for _, tf in sel)/len(sel):.2f}")
    else:
        print(w, "n=0")
n = len(allsel)
print("pooled n", n, "TV canon", sum(tv({r['canon'][0]: 1.0}, tf) for r, tf in allsel) / n, "TV uniform", sum(tv({a: 1 / len(tf) for a in tf}, tf) for r, tf in allsel) / n,
      "canon is mode", sum(tf[r['canon'][0]] >= max(tf.values()) - 1e-12 for r, tf in allsel) / n,
      "target exactly uniform", sum(max(tf.values()) - min(tf.values()) < 1e-9 for r, tf in allsel) / n)
# argmax of the target (RT-post's ideal committed action) equals canon?
import random
rng = random.Random(0)
print("share where the target's unique argmax is the canonical action:", sum((tf[r['canon'][0]] > max(v for a, v in tf.items() if a != r['canon'][0]) + 1e-12) for r, tf in allsel) / n)
# is the target a function of (world, s, state) across instances? check on eval5 post entries
for w in TRAIN_WORLDS:
    seen = {}; clash = 0; tot_ = 0
    for r in e5[w]:
        for p in r["post"]:
            key = (r["s"], p["state"]); tot_ += 1
            if key in seen and any(abs(seen[key].get(a, 0) - p["target"].get(a, 0)) > 1e-9 for a in set(seen[key]) | set(p["target"])):
                clash += 1
            seen.setdefault(key, p["target"])
    print(w, "post states", tot_, "distinct (s,state)", len(seen), "conflicting targets", clash)
