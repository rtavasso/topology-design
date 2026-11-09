"""P3: (1) the matched ratio of analyze.py applied to the symbolic rungs themselves (is eval7 easier than eval5 on matched cells?);
(2) cell sizes behind the matched denominator; (3) flagged-unaddressable instances that still have a route within T = 2k+2 using
prefix positions <= 8 only (the reference counts shortest proofs only)."""
import sys, time
from collections import deque, Counter
from common import *
from topology_design.rewrite.rules import valid_actions, world_from_name
from topology_design.rewrite.term import parse, nodes
e5 = load("eval5"); e7 = load("eval7")
def matched(worlds, okf):
    c7, c5 = {}, {}
    for w in worlds:
        for r in e7[w]: c7.setdefault((w, r["k"], bool(r["binding"])), []).append(r)
        for r in e5[w]:
            if r["k"] <= 4: c5.setdefault((w, r["k"], bool(r["binding"])), []).append(r)
    num = den = used = 0
    for key, rows in c7.items():
        if key not in c5: continue
        num += sum(okf(r) for r in rows); den += len(rows) * (sum(okf(r) for r in c5[key]) / len(c5[key])); used += len(rows)
    return num / den, used, sum(len(v) for v in c7.values()), c7, c5
HELD = PAIRS + TRIPLES + ["ACID"]
for nm, ws in (("train", TRAIN_WORLDS), ("held-out", HELD)):
    for d in (1, 2, 3):
        ratio, used, n7, c7, c5 = matched(ws, lambda r: bool(r["ladder"][str(d)]))
        print(f"{nm}: ladder d={d} matched ratio eval7/eval5 = {ratio:.3f}  (eval7 n matched {used}/{n7})")
    small = sum(len(v) for k_, v in c7.items() if k_ in c5 and len(c5[k_]) < 10)
    unm = sum(len(v) for k_, v in c7.items() if k_ not in c5)
    print(f"  {nm}: eval7 instances whose eval5 cell has n<10: {small}; unmatched: {unm}; cells7={len(c7)} cells5={len(c5)}")
    print("   small eval5 cells:", {k_: (len(c5[k_]), len(c7[k_])) for k_ in c7 if k_ in c5 and len(c5[k_]) < 10})
    g7 = sum(r["first"]["GP1"] for w in ws for r in e7[w]) / sum(len(e7[w]) for w in ws)
    g5 = [r for w in ws for r in e5[w] if r["k"] <= 4]
    print(f"   GP1 share: eval7 {g7:.3f}, eval5 k<=4 {sum(r['first']['GP1'] for r in g5)/len(g5):.3f}; n eval5 k<=4 = {len(g5)}")
# per-world matched ratios of the rungs
for w in ALL:
    out = []
    for d in (1, 2, 3):
        try:
            out.append(f"{matched([w], lambda r: bool(r['ladder'][str(d)]))[0]:.2f}")
        except ZeroDivisionError:
            out.append("nan")
    print("  ", w, "ladder matched ratios d1/d2/d3:", out)
# (3) addressable-within-T routes on flagged instances
un = DER["eval7_unaddressable"]
def route_within(t, s, W, cap, T, max_nodes=400000):
    seen = {t}; fr = [t]
    for depth in range(T):
        nxt = []
        for u in fr:
            idx = {p: i for i, (p, _, _) in enumerate(nodes(u))}
            for rule, path, v in valid_actions(u, W, cap):
                if idx[path] > 8 or v in seen: continue
                if v == s: return depth + 1
                seen.add(v); nxt.append(v)
                if len(seen) > max_nodes: return -1
        fr = nxt
        if not fr: break
    return None
tot = Counter(); 
for w in TRAIN_WORLDS + HELD:
    if w not in ("A","C","I","D","AC","AD","CD","ID","AI","CI"): continue
    W = world_from_name(w); t0 = time.time()
    flagged = [r for r in e7[w] if un[w][f"{r['t']}|{r['s']}"]]
    res = Counter()
    for r in flagged:
        x = route_within(parse(r["t"]), parse(r["s"]), W, r["cap"], 2 * r["k"] + 2)
        res["undetermined" if x == -1 else ("none" if x is None else "addressable route within T")] += 1
        if x and x > 0: res[f"extra steps {x - r['k']}"] += 1
    print(w, "flagged unaddressable:", len(flagged), "of", len(e7[w]), dict(res), f"{time.time()-t0:.0f}s", flush=True)
