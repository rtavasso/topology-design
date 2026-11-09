"""Q4a: the needs-the-unseen-combination flag, recomputed with the reviewer's own engine for every held-out binding instance."""
import json, sys, time, random
from collections import deque
from eng import *
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
flags = D["p2_comb"]
t0 = time.time()
def dist_in(t, s, U, cap, T):
    """shortest distance t -> s inside world U, searched to depth T (None if > T)."""
    if t == s: return 0
    seen = {t}; fr = [t]
    for d in range(1, T + 1):
        nx = []
        for u in fr:
            for rule, i, v in moves(u, cap):
                if AX[rule] not in U or v in seen: continue
                if v == s: return d
                seen.add(v); nx.append(v)
        fr = nx
        if not fr: break
    return None
tot = {}
mism = 0
for w in HELD_B:
    recs = [r for r in read("eval5", w) if r["binding"]]
    subs = [u for u in TRAIN if set(u) < set(w)]
    ncomb = 0; nm = 0
    for r in recs:
        t, s, T = parse(r["t"]), parse(r["s"]), 2 * r["k"] + 2
        # also check the binding flag's premise: distance inside W equals k
        within = [u for u in subs if dist_in(t, s, set(u), r["cap"], T) is not None]
        comb = not within
        f = flags[w][f"{r['t']}|{r['s']}"]["comb"]
        if comb != f: nm += 1
        ncomb += comb
    tot[w] = (len(recs), ncomb, nm); mism += nm
    print(w, "binding", len(recs), "comb", ncomb, "mismatch vs derived", nm, f"{time.time()-t0:.0f}s", flush=True)
print("pairs", tot["AI"][1] + tot["CI"][1], "triples", sum(tot[w][1] for w in ["ACI", "ACD", "AID", "CID"]), "mismatches", mism)
