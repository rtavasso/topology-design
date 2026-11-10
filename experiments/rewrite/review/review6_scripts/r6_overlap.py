"""Review 6, part C — train / eval overlap: could P4's reach (and the training-world rates) be memorisation?

    PYTHONPATH=src python3 experiments/rewrite/review/review6_scripts/r6_overlap.py > /tmp/r6_overlap.txt

Streams the eight training files once (one line at a time; the PF block of each record is cut off before parsing) and
builds, per training world: the set of (t, s) instance pairs, the set of teacher-forced (state, goal) rows of RT-canon
(the states of the canonical proof) and of RT-post (the states of the shortest DAG), and the sets of goals and starts.
Then every eval5 / evalk78 instance is looked up, and RT-canon's success is split by seen / unseen."""
import sys, os, gzip, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *
from topology_design.rewrite.term import parse, to_str
from topology_design.rewrite.rules import step, world_from_name

D = load(); dv = derived()
M5, M78 = D["meta"]["eval5"], D["meta"]["evalk78"]
RC = seeds(D, "RT-canon")
pairs, rows_c, rows_p, goals, starts, ntrain, caps = {}, {}, {}, {}, {}, {}, collections.Counter()
kdist = {}
for w in TRAIN:
    P, Rc, Rp, G, S = set(), set(), set(), set(), set(); n = 0; kd = collections.Counter()
    with gzip.open(os.path.join(ROOT, "data", "rewrite", f"train_{w}.jsonl.gz"), "rt") as f:
        for line in f:
            j = line.find(', "pf": ')
            r = json.loads(line[:j] + "}" if j > 0 else line)
            n += 1; caps[r["cap"]] += 1; kd[r["k"]] += 1
            t, s = r["t"], r["s"]
            P.add(t + "|" + s); G.add(s); S.add(t)
            W = world_from_name(w); u = parse(t)
            Rc.add(t + "|" + s)
            for a in r["canon"][:-1]:
                rule, path = a.split("@")
                u = step(u, rule, path, W, r["cap"]); Rc.add(to_str(u) + "|" + s)
            assert to_str(u) == s
            for p in r["post"]:
                Rp.add(p["state"] + "|" + s)
    pairs[w], rows_c[w], rows_p[w], goals[w], starts[w], ntrain[w], kdist[w] = P, Rc, Rp, G, S, n, kd
    print(f"train {w}: n {n} distinct (t,s) {len(P)} distinct canon rows {len(Rc)} distinct post rows {len(Rp)} distinct goals {len(G)} distinct starts {len(S)} k {dict(sorted(kd.items()))}", flush=True)
print("training caps:", dict(caps))

okc5 = np.mean([r["out"]["eval5"]["ok_m"] for r in RC], 0); okc78 = np.mean([r["out"]["evalk78"]["ok_m"] for r in RC], 0)
def look(m, w=None):
    w = w or m["world"]; key = m["t"] + "|" + m["s"]
    return key in pairs[w], key in rows_c[w], key in rows_p[w], m["s"] in goals[w], m["t"] in starts[w]

print("\n== eval5, training worlds: share of eval instances whose (t, s) is a training instance / a teacher-forced row ==")
print("world | n | (t,s) is a training instance | (t,s) is an RT-canon row | an RT-post row | goal s seen as a goal | start t seen as a start")
tot = collections.Counter()
F = {}
for i, m in enumerate(M5):
    if m["world"] in TRAIN:
        F[i] = look(m)
for w in TRAIN:
    ix = [i for i in F if M5[i]["world"] == w]
    print(f"  {w} | {len(ix)} | " + " | ".join(f"{np.mean([F[i][j] for i in ix]):.3f}" for j in range(5)))
ix = list(F)
print(f"  all | {len(ix)} | " + " | ".join(f"{np.mean([F[i][j] for i in ix]):.3f}" for j in range(5)))
print("\nby k (training worlds pooled): n | pair seen | canon-row seen | post-row seen | goal seen | RT-canon ok on row-seen | on row-unseen | on goal-unseen")
for k in range(1, 7):
    ix = [i for i in F if M5[i]["k"] == k]
    a = [i for i in ix if F[i][1]]; b = [i for i in ix if not F[i][1]]; c = [i for i in ix if not F[i][3]]
    print(f"  k={k} | {len(ix)} | {np.mean([F[i][0] for i in ix]):.3f} | {np.mean([F[i][1] for i in ix]):.3f} | {np.mean([F[i][2] for i in ix]):.3f} | {np.mean([F[i][3] for i in ix]):.3f} | "
          f"{okc5[a].mean() if a else float('nan'):.3f} (n {len(a)}) | {okc5[b].mean() if b else float('nan'):.3f} (n {len(b)}) | {okc5[c].mean() if c else float('nan'):.3f} (n {len(c)})")

# the §16 cells
cells16 = set(dv["p4_amended_pooled"]["cells"])
ix16 = [i for i in F if f"{M5[i]['world']}|{int(M5[i]['binding'])}|{int(M5[i]['GP1'])}|{M5[i]['k']}" in cells16]
print(f"\n== the §16 cells (n {len(ix16)}) ==")
print("  (t,s) is a training instance:", sum(F[i][0] for i in ix16), "| an RT-canon teacher-forced row:", sum(F[i][1] for i in ix16), "| an RT-post row:", sum(F[i][2] for i in ix16), "| goal seen as a training goal:", sum(F[i][3] for i in ix16), "| start seen:", sum(F[i][4] for i in ix16))
for lab, pred in (("pair seen", lambda i: F[i][0]), ("pair unseen", lambda i: not F[i][0]), ("row unseen (canon)", lambda i: not F[i][1]), ("goal unseen", lambda i: not F[i][3]), ("goal and start unseen", lambda i: not F[i][3] and not F[i][4])):
    sel = [i for i in ix16 if pred(i)]
    if sel:
        per = [float(r["out"]["eval5"]["ok_m"][sel].mean()) for r in RC]
        d1, d2, d3 = (lad(M5, sel, d) for d in (1, 2, 3))
        print(f"  {lab}: n {len(sel)} RT-canon {mmm(per, 3)} rungs {d1:.2f}/{d2:.2f}/{d3:.2f} seeds >= d3: {sum(p >= d3 for p in per)}/10")
# all k >= 4 training-world instances
for kk in (4, 5, 6):
    sel_u = [i for i in F if M5[i]["k"] == kk and not F[i][0]]; sel_s = [i for i in F if M5[i]["k"] == kk and F[i][0]]
    print(f"  k={kk}: pair unseen n {len(sel_u)} RT-canon {okc5[sel_u].mean():.3f} d3 {lad(M5, sel_u, 3):.2f} | pair seen n {len(sel_s)} RT-canon {okc5[sel_s].mean() if sel_s else float('nan'):.3f}")

# evalk78
print("\n== evalk78 ({A, C}, k = 7-8) ==")
f78 = [look(m, "AC") for m in M78]
print("  (t,s) training instance:", sum(f[0] for f in f78), "| canon row:", sum(f[1] for f in f78), "| goal seen:", sum(f[3] for f in f78), "| start seen:", sum(f[4] for f in f78), "of", len(M78))
a = [i for i, f in enumerate(f78) if f[3]]; b = [i for i, f in enumerate(f78) if not f[3]]
print(f"  RT-canon seed-mean: goal seen {okc78[a].mean() if a else float('nan'):.3f} (n {len(a)}); goal unseen {okc78[b].mean() if b else float('nan'):.3f} (n {len(b)})")

# held-out worlds: is the eval (t, s) a training instance / row of some training world (any; contained in W)?
print("\n== eval5, held-out worlds: (t, s) seen in a training world ==")
flags = dv["p2_comb"]
for w in HELD:
    ix = [i for i, m in enumerate(M5) if m["world"] == w]
    anyp = [any(M5[i]["t"] + "|" + M5[i]["s"] in pairs[u] for u in TRAIN) for i in ix]
    anyr = [any(M5[i]["t"] + "|" + M5[i]["s"] in rows_c[u] for u in TRAIN) for i in ix]
    sub = [u for u in TRAIN if set(u) <= set(w)]
    subr = [any(M5[i]["t"] + "|" + M5[i]["s"] in rows_c[u] for u in sub) for i in ix]
    line = f"  {w}: n {len(ix)} pair in any training world {np.mean(anyp):.3f}; canon row in any {np.mean(anyr):.3f}; canon row in a contained training world {np.mean(subr):.3f}"
    if w in flags:
        cix = [j for j, i in enumerate(ix) if M5[i]["binding"] and flags[w][M5[i]["t"] + "|" + M5[i]["s"]].get("comb")]
        line += f" | needs-comb n {len(cix)}: canon row in any training world {np.mean([anyr[j] for j in cix]) if cix else float('nan'):.3f}"
    print(line)

# ---------- the strata of P6 / P7 / P8 and the "learned" rates, with verbatim training instances removed
print("\n== training-world strata without the eval instances that are verbatim training instances / rows ==")
tb = [i for i in F if M5[i]["binding"]]
print(f"training binding stratum (P7, P8) n {len(tb)}: (t,s) a training instance {np.mean([F[i][0] for i in tb]):.3f}; an RT-canon row {np.mean([F[i][1] for i in tb]):.3f}")
RSH, REX = seeds(D, "RT-shuffled"), seeds(D, "RT-examples")
for lab, sel in (("all", tb), ("pair unseen", [i for i in tb if not F[i][0]]), ("row unseen", [i for i in tb if not F[i][1]]), ("goal unseen", [i for i in tb if not F[i][3]])):
    for rd in ("ok_m", "ok_u"):
        m = med([float(r["out"]["eval5"][rd][sel].mean()) for r in RC]); sh = [float(r["out"]["eval5"][rd][sel].mean()) for r in RSH]
        ex = [float(r["out"]["eval5"][rd][sel].mean()) for r in REX]
        print(f"  {lab} (n {len(sel)}) {rd}: RT-canon median {m:.3f}; RT-shuffled {[round(x, 3) for x in sh]} gaps {[round(m - x, 3) for x in sh]}; RT-examples {[round(x, 3) for x in ex]}")
p6 = [i for i in F if M5[i]["n_classes"] > 1 and len(M5[i]["target_first"]) > 1]
print(f"P6's multi-class training instances n {len(p6)}: (t,s) a training instance {np.mean([F[i][0] for i in p6]):.3f}; an RT-post row {np.mean([F[i][2] for i in p6]):.3f}")
allix = list(F)
for arm in ARMS:
    rs = seeds(D, arm)
    un = [i for i in allix if not F[i][0]]; se = [i for i in allix if F[i][0]]
    print(f"  {arm}: training worlds eval5 masked, pair seen (n {len(se)}) {mmm([float(r['out']['eval5']['ok_m'][se].mean()) for r in rs], 3)} | pair unseen (n {len(un)}) {mmm([float(r['out']['eval5']['ok_m'][un].mean()) for r in rs], 3)}")
# size of t among seen / unseen
from topology_design.rewrite.term import size as tsize
print("  share of eval5 training-world instances that are verbatim training instances, by size of t:", {s_: round(float(np.mean([F[i][0] for i in allix if M5[i]["size"] == s_])), 3) for s_ in range(1, 6)},
      "n", {s_: sum(1 for i in allix if M5[i]["size"] == s_) for s_ in range(1, 6)})

# ---------- held-out worlds: the complement of P2's strata, split by whether (t, s) is a teacher-forced row of a contained training world
print("\n== held-out binding instances that need no unseen combination: verbatim rows of a contained training world ==")
okh = np.mean([r["out"]["eval5"]["ok_m"] for r in RC], 0)
for nm, WS in (("pairs", PAIRS), ("triples", TRIPLES)):
    ix = [i for i, m in enumerate(M5) if m["world"] in WS and m["binding"] and not flags[m["world"]][m["t"] + "|" + m["s"]].get("comb")]
    seen = [i for i in ix if any(M5[i]["t"] + "|" + M5[i]["s"] in rows_c[u] for u in TRAIN if set(u) <= set(M5[i]["world"]))]
    uns = [i for i in ix if i not in set(seen)]
    print(f"  {nm} complement n {len(ix)}: verbatim RT-canon row of a contained training world {len(seen)} ({len(seen) / len(ix):.2f}); RT-canon seed-mean on those {okh[seen].mean():.3f}, on the rest {okh[uns].mean():.3f}")
for nm, WS in (("pairs", PAIRS), ("triples", TRIPLES), ("full", FULL)):
    ix = [i for i, m in enumerate(M5) if m["world"] in WS]
    seen = [i for i in ix if any(M5[i]["t"] + "|" + M5[i]["s"] in rows_c[u] for u in TRAIN if set(u) <= set(M5[i]["world"]))]
    uns = [i for i in ix if i not in set(seen)]
    print(f"  {nm}, all instances n {len(ix)}: verbatim row of a contained training world {len(seen) / len(ix):.2f}; RT-canon on those {okh[seen].mean():.3f}, on the rest {okh[uns].mean():.3f}")
