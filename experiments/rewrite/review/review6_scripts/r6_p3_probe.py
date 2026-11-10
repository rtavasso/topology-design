"""Review 6, part C — sizes 6-7: is the failure where RESULTS.md says it is, and is it an evaluation bug?

    PYTHONPATH=src python3 experiments/rewrite/review/review6_scripts/r6_p3_probe.py > /tmp/r6_p3.txt

(1) integrity of the eval7 instances against the executor (my own BFS with rules.step / valid_actions): k, cap, the
    first-action targets, action-slot range at 13 nodes; (2) success against nesting depth with size, k and world held
    fixed; (3) where the wrong halts are; (4) which features are actually out of the training range (model.py clamps)."""
import sys, os, gzip, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *
from topology_design.rewrite.term import parse, to_str, nodes, size as tsize, depth as tdep
from topology_design.rewrite.rules import valid_actions, step, world_from_name, RULES
from topology_design.rewrite.model import MAX_NODES, N_RULES, HALT_INDEX, TRAIN_MAX_DEPTH, TRAIN_MAX_SIZE, encode, action_slot

D = load()
M5, M7 = D["meta"]["eval5"], D["meta"]["eval7"]
RC, RA, RP = seeds(D, "RT-canon"), seeds(D, "RT-all"), seeds(D, "RT-post")

# ---------- (1) integrity of the eval7 file against the executor
print("== (1) eval7 integrity (training worlds) ==")
bad = collections.Counter(); caps = collections.Counter(); nn = collections.Counter()
recs = {}
for w in TRAIN:
    with gzip.open(os.path.join(ROOT, "data", "rewrite", f"eval7_{w}.jsonl.gz"), "rt") as f:
        recs[w] = [json.loads(l) for l in f]
pos = collections.Counter()
for w in TRAIN:
    mw = [m for m in M7 if m["world"] == w]
    assert len(mw) == len(recs[w])
    for m, r in zip(mw, recs[w]):
        assert (m["t"], m["s"], m["k"]) == (r["t"], r["s"], r["k"]), "order of per_instance differs from the file"
        caps[r["cap"]] += 1
        W = world_from_name(w); t, s = parse(r["t"]), parse(r["s"])
        nn[len(nodes(t))] += 1
        # BFS distance in W with cap 7 (own code), up to k
        front, seen, d = {t}, {t}, 0
        while s not in seen and d < r["k"]:
            nxt = set()
            for u in front:
                for _, _, v in valid_actions(u, W, r["cap"]):
                    if v not in seen:
                        seen.add(v); nxt.add(v)
            front = nxt; d += 1
        if not (s in seen and d == r["k"]):
            bad["distance != k"] += 1
        # the stored first actions: each valid and leads to a state at distance k-1 (checked for k = 1: leads to s)
        if r["k"] == 1:
            for a in m["target_first"]:
                rule, path = a.split("@")
                v = step(t, rule, path, W, r["cap"])
                if v != s:
                    bad["k=1 target action does not reach s"] += 1
            allv = {f"{ru}@{pa}" for ru, pa, v in valid_actions(t, W, r["cap"]) if v == s}
            if allv != set(m["target_first"]):
                bad["k=1 target set != set of valid actions reaching s"] += 1
        pi = {p: i for i, (p, _, _) in enumerate(nodes(t))}
        for a in m["target_first"]:
            rule, path = a.split("@")
            sl = action_slot((rule, path), pi)
            assert 0 <= sl < HALT_INDEX
            pos[pi[path]] += 1
        e = encode(w, r["s"], r["t"])
        assert len(e["term_paths"]) <= MAX_NODES and len(e["tok"]) <= 128
print("caps:", dict(caps), "| node counts of t:", dict(sorted(nn.items())), "| problems:", dict(bad) or "none")
print("prefix positions of correct first actions:", dict(sorted(pos.items())))
print("horizon: T = 2k+2 for the router (train.rollout) and for the ladder (data.complete_instance) - same instances, same T (code read).")

# ---------- (2) depth, with size / k / world fixed
print("\n== (2) RT-canon masked success at sizes 6-7 by nesting depth, within (size, k) and within (world, size, k) ==")
tr7 = [i for i, m in enumerate(M7) if m["world"] in TRAIN]
dep = {i: max(tdepth(M7[i]["t"]), tdepth(M7[i]["s"])) for i in tr7}
okc = np.mean([r["out"]["eval7"]["ok_m"] for r in RC], 0)          # seed-mean success per instance
oka = np.mean([r["out"]["eval7"]["ok_m"] for r in RA], 0)
for size in (6, 7):
    for k in (1, 2, 3, 4):
        row = []
        for d in (3, 4, 5, 6):
            ix = [i for i in tr7 if M7[i]["size"] == size and M7[i]["k"] == k and dep[i] == d]
            row.append(f"d{d}: {okc[ix].mean():.2f} (n {len(ix)})" if ix else f"d{d}: -")
        print(f"  size {size} k {k}: " + " | ".join(row))
# stratified (Mantel-Haenszel style) contrast: depth >= 5 minus depth <= 4 within (world, size, k) cells
for nm, ok in (("RT-canon", okc), ("RT-all", oka)):
    num = den = 0.0; cells = 0
    for w in TRAIN:
        for size in (6, 7):
            for k in (1, 2, 3, 4):
                lo = [i for i in tr7 if M7[i]["world"] == w and M7[i]["size"] == size and M7[i]["k"] == k and dep[i] <= 4]
                hi = [i for i in tr7 if M7[i]["world"] == w and M7[i]["size"] == size and M7[i]["k"] == k and dep[i] >= 5]
                if len(lo) >= 5 and len(hi) >= 5:
                    wgt = len(lo) * len(hi) / (len(lo) + len(hi)); num += wgt * (ok[hi].mean() - ok[lo].mean()); den += wgt; cells += 1
    print(f"  {nm}: depth>=5 minus depth<=4, weighted within (world, size, k): {num / den:+.3f} over {cells} cells")
# per seed sign of the within-cell contrast
sg = []
for r in RC:
    o = r["out"]["eval7"]["ok_m"].astype(float); num = den = 0.0
    for w in TRAIN:
        for size in (6, 7):
            for k in (1, 2, 3, 4):
                lo = [i for i in tr7 if M7[i]["world"] == w and M7[i]["size"] == size and M7[i]["k"] == k and dep[i] <= 4]
                hi = [i for i in tr7 if M7[i]["world"] == w and M7[i]["size"] == size and M7[i]["k"] == k and dep[i] >= 5]
                if len(lo) >= 5 and len(hi) >= 5:
                    wgt = len(lo) * len(hi) / (len(lo) + len(hi)); num += wgt * (o[hi].mean() - o[lo].mean()); den += wgt
    sg.append(num / den)
print("  per RT-canon seed:", " ".join(f"{x:+.2f}" for x in sg))
print("  by world (k=1, sizes 6-7): depth<=4 vs depth>=5, seed-mean")
for w in TRAIN:
    lo = [i for i in tr7 if M7[i]["world"] == w and M7[i]["k"] == 1 and dep[i] <= 4]; hi = [i for i in tr7 if M7[i]["world"] == w and M7[i]["k"] == 1 and dep[i] >= 5]
    print(f"    {w}: {okc[lo].mean() if lo else float('nan'):.2f} (n {len(lo)}) vs {okc[hi].mean() if hi else float('nan'):.2f} (n {len(hi)})")

# ---------- (3) k = 1: where is the required action, and where are the wrong halts?
print("\n== (3) k = 1 at sizes 6-7 (training worlds): by the depth of the node the (canonical) correct action rewrites ==")
k1 = [i for i in tr7 if M7[i]["k"] == 1]
wh = np.mean([(r["out"]["eval7"]["cat_m"] == 2) for r in RC], 0)   # seed-mean wrong-halt indicator
hz = np.mean([(r["out"]["eval7"]["cat_m"] == 1) for r in RC], 0)
def act_depth(m):
    return min(len(a.split("@")[1]) for a in m["target_first"])
def act_rule(m):
    return sorted(m["target_first"])[0].split("@")[0]
print("  action depth | n | ok | wrong halt | horizon   (seed-mean, masked)")
for d in range(0, 6):
    ix = [i for i in k1 if act_depth(M7[i]) == d]
    if ix: print(f"   {d} | {len(ix)} | {okc[ix].mean():.2f} | {wh[ix].mean():.2f} | {hz[ix].mean():.2f}")
print("  term depth x action depth (ok / wrong halt, n):")
for td in (3, 4, 5, 6):
    row = []
    for ad in range(0, 6):
        ix = [i for i in k1 if dep[i] == td and act_depth(M7[i]) == ad]
        row.append(f"a{ad}: {okc[ix].mean():.2f}/{wh[ix].mean():.2f} ({len(ix)})" if len(ix) >= 5 else f"a{ad}: -")
    print(f"   term depth {td}: " + " | ".join(row))
print("  by rule of the correct action (k=1): n | ok | wrong halt")
for ru in RULES:
    ix = [i for i in k1 if any(a.split('@')[0] == ru for a in M7[i]["target_first"])]
    if ix: print(f"   {ru}: {len(ix)} | {okc[ix].mean():.2f} | {wh[ix].mean():.2f}")
print("  by size change (size s - size t), k=1: n | ok | wrong halt")
for ds in (-3, -2, -1, 0, 1, 2, 3):
    ix = [i for i in k1 if tsize(parse(M7[i]["s"])) - M7[i]["size"] == ds]
    if ix: print(f"   {ds:+d}: {len(ix)} | {okc[ix].mean():.2f} | {wh[ix].mean():.2f}")
print("  wrong-halt share by term depth, all k, RT-canon / RT-all / RT-post (seed-mean):")
wha = np.mean([(r["out"]["eval7"]["cat_m"] == 2) for r in RA], 0); whp = np.mean([(r["out"]["eval7"]["cat_m"] == 2) for r in RP], 0)
for td in (3, 4, 5, 6):
    ix = [i for i in tr7 if dep[i] == td]
    print(f"   depth {td}: {wh[ix].mean():.2f} / {wha[ix].mean():.2f} / {whp[ix].mean():.2f} (n {len(ix)})")
print("  wrong halts at size <= 5 (eval5, training worlds), RT-canon pooled:", round(float(np.mean([(r['out']['eval5']['cat_m'][[i for i, m in enumerate(M5) if m['world'] in TRAIN]] == 2).mean() for r in RC])), 4))
# first decision at k=1: is the first action right but the episode lost later, or the first action wrong?
fo = np.mean([r["out"]["eval7"]["first_ok"] for r in RC], 0)
print(f"  k=1: first decision correct (masked) {fo[k1].mean():.2f}; ok {okc[k1].mean():.2f}; so {fo[k1].mean() - okc[k1].mean():+.2f} of episodes take the right step and then fail to halt at the goal within T=4 (or succeed by another route)")
st = np.concatenate([r["out"]["eval7"]["steps_m"][k1][r["out"]["eval7"]["cat_m"][k1] == 2] for r in RC])
print("  wrong halts at k=1 by the step at which the halt was taken:", dict(sorted(collections.Counter(st.tolist()).items())))

# ---------- (4) what is out of the training range at sizes 6-7 (model.py)
print("\n== (4) features outside the size<=5 range at sizes 6-7 (structural router), training worlds eval7, k=1 ==")
def flags_of(m):
    out = set()
    for which in ("t", "s"):
        for path, u, d in nodes(parse(m[which])):
            sz = tsize(u)
            if d > TRAIN_MAX_DEPTH: out.add("node deeper than 4 (depth and path level clamped)")
            if sz > TRAIN_MAX_SIZE: out.add("subtree > 5 leaves (size clamped)")
            if d == TRAIN_MAX_DEPTH and not isinstance(u, str): out.add("internal node at depth 4 (unseen token x depth combination)")
    return out
cnt = collections.Counter(); tot = 0
none_ix = []
for i in k1:
    f = flags_of(M7[i]); tot += 1
    for x in f: cnt[x] += 1
    if f == {"subtree > 5 leaves (size clamped)"}: none_ix.append(i)
for k_, v in cnt.items(): print(f"   {k_}: {v}/{tot}")
print(f"   only the size clamp applies (depth <= 4, no internal node at depth 4): n {len(none_ix)}; RT-canon ok {okc[none_ix].mean():.2f}, wrong halt {wh[none_ix].mean():.2f}")
dd = [i for i in k1 if dep[i] <= 4 and i not in set(none_ix)]
print(f"   depth <= 4 but with an internal node at depth 4: n {len(dd)}; ok {okc[dd].mean() if dd else float('nan'):.2f}")
# size-5 terms inside eval7 episodes cannot be isolated (t is 6-7), but eval5 size-5, depth-4 terms can: the in-range baseline
for d in (3, 4):
    ix = [i for i, m in enumerate(M5) if m["world"] in TRAIN and m["k"] == 1 and max(tdepth(m["t"]), tdepth(m["s"])) == d]
    print(f"   baseline eval5 k=1 depth {d}: n {len(ix)} ok {np.mean([r['out']['eval5']['ok_m'][ix].mean() for r in RC]):.3f}")
# sequence lengths
L5 = [len(encode(m['world'], m['s'], m['t'])['tok']) for m in M5 if m['world'] in TRAIN]; L7 = [len(encode(M7[i]['world'], M7[i]['s'], M7[i]['t'])['tok']) for i in tr7]
print(f"   sequence length: eval5 max {max(L5)}, eval7 min {min(L7)} max {max(L7)}; SEP index eval5 max {max(4 + len(nodes(parse(m['s']))) for m in M5 if m['world'] in TRAIN)}, eval7 range {min(4 + len(nodes(parse(M7[i]['s']))) for i in tr7)}-{max(4 + len(nodes(parse(M7[i]['s']))) for i in tr7)}")

# ---------- (5) k = 1 cross-tabs: action depth x size change; the in-range subset; RT-absolute's position ranges
print("\n== (5) k=1 cross-tabs (RT-canon seed-mean ok / wrong halt (n)) ==")
def dsz(i): return tsize(parse(M7[i]["s"])) - M7[i]["size"]
for lab, pred in (("size change +1 (IbR/IbL/Db)", lambda i: dsz(i) == 1), ("size change 0 (A/C)", lambda i: dsz(i) == 0), ("size change -1 (If/Df)", lambda i: dsz(i) < 0)):
    row = []
    for ad in range(0, 6):
        ix = [i for i in k1 if pred(i) and act_depth(M7[i]) == ad]
        row.append(f"a{ad}: {okc[ix].mean():.2f}/{wh[ix].mean():.2f} ({len(ix)})" if len(ix) >= 5 else f"a{ad}: - ({len(ix)})")
    print(f"  {lab}: " + " | ".join(row))
print("  in-range-depth subset (no node deeper than 4 in t or s), k=1, by size change:")
for ds in (-1, 0, 1):
    ix = [i for i in none_ix if (dsz(i) == ds if ds >= 0 else dsz(i) < 0)]
    print(f"    {ds:+d}: n {len(ix)} ok {okc[ix].mean():.2f} wrong halt {wh[ix].mean():.2f} | RT-all {oka[ix].mean():.2f}")
print("  same, all k (no node deeper than 4): by k")
inr = [i for i in tr7 if dep[i] <= 4]
for k in (1, 2, 3, 4):
    ix = [i for i in inr if M7[i]["k"] == k]
    e5 = [i for i, m in enumerate(M5) if m["world"] in TRAIN and m["k"] == k]
    print(f"    k={k}: n {len(ix)} RT-canon {okc[ix].mean():.2f} (per-seed min {min(r['out']['eval7']['ok_m'][ix].mean() for r in RC):.2f} max {max(r['out']['eval7']['ok_m'][ix].mean() for r in RC):.2f}) greedy {lad(M7, ix, 1):.2f} | eval5 same k {np.mean([r['out']['eval5']['ok_m'][e5].mean() for r in RC]):.2f}")
# per-seed: k=1, action depth <=3 vs >=4
print("  per RT-canon seed, k=1: action depth <= 3 | >= 4 :", " ".join(f"{r['out']['eval7']['ok_m'][[i for i in k1 if act_depth(M7[i]) <= 3]].mean():.2f}|{r['out']['eval7']['ok_m'][[i for i in k1 if act_depth(M7[i]) >= 4]].mean():.2f}" for r in RC))
print("  RT-all seed-mean, k=1: action depth <= 3 | >= 4 :", f"{oka[[i for i in k1 if act_depth(M7[i]) <= 3]].mean():.2f} | {oka[[i for i in k1 if act_depth(M7[i]) >= 4]].mean():.2f}")
# RT-absolute: is any eval7 input inside the training range of its (unclamped, sinusoidal) positions?
tl = collections.Counter(len(nodes(parse(m["t"]))) for m in M5 if m["world"] in TRAIN)
print("  RT-absolute: term node counts at size<=5 (eval5):", dict(sorted(tl.items())), "; at sizes 6-7:", dict(sorted(collections.Counter(len(nodes(parse(M7[i]['t']))) for i in tr7).items())),
      "-> every eval7 start term is longer (11 or 13 prefix tokens) than any term the control was trained on (<= 9); its positions are sinusoidal and NOT clamped (model.py Router.encode, else-branch)")

# ---------- (6) the three-way split at k = 1, per arm, with per-seed ranges
print("\n== (6) k=1, sizes 6-7, training worlds: three-way split (masked; median [min, max] over seeds) ==")
grp = {"edit at depth <= 3, no growth": [i for i in k1 if act_depth(M7[i]) <= 3 and dsz(i) <= 0],
       "edit at depth <= 3, growth (+1 leaf)": [i for i in k1 if act_depth(M7[i]) <= 3 and dsz(i) == 1],
       "edit at depth >= 4": [i for i in k1 if act_depth(M7[i]) >= 4],
       "no node deeper than 4, no growth": [i for i in none_ix if dsz(i) <= 0],
       "no node deeper than 4, growth": [i for i in none_ix if dsz(i) == 1]}
for arm in ("RT-canon", "RT-all", "RT-post"):
    for nm, ix in grp.items():
        rs = seeds(D, arm)
        print(f"  {arm:9s} {nm:38s} n {len(ix):4d}: ok {mmm([float(r['out']['eval7']['ok_m'][ix].mean()) for r in rs], 2)} | wrong halt {mmm([float((r['out']['eval7']['cat_m'][ix] == 2).mean()) for r in rs], 2)}")
print("  share of the k=1 eval7 set:", {nm: round(len(ix) / len(k1), 2) for nm, ix in grp.items()})
fails = 1 - okc[k1]
print("  share of RT-canon's k=1 failures (seed-mean) in: growth", round(float(fails[[j for j, i in enumerate(k1) if dsz(i) == 1]].sum() / fails.sum()), 2), "; edit at depth >= 4", round(float(fails[[j for j, i in enumerate(k1) if act_depth(M7[i]) >= 4]].sum() / fails.sum()), 2),
      "; either", round(float(fails[[j for j, i in enumerate(k1) if act_depth(M7[i]) >= 4 or dsz(i) == 1]].sum() / fails.sum()), 2))

# ---------- (7) the same split at k = 2..4 (endpoints only: intermediate states are not in the records)
print("\n== (7) k = 2..4, sizes 6-7, training worlds: endpoints with no node deeper than 4, split by size(s) - size(t) (masked) ==")
for k in (2, 3, 4):
    for lab, pred in (("size(s) <= size(t)", lambda i: dsz(i) <= 0), ("size(s) > size(t)", lambda i: dsz(i) > 0)):
        ix = [i for i in tr7 if M7[i]["k"] == k and dep[i] <= 4 and pred(i)]
        dx = [i for i in tr7 if M7[i]["k"] == k and dep[i] >= 5 and pred(i)]
        e5 = [i for i, m in enumerate(M5) if m["world"] in TRAIN and m["k"] == k]
        print(f"  k={k} {lab}: depth<=4 n {len(ix)} RT-canon {mmm([float(r['out']['eval7']['ok_m'][ix].mean()) for r in RC], 2)} greedy {lad(M7, ix, 1):.2f} d3 {lad(M7, ix, 3):.2f} | depth>=5 n {len(dx)} RT-canon {mmm([float(r['out']['eval7']['ok_m'][dx].mean()) for r in RC], 2)} greedy {lad(M7, dx, 1):.2f}")
# the k=1 clean subset as a matched ratio against eval5 k=1 (training worlds)
e5k1 = [i for i, m in enumerate(M5) if m["world"] in TRAIN and m["k"] == 1]
clean = grp["no node deeper than 4, no growth"]
print("  k=1 clean subset / eval5 k=1 success, per RT-canon seed:", " ".join(f"{r['out']['eval7']['ok_m'][clean].mean() / r['out']['eval5']['ok_m'][e5k1].mean():.3f}" for r in RC))
print("  worlds in the clean subset:", dict(collections.Counter(M7[i]["world"] for i in clean)), "| rules:", dict(collections.Counter(act_rule(M7[i]) for i in clean)), "| sizes:", dict(collections.Counter(M7[i]["size"] for i in clean)))
