"""Review 6, part A — independent recomputation of the post hoc companions (main_comb_by_arm.md, main_extra.md) and
the fragility checks on P2's count. Own code; analyze.py is not imported.

    python3 experiments/rewrite/review/review6_scripts/r6_posthoc.py > /tmp/r6_posthoc.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *

D = load(); dv = derived()
M5, M7 = D["meta"]["eval5"], D["meta"]["eval7"]
key = lambda m: f"{m['t']}|{m['s']}"
flags = dv["p2_comb"]
def comb_ix(worlds, want=True):
    return idx(M5, lambda m: m["world"] in worlds and m["binding"] and bool(flags[m["world"]][key(m)].get("comb")) == want)

print("===== needs-the-combination vs complement, every arm (median [min,max]) =====")
for nm, W in (("pairs", PAIRS), ("triples", TRIPLES)):
    for want in (True, False):
        ix = comb_ix(W, want)
        tw = np.mean([bool(flags[M5[i]["world"]][key(M5[i])]["trainw"]) for i in ix]); gl = np.mean([bool(flags[M5[i]["world"]][key(M5[i])].get("glue")) for i in ix])
        print(f"{nm} {'needs-comb' if want else 'complement'} n={len(ix)} greedy {lad(M5, ix, 1):.3f} d3 {lad(M5, ix, 3):.3f} trainw {tw:.3f} glue {gl:.3f}")
        for arm in ARMS:
            rs = seeds(D, arm)
            print(f"   {arm:12s} masked {mmm([rate(r, 'eval5', ix) for r in rs], 2)} | unmasked {mmm([rate(r, 'eval5', ix, 'ok_u') for r in rs], 2)} | per-seed masked {[round(rate(r, 'eval5', ix), 3) for r in rs]}")

print("\n===== P2 fragility =====")
ixp = comb_ix(PAIRS); n = len(ixp)
RC = seeds(D, "RT-canon")
ok = np.stack([r["out"]["eval5"]["ok_m"][ixp] for r in RC]).astype(float)        # seeds x instances
tw = np.array([bool(flags[M5[i]["world"]][key(M5[i])]["trainw"]) for i in ixp], float)
gr = np.array([M5[i]["ladder"]["1"] for i in ixp], float)
print("per-seed minus floor, in instances:", (ok.sum(1) - tw.sum()).astype(int).tolist(), " binomial SE per seed (instances):", [round(float(np.sqrt(n * p * (1 - p))), 1) for p in ok.mean(1)])
rng = np.random.default_rng(1)
cnts, cnts_fixed = [], []
for _ in range(20000):
    b = rng.integers(0, n, n)
    fl = max(tw[b].mean(), gr[b].mean())
    cnts.append(int((ok[:, b].mean(1) > fl + 1e-9).sum()))
    cnts_fixed.append(int((ok[:, b].mean(1) > 0.5311 + 1e-9).sum()))
cnts, cnts_fixed = np.array(cnts), np.array(cnts_fixed)
print("joint instance bootstrap (floor resampled with the instances): distribution of '#seeds above the floor':", {k: round(float((cnts == k).mean()), 3) for k in range(11)})
print("   P(count <= 3) =", round(float((cnts <= 3).mean()), 3), "; P(count >= 6) =", round(float((cnts >= 6).mean()), 3))
print("instance bootstrap with the floor fixed at 0.531: P(count <= 3) =", round(float((cnts_fixed <= 3).mean()), 3))
# paired per-instance difference between the median-ish policy and the floor: seed-mean success vs trainw
pm = ok.mean(0)
diff = pm - tw
bs = [diff[rng.integers(0, n, n)].mean() for _ in range(10000)]
print(f"seed-mean RT-canon minus floor on the pairs' stratum: {diff.mean():+.3f}  instance-bootstrap 95% [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")
# seed bootstrap of the median
meds = [np.median(rng.choice(ok.mean(1), 10)) for _ in range(10000)]
print(f"seed-bootstrap of the seed-median: 95% [{np.quantile(meds, .025):.3f}, {np.quantile(meds, .975):.3f}] (floor 0.531)")
# alternative floors from review 5 / DESIGN §17a.2: 0.354, 0.48-0.54, 0.727, 0.751
for fl in (0.354, 0.397, 0.48, 0.531, 0.54, 0.727, 0.751):
    print(f"   seeds above an alternative non-composing floor {fl}: {(ok.mean(1) > fl).sum()}/10")
# overlap of solved instances between RT-canon and the training-world reference
print("share of trainw-solved instances solved by RT-canon (seed-mean):", round(float(pm[tw == 1].mean()), 3), "; of trainw-unsolved:", round(float(pm[tw == 0].mean()), 3), f"(n {int(tw.sum())} / {int((1 - tw).sum())})")
gl = np.array([bool(flags[M5[i]["world"]][key(M5[i])].get("glue")) for i in ixp], float)
print("   ... of glue-solved:", round(float(pm[gl == 1].mean()), 3), "; glue-unsolved:", round(float(pm[gl == 0].mean()), 3))
for arm in ("RT-all", "RT-post", "RT-shuffled", "EI16"):
    o = np.stack([r["out"]["eval5"]["ok_m"][ixp] for r in seeds(D, arm)]).mean(0)
    print(f"   {arm}: on trainw-solved {o[tw == 1].mean():.3f} trainw-unsolved {o[tw == 0].mean():.3f}; glue-solved {o[gl == 1].mean():.3f} glue-unsolved {o[gl == 0].mean():.3f}")
# by k on the pairs stratum
for k in sorted({M5[i]["k"] for i in ixp}):
    sel = np.array([j for j, i in enumerate(ixp) if M5[i]["k"] == k])
    print(f"   pairs stratum k={k} n={len(sel)} RT-canon seed-mean {pm[sel].mean():.2f} trainw {tw[sel].mean():.2f} greedy {gr[sel].mean():.2f} RT-all {np.stack([r['out']['eval5']['ok_m'][ixp][sel] for r in seeds(D, 'RT-all')]).mean():.2f}")

print("\n===== RT-shuffled vs RT-canon on the needs-the-combination strata =====")
for nm, W in (("pairs", PAIRS), ("triples", TRIPLES)):
    ix = comb_ix(W)
    for rd in ("ok_m", "ok_u"):
        c = sorted(rate(r, "eval5", ix, rd) for r in RC); s = [rate(r, "eval5", ix, rd) for r in seeds(D, "RT-shuffled")]
        print(f"{nm} {rd}: RT-canon sorted {[round(x, 3) for x in c]} ; RT-shuffled {[round(x, 3) for x in s]} ; RT-canon seeds >= min shuffled: {sum(x >= min(s) for x in c)}/10 ; rank-sum exact p (2 vs 10, one-sided shuffled > canon): ", end="")
        from itertools import combinations
        allv = c + s; obs = sum(s); tot = 0; ge = 0
        for cb in combinations(range(12), 2):
            tot += 1; ge += (allv[cb[0]] + allv[cb[1]] >= obs - 1e-12)
        print(f"{ge}/{tot} = {ge / tot:.3f}")

print("\n===== sizes 6-7 (eval7), training worlds =====")
tr7 = idx(M7, lambda m: m["world"] in TRAIN)
for arm in ("RT-canon", "RT-post", "RT-all", "RT-absolute"):
    rs = seeds(D, arm)
    for size in (6, 7):
        ix = idx(M7, lambda m: m["world"] in TRAIN and m["size"] == size)
        print(f"{arm} size {size} n={len(ix)} masked {mmm([rate(r, 'eval7', ix) for r in rs], 2)} unmasked {mmm([rate(r, 'eval7', ix, 'ok_u') for r in rs], 2)}")
for arm in ("RT-canon", "RT-all", "RT-absolute", "RT-post"):
    rs = seeds(D, arm)
    for k in (1, 2, 3, 4):
        ix7 = idx(M7, lambda m: m["world"] in TRAIN and m["k"] == k); ix5 = idx(M5, lambda m: m["world"] in TRAIN and m["k"] == k)
        cm = np.concatenate([r["out"]["eval7"]["cat_m"][ix7] for r in rs]); cu = np.concatenate([r["out"]["eval7"]["cat_u"][ix7] for r in rs])
        print(f"{arm} k={k} n={len(ix7)} 6-7 masked {mmm([rate(r, 'eval7', ix7) for r in rs], 2)} | <=5 masked {mmm([rate(r, 'eval5', ix5) for r in rs], 2)} | greedy {lad(M7, ix7, 1):.2f} d3 {lad(M7, ix7, 3):.2f} | pooled masked cats "
              + " ".join(f"{CATS[c]}:{(cm == c).mean():.2f}" for c in range(6) if (cm == c).any()) + " | unmasked " + " ".join(f"{CATS[c]}:{(cu == c).mean():.2f}" for c in range(6) if (cu == c).any()))
        print(f"      seeds below the greedy rung: {sum(rate(r, 'eval7', ix7) < lad(M7, ix7, 1) for r in rs)}/{len(rs)}; wrong-halt share per seed (masked) {[round(float((r['out']['eval7']['cat_m'][ix7] == 2).mean()), 2) for r in rs]}")
print("by nesting depth of the deeper of t and s (pooled over seeds; n per seed):")
for arm in ("RT-canon", "RT-all"):
    rs = seeds(D, arm)
    for tag, M, ks in (("eval5", M5, (None,)), ("eval7", M7, (1, 2, 3, 4))):
        for k in ks:
            out = []
            for d in range(1, 7):
                ix = idx(M, lambda m: m["world"] in TRAIN and (k is None or m["k"] == k) and max(tdepth(m["t"]), tdepth(m["s"])) == d)
                out.append(f"{np.mean([rate(r, tag, ix) for r in rs]):.2f} ({len(ix)})" if len(ix) else "-")
            print(f"   {arm} {tag} k={k}: " + " | ".join(out))

print("\n===== curves =====")
STEPS = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000]
def series(run, worlds, rd="ok_m", tag="eval5"):
    xs, ys = [], []
    for pt in run["curve"]:
        num = den = 0
        for ck, c in pt.get(tag, {}).items():
            if ck.split("|")[0] in worlds:
                num += c[rd]; den += c["n"]
        xs.append(pt["step"]); ys.append(num / den)
    return xs, ys
print("curve points per run:", sorted({len(r["curve"]) for r in D["runs"].values()}), "; n per curve point (eval5, all worlds) at step 11750 vs 12000:",
      sorted({(sum(c["n"] for c in r["curve"][-2]["eval5"].values()), sum(c["n"] for c in r["curve"][-1]["eval5"].values())) for r in D["runs"].values()}))
for label, W in (("training worlds", TRAIN), ("held-out worlds", HELD)):
    print(label)
    for arm in ARMS:
        rs = seeds(D, arm)
        cols = []
        for s in STEPS:
            vm = [series(r, W)[1][series(r, W)[0].index(s)] for r in rs]; vu = [series(r, W, "ok_u")[1][series(r, W)[0].index(s)] for r in rs]
            cols.append(f"{med(vm):.2f}|{med(vu):.2f}")
        peaks, drops = [], []
        for r in rs:
            xs, ym = series(r, W); i = int(np.argmax(ym)); peaks.append(xs[i]); drops.append(ym[-1] - ym[i])
        print(f"   {arm:12s} " + " ".join(cols) + " | peak step " + " ".join(map(str, peaks)) + " | final-max " + " ".join(f"{d:+.2f}" for d in drops))
