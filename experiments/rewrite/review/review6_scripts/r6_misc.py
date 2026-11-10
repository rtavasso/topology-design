"""Review 6 — remaining checks: provenance numbers, P6's post hoc statistics, RT-post / other arms under P2's rule,
RT-shuffled against RT-canon on all held-out strata, P4 sensitivity, checkpoint-to-checkpoint noise of the held-out rates.

    python3 experiments/rewrite/review/review6_scripts/r6_misc.py > /tmp/r6_misc.txt
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *

D = load(); dv = derived()
M5, M7 = D["meta"]["eval5"], D["meta"]["eval7"]
key = lambda m: f"{m['t']}|{m['s']}"
flags = dv["p2_comb"]
RC = seeds(D, "RT-canon")

print("== provenance ==")
tot = sum(r["seconds"] for r in D["runs"].values())
print(f"sum of run seconds: {tot / 3600:.1f} h; EI16 minutes: {[round(r['seconds'] / 60) for r in seeds(D, 'EI16')]}; shards: {sorted({r['shard'] for r in D['runs'].values()})}")
print("curve losses finite at every point:", all(np.isfinite(pt["loss"]) for r in D["runs"].values() for pt in r["curve"][1:]))
print("training-world final (full eval5) masked, per arm min/max:", {a: (round(min(rate(r, 'eval5', idx(M5, lambda m: m['world'] in TRAIN)) for r in seeds(D, a)), 3), round(max(rate(r, 'eval5', idx(M5, lambda m: m['world'] in TRAIN)) for r in seeds(D, a)), 3)) for a in ARMS})

print("\n== P2's rule applied to every arm (unscored; the rule names RT-canon) ==")
def comb_ix(W): return idx(M5, lambda m: m["world"] in W and m["binding"] and flags[m["world"]][key(m)].get("comb"))
ixp, ixt = comb_ix(PAIRS), comb_ix(TRIPLES)
g = lad(M5, ixp, 1); fl_p = 111 / 209; fl_t = 108 / 509
for arm in ARMS:
    rs = seeds(D, arm)
    for rd in ("ok_m", "ok_u"):
        pr = [rate(r, "eval5", ixp, rd) for r in rs]; tr = [rate(r, "eval5", ixt, rd) for r in rs]
        print(f"  {arm:12s} {rd}: pair conds {sum(x - g >= 0.2 - 1e-9 and x >= 0.7 - 1e-9 for x in pr)}/{len(rs)}; > pairs' floor {sum(x > fl_p + 1e-9 for x in pr)}/{len(rs)}; > triples' floor {sum(x > fl_t + 1e-9 for x in tr)}/{len(rs)}; >= 0.7 on pairs {sum(x >= 0.7 for x in pr)}/{len(rs)}; pairs per seed {[round(x, 3) for x in pr]}")

print("\n== RT-shuffled against RT-canon, masked, every held-out stratum (two shuffled seeds against ten) ==")
strata = {"held-out, all": lambda m: m["world"] in HELD, "pairs binding": lambda m: m["world"] in PAIRS and m["binding"], "pairs non-binding": lambda m: m["world"] in PAIRS and not m["binding"],
          "triples binding": lambda m: m["world"] in TRIPLES and m["binding"], "triples non-binding": lambda m: m["world"] in TRIPLES and not m["binding"], "full world": lambda m: m["world"] in FULL,
          "pairs needs-comb": lambda m: m["world"] in PAIRS and m["binding"] and flags[m["world"]][key(m)].get("comb"), "pairs complement": lambda m: m["world"] in PAIRS and m["binding"] and not flags[m["world"]][key(m)].get("comb"),
          "triples needs-comb": lambda m: m["world"] in TRIPLES and m["binding"] and flags[m["world"]][key(m)].get("comb"), "triples complement": lambda m: m["world"] in TRIPLES and m["binding"] and not flags[m["world"]][key(m)].get("comb"),
          "training, all": lambda m: m["world"] in TRAIN, "training binding": lambda m: m["world"] in TRAIN and m["binding"]}
for nm, pred in strata.items():
    ix = idx(M5, pred)
    for rd in ("ok_m", "ok_u"):
        c = sorted(rate(r, "eval5", ix, rd) for r in RC); s = [rate(r, "eval5", ix, rd) for r in seeds(D, "RT-shuffled")]
        allv = c + s; obs = sum(s); ge = sum(allv[a] + allv[b] >= obs - 1e-12 for a, b in itertools.combinations(range(12), 2))
        print(f"  {nm:20s} n {len(ix):5d} {rd}: RT-canon {med(c):.3f} [{c[0]:.3f}, {c[-1]:.3f}] | RT-shuffled {s[0]:.3f} {s[1]:.3f} | RT-canon seeds below both shuffled seeds: {sum(x < min(s) for x in c)}/10 | exact rank-sum P(shuffled this high) = {ge}/66")
for w in HELD:
    ix = idx(M5, lambda m: m["world"] == w)
    c = sorted(rate(r, "eval5", ix) for r in RC); s = [rate(r, "eval5", ix) for r in seeds(D, "RT-shuffled")]
    print(f"     world {w}: RT-canon {med(c):.3f} [{c[0]:.3f}, {c[-1]:.3f}] | RT-shuffled {s[0]:.3f} {s[1]:.3f} | RT-all {med([rate(r, 'eval5', ix) for r in seeds(D, 'RT-all')]):.3f}")

print("\n== P4 sensitivity ==")
cells = dv["p4_amended_pooled"]["cells"]
def cell_of(m): return f"{m['world']}|{int(m['binding'])}|{int(m['GP1'])}|{m['k']}"
for drop in (None, "ID|1|1|4", "ID|0|0|4"):
    ix = idx(M5, lambda m: cell_of(m) in cells and cell_of(m) != drop)
    pm = [rate(r, "eval5", ix) for r in RC]; d2, d3 = lad(M5, ix, 2), lad(M5, ix, 3)
    print(f"  without {drop}: n {len(ix)} rungs {lad(M5, ix, 1):.3f}/{d2:.3f}/{d3:.3f} RT-canon {mmm(pm)} >= d3 {sum(x >= d3 for x in pm)}/10")
# lowest RT-canon seed against d3 with a binomial interval
ix = idx(M5, lambda m: cell_of(m) in cells)
print("  lowest seed 0.882 = 150/170 against d3 = 123/170; paired per-instance: instances where lookahead-3 succeeds and RT-canon (seed-mean) fails vs the reverse:")
okc = np.mean([r["out"]["eval5"]["ok_m"][ix] for r in RC], 0); l3 = np.array([M5[i]["ladder"]["3"] for i in ix])
print(f"     RT-canon seed-mean on lookahead-3 successes {okc[l3].mean():.3f} (n {l3.sum()}); on lookahead-3 failures {okc[~l3].mean():.3f} (n {(~l3).sum()})")

print("\n== P6 post hoc ==")
def p6(run):
    pf = run["out"]["eval5"]["p_first"]
    tv, tu, dm, dt, closer = [], [], [], [], []
    for i, m in enumerate(M5):
        p = pf[i]
        if m["world"] in TRAIN and p and m["n_classes"] > 1 and len(p) > 1:
            tg = m["target_first"]; tot = sum(p.values()); q = {a: v / tot for a, v in p.items()}; u = 1 / len(p)
            a_ = 0.5 * sum(abs(q.get(x, 0) - tg.get(x, 0)) for x in set(q) | set(tg)); b_ = 0.5 * sum(abs((u if x in p else 0) - tg.get(x, 0)) for x in set(p) | set(tg))
            tv.append(a_); tu.append(b_)
            if b_ >= 1e-9:
                closer.append(a_ < b_)
            for x in p:
                dm.append(q[x] - u); dt.append(tg.get(x, 0) - u)
    tv, tu = np.array(tv), np.array(tu); non = tu >= 1e-9
    return tv[non].mean(), tu[non].mean(), tv[~non].mean(), float(np.corrcoef(dm, dt)[0, 1]), float(np.mean(closer)), int(non.sum()), int((~non).sum())
for arm in ("RT-post", "PF", "RT-canon", "RT-all"):
    out = [p6(r) for r in seeds(D, arm)]
    print(f"  {arm}: non-uniform-target TV {' '.join(f'{o[0]:.3f}' for o in out)} (uniform ref {out[0][1]:.3f}, n {out[0][5]}); on uniform targets {' '.join(f'{o[2]:.3f}' for o in out)} (n {out[0][6]}); corr {' '.join(f'{o[3]:.2f}' for o in out)}; share of non-uniform instances closer than uniform {' '.join(f'{o[4]:.2f}' for o in out)}")
need = 0.1858 - (1003 / 1210) * 0.08
print(f"  for RT-post to beat the uniform reference overall with its 0.08 error on the uniform-target instances it would need TV <= {need:.3f} on the others (it has 0.150-0.170)")

print("\n== checkpoint-to-checkpoint noise of RT-canon's held-out rates (curve subset; AI+CI binding cells, n per point below) ==")
def ser(run, pred):
    ys = []
    for pt in run["curve"][:-1]:
        num = den = 0
        for ck, c in pt["eval5"].items():
            w, b, g_, k = ck.split("|")
            if pred(w, int(b)):
                num += c["ok_m"]; den += c["n"]
        ys.append(num / den)
    return np.array(ys), den
for nm, pred in (("held-out pairs, binding", lambda w, b: w in PAIRS and b == 1), ("held-out triples, binding", lambda w, b: w in TRIPLES and b == 1), ("training worlds, binding", lambda w, b: w in TRAIN and b == 1)):
    print(f"  {nm}:")
    for r in RC:
        ys, n = ser(r, pred); last = ys[-9:]
        print(f"     s{r['cfg']['seed']}: n {n}; last 9 checkpoints (10,000-11,750 step) min {last.min():.2f} max {last.max():.2f} sd {last.std():.3f} (binomial sd at this n {np.sqrt(last.mean() * (1 - last.mean()) / n):.3f}); 11,750 -> {ys[-1]:.2f}")
    sds = [ser(r, pred)[0][-9:].std() for r in RC]; rng_ = [np.ptp(ser(r, pred)[0][-9:]) for r in RC]
    print(f"     median within-run sd over the last 2,000 steps {np.median(sds):.3f}; median range {np.median(rng_):.2f}; between-seed sd of the final full-set rate {np.std([rate(r, 'eval5', idx(M5, lambda m: pred(m['world'], int(m['binding'])))) for r in RC]):.3f}")
