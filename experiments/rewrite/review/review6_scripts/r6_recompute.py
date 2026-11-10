"""Review 6, part A/B — independent recomputation of every number behind the verdicts of RESULTS.md §1
(P2, P3, P4, P5, P6, P7, P8, P1) from the raw records and derive/derived.json. Own code; analyze.py is not imported.

    python3 experiments/rewrite/review/review6_scripts/r6_recompute.py > /tmp/r6_recompute.txt
"""
import math, random, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *

D = load(); dv = derived()
M5, M7, M78 = D["meta"]["eval5"], D["meta"]["eval7"], D["meta"]["evalk78"]
key = lambda m: f"{m['t']}|{m['s']}"
flags = dv["p2_comb"]
EPS = 1e-9
RC, RA, RP, RAB, REI, RPF, REX, RSH = (seeds(D, a) for a in ["RT-canon", "RT-all", "RT-post", "RT-absolute", "EI16", "PF", "RT-examples", "RT-shuffled"])
print("runs per arm:", {a: len(seeds(D, a)) for a in ARMS})
print("lr per arm:", {a: sorted({r['cfg']['lr'] for r in seeds(D, a)}) for a in ARMS}, "steps", {r['cfg']['steps'] for r in D['runs'].values()})
print("params:", {a: sorted({r['params'] for r in seeds(D, a)}) for a in ARMS})

# ------------------------------------------------------------------ P2
print("\n===== P2 =====")
def comb_ix(worlds, want=True):
    return idx(M5, lambda m: m["world"] in worlds and m["binding"] and bool(flags.get(m["world"], {}).get(key(m), {}).get("comb")) == want)
missing = sum(1 for m in M5 if m["world"] in PAIRS + TRIPLES and m["binding"] and key(m) not in flags.get(m["world"], {}))
print("held-out binding instances with no p2_comb flag:", missing)
for nm, W in (("pairs", PAIRS), ("triples", TRIPLES)):
    ix = comb_ix(W)
    g, d3 = lad(M5, ix, 1), lad(M5, ix, 3)
    tw = np.mean([bool(flags[M5[i]["world"]][key(M5[i])]["trainw"]) for i in ix]); gl = np.mean([bool(flags[M5[i]["world"]][key(M5[i])].get("glue")) for i in ix])
    floor = max(g, tw)
    print(f"{nm}: n={len(ix)} greedy={g:.4f} d2={lad(M5, ix, 2):.4f} d3={d3:.4f} trainw={tw:.4f} glue={gl:.4f} floor={floor:.4f}  (floor*n={floor*len(ix):.2f})")
    for rd in ("ok_m", "ok_u"):
        pr = [rate(r, "eval5", ix, rd) for r in RC]
        print(f"  RT-canon {rd} per seed: " + " ".join(f"{x:.4f}" for x in pr), "| counts", [int(round(x * len(ix))) for x in pr], "|", mmm(pr))
        if nm == "pairs":
            print(f"    pair conds (>= g+0.2 and >= 0.7): {sum(x - g >= 0.2 - EPS and x >= 0.7 - EPS for x in pr)}/10 ; > floor: {sum(x > floor + EPS for x in pr)}/10 ; > greedy: {sum(x > g + EPS for x in pr)}/10 ; >= glue: {sum(x >= gl for x in pr)}/10 ; >= d3: {sum(x >= d3 for x in pr)}/10")
        else:
            print(f"    > floor: {sum(x > floor + EPS for x in pr)}/10 ; >= glue: {sum(x >= gl for x in pr)}/10; >= d3 {sum(x >= d3 for x in pr)}/10")
        pa = [rate(r, "eval5", ix, rd) for r in RA]
        print(f"  RT-all {rd}: " + " ".join(f"{x:.4f}" for x in pa), "median", med(pa), "meets pair conds:", med(pa) - g >= 0.2 - EPS and med(pa) >= 0.7 - EPS)
    # beyond glue
    bx = np.array([i for i in ix if not flags[M5[i]["world"]][key(M5[i])].get("glue")])
    print(f"  beyond glue n={len(bx)} d3={lad(M5, bx, 3):.3f} RT-canon m {mmm([rate(r, 'eval5', bx) for r in RC])} u {mmm([rate(r, 'eval5', bx, 'ok_u') for r in RC])}")
for w in PAIRS + TRIPLES:
    ix = comb_ix([w])
    if len(ix):
        print(f"  world {w} n={len(ix)} m {mmm([rate(r, 'eval5', ix) for r in RC], 2)} u {mmm([rate(r, 'eval5', ix, 'ok_u') for r in RC], 2)} greedy {lad(M5, ix, 1):.3f} d2 {lad(M5, ix, 2):.3f} d3 {lad(M5, ix, 3):.3f}"
              f" trainw {np.mean([bool(flags[w][key(M5[i])]['trainw']) for i in ix]):.3f} glue {np.mean([bool(flags[w][key(M5[i])].get('glue')) for i in ix]):.3f}")
# instance bootstrap envelope (my own seeds; the author's uses random.Random(0) with 1000 draws)
ixp = comb_ix(PAIRS); rng = np.random.default_rng(0)
los, his = [], []
for r in RC:
    v = r["out"]["eval5"]["ok_m"][ixp]
    b = rng.choice(v, size=(4000, len(v))).mean(1)
    los.append(np.quantile(b, 0.025)); his.append(np.quantile(b, 0.975))
print(f"  pairs instance-bootstrap envelope masked [{min(los):.3f}, {max(his):.3f}]")
# §6 preregistered strata
for nm, W in (("pairs binding (§6)", PAIRS), ("triples(+full) binding (§6)", TRIPLES + FULL)):
    ix = idx(M5, lambda m: m["world"] in W and m["binding"])
    g = lad(M5, ix, 1)
    pr = [rate(r, "eval5", ix) for r in RC]
    print(f"  {nm} n={len(ix)} greedy {g:.3f} RT-canon {mmm(pr)} ; >greedy {sum(x > g for x in pr)}/10 ; joint {sum(x - g >= 0.2 - EPS and x >= 0.7 - EPS for x in pr)}/10 ; RT-all {mmm([rate(r, 'eval5', ix) for r in RA])} per seed {[round(rate(r, 'eval5', ix), 4) for r in RA]}")

# ------------------------------------------------------------------ P3
print("\n===== P3 =====")
def matched(run, ix7, ix5, okf7, okf5):
    c7, c5 = {}, {}
    for i in ix7:
        m = M7[i]; c7.setdefault((m["world"], m["k"], bool(m["binding"])), []).append(i)
    for i in ix5:
        m = M5[i]
        if m["k"] <= 4:
            c5.setdefault((m["world"], m["k"], bool(m["binding"])), []).append(i)
    num = den = n = 0.0
    for kk, rows in c7.items():
        if kk not in c5:
            continue
        num += sum(okf7(i) for i in rows); den += len(rows) * np.mean([okf5(i) for i in c5[kk]]); n += len(rows)
    return num / den, den / n, int(n)
def rho_for(base, pts):
    pts = sorted(pts)
    if base <= pts[0][0]: rho = pts[0][1]
    elif base >= pts[-1][0]: rho = pts[-1][1]
    else:
        for (b0, r0), (b1, r1) in zip(pts, pts[1:]):
            if b0 <= base <= b1:
                rho = r0 + (r1 - r0) * (base - b0) / (b1 - b0); break
    return min(rho, 1 / base)
def p3(run, worlds, rd, ix7=None):
    ix7 = idx(M7, lambda m: m["world"] in worlds) if ix7 is None else ix7
    ix5 = idx(M5, lambda m: m["world"] in worlds)
    o7, o5 = run["out"]["eval7"][rd], run["out"]["eval5"][rd]
    R, base, n = matched(run, ix7, ix5, lambda i: o7[i], lambda i: o5[i])
    pts = []
    for d in (1, 2, 3):
        r_, b_, _ = matched(run, ix7, ix5, lambda i: M7[i]["ladder"][str(d)], lambda i: M5[i]["ladder"][str(d)])
        pts.append((b_, r_))
    return R, base, rho_for(base, pts), n, pts
res = [p3(r, TRAIN, "ok_m") for r in RC]
print("ladder matched (base, ratio) d=1,2,3:", [(round(b, 4), round(r, 4)) for b, r in res[0][4]], "n matched", res[0][3])
print("clause1 R   :", " ".join(f"{x[0]:.4f}" for x in res))
print("clause1 base:", " ".join(f"{x[1]:.4f}" for x in res))
print("clause1 rho :", " ".join(f"{x[2]:.4f}" for x in res))
print("R >= 0.9 rho:", sum(x[0] >= 0.9 * x[2] - EPS for x in res), "/10 ; R >= 0.9:", sum(x[0] >= 0.9 for x in res), "/10 ; median R", med([x[0] for x in res]))
for nm, W in (("single-axiom", ["A", "C", "I", "D"]), ("training pairs", ["AC", "AD", "CD", "ID"]), ("held-out", HELD)):
    rr = [p3(r, W, "ok_m") for r in RC]
    print(f"  {nm}: R " + " ".join(f"{x[0]:.2f}" for x in rr) + " | rho " + " ".join(f"{x[2]:.2f}" for x in rr) + " | " + mmm([x[0] for x in rr], 2))
for arm, rs in (("RT-all", RA), ("RT-post", RP), ("RT-absolute", RAB)):
    for nm, W in (("train", TRAIN), ("held-out", HELD)):
        rr = [p3(r, W, "ok_m") for r in rs]
        print(f"  {arm} {nm}: R {mmm([x[0] for x in rr], 2)}; raw eval7 {mmm([rate(r, 'eval7', idx(M7, lambda m: m['world'] in W)) for r in rs], 2)}")
grp = dv["eval7_abs_group"]
gi = {g: idx(M7, lambda m, g=g: m["world"] in TRAIN and grp[m["world"]].get(key(m)) == g) for g in ("mustfail", "detour", "shortest")}
print("clause2 groups n:", {g: len(v) for g, v in gi.items()}, "unflagged:", sum(1 for m in M7 if m["world"] in TRAIN and key(m) not in grp[m["world"]]))
for g in ("mustfail", "detour", "shortest"):
    print(f"  {g}: unmasked " + " ".join(f"{rate(r, 'eval7', gi[g], 'ok_u'):.4f}" for r in RAB) + " | counts " + str([int(r['out']['eval7']['ok_u'][gi[g]].sum()) for r in RAB])
          + " | masked " + " ".join(f"{rate(r, 'eval7', gi[g]):.4f}" for r in RAB) + " | RT-canon masked " + mmm([rate(r, 'eval7', gi[g]) for r in RC], 2) + " RT-canon unmasked " + mmm([rate(r, 'eval7', gi[g], 'ok_u') for r in RC], 2))
rr = [p3(r, TRAIN, "ok_u", gi["shortest"]) for r in RAB]
print("  clause2 R:", " ".join(f"{x[0]:.4f}" for x in rr), "| rho", " ".join(f"{x[2]:.3f}" for x in rr), "| base(u)", " ".join(f"{x[1]:.3f}" for x in rr), "| ok", sum(x[0] >= 0.9 * x[2] - EPS for x in rr), "/5")

# ------------------------------------------------------------------ P4
print("\n===== P4 =====")
def sep_cells(prereg=False, upper_only=False):
    cells = {}
    for i, m in enumerate(M5):
        if m["world"] in TRAIN and m["k"] in (4, 5, 6) and (m["GP1"] or not prereg):
            kk = (m["world"], m["k"]) if prereg else (m["world"], int(bool(m["binding"])), int(bool(m["GP1"])), m["k"])
            cells.setdefault(kk, []).append(i)
    out = {}
    for kk, rows in cells.items():
        if len(rows) < 10: continue
        d1, d2, d3 = (lad(M5, rows, d) for d in (1, 2, 3))
        ok = (d3 - d2 >= 0.1) if upper_only else (d2 - d1 >= 0.1 and d3 - d2 >= 0.1)   # NB: plain float comparison, as written
        okeps = (d3 - d2 >= 0.1 - EPS) if upper_only else (d2 - d1 >= 0.1 - EPS and d3 - d2 >= 0.1 - EPS)
        out[kk] = (rows, d1, d2, d3, ok, okeps)
    return out
c16 = sep_cells()
print("cells with n>=10:", len(c16), "; separated (float >=):", sorted("|".join(map(str, k)) for k, v in c16.items() if v[4]), "; with 1e-9 tolerance:", sorted("|".join(map(str, k)) for k, v in c16.items() if v[5]))
print("derive step's:", sorted(dv["p4_amended_pooled"]["cells"]))
for kk, v in sorted(c16.items()):
    if v[5] or abs(v[2] - v[1] - 0.1) < 0.03 and v[3] - v[2] >= 0.1:
        print(f"   cell {kk} n={len(v[0])} d1={v[1]:.4f} d2={v[2]:.4f} d3={v[3]:.4f} d2-d1={v[2]-v[1]:.4f} d3-d2={v[3]-v[2]:.4f} sep={v[4]}")
ix16 = np.array(sorted(i for v in c16.values() if v[4] for i in v[0]))
d1, d2, d3 = (lad(M5, ix16, d) for d in (1, 2, 3))
print(f"§16 set n={len(ix16)} rungs {d1:.4f} {d2:.4f} {d3:.4f} (d3*n={d3*len(ix16):.1f}); binding share {np.mean([M5[i]['binding'] for i in ix16]):.2f} GP1 share {np.mean([M5[i]['GP1'] for i in ix16]):.2f}")
for arm in ARMS:
    rs = seeds(D, arm)
    pm = [rate(r, "eval5", ix16) for r in rs]; pu = [rate(r, "eval5", ix16, "ok_u") for r in rs]
    print(f"  {arm}: masked {mmm(pm)} per-seed {[round(x, 3) for x in pm]} unmasked {mmm(pu, 2)} | between {sum(d2 < x < d3 for x in pm)} >=d3 {sum(x >= d3 for x in pm)} <=d1 {sum(x <= d1 for x in pm)} of {len(pm)}")
cu = sep_cells(upper_only=True); ixu = np.array(sorted(i for v in cu.values() if v[4] for i in v[0]))
print(f"upper-only: {sum(v[4] for v in cu.values())} cells n={len(ixu)} rungs {lad(M5, ixu, 1):.3f} {lad(M5, ixu, 2):.3f} {lad(M5, ixu, 3):.3f} RT-canon {mmm([rate(r, 'eval5', ixu) for r in RC], 2)}")
c6 = sep_cells(prereg=True); print("§6 GP1 cells n>=10:", len(c6), "separated:", sum(v[4] for v in c6.values()))
# per-cell: RT-canon against d3 on each of the six cells
for kk, v in sorted(c16.items()):
    if v[4]:
        ix = np.array(v[0]); pm = [rate(r, "eval5", ix) for r in RC]
        print(f"   cell {kk} n={len(ix)} rungs {v[1]:.2f}/{v[2]:.2f}/{v[3]:.2f} RT-canon {mmm(pm, 2)} ; seeds >= d3: {sum(x >= v[3] for x in pm)}/10")
print("by k x binding x GP1 (training worlds pooled):")
for k in (4, 5, 6):
    for b in (0, 1):
        for g in (0, 1):
            ix = idx(M5, lambda m: m["world"] in TRAIN and m["k"] == k and int(bool(m["binding"])) == b and int(bool(m["GP1"])) == g)
            if len(ix):
                print(f"   k={k} b={b} g={g} n={len(ix)} RT-canon {mmm([rate(r, 'eval5', ix) for r in RC], 2)} rungs {lad(M5, ix, 1):.2f}/{lad(M5, ix, 2):.2f}/{lad(M5, ix, 3):.2f}")
print("P1 (evalk78):")
ixk = np.arange(len(M78))
print(f"   n={len(ixk)} rungs {lad(M78, ixk, 1):.3f}/{lad(M78, ixk, 2):.3f}/{lad(M78, ixk, 3):.3f}")
for arm in ARMS:
    print(f"   {arm}: masked {mmm([rate(r, 'evalk78', ixk) for r in seeds(D, arm)], 2)} unmasked {mmm([rate(r, 'evalk78', ixk, 'ok_u') for r in seeds(D, arm)], 2)}")

# ------------------------------------------------------------------ P5
print("\n===== P5 =====")
ixh = idx(M5, lambda m: m["world"] in HELD and m["binding"])
ixc = comb_ix(PAIRS + TRIPLES)
g = lad(M5, ixh, 1)
rt, ei = [rate(r, "eval5", ixh) for r in RC], [rate(r, "eval5", ixh) for r in REI]
print(f"n={len(ixh)} greedy {g:.4f} d3 {lad(M5, ixh, 3):.4f} | RT-canon {mmm(rt)} | EI16 per seed {[round(x, 4) for x in ei]} median {med(ei):.4f} | diff {med(ei) - med(rt):+.4f}")
print(f"  unmasked medians RT {med([rate(r, 'eval5', ixh, 'ok_u') for r in RC]):.4f} EI {med([rate(r, 'eval5', ixh, 'ok_u') for r in REI]):.4f}; comb n={len(ixc)} RT {med([rate(r, 'eval5', ixc) for r in RC]):.4f} EI {med([rate(r, 'eval5', ixc) for r in REI]):.4f}")
print("  EI16 found/tried:", [f"{r['ei_found'][0] / r['ei_found'][1]:.4f}" for r in REI])
print("  verdict:", "falsified" if (g > med(ei) and med(rt) > g) else ("holds" if abs(med(ei) - med(rt)) <= 0.1 + EPS else "differs by more than 0.1"))
print("  sensitivity: EI16 mean", np.mean(ei), "RT-canon mean", np.mean(rt), "; EI16 seeds within 0.1 of RT median:", sum(abs(x - med(rt)) <= 0.1 for x in ei), "/5")

# ------------------------------------------------------------------ P6
print("\n===== P6 =====")
def p6(run):
    pf = run["out"]["eval5"]["p_first"]; fc = run["out"]["eval5"]["first_canon"]
    tv, tu, canon, mass, unif = [], [], [], [], []
    for i, m in enumerate(M5):
        p = pf[i]
        if m["world"] in TRAIN and p and m["target_first"] and m["n_classes"] > 1 and len(p) > 1:
            tg = m["target_first"]; tot = sum(p.values()); ks = set(p) | set(tg)
            q = {a: v / tot for a, v in p.items()}
            tv.append(0.5 * sum(abs(q.get(a, 0) - tg.get(a, 0)) for a in ks))
            u = 1 / len(p); t_u = 0.5 * sum(abs((u if a in p else 0) - tg.get(a, 0)) for a in ks)
            tu.append(t_u); canon.append(fc[i]); mass.append(tot); unif.append(t_u < 1e-9)
    tv, tu, unif = np.array(tv), np.array(tu), np.array(unif)
    return dict(n=len(tv), tv=tv.mean(), tu=tu.mean(), canon=np.mean(canon), mass=np.mean(mass), n_unif=int(unif.sum()),
                tv_non=tv[~unif].mean(), tu_non=tu[~unif].mean(), tv_unif=tv[unif].mean())
sp, sc = [p6(r) for r in RP], [p6(r) for r in RC]
print("n", sp[0]["n"], "uniform-target instances", sp[0]["n_unif"], "non-uniform", sp[0]["n"] - sp[0]["n_unif"])
print("RT-post TV:", " ".join(f"{x['tv']:.4f}" for x in sp), "| uniform ref", f"{sp[0]['tu']:.4f}", "| below:", sum(x["tv"] < x["tu"] - EPS for x in sp), "/10")
print("RT-post on non-uniform:", " ".join(f"{x['tv_non']:.4f}" for x in sp), "vs", f"{sp[0]['tu_non']:.4f}", "| below:", sum(x["tv_non"] < x["tu_non"] for x in sp), "/10 ; on uniform-target:", " ".join(f"{x['tv_unif']:.3f}" for x in sp))
print("RT-post mass:", " ".join(f"{x['mass']:.3f}" for x in sp), "| canonical-first:", " ".join(f"{x['canon']:.3f}" for x in sp))
print("RT-canon canon-first:", " ".join(f"{x['canon']:.4f}" for x in sc), "| >=0.9:", sum(x["canon"] >= 0.9 - EPS for x in sc), "/10 | TV", " ".join(f"{x['tv']:.3f}" for x in sc), "| mass", " ".join(f"{x['mass']:.3f}" for x in sc))
print("PF: TV", " ".join(f"{p6(r)['tv']:.3f}" for r in RPF), "mass", " ".join(f"{p6(r)['mass']:.3f}" for r in RPF), "non-uniform", " ".join(f"{p6(r)['tv_non']:.3f}" for r in RPF))

# ------------------------------------------------------------------ P7
print("\n===== P7 =====")
exc = dv["examples_ceiling"]
ixtb = idx(M5, lambda m: m["world"] in TRAIN and m["binding"]); ixhb = ixh
ws = [w for w in TRAIN if exc[w]["binding_n"] > 0]
ceil = sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in ws) / sum(exc[w]["binding_n"] for w in ws)
mt, mh = med([rate(r, "eval5", ixtb, "ok_u") for r in RC]), med([rate(r, "eval5", ixhb, "ok_u") for r in RC])
ext, exh = [rate(r, "eval5", ixtb, "ok_u") for r in REX], [rate(r, "eval5", ixhb, "ok_u") for r in REX]
print(f"train binding n={len(ixtb)} ceiling {ceil:.4f} RT-canon median(u) {mt:.4f} target {ceil * mt:.4f} RT-examples {[round(x, 4) for x in ext]} within 0.1: {sum(abs(x - ceil * mt) <= 0.1 + EPS for x in ext)}/3")
print(f"held-out binding n={len(ixhb)} RT-canon median(u) {mh:.4f} RT-examples {[round(x, 4) for x in exh]} below by >0.05: {sum(mh - 0.05 > x + EPS for x in exh)}/3")
print(f"  masked: train RT-ex {[round(rate(r, 'eval5', ixtb), 3) for r in REX]} vs RT-canon median {med([rate(r, 'eval5', ixtb) for r in RC]):.3f}; held-out RT-ex {[round(rate(r, 'eval5', ixhb), 3) for r in REX]} vs {med([rate(r, 'eval5', ixhb) for r in RC]):.3f}")
print(f"  RT-shuffled unmasked train binding {[round(rate(r, 'eval5', ixtb, 'ok_u'), 4) for r in RSH]}; held-out binding: pairs {[round(rate(r, 'eval5', idx(M5, lambda m: m['world'] in PAIRS and m['binding']), 'ok_u'), 3) for r in RSH]} triples {[round(rate(r, 'eval5', idx(M5, lambda m: m['world'] in TRIPLES and m['binding']), 'ok_u'), 3) for r in RSH]} all {[round(rate(r, 'eval5', ixhb, 'ok_u'), 3) for r in RSH]}")
print(f"  RT-examples unmasked held-out: pairs {med([rate(r, 'eval5', idx(M5, lambda m: m['world'] in PAIRS and m['binding']), 'ok_u') for r in REX]):.3f} triples {med([rate(r, 'eval5', idx(M5, lambda m: m['world'] in TRIPLES and m['binding']), 'ok_u') for r in REX]):.3f}")

# ------------------------------------------------------------------ P8
print("\n===== P8 =====")
for rd in ("ok_m", "ok_u"):
    m = med([rate(r, "eval5", ixtb, rd) for r in RC]); sh = [rate(r, "eval5", ixtb, rd) for r in RSH]
    gaps = [m - x for x in sh]
    print(f"{rd}: RT-canon median {m:.4f} per-seed [{min(rate(r, 'eval5', ixtb, rd) for r in RC):.4f}, {max(rate(r, 'eval5', ixtb, rd) for r in RC):.4f}] shuffled {[round(x, 4) for x in sh]} gaps {[round(x, 4) for x in gaps]} -> {'holds' if all(x > 0.05 + EPS for x in gaps) else ('falsified' if not any(x > 0.05 + EPS for x in gaps) else 'undecided')}")
    print("     RT-canon seeds at or below the shuffled seeds:", sum(rate(r, 'eval5', ixtb, rd) <= max(sh) for r in RC), "/10")
for cls, W in (("train", TRAIN), ("held-out", HELD)):
    ix = idx(M5, lambda m: m["world"] in W)
    for r in RSH:
        cu = r["out"]["eval5"]["cat_u"][ix]
        print(f"  RT-shuffled s{r['cfg']['seed']} {cls} unmasked cats:", {CATS[c]: round(float((cu == c).mean()), 3) for c in range(6)})
print("\nfailure categories, RT-canon held-out eval5 (pooled over seeds):")
ix = idx(M5, lambda m: m["world"] in HELD)
for rd in ("cat_m", "cat_u"):
    allc = np.concatenate([r["out"]["eval5"][rd][ix] for r in RC])
    print("  ", rd, {CATS[c]: round(float((allc == c).mean()), 3) for c in range(6)})
