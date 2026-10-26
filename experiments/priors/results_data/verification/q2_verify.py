"""Independent recomputation of Q2 (drift vs null jumps) and replication against E2. usage: python q2_verify.py ROOT"""
import gzip, json, glob, sys, math, statistics as stt
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/topology-design"


def sustained(traj, th=0.95, hold=2000, sub=None):
    tj = traj if sub is None else [r for r in traj if r["step"] % sub == 0]
    steps = [r["step"] for r in tj]
    acc = [r["test_acc"] for r in tj]
    for i, t in enumerate(steps):
        if acc[i] < th or steps[-1] < t + hold:
            continue
        if all(a >= th for s, a in zip(steps[i:], acc[i:]) if s <= t + hold):
            return t
    return None


def sign_one_sided(d):
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    p = sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n
    return pos, neg, p


def sign_two_sided(d):
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


runs = []
for f in glob.glob(f"{ROOT}/results/kaggle_q2/results/*.json.gz"):
    with gzip.open(f) as fh:
        runs.append(json.load(fh))
H = 15000
base = {r["config"]["model_seed"]: r for r in runs if r["config"]["tag"] == "pQ2base"}
arms = defaultdict(dict)  # (at,h) -> seed -> dir -> run
for r in runs:
    c = r["config"]
    if c["tag"] == "pQ2jump":
        arms[(c["drift_jump_at"], c["drift_jump_h"])].setdefault(c["model_seed"], {})[c["drift_jump_dir"]] = r


def t_of(r, sub=None):
    t = sustained(r["trajectory"], sub=sub)
    return t


def analysis(sub=None, label="native evals"):
    print(f"\n#### Q2 ({label})")
    tb = {s: t_of(r, sub) for s, r in base.items()}
    print("base sustained t:", dict(sorted(tb.items())))
    cells = sorted(arms)
    res = {}
    for null in ("radial", "random", "rescale"):
        rows, ps = [], []
        for k in cells:
            at = k[0]
            ds, nexcl_base, nexcl_cens = [], 0, 0
            for s, a in sorted(arms[k].items()):
                if tb[s] is not None and tb[s] <= at:
                    nexcl_base += 1
                    continue
                td0, tn0 = t_of(a["drift"], sub), t_of(a[null], sub)
                if td0 is None and tn0 is None:
                    nexcl_cens += 1
                    continue
                td = H if td0 is None else td0
                tn = H if tn0 is None else tn0
                ds.append(tn - td)
            pos, neg, p = sign_one_sided(ds)
            rows.append((k, ds, pos, neg, p, nexcl_base, nexcl_cens))
            ps.append(p)
        adj = holm(ps)
        pooled = [d for r_ in rows for d in r_[1]]
        print(f"\n null={null}: pooled median D = {stt.median(pooled) if pooled else None}, n={len(pooled)}; "
              f"pooled sign test one-sided p = {sign_one_sided(pooled)[2]:.4g}")
        print(" | cell (at,H) | n elig | excl base-early / both-censored | median D | +/− | one-sided p | Holm p |")
        for (k, ds, pos, neg, p, eb, ec), pa in zip(rows, adj):
            print(f" | {k} | {len(ds)} | {eb}/{ec} | {stt.median(ds) if ds else None} | +{pos}/−{neg} | {p:.4g} | {pa:.4g} |  D={ds}")
        res[null] = (rows, adj, pooled)
    rows, adj, pooled = res["radial"]
    nsig = sum(pa < 0.05 for pa in adj)
    md = stt.median(pooled)
    verdict = "supported" if (nsig >= 3 and md > 0) else ("falsified" if md <= 0 else "inconclusive")
    print(f"\nP5 (radial): cells with Holm p<0.05: {nsig}/4; pooled median D {md} -> {verdict}")
    # drift saving vs base (E2's quantity)
    print("\n drift-arm savings vs base (t_base - t_drift), E2 eligibility (base not grokked before jump, not both censored):")
    for k in cells:
        at = k[0]
        ds = []
        for s, a in sorted(arms[k].items()):
            if tb[s] is not None and tb[s] <= at:
                continue
            td0 = t_of(a["drift"], sub)
            if td0 is None and tb[s] is None:
                continue
            ds.append((H if tb[s] is None else tb[s]) - (H if td0 is None else td0))
        print(f"  {k}: n={len(ds)} median {stt.median(ds) if ds else None}, +{sum(d>0 for d in ds)}/−{sum(d<0 for d in ds)}, "
              f"two-sided sign p={sign_two_sided(ds):.4g}  {ds}")
    return res


analysis()
analysis(sub=100, label="evals subsampled to every 100 steps, as E2")

# ---- replication vs E2
with gzip.open(f"{ROOT}/experiments/walk/results_data/e2_runs.json.gz") as fh:
    e2 = json.load(fh)
e2b = {r["config"]["model_seed"]: r for r in e2 if r["config"]["tag"] == "wE2-sym"}
print("\n#### Base replication vs E2 wE2-sym")
for s in sorted(base):
    a, b = base[s], e2b[s]
    ta = {r["step"]: r["test_acc"] for r in a["trajectory"]}
    tb_ = {r["step"]: r["test_acc"] for r in b["trajectory"]}
    common = sorted(set(ta) & set(tb_))
    maxdiff = max(abs(ta[t] - tb_[t]) for t in common)
    diffcfg = {k: (a["config"][k], b["config"].get(k)) for k in a["config"] if a["config"][k] != b["config"].get(k)}
    print(f" seed {s}: t_sus Q2 {sustained(a['trajectory'])} (sub100 {sustained(a['trajectory'], sub=100)}) vs E2 {sustained(b['trajectory'])}; "
          f"t_grok {a['summary']['t_grok']} vs {b['summary']['t_grok']}; max |Δtest_acc| on {len(common)} common evals {maxdiff:.2e}; "
          f"collapses {a['summary']['n_collapses']} vs {b['summary']['n_collapses']}")
    if s == min(base):
        print("   config diffs:", diffcfg)

print("\n#### E2 drift-jump savings vs Q2 drift-arm savings (per seed, t_base - t_jump, censored at 15000)")
e2j = defaultdict(dict)
for r in e2:
    c = r["config"]
    if c["tag"] == "wE2jump-sym":
        e2j[(c["drift_jump_at"], c["drift_jump_h"])][c["model_seed"]] = r
for k in sorted(arms):
    out = []
    for s in sorted(arms[k]):
        tq = sustained(arms[k][s]["drift"]["trajectory"])
        te = sustained(e2j[k][s]["trajectory"])
        tbq, tbe = sustained(base[s]["trajectory"]), sustained(e2b[s]["trajectory"])
        out.append(f"s{s}: Q2 {tq} / E2 {te}")
    # identity check for the drift arms
    md = []
    for s in sorted(arms[k]):
        ta = {r["step"]: r["test_acc"] for r in arms[k][s]["drift"]["trajectory"]}
        tb_ = {r["step"]: r["test_acc"] for r in e2j[k][s]["trajectory"]}
        common = sorted(set(ta) & set(tb_))
        md.append(max(abs(ta[t] - tb_[t]) for t in common))
    print(f" {k}: max |Δacc| per seed vs E2 drift arm: {[f'{m:.1e}' for m in md]}")
    print("    ", "; ".join(out))
