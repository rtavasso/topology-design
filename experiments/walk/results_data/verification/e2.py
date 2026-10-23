"""Independent recomputation of E2 from raw run files (does not use experiments/walk/analyze.py)."""
import glob, gzip, json, math
from collections import defaultdict
import numpy as np

REPO = "/home/claude/topology-design"
runs = [json.load(gzip.open(f)) for f in sorted(glob.glob(REPO + "/results/kaggle_e2/results/*.json.gz"))]
print("n runs", len(runs))


def binom_tail(k, n):  # P(X >= k), X~Bin(n, .5)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_test(diffs):
    pos = sum(d > 0 for d in diffs); neg = sum(d < 0 for d in diffs); n = pos + neg
    if n == 0:
        return pos, neg, 1.0, 1.0
    one = binom_tail(pos, n)
    two = min(1.0, 2 * min(binom_tail(pos, n), binom_tail(neg, n)))
    return pos, neg, one, two


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0] * m; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def first_cross(traj, key, thr):
    for e in traj:
        v = e.get(key)
        if v is not None and v >= thr:
            return e["step"]
    return None


def sustained(traj, thr=0.95, window=2000, need_full=True):
    steps = [e["step"] for e in traj]; acc = [e["test_acc"] for e in traj]; last = steps[-1]
    for i, s in enumerate(steps):
        if acc[i] < thr:
            continue
        if need_full and s + window > last:
            return None
        ok = all(acc[j] >= thr for j in range(i, len(steps)) if steps[j] <= s + window)
        if ok:
            return s
    return None


def graph_of(tag):
    return tag.split("-")[1]


base, jumps, noise = {}, defaultdict(dict), {}
for r in runs:
    c = r["config"]; tag = c["tag"]; g = graph_of(tag); seed = c["model_seed"]
    assert c["data_seed"] == seed
    if tag.startswith("wE2-"):
        assert c["drift_jump_at"] is None
        base[(g, seed)] = r
    elif tag.startswith("wE2jump-"):
        assert c["drift_jump_h"] > 0
        jumps[(g, c["drift_jump_at"], c["drift_jump_h"])][seed] = r
    elif tag.startswith("wE2noise-"):
        assert c["drift_jump_h"] == 0
        noise[(g, seed)] = r
print("base", len(base), "jump cells", {k: len(v) for k, v in sorted(jumps.items())}, "noise", len(noise))

# Every pair must agree on everything except the jump fields
IGN = {"drift_jump_at", "drift_jump_h", "tag", "measure_refit"}
for (g, s, *_), r in [((k[0], sd), v) for k, d in jumps.items() for sd, v in d.items()] + list(noise.items()):
    b = base[(g, s)]
    diff = {k for k in r["config"] if r["config"][k] != b["config"][k]} - IGN
    assert not diff, (g, s, diff)
print("pairing: all jump/noise configs match their base except jump fields + measure_refit")

# ---------------- P1 / P1b ----------------
for name, key, thr, minlead in [("P1 refit@0.9", "refit_test_acc", 0.9, 0.2), ("P1b extrap1000@0.5", "extrap_acc_1000", 0.5, 0.1)]:
    print(f"\n=== {name} ===")
    leads = {}
    for (g, s), r in sorted(base.items()):
        t = r["trajectory"]
        tm = first_cross(t, "test_acc", 0.9 if thr == 0.9 else 0.5)
        to = first_cross(t, key, thr)
        if tm is None:
            print("  model never reaches", g, s); continue
        if to is None:
            to_c = t[-1]["step"]
            print("  operator never reaches", g, s)
        lead = (tm - (to if to is not None else to_c)) / tm
        leads[(g, s)] = (tm, to, lead)
    for gsel in ["both", "sym", "asym"]:
        L = [v[2] for k, v in leads.items() if gsel == "both" or k[0] == gsel]
        pos, neg, one, two = sign_test(L)
        print(f"  {gsel}: n={len(L)} median lead {np.median(L):.4f}  +{pos}/-{neg} (ties {len(L)-pos-neg}) "
              f"sign p one-sided {one:.4f} two-sided {two:.4f}  -> supported={np.median(L) >= minlead and one < 0.05}")
    for k, v in sorted(leads.items()):
        print("   ", k, "t_model", v[0], "t_op", v[1], "lead", round(v[2], 4))

# ---------------- P2 ----------------
print("\n=== P2 ===")
H = {"sym": 15000, "asym": 40000}
for need_full in [True, False]:
    print(f"\n--- sustained definition need_full_window={need_full} ---")
    tb = {k: sustained(r["trajectory"], need_full=need_full) for k, r in base.items()}
    # noise floor
    nd = []
    for (g, s), r in sorted(noise.items()):
        tn = sustained(r["trajectory"], need_full=need_full)
        b = tb[(g, s)]
        if tn is None and b is None:
            continue
        d = (b if b is not None else H[g]) - (tn if tn is not None else H[g])
        nd.append(abs(d))
    print("noise |delta|:", nd, "median", np.median(nd))
    floor = float(np.median(nd))
    res = []
    for cell in sorted(jumps):
        g, sj, h = cell
        sav, dcol, excl = [], [], []
        for s, r in sorted(jumps[cell].items()):
            b = tb[(g, s)]
            tj = sustained(r["trajectory"], need_full=need_full)
            if b is not None and sj >= b:
                excl.append((s, "jump after base grok")); continue
            if b is None and tj is None:
                excl.append((s, "both censored")); continue
            sav.append((b if b is not None else H[g]) - (tj if tj is not None else H[g]))
            dcol.append(r["summary"]["n_collapses"] - base[(g, s)]["summary"]["n_collapses"])
        pos, neg, one, two = sign_test(sav)
        res.append((cell, sav, dcol, excl, pos, neg, one, two))
    adj1 = holm([x[6] for x in res]); adj2 = holm([x[7] for x in res])
    for (cell, sav, dcol, excl, pos, neg, one, two), a1, a2 in zip(res, adj1, adj2):
        testable = len(sav) >= 5
        med = float(np.median(sav)) if sav else float("nan")
        mc = float(np.median(dcol)) if dcol else float("nan")
        ok = testable and a1 < 0.05 and med > floor and mc <= 0
        print(f"{cell}: n={len(sav)} median saving {med:.0f} +{pos}/-{neg} p1={one:.4f} holm1={a1:.4f} "
              f"p2={two:.4f} holm2={a2:.4f} med dcollapse {mc} -> qualifies(one-sided)={ok}")
        print("     savings", sav, "dcol", dcol, "excluded", excl)
    print("base sustained:", {k: v for k, v in sorted(tb.items())})

# ---------------- noise arm identity ----------------
print("\n=== noise arms vs base: trajectory identity ===")
for (g, s), r in sorted(noise.items()):
    b = base[(g, s)]
    tn, tbb = r["trajectory"], b["trajectory"]
    assert [e["step"] for e in tn] == [e["step"] for e in tbb]
    dacc = max(abs(x["test_acc"] - y["test_acc"]) for x, y in zip(tn, tbb))
    dtr = max(abs(x["train_acc"] - y["train_acc"]) for x, y in zip(tn, tbb))
    dloss = max(abs(x["test_loss"] - y["test_loss"]) / max(abs(y["test_loss"]), 1e-12) for x, y in zip(tn, tbb))
    dwn = max(max(abs(a - c) for a, c in zip(x["weight_norms"], y["weight_norms"])) for x, y in zip(tn, tbb))
    nexact = sum(x["test_acc"] == y["test_acc"] for x, y in zip(tn, tbb))
    print(f"{g} {s}: max|dtest_acc|={dacc:.2e} max|dtrain_acc|={dtr:.2e} max rel dtest_loss={dloss:.2e} "
          f"max|dweightnorm|={dwn:.2e} exact test_acc evals {nexact}/{len(tn)} "
          f"ncol {r['summary']['n_collapses']} vs {b['summary']['n_collapses']} jump={r['summary'].get('jump')}")
