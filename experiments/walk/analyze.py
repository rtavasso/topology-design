"""Preregistered analyses for "diagnose the walk" (PREREGISTRATION.md). Written before any data.

    python experiments/walk/analyze.py e1 results/kaggle_e1/results
    python experiments/walk/analyze.py e2 results/kaggle_e2/results       (same for e3, e4)

Each writes <results dir>/../<exp>_summary.md and _summary.json and prints the summary. Verdicts follow the rules
in PREREGISTRATION.md exactly; anything else is labelled exploratory.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics as stt
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "fleet"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topology_design import fleet as fl  # noqa: E402
from topology_design import program_prior as pp  # noqa: E402
from run_fleet import _tuplify, load_result  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

N_PERM = 10000


def load(rdir: str, prefix: str) -> list[dict]:
    out = []
    for f in sorted(os.listdir(rdir)):
        if f.endswith((".json", ".json.gz")):
            r = load_result(os.path.join(rdir, f))
            if r["config"]["tag"].startswith(prefix):
                out.append(r)
    if not out:
        sys.exit(f"no runs with tag prefix {prefix} in {rdir}")
    return out


def kendall(x, y) -> float:
    """Kendall tau-b."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def cfg_of(r) -> fl.FleetConfig:
    kw = {k: v for k, v in r["config"].items() if k in fl.FleetConfig.__dataclass_fields__}
    return fl.FleetConfig(**_tuplify(kw))


def perm_tau(x, y, strata, rng, n=N_PERM) -> tuple[float, float]:
    """Kendall tau and a one-sided (tau < observed is extreme for negative, > for positive) permutation p, permuting
    x within strata. Returns (tau, p_two_sided)."""
    tau = kendall(x, y)
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    hits = 0
    for _ in range(n):
        xp = list(x)
        for idx in groups.values():
            vals = [x[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                xp[i] = v
        hits += abs(kendall(xp, y)) >= abs(tau) - 1e-12
    return tau, (hits + 1) / (n + 1)


def sign_test(diffs) -> tuple[int, int, float]:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def first_step(traj, key, th):
    return next((r["step"] for r in traj if r.get(key) is not None and r[key] >= th), None)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return stt.median(xs) if xs else None


def fmt(x, nd=2):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def sustained_grok(traj, th=0.95, hold=2000):
    """First step t with test_acc >= th at every eval in [t, t + hold] (None if never, or not observed long enough)."""
    steps = [r["step"] for r in traj]
    accs = [r["test_acc"] for r in traj]
    for i, t in enumerate(steps):
        if accs[i] < th or steps[-1] < t + hold:
            continue
        if all(a >= th for s_, a in zip(steps[i:], accs[i:]) if s_ <= t + hold):
            return t
    return None


def verdict(ok, testable=True, why=""):
    if not testable:
        return f"**untestable**{(' (' + why + ')') if why else ''}"
    return "**supported**" if ok else "**NOT supported**"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def logit(x, lo=0.02):
    x = min(1 - lo, max(lo, x))
    return math.log(x / (1 - x))


# ---------------------------------------------------------------------------- E1 (measurement)


def e1(rdir):
    allr = load(rdir, "wE1")
    streams = [r for r in allr if r["config"]["tag"].startswith("wE1-")]
    floor = [r for r in allr if r["config"]["tag"].startswith("wE1floor")]
    lines = ["# E1 — information gap (measurement)\n", f"{len(streams)} stream runs, {len(floor)} data-floor runs.\n"]
    prog_cache, rows = {}, []
    for r in streams:
        c = cfg_of(r)
        pq = r["summary"]["prequential"]
        n_tot = int(sum(pq["n"]))
        cum = np.cumsum([0] + pq["n"][:-1])
        prod = next((int(cum[k]) for k in range(1, len(pq["n"])) if pq["n"][k] > 0 and pq["acc"][k] >= 0.9), None)
        key = (c.data_seed, c.p, c.train_frac, c.arrive_start, c.arrive_n)
        if key not in prog_cache:
            t = fl.build_task(c)
            n = len(t["tr"])
            _, inc, K = fl._arrivals([c], n, "cpu")
            tr, y = t["tr"].numpy(), t["y"].numpy()
            T, logm, _ = pp.enumerate_functions(c.p, 7)
            blocks = [tr[(inc[0] == k).numpy()] for k in range(K + 1)]
            res_b = pp.prequential(T, logm, blocks, [y[b] for b in blocks], c.p, true_fn=y)
            g = torch.Generator().manual_seed(40_000 + c.data_seed)
            order = tr[torch.randperm(n, generator=g).numpy()]
            m = min(n, 200)
            res_e = pp.prequential(T, logm, [order[i:i + 1] for i in range(m)], [y[order[i:i + 1]] for i in range(m)],
                                   c.p, true_fn=y)
            k_star = next((i + 1 for i, v in enumerate(res_e["post_true"]) if v >= 0.99), None)
            prog_cache[key] = {"nats": sum(res_b["nats"]), "k_star": k_star, "uniform": n * math.log(c.p),
                               "start": c.arrive_start}
        pr = prog_cache[key]
        rows.append({"graph": r["config"]["tag"].split("-")[1], "lr": c.lr, "budget": c.arrive_every,
                     "seed": c.model_seed, "gd_nats": sum(pq["nats"]), "prog_nats": pr["nats"],
                     "uniform": pr["uniform"], "k_star": pr["k_star"], "prod": prod, "n_tot": n_tot})
    lines += ["## Streams (32 new pairs per increment, after an initial 140)\n",
              "| graph | lr | steps per increment | GD code length (nats) | program prior | uniform | GD / program | "
              "GD production point (examples; median) | censored |", "|---|---|---|---|---|---|---|---|---|"]
    cells = defaultdict(list)
    for x in rows:
        cells[(x["graph"], x["lr"], x["budget"])].append(x)
    for (g, lr, b), xs in sorted(cells.items()):
        prods = [x["prod"] if x["prod"] is not None else x["n_tot"] for x in xs]
        lines.append(f"| {g} | {lr:g} | {b} | {med([x['gd_nats'] for x in xs]):.0f} | {med([x['prog_nats'] for x in xs]):.0f} | "
                     f"{med([x['uniform'] for x in xs]):.0f} | {med([x['gd_nats'] / x['prog_nats'] for x in xs]):.1f} | "
                     f"{med(prods)} | {sum(x['prod'] is None for x in xs)}/{len(xs)} |")
    fl_rows = defaultdict(list)
    for r in floor:
        c = r["config"]
        ok = r["summary"]["t_grok"] is not None and (r["summary"]["final_test_acc"] or 0) >= 0.9
        fl_rows[(c["tag"].split("-")[1], c["train_frac"])].append(ok)
    lines += ["", "## Data floor (fixed dataset, up to 60,000 steps)\n", "| graph | train fraction | examples | grokked |",
              "|---|---|---|---|"]
    floors = {}
    for (g, f_), oks in sorted(fl_rows.items()):
        lines.append(f"| {g} | {f_:g} | {round(f_ * 59 * 59)} | {sum(oks)}/{len(oks)} |")
        if sum(oks) >= 2 and (g not in floors or f_ < floors[g]):
            floors[g] = f_
    ks = sorted(set(x["k_star"] for x in rows if x["k_star"] is not None))
    lines += ["", "## Measurements",
              f"- Program prior: the rule is pinned down (posterior ≥ 0.99) after k* = {ks} examples "
              f"(calibration check, expected ≤ 16: {'passed' if ks and max(ks) <= 16 else 'FAILED'}).",
              f"- Gradient descent's code length is {fmt(min(x['gd_nats'] / x['prog_nats'] for x in rows), 1)}–"
              f"{fmt(max(x['gd_nats'] / x['prog_nats'] for x in rows), 1)}× the program prior's.",
              "- Data floor (smallest training fraction where ≥ 2 of 3 seeds grok): " +
              ", ".join(f"{g}: {f_:g} ({round(f_ * 59 * 59)} examples, {round(f_ * 59 * 59 / max(ks) if ks else float('nan'))}× k*)"
                        for g, f_ in sorted(floors.items())) +
              ("" if floors else "none within the sweep"),
              "- Walk versus data: compare the production point at 400 vs 1,600 steps per increment (a drop means more "
              "walking per example helps) and both against the data floor."]
    return lines, {"rows": rows, "floor": {f"{k[0]}|{k[1]}": v for k, v in fl_rows.items()}, "floors": floors,
                   "k_star": ks}


# ---------------------------------------------------------------------------- E2


def e2(rdir):
    allr = load(rdir, "wE2")
    base = [r for r in allr if r["config"]["tag"].startswith("wE2-")]
    jumps = [r for r in allr if r["config"]["tag"].startswith("wE2jump")]
    noise = [r for r in allr if r["config"]["tag"].startswith("wE2noise")]
    lines = ["# E2 — is the answer visible before the network produces it, and can a learner act on it?\n",
             f"{len(base)} base runs, {len(jumps)} jump runs, {len(noise)} noise-floor runs.\n"]
    rows = []
    by_cell = defaultdict(list)
    for r in base:
        c, tj = r["config"], r["trajectory"]
        row = {"graph": c["tag"].split("-")[1], "lr": c["lr"], "seed": c["model_seed"],
               "collapses": r["summary"]["n_collapses"], "t_sus": sustained_grok(tj)}
        for th in (0.5, 0.9):
            row[f"model_{th}"] = first_step(tj, "test_acc", th)
            row[f"refit_{th}"] = first_step(tj, "refit_test_acc", th)
            for H in c["horizon_extrap"]:
                row[f"ex{H}_{th}"] = first_step(tj, f"extrap_acc_{H}", th)
                row[f"exd{H}_{th}"] = first_step(tj, f"extrapd_acc_{H}", th)
        rows.append(row)
        by_cell[row["graph"]].append(row)

    def lead_stats(key, th):
        d, rel = [], []
        for x in rows:
            tm, tk = x[f"model_{th}"], x.get(f"{key}_{th}")
            if tm is not None and tk is not None:
                d.append(tm - tk)
                rel.append((tm - tk) / tm if tm > 0 else 0.0)
        pos, neg, p = sign_test(d)
        return {"median_rel": med(rel), "pos": pos, "neg": neg, "p": p, "n": len(d)}

    lines.append("| graph | runs | model 0.5 / 0.9 (median step) | refit readout 0.5 / 0.9 | linear jump-ahead H=1000 "
                 "0.5 / 0.9 | sustained grok |")
    lines.append("|---|---|---|---|---|---|")
    for g, xs in sorted(by_cell.items()):
        f2 = lambda k: f"{med([x[k + '_0.5'] for x in xs])} / {med([x[k + '_0.9'] for x in xs])}"
        lines.append(f"| {g} | {len(xs)} | {f2('model')} | {f2('refit')} | {f2('ex1000')} | "
                     f"{sum(x['t_sus'] is not None for x in xs)}/{len(xs)} |")
    r9, e5 = lead_stats("refit", 0.9), lead_stats("ex1000", 0.5)
    p1_ok = r9["median_rel"] is not None and r9["median_rel"] >= 0.2 and r9["pos"] > r9["neg"] and r9["p"] < 0.05
    p1b_ok = e5["median_rel"] is not None and e5["median_rel"] >= 0.1 and e5["pos"] > e5["neg"] and e5["p"] < 0.05
    # jumps vs paired base, with the H = 0 noise floor
    base_by = {(r["config"]["tag"].split("-")[1], r["config"]["model_seed"]): r for r in base}

    def delta(r):
        c = r["config"]
        b = base_by.get((c["tag"].split("-")[1], c["model_seed"]))
        if b is None:
            return None, None, None
        tb0, tj0 = sustained_grok(b["trajectory"]), sustained_grok(r["trajectory"])
        tb = c["steps"] if tb0 is None else tb0
        tj_ = c["steps"] if tj0 is None else tj0
        # eligible: the jump came before the base's sustained grokking, and at least one of the pair grokked
        # (if neither did, both are censored at the horizon and the pair carries no information)
        eligible = (tb0 is None or tb0 > c["drift_jump_at"]) and (tb0 is not None or tj0 is not None)
        return tb - tj_, eligible, r["summary"]["n_collapses"] - b["summary"]["n_collapses"]

    noise_d = defaultdict(list)
    for r in noise:
        d, _, _ = delta(r)
        if d is not None:
            noise_d[r["config"]["tag"].split("-")[1]].append(abs(d))
    cells = defaultdict(list)
    for r in jumps:
        c = r["config"]
        d, el, dc = delta(r)
        jm = r["summary"]["jump"]
        cells[(c["tag"].split("-")[1], c["drift_jump_at"], c["drift_jump_h"])].append(
            {"d": d, "eligible": el, "dcoll": dc, "before": jm["test_acc_before"], "after": jm["test_acc_after"]})
    keys = sorted(cells)
    pvals, stats = [], []
    for k in keys:
        xs = [x for x in cells[k] if x["eligible"] and x["d"] is not None]
        pos, neg, p = sign_test([x["d"] for x in xs])
        stats.append((xs, pos, neg, p))
        pvals.append(p if not math.isnan(p) else 1.0)
    adj = holm(pvals) if pvals else []
    lines += ["", "| graph | jump at | H | eligible | test acc before → right after | median steps saved | Holm p | "
              "median Δ collapses | noise floor (median abs Δ, H = 0) | cell supports |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    any_support, any_testable = False, False
    for k, (xs, pos, neg, p), pa in zip(keys, stats, adj):
        g = k[0]
        nf = med(noise_d.get(g, []))
        md, mc = med([x["d"] for x in xs]), med([x["dcoll"] for x in xs])
        testable = len(xs) >= 5
        any_testable |= testable
        sup = testable and pa < 0.05 and md is not None and md > 0 and (nf is None or md > nf) and (mc is None or mc <= 0)
        any_support |= sup
        lines.append(f"| {g} | {k[1]} | {k[2]} | {len(xs)} | {fmt(med([x['before'] for x in cells[k]]))} → "
                     f"{fmt(med([x['after'] for x in cells[k]]))} | {md} | {fmt(pa, 4)} | {mc} | {nf} | "
                     f"{'yes' if sup else ('—' if not testable else 'no')} |")
    lines += ["", "## Verdicts (preregistered)",
              f"- **P1 (readout visibility):** refit reaches test accuracy 0.9 earlier than the model, median relative "
              f"lead ≥ 0.2 with a sign test: median {fmt(r9['median_rel'])}, +{r9['pos']}/−{r9['neg']}, p = {fmt(r9['p'], 4)} "
              f"→ {verdict(p1_ok, r9['n'] >= 5, 'fewer than 5 runs reach 0.9')} (pilot expectation: not supported)",
              f"- **P1b (drift visibility):** the linear jump-ahead (H = 1,000) reaches 0.5 earlier than the model, median "
              f"relative lead ≥ 0.1 with a sign test: median {fmt(e5['median_rel'])}, +{e5['pos']}/−{e5['neg']}, "
              f"p = {fmt(e5['p'], 4)} → {verdict(p1b_ok, e5['n'] >= 5, 'fewer than 5 runs reach 0.5')}",
              f"- **P2 (actionability):** at least one jump cell groks (sustained) earlier than its paired bases "
              f"(Holm-adjusted sign test p < 0.05), by more than the H = 0 noise floor, without more collapses → "
              f"{verdict(any_support, any_testable, 'no cell has 5 eligible runs')}",
              "- *Exploratory:* the decay-aware operator (extrapd), longer horizons, and refit at 0.5 are in the json."]
    return lines, {"base": rows, "P1": p1_ok, "P1b": p1b_ok, "P2": any_support,
                   "leads": {"refit_0.9": r9, "ex1000_0.5": e5, "refit_0.5": lead_stats("refit", 0.5),
                             **{f"{op}{H}_{th}": lead_stats(f"{op}{H}", th) for op in ("ex", "exd")
                                for H in (250, 1000, 4000) for th in (0.5, 0.9)}}}


# ---------------------------------------------------------------------------- E3


def _branch(rec, key, idx):
    return sum(rec[key][i] for i in idx)


def e3(rdir):
    runs = load(rdir, "wE3")
    rng = random.Random(0)
    A, B = (1, 2, 3), (4, 5, 6)
    rows = []
    for r in runs:
        c, tj = r["config"], r["trajectory"]
        kind = c["tag"].split("-")[0]  # wE3 / wE3lr / wE3wd
        lm = c.get("branch_lr_mult") or [1.0, 1.0]
        wm = c.get("branch_wd_mult") or [1.0, 1.0]
        rho = float(c["tag"].split("rho")[1])
        tot_mem = [sum(x["mem"]) for x in tj]
        rm = tj[int(np.argmax(tot_mem))]
        ma, mb = _branch(rm, "mem", A), _branch(rm, "mem", B)
        tail = [x for x in tj if x["step"] > c["steps"] - 2000]
        if not tail:
            continue
        ga = stt.fmean(_branch(x, "gen", A) for x in tail)
        gb = stt.fmean(_branch(x, "gen", B) for x in tail)
        tg = r["summary"]["t_grok"]
        stable_tail = min(x["train_acc"] for x in tail) >= 0.9
        no_cancel = abs(ga + gb) >= 0.5 * (abs(ga) + abs(gb)) and abs(ma + mb) >= 0.5 * (abs(ma) + abs(mb))
        eligible = (tg is not None and tg <= c["steps"] - 5000 and stable_tail and no_cancel
                    and r["summary"]["diverged_at"] is None)
        rows.append({"graph": c["tag"].split("-")[1], "kind": kind, "rho": rho, "seed": c["model_seed"],
                     "s_gen": ga / (ga + gb) if ga + gb else float("nan"),
                     "s_mem": ma / (ma + mb) if ma + mb else float("nan"),
                     "eligible": eligible, "collapses": r["summary"]["n_collapses"]})
    lines = ["# E3 — credit by speed\n", f"{len(rows)} runs. Main arms: branch 0 (blocks 1–3) steps ρ× faster than branch 1 "
             "at equal lr·wd. Controls: lr-only and wd-only at ρ = ¼ and 4.\n",
             "| graph | arm | ρ | eligible | share of gen (end, median) | share of mem (peak) | collapses (median) |",
             "|---|---|---|---|---|---|---|"]
    for g in sorted({x["graph"] for x in rows}):
        for kind in ("wE3", "wE3lr", "wE3wd"):
            for rho in sorted({x["rho"] for x in rows if x["graph"] == g and x["kind"] == kind}):
                ys = [x for x in rows if x["graph"] == g and x["kind"] == kind and x["rho"] == rho]
                el = [y for y in ys if y["eligible"]]
                lines.append(f"| {g} | { {'wE3': 'main', 'wE3lr': 'lr-only', 'wE3wd': 'wd-only'}[kind]} | {rho:g} | {len(el)}/{len(ys)} | "
                             f"{fmt(med([y['s_gen'] for y in el]))} | {fmt(med([y['s_mem'] for y in el]))} | "
                             f"{med([y['collapses'] for y in ys])} |")
    verdicts = {}
    lines.append("\n## Verdicts (preregistered)")
    for g in sorted({x["graph"] for x in rows}):
        main = [x for x in rows if x["graph"] == g and x["kind"] == "wE3" and x["eligible"]]
        testable_rhos = [rho for rho in sorted({x["rho"] for x in main}) if sum(x["rho"] == rho for x in main) >= 6]
        main_t = [x for x in main if x["rho"] in testable_rhos]
        t1 = len(testable_rhos) >= 3
        if t1:
            tg_, pg = perm_tau([x["rho"] for x in main_t], [logit(x["s_gen"]) for x in main_t], [x["seed"] for x in main_t], rng)
            tm_, pm = perm_tau([x["rho"] for x in main_t], [logit(x["s_mem"]) for x in main_t], [x["seed"] for x in main_t], rng)
        else:
            tg_ = pg = tm_ = pm = float("nan")
        p1 = t1 and tg_ > 0 and pg < 0.05 and tm_ > 0 and pm < 0.05
        dmed = {rho: med([logit(x["s_gen"]) - logit(x["s_mem"]) for x in main_t if x["rho"] == rho]) for rho in testable_rhos}
        t2 = t1 and 1.0 in testable_rhos
        if t2:
            td, pdv = perm_tau([x["rho"] for x in main_t], [logit(x["s_gen"]) - logit(x["s_mem"]) for x in main_t],
                               [x["seed"] for x in main_t], rng)
            shifts = {rho: dmed[rho] - dmed[1.0] for rho in testable_rhos}
            big = any(abs(v) >= 0.5 for v in shifts.values())
            p2 = "supported" if (pdv >= 0.05 and not big) else ("FALSIFIED (speed induces a role split)" if (pdv < 0.05 and big) else "inconclusive")
        else:
            td = pdv = float("nan")
            shifts, p2 = {}, "untestable"
        lr_arm = [x for x in rows if x["graph"] == g and x["kind"] == "wE3lr" and x["eligible"]]
        by_seed = defaultdict(dict)
        for x in lr_arm:
            by_seed[x["seed"]][x["rho"]] = x
        pairs = [(v[4.0], v[0.25]) for v in by_seed.values() if 4.0 in v and 0.25 in v]
        t3 = len(pairs) >= 6
        sg = sign_test([logit(a["s_gen"]) - logit(b["s_gen"]) for a, b in pairs])
        sm = sign_test([logit(a["s_mem"]) - logit(b["s_mem"]) for a, b in pairs])
        p3 = t3 and sg[0] > sg[1] and sg[2] < 0.05 and sm[0] > sm[1] and sm[2] < 0.05

        def effect(kind):
            xs = [x for x in rows if x["graph"] == g and x["kind"] == kind and x["eligible"]]
            hi = med([logit(x["s_gen"]) for x in xs if x["rho"] == 4.0])
            lo = med([logit(x["s_gen"]) for x in xs if x["rho"] == 0.25])
            return None if hi is None or lo is None else hi - lo
        verdicts[g] = {"P1": p1, "P2": p2, "P3": p3, "tau_gen": tg_, "tau_mem": tm_, "d_shifts": shifts,
                       "effect_main": effect("wE3"), "effect_lr": effect("wE3lr"), "effect_wd": effect("wE3wd")}
        lines += [f"- **{g}, P1** shares rise with ρ (testable ρ: {testable_rhos}): τ(gen) = {fmt(tg_, 3)} (p = {fmt(pg, 4)}), "
                  f"τ(mem) = {fmt(tm_, 3)} (p = {fmt(pm, 4)}) → {verdict(p1, t1, 'fewer than 3 ρ cells with 6 eligible runs')}",
                  f"- **{g}, P2** co-location persists (logit gen − logit mem): trend τ = {fmt(td, 3)} (p = {fmt(pdv, 4)}), "
                  f"shifts vs ρ = 1: { {f'{k:g}': fmt(v) for k, v in shifts.items()} } → **{p2}**",
                  f"- **{g}, P3** step size alone moves credit (lr-only arm, ρ = 4 vs ¼, paired): gen +{sg[0]}/−{sg[1]} "
                  f"(p = {fmt(sg[2], 4)}), mem +{sm[0]}/−{sm[1]} (p = {fmt(sm[2], 4)}) → {verdict(p3, t3, 'fewer than 6 eligible pairs')}",
                  f"- *{g}, diagnostic:* logit-share effect ρ 4 vs ¼ — main {fmt(verdicts[g]['effect_main'])}, lr-only "
                  f"{fmt(verdicts[g]['effect_lr'])}, wd-only {fmt(verdicts[g]['effect_wd'])}"]
    return lines, {"rows": rows, "verdict": verdicts}


# ---------------------------------------------------------------------------- E4


def window_stable(r, window=6000):
    s, tj = r["summary"], r["trajectory"]
    t_fit = s["t_fit"]
    if t_fit is None or t_fit > r["config"]["steps"] - window:
        return False
    if s["diverged_at"] is not None and s["diverged_at"] <= t_fit + window:
        return False
    return min(x["train_acc"] for x in tj if t_fit <= x["step"] <= t_fit + window) >= 0.9


def e4(rdir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("walk_make_grids", os.path.join(HERE, "make_grids.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    runs = load(rdir, "wE4")
    cell = defaultdict(list)
    deep = defaultdict(list)
    for r in runs:
        c, s = r["config"], r["summary"]
        st_ = window_stable(r)
        cell[(c["tag"].split("-")[1], c["arm"], c["weight_decay"], c["lr"])].append((st_, s["t_grok"] is not None))
        if c["arm"] == "pk0123" and st_ and r["trajectory"]:
            deep[c["tag"].split("-")[1]].append(r["trajectory"][-1]["abl_block_test_acc"][3])
    params, arms = ["std", "P1", "P2"], list(mg.E4_ARMS)
    okc = lambda v: len(v) >= 2 and sum(x[0] for x in v) >= 2
    lines = ["# E4 — stability regions\n", f"{len(runs)} runs. Stable cell: ≥ 2 of 3 seeds fit and keep train accuracy "
             "≥ 0.9 for 6,000 steps after first fitting, without diverging (Study 2's window). Grokking within 12,000 steps "
             "is reported separately (g).\n"]
    out = {}
    for pn in params:
        lines += [f"\n## {pn}\n", "| arm (C) | wd | " + " | ".join(f"{lr:g}" for lr in mg.E4_LRS) + " | lr* |",
                  "|---|---|" + "---|" * len(mg.E4_LRS) + "---|"]
        for arm in arms:
            for wd in mg.E4_WDS:
                marks, lstar = [], None
                for lr in mg.E4_LRS:
                    v = cell.get((pn, arm, wd, lr), [])
                    ok = okc(v)
                    gk_ = len(v) >= 2 and sum(x[1] for x in v) >= 2
                    marks.append(("✓" if ok else "·") + ("g" if gk_ else ""))
                    if ok:
                        lstar = lr
                out[(pn, arm, wd)] = lstar
                lines.append(f"| {arm} ({mg.E4_ARMS[arm]}) | {wd:g} | " + " | ".join(marks) + f" | {fmt(lstar)} |")
    top, bottom = mg.E4_LRS[-1], mg.E4_LRS[0] / 2

    def gap(pn):
        l1, l4 = out[(pn, "pk0000", 1.0)], out[(pn, "pk0123", 1.0)]
        if l1 is None:
            return None, l1, l4
        return round(math.log2(l1 / (l4 if l4 is not None else bottom))), l1, l4
    lines.append("\n## Verdicts (preregistered, wd = 1)")
    res = {}
    g0, l1, l4 = gap("std")
    p0_testable = l1 is not None and not (l1 == top and l4 == top)
    p0 = p0_testable and g0 is not None and g0 >= 2
    res["P0"] = {"gap": g0, "lr*_C1": l1, "lr*_C4": l4, "supported": p0}
    lines.append(f"- **P0 (replicates Study 2):** under std, lr*(C1) = {fmt(l1)}, lr*(C4) = {fmt(l4)}, gap = {g0} grid steps "
                 f"(need ≥ 2) → {verdict(p0, p0_testable, 'C1 never stable, or both at the top of the grid')}")
    for pn in ("P1", "P2"):
        gp, a1, a4 = gap(pn)
        testable = p0 and a1 is not None and not (a1 == top and (a4 == top or (gp is not None and abs(gp) <= 1)))
        sup = testable and gp is not None and abs(gp) <= 1
        res[pn] = {"gap": gp, "lr*_C1": a1, "lr*_C4": a4, "H8_supported": sup}
        lines.append(f"- **H8 under {pn}:** lr*(C1) = {fmt(a1)}, lr*(C4) = {fmt(a4)}, |gap| = {None if gp is None else abs(gp)} "
                     f"(need ≤ 1, with P0 supported) → {verdict(sup, testable, 'P0 not supported, C1 never stable, or C1 censored at the top of the grid')}")
    lines.append(f"- *Silencing check (exploratory):* test accuracy with the deepest block of pk0123 removed, in stable "
                 f"runs (lower = the block matters): " + ", ".join(f"{pn} {fmt(med(deep.get(pn, [])))}" for pn in params))
    return lines, {"cells": {str(k): v for k, v in out.items()}, "verdict": res}


def main():
    exp, rdir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, HERE)
    lines, data = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}[exp](rdir)
    out_dir = os.path.dirname(os.path.abspath(rdir.rstrip("/")))
    open(os.path.join(out_dir, f"{exp}_summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(data, open(os.path.join(out_dir, f"{exp}_summary.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
