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


# ---------------------------------------------------------------------------- E1


def e1(rdir):
    runs = load(rdir, "wE1")
    rng = random.Random(0)
    lines = ["# E1 — information gap (prequential code length)\n", f"{len(runs)} runs.\n"]
    prog_cache = {}
    rows = []
    for r in runs:
        c = cfg_of(r)
        pq = r["summary"]["prequential"]
        n_tot = int(sum(pq["n"]))
        cum = np.cumsum([0] + pq["n"][:-1])
        prod = next((int(cum[k]) for k in range(len(pq["n"])) if pq["n"][k] > 0 and pq["acc"][k] >= 0.9), None)
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
            prog_cache[key] = {"nats": sum(res_b["nats"]), "k_star": k_star, "uniform": n * math.log(c.p)}
        pr = prog_cache[key]
        graph = r["config"]["tag"].split("-")[1]
        rows.append({"graph": graph, "lr": c.lr, "budget": c.arrive_every, "seed": c.model_seed,
                     "gd_nats": sum(pq["nats"]), "prog_nats": pr["nats"], "uniform": pr["uniform"],
                     "k_star": pr["k_star"], "prod": prod, "n_tot": n_tot})
    lines.append("| graph | lr | steps per increment | GD code length (nats, median) | program prior | uniform | "
                 "GD / program | GD production point (examples; median, censored) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    cells = defaultdict(list)
    for x in rows:
        cells[(x["graph"], x["lr"], x["budget"])].append(x)
    for (g, lr, b), xs in sorted(cells.items()):
        prods = [x["prod"] if x["prod"] is not None else x["n_tot"] for x in xs]
        cens = sum(x["prod"] is None for x in xs)
        lines.append(f"| {g} | {lr:g} | {b} | {med([x['gd_nats'] for x in xs]):.0f} | {med([x['prog_nats'] for x in xs]):.0f} | "
                     f"{med([x['uniform'] for x in xs]):.0f} | {med([x['gd_nats'] / x['prog_nats'] for x in xs]):.1f} | "
                     f"{med(prods)} ({cens}/{len(xs)} censored) |")
    ks = [x["k_star"] for x in rows]
    p1 = all(k is not None and k <= 16 for k in ks)
    ratios = [x["gd_nats"] / x["prog_nats"] for x in rows]
    p2 = all(q >= 5 for q in ratios)
    prods = [x["prod"] if x["prod"] is not None else x["n_tot"] for x in rows]
    tau, pval = perm_tau([x["budget"] for x in rows], prods, [(x["graph"], x["lr"], x["seed"]) for x in rows], rng)
    p3 = tau < 0 and pval < 0.05
    lines += ["", "## Verdicts (preregistered)",
              f"- **P1** program prior determines the rule within 16 examples (posterior ≥ 0.99) for every stream: "
              f"k* = {sorted(set(k for k in ks if k is not None))} → **{'supported' if p1 else 'NOT supported'}**",
              f"- **P2** GD code length ≥ 5× the program prior's in every run: min ratio {min(ratios):.1f} → "
              f"**{'supported' if p2 else 'NOT supported'}**",
              f"- **P3 (primary)** more steps per example lowers GD's production point: Kendall τ(budget, production "
              f"point) = {tau:+.3f}, permutation p = {pval:.4f} (within graph × lr × seed) → "
              f"**{'supported' if p3 else 'NOT supported'}**"]
    return lines, {"rows": rows, "P1": p1, "P2": p2, "P3": {"tau": tau, "p": pval, "supported": p3}}


# ---------------------------------------------------------------------------- E2


def e2(rdir):
    allr = load(rdir, "wE2")
    base = [r for r in allr if r["config"]["tag"].startswith("wE2-")]
    jumps = [r for r in allr if r["config"]["tag"].startswith("wE2jump")]
    lines = ["# E2 — is the answer visible before the network produces it, and can a learner act on it?\n",
             f"{len(base)} base runs, {len(jumps)} drift-jump runs.\n"]
    rows = []
    by_cell = defaultdict(list)
    for r in base:
        c, tj = r["config"], r["trajectory"]
        row = {"graph": c["tag"].split("-")[1], "lr": c["lr"], "seed": c["model_seed"],
               "collapses": r["summary"]["n_collapses"], "t_grok": r["summary"]["t_grok"]}
        for th in (0.5, 0.9):
            row[f"model_{th}"] = first_step(tj, "test_acc", th)
            row[f"refit_{th}"] = first_step(tj, "refit_test_acc", th)
            for H in c["horizon_extrap"]:
                row[f"ex{H}_{th}"] = first_step(tj, f"extrap_acc_{H}", th)
        rows.append(row)
        by_cell[(row["graph"], row["lr"])].append(row)

    def lead_stats(key, th):
        d, rel = [], []
        for x in rows:
            tm, tk = x[f"model_{th}"], x[f"{key}_{th}"]
            if tm is not None and tk is not None:
                d.append(tm - tk)
                rel.append((tm - tk) / tm if tm > 0 else 0.0)
        pos, neg, p = sign_test(d)
        return med(rel), pos, neg, p

    lines.append("| graph | lr | runs | model reaches 0.5 / 0.9 (median step) | refit readout 0.5 / 0.9 | "
                 "drift-extrapolated (H = 1000) 0.5 / 0.9 | grokked |")
    lines.append("|---|---|---|---|---|---|---|")
    for (g, lr), xs in sorted(by_cell.items()):
        f2 = lambda k: f"{med([x[k + '_0.5'] for x in xs])} / {med([x[k + '_0.9'] for x in xs])}"
        lines.append(f"| {g} | {lr:g} | {len(xs)} | {f2('model')} | {f2('refit')} | {f2('ex1000')} | "
                     f"{sum(x['t_grok'] is not None for x in xs)}/{len(xs)} |")
    r9 = lead_stats("refit", 0.9)
    e5 = lead_stats("ex1000", 0.5)
    p1 = r9[0] is not None and r9[0] >= 0.2 and r9[1] > r9[2] and r9[3] < 0.05
    p1b = e5[0] is not None and e5[0] >= 0.1 and e5[1] > e5[2] and e5[3] < 0.05
    # jumps
    base_by = {(r["config"]["tag"].split("-")[1], r["config"]["lr"], r["config"]["model_seed"]): r for r in base}
    jrows = []
    for r in jumps:
        c = r["config"]
        b = base_by.get((c["tag"].split("-")[1], c["lr"], c["model_seed"]))
        tg_j = r["summary"]["t_grok"] if r["summary"]["t_grok"] is not None else c["steps"]
        tg_b = None if b is None else (b["summary"]["t_grok"] if b["summary"]["t_grok"] is not None else c["steps"])
        jm = r["summary"]["jump"]
        jrows.append({"graph": c["tag"].split("-")[1], "lr": c["lr"], "seed": c["model_seed"], "at": c["drift_jump_at"],
                      "h": c["drift_jump_h"], "before": jm["test_acc_before"], "after": jm["test_acc_after"],
                      "delta": None if tg_b is None else tg_b - tg_j,
                      "already_grokked": b is not None and b["summary"]["t_grok"] is not None and b["summary"]["t_grok"] <= c["drift_jump_at"],
                      "collapses": r["summary"]["n_collapses"]})
    lines += ["", "| graph | lr | jump at | H | runs | test acc before → right after (median) | median steps saved vs "
              "paired base | collapses (median) |", "|---|---|---|---|---|---|---|---|"]
    cells = defaultdict(list)
    for x in jrows:
        cells[(x["graph"], x["lr"], x["at"], x["h"])].append(x)
    for (g, lr, at, h), xs in sorted(cells.items()):
        lines.append(f"| {g} | {lr:g} | {at} | {h} | {len(xs)} | {fmt(med([x['before'] for x in xs]))} → "
                     f"{fmt(med([x['after'] for x in xs]))} | {med([x['delta'] for x in xs])} | {med([x['collapses'] for x in xs])} |")
    elig = [x for x in jrows if x["delta"] is not None and not x["already_grokked"]]
    jp, jn, jpv = sign_test([x["delta"] for x in elig])
    p2 = jp > jn and jpv < 0.05
    lines += ["", "## Verdicts (preregistered)",
              f"- **P1 (readout visibility)** the refit readout reaches 0.9 test accuracy earlier than the model, median "
              f"relative lead ≥ 0.2 with a sign test: median {fmt(r9[0])}, +{r9[1]}/−{r9[2]}, p = {fmt(r9[3], 4)} → "
              f"**{'supported' if p1 else 'NOT supported'}** (pilot expectation: not supported)",
              f"- **P1b (drift visibility)** jumping 1,000 steps ahead along the drift reaches 0.5 test accuracy earlier "
              f"than the model, median relative lead ≥ 0.1 with a sign test: median {fmt(e5[0])}, +{e5[1]}/−{e5[2]}, "
              f"p = {fmt(e5[3], 4)} → **{'supported' if p1b else 'NOT supported'}**",
              f"- **P2 (actionability)** a drift jump made before grokking makes the run grok earlier than its paired "
              f"base ({len(elig)} eligible runs): +{jp}/−{jn}, p = {fmt(jpv, 4)} → **{'supported' if p2 else 'NOT supported'}**",
              "- *Exploratory:* longer horizons (4,000 and 16,000 steps), the 0.5 threshold for the readout, and "
              "sym vs asym leads are in the json."]
    return lines, {"base": rows, "jumps": jrows, "P1": p1, "P1b": p1b, "P2": p2,
                   "leads": {"refit_0.9": r9, "ex1000_0.5": e5, "refit_0.5": lead_stats("refit", 0.5),
                             **{f"ex{H}_{th}": lead_stats(f"ex{H}", th) for H in (1000, 4000, 16000) for th in (0.5, 0.9)}}}


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
        rho = c["branch_lr_mult"][0] / c["branch_lr_mult"][1]
        tot_mem = [sum(x["mem"]) for x in tj]
        i_mem = int(np.argmax(tot_mem))
        rm = tj[i_mem]
        ma, mb = _branch(rm, "mem", A), _branch(rm, "mem", B)
        tail = [x for x in tj if x["step"] > c["steps"] - 2000]
        ga = stt.fmean(_branch(x, "gen", A) for x in tail)
        gb = stt.fmean(_branch(x, "gen", B) for x in tail)
        clip = lambda v: max(-1.0, min(2.0, v))
        s_mem = clip(ma / (ma + mb)) if (ma + mb) != 0 else float("nan")
        s_gen = clip(ga / (ga + gb)) if (ga + gb) != 0 else float("nan")
        rows.append({"graph": c["tag"].split("-")[1], "rho": rho, "seed": c["model_seed"], "s_gen": s_gen,
                     "s_mem": s_mem, "t_mem": rm["step"], "collapses": r["summary"]["n_collapses"],
                     "grokked": r["summary"]["grokked"]})
    lines = ["# E3 — credit by speed\n", f"{len(rows)} runs. Branch 0 (blocks 1–3) steps ρ× faster than branch 1 at "
             "equal lr·wd.\n", "| graph | ρ | share of gen (end) | share of mem (peak) | median |gen − mem| | "
             "collapses (median) | grokked |", "|---|---|---|---|---|---|---|"]
    verdict = {}
    for g in sorted({x["graph"] for x in rows}):
        xs = [x for x in rows if x["graph"] == g]
        for rho in sorted({x["rho"] for x in xs}):
            ys = [x for x in xs if x["rho"] == rho]
            lines.append(f"| {g} | {rho:g} | {fmt(med([y['s_gen'] for y in ys]))} | {fmt(med([y['s_mem'] for y in ys]))} | "
                         f"{fmt(med([abs(y['s_gen'] - y['s_mem']) for y in ys]))} | {med([y['collapses'] for y in ys])} | "
                         f"{sum(y['grokked'] for y in ys)}/{len(ys)} |")
        ok = [x for x in xs if not (math.isnan(x["s_gen"]) or math.isnan(x["s_mem"]))]
        tg, pg = perm_tau([x["rho"] for x in ok], [x["s_gen"] for x in ok], [x["seed"] for x in ok], rng)
        tm, pm = perm_tau([x["rho"] for x in ok], [x["s_mem"] for x in ok], [x["seed"] for x in ok], rng)
        diffs = {rho: med([abs(x["s_gen"] - x["s_mem"]) for x in ok if x["rho"] == rho]) for rho in sorted({x["rho"] for x in ok})}
        p1 = tg > 0 and pg < 0.05 and tm > 0 and pm < 0.05
        p2 = all(d is not None and d < 0.15 for d in diffs.values())
        split = (diffs.get(1.0) is not None and diffs[1.0] < 0.15
                 and any(d is not None and d >= 0.15 for r_, d in diffs.items() if r_ != 1.0))
        # exploratory: logit slope of the share against log rho
        def slope(key):
            xv = [math.log(x["rho"]) for x in ok]
            yv = [math.log(min(0.98, max(0.02, x[key])) / (1 - min(0.98, max(0.02, x[key])))) for x in ok]
            return float(np.polyfit(xv, yv, 1)[0]) if len(set(xv)) > 1 else float("nan")
        verdict[g] = {"tau_gen": tg, "p_gen": pg, "tau_mem": tm, "p_mem": pm, "P1": p1, "P2": p2, "role_split": split,
                      "diffs": diffs, "kappa_gen": slope("s_gen"), "kappa_mem": slope("s_mem")}
    lines.append("\n## Verdicts (preregistered)")
    for g, v in verdict.items():
        lines.append(f"- **{g}, P1** shares rise with ρ: τ(gen) = {v['tau_gen']:+.3f} (p = {v['p_gen']:.4f}), "
                     f"τ(mem) = {v['tau_mem']:+.3f} (p = {v['p_mem']:.4f}) → **{'supported' if v['P1'] else 'NOT supported'}**")
        lines.append(f"- **{g}, P2** co-location at every ρ (median |gen − mem| < 0.15): "
                     f"{ {k: fmt(d) for k, d in v['diffs'].items()} } → **{'supported' if v['P2'] else 'NOT supported'}**"
                     + (" — **speed induced a role split (falsifies H1)**" if v["role_split"] else ""))
        lines.append(f"- *{g}, exploratory:* logit slope κ(gen) = {v['kappa_gen']:.2f}, κ(mem) = {v['kappa_mem']:.2f} "
                     "(lazy-regime prediction for sym: 1; > 1 means rich-get-richer).")
    return lines, {"rows": rows, "verdict": verdict}


# ---------------------------------------------------------------------------- E4


def e4(rdir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("walk_make_grids", os.path.join(HERE, "make_grids.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    runs = load(rdir, "wE4")
    cell = defaultdict(list)
    for r in runs:
        c, s = r["config"], r["summary"]
        stable = s["grokked"] and s["diverged_at"] is None and (s["post_fit_min_train_acc"] or 0) >= 0.9
        cell[(c["tag"].split("-")[1], c["arm"], c["weight_decay"], c["lr"])].append((stable, s["t_grok"]))
    params, arms = ["std", "P1", "P2"], list(mg.E4_ARMS)
    lines = ["# E4 — stability regions\n", f"{len(runs)} runs. A cell is stable if ≥ 2 of 3 seeds grok with no "
             "post-fit collapse below 0.9 train accuracy and no divergence.\n"]
    out = {}
    for pn in params:
        lines += [f"\n## {pn}\n", "| arm (C) | wd | " + " | ".join(f"{lr:g}" for lr in mg.E4_LRS) + " | lr* |",
                  "|---|---|" + "---|" * len(mg.E4_LRS) + "---|"]
        for arm in arms:
            for wd in mg.E4_WDS:
                marks, lstar = [], None
                for lr in mg.E4_LRS:
                    v = cell.get((pn, arm, wd, lr), [])
                    ok = len(v) >= 2 and sum(x[0] for x in v) >= 2
                    marks.append("✓" if ok else "·")
                    if ok:
                        lstar = lr
                out[(pn, arm, wd)] = lstar
                lines.append(f"| {arm} ({mg.E4_ARMS[arm]}) | {wd:g} | " + " | ".join(marks) + f" | {fmt(lstar)} |")
    steps = lambda a, b: None if (a is None or b is None) else round(math.log2(a / b))
    lines.append("\n## Verdicts (preregistered, wd = 1)")
    res = {}
    for pn in params:
        l1, l4 = out[(pn, "pk0000", 1.0)], out[(pn, "pk0123", 1.0)]
        okc = lambda v: len(v) >= 2 and sum(x[0] for x in v) >= 2
        st1 = {(lr, wd) for lr in mg.E4_LRS for wd in mg.E4_WDS if okc(cell.get((pn, "pk0000", wd, lr), []))}
        st4 = {(lr, wd) for lr in mg.E4_LRS for wd in mg.E4_WDS if okc(cell.get((pn, "pk0123", wd, lr), []))}
        jac = len(st1 & st4) / len(st1 | st4) if (st1 | st4) else None
        taus = kendall([mg.E4_ARMS[a] for a in arms], [math.log(out[(pn, a, 1.0)]) if out[(pn, a, 1.0)] else math.log(mg.E4_LRS[0] / 2) for a in arms])
        res[pn] = {"lr*_C1": l1, "lr*_C4": l4, "grid_steps_C1_minus_C4": steps(l1, l4) if l4 else (None if l1 is None else 99),
                   "jaccard_lr_sets": jac, "tau_C_loglr*": taus}
        lines.append(f"- {pn}: lr*(C1) = {fmt(l1)}, lr*(C4) = {fmt(l4)}, gap = {res[pn]['grid_steps_C1_minus_C4']} grid steps; "
                     f"τ(C, log lr*) over the 4 arms = {taus:+.2f}; Jaccard of stable (lr, wd) cells C1 vs C4 = {fmt(jac)}")
    std_gap = res["std"]["grid_steps_C1_minus_C4"]
    for pn in ("P1", "P2"):
        g = res[pn]["grid_steps_C1_minus_C4"]
        sup = std_gap is not None and std_gap >= 2 and g is not None and g <= 1
        res[pn]["H8_supported"] = sup
        lines.append(f"- **H8 under {pn}:** standard gap ≥ 2 steps and {pn} gap ≤ 1 step → **{'supported' if sup else 'NOT supported'}**")
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
