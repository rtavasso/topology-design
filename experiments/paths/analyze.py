"""Score Study 4 (what each path does) against experiments/paths/PREREGISTRATION.md.

Usage:
  PYTHONPATH=src python experiments/paths/analyze.py --runs results/paths/stage1 \
      --selection results/paths/selection.json --out results/paths/report

Writes <out>/summary.md, <out>/compact.json (per-run time-point snapshots + thinned trajectories, small enough to
transfer), and figures when matplotlib is available.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from topology_design import grokking as gk

MIXED = ["pk0100", "pk0111", "pk0120", "pk0122", "pk0123"]
ALL = ["pk0000"] + MIXED
H2_CONTRASTS = {"H2a": ("pk0120", "pk0123", [2, 3]), "H2b": ("pk0100", "pk0120", [2])}  # (A: extra short reader, B, blocks)
BUDGET = 40000
N_BOOT, N_PERM = 10000, 10000
RNG = np.random.default_rng(20261021)


# ---------------------------------------------------------------------------
# loading and time points
# ---------------------------------------------------------------------------


def load(runs_dir: str) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        r = json.load(open(f))
        c = r["config"]
        parents, _ = gk.arm_spec(c["arm"])
        runs.append({"arm": c["arm"], "seed": c["model_seed"], "lr": c["lr"], "depth": gk.block_depths(parents),
                     "C": gk.longest_chain(parents), "Q": gk.root_readers(parents),
                     "traj": r["trajectory"], "summary": r["summary"]})
    return runs


def timepoints(traj: list[dict]) -> dict[str, int | None]:
    """Indices into traj: t_fit, t_mem (first peak of excluded-train accuracy), t_grok, t_end."""
    def first(pred):
        return next((i for i, r in enumerate(traj) if pred(r)), None)

    peak = max(r["excluded_train_acc"] for r in traj)
    return {"t_fit": first(lambda r: r["train_acc"] >= 0.99),
            "t_mem": first(lambda r: r["excluded_train_acc"] >= peak),
            "t_grok": first(lambda r: r["test_acc"] >= 0.95),
            "t_end": len(traj) - 1}


def mask(blocks: list[int], L: int = 4) -> str:
    """Blocks numbered 1..L -> abl_sets key."""
    return "".join("1" if (i + 1) in blocks else "0" for i in range(L))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bootstrap_mean(v, n=N_BOOT):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "boot": np.array([])}
    boot = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "n": int(len(v)), "boot": boot}


def p_one_sided(boot, direction: str) -> float | None:
    if len(boot) == 0:
        return None
    wrong = (boot <= 0) if direction == ">" else (boot >= 0)
    return float(max(wrong.mean(), 1.0 / len(boot)))


def kendall_tau_b(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    dx, dy = dx[iu], dy[iu]
    s = (dx * dy).sum()
    n0 = len(dx)
    denom = math.sqrt((n0 - (dx == 0).sum()) * (n0 - (dy == 0).sum()))
    return float(s / denom) if denom > 0 else 0.0


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted([(p, k) for k, p in pvals.items() if p is not None])
    m, out, run = len(items), {k: None for k in pvals}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------


def h1_D(run: dict) -> float | None:
    """Short-path share of block gen at t_end minus short-path share of block mem at t_mem (None if undefined)."""
    tp = timepoints(run["traj"])
    short = [j for j, d in enumerate(run["depth"], start=1) if d == 1]  # component index = block number
    pe, pm = run["traj"][tp["t_end"]]["paths"], run["traj"][tp["t_mem"]]["paths"]
    gen_blocks, mem_blocks = sum(pe["gen"][1:]), sum(pm["mem"][1:])
    run["_h1_den"] = (gen_blocks, mem_blocks)
    if gen_blocks == 0 or mem_blocks == 0:
        return None
    return sum(pe["gen"][j] for j in short) / gen_blocks - sum(pm["mem"][j] for j in short) / mem_blocks


def h1(runs: list[dict], graphs: list[str]) -> dict:
    per_graph, excluded = defaultdict(dict), defaultdict(int)
    for g in graphs:
        rs = [r for r in runs if r["arm"] == g]
        vals = {r["seed"]: h1_D(r) for r in rs}
        med_g = np.median([abs(r["_h1_den"][0]) for r in rs]) if rs else 0.0
        med_m = np.median([abs(r["_h1_den"][1]) for r in rs]) if rs else 0.0
        for r in rs:
            gden, mden = r["_h1_den"]
            if vals[r["seed"]] is None or abs(gden) < 0.05 * med_g or abs(mden) < 0.05 * med_m:
                excluded[g] += 1
                continue
            per_graph[g][r["seed"]] = vals[r["seed"]]
    seeds = sorted({s for g in per_graph for s in per_graph[g]})
    per_seed = [np.mean([per_graph[g][s] for g in graphs if s in per_graph[g]]) for s in seeds]
    bm = bootstrap_mean(per_seed)
    return {"pooled": {k: v for k, v in bm.items() if k != "boot"}, "p": p_one_sided(bm["boot"], ">"),
            "per_graph": {g: {k: v for k, v in bootstrap_mean(list(per_graph[g].values())).items() if k != "boot"}
                          for g in graphs},
            "excluded": dict(excluded)}


def utility(run: dict, blocks: list[int], when: str = "t_end") -> float | None:
    tp = timepoints(run["traj"])
    idx = tp[when]
    if idx is None:
        return None
    pa = run["traj"][idx]["paths"]
    g = pa["gen_total"]
    if g <= 0:
        return None
    return 1.0 - pa["abl_sets"][mask(blocks)]["gen"] / g


def h2(runs: list[dict]) -> dict:
    out = {}
    by = {(r["arm"], r["seed"]): r for r in runs}
    for name, (A, B, blocks) in H2_CONTRASTS.items():
        seeds = sorted({s for (a, s) in by if a == A} & {s for (a, s) in by if a == B})
        diffs, dropped = [], 0
        for s in seeds:
            ua, ub = utility(by[(A, s)], blocks), utility(by[(B, s)], blocks)
            if ua is None or ub is None:
                dropped += 1
                continue
            diffs.append(ua - ub)
        bm = bootstrap_mean(diffs)
        out[name] = {"A": A, "B": B, "blocks": blocks, **{k: v for k, v in bm.items() if k != "boot"},
                     "p": p_one_sided(bm["boot"], "<"), "dropped": dropped,
                     "U_A": nanmean([utility(by[(A, s)], blocks) for s in seeds]),
                     "U_B": nanmean([utility(by[(B, s)], blocks) for s in seeds])}
    ps = [out[k]["p"] for k in out]
    out["p"] = max(ps) if all(p is not None for p in ps) else None  # intersection-union: both must hold
    out["supported"] = all(out[k]["hi"] is not None and out[k]["hi"] < 0 for k in H2_CONTRASTS)
    return out


def nanmean(v) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def delay(run: dict) -> float:
    s = run["summary"]
    if s["t_fit"] is None or s["t_grok"] is None:
        return 1e9  # censored: ties at the top
    return float(s["t_grok"] - s["t_fit"])


def h3(runs: list[dict], graphs: list[str]) -> dict:
    rs = [r for r in runs if r["arm"] in graphs]
    C = np.array([r["C"] for r in rs], float)
    D = np.array([delay(r) for r in rs], float)
    tau = kendall_tau_b(C, D)
    seeds = np.array([r["seed"] for r in rs])
    groups = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    count = 0
    for _ in range(N_PERM):
        Cp = C.copy()
        for g in groups:
            Cp[g] = C[g][RNG.permutation(len(g))]
        count += kendall_tau_b(Cp, D) >= tau
    med = {g: float(np.median([delay(r) for r in rs if r["arm"] == g])) for g in graphs if any(r["arm"] == g for r in rs)}
    return {"tau": tau, "p": (count + 1) / (N_PERM + 1), "n": len(rs),
            "censored": int((D >= 1e9).sum()), "median_delay": med}


# ---------------------------------------------------------------------------
# exploratory
# ---------------------------------------------------------------------------


def snapshot(run: dict, idx: int | None) -> dict | None:
    if idx is None:
        return None
    r = run["traj"][idx]
    pa = r["paths"]
    return {"step": r["step"], "train_acc": r["train_acc"], "test_acc": r["test_acc"], "train_loss": r["train_loss"],
            "rule_frac": r["rule_frac"], "excluded_train_acc": r["excluded_train_acc"],
            "grad_norms": r.get("grad_norms"), "residual_norm": r.get("residual_norm"),
            **{k: pa[k] for k in ("gen", "mem", "noise", "gen_total", "mem_total", "shap_rule", "shap_mem", "own_rule",
                                  "own_mem", "comp_dep_state", "comp_dep_gen", "comp_dep_mem", "abl_short", "abl_long",
                                  "abl_sets")}}


def exploratory(runs: list[dict]) -> dict:
    ex = {"role_map": {}, "destruction": {}, "effective_depth": {}, "post_fit_long": {}, "grad_at_loss": {},
          "qc": {}, "preservation": {}}
    for g in ALL:
        rs = [r for r in runs if r["arm"] == g]
        if not rs:
            continue
        depth = rs[0]["depth"]
        long_b = [j for j, d in enumerate(depth, start=1) if d >= 2]
        short_b = [j for j, d in enumerate(depth, start=1) if d == 1]
        # E1 role map
        rm = {}
        for tpn in ("t_fit", "t_mem", "t_grok", "t_end"):
            snaps = [snapshot(r, timepoints(r["traj"])[tpn]) for r in rs]
            snaps = [s for s in snaps if s is not None]
            if snaps:
                rm[tpn] = {k: np.mean([s[k] for s in snaps], 0).round(4).tolist() for k in ("gen", "mem", "noise")}
                rm[tpn]["n"] = len(snaps)
        ex["role_map"][g] = rm
        # E2 destruction
        dest, canc = [], defaultdict(list)
        for r in rs:
            tp = timepoints(r["traj"])
            for tpn in ("t_mem", "t_grok", "t_end"):
                if tp[tpn] is not None:
                    pa = r["traj"][tp[tpn]]["paths"]
                    if abs(pa["mem_total"]) > 1e-6:
                        canc[tpn].append(sum(abs(m) for m in pa["mem"]) / abs(pa["mem_total"]))
            pa = r["traj"][tp["t_end"]]["paths"]
            neg = [j for j, m in enumerate(pa["mem"]) if m < -0.1 * abs(pa["mem_total"])]
            dest.append(neg)
        ex["destruction"][g] = {"frac_runs_with_destroyer_at_end": float(np.mean([len(d) > 0 for d in dest])),
                                "destroyer_components_at_end": dest,
                                "cancellation_ratio_median": {k: float(np.median(v)) for k, v in canc.items()}}
        # E3 effective depth
        if long_b:
            ed = []
            for r in rs:
                pa = r["traj"][timepoints(r["traj"])["t_end"]]["paths"]
                ed.append({"comp_dep_state": [pa["comp_dep_state"][j - 1] for j in long_b],
                           "composed_share_of_gen": [pa["comp_dep_gen"][j - 1] / pa["gen"][j] if abs(pa["gen"][j]) > 1e-6
                                                     else None for j in long_b]})
            ex["effective_depth"][g] = {"long_blocks": long_b,
                                        "comp_dep_state_mean": np.mean([e["comp_dep_state"] for e in ed], 0).round(3).tolist(),
                                        "per_run": ed}
        # E4 post-fit long-path learning
        if long_b:
            rows = []
            for r in rs:
                tp = timepoints(r["traj"])
                row = {}
                for tpn in ("t_fit", "t_grok", "t_end"):
                    if tp[tpn] is None:
                        continue
                    pa = r["traj"][tp[tpn]]["paths"]
                    gb = sum(pa["gen"][1:])
                    row[f"long_gen_share@{tpn}"] = sum(pa["gen"][j] for j in long_b) / gb if abs(gb) > 1e-6 else None
                    row[f"U_long@{tpn}"] = utility(r, long_b, tpn)
                rows.append(row)
            keys = sorted({k for row in rows for k in row})
            ex["post_fit_long"][g] = {k: float(np.nanmedian([row.get(k) if row.get(k) is not None else np.nan
                                                             for row in rows])) for k in keys}
        # E5 gradient allocation at matched train loss
        gal = {}
        for thr in (1.0, 0.1, 0.01):
            sh, lo = [], []
            for r in rs:
                rec = next((x for x in r["traj"] if x["train_loss"] <= thr), None)
                if rec is None or not rec.get("grad_norms"):
                    continue
                gn = rec["grad_norms"]
                sh.append(np.mean([gn[f"block{j}"] for j in short_b]))
                if long_b:
                    lo.append(np.mean([gn[f"block{j}"] for j in long_b]))
            gal[str(thr)] = {"short_mean_grad": float(np.median(sh)) if sh else None,
                             "long_mean_grad": float(np.median(lo)) if lo else None, "n": len(sh)}
        ex["grad_at_loss"][g] = gal
        # E6 Q-C descriptors
        qc = []
        for r in rs:
            tp = timepoints(r["traj"])
            s = r["summary"]
            qc.append({"delay": delay(r), "peak_excl": s["peak_excluded_train_acc"],
                       "persistence": (r["traj"][tp["t_grok"]]["step"] - r["traj"][tp["t_mem"]]["step"])
                       if tp["t_grok"] is not None else None,
                       "final_rule_frac": s["final_rule_frac"], "stable": s["stable"], "grokked": s["grokked"]})
        ex["qc"][g] = {"C": rs[0]["C"], "Q": rs[0]["Q"], "lr": rs[0]["lr"], "n": len(rs),
                       "median_delay": float(np.median([q["delay"] for q in qc])),
                       "median_peak_excl": float(np.median([q["peak_excl"] for q in qc])),
                       "median_persistence": float(np.median([q["persistence"] for q in qc if q["persistence"] is not None]))
                       if any(q["persistence"] is not None for q in qc) else None,
                       "median_final_rule_frac": float(np.median([q["final_rule_frac"] for q in qc])),
                       "stable_frac": float(np.mean([q["stable"] for q in qc])),
                       "grok_frac": float(np.mean([q["grokked"] for q in qc]))}
        # E3b preservation: embedding path's own vs Shapley rule/mem energy at t_end
        pres = []
        for r in rs:
            pa = r["traj"][timepoints(r["traj"])["t_end"]]["paths"]
            pres.append({"gen0": pa["gen"][0], "mem0": pa["mem"][0],
                         "rule_cancelled": pa["own_rule"][0] - pa["shap_rule"][0],
                         "mem_cancelled": pa["own_mem"][0] - pa["shap_mem"][0]})
        ex["preservation"][g] = {k: float(np.median([p[k] for p in pres])) for k in pres[0]}
    return ex


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def compact(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        tp = timepoints(r["traj"])
        thin = [{"step": x["step"], "train_acc": x["train_acc"], "test_acc": x["test_acc"],
                 "excluded_train_acc": x["excluded_train_acc"], "gen": x["paths"]["gen"], "mem": x["paths"]["mem"],
                 "gen_without_short": None if x["paths"]["abl_short"] is None else x["paths"]["abl_short"]["gen"],
                 "gen_without_long": None if x["paths"]["abl_long"] is None else x["paths"]["abl_long"]["gen"],
                 "gen_total": x["paths"]["gen_total"]}
                for i, x in enumerate(r["traj"]) if i % 5 == 0 or i == len(r["traj"]) - 1]
        out.append({"arm": r["arm"], "seed": r["seed"], "lr": r["lr"], "depth": r["depth"], "C": r["C"], "Q": r["Q"],
                    "summary": {k: v for k, v in r["summary"].items() if k not in ("final_branches", "final_ablate")},
                    "timepoints": {k: snapshot(r, v) for k, v in tp.items()}, "thin": thin})
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x >= 1e9:
        return "cens."
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def figures(runs: list[dict], out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    graphs = [g for g in ALL if any(r["arm"] == g for r in runs)]
    fig, axes = plt.subplots(2, len(graphs), figsize=(3.2 * len(graphs), 5.5), sharex=False, squeeze=False)
    colors = ["#888888", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for col, g in enumerate(graphs):
        rs = [r for r in runs if r["arm"] == g]
        n = min(len(r["traj"]) for r in rs)
        steps = [rs[0]["traj"][i]["step"] for i in range(n)]
        for row, key in enumerate(("gen", "mem")):
            ax = axes[row][col]
            for j in range(5):
                m = np.mean([[r["traj"][i]["paths"][key][j] for i in range(n)] for r in rs], 0)
                lab = "emb" if j == 0 else f"b{j} (d{rs[0]['depth'][j - 1]})"
                ax.plot(steps, m, color=colors[j], lw=1.2, label=lab)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{g}: {key}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{key} (label-logit support)")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Study 4: what each path does (mean over seeds; truncated at shortest run)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roles.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.runs)
    sel = json.load(open(args.selection)) if args.selection else {}
    flagged = [g for g, v in sel.items() if v["tier"] != "stable"]
    res = {"H1": h1(runs, MIXED), "H2": h2(runs), "H3": h3(runs, ALL)}
    holm_p = holm({k: res[k]["p"] for k in res})
    res["holm"] = holm_p
    clean = [g for g in ALL if g not in flagged]
    res["without_flagged"] = {"flagged": flagged, "H1": h1(runs, [g for g in MIXED if g in clean]),
                              "H3": h3(runs, clean) if len(clean) > 1 else None}
    ex = exploratory(runs)
    json.dump({"results": res, "exploratory": ex, "selection": sel, "runs": compact(runs)},
              open(os.path.join(args.out, "compact.json"), "w"))
    figures(runs, args.out)

    L = ["# Study 4 automatic report", "", f"{len(runs)} Stage-1 runs. Selection: " +
         ", ".join(f"{g} lr {v['lr']:g} ({v['tier']})" for g, v in sel.items()), ""]
    h = res["H1"]
    L += ["## H1 — short paths construct, long paths memorize (D > 0)",
          f"pooled D = {fmt(h['pooled']['mean'])} [{fmt(h['pooled']['lo'])}, {fmt(h['pooled']['hi'])}], n seeds = "
          f"{h['pooled']['n']}, p = {fmt(h['p'], 4)}, Holm p = {fmt(holm_p['H1'], 4)}; excluded runs {h['excluded']}",
          "", "| graph | D | 95% CI | n |", "|---|---|---|---|"]
    L += [f"| {g} | {fmt(v['mean'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {v['n']} |" for g, v in h["per_graph"].items()]
    h = res["H2"]
    L += ["", "## H2 — an extra short reader lowers the long blocks' utility (A − B < 0 in both)",
          f"supported: {h['supported']}; intersection-union p = {fmt(h['p'], 4)}, Holm p = {fmt(holm_p['H2'], 4)}", "",
          "| contrast | blocks | U(A) | U(B) | mean A−B | 95% CI | n | dropped |", "|---|---|---|---|---|---|---|---|"]
    for k in H2_CONTRASTS:
        v = h[k]
        L.append(f"| {k}: {v['A']} vs {v['B']} | {v['blocks']} | {fmt(v['U_A'])} | {fmt(v['U_B'])} | {fmt(v['mean'])} | "
                 f"[{fmt(v['lo'])}, {fmt(v['hi'])}] | {v['n']} | {v['dropped']} |")
    h = res["H3"]
    L += ["", "## H3 — composition delays generalization (τ(C, delay) > 0)",
          f"τ_b = {fmt(h['tau'])}, permutation p = {fmt(h['p'], 4)}, Holm p = {fmt(holm_p['H3'], 4)}, n = {h['n']}, "
          f"censored = {h['censored']}", "", "median delay: " + ", ".join(f"{g} {fmt(v, 0)}" for g, v in h["median_delay"].items())]
    w = res["without_flagged"]
    L += ["", f"## Without flagged graphs {w['flagged']}",
          f"H1 pooled D = {fmt(w['H1']['pooled']['mean'])} [{fmt(w['H1']['pooled']['lo'])}, {fmt(w['H1']['pooled']['hi'])}]",
          f"H3 τ = {fmt(w['H3']['tau']) if w['H3'] else '—'}, p = {fmt(w['H3']['p'], 4) if w['H3'] else '—'}"]
    L += ["", "## Exploratory (no claims)", "", "### Q-C descriptors", "",
          "| graph | C | Q | lr | grok frac | stable frac | median delay | peak excl-train acc | persistence | final rule frac |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, q in ex["qc"].items():
        L.append(f"| {g} | {q['C']} | {q['Q']} | {q['lr']:g} | {fmt(q['grok_frac'], 2)} | {fmt(q['stable_frac'], 2)} | "
                 f"{fmt(q['median_delay'], 0)} | {fmt(q['median_peak_excl'])} | {fmt(q['median_persistence'], 0)} | "
                 f"{fmt(q['median_final_rule_frac'])} |")
    L += ["", "### Role map at t_end (mean gen / mem per component: emb, b1..b4)", ""]
    for g, rm in ex["role_map"].items():
        if "t_end" in rm:
            L.append(f"- {g}: gen {rm['t_end']['gen']}, mem {rm['t_end']['mem']}"
                     + (f"; at t_mem mem {rm['t_mem']['mem']}" if "t_mem" in rm else ""))
    L += ["", "### Destruction", ""]
    for g, d in ex["destruction"].items():
        L.append(f"- {g}: runs with a destroyer at end {fmt(d['frac_runs_with_destroyer_at_end'], 2)}; "
                 f"cancellation ratio (median) {{{', '.join(f'{k}: {v:.2f}' for k, v in d['cancellation_ratio_median'].items())}}}")
    L += ["", "### Effective depth (comp_dep_state of long blocks at t_end)", ""]
    for g, e in ex["effective_depth"].items():
        L.append(f"- {g}: long blocks {e['long_blocks']} → {e['comp_dep_state_mean']}")
    L += ["", "### Post-fit long-path learning (medians)", ""]
    for g, e in ex["post_fit_long"].items():
        L.append(f"- {g}: " + ", ".join(f"{k} {fmt(v)}" for k, v in e.items()))
    L += ["", "### Gradient allocation at matched train loss (median mean-block grad norm, short / long)", ""]
    for g, e in ex["grad_at_loss"].items():
        L.append(f"- {g}: " + "; ".join(f"loss≤{t}: {fmt(v['short_mean_grad'], 4)} / {fmt(v['long_mean_grad'], 4)}"
                                        for t, v in e.items()))
    L += ["", "### Preservation (embedding path, medians at t_end)", ""]
    for g, e in ex["preservation"].items():
        L.append(f"- {g}: " + ", ".join(f"{k} {fmt(v)}" for k, v in e.items()))
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
