"""Score Study 4 (what each path does) against experiments/paths/PREREGISTRATION.md.

Usage:
  PYTHONPATH=src python experiments/paths/analyze.py --runs results/paths/stage1 --stage0 results/paths/stage0 \
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

ALL = ["pk0000", "pk0100", "pk0110", "pk0111", "pk0120", "pk0121", "pk0122", "pk0123"]
MIXED = ALL[1:]
# (A: has the extra short reader, B: same graph with that block reading h_1 instead, S: blocks whose utility is compared)
# In both A and B the rewired block does NOT read S, so removing S leaves its input unchanged in both.
H2_CONTRASTS = {"H2a": ("pk0120", "pk0121", [2, 3]), "H2b": ("pk0100", "pk0110", [2])}
N_END = 5  # t_end = mean over the last N_END evals
N_BOOT, N_PERM = 10000, 10000
CLIP = (-1.0, 2.0)  # share clipping for H1
RNG = np.random.default_rng(20261021)


# ---------------------------------------------------------------------------
# loading, time points, end averages
# ---------------------------------------------------------------------------


def load(runs_dir: str) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        r = json.load(open(f))
        c = r["config"]
        parents, _ = gk.arm_spec(c["arm"])
        runs.append({"arm": c["arm"], "seed": c["model_seed"], "lr": c["lr"], "wd": c["weight_decay"],
                     "depth": gk.block_depths(parents), "C": gk.longest_chain(parents), "Q": gk.root_readers(parents),
                     "traj": r["trajectory"], "summary": r["summary"]})
    return runs


def timepoints(traj: list[dict]) -> dict[str, int | None]:
    """Indices into traj: t_fit, t_mem (peak of total train-specific support mem_total), t_grok, t_end (last eval)."""
    def first(pred):
        return next((i for i, r in enumerate(traj) if pred(r)), None)

    mems = [r["paths"]["mem_total"] for r in traj] if "paths" in traj[0] else [r["excluded_train_acc"] for r in traj]
    return {"t_fit": first(lambda r: r["train_acc"] >= 0.99), "t_mem": int(np.argmax(mems)),
            "t_grok": first(lambda r: r["test_acc"] >= 0.95), "t_end": len(traj) - 1}


def endavg(run: dict, f) -> float | None:
    vals = [f(r) for r in run["traj"][-N_END:]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def endavg_vec(run: dict, key: str) -> list[float]:
    return np.mean([r["paths"][key] for r in run["traj"][-N_END:]], 0).tolist()


def mask(blocks: list[int], L: int = 4) -> str:
    """Blocks numbered 1..L -> abl_sets key (blocks removed)."""
    return "".join("1" if (i + 1) in blocks else "0" for i in range(L))


def grokked(run: dict) -> bool:
    return run["summary"]["t_grok"] is not None


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


def strip(bm: dict) -> dict:
    return {k: v for k, v in bm.items() if k != "boot"}


def p_one_sided(boot, direction: str) -> float | None:
    if len(boot) == 0:
        return None
    wrong = (boot <= 0) if direction == ">" else (boot >= 0)
    return float(max(wrong.mean(), 1.0 / len(boot)))


def kendall_tau_b(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    iu = np.triu_indices(len(x), 1)
    dx = np.sign(x[:, None] - x[None, :])[iu]
    dy = np.sign(y[:, None] - y[None, :])[iu]
    n0 = len(dx)
    denom = math.sqrt((n0 - (dx == 0).sum()) * (n0 - (dy == 0).sum()))
    return float((dx * dy).sum() / denom) if denom > 0 else 0.0


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted([(p, k) for k, p in pvals.items() if p is not None])
    m, out, run = len(items), {k: None for k in pvals}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------------------
# H1: roles split by depth, under direct and total-effect attribution
# ---------------------------------------------------------------------------


def shares(run: dict, attribution: str) -> tuple[float | None, float | None, float, float]:
    """(short share of block gen at t_end [mean of last evals], short share of block mem at t_mem, denominators)."""
    tp = timepoints(run["traj"])
    short = [j for j, d in enumerate(run["depth"], start=1) if d == 1]  # block numbers
    if attribution == "direct":
        gen = endavg_vec(run, "gen")[1:]  # drop the embedding component -> index 0 = block 1
        mem = run["traj"][tp["t_mem"]]["paths"]["mem"][1:]
    else:
        gen = endavg_vec(run, "gen_tshap")
        mem = run["traj"][tp["t_mem"]]["paths"]["mem_tshap"]
    gden, mden = sum(gen), sum(mem)
    clip = lambda x: float(np.clip(x, *CLIP))
    gs = clip(sum(gen[j - 1] for j in short) / gden) if gden != 0 else None
    ms = clip(sum(mem[j - 1] for j in short) / mden) if mden != 0 else None
    return gs, ms, gden, mden


def h1(runs: list[dict], graphs: list[str], attribution: str) -> dict:
    per_graph, excluded, not_grokked = defaultdict(dict), defaultdict(int), defaultdict(int)
    for g in graphs:
        rs = [r for r in runs if r["arm"] == g]
        not_grokked[g] = sum(not grokked(r) for r in rs)
        rs = [r for r in rs if grokked(r)]
        if not rs:
            continue
        sh = {r["seed"]: shares(r, attribution) for r in rs}
        med_g = np.median([abs(v[2]) for v in sh.values()])
        med_m = np.median([abs(v[3]) for v in sh.values()])
        for seed, (gs, ms, gden, mden) in sh.items():
            if gs is None or ms is None or abs(gden) < 0.05 * med_g or abs(mden) < 0.05 * med_m:
                excluded[g] += 1
                continue
            per_graph[g][seed] = gs - ms
    seeds = sorted({s for g in per_graph for s in per_graph[g]})
    per_seed = [np.mean([per_graph[g][s] for g in graphs if s in per_graph[g]]) for s in seeds]
    bm = bootstrap_mean(per_seed)
    return {"pooled": strip(bm), "p": p_one_sided(bm["boot"], ">"),
            "per_graph": {g: strip(bootstrap_mean(list(per_graph[g].values()))) for g in graphs},
            "excluded": dict(excluded), "not_grokked": dict(not_grokked)}


def h1_both(runs, graphs):
    d, t = h1(runs, graphs, "direct"), h1(runs, graphs, "total")
    ok = all(x["pooled"]["lo"] is not None and x["pooled"]["lo"] > 0 for x in (d, t))
    ps = [d["p"], t["p"]]
    return {"direct": d, "total": t, "p": max(ps) if None not in ps else None, "ci_both_above_0": ok}


# ---------------------------------------------------------------------------
# H2: friction
# ---------------------------------------------------------------------------


def utility_at(rec: dict, blocks: list[int]) -> float | None:
    pa = rec["paths"]
    g = pa["gen_total"]
    if g <= 0:
        return None
    return 1.0 - pa["abl_sets"][mask(blocks)]["gen"] / g


def utility(run: dict, blocks: list[int], when: str = "t_end") -> float | None:
    if when == "t_end":
        return endavg(run, lambda r: utility_at(r, blocks))
    idx = timepoints(run["traj"])[when]
    return None if idx is None else utility_at(run["traj"][idx], blocks)


def cmedian(v) -> float:
    """Median with censored values (>= 1e9) treated as +inf, so a censored median prints as censored."""
    return float(np.median([np.inf if x >= 1e9 else x for x in v]))


def nanmean(v) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def h2(runs: list[dict], contrasts=H2_CONTRASTS) -> dict:
    out = {}
    by = {(r["arm"], r["seed"]): r for r in runs}
    for name, (A, B, blocks) in contrasts.items():
        seeds = sorted({s for (a, s) in by if a == A} & {s for (a, s) in by if a == B})
        diffs, not_grok, undefined = [], 0, 0
        for s in seeds:
            ra, rb = by[(A, s)], by[(B, s)]
            if not (grokked(ra) and grokked(rb)):
                not_grok += 1
                continue
            ua, ub = utility(ra, blocks), utility(rb, blocks)
            if ua is None or ub is None:
                undefined += 1
                continue
            diffs.append(ua - ub)
        bm = bootstrap_mean(diffs)
        used = [s for s in seeds if grokked(by[(A, s)]) and grokked(by[(B, s)])]
        out[name] = {"A": A, "B": B, "blocks": blocks, **strip(bm), "p": p_one_sided(bm["boot"], "<"),
                     "pairs_not_both_grokked": not_grok, "pairs_undefined": undefined,
                     "U_A": nanmean([utility(by[(A, s)], blocks) for s in used]),
                     "U_B": nanmean([utility(by[(B, s)], blocks) for s in used])}
    ps = [out[k]["p"] for k in contrasts]
    out["p"] = max(ps) if all(p is not None for p in ps) else None  # intersection-union: both must hold
    out["ci_both_below_0"] = all(out[k]["hi"] is not None and out[k]["hi"] < 0 for k in contrasts)
    return out


# ---------------------------------------------------------------------------
# H3: composition delays generalization
# ---------------------------------------------------------------------------


def delay(run: dict, normalized: bool = False) -> float:
    s = run["summary"]
    if s["t_fit"] is None or s["t_grok"] is None:
        return 1e9  # censored: ties at the top
    d = float(s["t_grok"] - s["t_fit"])
    return d * run["lr"] * run["wd"] if normalized else d


def h3(runs: list[dict], graphs: list[str], normalized: bool = False, n_perm=N_PERM) -> dict | None:
    rs = [r for r in runs if r["arm"] in graphs]
    if len({r["C"] for r in rs}) < 2:
        return None
    C = np.array([r["C"] for r in rs], float)
    D = np.array([delay(r, normalized) for r in rs], float)
    seeds = np.array([r["seed"] for r in rs])
    tau = kendall_tau_b(C, D)
    groups = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    count = 0
    for _ in range(n_perm):
        Cp = C.copy()
        for g in groups:
            Cp[g] = C[g][RNG.permutation(len(g))]
        count += kendall_tau_b(Cp, D) >= tau
    boots = []
    for _ in range(1000):  # bootstrap over seeds (each seed keeps all its graphs)
        pick = RNG.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        boots.append(kendall_tau_b(C[idx], D[idx]))
    med = {g: cmedian([delay(r, normalized) for r in rs if r["arm"] == g]) for g in graphs
           if any(r["arm"] == g for r in rs)}
    return {"tau": tau, "p": (count + 1) / (n_perm + 1), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "n": len(rs), "censored": int((D >= 1e9).sum()), "median_delay": med}


# ---------------------------------------------------------------------------
# exploratory
# ---------------------------------------------------------------------------


SNAP_KEYS = ("gen", "mem", "gen_total", "mem_total", "gen_tshap", "mem_tshap", "shap_rule", "shap_mem", "own_rule",
             "own_mem", "comp_dep_state", "comp_dep_gen", "comp_dep_mem", "abl_short", "abl_long", "abl_sets")


def snapshot(run: dict, idx: int | None) -> dict | None:
    if idx is None:
        return None
    r = run["traj"][idx]
    return {"step": r["step"], "train_acc": r["train_acc"], "test_acc": r["test_acc"], "train_loss": r["train_loss"],
            "rule_frac": r["rule_frac"], "excluded_train_acc": r["excluded_train_acc"],
            "grad_norms": r.get("grad_norms"), "residual_norm": r.get("residual_norm"),
            **{k: r["paths"][k] for k in SNAP_KEYS}}


def exploratory(runs: list[dict]) -> dict:
    ex = defaultdict(dict)
    for g in ALL:
        rs = [r for r in runs if r["arm"] == g]
        if not rs:
            continue
        depth = rs[0]["depth"]
        long_b = [j for j, d in enumerate(depth, start=1) if d >= 2]
        short_b = [j for j, d in enumerate(depth, start=1) if d == 1]
        # E1 role map (direct components emb, b1..b4; total-effect blocks b1..b4)
        rm = {}
        for tpn in ("t_fit", "t_mem", "t_grok", "t_end"):
            snaps = [snapshot(r, timepoints(r["traj"])[tpn]) for r in rs]
            snaps = [s for s in snaps if s is not None]
            if snaps:
                rm[tpn] = {k: np.mean([s[k] for s in snaps], 0).round(3).tolist()
                           for k in ("gen", "mem", "gen_tshap", "mem_tshap")}
                rm[tpn]["n"] = len(snaps)
        ex["role_map"][g] = rm
        # E2 destruction: components with mem < -0.1 * peak |mem_total|
        dest, canc = [], defaultdict(list)
        for r in rs:
            tp = timepoints(r["traj"])
            peak = max(abs(x["paths"]["mem_total"]) for x in r["traj"])
            for tpn in ("t_mem", "t_grok", "t_end"):
                if tp[tpn] is not None:
                    pa = r["traj"][tp[tpn]]["paths"]
                    canc[tpn].append(sum(abs(m) for m in pa["mem"]) / max(abs(pa["mem_total"]), 1e-3 * peak, 1e-9))
            mem_end = endavg_vec(r, "mem")
            dest.append([j for j, m in enumerate(mem_end) if m < -0.1 * peak])
        ex["destruction"][g] = {"frac_runs_with_destroyer_at_end": float(np.mean([len(d) > 0 for d in dest])),
                                "destroyer_components_at_end": dest,
                                "cancellation_ratio_median": {k: float(np.median(v)) for k, v in canc.items()}}
        # E3 effective depth
        if long_b:
            cds = [endavg_vec(r, "comp_dep_state") for r in rs]
            ex["effective_depth"][g] = {"long_blocks": long_b,
                                        "comp_dep_state_mean": [round(float(np.mean([c[j - 1] for c in cds])), 3)
                                                                for j in long_b]}
        # E4 post-fit long-path learning
        if long_b:
            rows = []
            for r in rs:
                row = {}
                for tpn in ("t_fit", "t_mem", "t_grok", "t_end"):
                    if tpn == "t_end":
                        gen = endavg_vec(r, "gen")
                    else:
                        idx = timepoints(r["traj"])[tpn]
                        if idx is None:
                            continue
                        gen = r["traj"][idx]["paths"]["gen"]
                    gb = sum(gen[1:])
                    row[f"long_gen_share@{tpn}"] = sum(gen[j] for j in long_b) / gb if abs(gb) > 1e-6 else None
                    row[f"U_long@{tpn}"] = utility(r, long_b, tpn)
                rows.append(row)
            keys = sorted({k for row in rows for k in row})
            ex["post_fit_long"][g] = {k: float(np.nanmedian([row[k] if row.get(k) is not None else np.nan for row in rows]))
                                      for k in keys}
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
            gal[str(thr)] = {"short": float(np.median(sh)) if sh else None, "long": float(np.median(lo)) if lo else None,
                             "n": len(sh)}
        ex["grad_at_loss"][g] = gal
        # E6 Q-C descriptors
        qc = []
        for r in rs:
            tp = timepoints(r["traj"])
            s = r["summary"]
            qc.append({"delay": delay(r), "delay_norm": delay(r, True),
                       "peak_mem": r["traj"][tp["t_mem"]]["paths"]["mem_total"],
                       "persistence": (r["traj"][tp["t_grok"]]["step"] - r["traj"][tp["t_mem"]]["step"])
                       if tp["t_grok"] is not None else None,
                       "final_rule_frac": s["final_rule_frac"], "stable": s["stable"], "grokked": s["grokked"]})
        pers = [q["persistence"] for q in qc if q["persistence"] is not None]
        ex["qc"][g] = {"C": rs[0]["C"], "Q": rs[0]["Q"], "lr": rs[0]["lr"], "n": len(rs),
                       "median_delay": cmedian([q["delay"] for q in qc]),
                       "median_delay_lr_wd_normalized": cmedian([q["delay_norm"] for q in qc]),
                       "median_peak_mem": float(np.median([q["peak_mem"] for q in qc])),
                       "median_persistence": float(np.median(pers)) if pers else None,
                       "median_final_rule_frac": float(np.median([q["final_rule_frac"] for q in qc])),
                       "stable_frac": float(np.mean([q["stable"] for q in qc])),
                       "grok_frac": float(np.mean([q["grokked"] for q in qc]))}
        # E7 preservation: embedding path
        pres = [{"gen0": endavg(r, lambda x: x["paths"]["gen"][0]), "mem0": endavg(r, lambda x: x["paths"]["mem"][0]),
                 "rule_cancelled": endavg(r, lambda x: x["paths"]["own_rule"][0] - x["paths"]["shap_rule"][0]),
                 "mem_cancelled": endavg(r, lambda x: x["paths"]["own_mem"][0] - x["paths"]["shap_mem"][0])} for r in rs]
        ex["preservation"][g] = {k: float(np.median([p[k] for p in pres])) for k in pres[0]}
    return dict(ex)


def shared_lr(stage0: list[dict]) -> dict:
    """Exploratory: H2 contrasts and H3 at each shared lr of Stage 0 (3 seeds)."""
    out = {}
    for lr in sorted({r["lr"] for r in stage0}):
        rs = [r for r in stage0 if r["lr"] == lr]
        h3r = h3(rs, ALL, n_perm=2000)
        h2r = h2(rs)
        out[str(lr)] = {"H3_tau": None if h3r is None else h3r["tau"], "H3_p": None if h3r is None else h3r["p"],
                        "median_delay": None if h3r is None else h3r["median_delay"],
                        "H2a_mean": h2r["H2a"]["mean"], "H2b_mean": h2r["H2b"]["mean"],
                        "grok_frac": {g: float(np.mean([grokked(r) for r in rs if r["arm"] == g])) for g in ALL
                                      if any(r["arm"] == g for r in rs)}}
    return out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def compact(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        tp = timepoints(r["traj"])
        n = len(r["traj"])
        thin = [{"step": x["step"], "train_acc": x["train_acc"], "test_acc": x["test_acc"],
                 "gen": x["paths"]["gen"], "mem": x["paths"]["mem"], "gen_tshap": x["paths"]["gen_tshap"],
                 "mem_tshap": x["paths"]["mem_tshap"], "gen_total": x["paths"]["gen_total"],
                 "mem_total": x["paths"]["mem_total"],
                 "gen_without_short": None if x["paths"]["abl_short"] is None else x["paths"]["abl_short"]["gen"],
                 "gen_without_long": None if x["paths"]["abl_long"] is None else x["paths"]["abl_long"]["gen"]}
                for i, x in enumerate(r["traj"]) if x["step"] < 1000 or i % 5 == 0 or i == n - 1]
        out.append({"arm": r["arm"], "seed": r["seed"], "lr": r["lr"], "depth": r["depth"], "C": r["C"], "Q": r["Q"],
                    "summary": {k: v for k, v in r["summary"].items() if k not in ("final_branches", "final_ablate")},
                    "timepoints": {k: snapshot(r, v) for k, v in tp.items()}, "thin": thin})
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)) and (x >= 1e9 or np.isinf(x)):
        return "cens."
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def figures(runs: list[dict], out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    graphs = [g for g in ALL if any(r["arm"] == g for r in runs)]
    colors = ["#888888", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    ncol = (len(graphs) + 1) // 2
    for key, fname in (("gen", "roles_gen.png"), ("mem", "roles_mem.png")):
        fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 5.6), squeeze=False)
        for k, g in enumerate(graphs):
            ax = axes[k % 2][k // 2]
            rs = [r for r in runs if r["arm"] == g]
            n = min(len(r["traj"]) for r in rs)
            steps = [rs[0]["traj"][i]["step"] for i in range(n)]
            for j in range(5):
                m = np.mean([[r["traj"][i]["paths"][key][j] for i in range(n)] for r in rs], 0)
                lab = "emb" if j == 0 else f"b{j} (d{rs[0]['depth'][j - 1]})"
                ax.plot(steps, m, color=colors[j], lw=1.1, label=lab)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xscale("symlog", linthresh=1000)
            ax.set_title(g, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
        fig.suptitle(f"Study 4: {key} per component (direct attribution; mean over seeds; truncated at shortest run)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=130)
        plt.close(fig)


def verdict(ok: bool, holm_p: float | None) -> str:
    if not ok or holm_p is None or holm_p >= 0.05:
        return "NOT SUPPORTED"
    return "SUPPORTED" if holm_p <= 0.01 else "SUPPORTED, escalate to 29 seeds before claiming (0.01 < Holm p < 0.05)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--stage0", default=None)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.runs)
    sel = json.load(open(args.selection)) if args.selection else {}
    flagged = [g for g, v in sel.items() if v["tier"] != "stable"]
    res = {"H1": h1_both(runs, MIXED), "H2": h2(runs), "H3": h3(runs, ALL)}
    hp = holm({k: res[k]["p"] for k in ("H1", "H2", "H3")})
    res["holm"] = hp
    res["verdicts"] = {"H1": verdict(res["H1"]["ci_both_above_0"], hp["H1"]),
                       "H2": verdict(res["H2"]["ci_both_below_0"], hp["H2"]),
                       "H3": verdict(res["H3"]["tau"] > 0 and res["H3"]["lo"] > 0, hp["H3"])}
    clean = [g for g in ALL if g not in flagged]
    res["without_flagged"] = {
        "flagged": flagged, "H1": h1_both(runs, [g for g in MIXED if g in clean]),
        "H2": {k: {**h2(runs, {k: v})[k], "involves_flagged": v[0] in flagged or v[1] in flagged}
               for k, v in H2_CONTRASTS.items()},
        "H3": h3(runs, clean)}
    res["H3_lr_wd_normalized"] = h3(runs, ALL, normalized=True)
    ex = exploratory(runs)
    s0 = shared_lr(load(args.stage0)) if args.stage0 else None
    json.dump({"results": res, "exploratory": ex, "shared_lr": s0, "selection": sel, "runs": compact(runs)},
              open(os.path.join(args.out, "compact.json"), "w"), default=float)
    figures(runs, args.out)

    L = ["# Study 4 automatic report", "", f"{len(runs)} Stage-1 runs. Selection: " +
         ", ".join(f"{g} lr {v['lr']:g} ({v['tier']}{', grid edge' if v['grid_edge'] else ''})" for g, v in sel.items()), ""]
    L += ["## Verdicts (Holm over H1–H3)", ""] + [f"- **{k}:** {v} (Holm p = {fmt(hp[k], 4)})"
                                                for k, v in res["verdicts"].items()]
    for att in ("direct", "total"):
        h = res["H1"][att]
        L += ["", f"## H1 ({att} attribution): short paths' share of gen minus their share of mem (> 0)",
              f"pooled D = {fmt(h['pooled']['mean'])} [{fmt(h['pooled']['lo'])}, {fmt(h['pooled']['hi'])}], seeds = "
              f"{h['pooled']['n']}, one-sided p = {fmt(h['p'], 4)}; excluded {h['excluded']}; not grokked {h['not_grokked']}",
              "", "| graph | D | 95% CI | n |", "|---|---|---|---|"]
        L += [f"| {g} | {fmt(v['mean'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {v['n']} |" for g, v in h["per_graph"].items()]
    h = res["H2"]
    L += ["", "## H2: an extra short reader lowers the utility of existing long blocks (A − B < 0 in both)",
          f"intersection-union p = {fmt(h['p'], 4)}", "",
          "| contrast | blocks | U(A) | U(B) | mean A−B | 95% CI | pairs | not both grokked | undefined |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in H2_CONTRASTS:
        v = h[k]
        L.append(f"| {k}: {v['A']} vs {v['B']} | {v['blocks']} | {fmt(v['U_A'])} | {fmt(v['U_B'])} | {fmt(v['mean'])} | "
                 f"[{fmt(v['lo'])}, {fmt(v['hi'])}] | {v['n']} | {v['pairs_not_both_grokked']} | {v['pairs_undefined']} |")
    h = res["H3"]
    L += ["", "## H3: composition delays generalization (τ(C, delay) > 0)",
          f"τ_b = {fmt(h['tau'])} [seed bootstrap {fmt(h['lo'])}, {fmt(h['hi'])}], permutation p = {fmt(h['p'], 4)}, "
          f"n = {h['n']}, censored = {h['censored']}",
          "median delay: " + ", ".join(f"{g} {fmt(v, 0)}" for g, v in h["median_delay"].items())]
    hn = res["H3_lr_wd_normalized"]
    L += [f"lr·wd-normalized delay (exploratory): τ_b = {fmt(hn['tau'])}, p = {fmt(hn['p'], 4)}"]
    w = res["without_flagged"]
    L += ["", f"## Without flagged graphs {w['flagged']}",
          f"- H1 direct D = {fmt(w['H1']['direct']['pooled']['mean'])} [{fmt(w['H1']['direct']['pooled']['lo'])}, "
          f"{fmt(w['H1']['direct']['pooled']['hi'])}]; total D = {fmt(w['H1']['total']['pooled']['mean'])} "
          f"[{fmt(w['H1']['total']['pooled']['lo'])}, {fmt(w['H1']['total']['pooled']['hi'])}]"]
    for k, v in w["H2"].items():
        L.append(f"- {k}: mean A−B {fmt(v['mean'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]"
                 f"{' (involves a flagged graph)' if v['involves_flagged'] else ''}")
    L.append(f"- H3 τ = {fmt(w['H3']['tau']) if w['H3'] else '—'}, p = {fmt(w['H3']['p'], 4) if w['H3'] else '—'}")
    if s0:
        L += ["", "## Shared learning rate (Stage 0, 3 seeds, exploratory)", "",
              "| lr | H3 τ | H3 p | H2a A−B | H2b A−B |", "|---|---|---|---|---|"]
        L += [f"| {lr} | {fmt(v['H3_tau'])} | {fmt(v['H3_p'], 3)} | {fmt(v['H2a_mean'])} | {fmt(v['H2b_mean'])} |"
              for lr, v in s0.items()]
    L += ["", "## Exploratory (no claims)", "", "### Q-C descriptors", "",
          "| graph | C | Q | lr | grok | stable | median delay | lr·wd·delay | peak mem_total | persistence | final rule frac |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, q in ex["qc"].items():
        L.append(f"| {g} | {q['C']} | {q['Q']} | {q['lr']:g} | {fmt(q['grok_frac'], 2)} | {fmt(q['stable_frac'], 2)} | "
                 f"{fmt(q['median_delay'], 0)} | {fmt(q['median_delay_lr_wd_normalized'], 2)} | {fmt(q['median_peak_mem'], 2)} | "
                 f"{fmt(q['median_persistence'], 0)} | {fmt(q['median_final_rule_frac'])} |")
    L += ["", "### Role map (mean over seeds; direct: emb, b1..b4; total-effect: b1..b4)", ""]
    for g, rm in ex["role_map"].items():
        if "t_end" in rm and "t_mem" in rm:
            L.append(f"- {g}: gen@end {rm['t_end']['gen']} | tshap {rm['t_end']['gen_tshap']}; "
                     f"mem@t_mem {rm['t_mem']['mem']} | tshap {rm['t_mem']['mem_tshap']}")
    L += ["", "### Destruction", ""]
    for g, d in ex["destruction"].items():
        L.append(f"- {g}: runs with a destroyer at end {fmt(d['frac_runs_with_destroyer_at_end'], 2)}; cancellation ratio "
                 + ", ".join(f"{k} {v:.2f}" for k, v in d["cancellation_ratio_median"].items()))
    L += ["", "### Effective depth (comp_dep_state of long blocks, end)", ""]
    L += [f"- {g}: blocks {e['long_blocks']} → {e['comp_dep_state_mean']}" for g, e in ex["effective_depth"].items()]
    L += ["", "### Post-fit long-path learning (medians)", ""]
    L += [f"- {g}: " + ", ".join(f"{k} {fmt(v)}" for k, v in e.items()) for g, e in ex["post_fit_long"].items()]
    L += ["", "### Gradient allocation at matched train loss (median mean-block grad norm, short / long)", ""]
    L += [f"- {g}: " + "; ".join(f"loss≤{t}: {fmt(v['short'], 4)} / {fmt(v['long'], 4)}" for t, v in e.items())
          for g, e in ex["grad_at_loss"].items()]
    L += ["", "### Preservation (embedding path, end medians)", ""]
    L += [f"- {g}: " + ", ".join(f"{k} {fmt(v)}" for k, v in e.items()) for g, e in ex["preservation"].items()]
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
