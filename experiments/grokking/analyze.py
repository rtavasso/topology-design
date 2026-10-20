"""Score a results directory against the preregistered endpoints (PREREGISTRATION.md).

Usage:
    PYTHONPATH=src python experiments/grokking/analyze.py --runs results/main --out results/main/report
Writes summary.md, summary.json and figures into --out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

SINK = 3  # branch index of block 3 in the per-branch lists (0 = embedding path)


def load(run_dir: str) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if f.endswith("summary.json"):
            continue
        with open(f) as fh:
            r = json.load(fh)
        if "trajectory" in r:
            runs.append(r)
    return runs


def group_key(cfg: dict) -> tuple:
    return (cfg["arm"], cfg["weight_decay"], cfg["lr"], cfg["out_scale"], cfg["beta"], cfg["train_frac"], cfg["p"])


def censored_delay(r: dict) -> tuple[float, bool]:
    """Delay with right-censoring: runs that never grok get (budget - t_fit + 1), flagged censored."""
    s = r["summary"]
    budget = r["config"]["steps"]
    t_fit = s["t_fit"] if s["t_fit"] is not None else budget
    if s["t_grok"] is not None:
        return float(s["t_grok"] - t_fit), False
    return float(budget - t_fit + 1), True


def localization_index(r: dict) -> float:
    traj = r["trajectory"]
    k = int(np.argmax([t["excluded_train_acc"] for t in traj]))
    br = traj[k]["branches"]
    rem = np.array([b["rem_energy"] for b in br])
    rule = np.array([b["rule_energy"] for b in br])
    return float(rem[SINK] / max(rem.sum(), 1e-12) - rule[SINK] / max(rule.sum(), 1e-12))


def max_branch_dominance(r: dict) -> float:
    """Stability diagnostic (amendment A1): max over training of the largest block's share of direct logit energy."""
    best = 0.0
    for t in r["trajectory"][1:]:
        e = np.array([b["rem_energy"] + b["rule_energy"] for b in t["branches"][1:]])
        if e.sum() > 0:
            best = max(best, float(e.max() / e.sum()))
    return best


def final_rule_share_in_sink(r: dict) -> float:
    br = r["trajectory"][-1]["branches"]
    rule = np.array([b["rule_energy"] for b in br])
    return float(rule[SINK] / max(rule.sum(), 1e-12))


def p_less(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Paired P(A < B) (ties = 1/2) with a percentile bootstrap CI over seeds."""
    ind = (a < b).astype(float) + 0.5 * (a == b)
    rng = np.random.default_rng(seed)
    boots = [ind[rng.integers(0, len(ind), len(ind))].mean() for _ in range(n_boot)]
    return float(ind.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mean_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.runs)
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in runs:
        c = r["config"]
        groups[group_key(c)][c["model_seed"] * 1000 + c["data_seed"]] = r

    rows, table = [], {}
    for key in sorted(groups, key=lambda k: (k[1], k[0])):
        g = groups[key]
        seeds = sorted(g)
        d = np.array([censored_delay(g[s])[0] for s in seeds])
        cens = np.array([censored_delay(g[s])[1] for s in seeds])
        loc = np.array([localization_index(g[s]) for s in seeds])
        tfit = np.array([g[s]["summary"]["t_fit"] if g[s]["summary"]["t_fit"] is not None else np.nan for s in seeds])
        abl_test = np.array([g[s]["summary"]["final_ablate"][SINK - 1]["test_acc"] for s in seeds])
        final_test = np.array([g[s]["summary"]["final_test_acc"] for s in seeds])
        sink_rule = np.array([final_rule_share_in_sink(g[s]) for s in seeds])
        peak_mem = np.array([g[s]["summary"]["peak_excluded_train_acc"] for s in seeds])
        dom = np.array([max_branch_dominance(g[s]) for s in seeds])
        table[key] = dict(seeds=seeds, delay=d, censored=cens, loc=loc)
        lm, ll, lh = mean_ci(loc)
        rows.append(
            f"| {key[0]} | {key[1]:g} | {key[2]:g} | {len(seeds)} | {np.nanmedian(tfit):.0f} | {np.median(d):.0f} | "
            f"{int(cens.sum())} | {lm:+.3f} [{ll:+.3f}, {lh:+.3f}] | {np.mean(peak_mem):.2f} | "
            f"{np.mean(final_test):.3f} | {np.mean(abl_test):.3f} | {np.mean(sink_rule):.2f} | "
            f"{np.median(dom):.2f}{' ⚠' if np.median(dom) > 0.9 else ''} |"
        )

    md = ["# Grokking × topology — results summary", "",
          f"{len(runs)} runs. Endpoints as preregistered; delay = t_grok − t_fit (censored runs counted at budget).", "",
          "| arm | wd | lr | n | median t_fit | median delay | # censored | localization index [95% CI] | "
          "peak memorization (excl. train acc) | final test acc | test acc, block 3 ablated | final rule share in block 3 | "
          "median max branch dominance (⚠ >0.9) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, ""]

    # paired contrasts on delay among arms sharing wd/lr settings and seeds
    md += ["## Paired contrasts on grokking delay", "", "P(row arm groks faster than column arm), paired over shared seeds [95% bootstrap CI].", ""]
    by_setting = defaultdict(dict)
    for key, v in table.items():
        by_setting[(key[1], key[2]) + key[3:]][key[0]] = (key, v)  # hold wd AND lr fixed within a contrast
    contrasts = {}
    for setting, arms in by_setting.items():
        names = sorted(arms)
        for i, A in enumerate(names):
            for B in names[i + 1 :]:
                ka, va = arms[A]
                kb, vb = arms[B]
                shared = sorted(set(va["seeds"]) & set(vb["seeds"]))
                if len(shared) < 2:
                    continue
                da = np.array([va["delay"][va["seeds"].index(s)] for s in shared])
                db = np.array([vb["delay"][vb["seeds"].index(s)] for s in shared])
                pm, pl, ph = p_less(da, db)
                contrasts[f"{A} vs {B} (wd={setting[0]:g}, lr={setting[1]:g})"] = (pm, pl, ph, len(shared))
                md.append(f"- wd={setting[0]:g}, lr={setting[1]:g}: P({A} faster than {B}) = {pm:.2f} [{pl:.2f}, {ph:.2f}] (n={len(shared)})")
    md.append("")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"contrasts": contrasts}, f, indent=1)
    plot(groups, args.out)
    print("\n".join(md))


def _ffill(traj, field, n):
    v = [t[field] for t in traj]
    return v + [v[-1]] * (n - len(v))  # runs stop early after grokking: carry the last value forward


def plot(groups, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cmap = plt.get_cmap("tab10")
    # 1) median trajectories, one figure per (weight decay, lr)
    by_set = defaultdict(list)
    for key, g in groups.items():
        by_set[(key[1], key[2])].append((key, g))
    for (wd, lr), items in sorted(by_set.items()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ci, (key, g) in enumerate(sorted(items, key=lambda kv: kv[0][0])):
            seeds = sorted(g)
            n = max(len(g[s]["trajectory"]) for s in seeds)
            ev = g[seeds[0]]["config"]["eval_every"]
            steps = np.arange(n) * ev
            for ax, field in zip(axes.flat, ["test_acc", "excluded_train_acc", "rule_frac", "embed_top5_freq_frac"]):
                vals = np.array([_ffill(g[s]["trajectory"], field, n) for s in seeds])
                ax.plot(steps, np.median(vals, 0), color=cmap(ci % 10), ls="-" if ci < 10 else "--", label=key[0])
                ax.set_title(field)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("step")
        fig.suptitle(f"Median over seeds (stopped runs carried forward), weight decay {wd:g}, lr {lr:g}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trajectories_wd{wd:g}_lr{lr:g}.png"), dpi=110)
        plt.close(fig)
    # 2) sink dose-response per (wd, lr), with par / seq reference bands
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sets = sorted({(k[1], k[2]) for k in groups if k[0].startswith("sink_")})
    for ci, (wd, lr) in enumerate(sets):
        pts = sorted([(float(k[0].split("_")[1]), g) for k, g in groups.items()
                      if k[0].startswith("sink_") and k[1] == wd and k[2] == lr], key=lambda x: x[0])
        xs = np.array([s for s, _ in pts])
        xplot = np.where(xs == 0, 0.125, xs)  # put s=0 at the left edge of the log axis
        d = [np.array([censored_delay(g[sd])[0] for sd in g]) for _, g in pts]
        med = np.array([np.median(v) for v in d])
        lo = np.array([np.percentile(v, 25) for v in d])
        hi = np.array([np.percentile(v, 75) for v in d])
        col = cmap(ci)
        axes[0].plot(xplot, med, "o-", color=col, label=f"sinks, wd {wd:g}, lr {lr:g}")
        axes[0].fill_between(xplot, lo, hi, color=col, alpha=0.15)
        for ref, ls in (("par", ":"), ("seq", "--")):
            k = next((k for k in groups if k[0] == ref and k[1] == wd and k[2] == lr), None)
            if k:
                axes[0].axhline(np.median([censored_delay(groups[k][sd])[0] for sd in groups[k]]), color=col, ls=ls, lw=1,
                                label=f"{ref}, wd {wd:g}, lr {lr:g}")
        locs = [np.array([localization_index(g[sd]) for sd in g]) for _, g in pts]
        m = np.array([v.mean() for v in locs])
        ci_ = np.array([mean_ci(v)[1:] for v in locs])
        axes[1].errorbar(xplot, m, yerr=[m - ci_[:, 0], ci_[:, 1] - m], fmt="o-", color=col, capsize=3,
                         label=f"wd {wd:g}, lr {lr:g}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0", "0.25", "0.5", "1", "2", "4"])
        ax.set_xlabel("sink multiplier s")
    axes[0].set_ylabel("grokking delay, median & IQR (censored at budget)")
    axes[1].set_ylabel("localization index (mean, 95% CI)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sink_dose_response.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
