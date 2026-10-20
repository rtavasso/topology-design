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
        table[key] = dict(seeds=seeds, delay=d, censored=cens, loc=loc)
        lm, ll, lh = mean_ci(loc)
        rows.append(
            f"| {key[0]} | {key[1]:g} | {key[2]:g} | {len(seeds)} | {np.nanmedian(tfit):.0f} | {np.median(d):.0f} | "
            f"{int(cens.sum())} | {lm:+.3f} [{ll:+.3f}, {lh:+.3f}] | {np.mean(peak_mem):.2f} | "
            f"{np.mean(final_test):.3f} | {np.mean(abl_test):.3f} | {np.mean(sink_rule):.2f} |"
        )

    md = ["# Grokking × topology — results summary", "",
          f"{len(runs)} runs. Endpoints as preregistered; delay = t_grok − t_fit (censored runs counted at budget).", "",
          "| arm | wd | lr | n | median t_fit | median delay | # censored | localization index [95% CI] | "
          "peak memorization (excl. train acc) | final test acc | test acc, block 3 ablated | final rule share in block 3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, ""]

    # paired contrasts on delay among arms sharing wd/lr settings and seeds
    md += ["## Paired contrasts on grokking delay", "", "P(row arm groks faster than column arm), paired over shared seeds [95% bootstrap CI].", ""]
    by_setting = defaultdict(dict)
    for key, v in table.items():
        by_setting[(key[1],) + key[3:]][key[0]] = (key, v)
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
                contrasts[f"{A} vs {B} (wd={setting[0]:g})"] = (pm, pl, ph, len(shared))
                md.append(f"- wd={setting[0]:g}: P({A} faster than {B}) = {pm:.2f} [{pl:.2f}, {ph:.2f}] (n={len(shared)})")
    md.append("")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"contrasts": contrasts}, f, indent=1)
    plot(groups, args.out)
    print("\n".join(md))


def plot(groups, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # trajectories per arm (median over seeds), one figure per weight decay
    by_wd = defaultdict(list)
    for key, g in groups.items():
        by_wd[key[1]].append((key, g))
    for wd, items in by_wd.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for key, g in sorted(items, key=lambda kv: kv[0][0]):
            seeds = sorted(g)
            steps = [t["step"] for t in g[seeds[0]]["trajectory"]]
            for ax, field in zip(axes.flat, ["train_acc", "test_acc", "excluded_train_acc", "rule_frac"]):
                n = min(len(g[s]["trajectory"]) for s in seeds)
                vals = np.array([[t[field] for t in g[s]["trajectory"][:n]] for s in seeds])
                ax.plot(steps[:n], np.median(vals, 0), label=f"{key[0]} (lr {key[2]:g})")
                ax.set_title(field)
        axes[0, 0].legend(fontsize=7)
        for ax in axes[1]:
            ax.set_xlabel("step")
        fig.suptitle(f"Median trajectories, weight decay {wd:g}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trajectories_wd{wd:g}.png"), dpi=110)
        plt.close(fig)
    # sink dose-response
    sink = sorted([(float(k[0].split("_")[1]), k, g) for k, g in groups.items() if k[0].startswith("sink_")], key=lambda x: (x[1][1], x[0]))
    if sink:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for wd in sorted({k[1] for _, k, _ in sink}):
            pts = [(s, g) for s, k, g in sink if k[1] == wd]
            xs = [s for s, _ in pts]
            dmed = [np.median([censored_delay(g[sd])[0] for sd in g]) for _, g in pts]
            lmean = [np.mean([localization_index(g[sd]) for sd in g]) for _, g in pts]
            axes[0].plot(xs, dmed, "o-", label=f"wd {wd:g}")
            axes[1].plot(xs, lmean, "o-", label=f"wd {wd:g}")
        axes[0].set_xscale("symlog", linthresh=0.25)
        axes[1].set_xscale("symlog", linthresh=0.25)
        axes[0].set_xlabel("sink multiplier s")
        axes[1].set_xlabel("sink multiplier s")
        axes[0].set_ylabel("median grokking delay (steps)")
        axes[1].set_ylabel("localization index (mean)")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "sink_dose_response.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
