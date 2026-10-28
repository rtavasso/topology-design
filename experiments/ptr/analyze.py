"""Analysis of rung-0 results (experiments/ptr/DESIGN.md §§7–8).

    python experiments/ptr/analyze.py results/ptr_main [results/ptr_ppool ...] [--md out.md]

Reads every results/*.json.gz under the given folders and prints, per arm (and per tail factor):
accuracy and steps by k with run-level mean and standard error; the headline gaps (H2: LHs vs LO by k; H4: step
slope/excess; H5: accuracy at k = 6..8 against example count); the residual-trigger analysis on traces (H7) and the
random-halting control; the step-index probe. Verdicts here are descriptive; the preregistration fixes thresholds.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np


def load(folders):
    recs = []
    for fo in folders:
        for p in sorted(glob.glob(os.path.join(fo, "results", "*.json.gz")) + glob.glob(os.path.join(fo, "*.json.gz"))):
            with gzip.open(p, "rt") as f:
                recs.append(json.load(f))
    return recs


def group_key(r):
    cfg = r["cfg"]
    tag = r.get("arm", "?")
    extras = []
    if cfg.get("tail_factor", 1.0) != 1.0:
        extras.append(f"f={cfg['tail_factor']:g}")
    if cfg.get("pair_pair_mask", True) is False:
        extras.append("doubling")
    if cfg.get("injection", True) is False:
        extras.append("no-inj")
    if cfg.get("d") not in (None, 264):
        extras.append(f"d={cfg['d']}")
    if r["name"].startswith("pilot"):
        extras.append(f"lr={cfg.get('lr')}")
        if "beta" in cfg:
            extras.append(f"b={cfg['beta']},l={cfg['lam_p']}")
    return tag + ("[" + ",".join(extras) + "]" if extras else "")


def final(r):
    return r["evals"][-1]["eval"]["in"]


def mean_se(x):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def table(recs, out):
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
        if "eval_T_train" in r:   # the fixed loop evaluated at its training depth, as a separate row
            twin = dict(r); twin["evals"] = [{"eval": r["eval_T_train"]}]
            groups[group_key(r).replace("LF", f"LF@T{r['model']['T']}", 1)].append(twin)
    ks = sorted({int(k) for r in recs for k in final(r)})
    out("## Accuracy by k (run mean ± s.e.; n runs)\n")
    out("| arm | n | " + " | ".join(f"k={k}" for k in ks) + " |")
    out("|---|---|" + "---|" * len(ks))
    for g, rs in sorted(groups.items()):
        cells = []
        for k in ks:
            m, se = mean_se([final(r).get(str(k), {}).get("acc") for r in rs])
            cells.append(f"{m:.2f}±{se:.2f}" if not np.isnan(m) else "–")
        out(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    out("\n## Query steps by k (run mean)\n")
    out("| arm | " + " | ".join(f"k={k}" for k in ks) + " |")
    out("|---|" + "---|" * len(ks))
    for g, rs in sorted(groups.items()):
        cells = []
        for k in ks:
            m, _ = mean_se([final(r).get(str(k), {}).get("q_steps") for r in rs])
            cells.append(f"{m:.1f}" if not np.isnan(m) else "–")
        out(f"| {g} | " + " | ".join(cells) + " |")
    base = [final(r).get(str(ks[0]), {}).get("baseline_acc") for r in recs]
    out(f"\nNo-hop baseline accuracy ≈ {np.nanmean([b for b in base if b is not None]):.3f}")
    return groups, ks


def calibration(groups, ks, out):
    out("\n## H4 calibration: query steps vs k (least-squares slope, mean excess over k)\n")
    for g, rs in sorted(groups.items()):
        if not any(r["model"]["halting"] == "learned" for r in rs):
            continue
        xs, ys = [], []
        for r in rs:
            for k in ks:
                v = final(r).get(str(k), {}).get("q_steps")
                if v is not None:
                    xs.append(k); ys.append(v)
        xs, ys = np.array(xs, float), np.array(ys, float)
        slope = np.polyfit(xs, ys, 1)[0]
        out(f"- {g}: slope {slope:.2f}, mean excess {np.mean(ys - xs):.2f} steps, "
            f"max steps {max(final(r).get(str(k), {}).get('max_steps', 0) for r in rs for k in ks):.0f}")


def headline(groups, ks, out):
    out("\n## H2: halting extrapolation gap (LHs − LO) by k\n")
    lo = groups.get("LO", []); lh = groups.get("LHs", []); lf = groups.get("LF", [])
    if lo and lh:
        for k in ks:
            a, sa = mean_se([final(r).get(str(k), {}).get("acc") for r in lh])
            b, sb = mean_se([final(r).get(str(k), {}).get("acc") for r in lo])
            c, _ = mean_se([final(r).get(str(k), {}).get("acc") for r in lf]) if lf else (float("nan"), 0)
            out(f"- k={k}: LHs {a:.2f}±{sa:.2f}  LO {b:.2f}±{sb:.2f}  gap {a-b:+.2f}" + (f"  LF(T=40) {c:.2f}" if lf else ""))


def ppool(recs, out):
    rows = defaultdict(dict)
    for r in recs:
        f = r["cfg"].get("tail_factor", 1.0)
        for k in (6, 7, 8):
            rows[(r.get("arm"), f)].setdefault(k, []).append(final(r).get(str(k), {}).get("acc"))
    if len({f for (_, f) in rows}) < 2:
        return
    out("\n## H5 P-pool: accuracy at k = 6/7/8 by tail factor (deep-example count ∝ factor)\n")
    out("| arm | factor | k=6 | k=7 | k=8 |")
    out("|---|---|---|---|---|")
    for (arm, f), d in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        cells = [f"{mean_se(d.get(k, []))[0]:.2f}" for k in (6, 7, 8)]
        out(f"| {arm} | {f:g} | " + " | ".join(cells) + " |")


def residual_triggers(groups, out):
    """LR: on fixed-arm traces, halt when the readout's step-KL drops below a threshold; accuracy and steps."""
    rs = [r for r in groups.get("LF", []) if r.get("traces")]
    if not rs:
        return
    out("\n## H7 residual-triggered halting on LF traces (step-KL threshold; accuracy at the halting step, mean steps)\n")
    for thr in (1e-1, 1e-2, 1e-3, 1e-4):
        accs, steps = [], []
        for r in rs:
            for k, tr in r["traces"].items():
                kl = np.array(tr["kl"]); corr = np.array(tr["correct"])     # (T, n)
                T, n = kl.shape
                for j in range(n):
                    t_halt = next((t for t in range(1, T) if kl[t, j] < thr), T - 1)
                    accs.append(corr[t_halt, j]); steps.append(t_halt + 1)
        out(f"- threshold {thr:g}: accuracy {np.mean(accs):.3f}, mean steps {np.mean(steps):.1f} (n={len(accs)})")
    out("\nOracle on the same traces (halt at exactly k): accuracy "
        + f"{np.mean([np.array(tr['correct'])[min(int(kk)-1, len(tr['correct'])-1), j] for r in rs for tr in r['traces'].values() for j, kk in enumerate(tr['k'])]):.3f}")


def random_halting(groups, ks, out):
    """Halting at a random step drawn from LHs's empirical step distribution, applied to LO/LF traces."""
    lh = groups.get("LHs", []); lf = [r for r in groups.get("LF", []) if r.get("traces")]
    if not lh or not lf:
        return
    out("\n## Random-halting control (LHs's step distribution, independent of the token, applied to LF traces)\n")
    rng = np.random.default_rng(0)
    for k in ks:
        steps = [final(r).get(str(k), {}).get("q_steps") for r in lh]
        steps = [s for s in steps if s is not None]
        if not steps:
            continue
        mean_steps = float(np.mean(steps))
        accs = []
        for r in lf:
            tr = r["traces"].get(str(k))
            if not tr:
                continue
            corr = np.array(tr["correct"])
            T, n = corr.shape
            t = np.clip(rng.poisson(max(mean_steps - 1, 0), size=n), 0, T - 1)
            accs.append(corr[t, np.arange(n)].mean())
        if accs:
            out(f"- k={k}: random halting at mean {mean_steps:.1f} steps: accuracy {np.mean(accs):.2f}")


def probes(groups, out):
    out("\n## Step-index probe (linear decodability of t from the normalized chasing state)\n")
    for g, rs in sorted(groups.items()):
        v = [r["evals"][-1]["eval"].get("step_probe_acc") for r in rs]
        v = [x for x in v if x is not None]
        if v:
            out(f"- {g}: {np.mean(v):.2f} (chance ≈ {1/32:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = load(args.folders)
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-0 results: {len(recs)} runs\n")
    groups, ks = table(recs, out)
    headline(groups, ks, out)
    calibration(groups, ks, out)
    ppool(recs, out)
    residual_triggers(groups, out)
    random_halting(groups, ks, out)
    probes(groups, out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
