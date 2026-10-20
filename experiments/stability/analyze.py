"""Score Study 2 (stability ceiling vs composition depth) against experiments/stability/PREREGISTRATION.md.

Usage: PYTHONPATH=src python experiments/stability/analyze.py --runs results/stability/runs --out results/stability/report
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

LRS = [0.0015, 0.003, 0.006, 0.012, 0.024, 0.048]
BELOW = math.log(LRS[0] / 2)  # code for "no stable lr in the grid"


def kendall_tau_b(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom > 0 else 0.0


def ranks(v) -> np.ndarray:
    v = np.asarray(v, float)
    order = v.argsort(kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for val in np.unique(v):  # average ties
        m = v == val
        r[m] = r[m].mean()
    return r


def failure_mode(s: dict) -> str:
    if s.get("diverged_at") is not None:
        return "diverged"
    if s["t_fit"] is None:
        return "never_fit"
    return "stable" if s["stable"] else "collapsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = defaultdict(list)  # (wd, arm, lr) -> list of summaries
    for f in glob.glob(os.path.join(args.runs, "*.json")):
        r = json.load(open(f))
        if "summary" not in r:
            continue
        c = r["config"]
        cells[(c["weight_decay"], c["arm"], c["lr"])].append(r["summary"])

    def lr_star(wd, arm):
        best = None
        for lr in LRS:
            ss = cells.get((wd, arm, lr), [])
            if len(ss) >= 2 and sum(s["stable"] for s in ss) >= 2:
                best = lr
        return math.log(best) if best else BELOW, best

    out = {}
    md = ["# Study 2 — stability ceiling vs composition depth", ""]
    for wd in sorted({k[0] for k in cells}, reverse=True):
        arms = sorted({k[1] for k in cells if k[0] == wd})
        rows = []
        for arm in arms:
            parents, _ = gk.arm_spec(arm)
            C, Q = gk.longest_chain(parents), gk.root_readers(parents)
            loglr, best = lr_star(wd, arm)
            frac = [np.mean([s["stable"] for s in cells.get((wd, arm, lr), [])]) if cells.get((wd, arm, lr)) else np.nan for lr in LRS]
            modes = {lr: [failure_mode(s) for s in cells.get((wd, arm, lr), [])] for lr in LRS}
            best_delay = min([s["grok_delay"] for lr in LRS for s in cells.get((wd, arm, lr), [])
                              if s["stable"] and s["grok_delay"] is not None], default=None)
            rows.append(dict(arm=arm, C=C, Q=Q, loglr=loglr, lr_star=best, frac=frac, modes=modes, best_delay=best_delay))
        C = np.array([r["C"] for r in rows]); Q = np.array([r["Q"] for r in rows]); Y = np.array([r["loglr"] for r in rows])
        res = {"n_graphs": len(rows)}
        md += [f"## Weight decay {wd:g}", "", "| graph | C | Q | lr* | fraction stable by lr (" + ", ".join(f"{x:g}" for x in LRS) + ") | best stable grok delay |",
               "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["C"], r["Q"], r["arm"])):
            lrs = f"{r['lr_star']:g}" if r["lr_star"] else "< grid"
            if r["lr_star"] == LRS[-1]:
                lrs += " (≥, censored)"
            md.append(f"| {r['arm']} | {r['C']} | {r['Q']} | {lrs} | " + " ".join("–" if np.isnan(f) else f"{f:.2f}" for f in r["frac"]) +
                      f" | {r['best_delay'] if r['best_delay'] is not None else '–'} |")
        md.append("")
        if len(rows) >= 4:
            tau = kendall_tau_b(C, Y)
            rng = np.random.default_rng(0)
            perm = np.array([kendall_tau_b(C, rng.permutation(Y)) for _ in range(10000 if len(rows) > 8 else 2000)])
            p1 = float((perm <= tau).mean())
            res.update(tau_C=tau, p_one_sided=p1)
            tag = "" if wd == 1.0 else "(informational only; preregistered tests use wd 1.0) "
            md.append(f"{tag}**P1** Kendall τ_b(C, log lr*) = {tau:+.3f}, one-sided permutation p = {p1:.4f} "
                      f"→ {'SUPPORTED' if (tau < 0 and p1 < 0.05) else 'NOT supported'}")
            ends = {r["arm"]: r for r in rows}
            if "pk0123" in ends and "pk0000" in ends:
                a, b = ends["pk0123"]["loglr"], ends["pk0000"]["loglr"]
                ok = a <= b - math.log(4) + 1e-9
                md.append(f"{tag}**P2** lr*(pk0123) = {math.exp(a):.2g}{' (below grid)' if a == BELOW else ''}, lr*(pk0000) = {math.exp(b):.2g} "
                          f"→ {'SUPPORTED' if ok else 'NOT supported'} (needs ≥ 4× gap)")
                res.update(P2=bool(ok))
            if len(rows) > 8:
                within = {}
                for c in (2, 3):
                    m = C == c
                    within[c] = kendall_tau_b(Q[m], Y[m]) if m.sum() > 2 else float("nan")
                X = np.column_stack([np.ones(len(rows)), ranks(C), ranks(Q)])
                coef = np.linalg.lstsq(X, ranks(Y), rcond=None)[0]
                boots = []
                for _ in range(5000):
                    idx = rng.integers(0, len(rows), len(rows))
                    Xb, yb = X[idx], ranks(Y)[idx]
                    if np.linalg.matrix_rank(Xb) == 3:
                        boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0][1])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                p3 = within[2] <= 0.2 and within[3] <= 0.2 and hi < 0
                q_wins = within[2] > 0.2 and within[3] > 0.2 and lo <= 0 <= hi
                res.update(within_tau_Q=within, coef_C=float(coef[1]), coef_C_ci=[float(lo), float(hi)], P3=bool(p3), Q_carries=bool(q_wins))
                md.append(f"**P3** within-C τ(Q, log lr*): C=2 {within[2]:+.2f}, C=3 {within[3]:+.2f}; rank-regression coefficient of C = "
                          f"{coef[1]:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] → {'SUPPORTED' if p3 else ('Q CARRIES THE EFFECT' if q_wins else 'NOT supported / inconclusive')}")
            md.append("")
        # failure modes by C and lr
        md += ["Failure modes (counts over seeds and graphs) by C × lr:", "", "| C | lr | stable | collapsed | never_fit | diverged |", "|---|---|---|---|---|---|"]
        for c in sorted(set(C)):
            for lr in LRS:
                ms = [m for r in rows if r["C"] == c for m in r["modes"][lr]]
                if ms:
                    md.append(f"| {c} | {lr:g} | " + " | ".join(str(ms.count(k)) for k in ("stable", "collapsed", "never_fit", "diverged")) + " |")
        md.append("")
        out[wd] = (rows, res)
    if 1.0 in out and 0.1 in out:
        sub = {r["arm"]: r["loglr"] for r in out[0.1][0]}
        full = {r["arm"]: r["loglr"] for r in out[1.0][0] if r["arm"] in sub}
        if len(sub) >= 3:
            arms = sorted(sub)
            rho = np.corrcoef(ranks([full[a] for a in arms]), ranks([sub[a] for a in arms]))[0, 1]
            md.append(f"**P4** Spearman ρ of lr* across the robustness subset, wd 1.0 vs 0.1 = {rho:+.2f} → "
                      f"{'SUPPORTED' if rho >= 0.8 else 'NOT supported'}")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({str(k): v[1] for k, v in out.items()}, f, indent=1, default=float)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = out[1.0][0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        rng = np.random.default_rng(1)
        for r in rows:
            axes[0].scatter(r["C"] + rng.uniform(-0.15, 0.15), math.exp(r["loglr"]), c=[plt.cm.viridis(r["Q"] / 4)], s=40)
        axes[0].set_yscale("log"); axes[0].set_xlabel("longest chain C"); axes[0].set_ylabel("lr* (largest stable lr)")
        axes[0].set_xticks([1, 2, 3, 4]); axes[0].set_title("colour = root readers Q (dark=1, light=4)")
        order = sorted(rows, key=lambda r: (r["C"], r["Q"]))
        M = np.array([r["frac"] for r in order])
        axes[1].imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        axes[1].set_yticks(range(len(order))); axes[1].set_yticklabels([f"{r['arm']} C{r['C']}" for r in order], fontsize=6)
        axes[1].set_xticks(range(len(LRS))); axes[1].set_xticklabels([f"{x:g}" for x in LRS], rotation=45)
        axes[1].set_title("fraction of seeds stable (wd 1.0)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "stability.png"), dpi=110)
    except Exception:
        pass
    print("\n".join(md))


if __name__ == "__main__":
    main()
