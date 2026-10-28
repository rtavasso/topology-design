"""Analysis of rung-0 results (experiments/ptr/DESIGN.md §§7–8, revision 3).

    python experiments/ptr/analyze.py results/ptr_rev3 [more folders ...] [--md out.md]

Reads every results/*.json.gz under the given folders and prints, per arm (and per tail factor):
accuracy and steps by k with run-level mean and standard error (LF at its training depth and LR, the stability exit
on LF's readouts, as their own rows); the extrapolation gaps of every halting arm against LO and LR; step calibration;
steps to criterion by k (time to learn each depth) and the pooling test as steps to criterion at k = k_deep against
the tail factor; the halting probability by loop; the residual-trigger analysis on traces and the random-halting
control; the step-index probe. Verdicts here are descriptive; the preregistration (DESIGN.md §8) fixes thresholds.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

CRIT = 0.95          # steps-to-criterion threshold on the 100-step training window (confirmed by the next window)
LR_PRIMARY = "0.01"  # the preregistered stability-exit threshold (Entry 56)


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


def derived_rows(r):
    """Extra rows from one record: the fixed loop at its training depth (LF@T) and the stability exit (LR)."""
    rows = []
    if "eval_T_train" in r:
        twin = dict(r); twin["evals"] = [{"eval": r["eval_T_train"]}]
        rows.append((group_key(r).replace("LF", f"LF@T{r['model']['T']}", 1), twin))
    fin = final(r)
    if any("kl_exit" in v for v in fin.values() if isinstance(v, dict)):
        thrs = sorted({t for v in fin.values() if isinstance(v, dict) for t in v.get("kl_exit", {})}, key=float)
        for thr in thrs:
            ins = {}
            for k, v in fin.items():
                if isinstance(v, dict) and thr in v.get("kl_exit", {}):
                    ins[k] = {**{kk: vv for kk, vv in v.items() if kk != "kl_exit"}, "acc": v["kl_exit"][thr]["acc"],
                              "q_steps": v["kl_exit"][thr]["q_steps"]}
            twin = dict(r); twin["evals"] = [{"eval": {"in": ins}}]
            twin["model"] = {**r["model"], "halting": "kl_exit"}
            name = "LR" if thr == LR_PRIMARY else f"LR[thr={thr}]"
            rows.append((group_key(r).replace("LF", name, 1), twin))
    return rows


def build_groups(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
        for g, twin in derived_rows(r):
            groups[g].append(twin)
    return groups


def table(groups, recs, out):
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
    return ks


def is_halting_group(g, rs):
    return g.startswith("LH") or g.startswith("LR") or any(r["model"].get("halting") in ("learned", "kl_exit") for r in rs)


def calibration(groups, ks, out):
    out("\n## Calibration: query steps vs k (least-squares slope, mean excess over k; k ≤ 32)\n")
    for g, rs in sorted(groups.items()):
        if not is_halting_group(g, rs):
            continue
        xs, ys = [], []
        for r in rs:
            for k in ks:
                v = final(r).get(str(k), {}).get("q_steps")
                if v is not None:
                    xs.append(k); ys.append(v)
        if len(xs) < 2:
            continue
        xs, ys = np.array(xs, float), np.array(ys, float)
        slope = np.polyfit(xs, ys, 1)[0]
        out(f"- {g}: slope {slope:.2f}, mean excess {np.mean(ys - xs):.2f} steps, "
            f"max steps {max(final(r).get(str(k), {}).get('max_steps', 0) for r in rs for k in ks):.0f}")


def headline(groups, ks, out):
    """Extrapolation gaps: each halting arm against LO (block extrapolation alone) and LR (the free stability exit)."""
    refs = {n: groups.get(n, []) for n in ("LO", "LR", "LF")}
    show = [k for k in (8, 12, 16, 24, 32) if k in ks]
    out("\n## Extrapolation: accuracy by k, and the gaps against LO and LR (run means)\n")
    out("| arm | " + " | ".join(f"k={k}" for k in show) + " | gap vs LO @16/32 | gap vs LR @16/32 |")
    out("|---|" + "---|" * len(show) + "---|---|")
    for g, rs in sorted(groups.items()):
        if not (is_halting_group(g, rs) or g in ("LO", "LF")):
            continue
        acc = {k: mean_se([final(r).get(str(k), {}).get("acc") for r in rs])[0] for k in show}
        gaps = []
        for ref in ("LO", "LR"):
            rr = refs[ref]
            if rr and g != ref:
                ga = [acc[k] - mean_se([final(r).get(str(k), {}).get("acc") for r in rr])[0] for k in (16, 32) if k in acc]
                gaps.append(" / ".join(f"{v:+.2f}" for v in ga))
            else:
                gaps.append("–")
        out(f"| {g} | " + " | ".join(f"{acc[k]:.2f}" for k in show) + f" | {gaps[0]} | {gaps[1]} |")


def t95_of(rec, k, key):
    """First logged step at which the training-window accuracy at k is ≥ CRIT in that window and the next (or the
    last window alone); None if never (censored at the budget)."""
    cur = rec["curve"]
    vals = [(c["step"], c.get(key, {}).get(str(k), c.get(key, {}).get(k))) for c in cur]
    for i, (s, a) in enumerate(vals):
        if a is None or a < CRIT:
            continue
        nxt = vals[i + 1][1] if i + 1 < len(vals) else None
        if nxt is None or nxt >= CRIT:
            return s
    return None


def own_key(r):
    """The arm's own prediction: the halted readout for learned halting, the final readout otherwise."""
    return "acc_halt_by_k" if any("acc_halt_by_k" in c for c in r["curve"]) else "acc_by_k"


def fmt_t95(vals, budget):
    v = [x for x in vals if x is not None]
    cens = len(vals) - len(v)
    if not v:
        return f"> {budget}"
    s = f"{int(np.median(v))}" + (f" [{min(v)}–{max(v)}]" if len(v) > 1 else "")
    return s + (f" ({cens}/{len(vals)} > {budget})" if cens else "")


def firsts(groups, out):
    out(f"\n## Steps to criterion by k (first 100-step training window with accuracy ≥ {CRIT}, confirmed by the next; "
        "median [min–max] over runs; the arm's own prediction, i.e. the halted readout for learned halting)\n")
    kmax = max(r["data"]["kmax"] for rs in groups.values() for r in rs)
    kk = list(range(1, kmax + 1))
    out("| arm | n | " + " | ".join(f"k={k}" for k in kk) + " |")
    out("|---|---|" + "---|" * len(kk))
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if r.get("curve")]
        if not rs or g.startswith("LF@") or g.startswith("LR"):
            continue
        budget = rs[0]["train"]["steps"]
        out(f"| {g} | {len(rs)} | " + " | ".join(fmt_t95([t95_of(r, k, own_key(r)) for r in rs], budget) for k in kk) + " |")
        if own_key(rs[0]) == "acc_halt_by_k":    # the block's T-loop readout as well, for the learned-halting arms
            out(f"| {g} (T-loop readout) | {len(rs)} | "
                + " | ".join(fmt_t95([t95_of(r, k, "acc_by_k") for r in rs], budget) for k in kk) + " |")


def ppool_time(groups, out):
    """Pooling as time: steps to criterion at k = k_deep by tail factor, per arm, with the ratio to factor 1."""
    by_arm = defaultdict(dict)
    for g, rs in groups.items():
        rs = [r for r in rs if r.get("curve")]
        if not rs or g.startswith("LF@") or g.startswith("LR"):
            continue
        f = rs[0]["cfg"].get("tail_factor", 1.0)
        by_arm[g.split("[")[0]][f] = rs
    multi = {a: d for a, d in by_arm.items() if len(d) > 1}
    if not multi:
        return
    out("\n## Pooling by time: steps to criterion at k = k_deep (and k_deep + 1) against the tail factor "
        "(median over runs; ratio to factor 1)\n")
    out("| arm | factor | k_deep | n | t95(k_deep) | ratio | t95(k_deep+1) | share of k ≥ k_deep |")
    out("|---|---|---|---|---|---|---|---|")
    for a, d in sorted(multi.items()):
        base = None
        for f in sorted(d, reverse=True):
            rs = d[f]
            kd = rs[0]["data"].get("k_deep", 6)
            budget = rs[0]["train"]["steps"]
            key = own_key(rs[0])
            t = [t95_of(r, kd, key) for r in rs]
            t1 = [t95_of(r, kd + 1, key) for r in rs]
            med = np.median([x for x in t if x is not None]) if any(x is not None for x in t) else None
            if f == 1.0:
                base = med
            ratio = f"{med / base:.1f}×" if (med is not None and base) else ("≥ " + f"{budget / base:.1f}×" if (med is None and base) else "–")
            share = np.mean([sum(c["n_by_k"].get(str(k), 0) for k in range(kd, 64)) / max(sum(c["n_by_k"].values()), 1)
                             for r in rs for c in r["curve"] if "n_by_k" in c]) if any("n_by_k" in c for c in rs[0]["curve"]) else float("nan")
            out(f"| {a} | {f:g} | {kd} | {len(rs)} | {fmt_t95(t, budget)} | {ratio} | {fmt_t95(t1, budget)} | {share:.4f} |")


def ppool_acc(recs, out):
    rows = defaultdict(dict)
    kd = min(r["data"].get("k_deep", 6) for r in recs)
    for r in recs:
        f = r["cfg"].get("tail_factor", 1.0)
        for k in (kd, kd + 1, kd + 2):
            rows[(r.get("arm"), f)].setdefault(k, []).append(final(r).get(str(k), {}).get("acc"))
    if len({f for (_, f) in rows}) < 2:
        return
    out(f"\n## Pooling by accuracy at the budget: k = {kd}/{kd+1}/{kd+2} by tail factor\n")
    out("| arm | factor | " + " | ".join(f"k={k}" for k in (kd, kd + 1, kd + 2)) + " |")
    out("|---|---|---|---|---|")
    for (arm, f), d in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        cells = [f"{mean_se(d.get(k, []))[0]:.2f}" for k in (kd, kd + 1, kd + 2)]
        out(f"| {arm} | {f:g} | " + " | ".join(cells) + " |")


def lam_by_loop(groups, out):
    out("\n## Halting probability λ by loop (run mean) at k = 8, 16, 32; the training horizon T in brackets\n")
    for g, rs in sorted(groups.items()):
        if not any(r["model"].get("halting") == "learned" for r in rs):
            continue
        for k in (8, 16, 32):
            lam = [final(r).get(str(k), {}).get("lam_by_step") for r in rs]
            lam = [l for l in lam if l]
            if not lam:
                continue
            L = min(len(l) for l in lam)
            m = np.mean([l[:L] for l in lam], 0)
            out(f"- {g} [T={rs[0]['model']['T']}] k={k}: " + " ".join(f"{v:.2f}" for v in m[:min(L, 36)]))


def residual_triggers(groups, out):
    """On fixed-arm traces: halt when the readout's step-KL drops below a threshold (the same rule as LR, here on
    the trace subset, for continuity with Entry 56)."""
    rs = [r for r in groups.get("LF", []) if r.get("traces")]
    if not rs:
        return
    out("\n## Residual-triggered halting on LF traces (step-KL threshold; accuracy at the halting step, mean steps)\n")
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
    """Halting at a random step drawn from a learned arm's empirical step distribution, applied to LF traces."""
    lf = [r for r in groups.get("LF", []) if r.get("traces")]
    if not lf:
        return
    for arm in ("LHs", "LHd", "LHs24"):
        lh = groups.get(arm, [])
        if not lh:
            continue
        out(f"\n## Random-halting control ({arm}'s step distribution, independent of the token, applied to LF traces)\n")
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
    groups = build_groups(recs)
    ks = table(groups, recs, out)
    headline(groups, ks, out)
    calibration(groups, ks, out)
    firsts(groups, out)
    ppool_time(groups, out)
    ppool_acc(recs, out)
    lam_by_loop(groups, out)
    residual_triggers(groups, out)
    random_halting(groups, ks, out)
    probes(groups, out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
