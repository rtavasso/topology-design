"""Analysis of rung-2 results (experiments/routefind/DESIGN.md §§4, 7).

    PYTHONPATH=src python experiments/routefind/analyze.py results/kaggle_routefind-main [--ref experiments/route/results_data/main] [--md out.md]

Arms are estimator × budget × deferral (e.g. IW16d1). --ref adds rung 1's exact-marginal runs (RM -> EXd1, RMd0 -> EXd0)
as a reference distribution for the SET-flip and criterion times (same code path, rung-1 commit). Verdicts here are
descriptive; DESIGN.md §7 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "route"))
import analyze as A  # noqa: E402  (rung 1's analysis helpers)

STRATA = (["in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8", "none8", "len16", "len32", "alt8", "deepP8",
           "pp4", "pp8", "pp32"])


def arm_of(r):
    a = r.get("arm") or "?"
    if r["name"].startswith("rfpilot"):
        a += f"[lr={r['cfg'].get('lr')}]"
    return a


def ref_arm(r):
    return {"RM": "EXd1(rung1)", "RMd0": "EXd0(rung1)"}.get(r.get("arm"))


def crit_step(rec):
    """First held-out point with route accuracy >= 0.95 at every k <= 8, confirmed by the next point; None if never."""
    pts = [(c["step"], min((c.get("acc_by_k") or {}).values(), default=0.0)) for c in rec["curve"] if c.get("acc_by_k")]
    for i, (s, a) in enumerate(pts):
        if a >= A.CRIT and (i + 1 >= len(pts) or pts[i + 1][1] >= A.CRIT):
            return s
    return None


def flip(rec, action, thr=0.5):
    return A.flip_of(rec, action, thr)


def fmt_med(vals, budget):
    return A.fmt_t95(vals, budget)


def learned(r, deferral):
    e = A.final(r)
    return all((e.get(st, {}).get("route_acc_P") or 0) >= 0.95 for st in ("in4", "in8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--ref", default=None, help="rung-1 records folder for the EX reference distribution")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = A.load(args.folders)
    groups = {}
    for r in recs:
        groups.setdefault(arm_of(r), []).append(r)
    if args.ref:
        for r in A.load([args.ref]):
            g = ref_arm(r)
            if g:
                groups.setdefault(g, []).append(r)
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-2 results: {len(recs)} runs" + (f" (+ rung-1 reference from {args.ref})" if args.ref else "") + "\n")
    for field, title in (("route_acc", "Route accuracy (committed)"), ("route_acc_P", "Route accuracy on P-words"),
                         ("acc", "Answer accuracy (committed)"), ("samp_acc_P", "Sampled-route accuracy on P-words"),
                         ("p_correct_marg_P", "p(target) under the route mixture, P-words"),
                         ("cf_count2_prev", "Probe (a): count 2 -> PREV"), ("cf_pending_set", "Probe (b): flagged ∧ pending -> SET")):
        A.table(groups, out, field, title, strata=STRATA)

    out("\n## Learning: held-out criterion (all k ≤ 8 at route accuracy ≥ 0.95, confirmed), SET / PREV flips, episodes and rollouts to criterion\n")
    out("| arm | n | learned (P-words ≥ 0.95 at in4, in8) | criterion step | SET flip | PREV flip | episodes to criterion (median) | rollouts / query | final crit_obj |")
    out("|---|---|---|---|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        budget = rs[0]["train"]["steps"]; batch = rs[0]["train"]["batch"]; rpq = rs[0]["train"].get("rollouts_per_query", 0)
        cs = [crit_step(r) for r in rs]
        med = A.cmedian(cs)
        n_learn = sum(learned(r, 0) for r in rs)
        obj = A.mean_se([r["curve"][-1].get("crit_obj") for r in rs])[0]
        out(f"| {g} | {len(rs)} | {n_learn}/{len(rs)} | {fmt_med(cs, budget)} | {fmt_med([flip(r, 'SET') for r in rs], budget)} | "
            f"{fmt_med([flip(r, 'PREV') for r in rs], budget)} | {'–' if med is None else int(med * batch)} | {rpq} | {obj:.3f} |")

    out("\n## What each estimator sees (training batches, 100-step windows): label reached / oracle route among the samples or kept by the beam, P-word-heavy lengths k = 3, 5, 8, at steps 100, 500, 1000, 2000, end\n")
    for g, rs in sorted(groups.items()):
        if "search_hit" not in rs[0]["curve"][-1]:
            continue
        cells = []
        for step in (100, 500, 1000, 2000, rs[0]["curve"][-1]["step"]):
            vals = []
            for r in rs:
                c = next((c for c in r["curve"] if c["step"] == step), None)
                if c:
                    vals.append([c["search_hit_by_k"].get(str(k), c["search_hit_by_k"].get(k, np.nan)) for k in (3, 5, 8)]
                                + [c["search_oracle_by_k"].get(str(k), c["search_oracle_by_k"].get(k, np.nan)) for k in (3, 5, 8)])
            if vals:
                v = np.nanmean(np.array(vals, float), 0)
                cells.append(f"@{step}: hit {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} oracle {v[3]:.2f}/{v[4]:.2f}/{v[5]:.2f}")
        out(f"- {g}: " + "; ".join(cells))

    out("\n## Preregistered (DESIGN.md §7)\n")
    def counts(arm, test, label):
        rs = groups.get(arm, [])
        if rs:
            out(f"- {label}: {sum(bool(test(r)) for r in rs)}/{len(rs)}")
    ex_set = {d: A.cmedian([flip(r, "SET") for r in groups.get(f"EXd{d}", []) + groups.get(f"EXd{d}(rung1)", [])]) for d in (1, 0)}
    ex_crit = {d: A.cmedian([crit_step(r) for r in groups.get(f"EXd{d}", []) + groups.get(f"EXd{d}(rung1)", [])]) for d in (1, 0)}
    out(f"- EX reference (this grid + rung 1): SET flip median δ1 {ex_set[1]}, criterion median δ1 {ex_crit[1]}; δ0 {ex_set[0]} / {ex_crit[0]}")
    for arm, factor in (("IW16d1", 2), ("IW4d1", 4), ("RF16d1", 4), ("EI16d1", 4), ("EI4d1", 4)):
        counts(arm, lambda r: learned(r, 1), f"P1/P2 {arm} learned")
        if ex_crit[1]:
            counts(arm, lambda r, f=factor: (crit_step(r) or 1e9) <= f * ex_crit[1], f"P1/P2 {arm} criterion within {factor}× EX")
            counts(arm, lambda r, f=factor: (flip(r, 'SET') or 1e9) <= f * (ex_set[1] or 1e9), f"P1/P2 {arm} SET flip within {factor}× EX")
    x, y = A.paired(groups, "RF16d1", "IW16d1", lambda r: crit_step(r) or 1e5)
    if x:
        (pg, pl), d = A.p_gt(x, y)
        out(f"- P2 criterion step RF16d1 − IW16d1: n = {len(x)}, mean {d:+.0f}, P(RF slower) = {pg:.3f}, P(RF faster) = {pl:.3f}")
    for arm in ("EXd0", "IW4d0", "IW16d0", "RF16d0", "EI4d0", "EI16d0"):
        counts(arm, lambda r: learned(r, 0), f"P3 {arm} learned (δ0)")
    for arm in sorted(groups):
        rs = [r for r in groups[arm] if learned(r, 0)]
        if not rs:
            continue
        e = lambda r, st, f: A.final(r).get(st, {}).get(f)
        d = 0 if "d0" in arm else 1
        prof = [(e(r, "len32", "route_acc"), e(r, "in8", "cf_count2_prev"), e(r, "in8", "cf_pending_set"), e(r, "pp8", "route_acc"),
                 e(r, "in8", "samp_acc_P"), (e(r, "in8", "e_steps_P") or 0) - (e(r, "in8", "oracle_steps_P") or 0), r["curve"][-1].get("crit_obj")) for r in rs]
        p = np.array([[np.nan if v is None else v for v in row] for row in prof], float)
        mn = np.nanmin(p, 0); mx = np.nanmax(p, 0)
        out(f"- P4/P5 {arm} (learned runs n = {len(rs)}, δ{d}): len32 {mn[0]:.2f}–{mx[0]:.2f}; probe a {mn[1]:.2f}–{mx[1]:.2f}; probe b {mn[2]:.2f}–{mx[2]:.2f}; "
            f"pp8 {mn[3]:.2f}–{mx[3]:.2f}; sampled acc P {mn[4]:.2f}–{mx[4]:.2f}; E[steps] − oracle {mn[5]:+.2f}…{mx[5]:+.2f}; crit_obj {mn[6]:.3f}–{mx[6]:.3f}")
    A.deviations(groups, out)
    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
