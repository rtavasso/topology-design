"""Analysis of rung-2 results (experiments/routefind/DESIGN.md §§4, 7; revision 4).

    PYTHONPATH=src python experiments/routefind/analyze.py results/kaggle_route-rf_main [--ref experiments/route/results_data/main] [--md out.md]

Arms are signal × deferral [× answer pool] (e.g. IW16d1, EI16Bd0, PFIS16d1p4). --ref adds rung 1's exact-marginal runs (RM -> EXd1(rung1), RMd0 ->
EXd0(rung1)) as a reference distribution for SET-flip and criterion times (rung-1 records have no crit_obj). Verdicts
here are descriptive; DESIGN.md §7 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "route"))
import analyze as A  # noqa: E402  (rung 1's analysis helpers)

STRATA = ["in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8", "none8", "len16", "len32", "alt8", "deepP8",
          "pp4", "pp6", "pp8", "pp16", "pp32"]
QN = 6


def arm_of(r):
    a = r.get("arm") or "?"
    if r["name"].startswith("rfpilot"):
        a += f"[lr={r['cfg'].get('lr')}]"
    return a


def crit_step(rec, key="min_acc_by_k"):
    """First held-out point with shortest-correct-route accuracy (any shortest correct program; key="acc_by_k" for the
    oracle's program) >= 0.95 at every k <= 8, CONFIRMED by the next point (a crossing at the last point does not
    count: code review 1, finding 8); None if never. Rung-1 records have no min_acc_by_k: fall back to acc_by_k."""
    if rec["curve"] and key not in rec["curve"][-1]:
        key = "acc_by_k"
    pts = [(c["step"], min((c.get(key) or {}).values(), default=0.0)) for c in rec["curve"] if c.get(key)]
    for i in range(len(pts) - 1):
        if pts[i][1] >= A.CRIT and pts[i + 1][1] >= A.CRIT:
            return pts[i][0]
    return None


def learned(r, field="min_acc_P"):
    """Reached a shortest correct program on P-words at in4 and in8 (program-agnostic; field="route_acc_P" for the
    oracle's program). Rung-1 records lack min_acc_P: fall back to route_acc_P."""
    e = A.final(r)
    if field not in e.get("in4", {}):
        field = "route_acc_P"
    return all((e.get(st, {}).get(field) or 0) >= 0.95 for st in ("in4", "in8"))


def program(r):
    """Which shortest program a run uses on P-words (in8, committed route, shortest-correct routes only): the fraction
    of its SETs taken AT the P-leaving state (L, "late": 1; E, "early", SET one state before and carry: 0), and the
    fraction that are neither. The oracle is E under deferral 1, L under 0."""
    e = A.final(r).get("in8", {})
    return e.get("set_late_P"), e.get("set_other_P")


def prog_label(r):
    p, o = program(r)
    if p is None:
        return "–"
    if (o or 0) > 0.25:
        return "other"
    return "L" if p >= 0.75 else ("E" if p <= 0.25 else "mix")


def success(r, ex_crit):
    """DESIGN.md §1 (revision 4): learned (shortest-correct committed route on P-words at in4 and in8 >= 0.95) AND the
    held-out criterion within 3x EX's median criterion step on the same deferral in this grid. Determinism is reported
    separately (a well-fit posterior-matching router is stochastic by derivation)."""
    c = crit_step(r)
    return learned(r) and ex_crit is not None and c is not None and c <= 3 * ex_crit


def g_(d, k):
    return (d or {}).get(str(k), (d or {}).get(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = A.load(args.folders)
    groups = {}
    for r in recs:
        groups.setdefault(arm_of(r), []).append(r)
    if args.ref:
        for r in A.load([args.ref]):
            g = {"RM": "EXd1(rung1)", "RMd0": "EXd0(rung1)", "RMT": "RTd1(rung1)"}.get(r.get("arm"))
            if g:
                groups.setdefault(g, []).append(r)
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-2 results: {len(recs)} runs" + (f" (+ rung-1 reference from {args.ref})" if args.ref else "") + "\n")
    for field, title in (("min_acc_P", "Shortest-correct-route accuracy on P-words (any shortest correct program)"),
                         ("set_late_P", "Program identity on P-words (shortest-correct routes): fraction of SETs at the P-leaving state (L = 1, E = 0)"),
                         ("set_other_P", "Program identity on P-words: fraction of SETs neither early nor late"),
                         ("route_acc", "Route accuracy (committed = the oracle's program)"), ("route_acc_P", "Oracle-route accuracy on P-words"),
                         ("acc", "Answer accuracy (committed)"), ("samp_acc_P", "Sampled-route accuracy on P-words"),
                         ("q_steps", "Committed steps"),
                         ("cf_count2_prev", "Probe (a): count 2 -> PREV"), ("cf_pending_set", "Probe (b): flagged ∧ pending -> SET")):
        A.table(groups, out, field, title, strata=STRATA)

    out("\n## Learning: held-out criterion (all k ≤ 8 at shortest-correct-route accuracy ≥ 0.95, confirmed by the next point), flips, episodes and rollouts to criterion, clipping\n")
    out("Learned and criterion are program-agnostic (any shortest correct route); 'E / L' counts learned runs by program (E: SET early and carry; L: SET late at the P-leaving state; the oracle is E under δ1, L under δ0); flips are teacher-forced at the oracle's states.\n")
    out("| arm | n | learned | E / L / other-or-mixed | criterion step | oracle-route criterion | SET flip (oracle) | PREV flip | episodes to crit. (median) | rollouts to crit. (median) | final crit_obj | grad-clip binding (mean frac.) |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ex_obj = {}
    for g, rs in sorted(groups.items()):
        budget = rs[0]["train"]["steps"]; batch = rs[0]["train"]["batch"]; rpq = rs[0]["train"].get("rollouts_per_query", 0)
        cs = [crit_step(r) for r in rs]; med = A.cmedian(cs)
        objs = [r["curve"][-1].get("crit_obj") for r in rs]
        obj = A.mean_se(objs)[0]
        if g in ("EXd0", "EXd1"):
            ex_obj[g[3]] = obj
        clip = A.mean_se([np.mean([c.get("clip_frac", np.nan) for c in r["curve"]]) for r in rs])[0]
        lab = [prog_label(r) for r in rs if learned(r)]
        nE, nL, nX = lab.count("E"), lab.count("L"), len(lab) - lab.count("E") - lab.count("L")
        out(f"| {g} | {len(rs)} | {sum(learned(r) for r in rs)}/{len(rs)} | {nE} / {nL} / {nX} | {A.fmt_t95(cs, budget)} | {A.fmt_t95([crit_step(r, 'acc_by_k') for r in rs], budget)} | "
            f"{A.fmt_t95([A.flip_of(r, 'SET') for r in rs], budget)} | {A.fmt_t95([A.flip_of(r, 'PREV') for r in rs], budget)} | "
            f"{'–' if med is None else int(med * batch)} | {'–' if med is None or not rpq else int(med * batch * QN * rpq)} | "
            f"{obj:.3f} | {clip:.2f} |")

    out("\n## What each estimator sees (held-out set, words WITH a P, k = 3 / 4 / 5 / 8): at init and over training\n")
    out("oracle_kept = the route imitation would use is the oracle's (EI: beam MAP; SI: best correct sample); "
        "oracle_any = some sampled route is the oracle's (sampled arms); hit = the label is reached\n")
    for g, rs in sorted(groups.items()):
        if "search_init" not in rs[0]:
            continue
        for key in ("oracle_kept", "oracle_any", "hit"):
            if key + "_P_by_k" not in rs[0]["search_init"]:
                continue
            cells = []
            v0 = np.nanmean([[np.nan if g_(r["search_init"][key + "_P_by_k"], k) is None else g_(r["search_init"][key + "_P_by_k"], k)
                              for k in (3, 4, 5, 8)] for r in rs], 0)
            cells.append("init " + "/".join(f"{x:.2f}" for x in v0))
            for step in (500, 1000, 2000, 4000, rs[0]["curve"][-1]["step"]):
                vals = []
                for r in rs:
                    c = next((c for c in r["curve"] if c["step"] == step and "search" in c), None)
                    if c:
                        vals.append([np.nan if g_(c["search"][key + "_P_by_k"], k) is None else g_(c["search"][key + "_P_by_k"], k) for k in (3, 4, 5, 8)])
                if vals:
                    cells.append(f"@{step} " + "/".join(f"{x:.2f}" for x in np.nanmean(np.array(vals, float), 0)))
            out(f"- {g} {key}: " + "; ".join(dict.fromkeys(cells)))

    out("\n## Per-seed: what the beam / samples saw at init (oracle_kept on k = 3 P-words) against the outcome (EI, SI arms)\n")
    for g, rs in sorted(groups.items()):
        if not (g.startswith("EI") or g.startswith("SI")) or "search_init" not in rs[0]:
            continue
        def bl(r):
            v = r["search_init"].get("beam_set_late_P")
            return "–" if v is None else ("L" if v >= 0.5 else "E") + f" ({v:.2f})"
        cells = [f"s{r['train']['seed']}: init kept {g_(r['search_init']['oracle_kept_P_by_k'], 3) or 0:.2f}, init beam program {bl(r)} → "
                 f"{'learned ' + prog_label(r) if learned(r) else 'not'} (crit {crit_step(r)})" for r in sorted(rs, key=lambda r: r['train']['seed'])]
        out(f"- {g}: " + "; ".join(cells))

    out("\n## Critic and importance-weight diagnostics (mean over the last 1,000 steps)\n")
    for g, rs in sorted(groups.items()):
        ks = sorted({k for r in rs for c in r["curve"] for k in (c.get("diag") or {})})
        if not ks:
            continue
        last = rs[0]["curve"][-1]["step"]
        cells = []
        for k in ks:
            v = [np.mean([c["diag"][k] for c in r["curve"] if c["step"] > last - 1000 and c.get("diag")]) for r in rs]
            cells.append(f"{k} {np.nanmin(v):.3f}…{np.nanmax(v):.3f}")
        out(f"- {g}: " + "; ".join(cells))

    out("\n## Preregistered (DESIGN.md §7, revision 4)\n")
    def counts(arm, test, label):
        rs = groups.get(arm, [])
        if rs:
            out(f"- {label}: {sum(bool(test(r)) for r in rs)}/{len(rs)}")
    ex_set = {d: A.cmedian([A.flip_of(r, "SET") for r in groups.get(f"EXd{d}", []) + groups.get(f"EXd{d}(rung1)", [])]) for d in (1, 0)}
    ex_crit_ref = {d: A.cmedian([crit_step(r) for r in groups.get(f"EXd{d}", []) + groups.get(f"EXd{d}(rung1)", [])]) for d in (1, 0)}
    ex_crit = {"d1": A.cmedian([crit_step(r) for r in groups.get("EXd1", [])]),
               "d1p4": A.cmedian([crit_step(r) for r in groups.get("EXd1p4", [])])}
    out(f"- EX reference: δ1 SET flip {ex_set[1]}, criterion {ex_crit_ref[1]} (this grid + rung 1); δ0 {ex_set[0]} / {ex_crit_ref[0]}; "
        f"success clock (this grid only): δ1 {ex_crit['d1']}, δ1p4 {ex_crit['d1p4']}; crit_obj {ex_obj}")
    core = ("EX", "EXAL", "PF", "IW16", "PFIS16", "TB16", "RL16", "ACMC16", "AC16", "EI16")
    arms = [f"{a}d1" for a in core] + [f"{a}d0" for a in core] + [f"{a}d1p4" for a in core] + ["SI16d1", "ELL16d1", "AC16T0d1", "EI16Bd0"]
    for arm in arms:
        counts(arm, learned, f"{arm} learned (shortest-correct committed route on P-words ≥ 0.95 at in4 and in8)")
        clock = ex_crit.get(arm[arm.index("d"):]) if arm.endswith(("d1", "d1p4")) else None
        if clock:
            counts(arm, lambda r: success(r, clock), f"{arm} SUCCESS (learned and criterion ≤ 3× EX = {3 * clock:.0f} steps)")
        for lab in ("E", "L", "mix", "other"):
            counts(arm, lambda r: learned(r) and prog_label(r) == lab, f"{arm} learned with program {lab}")
        counts(arm, lambda r: learned(r) and (A.final(r).get("pp8", {}).get("route_acc") or 0) >= 0.95, f"{arm} learned and passes pp8 (PP precedence)")
        counts(arm, lambda r: learned(r) and (A.final(r).get("in8", {}).get("samp_acc_P") or 0) >= 0.95, f"{arm} learned and deterministic (sampled acc. on P-words ≥ 0.95)")
    for a, b_ in (("EXALd0", "EXd0"), ("PFd0", "EXALd0"), ("PFIS16d1", "IW16d1"), ("PFIS16d0", "EXALd0"), ("TB16d1", "PFIS16d1"),
                  ("ACMC16d1", "RL16d1"), ("AC16d1", "ACMC16d1"), ("AC16d1", "RL16d1"), ("AC16d0", "RL16d0"),
                  ("EI16d1p4", "IW16d1p4"), ("PFIS16d1p4", "PFIS16d1")):
        x, y = A.paired(groups, a, b_, lambda r: crit_step(r) or 1e5)
        if x:
            (pg, pl), d = A.p_gt(x, y)
            out(f"- criterion step {a} − {b_}: n = {len(x)}, mean {d:+.0f}, P(A slower) = {pg:.3f}, P(A faster) = {pl:.3f}")
    out("\n### Checks on learned runs (P4, P5): the generalization profile and determinism\n")
    for arm in sorted(groups):
        rs = [r for r in groups[arm] if learned(r)]
        if not rs:
            continue
        d = "0" if "d0" in arm else "1"   # crit_obj−EX compares within deferral
        e = lambda r, st, f: A.final(r).get(st, {}).get(f)
        cols = {"len16−in8": lambda r: (e(r, "len16", "route_acc") or 0) - (e(r, "in8", "route_acc") or 0),
                "len32−in8": lambda r: (e(r, "len32", "route_acc") or 0) - (e(r, "in8", "route_acc") or 0),
                "probe a": lambda r: e(r, "in8", "cf_count2_prev"), "probe b": lambda r: e(r, "in8", "cf_pending_set"),
                "pp4": lambda r: e(r, "pp4", "route_acc"), "pp8": lambda r: e(r, "pp8", "route_acc"), "pp32": lambda r: e(r, "pp32", "route_acc"),
                "samp acc P": lambda r: e(r, "in8", "samp_acc_P"),
                "E[steps]−oracle": lambda r: (e(r, "in8", "e_steps_P") or 0) - (e(r, "in8", "oracle_steps_P") or 0),
                "crit_obj−EX": lambda r: (r["curve"][-1].get("crit_obj") or np.nan) - ex_obj.get(d, np.nan)}
        cells = []
        for name, f in cols.items():
            v = np.array([np.nan if f(r) is None else f(r) for r in rs], float)
            cells.append(f"{name} {np.nanmin(v):+.2f}…{np.nanmax(v):+.2f}" if np.isfinite(v).any() else f"{name} –")
        out(f"- {arm} (n = {len(rs)}): " + "; ".join(cells))
    A.deviations(groups, out)
    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
