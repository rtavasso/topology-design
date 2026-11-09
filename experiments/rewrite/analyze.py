"""Analysis of a rung-4 grid on the rewrite carrier (experiments/rewrite/DESIGN.md §6 predictions, §12a.7 protocol).

    PYTHONPATH=src python3 experiments/rewrite/analyze.py --results results/kaggle_rewrite-main/results \
        --derived experiments/rewrite/derive/derived.json --out experiments/rewrite/results_data/main_report.md

Everything is computed from the per-instance final outcomes saved with each run (masked and unmasked success, failure
category, first-decision correctness, the stored ladder outcomes of the same instance) and the curves; the symbolic
rungs are evaluated on exactly the instances each cell contains, so floors and rungs are matched to the router's
denominator. Every cell prints its n. Run classification from the training-world curve (L32). Tested on fixtures cut
from real run records (tests/test_rewrite.py)."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import statistics
import sys

TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
PAIRS_HELD = ["AI", "CI"]
TRIPLES = ["ACI", "ACD", "AID", "CID"]
FULL = ["ACID"]
CLASS = {**{w: "train" for w in TRAIN_WORLDS}, **{w: "pair" for w in PAIRS_HELD}, **{w: "triple" for w in TRIPLES}, "ACID": "full"}


def load_results(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            out.append(json.load(f))
    return out


def arm_of(r):
    return r["cfg"]["arm"]


# ---------------- per-instance selections ----------------

def instances(run, tag="eval5", **where):
    """Per-instance final outcomes of one run filtered by equality on keys (world, binding, GP1, k, ...) or by a callable."""
    rows = run["final"].get(tag, {}).get("per_instance", [])
    out = []
    for pi in rows:
        ok = True
        for key, val in where.items():
            if callable(val):
                if not val(pi):
                    ok = False; break
            elif isinstance(val, (list, tuple, set)):
                if pi[key] not in val:
                    ok = False; break
            elif pi[key] != val:
                ok = False; break
        if ok:
            out.append(pi)
    return out


def rate(rows, key="ok_m"):
    return (sum(bool(r[key]) for r in rows) / len(rows)) if rows else None


def ladder_rate(rows, d):
    return (sum(bool(r["ladder"][str(d)]) for r in rows) / len(rows)) if rows else None


def bootstrap_ci(rows, key="ok_m", n=1000, seed=0):
    import random
    rng = random.Random(seed)
    if not rows:
        return (None, None)
    vals = [bool(r[key]) for r in rows]
    means = []
    for _ in range(n):
        means.append(sum(rng.choice(vals) for _ in vals) / len(vals))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def by_seed(runs, arm):
    return sorted([r for r in runs if arm_of(r) == arm], key=lambda r: r["cfg"]["seed"])


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def summarize_seeds(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{statistics.median(vals):.2f} [{min(vals):.2f}, {max(vals):.2f}] n={len(vals)}"


# ---------------- run classification from the curve (L32) ----------------

def curve_series(run, tag="eval5", worlds=None, key="ok_m"):
    xs, ys = [], []
    for pt in run["curve"]:
        cells = pt.get(tag, {})
        num = den = 0
        for ck, c in cells.items():
            w = ck.split("|")[0]
            if worlds and w not in worlds:
                continue
            num += c[key]; den += c["n"]
        if den:
            xs.append(pt["step"]); ys.append(num / den)
    return xs, ys


def classify(ys):
    """learned: final ≥ 0.9·max and max ≥ 0.5; collapsed: max − final ≥ 0.2; slow-rising: last quarter's mean above the
    previous quarter's by ≥ 0.05 and final < 0.5; flat otherwise."""
    if not ys:
        return "no curve"
    mx, fin = max(ys), ys[-1]
    q = max(1, len(ys) // 4)
    last, prev = statistics.mean(ys[-q:]), statistics.mean(ys[-2 * q:-q]) if len(ys) >= 2 * q else ys[0]
    if mx - fin >= 0.2:
        return "learned-then-collapsed"
    if mx >= 0.5 and fin >= 0.9 * mx:
        return "learned"
    if last - prev >= 0.05 and fin < 0.5:
        return "slow-rising"
    return "flat" if mx < 0.5 else "learned (partial)"


# ---------------- the report ----------------

def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def report(runs, derived=None):
    md = []
    arms = sorted({arm_of(r) for r in runs}, key=lambda a: ["RT-canon", "RT-post", "RT-all", "RT-absolute", "EI16", "PF", "RT-examples", "RT-shuffled"].index(a) if a in ["RT-canon", "RT-post", "RT-all", "RT-absolute", "EI16", "PF", "RT-examples", "RT-shuffled"] else 99)
    md.append(f"runs: { {a: len(by_seed(runs, a)) for a in arms} }")

    # run classification
    rows = []
    for a in arms:
        for r in by_seed(runs, a):
            xs, ys = curve_series(r, "eval5", TRAIN_WORLDS)
            xs2, ys2 = curve_series(r, "eval5", PAIRS_HELD + TRIPLES + FULL)
            rows.append([r["cfg"]["name"], classify(ys), fmt(max(ys) if ys else None), fmt(ys[-1] if ys else None), classify(ys2), fmt(ys2[-1] if ys2 else None),
                         fmt(r["curve"][-1].get("loss"), 3), f"{r['seconds'] / 60:.0f} min"])
    md.append("\n## Runs classified from the curve (masked success, eval5 curve subsets; training worlds | held-out worlds)\n\n" +
              table(rows, ["run", "class (train)", "max", "final", "class (held-out)", "final", "loss", "time"]))

    # success by arm × world class × binding (eval5, masked / unmasked), seed medians
    rows = []
    for a in arms:
        for cls in ("train", "pair", "triple", "full"):
            for b in (1, 0):
                worlds = [w for w, c in CLASS.items() if c == cls]
                vm = [rate(instances(r, "eval5", world=worlds, binding=bool(b))) for r in by_seed(runs, a)]
                vu = [rate(instances(r, "eval5", world=worlds, binding=bool(b)), "ok_u") for r in by_seed(runs, a)]
                vf = [rate(instances(r, "eval5", world=worlds, binding=bool(b)), "first_ok") for r in by_seed(runs, a)]
                n = len(instances(by_seed(runs, a)[0], "eval5", world=worlds, binding=bool(b))) if by_seed(runs, a) else 0
                g = [ladder_rate(instances(r, "eval5", world=worlds, binding=bool(b)), 1) for r in by_seed(runs, a)[:1]]
                rows.append([a, cls, b, n, summarize_seeds(vm), summarize_seeds(vu), summarize_seeds(vf), fmt(g[0] if g else None)])
    md.append("\n## Success on eval5 by arm × world class × binding: masked | unmasked | first decision correct; greedy rung on the same instances\n\n" +
              table(rows, ["arm", "class", "binding", "n/seed", "masked", "unmasked", "first ok", "greedy d=1"]))

    # P2 — masked is the pass reading (the fair comparison with the ladder and EI16); unmasked reported alongside (review 2, S2)
    md.append("\n## P2 (Q1): held-out pairs' binding stratum, eval5 — RT-canon against the greedy rung on the same instances; RT-all's ceiling. Pass rule on the masked reading; unmasked alongside.")
    rows = []
    for a in ("RT-canon", "RT-post", "RT-all", "EI16", "RT-examples", "RT-shuffled", "PF", "RT-absolute"):
        if a not in arms:
            continue
        for cls, worlds in (("pair", PAIRS_HELD), ("triple+full", TRIPLES + FULL)):
            per = []
            for r in by_seed(runs, a):
                sel = instances(r, "eval5", world=worlds, binding=True)
                per.append((rate(sel), ladder_rate(sel, 1), len(sel), rate(sel, "ok_u")))
            above0 = sum(1 for s_, g_, _, _ in per if s_ is not None and s_ > g_)
            joint = sum(1 for s_, g_, _, _ in per if s_ is not None and s_ - g_ >= 0.2 and s_ >= 0.7)     # both conditions in the same seed
            ge09 = sum(1 for s_, g_, _, _ in per if s_ is not None and s_ >= 0.9)
            rows.append([a, cls, per[0][2] if per else 0, summarize_seeds([p[0] for p in per]), summarize_seeds([p[3] for p in per]),
                         fmt(per[0][1] if per else None), f"{above0}/{len(per)}", f"{joint}/{len(per)}", f"{ge09}/{len(per)}"])
    md.append(table(rows, ["arm", "worlds", "n/seed", "masked (median [min,max])", "unmasked", "greedy d=1", "> greedy", "≥ greedy+0.2 and ≥ 0.7 (same seed)", "≥ 0.9"]))
    md.append("P2 passes when RT-canon on the pairs meets both conditions in ≥ 6/10 seeds, exceeds greedy on triples+full in ≥ 6/10, and RT-all is ≥ 0.9 on the same strata.")

    # P3 — matched on (world, k, binding) cells reweighted to eval7's composition (review 2, S1); training worlds primary
    md.append("\n## P3 (Q2): sizes 6–7 (eval7, k ≤ 4) against size ≤ 5 on the same (world, k, binding) cells reweighted to eval7's composition, masked; RT-absolute split by addressability")
    un = (derived or {}).get("eval7_unaddressable", {})

    def matched_ratio(r, worlds):
        e7 = instances(r, "eval7", world=worlds)
        e5 = instances(r, "eval5", world=worlds, k=[1, 2, 3, 4])
        cells7, cells5 = {}, {}
        for pi in e7:
            cells7.setdefault((pi["world"], pi["k"], pi["binding"]), []).append(pi)
        for pi in e5:
            cells5.setdefault((pi["world"], pi["k"], pi["binding"]), []).append(pi)
        num = den = n_used = 0.0
        for key, rows7 in cells7.items():
            if key not in cells5:
                continue
            num += sum(pi["ok_m"] for pi in rows7)
            den += len(rows7) * rate(cells5[key])
            n_used += len(rows7)
        if n_used == 0 or den == 0:
            return None, 0, len(e7)
        return num / den, int(n_used), len(e7)

    rows = []
    for a in ("RT-canon", "RT-absolute", "RT-post", "RT-all"):
        if a not in arms:
            continue
        for cls, worlds in (("train", TRAIN_WORLDS), ("held-out", PAIRS_HELD + TRIPLES + FULL)):
            ratios, covered = [], None
            abs_rows = []
            for r in by_seed(runs, a):
                ratio, n_used, n7 = matched_ratio(r, worlds)
                ratios.append(ratio); covered = (n_used, n7)
                if un:
                    e7 = instances(r, "eval7", world=worlds)
                    flag = [un.get(pi["world"], {}).get(f"{pi['t']}|{pi['s']}") for pi in e7]
                    addr = [pi for pi, f in zip(e7, flag) if f is False]
                    unad = [pi for pi, f in zip(e7, flag) if f is True]
                    abs_rows.append((rate(addr), rate(unad), len(addr), len(unad)))
            n_ok = sum(1 for x in ratios if x is not None and x >= 0.9)
            extra = ""
            if abs_rows:
                extra = f"addressable {summarize_seeds([x[0] for x in abs_rows])} (n {abs_rows[0][2]}); unaddressable {summarize_seeds([x[1] for x in abs_rows])} (n {abs_rows[0][3]})"
            rows.append([a, cls, summarize_seeds([rate(instances(r, 'eval5', world=worlds, k=[1, 2, 3, 4])) for r in by_seed(runs, a)]),
                         summarize_seeds([rate(instances(r, 'eval7', world=worlds)) for r in by_seed(runs, a)]),
                         summarize_seeds(ratios), f"{covered[0]}/{covered[1]}" if covered else "–", f"{n_ok}/{len(ratios)}", extra])
    md.append(table(rows, ["arm", "worlds", "size ≤ 5, k ≤ 4", "sizes 6–7", "matched ratio", "eval7 n matched", "ratio ≥ 0.9", "RT-absolute split"]))
    md.append("P3 passes when the structural RT-canon's matched ratio on the training worlds is ≥ 0.9 in ≥ 7/10 seeds, and RT-absolute's failures are the unaddressable instances (success on addressable ≈ its size-5 success, ≈ 0 on unaddressable).")

    # P4 — on cells where the rungs are separated (review 2, B5/S7): d2 − d1 ≥ 0.1 and d3 − d2 ≥ 0.1 on the eval instances
    # themselves; strict inequalities; unmasked alongside. The pass reading is on the cells amended in DESIGN §16 from the
    # derive step, before any run: every (world, binding, GP1, k) cell of the training worlds at k ≥ 4 with n ≥ 10. The §6
    # preregistered set (GP1 instances by (world, k)) is printed beside it; on it the d = 1 and d = 2 rungs tie near zero.
    md.append("\n## P4 (Q3): training worlds at k ≥ 4 (eval5), on cells whose rungs are separated (d2 − d1 ≥ 0.1 and d3 − d2 ≥ 0.1); pass reading on the §16 cells (all strata, (world, binding, GP1, k), n ≥ 10), masked; the §6 cells (GP1 by (world, k)) beside it")

    def separated_cells(r, prereg=False, upper_only=False, with_keys=False):
        cells = {}
        pool = instances(r, "eval5", world=TRAIN_WORLDS, GP1=True, k=[4, 5, 6]) if prereg else instances(r, "eval5", world=TRAIN_WORLDS, k=[4, 5, 6])
        for pi in pool:
            key = (pi["world"], pi["k"]) if prereg else (pi["world"], int(bool(pi["binding"])), int(bool(pi["GP1"])), pi["k"])
            cells.setdefault(key, []).append(pi)
        out, keys = [], []
        for key, rows_ in cells.items():
            if len(rows_) < 10:
                continue
            d1, d2, d3 = ladder_rate(rows_, 1), ladder_rate(rows_, 2), ladder_rate(rows_, 3)
            if (d3 - d2 >= 0.1) if upper_only else (d2 - d1 >= 0.1 and d3 - d2 >= 0.1):
                out.extend(rows_)
                keys.append("|".join(str(x) for x in key))
        return (out, sorted(keys)) if with_keys else out

    # the cell set must be the one the derive step fixed before the runs (DESIGN §16); a drift of the data would move it silently
    expected = sorted((derived or {}).get("p4_amended_pooled", {}).get("cells", []))
    if expected and "RT-canon" in arms:
        got = sorted(separated_cells(by_seed(runs, "RT-canon")[0], with_keys=True)[1])
        if got != expected:
            md.append(f"\n**P4 CELL SET MISMATCH** — derive step fixed {expected}; the run records give {got}. The dataset or the records drifted; P4 is not read until this is explained.\n")
            print("P4 CELL SET MISMATCH", expected, got, file=sys.stderr)
        else:
            md.append(f"\nThe §16 cells recomputed from the run records equal the derive step's ({len(expected)} cells).")
    # the rungs under the random tie-break on the same cells, from the derive step (the lower separation is tie-break-dependent, §16)
    pr = (derived or {}).get("p4_amended_pooled", {})
    if pr.get("d1r") is not None:
        md.append(f"\nRandom-tie-break rungs on the §16 cells (derive step): d=1 {pr['d1r']:.2f}, d=2 {pr['d2r']:.2f}, d=3 {pr['d3r']:.2f}; the lower separation fails under it on {len(pr.get('cells_lower_separation_fails_random', []))} of {len(pr.get('cells', []))} cells.")

    rows = []
    for a in arms:
        for label, kw in (("§16 cells", {}), ("§6 cells (GP1)", {"prereg": True}), ("upper separation only (sensitivity)", {"upper_only": True})):
            per = []
            for r in by_seed(runs, a):
                sel = separated_cells(r, **kw)
                per.append((rate(sel), ladder_rate(sel, 1), ladder_rate(sel, 2), ladder_rate(sel, 3), len(sel), rate(sel, "ok_u"), bootstrap_ci(sel) if sel else (None, None)))
            between = sum(1 for s_, d1, d2, d3, n, _, _ in per if s_ is not None and d2 < s_ < d3)
            ge3 = sum(1 for s_, d1, d2, d3, n, _, _ in per if s_ is not None and s_ >= d3)
            le1 = sum(1 for s_, d1, d2, d3, n, _, _ in per if s_ is not None and s_ <= d1)
            n_cells = per[0][4] if per else 0
            los = [p[6][0] for p in per if p[6][0] is not None]; his = [p[6][1] for p in per if p[6][1] is not None]
            ci = f"[{min(los):.2f}, {max(his):.2f}]" if los else "—"
            rows.append([a, label, n_cells, summarize_seeds([p[0] for p in per]), ci, summarize_seeds([p[5] for p in per]), fmt(per[0][1] if per else None), fmt(per[0][2] if per else None), fmt(per[0][3] if per else None),
                         f"{between}/{len(per)}" if n_cells else "rungs not separated", f"{ge3}/{len(per)}", f"{le1}/{len(per)}"])
    md.append(table(rows, ["arm", "cells", "n/seed (separated cells)", "masked", "instance bootstrap 95 % (seeds' envelope)", "unmasked", "d=1", "d=2", "d=3", "strictly between d=2 and d=3", "≥ d=3", "≤ d=1"]))
    md.append("P4 passes when RT-canon lies strictly between the d = 2 and d = 3 rungs in ≥ 7/10 seeds on the §16 cells; ≥ d = 3 in ≥ 7/10 falsifies toward 'three steps absorbed', ≤ d = 1 in ≥ 7/10 toward 'none'; if no count reaches 7/10 (the router within ≈ 0.03 of a rung), P4 is **undecided** and reported so. The seed count covers training variance only (the instances are fixed); the instance bootstrap column is the envelope over seeds of each seed's 95 % interval. The §6 row is the preregistered GP1-only reading and the upper-only row a sensitivity set; both reported, not scored.")
    # Q3's quantity per cell: the largest d whose rung RT-canon matches (0 = below greedy), with the rungs
    if "RT-canon" in arms:
        rows = []
        for k in (2, 3, 4, 5, 6):
            for b in (0, 1):
                for g in (0, 1):
                    per = [instances(r, "eval5", world=TRAIN_WORLDS, binding=lambda pi, b=b: int(bool(pi["binding"])) == b, GP1=bool(g), k=k) for r in by_seed(runs, "RT-canon")]
                    if not per or not per[0]:
                        continue
                    d1, d2, d3 = ladder_rate(per[0], 1), ladder_rate(per[0], 2), ladder_rate(per[0], 3)
                    reach = []
                    for p_ in per:
                        s_ = rate(p_)
                        # the largest d whose rung is matched, counting a rung only where it rises above the one below (ties at
                        # zero would otherwise be "matched" by a router that solves nothing; review 2, B5)
                        rungs = [0.0, d1, d2, d3]
                        reach.append(max([d for d in (1, 2, 3) if s_ >= rungs[d] and rungs[d] > rungs[d - 1]], default=0))
                    rows.append([k, b, g, len(per[0]), summarize_seeds([rate(p_) for p_ in per]), summarize_seeds([rate(p_, "ok_u") for p_ in per]), fmt(d1), fmt(d2), fmt(d3),
                                 f"median {statistics.median(reach)} (min {min(reach)}, max {max(reach)})", "yes" if (d2 - d1 >= 0.1 and d3 - d2 >= 0.1) else "no"])
        md.append("\nRT-canon by k × binding × GP1 (training worlds pooled, eval5): reach = the largest d whose rung RT-canon matches\n\n" +
                  table(rows, ["k", "binding", "GP1", "n/seed", "masked", "unmasked", "d=1", "d=2", "d=3", "reach over seeds", "separated"]))

    # P5
    if "EI16" in arms and "RT-canon" in arms:
        sel_rt = [rate(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "RT-canon")]
        sel_ei = [rate(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "EI16")]
        cis = [bootstrap_ci(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "EI16")]
        md.append(f"\n## P5 (Q5): held-out binding stratum — RT-canon median {fmt(statistics.median([v for v in sel_rt if v is not None]) if any(v is not None for v in sel_rt) else None)}, "
                  f"EI16 {summarize_seeds(sel_ei)}; EI16 bootstrap 95 % CIs per seed: {[(fmt(a), fmt(b)) for a, b in cis]}; EI16 [routes found, instances tried, rows imitated]: "
                  f"{[r.get('ei_found') for r in by_seed(runs, 'EI16')]}")

    # P6 — against the RT-post target at t (review 2, S3): total variation between the router's masked distribution over the
    # shortest-path first actions (renormalised) and the target; the lexicographic-class share for RT-canon
    rows = []
    for a in ("RT-canon", "RT-post", "PF", "RT-all"):
        if a not in arms:
            continue
        for r in by_seed(runs, a):
            sel = [pi for pi in instances(r, "eval5", world=TRAIN_WORLDS) if pi.get("p_first") and pi.get("target_first") and pi["n_classes"] > 1 and len(pi["p_first"]) > 1]
            if not sel:
                continue
            tvs, mass, canon = [], [], []
            for pi in sel:
                tot = sum(pi["p_first"].values())
                mass.append(tot)
                if tot > 0:
                    q = {k_: v / tot for k_, v in pi["p_first"].items()}
                    tvs.append(0.5 * sum(abs(q.get(k_, 0.0) - pi["target_first"].get(k_, 0.0)) for k_ in set(q) | set(pi["target_first"])))
                canon.append(bool(pi.get("first_canon")))
            rows.append([r["cfg"]["name"], len(sel), fmt(statistics.mean(mass)), fmt(statistics.mean(tvs)) if tvs else "–", fmt(statistics.mean(canon))])
    if rows:
        md.append("\n## P6 (Q6): multi-class training instances — the router's mass on shortest-path first actions, its total-variation distance to the RT-post target (0 = the target mixture), and the share of first decisions equal to the lexicographic (canonical) action\n\n" +
                  table(rows, ["run", "n", "mass on shortest-path actions", "TV to RT-post target", "canonical first action"]))

    # P7 — the ceiling weighted by the binding count per world (review 2, M3)
    if "RT-examples" in arms:
        exc = (derived or {}).get("examples_ceiling", {})
        rows = []
        for r in by_seed(runs, "RT-examples"):
            for cls, worlds in (("train", TRAIN_WORLDS), ("pair", PAIRS_HELD), ("triple+full", TRIPLES + FULL)):
                sel = instances(r, "eval5", world=worlds, binding=True)
                ws = [w for w in worlds if w in exc and exc[w]["binding_n"] > 0]
                ceil = (sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in ws) / sum(exc[w]["binding_n"] for w in ws)) if ws else None
                rt = [rate(instances(x, "eval5", world=worlds, binding=True)) for x in by_seed(runs, "RT-canon")]
                rows.append([r["cfg"]["name"], cls, len(sel), fmt(rate(sel)), fmt(rate(sel, "ok_u")), fmt(ceil), summarize_seeds(rt)])
        md.append("\n## P7 (Q7): RT-examples on the binding stratum against its derived ceiling (weighted by binding count) and stated-axiom RT-canon\n\n" + table(rows, ["run", "class", "n", "RT-examples masked", "unmasked", "derived ceiling", "RT-canon"]))

    # P8 — two readings (review 2, M4): shuffled against the cue-blind ceiling, and against ceiling × RT-canon's non-binding
    # success (what a cue-blind router of RT-canon's planning ability would score); binding stratum against RT-canon
    if "RT-shuffled" in arms:
        cb = (derived or {}).get("cue_blind", {})
        rows = []
        for w in CLASS:
            for r in by_seed(runs, "RT-shuffled"):
                sel = instances(r, "eval5", world=w)
                selb = instances(r, "eval5", world=w, binding=True)
                rt_nb = [rate(instances(x, "eval5", world=w, binding=False)) for x in by_seed(runs, "RT-canon")]
                rt_b = [rate(instances(x, "eval5", world=w, binding=True)) for x in by_seed(runs, "RT-canon")]
                ceil = cb.get(w, {}).get("ceiling")
                med_nb = statistics.median([v for v in rt_nb if v is not None]) if any(v is not None for v in rt_nb) else None
                rows.append([w, r["cfg"]["seed"], len(sel), fmt(rate(sel)), fmt(ceil), fmt(ceil * med_nb) if (ceil is not None and med_nb is not None) else "–",
                             len(selb), fmt(rate(selb)), summarize_seeds(rt_b)])
        md.append("\n## P8 (null): RT-shuffled per world against the cue-blind ceiling and against ceiling × RT-canon's non-binding success; binding stratum against RT-canon\n\n" +
                  table(rows, ["world", "seed", "n", "shuffled (all)", "cue-blind ceiling", "ceiling × RT-canon non-binding", "binding n", "shuffled (binding)", "RT-canon (binding)"]))

    # failure categories
    rows = []
    for a in arms:
        for cls, worlds in (("train", TRAIN_WORLDS), ("held-out", PAIRS_HELD + TRIPLES + FULL)):
            cats_m, cats_u, n = {}, {}, 0
            for r in by_seed(runs, a):
                for pi in instances(r, "eval5", world=worlds):
                    cats_m[pi["cat_m"]] = cats_m.get(pi["cat_m"], 0) + 1
                    cats_u[pi["cat_u"]] = cats_u.get(pi["cat_u"], 0) + 1
                    n += 1
            rows.append([a, cls, n, " ".join(f"{k}:{v / n:.2f}" for k, v in sorted(cats_m.items(), key=lambda kv: -kv[1])),
                         " ".join(f"{k}:{v / n:.2f}" for k, v in sorted(cats_u.items(), key=lambda kv: -kv[1]))])
    md.append("\n## Failure categories (eval5, pooled over seeds): masked | unmasked\n\n" + table(rows, ["arm", "class", "n", "masked", "unmasked"]))

    # k = 7–8 stratum (P1, reported)
    rows = []
    for a in arms:
        per = [instances(r, "evalk78") for r in by_seed(runs, a)]
        if per and per[0]:
            rows.append([a, len(per[0]), summarize_seeds([rate(p) for p in per]), summarize_seeds([rate(p, "ok_u") for p in per]), fmt(ladder_rate(per[0], 1), 3), fmt(ladder_rate(per[0], 2), 3), fmt(ladder_rate(per[0], 3), 3)])
    if rows:
        md.append("\n## P1 (reported): {A, C} at k = 7–8, size ≤ 5 (the symbolic rungs are at their floor here — there is no reference; a rate is reported, not compared)\n\n" + table(rows, ["arm", "n/seed", "masked", "unmasked", "d=1", "d=2", "d=3"]))
    md.append(render_verdicts(verdicts(runs, derived)))
    return "\n".join(md) + "\n"


# ---------------- the amended pass rules (DESIGN §17): one verdict per prediction, computed, not read off a table ----------------

SEEDS = {"RT-canon": 10, "RT-post": 10, "RT-all": 3, "RT-absolute": 5, "EI16": 5, "PF": 3, "RT-examples": 3, "RT-shuffled": 2}


def _key(pi):
    return f"{pi['t']}|{pi['s']}"


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _count(n_ok, n, need_frac):
    """A count over seeds against a preregistered fraction (6/10, 7/10, …) of the seeds actually present."""
    return n > 0 and n_ok >= math.ceil(need_frac * n - 1e-9)


def matched_cells(rows7, rows5):
    c7, c5 = {}, {}
    for pi in rows7:
        c7.setdefault((pi["world"], pi["k"], bool(pi["binding"])), []).append(pi)
    for pi in rows5:
        if pi["k"] <= 4:
            c5.setdefault((pi["world"], pi["k"], bool(pi["binding"])), []).append(pi)
    return [(rows, c5[key]) for key, rows in c7.items() if key in c5]


def matched_stat(cells, okf):
    """(ratio, size-<=5 success reweighted to eval7's composition, n): the P3 statistic for any per-instance outcome."""
    num = den = n = 0.0
    for rows7, rows5 in cells:
        num += sum(bool(okf(pi)) for pi in rows7)
        den += len(rows7) * sum(bool(okf(pi)) for pi in rows5) / len(rows5)
        n += len(rows7)
    return (num / den if den else None), (den / n if n else None), int(n)


def normalised_ratio(rows7, rows5, key):
    """The router's matched ratio R and the same statistic rho for the symbolic rung whose size-<=5 success on these cells
    is nearest the router's (the rungs are rules over structure: rho is what a size-blind policy of that strength scores,
    and it is above 1 where eval7's matched cells are easier). ok = R >= 0.9 rho."""
    cells = matched_cells(rows7, rows5)
    R, base, n = matched_stat(cells, lambda pi: pi[key])
    if R is None:
        return {"R": None, "rho": None, "rung": None, "n": n, "ok": None, "raw_ok": None}
    rungs = {d: matched_stat(cells, lambda pi, d=d: pi["ladder"][str(d)]) for d in (1, 2, 3)}
    usable = {d: v for d, v in rungs.items() if v[0] is not None}
    if not usable:
        return {"R": R, "rho": None, "rung": None, "n": n, "ok": None, "raw_ok": R >= 0.9}
    d = min(usable, key=lambda d: (abs(usable[d][1] - base), d))
    return {"R": R, "rho": usable[d][0], "rung": d, "n": n, "ok": R >= 0.9 * usable[d][0], "raw_ok": R >= 0.9}


def verdicts(runs, derived=None):
    """Every pass rule of DESIGN §17 as a computed verdict: {"P2": {...}, "P3": {...}, "P5": ..., "P6": ..., "P7": ..., "P8": ...}.
    Each carries "verdict" (pass / holds / falsified / undecided / not readable) and the numbers it was read from."""
    derived = derived or {}
    V = {}
    held_b = PAIRS_HELD + TRIPLES

    # ---- P2: held-out binding instances that need the unseen combination
    flags = derived.get("p2_comb")

    def comb(run, worlds):
        return [pi for pi in instances(run, "eval5", world=worlds, binding=True) if (flags or {}).get(pi["world"], {}).get(_key(pi), {}).get("comb")]

    def refs(rows):
        tw = sum(bool(flags[pi["world"]][_key(pi)]["trainw"]) for pi in rows) / len(rows) if rows else None
        return {"n": len(rows), "greedy": ladder_rate(rows, 1), "d3": ladder_rate(rows, 3), "trainw": tw,
                "floor": max(ladder_rate(rows, 1), tw) if rows else None}

    def pair_ok(rate_, g):
        return rate_ is not None and g is not None and rate_ - g >= 0.2 and rate_ >= 0.7

    p2 = {"verdict": "not readable", "why": "no p2_comb flags in derived.json" if not flags else None}
    rc, ra = by_seed(runs, "RT-canon"), by_seed(runs, "RT-all")
    if flags and rc:
        rp, rt_ = refs(comb(rc[0], PAIRS_HELD)), refs(comb(rc[0], TRIPLES))
        p2.update({"pairs": rp, "triples": rt_})
        if not rp["n"]:
            p2["why"] = "the pairs' needs-the-combination stratum is empty in these records"
        else:
            for rd in ("ok_m", "ok_u"):
                pr = [rate(comb(r, PAIRS_HELD), rd) for r in rc]
                tr = [rate(comb(r, TRIPLES), rd) for r in rc]
                n_pair = sum(pair_ok(x, rp["greedy"]) for x in pr)
                n_tri = sum(x is not None and rt_["floor"] is not None and x > rt_["floor"] for x in tr)
                n_above = sum(x is not None and x > rp["floor"] for x in pr)
                allp = [rate(comb(r, PAIRS_HELD), rd) for r in ra]
                p2[rd] = {"pair_rates": pr, "triple_rates": tr, "pair_ok": n_pair, "triple_above_floor": n_tri, "pair_above_floor": n_above, "n_seeds": len(rc),
                          "rt_all_pair_rates": allp, "rt_all_median": _median(allp), "rt_all_reaches": pair_ok(_median(allp), rp["greedy"]),
                          "ci": [bootstrap_ci(comb(r, PAIRS_HELD), rd) for r in rc],
                          "passes": _count(n_pair, len(rc), 0.6) and (rt_["n"] == 0 or _count(n_tri, len(rc), 0.6))}
            m = p2["ok_m"]
            few_above = m["pair_above_floor"] <= math.floor(0.3 * m["n_seeds"] + 1e-9)
            if m["passes"]:
                p2["verdict"] = "pass"
                p2["cue"] = "composes under the stated cue" if p2["ok_u"]["passes"] else "composes decisions when the executor supplies the action set; the cue alone does not carry the unseen combination"
            elif few_above and m["rt_all_reaches"]:
                p2["verdict"] = "falsified"
            elif few_above:
                p2["verdict"] = "undecided"; p2["why"] = "RT-canon is not above the non-composing floor, but RT-all does not reach the pair conditions either: reachability not shown"
            else:
                p2["verdict"] = "undecided"; p2["why"] = "above the non-composing floor in more than 3/10 seeds, short of the pair conditions in 6/10 (or of the triples' clause)"
    V["P2"] = p2

    # ---- P3: sizes 6–7 against size <= 5, normalised by the matched rung; the categorical head's must-fail set
    p3 = {"verdict": "not readable"}
    if rc:
        c1 = [normalised_ratio(instances(r, "eval7", world=TRAIN_WORLDS), instances(r, "eval5", world=TRAIN_WORLDS), "ok_m") for r in rc]
        ok1 = sum(bool(x["ok"]) for x in c1)
        readable1 = all(x["ok"] is not None for x in c1)
        p3["clause1"] = {"per_seed": c1, "ok": ok1, "n_seeds": len(c1), "raw_ok": sum(bool(x["raw_ok"]) for x in c1),
                         "single_axiom": [normalised_ratio(instances(r, "eval7", world=["A", "C", "I", "D"]), instances(r, "eval5", world=["A", "C", "I", "D"]), "ok_m") for r in rc],
                         "pairs": [normalised_ratio(instances(r, "eval7", world=["AC", "AD", "CD", "ID"]), instances(r, "eval5", world=["AC", "AD", "CD", "ID"]), "ok_m") for r in rc],
                         "verdict": "not readable" if not readable1 else ("pass" if _count(ok1, len(c1), 0.7) else ("falsified" if ok1 <= math.floor(0.3 * len(c1) + 1e-9) else "undecided"))}
        p3["verdict"] = p3["clause1"]["verdict"]
    rab, mf = by_seed(runs, "RT-absolute"), derived.get("eval7_mustfail")
    if rab and mf is not None:
        per = []
        for r in rab:
            e7 = instances(r, "eval7", world=TRAIN_WORLDS)
            must = [pi for pi in e7 if mf.get(pi["world"], {}).get(_key(pi)) is True]
            rest = [pi for pi in e7 if mf.get(pi["world"], {}).get(_key(pi)) is not True]
            nr = normalised_ratio(rest, instances(r, "eval5", world=TRAIN_WORLDS), "ok_u")
            per.append({"must_fail_n": len(must), "must_fail_rate": rate(must, "ok_u"), "rest_n": len(rest), "rest_rate": rate(rest, "ok_u"), **{f"rest_{k_}": v for k_, v in nr.items()}})
        ok2 = sum(bool(x["rest_ok"]) for x in per)
        art = sum(x["must_fail_rate"] is not None and x["must_fail_rate"] <= 0.05 for x in per)
        readable2 = all(x["rest_ok"] is not None for x in per)
        p3["clause2"] = {"per_seed": per, "ok": ok2, "artefact_ok": art, "n_seeds": len(per),
                         "verdict": "not readable" if not readable2 else ("holds" if _count(ok2, len(per), 0.8) else ("falsified" if ok2 <= math.floor(0.2 * len(per) + 1e-9) else "undecided"))}
    V["P3"] = p3

    # ---- P5: EI16 against RT-canon on the held-out binding stratum (masked), seed-medians
    rei = by_seed(runs, "EI16")
    p5 = {"verdict": "not readable"}
    if rc and rei:
        sel = lambda r: instances(r, "eval5", world=held_b + FULL, binding=True)
        m_rt, m_ei = _median([rate(sel(r)) for r in rc]), _median([rate(sel(r)) for r in rei])
        g = ladder_rate(sel(rc[0]), 1)
        if m_rt is not None and m_ei is not None:
            p5 = {"n": len(sel(rc[0])), "rt_median": m_rt, "ei_median": m_ei, "greedy": g, "diff": m_ei - m_rt,
                  "rt_median_u": _median([rate(sel(r), "ok_u") for r in rc]), "ei_median_u": _median([rate(sel(r), "ok_u") for r in rei]),
                  "comb": {"n": len(comb(rc[0], held_b)), "rt_median": _median([rate(comb(r, held_b)) for r in rc]), "ei_median": _median([rate(comb(r, held_b)) for r in rei])} if flags else None,
                  "verdict": "falsified" if (m_ei < g < m_rt) else ("holds" if abs(m_ei - m_rt) <= 0.1 else "differs by more than 0.1")}
    V["P5"] = p5

    # ---- P6: RT-post at the target mixture, RT-canon at the canonical class (first decision, multi-class training instances)
    def p6_stats(r):
        sel = [pi for pi in instances(r, "eval5", world=TRAIN_WORLDS) if pi.get("p_first") and pi.get("target_first") and pi["n_classes"] > 1 and len(pi["p_first"]) > 1]
        tvs, uni, canon = [], [], []
        for pi in sel:
            tot = sum(pi["p_first"].values())
            tg = pi["target_first"]
            if tot > 0:
                q = {k_: v / tot for k_, v in pi["p_first"].items()}
                tvs.append(0.5 * sum(abs(q.get(k_, 0.0) - tg.get(k_, 0.0)) for k_ in set(q) | set(tg)))
            u = 1.0 / len(pi["p_first"])
            uni.append(0.5 * sum(abs((u if k_ in pi["p_first"] else 0.0) - tg.get(k_, 0.0)) for k_ in set(pi["p_first"]) | set(tg)))
            canon.append(bool(pi.get("first_canon")))
        return {"n": len(sel), "tv": statistics.mean(tvs) if tvs else None, "tv_uniform": statistics.mean(uni) if uni else None, "canon": statistics.mean(canon) if canon else None}

    rpo = by_seed(runs, "RT-post")
    p6 = {"verdict": "not readable"}
    if rpo and rc:
        sp, sc = [p6_stats(r) for r in rpo], [p6_stats(r) for r in rc]
        if any(x["n"] for x in sp) and any(x["n"] for x in sc):
            okp = sum(x["tv"] is not None and x["tv"] < x["tv_uniform"] for x in sp)
            okc = sum(x["canon"] is not None and x["canon"] >= 0.9 for x in sc)
            p6 = {"post": sp, "canon": sc, "post_ok": okp, "canon_ok": okc,
                  "verdict": "holds" if (_count(okp, len(sp), 0.7) and _count(okc, len(sc), 0.7)) else "does not hold"}
    V["P6"] = p6

    # ---- P7: RT-examples against ceiling x RT-canon (unmasked), training binding stratum; below stated RT on held-out binding
    rex, exc = by_seed(runs, "RT-examples"), derived.get("examples_ceiling")
    p7 = {"verdict": "not readable"}
    if rex and rc and exc:
        tb = lambda r: instances(r, "eval5", world=TRAIN_WORLDS, binding=True)
        hb = lambda r: instances(r, "eval5", world=held_b + FULL, binding=True)
        ws = [w for w in TRAIN_WORLDS if w in exc and exc[w]["binding_n"] > 0]
        ceil = (sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in ws) / sum(exc[w]["binding_n"] for w in ws)) if ws else None
        m_t, m_h = _median([rate(tb(r), "ok_u") for r in rc]), _median([rate(hb(r), "ok_u") for r in rc])
        ex_t, ex_h = [rate(tb(r), "ok_u") for r in rex], [rate(hb(r), "ok_u") for r in rex]
        if ceil is not None and m_t is not None and all(x is not None for x in ex_t + ex_h) and m_h is not None:
            target = ceil * m_t
            within = sum(abs(x - target) <= 0.1 for x in ex_t)
            above = sum(x - target > 0.1 for x in ex_t)
            below_h = sum(x < m_h - 0.05 for x in ex_h)
            ge_h = sum(x >= m_h for x in ex_h)
            c1 = "holds" if within == len(ex_t) else ("above: W is read from more than the examples" if above * 3 >= 2 * len(ex_t) else ("below: the examples are not used to the ceiling" if (len(ex_t) - within - above) * 3 >= 2 * len(ex_t) else "undecided"))
            c2 = "holds" if below_h == len(ex_h) else ("falsified" if ge_h * 3 >= 2 * len(ex_h) else "undecided")
            p7 = {"ceiling": ceil, "rt_train_median_u": m_t, "target": target, "examples_train_u": ex_t, "clause1": c1, "rt_heldout_median_u": m_h, "examples_heldout_u": ex_h, "clause2": c2,
                  "train_n": len(tb(rc[0])), "heldout_n": len(hb(rc[0])),
                  "verdict": "holds" if (c1 == "holds" and c2 == "holds") else ("falsified" if c2 == "falsified" else "does not hold as stated")}
    V["P7"] = p7

    # ---- P8: RT-shuffled against RT-canon on the training worlds' binding stratum, each reading on its own
    rsh = by_seed(runs, "RT-shuffled")
    p8 = {"verdict": "not readable"}
    if rsh and rc:
        tb = lambda r: instances(r, "eval5", world=TRAIN_WORLDS, binding=True)
        out = {"n": len(tb(rc[0]))}
        for rd in ("ok_m", "ok_u"):
            m = _median([rate(tb(r), rd) for r in rc])
            sh = [rate(tb(r), rd) for r in rsh]
            if m is None or any(x is None for x in sh):
                out = None
                break
            gaps = [m - x for x in sh]
            n_gap = sum(g > 0.05 for g in gaps)
            out[rd] = {"rt_median": m, "shuffled": sh, "gaps": gaps, "verdict": "holds" if n_gap == len(gaps) else ("falsified" if n_gap == 0 else "undecided")}
        if out:
            vm, vu = out["ok_m"]["verdict"], out["ok_u"]["verdict"]
            reading = {("holds", "holds"): "the presence tokens carry the world under both readings",
                       ("falsified", "holds"): "under the mask the action set, not the presence tokens, carries the world; without the mask the tokens do",
                       ("falsified", "falsified"): "the presence tokens carry nothing the pair (t, s) does not"}.get((vm, vu), "mixed or undecided")
            p8 = {**out, "verdict": vm, "verdict_unmasked": vu, "reading": reading}
    V["P8"] = p8
    return V


def render_verdicts(V):
    f = lambda x, nd=2: "–" if x is None else f"{x:.{nd}f}"
    fl = lambda xs: " ".join(f(x) for x in xs)
    md = ["\n## Verdicts under the amended pass rules (DESIGN §17)\n"]
    rows = [[k_, v.get("verdict", "–") + (f"; unmasked: {v['verdict_unmasked']}" if "verdict_unmasked" in v else ""), v.get("cue") or v.get("reading") or v.get("why") or ""] for k_, v in V.items()]
    md.append(table(rows, ["prediction", "verdict", "reading"]))
    p2 = V["P2"]
    if "ok_m" in p2:
        rp, rt_ = p2["pairs"], p2["triples"]
        md.append(f"\n**P2** — held-out binding instances that need the unseen combination. Pairs n = {rp['n']}: greedy {f(rp['greedy'])}, lookahead-3 {f(rp['d3'])}, training-world reference {f(rp['trainw'])} (floor {f(rp['floor'])}); "
                  f"triples n = {rt_['n']}: greedy {f(rt_['greedy'])}, lookahead-3 {f(rt_['d3'])}, training-world reference {f(rt_['trainw'])} (floor {f(rt_['floor'])}).\n")
        rows = []
        for rd, name in (("ok_m", "masked"), ("ok_u", "unmasked")):
            x = p2[rd]
            lo, hi = [c[0] for c in x["ci"] if c[0] is not None], [c[1] for c in x["ci"] if c[1] is not None]
            rows.append([name, summarize_seeds(x["pair_rates"]), f"[{min(lo):.2f}, {max(hi):.2f}]" if lo else "–", f"{x['pair_ok']}/{x['n_seeds']}", f"{x['pair_above_floor']}/{x['n_seeds']}",
                         summarize_seeds(x["triple_rates"]), f"{x['triple_above_floor']}/{x['n_seeds']}", summarize_seeds(x["rt_all_pair_rates"]), "yes" if x["rt_all_reaches"] else "no", "yes" if x["passes"] else "no"])
        md.append(table(rows, ["reading", "RT-canon, pairs", "instance bootstrap 95 % (seeds' envelope)", "≥ greedy + 0.2 and ≥ 0.7", "> floor (pairs)", "RT-canon, triples", "> floor (triples)", "RT-all, pairs", "RT-all's median meets the pair conditions", "pass conditions met"]))
    p3 = V["P3"]
    if "clause1" in p3:
        c = p3["clause1"]
        md.append(f"\n**P3, clause 1** (RT-canon, masked, training worlds): matched ratio R against 0.9 × ρ of the nearest rung — per seed R {fl([x['R'] for x in c['per_seed']])}; ρ {fl([x['rho'] for x in c['per_seed']])} (rung {[x['rung'] for x in c['per_seed']]}); "
                  f"R ≥ 0.9 ρ in {c['ok']}/{c['n_seeds']} seeds (the unnormalised R ≥ 0.9 of §6: {c['raw_ok']}/{c['n_seeds']}) → {c['verdict']}. Single-axiom worlds R {fl([x['R'] for x in c['single_axiom']])}; training pairs R {fl([x['R'] for x in c['pairs']])} against ρ {fl([x['rho'] for x in c['pairs']])}.")
    if "clause2" in p3:
        c = p3["clause2"]
        md.append(f"\n**P3, clause 2** (RT-absolute, unmasked, training worlds, sizes 6–7): on the must-fail instances (n {c['per_seed'][0]['must_fail_n']}) success {fl([x['must_fail_rate'] for x in c['per_seed']])} (≤ 0.05 in {c['artefact_ok']}/{c['n_seeds']}); "
                  f"on the rest (n {c['per_seed'][0]['rest_n']}) success {fl([x['rest_rate'] for x in c['per_seed']])}, matched ratio R {fl([x['rest_R'] for x in c['per_seed']])} against ρ {fl([x['rest_rho'] for x in c['per_seed']])}; R ≥ 0.9 ρ in {c['ok']}/{c['n_seeds']} → {c['verdict']}.")
    p5 = V["P5"]
    if "rt_median" in p5:
        md.append(f"\n**P5** (held-out binding stratum, n {p5['n']}, masked): RT-canon's seed-median {f(p5['rt_median'])}, EI16's {f(p5['ei_median'])} (difference {p5['diff']:+.2f}), greedy rung {f(p5['greedy'])}; unmasked {f(p5['rt_median_u'])} / {f(p5['ei_median_u'])}"
                  + (f"; on the {p5['comb']['n']} that need the unseen combination {f(p5['comb']['rt_median'])} / {f(p5['comb']['ei_median'])}" if p5.get("comb") else "") + f" → {p5['verdict']}.")
    p6 = V["P6"]
    if "post" in p6:
        md.append(f"\n**P6** (first decision, multi-class training instances, n {p6['post'][0]['n']}): RT-post's TV to the target {fl([x['tv'] for x in p6['post']])} against the uniform-over-shortest-actions reference {f(p6['post'][0]['tv_uniform'])} (below it in {p6['post_ok']}/{len(p6['post'])}); "
                  f"RT-canon's canonical-first share {fl([x['canon'] for x in p6['canon']])} (≥ 0.9 in {p6['canon_ok']}/{len(p6['canon'])}); RT-canon's TV {fl([x['tv'] for x in p6['canon']])} → {p6['verdict']}.")
    p7 = V["P7"]
    if "ceiling" in p7:
        md.append(f"\n**P7** (unmasked): training binding stratum (n {p7['train_n']}) — RT-examples {fl(p7['examples_train_u'])} against ceiling {f(p7['ceiling'])} × RT-canon's median {f(p7['rt_train_median_u'])} = {f(p7['target'])} (± 0.1) → {p7['clause1']}; "
                  f"held-out binding stratum (n {p7['heldout_n']}) — RT-examples {fl(p7['examples_heldout_u'])} against RT-canon's median {f(p7['rt_heldout_median_u'])} (below by more than 0.05 in every seed) → {p7['clause2']}.")
    p8 = V["P8"]
    if "ok_m" in p8:
        md.append(f"\n**P8** (training worlds' binding stratum, n {p8['n']}): masked — RT-canon's median {f(p8['ok_m']['rt_median'])}, RT-shuffled {fl(p8['ok_m']['shuffled'])} (gaps {fl(p8['ok_m']['gaps'])}) → {p8['ok_m']['verdict']}; "
                  f"unmasked — {f(p8['ok_u']['rt_median'])}, {fl(p8['ok_u']['shuffled'])} (gaps {fl(p8['ok_u']['gaps'])}) → {p8['ok_u']['verdict']}. A gap of more than 0.05 in every shuffled seed holds; none falsifies.")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--derived", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load_results(args.results)
    derived = json.load(open(args.derived)) if args.derived and os.path.exists(args.derived) else None
    md = report(runs, derived)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
