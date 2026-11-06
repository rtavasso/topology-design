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
import os
import statistics

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

    # P2
    md.append("\n## P2 (Q1): held-out pairs' binding stratum, eval5 — RT-canon against the greedy rung; RT-all's ceiling")
    rows = []
    for a in ("RT-canon", "RT-post", "RT-all", "EI16", "RT-examples", "RT-shuffled", "PF", "RT-absolute"):
        if a not in arms:
            continue
        for cls, worlds in (("pair", PAIRS_HELD), ("triple+full", TRIPLES + FULL)):
            per = []
            for r in by_seed(runs, a):
                sel = instances(r, "eval5", world=worlds, binding=True)
                s_, g_ = rate(sel), ladder_rate(sel, 1)
                per.append((s_, g_, len(sel)))
            above02 = sum(1 for s_, g_, _ in per if s_ is not None and s_ - g_ >= 0.2)
            above0 = sum(1 for s_, g_, _ in per if s_ is not None and s_ > g_)
            ge07 = sum(1 for s_, g_, _ in per if s_ is not None and s_ >= 0.7)
            rows.append([a, cls, per[0][2] if per else 0, summarize_seeds([p[0] for p in per]), fmt(per[0][1] if per else None),
                         f"{above0}/{len(per)}", f"{above02}/{len(per)}", f"{ge07}/{len(per)}"])
    md.append(table(rows, ["arm", "worlds", "n/seed", "success (median [min,max])", "greedy", "> greedy", "≥ greedy+0.2", "≥ 0.7"]))
    md.append("P2 passes when RT-canon on the pairs is ≥ greedy + 0.2 and ≥ 0.7 in ≥ 6/10 seeds, > greedy on triples+full in ≥ 6/10, and RT-all ≥ 0.9 there.")

    # P3
    md.append("\n## P3 (Q2): sizes 6–7 (eval7, k ≤ 4) against size ≤ 5 at k ≤ 4 (eval5), masked; RT-absolute against the unaddressable share")
    rows = []
    un = (derived or {}).get("eval7_unaddressable", {})
    for a in ("RT-canon", "RT-absolute", "RT-post", "RT-all"):
        if a not in arms:
            continue
        ratios, abs_rows = [], []
        for r in by_seed(runs, a):
            s5 = rate(instances(r, "eval5", k=[1, 2, 3, 4]))
            s7 = rate(instances(r, "eval7"))
            ratios.append(s7 / s5 if s5 else None)
            if un:
                e7 = instances(r, "eval7")
                flag = [un.get(pi["world"], {}).get(f"{pi['t']}|{pi['s']}") for pi in e7]
                addr = [pi for pi, f in zip(e7, flag) if f is False]
                unad = [pi for pi, f in zip(e7, flag) if f is True]
                abs_rows.append((rate(addr), rate(unad), len(addr), len(unad)))
        n_ok = sum(1 for x in ratios if x is not None and x >= 0.9)
        extra = ""
        if abs_rows:
            extra = f"addressable: {summarize_seeds([x[0] for x in abs_rows])} (n {abs_rows[0][2]}); unaddressable: {summarize_seeds([x[1] for x in abs_rows])} (n {abs_rows[0][3]})"
        rows.append([a, summarize_seeds([rate(instances(r, 'eval5', k=[1, 2, 3, 4])) for r in by_seed(runs, a)]),
                     summarize_seeds([rate(instances(r, 'eval7')) for r in by_seed(runs, a)]), summarize_seeds(ratios), f"{n_ok}/{len(ratios)}", extra])
    md.append(table(rows, ["arm", "size ≤ 5, k ≤ 4", "sizes 6–7", "ratio", "ratio ≥ 0.9", "split by addressability"]))
    md.append("P3 passes when the structural RT-canon's ratio is ≥ 0.9 in ≥ 7/10 seeds, and RT-absolute's failures are the unaddressable instances (success on addressable ≈ size-5 success, ≈ 0 on unaddressable).")

    # P4
    md.append("\n## P4 (Q3): training worlds, GP1 instances at k ≥ 4 (eval5, masked) against the lookahead rungs on the same instances")
    rows = []
    for a in arms:
        per = []
        for r in by_seed(runs, a):
            sel = instances(r, "eval5", world=TRAIN_WORLDS, GP1=True, k=[4, 5, 6])
            per.append((rate(sel), ladder_rate(sel, 1), ladder_rate(sel, 2), ladder_rate(sel, 3), len(sel)))
        between = sum(1 for s_, d1, d2, d3, _ in per if s_ is not None and d2 <= s_ < d3)
        ge3 = sum(1 for s_, d1, d2, d3, _ in per if s_ is not None and s_ >= d3)
        le1 = sum(1 for s_, d1, d2, d3, _ in per if s_ is not None and s_ <= d1)
        rows.append([a, per[0][4] if per else 0, summarize_seeds([p[0] for p in per]), fmt(per[0][1] if per else None), fmt(per[0][2] if per else None), fmt(per[0][3] if per else None),
                     f"{between}/{len(per)}", f"{ge3}/{len(per)}", f"{le1}/{len(per)}"])
    md.append(table(rows, ["arm", "n/seed", "success", "d=1", "d=2", "d=3", "between d=2 and d=3", "≥ d=3", "≤ d=1"]))
    # per k and GP1 for RT-canon
    if "RT-canon" in arms:
        rows = []
        for k in (2, 3, 4, 5, 6):
            for g in (0, 1):
                per = [instances(r, "eval5", world=TRAIN_WORLDS, GP1=bool(g), k=k) for r in by_seed(runs, "RT-canon")]
                if not per or not per[0]:
                    continue
                rows.append([k, g, len(per[0]), summarize_seeds([rate(p) for p in per]), fmt(ladder_rate(per[0], 1)), fmt(ladder_rate(per[0], 2)), fmt(ladder_rate(per[0], 3))])
        md.append("\nRT-canon by k × GP1 (training worlds, eval5):\n\n" + table(rows, ["k", "GP1", "n/seed", "RT-canon", "d=1", "d=2", "d=3"]))

    # P5
    if "EI16" in arms and "RT-canon" in arms:
        sel_rt = [rate(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "RT-canon")]
        sel_ei = [rate(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "EI16")]
        cis = [bootstrap_ci(instances(r, "eval5", world=PAIRS_HELD + TRIPLES + FULL, binding=True)) for r in by_seed(runs, "EI16")]
        md.append(f"\n## P5 (Q5): held-out binding stratum — RT-canon median {fmt(statistics.median([v for v in sel_rt if v is not None]) if any(v is not None for v in sel_rt) else None)}, "
                  f"EI16 {summarize_seeds(sel_ei)}; EI16 bootstrap 95 % CIs per seed: {[(fmt(a), fmt(b)) for a, b in cis]}; EI16 found routes: "
                  f"{[r.get('ei_found') for r in by_seed(runs, 'EI16')]}")

    # P6
    if "RT-post" in arms or "RT-canon" in arms:
        rows = []
        for a in ("RT-canon", "RT-post"):
            if a not in arms:
                continue
            for r in by_seed(runs, a):
                sel = [pi for pi in instances(r, "eval5", world=TRAIN_WORLDS) if pi.get("p_first") and pi["n_classes"] > 1 and len(pi["p_first"]) > 1]
                if not sel:
                    continue
                mass = statistics.mean(sum(pi["p_first"].values()) for pi in sel)
                # share of the shortest-path mass on the single best action (1 = committed to one choice)
                conc = statistics.mean(max(pi["p_first"].values()) / max(1e-12, sum(pi["p_first"].values())) for pi in sel)
                rows.append([r["cfg"]["name"], len(sel), fmt(mass), fmt(conc)])
        md.append("\n## P6 (Q6): at multi-class training instances, the router's mass on shortest-path first actions and its concentration on one of them (RT-canon should commit; RT-post should spread toward the target mixture)\n\n" +
                  table(rows, ["run", "n", "mass on shortest-path actions", "concentration on best"]))

    # P7
    if "RT-examples" in arms:
        exc = (derived or {}).get("examples_ceiling", {})
        rows = []
        for r in by_seed(runs, "RT-examples"):
            for cls, worlds in (("train", TRAIN_WORLDS), ("pair", PAIRS_HELD), ("triple+full", TRIPLES + FULL)):
                sel = instances(r, "eval5", world=worlds, binding=True)
                ceil = statistics.mean([exc[w]["ceiling_canon_binding"] for w in worlds if w in exc and exc[w]["binding_n"] > 0]) if exc else None
                rt = [rate(instances(x, "eval5", world=worlds, binding=True)) for x in by_seed(runs, "RT-canon")]
                rows.append([r["cfg"]["name"], cls, len(sel), fmt(rate(sel)), fmt(ceil), summarize_seeds(rt)])
        md.append("\n## P7 (Q7): RT-examples on the binding stratum against its derived ceiling and stated-axiom RT-canon\n\n" + table(rows, ["run", "class", "n", "RT-examples", "derived ceiling", "RT-canon"]))

    # P8
    if "RT-shuffled" in arms:
        cb = (derived or {}).get("cue_blind", {})
        rows = []
        for w in CLASS:
            for r in by_seed(runs, "RT-shuffled"):
                sel = instances(r, "eval5", world=w)
                selb = instances(r, "eval5", world=w, binding=True)
                rt = [rate(instances(x, "eval5", world=w, binding=True)) for x in by_seed(runs, "RT-canon")]
                rows.append([w, r["cfg"]["seed"], len(sel), fmt(rate(sel)), fmt(cb.get(w, {}).get("ceiling")), fmt(rate(selb)), summarize_seeds(rt)])
        md.append("\n## P8 (null): RT-shuffled per world against the cue-blind ceiling; binding stratum against RT-canon\n\n" + table(rows, ["world", "seed", "n", "shuffled (all)", "cue-blind ceiling", "shuffled (binding)", "RT-canon (binding)"]))

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
            rows.append([a, len(per[0]), summarize_seeds([rate(p) for p in per]), fmt(ladder_rate(per[0], 1)), fmt(ladder_rate(per[0], 2)), fmt(ladder_rate(per[0], 3))])
    if rows:
        md.append("\n## P1 (reported): {A, C} at k = 7–8, size ≤ 5\n\n" + table(rows, ["arm", "n/seed", "success", "d=1", "d=2", "d=3"]))
    return "\n".join(md) + "\n"


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
