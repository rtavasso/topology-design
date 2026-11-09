"""The rung-4 pilot's readings (experiments/rewrite/PILOT.md): lr per arm, steps for the main grid, throughput, harness sanity.

    PYTHONPATH=src python3 experiments/rewrite/pilot_read.py --results results/kaggle_rewrite-pilot/results \
        --out experiments/rewrite/results_data/pilot_report.md [--compact experiments/rewrite/results_data/pilot]

It reads what PILOT.md says the pilot sets and nothing else (L30): the choices use each arm's own training-world rows of
eval5 only (RT-all's training worlds are all fifteen; its eight-world reading is printed beside), no held-out world, no
binding stratum, no sizes 6–7, no arm-against-arm comparison. The curve's intermediate points are on the first 100
instances per world and its last point on the full 600 (DESIGN §12a.7), so the last point is recomputed on the same 100
from the per-instance records before a curve is called rising or flat. --compact writes the run records without their
per-instance lists (cells, curves, timings; small enough for git)."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics

from analyze import TRAIN_WORLDS, arm_of, load_results

ALL_WORLDS = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
ARM_ORDER = ["RT-canon", "RT-post", "RT-all", "RT-absolute", "EI16", "PF", "RT-examples", "RT-shuffled"]
MAIN_SEEDS = {"RT-canon": 10, "RT-post": 10, "RT-all": 3, "RT-absolute": 5, "EI16": 5, "PF": 3, "RT-examples": 3, "RT-shuffled": 2}


def own_worlds(r):
    return ALL_WORLDS if arm_of(r) == "RT-all" else TRAIN_WORLDS


def cell_rate(cells, worlds, key="ok_m"):
    n = sum(c["n"] for k, c in cells.items() if k.split("|")[0] in worlds)
    return (sum(c[key] for k, c in cells.items() if k.split("|")[0] in worlds) / n if n else None), n


def first_n_rate(r, tag, worlds, n_per_world, key="ok_m"):
    """The final outcomes restricted to the first n_per_world instances of each world (the curve's subset)."""
    seen, ok, n = {}, 0, 0
    for pi in r["final"][tag]["per_instance"]:
        w = pi["world"]
        seen[w] = seen.get(w, 0) + 1
        if seen[w] <= n_per_world and w in worlds:
            ok += bool(pi[key]); n += 1
    return (ok / n if n else None), n


def subset_matches(r, tag="eval5"):
    """Is the first-100-per-world subset of the per-instance records the set the curve's intermediate points used?
    Compared through the cell sizes of the last intermediate point."""
    npw = int(r["cfg"].get("curve_per_world", 100))
    seen, cells = {}, {}
    for pi in r["final"][tag]["per_instance"]:
        w = pi["world"]
        seen[w] = seen.get(w, 0) + 1
        if seen[w] <= npw:
            key = f"{w}|{int(bool(pi['binding']))}|{int(bool(pi['GP1']))}|{pi['k']}"
            cells[key] = cells.get(key, 0) + 1
    ref = {k: c["n"] for k, c in r["curve"][-2][tag].items()}
    return cells == ref


def curve_on_subset(r, worlds, tag="eval5", key="ok_m"):
    npw = int(r["cfg"].get("curve_per_world", 100))
    pts = [(p["step"], cell_rate(p[tag], worlds, key)[0]) for p in r["curve"][:-1]]
    pts.append((r["curve"][-1]["step"], first_n_rate(r, tag, worlds, npw, key)[0]))
    return pts


def timing(r):
    """Seconds per training step and per evaluation, from the curve's cumulative times: an intermediate interval is
    eval_every steps plus one curve evaluation (100 per world), the last one eval_every steps plus the full evaluation
    (600 per world). With the full evaluation taken as 6 curve evaluations, the two give both unknowns."""
    c = r["curve"]
    every = c[1]["step"] - c[0]["step"]
    mids = [c[i]["time"] - c[i - 1]["time"] for i in range(1, len(c) - 1)]
    mid = statistics.median(mids)
    last = c[-1]["time"] - c[-2]["time"]
    ce = max(0.0, (last - mid) / 5.0)
    step = max(1e-9, (mid - ce) / every)
    return {"step_s": step, "curve_eval_s": ce, "full_eval_s": 6 * ce, "start_s": c[0]["time"] - ce, "total_s": r["seconds"]}


def project(t, steps, every):
    return t["start_s"] + steps * t["step_s"] + (steps // every) * t["curve_eval_s"] + t["full_eval_s"]


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def table(rows, head):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def report(runs, main_steps=12000, main_every=250):
    md = [f"# Rung 4 pilot: the readings PILOT.md names ({len(runs)} runs, seed {sorted({r['cfg']['seed'] for r in runs})}, "
          f"{sorted({r['cfg']['steps'] for r in runs})} steps)\n",
          "Read here: lr per arm, steps, throughput, harness sanity — on each arm's own training-world rows of eval5. "
          "Not read: held-out worlds, the binding stratum, sizes 6–7, k = 7–8, any arm against another (L30).\n"]
    by = {}
    for r in runs:
        by.setdefault(arm_of(r), []).append(r)
    arms = [a for a in ARM_ORDER if a in by] + sorted(a for a in by if a not in ARM_ORDER)
    for a in arms:
        by[a].sort(key=lambda r: r["cfg"]["lr"])

    # 0. the curve subset
    ok_subset = all(subset_matches(r) for r in runs)
    md.append(f"Curve subset check (first 100 per world of the per-instance records = the intermediate points' set): **{'holds' if ok_subset else 'FAILS'}** on {len(runs)} runs.\n")

    # 1. lr per arm
    md.append("## 1. lr per arm: training-world eval5 masked success at the last step (full 600 per world); ties by the lower loss\n")
    rows, chosen = [], {}
    for a in arms:
        best = None
        for r in by[a]:
            w = own_worlds(r)
            final_m, n = cell_rate(r["final"]["eval5"]["cells"], w)
            final_u, _ = cell_rate(r["final"]["eval5"]["cells"], w, "ok_u")
            eight, n8 = cell_rate(r["final"]["eval5"]["cells"], TRAIN_WORLDS)
            loss = r["curve"][-1]["loss"]
            pts = curve_on_subset(r, w)
            rows.append([a, f"{r['cfg']['lr']:g}", n, " ".join(fmt(p, 2) for _, p in pts), fmt(final_m), fmt(final_u), fmt(eight) if a == "RT-all" else "", fmt(loss)])
            key = (round(final_m, 3), -loss)
            if best is None or key > best[0]:
                best = (key, r["cfg"]["lr"])
        chosen[a] = best[1]
    for row in rows:
        row.append("**chosen**" if float(row[1]) == chosen[row[0]] else "")
    md.append(table(rows, ["arm", "lr", "n", "curve on the first 100 per world (steps " + ", ".join(str(p["step"]) for p in runs[0]["curve"]) + ")", "final masked", "final unmasked", "RT-all on the eight", "loss (last interval)", ""]))
    md.append("\nChosen: " + ", ".join(f"{a} {chosen[a]:g}" for a in arms) + ".\n")

    # 2. steps
    md.append("## 2. Steps: is any arm's training-world curve still rising at the last step (at its chosen lr)?\n")
    rows, rising = [], []
    for a in arms:
        r = [x for x in by[a] if x["cfg"]["lr"] == chosen[a]][0]
        pts = curve_on_subset(r, own_worlds(r))
        n = cell_rate(r["curve"][-2]["eval5"], own_worlds(r))[1]
        se = math.sqrt(max(1e-9, pts[-1][1] * (1 - pts[-1][1])) / n)
        g1, g2 = pts[-1][1] - pts[-2][1], pts[-1][1] - pts[-3][1]
        l = [p["loss"] for p in r["curve"]]
        dl = (l[-1] - l[-2]) / l[-2]
        up = (g2 > 2 * se) or (dl < -0.05)
        rising.append(up)
        rows.append([a, f"{chosen[a]:g}", n, fmt(pts[-3][1]), fmt(pts[-2][1]), fmt(pts[-1][1]), f"{g1:+.3f}", f"{g2:+.3f}", fmt(se), f"{100 * dl:+.1f} %", "rising" if up else "flat"])
    steps_pts = [p["step"] for p in runs[0]["curve"]]
    md.append(table(rows, ["arm", "lr", "n", f"step {steps_pts[-3]}", f"step {steps_pts[-2]}", f"step {steps_pts[-1]}", "gain, last interval", "gain, last two", "binomial s.e.", "loss change, last interval", "reading"]))
    md.append(f"\n\"Rising\" = the gain over the last two intervals exceeds two binomial standard errors, or the training loss still fell by more than 5 % over the last interval. "
              f"Rising: {sum(rising)} of {len(rising)} arms → **{'12,000 steps' if any(rising) else 'plateau rule applies'}** for the main grid (PILOT.md item 2).\n")

    # 3. throughput
    md.append(f"## 3. Throughput, and the main grid at {main_steps:,} steps with curves every {main_every}\n")
    rows, tot = [], 0.0
    cost = {}
    for a in arms:
        ts = [timing(r) for r in by[a]]
        t = {k: statistics.median(x[k] for x in ts) for k in ts[0]}
        proj = project(t, main_steps, main_every)
        cost[a] = proj
        tot += proj * MAIN_SEEDS.get(a, 0)
        rows.append([a, f"{1000 * t['step_s']:.0f} ms", f"{t['curve_eval_s']:.1f} s", f"{t['full_eval_s']:.0f} s", f"{t['total_s']:.0f} s", f"{proj / 60:.0f} min", MAIN_SEEDS.get(a, 0), f"{proj * MAIN_SEEDS.get(a, 0) / 3600:.1f} h"])
    md.append(table(rows, ["arm", "per training step", "curve evaluation", "full evaluation", "pilot run (median of 3)", "projected main run", "seeds", "arm total"]))
    base = cost.get("RT-canon", 1.0)
    md.append(f"\nSum over the 41 runs: **{tot / 3600:.1f} process-hours** → {tot / 3600 / 2:.1f} h wall on two shards (one per T4), {tot / 3600 / 4:.1f} h on four (two per T4) if the four processes do not slow one another. "
              "Relative cost per run (RT-canon = 1): " + ", ".join(f"{a} {cost[a] / base:.1f}" for a in arms) + ".\n")
    ei = [r for r in by.get("EI16", [])]
    if ei:
        rows = []
        for r in ei:
            c = r["curve"]
            cells = []
            for i in range(1, len(c)):
                f0, t0, r0 = c[i - 1].get("ei_found", [0, 0, 0])
                f1, t1, r1 = c[i]["ei_found"]
                cells.append(f"{(f1 - f0) / max(1, t1 - t0):.2f} ({(r1 - r0) / (c[i]['step'] - c[i - 1]['step']):.0f})")
            rows.append([f"{r['cfg']['lr']:g}"] + cells)
        md.append("EI16's beam: share of instances with a route found, per interval (rows imitated per step in brackets; RT trains on 256):\n\n" +
                  table(rows, ["lr"] + [f"to step {p['step']}" for p in ei[0]["curve"][1:]]) + "\n")

    # 4. harness sanity
    md.append("## 4. Harness sanity\n")
    rows = []
    for a in arms:
        viol = nan = 0
        cats_m, cats_u = set(), set()
        n_pi, n_curve, has_ladder = set(), set(), True
        gap = []
        for r in by[a]:
            nan += sum(1 for x in r["losses"] if x != x) + sum(1 for p in r["curve"] if p["loss"] is not None and p["loss"] != p["loss"])
            n_curve.add(len(r["curve"]))
            for tag, rep in r["final"].items():
                n_pi.add((tag, len(rep["per_instance"])))
                for c in rep["cells"].values():
                    viol += c["ok_u"] > c["ok_m"]
                    cats_m |= set(c["cat_m"]); cats_u |= set(c["cat_u"])
                has_ladder &= all("ladder" in pi for pi in rep["per_instance"][:50])
            m, _ = cell_rate(r["final"]["eval5"]["cells"], own_worlds(r)); u, _ = cell_rate(r["final"]["eval5"]["cells"], own_worlds(r), "ok_u")
            gap.append(f"{m - u:+.3f}")
        rows.append([a, viol, nan, ", ".join(sorted(cats_m)), ", ".join(sorted(cats_u)), "; ".join(f"{t} {n}" for t, n in sorted(n_pi)), sorted(n_curve), "yes" if has_ladder else "NO", " ".join(gap)])
    md.append(table(rows, ["arm", "cells with unmasked > masked", "NaN losses", "masked categories", "unmasked categories", "per-instance records", "curve points", "ladder stored", "masked − unmasked on its training worlds (per lr)"]))
    return "\n".join(md) + "\n", chosen, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compact", default=None, help="directory for the run records without their per-instance lists")
    args = ap.parse_args()
    runs = load_results(args.results)
    md, chosen, cost = report(runs)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(md)
    if args.compact:
        os.makedirs(args.compact, exist_ok=True)
        for r in runs:
            c = {**r, "final": {tag: {"cells": rep["cells"], "n_per_instance": len(rep["per_instance"])} for tag, rep in r["final"].items()}}
            with gzip.open(os.path.join(args.compact, r["cfg"]["name"] + ".compact.json.gz"), "wt") as f:
                json.dump(c, f)
    print(md)
    print("LR =", json.dumps(chosen))


if __name__ == "__main__":
    main()
