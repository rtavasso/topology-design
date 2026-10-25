"""Grids for the "priors" round (PREREGISTRATION.md).

Q1 (prior or search): ungrokking runs (grok on 40%, then train on a nested subset) and scratch runs on the same
    subsets, for both wirings at base width and for three widths in sym.
Q2 (does the drift direction matter): E2's sym drift jumps re-run with matched-norm random-direction jumps (I13).

    python experiments/priors/make_grids.py        # writes grids/q1u.json, q1s.json, q2.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = os.path.join(HERE, "grids")

COMMON = {"task": "modadd", "p": 59, "train_frac": 0.4, "weight_decay": 1.0, "eval_every": 100, "dense_early": True}
GRAPHS = {"sym": "pk000000", "asym": "pk012000"}
WIDTHS = {"narrow": (32, 64), "base": (128, 256), "wide": (256, 512)}  # (d, hidden): 30k / 416k / 1.6M parameters
LR0 = {"sym": 0.002, "asym": 0.001}  # as in E1-E3

# Q1 ungrokking: shrink to each fraction once test accuracy has held >= 0.95 for 2,000 steps
Q1_FRACS = [0.0125, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35]  # 44 ... 1,218 pairs, nested in
#   the 0.4 set (1,392); the low levels make a large search factor S representable without censoring (review B1)
Q1_SEEDS = [700, 701, 702, 703]
Q1_SHRINK_STEPS = {"sym": 30000, "asym": 40000}  # 60 and 40 weight-decay time constants 1/(lr*wd)
Q1_PHASE1_CAP = {"sym": 13000, "asym": 70000}  # steps allowed to reach sustained grokking before the shrink
# Q1 scratch: same subsets from initialization, best of three learning rates
Q1S_FRACS = {"sym": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], "asym": [0.25, 0.3, 0.35, 0.4]}
Q1S_LR_MULT = [1.0, 2.0, 4.0]  # grokking time scales ~1/lr, so the search is extended upward (review B3); sym 8e-3 is
#   inside the all-parallel graph's stable range (E4: stable to 0.016)
Q1S_SEEDS = [700, 701, 702]
Q1S_STEPS = {"sym": 60000, "asym": 90000}
# Q2: E2's sym jump cells and seeds, drift vs matched-norm random direction, plus the paired base runs
Q2_SEEDS = list(range(610, 620))
Q2_AT, Q2_H = [1000, 2000], [250, 1000]
Q2_STEPS = 15000
Q2_DIRS = ["drift", "radial", "random", "rescale"]  # radial = the primary null (review B2)


def sub(base, vary):
    return {"base": base, "vary": vary, "tie_data_seed": True}


def q1u() -> dict:
    grids = []
    for gname, arm in GRAPHS.items():
        widths = WIDTHS if gname == "sym" else {"base": WIDTHS["base"]}
        for wname, (d, h) in widths.items():
            grids.append(sub({**COMMON, "arm": arm, "d": d, "hidden": h, "lr": LR0[gname],
                              "steps": Q1_PHASE1_CAP[gname] + 2000 + Q1_SHRINK_STEPS[gname],
                              "shrink_after_grok": 2000, "shrink_steps": Q1_SHRINK_STEPS[gname],
                              "shrink_by": Q1_PHASE1_CAP[gname] + 2000,
                              "tag": f"pQ1u-{gname}-{wname}"},
                             {"shrink_frac": Q1_FRACS, "model_seed": Q1_SEEDS}))
    return {"_note": "Q1 ungrokking: grok on the 40% training set, then train only on a nested subset; does the rule "
                     "survive? Paired with Q1 scratch runs on the same subsets.", "grids": grids}


def q1s() -> dict:
    grids = []
    for gname, arm in GRAPHS.items():
        widths = WIDTHS if gname == "sym" else {"base": WIDTHS["base"]}
        for wname, (d, h) in widths.items():
            for frac in Q1S_FRACS[gname]:
                grids.append(sub({**COMMON, "arm": arm, "d": d, "hidden": h, "train_frac": frac,
                                  "steps": Q1S_STEPS[gname], "stop_after_grok": 5000,
                                  "tag": f"pQ1s-{gname}-{wname}"},
                                 {"lr": [round(LR0[gname] * m, 6) for m in Q1S_LR_MULT], "model_seed": Q1S_SEEDS}))
    return {"_note": "Q1 scratch: data floor on the same nested subsets, best of three learning rates.", "grids": grids}


def q2() -> dict:
    base = {**COMMON, "arm": GRAPHS["sym"], "d": 128, "hidden": 256, "lr": LR0["sym"], "steps": Q2_STEPS,
            "stop_after_grok": 0, "measure_horizon": True, "measure_refit": False, "dense_until": 6000}
    return {"_note": "Q2 (I13): E2's sym drift jumps vs null jumps of the same norm per tensor (radial-matched, "
                     "random, rescale-only), paired by seed with base runs.",
            "grids": [sub({**base, "tag": "pQ2base"}, {"model_seed": Q2_SEEDS}),
                      sub({**base, "tag": "pQ2jump"}, {"drift_jump_at": Q2_AT, "drift_jump_h": Q2_H,
                                                       "drift_jump_dir": Q2_DIRS,
                                                       "model_seed": Q2_SEEDS})]}


GRID_FNS = {"q1u": q1u, "q1s": q1s, "q2": q2}
EXPECTED = {"q1u": (3 + 1) * len(Q1_FRACS) * len(Q1_SEEDS),
            "q1s": (3 * len(Q1S_FRACS["sym"]) + len(Q1S_FRACS["asym"])) * len(Q1S_LR_MULT) * len(Q1S_SEEDS),
            "q2": len(Q2_SEEDS) * (1 + len(Q2_AT) * len(Q2_H) * len(Q2_DIRS))}


def main():
    os.makedirs(GRIDS, exist_ok=True)
    for name, fn in GRID_FNS.items():
        path = os.path.join(GRIDS, name + ".json")
        with open(path, "w") as f:
            json.dump(fn(), f, indent=1)
        print(path, EXPECTED[name], "runs")


if __name__ == "__main__":
    main()
