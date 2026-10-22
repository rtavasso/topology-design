"""Grids for "diagnose the walk" (PREREGISTRATION.md): E1 information gap, E2 visibility, E3 credit by speed,
E4 stability regions. Run with experiments/fleet/run_fleet.py, locally or via experiments/fleet/kaggle/launch.py.

    python experiments/walk/make_grids.py        # writes grids/e1.json ... grids/e4.json
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from topology_design import grokking as gk  # noqa: E402

GRIDS = os.path.join(HERE, "grids")
BR = [[1, 2, 3], [4, 5, 6]]
COMMON = {"task": "modadd", "p": 59, "train_frac": 0.4, "d": 128, "hidden": 256, "weight_decay": 1.0,
          "eval_every": 100, "dense_early": True}
GRAPHS6 = {"asym": "pk012000", "sym": "pk000000"}

# E1 (measurement): streams start at ~10% of the training set (140 pairs), +32 every b steps (40 increments).
#   No training after the last arrival: only arrival-time predictions are scored.
E1_START, E1_N = 140, 32
E1_BUDGETS = [400, 1600]  # steps between increments
E1_SEEDS = list(range(600, 605))
E1_LRS = [0.001, 0.002]
#   Data floor (fixed dataset, ample steps): smallest training fraction at which the network groks.
E1_FRACS = [0.15, 0.2, 0.25, 0.3, 0.4]
E1_FLOOR_LR = {"sym": 0.002, "asym": 0.001}
E1_FLOOR_SEEDS = [600, 601, 602]
# E2: visibility (base runs, fixed horizons, no early stop) and drift jumps (paired by seed)
E2_SEEDS = list(range(610, 620))
E2_STEPS = {"sym": 15000, "asym": 40000}
E2_BASE_LR = {"sym": 0.002, "asym": 0.001}
E2_JUMPS = {"sym": ([1000, 2000], [250, 1000]), "asym": ([4000, 8000], [250, 1000])}  # (jump steps, horizons H)
E2_NOISE = {"sym": 1000, "asym": 4000}  # H = 0 jump arms at this step: the batch-to-batch noise floor
# E3: per-branch speed ratio rho at fixed lr*wd, plus lr-only and wd-only controls at the extremes
E3_RHOS = [0.25, 0.5, 1.0, 2.0, 4.0]
E3_LR0 = {"sym": 0.002, "asym": 0.001}
E3_STEPS = {"sym": 30000, "asym": 50000}
E3_SEEDS = list(range(620, 630))
E3_CONTROL_RHOS = [0.25, 4.0]
# E4: stability regions (fixed 12k horizon; stability scored on a 6,000-step window after first fit, as in Study 2)
E4_ARMS = {"pk0000": 1, "pk0100": 2, "pk0120": 3, "pk0123": 4}  # arm -> longest chain C
E4_LRS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064]
E4_WDS = [0.3, 1.0, 3.0]
E4_SEEDS = [630, 631, 632]
E4_STEPS = 12000


def sub(base, vary):
    return {"base": base, "vary": vary, "tie_data_seed": True}


def block_depths(arm: str) -> list[int]:
    """Chain depth of each block: 1 + depth of the state it reads (state h_k holds blocks 1..k)."""
    parents = gk.arm_spec(arm)[0]
    state_depth = [0]
    depths = []
    for par in parents:
        d = 1 + state_depth[par]
        depths.append(d)
        state_depth.append(max(state_depth[-1], d))
    return depths


def e1() -> dict:
    grids = []
    K = math.ceil((1392 - E1_START) / E1_N)
    for budget in E1_BUDGETS:
        for name, arm in GRAPHS6.items():
            grids.append(sub({**COMMON, "arm": arm, "branches": BR, "steps": K * budget, "stop_after_grok": 0,
                              "arrive_start": E1_START, "arrive_n": E1_N, "arrive_every": budget,
                              "tag": f"wE1-{name}-b{budget}"}, {"lr": E1_LRS, "model_seed": E1_SEEDS}))
    for name, arm in GRAPHS6.items():
        for frac in E1_FRACS:
            grids.append(sub({**COMMON, "arm": arm, "branches": BR, "train_frac": frac, "lr": E1_FLOOR_LR[name],
                              "steps": 60000, "stop_after_grok": 3000, "tag": f"wE1floor-{name}"},
                             {"model_seed": E1_FLOOR_SEEDS}))
    return {"_note": "E1 information gap (measurement): prequential code length of gradient descent on streams vs a "
                     "program-prior learner (computed in analysis), and the data floor on fixed datasets.",
            "grids": grids}


def e2() -> dict:
    grids = []
    for name, arm in GRAPHS6.items():
        base = {**COMMON, "arm": arm, "branches": BR, "lr": E2_BASE_LR[name], "steps": E2_STEPS[name],
                "stop_after_grok": 0, "measure_horizon": True}
        grids.append(sub({**base, "tag": f"wE2-{name}"}, {"model_seed": E2_SEEDS}))
        ats, hs = E2_JUMPS[name]
        grids.append(sub({**base, "measure_refit": False, "tag": f"wE2jump-{name}"},
                         {"drift_jump_at": ats, "drift_jump_h": hs, "model_seed": E2_SEEDS}))
        grids.append(sub({**base, "measure_refit": False, "drift_jump_at": E2_NOISE[name], "drift_jump_h": 0,
                          "tag": f"wE2noise-{name}"}, {"model_seed": E2_SEEDS}))
    return {"_note": "E2 visibility: logistic readout refit and drift extrapolation at evals (base runs); drift-jump "
                     "interventions and H = 0 noise-floor arms, paired with the base run of the same seed.",
            "grids": grids}


def e3() -> dict:
    grids = []
    for name, arm in GRAPHS6.items():
        base = {**COMMON, "arm": arm, "branches": BR, "lr": E3_LR0[name], "steps": E3_STEPS[name], "stop_after_grok": 0}
        for rho in E3_RHOS:
            r = math.sqrt(rho)
            grids.append(sub({**base, "branch_lr_mult": [r, 1 / r], "branch_wd_mult": [1 / r, r],
                              "tag": f"wE3-{name}-rho{rho:g}"}, {"model_seed": E3_SEEDS}))
        for rho in E3_CONTROL_RHOS:
            r = math.sqrt(rho)
            grids.append(sub({**base, "branch_lr_mult": [r, 1 / r], "tag": f"wE3lr-{name}-rho{rho:g}"},
                             {"model_seed": E3_SEEDS}))
            grids.append(sub({**base, "branch_wd_mult": [1 / r, r], "tag": f"wE3wd-{name}-rho{rho:g}"},
                             {"model_seed": E3_SEEDS}))
    return {"_note": "E3 credit by speed: branch 0 steps rho times faster than branch 1 at equal lr*wd; lr-only and "
                     "wd-only controls separate step size from weight scale.", "grids": grids}


def e4() -> dict:
    grids = []
    for arm in E4_ARMS:
        dep = block_depths(arm)
        params = {"std": [1.0] * len(dep), "P1": [1 / math.sqrt(len(dep))] * len(dep),
                  "P2": [round(1 / math.sqrt(d), 6) for d in dep]}
        for pname, alphas in params.items():
            grids.append(sub({**COMMON, "arm": arm, "alphas": alphas, "steps": E4_STEPS, "stop_after_grok": 0,
                              "tag": f"wE4-{pname}"}, {"lr": E4_LRS, "weight_decay": E4_WDS, "model_seed": E4_SEEDS}))
    return {"_note": "E4 stability regions: lr x wd per structure under three parameterizations (std; P1 uniform "
                     "1/sqrt(L) block multiplier, a control; P2 per-block 1/sqrt(chain depth)).", "grids": grids}


def main():
    os.makedirs(GRIDS, exist_ok=True)
    for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
        path = os.path.join(GRIDS, name + ".json")
        with open(path, "w") as f:
            json.dump(fn(), f, indent=1)
        print(path)


if __name__ == "__main__":
    main()
