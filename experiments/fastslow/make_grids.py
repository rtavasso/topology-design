"""Grids for Phase 5 (experiments/fastslow/DESIGN.md), run with experiments/fleet/run_fleet.py (locally or on Kaggle).

    python experiments/fastslow/make_grids.py pilot
        -> grids/pilot.json (DESIGN §3 pilot + the 5B multiplier pilot)
    python experiments/fastslow/make_grids.py stage1 --base-lr 0.002 --alpha4 aL aS --alpha14 aL aS
        -> grids/stage1_5A.json, grids/stage1_5B.json (normally written by analyze_pilot.py)

Every run carries a tag naming its study and arm, so analysis never has to infer arms from settings.
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = os.path.join(HERE, "grids")
BR = [[1, 2, 3], [4, 5, 6]]  # L / A = blocks 1-3, S / B = blocks 4-6
COMMON = {"task": "modadd", "p": 59, "train_frac": 0.4, "d": 128, "hidden": 256, "weight_decay": 1.0,
          "eval_every": 100, "dense_early": True}
PILOT_LRS = [0.001, 0.002, 0.004]
PILOT_SEEDS = [490, 491, 492]
SEEDS20 = list(range(500, 520))
SEEDS10 = list(range(500, 510))
MEM4 = {"branch_wd_mult": [0.5, 2.0]}  # L remembers 4x longer than S (tau = 1/(lr wd))
MEM14 = {"branch_wd_mult": [2.0, 0.5]}
SPEED4 = {"branch_lr_mult": [2.0, 0.5], "branch_wd_mult": [0.5, 2.0]}  # L steps 4x faster, equal tau
SPEED14 = {"branch_lr_mult": [0.5, 2.0], "branch_wd_mult": [2.0, 0.5]}


def sub(base: dict, vary: dict) -> dict:
    return {"base": base, "vary": vary, "tie_data_seed": True}


def pilot() -> dict:
    base = {**COMMON, "steps": 40000, "stop_after_grok": 0, "branches": BR}
    # fixed 40k horizon (no early stop) so every run has a step-20,000 record for the multipliers
    return {"_note": "Phase 5 pilot (DESIGN §3, §5): asym/sym at three lrs for the base-lr rule and seed SDs; "
                     "unscaled memory arms of asym at the same lrs for the scale-matching multipliers.",
            "grids": [
                sub({**base, "tag": "p5pilot-base"},
                    {"arm": ["pk012000", "pk000000"], "lr": PILOT_LRS, "model_seed": PILOT_SEEDS}),
                sub({**base, "arm": "pk012000", "tag": "p5pilot-mem4", **MEM4},
                    {"lr": PILOT_LRS, "model_seed": PILOT_SEEDS}),
                sub({**base, "arm": "pk012000", "tag": "p5pilot-mem14", **MEM14},
                    {"lr": PILOT_LRS, "model_seed": PILOT_SEEDS}),
            ]}


def stage1_5A(lr: float) -> dict:
    base = {**COMMON, "lr": lr, "steps": 40000, "stop_after_grok": 5000, "measure_signal": True, "groups": 8,
            "route_tau": 0.75}
    return {"_note": "Study 5A (DESIGN §4): measurement only; asym/sym per branch, Study-4 graphs per block.",
            "grids": [
                sub({**base, "branches": BR, "tag": "p5A"}, {"arm": ["pk012000", "pk000000"], "model_seed": SEEDS10}),
                sub({**base, "tag": "p5A"}, {"arm": ["pk0000", "pk0120", "pk0123"], "model_seed": SEEDS10}),
            ]}


def stage1_5B(lr: float, alpha4: list[float], alpha14: list[float]) -> dict:
    base = {**COMMON, "lr": lr, "steps": 50000, "stop_after_grok": 0, "branches": BR, "measure_path_eff": True}
    a = {**base, "arm": "pk012000"}
    s = {**base, "arm": "pk000000"}
    arms = [  # (graph base, arm name, settings, seeds)
        (a, "base", {}, SEEDS20), (a, "mem4", MEM4, SEEDS20), (a, "mem14", MEM14, SEEDS20),
        (a, "mem4s", {**MEM4, "branch_alpha": alpha4}, SEEDS20),
        (a, "mem14s", {**MEM14, "branch_alpha": alpha14}, SEEDS20),
        (a, "speed4", SPEED4, SEEDS10), (a, "speed14", SPEED14, SEEDS10),
        (a, "unmaskedL", {"unmask_branch": 0}, SEEDS10),
        (s, "base", {}, SEEDS10), (s, "mem4", MEM4, SEEDS10), (s, "mem14", MEM14, SEEDS10),
        (s, "speed4", SPEED4, SEEDS10), (s, "speed14", SPEED14, SEEDS10),
    ]
    grids = []
    for g, name, extra, seeds in arms:
        graph = "asym" if g["arm"] == "pk012000" else "sym"
        grids.append(sub({**g, **extra, "tag": f"p5B-{graph}-{name}"}, {"model_seed": seeds}))
    return {"_note": "Study 5B (DESIGN §5): per-branch time constants; fixed 50k horizon; "
                     f"base lr {lr}; scaled-arm multipliers from the pilot: mem4 {alpha4}, mem14 {alpha14}.",
            "grids": grids}


def write(name: str, grid: dict) -> str:
    os.makedirs(GRIDS, exist_ok=True)
    path = os.path.join(GRIDS, name + ".json")
    with open(path, "w") as f:
        json.dump(grid, f, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pilot", "stage1"])
    ap.add_argument("--base-lr", type=float)
    ap.add_argument("--alpha4", type=float, nargs=2)
    ap.add_argument("--alpha14", type=float, nargs=2)
    args = ap.parse_args()
    if args.what == "pilot":
        print(write("pilot", pilot()))
    else:
        print(write("stage1_5A", stage1_5A(args.base_lr)))
        print(write("stage1_5B", stage1_5B(args.base_lr, args.alpha4, args.alpha14)))


if __name__ == "__main__":
    main()
