"""Grids for rung 4 on the rewrite carrier (experiments/rewrite/DESIGN.md §4, §9).

    PYTHONPATH=src python3 experiments/rewrite/make_grids.py            # writes grids/pilot.json and grids/main.json

pilot: one seed per arm, a learning-rate sweep {3e-4, 1e-3, 3e-3} for every arm (AGENTS rule 4; review 2 S5), 3,000 steps,
       curves every 500; it sets lr and steps per arm from the training-world curves and reads nothing about the
       predictions (L30).
main:  the 41 runs of §4 at the lr the pilot chose (fill LR below after the pilot; the defaults are placeholders).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"RT-canon": 10, "RT-post": 10, "RT-all": 3, "RT-absolute": 5, "EI16": 5, "PF": 3, "RT-examples": 3, "RT-shuffled": 2}
LR = {a: 1e-3 for a in ARMS}                 # placeholders until the pilot
STEPS = 12000
PILOT_STEPS = 3000
COMMON = {"rows": 256, "inst": 32, "eval_every": 250, "warmup": 200, "wd": 0.01, "curve_per_world": 100}


def pilot():
    out = []
    for arm in ARMS:
        for lr in (3e-4, 1e-3, 3e-3):
            out.append({"name": f"pilot_{arm}_lr{lr:g}", "arm": arm, "seed": 100, "lr": lr, "steps": PILOT_STEPS, **{**COMMON, "eval_every": 500}})
    return out


def main_grid(lr=LR, steps=STEPS):
    out = []
    for arm, n in ARMS.items():
        for s in range(n):
            out.append({"name": f"{arm}_s{s}", "arm": arm, "seed": s, "lr": lr[arm], "steps": steps, **COMMON})
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "grids"), exist_ok=True)
    p = pilot()
    json.dump(p, open(os.path.join(HERE, "grids", "pilot.json"), "w"), indent=1)
    m = main_grid()
    json.dump(m, open(os.path.join(HERE, "grids", "main.json"), "w"), indent=1)
    print(f"pilot {len(p)} runs, main {len(m)} runs")
