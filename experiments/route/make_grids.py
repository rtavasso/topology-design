"""Grids for rung 1 (experiments/route/DESIGN.md §8). Writes experiments/route/grids/{pilot,main}.json.

    python experiments/route/make_grids.py

A config's `model` field picks the trainer: "router" (train_route.run_stack: RM, RM-λ0, RM-idx, RM-T) or "neural"
(train_neural.run_stack: S8, S2, LF). Runs stack by everything except seed and lr.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

ARMS = {
    "RM": dict(model="router", kind="state", lam=0.02),
    "RMl0": dict(model="router", kind="state", lam=0.0),
    "RMidx": dict(model="router", kind="index", lam=0.02),
    "RMT": dict(model="router", kind="state", lam=0.02, teacher=True),
    "S8": dict(model="neural", kind="static", layers=8),
    "S2": dict(model="neural", kind="static", layers=2),
    "LF": dict(model="neural", kind="looped", halting="fixed", T=10),
}
DEFAULT_LR = {"RM": 1e-3, "RMl0": 1e-3, "RMidx": 1e-2, "RMT": 1e-3, "S8": 1e-4, "S2": 3e-4, "LF": 1e-3}
STEPS = {"RM": 6000, "RMl0": 6000, "RMidx": 6000, "RMT": 6000, "S8": 25000, "S2": 8000, "LF": 6000}
EVAL_EVERY = {"S8": 5000}
# T4 seconds per run-step (routed arms measured in the pilot; neural arms from rung 0 at d = 344: x1.1)
SEC = {"RM": 0.02, "RMl0": 0.02, "RMidx": 0.01, "RMT": 0.02, "S8": 0.058, "S2": 0.015, "LF": 0.072}


def run(name, arm, seed, **extra):
    return {"name": name, "arm": arm, "seed": seed, "lr": DEFAULT_LR[arm], **ARMS[arm], "steps": STEPS[arm],
            "eval_every": EVAL_EVERY.get(arm, 2000), **extra}


def pilot():
    g = []
    for lr in (3e-4, 1e-3, 3e-3):
        for seed in (0, 1):
            g.append(run(f"pilot_RM_lr{lr:g}_s{seed}", "RM", seed, lr=lr, steps=4000))
    for lam in (0.005, 0.1):
        for seed in (0, 1):
            g.append(run(f"pilot_RM_lam{lam:g}_s{seed}", "RM", seed, lam=lam, steps=4000))
    for lr in (3e-4, 1e-3):
        for seed in (0, 1):
            g.append(run(f"pilot_LF_lr{lr:g}_s{seed}", "LF", seed, lr=lr, steps=6000))
    for lr in (3e-5, 1e-4):
        for seed in (0, 1):
            g.append(run(f"pilot_S8_lr{lr:g}_s{seed}", "S8", seed, lr=lr, steps=15000, eval_every=5000))
    return g


def main_grid():
    g = []
    for arm, n in (("RM", 10), ("RMl0", 5), ("RMidx", 5), ("RMT", 5), ("S8", 5), ("LF", 10), ("S2", 3)):
        for seed in range(n):
            g.append(run(f"main_{arm}_s{seed}", arm, seed))
    return g


def cost_hours(grid):
    return sum(SEC[c["arm"]] * c["steps"] for c in grid) / 3600


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, g in (("pilot", pilot()), ("main", main_grid())):
        assert len({c["name"] for c in g}) == len(g)
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(g, f, indent=0)
        print(f"{name}: {len(g)} runs, ≈ {cost_hours(g):.1f} T4-hours")


if __name__ == "__main__":
    main()
