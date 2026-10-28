"""Grids for rung 1 (experiments/route/DESIGN.md §8, revision 2). Writes experiments/route/grids/{pilot,main}.json.

    python experiments/route/make_grids.py

A config's `model` field picks the trainer: "router" (train_route.run_stack: RM, RMs, RMd0, RMT, RMidx) or "neural"
(train_neural.run_stack: S8, S2, LF, LFd0). Runs stack by everything except seed and lr.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

ARMS = {
    "RM": dict(model="router", kind="state", state_input="full", deferral=1, lam=0.02),
    "RMs": dict(model="router", kind="state", state_input="symbol", deferral=1, lam=0.02),      # F4 null
    "RMd0": dict(model="router", kind="state", state_input="full", deferral=0, lam=0.02),       # local-trigger control
    "RMT": dict(model="router", kind="state", state_input="full", deferral=1, lam=0.02, teacher=True),
    "RMidx": dict(model="router", kind="index", deferral=1, lam=0.02),
    "S8": dict(model="neural", kind="static", layers=8, deferral=1),
    "S2": dict(model="neural", kind="static", layers=2, deferral=1),
    "LF": dict(model="neural", kind="looped", halting="fixed", T=10, deferral=1),
    "LFd0": dict(model="neural", kind="looped", halting="fixed", T=10, deferral=0),
}
DEFAULT_LR = {"RM": 1e-3, "RMs": 1e-3, "RMd0": 1e-3, "RMT": 1e-3, "RMidx": 1e-2, "S8": 1e-4, "S2": 3e-4, "LF": 1e-3, "LFd0": 1e-3}
STEPS = {"RM": 6000, "RMs": 6000, "RMd0": 6000, "RMT": 6000, "RMidx": 6000, "S8": 12000, "S2": 8000, "LF": 6000, "LFd0": 6000}
EVAL_EVERY = {"S8": 4000}
# T4 seconds per run-step, measured in the revision-2 pilot (PILOT.md): routed 0.09 (stacks of 2 and 6 alike),
# LF 0.113 (T = 10), S8 0.095 (8 layers); RMidx and S2 scaled from those
SEC = {"RM": 0.09, "RMs": 0.09, "RMd0": 0.09, "RMT": 0.09, "RMidx": 0.02, "S8": 0.095, "S2": 0.03, "LF": 0.113, "LFd0": 0.113}


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
    for seed in (0, 1):
        g.append(run(f"pilot_RMs_lr0.001_s{seed}", "RMs", seed, steps=4000))
        g.append(run(f"pilot_RMT_lr0.001_s{seed}", "RMT", seed, steps=4000))
    for lr in (3e-4, 1e-3):
        for seed in (0, 1):
            g.append(run(f"pilot_LF_lr{lr:g}_s{seed}", "LF", seed, lr=lr, steps=6000))
    for lr in (3e-5, 1e-4):
        g.append(run(f"pilot_S8_lr{lr:g}_s0", "S8", 0, lr=lr, steps=8000, eval_every=4000))
    return g


def main_grid():
    g = []
    for arm, n in (("RM", 10), ("RMs", 5), ("RMd0", 5), ("RMT", 5), ("RMidx", 5), ("LF", 10), ("LFd0", 5), ("S8", 3), ("S2", 3)):
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
