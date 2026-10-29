"""Grids for rung 2 (experiments/routefind/DESIGN.md §6). Writes experiments/routefind/grids/{pilot,main}.json.

    python experiments/routefind/make_grids.py

All runs use the rung-1 router trainer (model "router", run by experiments/route/run_route.py); arms differ only in
the training estimator. Runs stack by everything except seed and lr.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

BASE = dict(model="router", kind="state", state_input="full", lam=0.02, steps=6000, eval_every=2000)
ARMS = {
    "EX": dict(estimator="exact"),
    "IW4": dict(estimator="iw", K=4),
    "IW16": dict(estimator="iw", K=16),
    "RF16": dict(estimator="rf", K=16),
    "EI4": dict(estimator="ei", W=4),
    "EI16": dict(estimator="ei", W=16),
}
LR = {"EX": 1e-3, "IW4": 1e-3, "IW16": 1e-3, "RF16": 1e-3, "EI4": 1e-3, "EI16": 1e-3}   # set by the pilot rule (§6)
SEEDS = {"EX": 2, "IW4": 3, "IW16": 5, "RF16": 5, "EI4": 3, "EI16": 10}
SEC = 0.09                       # T4 seconds per run-step (rung-1 pilot); the estimators' overhead is measured in the pilot


def run(name, arm, deferral, seed, **extra):
    return {"name": name, "arm": f"{arm}d{deferral}", "seed": seed, "lr": LR[arm], "deferral": deferral,
            **BASE, **ARMS[arm], **extra}


def pilot():
    g = []
    for arm in ("IW16", "RF16", "EI16"):
        for lr in (3e-4, 1e-3, 3e-3):
            g.append(run(f"rfpilot_{arm}d1_lr{lr:g}_s0", arm, 1, 0, lr=lr, steps=4000))
        g.append(run(f"rfpilot_{arm}d0_lr0.001_s0", arm, 0, 0, lr=1e-3, steps=4000))
    return g


def main_grid():
    return [run(f"rfmain_{arm}d{d}_s{s}", arm, d, s) for d in (1, 0) for arm in ARMS for s in range(SEEDS[arm])]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, g in (("pilot", pilot()), ("main", main_grid())):
        assert len({c["name"] for c in g}) == len(g)
        json.dump(g, open(os.path.join(OUT, f"{name}.json"), "w"), indent=0)
        print(f"{name}: {len(g)} runs, ≈ {sum(SEC * c['steps'] for c in g) / 3600:.1f} T4-hours")


if __name__ == "__main__":
    main()
