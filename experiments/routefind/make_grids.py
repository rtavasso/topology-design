"""Grids for rung 2 (experiments/routefind/DESIGN.md §6, revision 3: the router's training signal). Writes
experiments/routefind/grids/{rf_pilot,rf_main}.json (names distinct from rung 1's: code review 1, finding 4).

    python experiments/routefind/make_grids.py

All runs use the rung-1 router trainer (model "router", run by experiments/route/run_route.py); arms differ only in
the training signal (and EI16B's warm start inside the exact marginal's basin). Every run seeds its own init and
sampling stream (per_run_init). Arm names: <signal>d<deferral>[a<n_ans> for the spurious-density dose].
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

BASE = dict(model="router", kind="state", state_input="full", lam=0.02, steps=4000, eval_every=2000, per_run_init=True)
ARMS = {
    # current-policy posterior: exact (rung 1's RM) and sampled
    "EX": dict(estimator="exact"),
    "IW16": dict(estimator="iw", K=16),
    # fixed-prior posterior (the PFN signal): exact and sampled (GFlowNet trajectory balance)
    "PF": dict(estimator="pf"),
    "TB16": dict(estimator="tb", K=16),
    # reward: Monte-Carlo (REINFORCE) and bootstrapped (actor-critic with the router's contextual value V(x))
    "RL16": dict(estimator="rl", K=16),
    "AC16": dict(estimator="ac", K=16, value_head=True, gae=0.0),
    # search + imitation
    "EI16": dict(estimator="ei", W=16),
    # checks (derivable or already observed on CPU)
    "EI16B": dict(estimator="ei", W=16, warm_steps=1000),
    "SI16": dict(estimator="si", K=16),
    "ELL16": dict(estimator="ell", K=16),
}
CORE = ("EX", "IW16", "PF", "TB16", "RL16", "AC16", "EI16")
LR = {a: 1e-3 for a in ARMS}                                    # set by the pilot rule (§6)
CELLS = [  # (deferral, n_ans, {arm: seeds})
    (1, 16, {"EX": 3, "IW16": 5, "PF": 5, "TB16": 5, "RL16": 5, "AC16": 5, "EI16": 5}),
    (0, 16, {"EX": 2, "PF": 3, "TB16": 3, "AC16": 3, "EI16": 3}),
    (1, 4, {a: 3 for a in CORE}),                               # the spurious-density dose (D104)
    (1, 16, {"SI16": 2, "ELL16": 2}),                           # checks
    (0, 16, {"EI16B": 2, "IW16": 2}),
]
SEC = 0.09                                                      # T4 seconds per run-step (rung-1 pilot; signals measured in the pilot)


def name_of(arm, d, n_ans):
    return f"{arm}d{d}" + ("" if n_ans == 16 else f"a{n_ans}")


def run(name, arm, deferral, n_ans, seed, **extra):
    return {"name": name, "arm": name_of(arm, deferral, n_ans), "seed": seed, "lr": LR[arm], "deferral": deferral,
            "n_ans": n_ans, **BASE, **ARMS[arm], **extra}


def pilot():
    g = []
    for arm in ("IW16", "PF", "TB16", "RL16", "AC16", "EI16"):
        for lr in (3e-4, 1e-3, 3e-3):
            g.append(run(f"rfpilot_{arm}d1_lr{lr:g}_s0", arm, 1, 16, 0, lr=lr))
    for arm in ("PF", "TB16", "AC16"):                          # the δ0 controls in the pilot (L26)
        g.append(run(f"rfpilot_{arm}d0_lr0.001_s0", arm, 0, 16, 0, lr=1e-3))
    return g


def main_grid():
    g = []
    for d, n_ans, arms in CELLS:
        for arm, n in arms.items():
            for s in range(n):
                g.append(run(f"rfmain_{name_of(arm, d, n_ans)}_s{s}", arm, d, n_ans, s))
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, g in (("rf_pilot", pilot()), ("rf_main", main_grid())):
        assert len({c["name"] for c in g}) == len(g)
        json.dump(g, open(os.path.join(OUT, f"{name}.json"), "w"), indent=0)
        print(f"{name}: {len(g)} runs, ≈ {sum(SEC * c['steps'] for c in g) / 3600:.1f} T4-hours")


if __name__ == "__main__":
    main()
