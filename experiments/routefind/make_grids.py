"""Grids for rung 2 (experiments/routefind/DESIGN.md §6, revision 2). Writes experiments/routefind/grids/{rf_pilot,rf_main}.json
(names distinct from rung 1's so Kaggle kernels and result folders do not collide: code review 1, finding 4).

    python experiments/routefind/make_grids.py

All runs use the rung-1 router trainer (model "router", run by experiments/route/run_route.py); arms differ only in
the training estimator (and EI16B's warm start inside the exact marginal's basin). Every run seeds its own init and
sampling stream (per_run_init), so a run does not depend on the stack it shares.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

BASE = dict(model="router", kind="state", state_input="full", lam=0.02, steps=6000, eval_every=2000, per_run_init=True)
ARMS = {
    "EX": dict(estimator="exact"),
    "RT": dict(estimator="teacher"),
    "IW16": dict(estimator="iw", K=16),
    "ELL16": dict(estimator="ell", K=16),
    "RL16": dict(estimator="rl", K=16),
    "SI16": dict(estimator="si", K=16),
    "EI16": dict(estimator="ei", W=16),
    "EI16B": dict(estimator="ei", W=16, warm_steps=1000),     # 1,000 exact-marginal steps first: started inside EX's basin
}
LR = {a: 1e-3 for a in ARMS}                                    # set by the pilot rule (§6)
SEEDS = {1: {"EX": 3, "IW16": 5, "ELL16": 3, "RL16": 5, "SI16": 3, "EI16": 5},
         0: {"EX": 3, "IW16": 5, "ELL16": 3, "RL16": 5, "SI16": 3, "EI16": 5, "EI16B": 5, "RT": 2}}
SEC = 0.09                                                      # T4 seconds per run-step (rung-1 pilot; estimators measured in the pilot)


def run(name, arm, deferral, seed, **extra):
    return {"name": name, "arm": f"{arm}d{deferral}", "seed": seed, "lr": LR[arm], "deferral": deferral,
            **BASE, **ARMS[arm], **extra}


def pilot():
    g = []
    for arm in ("IW16", "RL16", "EI16"):
        for lr in (3e-4, 1e-3, 3e-3):
            g.append(run(f"rfpilot_{arm}d1_lr{lr:g}_s0", arm, 1, 0, lr=lr, steps=4000))
    for arm in ("IW16", "RL16", "EI16B"):                         # the controls run in the pilot (L26)
        g.append(run(f"rfpilot_{arm}d0_lr0.001_s0", arm, 0, 0, lr=1e-3, steps=4000))
    g.append(run("rfpilot_IW16d0_lr0.0003_s0", "IW16", 0, 0, lr=3e-4, steps=4000))     # δ0 at a second lr (review N6)
    return g


def main_grid():
    return [run(f"rfmain_{arm}d{d}_s{s}", arm, d, s) for d in (1, 0) for arm, n in SEEDS[d].items() for s in range(n)]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, g in (("rf_pilot", pilot()), ("rf_main", main_grid())):
        assert len({c["name"] for c in g}) == len(g)
        json.dump(g, open(os.path.join(OUT, f"{name}.json"), "w"), indent=0)
        print(f"{name}: {len(g)} runs, ≈ {sum(SEC * c['steps'] for c in g) / 3600:.1f} T4-hours")


if __name__ == "__main__":
    main()
