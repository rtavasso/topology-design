"""Grids for rung 2 (experiments/routefind/DESIGN.md §6, revision 4: the router's training signal). Writes
experiments/routefind/grids/{rf_pilot,rf_main}.json (names distinct from rung 1's: code review 1, finding 4).

    python experiments/routefind/make_grids.py

All runs use the rung-1 router trainer (model "router", run by experiments/route/run_route.py); arms differ only in
the training signal (and EI16B's warm start inside the exact marginal's basin). Every run seeds its own init and
sampling stream (per_run_init). Arm names: <signal>d<deferral>[p<ans_pool> for the spurious-density dose].
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

BASE = dict(model="router", kind="state", state_input="full", lam=0.02, steps=5000, eval_every=2500, per_run_init=True)
ARMS = {
    # current-policy posterior: exact (rung 1's RM), exact with the answer-only likelihood, sampled
    "EX": dict(estimator="exact"),
    "EXAL": dict(estimator="exal"),
    "IW16": dict(estimator="iw", K=16),
    # fixed-prior posterior (the PFN signal): exact (a reference: it enumerates the states), sampled forward KL
    # (self-normalized importance weights), sampled reverse KL (trajectory balance)
    "PF": dict(estimator="pf"),
    "PFIS16": dict(estimator="pfis", K=16),
    "TB16": dict(estimator="tb", K=16),
    # reward: Monte-Carlo with route-level baselines (REINFORCE + RLOO), Monte-Carlo with the router's learned state
    # value as baseline, and bootstrapped through it (TD(0))
    "RL16": dict(estimator="rl", K=16),
    # (all: 200 critic-only steps first, policy evaluation before improvement; DESIGN.md §9). AC16 bootstraps with
    # TD(lambda = 0.9); pure TD(0) (AC16T0) collapses into halting at the start on CPU, so it is a check
    "ACMC16": dict(estimator="ac", K=16, value_head=True, gae=1.0, critic_warmup=200),
    "AC16": dict(estimator="ac", K=16, value_head=True, gae=0.9, critic_warmup=200),
    "AC16T0": dict(estimator="ac", K=16, value_head=True, gae=0.0, critic_warmup=200),
    # search + imitation
    "EI16": dict(estimator="ei", W=16),
    # checks (derivable or already observed on CPU)
    "EI16B": dict(estimator="ei", W=16, warm_steps=1000),
    "SI16": dict(estimator="si", K=16),
    "ELL16": dict(estimator="ell", K=16),
}
SAMPLED_CORE = ("IW16", "PFIS16", "TB16", "RL16", "ACMC16", "AC16", "EI16")   # the candidates (no state enumeration)
# lr per signal, set by the pilot rule (DESIGN.md §6; PILOT.md §2): the lr with the highest held-out shortest-correct
# accuracy at 5k on δ1, ties by the mean over the last 1,000 steps, then 1e-3. Pilot (seed 100): IW16 1e-3 (3e-4 and
# 1e-3 learned, 1e-3 earlier; 3e-3 collapsed), PFIS16 1e-3 (all three learned; 1e-3 earliest), TB16 1e-3 (0.97 vs
# 0.86 / 0.84, none at criterion), EI16 1e-3 (3e-4 and 1e-3 learned; 1e-3 earlier), RL16 / ACMC16 / AC16 1e-3 (every
# lr at the floor; 3e-3 collapsed into halting or never halting).
LR = {a: 1e-3 for a in ARMS}
CELLS = [  # (deferral, ans_pool, {arm: seeds}); ans_pool 0 = every episode draws from all 16 answers
    # D108 (PILOT.md §4): the reward arms RL16 / ACMC16 / AC16 collapsed into a deterministic policy within 200-400
    # steps at every pilot lr and both deferrals (distinct sampled routes per query -> 1, so their advantages are
    # identically zero); P4's preregistered uninterpretability condition (RL16 at the floor) is met, so they run as
    # 2-seed checks on δ1 only. Their δ0 and dose cells are dropped (−21 runs: 93 → 72).
    (1, 0, {"EX": 3, "EXAL": 2, "PF": 2, "IW16": 5, "PFIS16": 5, "TB16": 5, "RL16": 2, "ACMC16": 2, "AC16": 2, "EI16": 5}),
    (0, 0, {"EX": 2, "EXAL": 3, "PF": 2, "IW16": 2, "PFIS16": 3, "TB16": 3, "EI16": 2}),
    (1, 4, {"EX": 2, "IW16": 3, "PFIS16": 3, "TB16": 3, "EI16": 3}),   # the dose
    (1, 0, {"SI16": 2, "ELL16": 2, "AC16T0": 2}),               # checks
    (0, 0, {"EI16B": 2}),
]
SEC = 0.09                                                      # T4 seconds per run-step (rung-1 pilot; signals measured in the pilot)
PILOT_SEED = 100                                                # not a main-grid seed (design review 3, S5)


def name_of(arm, d, pool):
    return f"{arm}d{d}" + (f"p{pool}" if pool else "")


def run(name, arm, deferral, pool, seed, **extra):
    return {"name": name, "arm": name_of(arm, deferral, pool), "seed": seed, "lr": LR[arm], "deferral": deferral,
            "ans_pool": pool, **BASE, **ARMS[arm], **extra}


def pilot():
    g = []
    for arm in SAMPLED_CORE:
        for lr in (3e-4, 1e-3, 3e-3):
            g.append(run(f"rfpilot_{arm}d1_lr{lr:g}_s{PILOT_SEED}", arm, 1, 0, PILOT_SEED, lr=lr))
    for arm in ("EXAL", "PFIS16", "TB16", "RL16", "AC16"):     # the δ0 controls in the pilot (L26)
        g.append(run(f"rfpilot_{arm}d0_lr0.001_s{PILOT_SEED}", arm, 0, 0, PILOT_SEED, lr=1e-3))
    g.append(run(f"rfpilot_EXALd0_lr0.003_s{PILOT_SEED}", "EXAL", 0, 0, PILOT_SEED, lr=3e-3))   # P2 beyond one lr (review 4, S7)
    return g


def main_grid():
    g = []
    for d, pool, arms in CELLS:
        for arm, n in arms.items():
            for s in range(n):
                g.append(run(f"rfmain_{name_of(arm, d, pool)}_s{s}", arm, d, pool, s))
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, g in (("rf_pilot", pilot()), ("rf_main", main_grid())):
        assert len({c["name"] for c in g}) == len(g)
        json.dump(g, open(os.path.join(OUT, f"{name}.json"), "w"), indent=0)
        print(f"{name}: {len(g)} runs, ≈ {sum(SEC * c['steps'] for c in g) / 3600:.1f} T4-hours")


if __name__ == "__main__":
    main()
