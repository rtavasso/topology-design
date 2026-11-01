"""Grids for rung 3 (experiments/commit/DESIGN.md §6). Stage 1 (phase 1 + core B) is fixed; stage 2 (phase 2) is built
from the gate report after the derive step.

    python experiments/commit/make_grids.py                      # writes grids/cm_stage1.json
    python experiments/commit/make_grids.py --stage2 results/kaggle_route-cm_stage1/gate.json   # writes grids/cm_stage2.json

All runs use the rung-1 router trainer (model "router", run by experiments/route/run_route.py) at a CONSTANT lr of 1e-3
(L31). Arm names: <signal>[-<variant>]_<cell>; cells name the phase and the data (p1 = phase 1 on delta 1; A30 / A15 /
A08 = shift A at p_P 0.30 / 0.15 / 0.08; B100 = shift B at pp_frac 1.0; S = stability; d1 / d0 = from scratch).
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

BASE = dict(model="router", kind="state", state_input="full", lam=0.02, lr=1e-3, schedule="constant", per_run_init=True,
            eval_every=2500)
SIGNALS = {
    "RT": dict(teacher=True),
    "EX": dict(estimator="exact"),
    "IW16": dict(estimator="iw", K=16),
    "PF": dict(estimator="pf"),
    "PFIS16": dict(estimator="pfis", K=16),
    "EI16": dict(estimator="ei", W=16, ei_eps=0.1),
    "EI16e3": dict(estimator="ei", W=16, ei_eps=0.3),
    # core B
    "RL16B": dict(estimator="rlb", W=16, ei_eps=0.1),
    "ACMC16B": dict(estimator="acb", W=16, ei_eps=0.1, value_head=True, gae=1.0, critic_warmup=200),
    "AC16B": dict(estimator="acb", W=16, ei_eps=0.1, value_head=True, gae=0.9, critic_warmup=200),
    "RL16H": dict(estimator="rl", K=16),                       # beta set per run
    "PFB16": dict(estimator="pfb", W=16, ei_eps=0.1),
}
P1 = dict(steps=2500, warmup=500, save_state=True, deferral=1, p_P=0.3)
P2 = dict(steps=3000, warmup=100)
CB = dict(steps=5000, warmup=500, deferral=1)


def run(name, arm, sig, seed, **extra):
    return {"name": name, "arm": arm, "seed": seed, **BASE, **SIGNALS[sig], **extra}


def stage1():
    g = []
    for sig, n in (("EI16", 7), ("EX", 7), ("PF", 3), ("PFIS16", 5)):          # phase 1: the committed starting points and checks C
        for s in range(n):
            g.append(run(f"cm1_{sig}_p1_s{s}", f"{sig}_p1", sig, s, **P1))
    for sig, n in (("RL16B", 5), ("ACMC16B", 5), ("AC16B", 5)):             # core B: forced-breadth reward arms
        for s in range(n):
            g.append(run(f"cm1_{sig}_d1_s{s}", f"{sig}_d1", sig, s, **CB))
    for sig in ("RL16B", "AC16B"):
        for s in range(3):
            g.append(run(f"cm1_{sig}_d0_s{s}", f"{sig}_d0", sig, s, **{**CB, "deferral": 0}))
    for beta in (0.01, 0.03, 0.1):                                          # the entropy check
        for s in range(2):
            g.append(run(f"cm1_RL16H{beta:g}_d1_s{s}", f"RL16H{beta:g}_d1", "RL16H", s, beta=beta, **CB))
    for s in range(3):                                                      # PFB16 and its matched dose reference
        g.append(run(f"cm1_PFB16_d1_s{s}", "PFB16_d1", "PFB16", s, **CB))
        g.append(run(f"cm1_PFB16_d1p4_s{s}", "PFB16_d1p4", "PFB16", s, ans_pool=4, **CB))
        g.append(run(f"cm1_PFIS16_d1p4_s{s}", "PFIS16_d1p4", "PFIS16", s, ans_pool=4, **CB))
    return g


def stage2(gate_path):
    rep = json.load(open(gate_path))
    passed = {"E": [], "L": []}
    for name, g_ in sorted(rep["gate"].items(), key=lambda kv: kv[1]["seed"]):
        if g_["passes"]:
            passed[g_["start"]].append((g_["seed"], g_["ckpt"]))
    E_, L_ = passed["E"][:5], passed["L"][:5]
    assert len(E_) >= 3 and len(L_) >= 3, f"too few committed routers passed the gate: E {len(E_)}, L {len(L_)}"
    g = []
    A = lambda pP: dict(deferral=0, p_P=pP, **P2)
    def cont(sig, start, cell, cfg, routers, tag=None):
        for seed, ck in routers:
            arm = f"{sig}_{start}{cell}"
            g.append(run(f"cm2_{arm}_s{seed}", arm, sig, seed, init_from=ck, **cfg))
    # shift A from the E routers, p_P 0.30: the core
    cont("RT", "E", "A30", A(0.30), E_[:2])
    for sig in ("PF", "EX", "EI16", "PFIS16", "IW16"):
        cont(sig, "E", "A30", A(0.30), E_[:5])
    cont("EI16e3", "E", "A30", A(0.30), E_[:3])
    for pP, cell in ((0.15, "A15"), (0.08, "A08")):                        # the dose
        for sig in ("PF", "EX", "EI16", "PFIS16", "IW16"):
            cont(sig, "E", cell, A(pP), E_[:3])
    cont("RT", "L", "A30", A(0.30), L_[:2])                                 # from the L routers
    for sig in ("PF", "EX", "EI16", "PFIS16", "IW16"):
        cont(sig, "L", "A30", A(0.30), L_[:3])
    cont("EX", "E", "S", dict(deferral=1, p_P=0.3, **P2), E_[:2])           # stability
    cont("EI16", "L", "S", dict(deferral=1, p_P=0.3, **P2), L_[:2])
    for sig in ("EX", "EI16", "PF", "PFIS16"):                              # matched from-scratch references on delta 0
        for s in range(3):
            g.append(run(f"cm2_{sig}_d0_s{s}", f"{sig}_d0", sig, s, deferral=0, p_P=0.3, steps=3000, warmup=500))
    # shift B (secondary): filling in, from the L routers, and f = 0 controls
    Bc = dict(deferral=1, p_P=0.3, pp_frac=1.0, **P2)
    cont("RT", "L", "B100", Bc, L_[:2])
    for sig in ("PF", "EX", "EI16", "PFIS16"):
        cont(sig, "L", "B100", Bc, L_[:3])
        cont(sig, "L", "B0", dict(deferral=1, p_P=0.3, pp_frac=0.0, **P2), L_[:2])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", default=None, help="gate.json from experiments/commit/gate.py")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    grids = [("cm_stage1", stage1())] if not args.stage2 else [("cm_stage2", stage2(args.stage2))]
    for name, g in grids:
        assert len({c["name"] for c in g}) == len(g)
        json.dump(g, open(os.path.join(OUT, f"{name}.json"), "w"), indent=0)
        print(f"{name}: {len(g)} runs, ≈ {sum(0.085 * c['steps'] for c in g) / 3600:.1f} T4-hours")


if __name__ == "__main__":
    main()
