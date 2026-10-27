"""Grids for rung 0 (experiments/ptr/DESIGN.md §6). Writes experiments/ptr/grids/*.json, each a list of run configs.

    python experiments/ptr/make_grids.py            # writes all grids and prints counts
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "grids")

ARMS = {
    "S2": dict(kind="static", layers=2),
    "S8": dict(kind="static", layers=8),
    "S8pm": dict(kind="static", layers=8, d=96),
    "LF": dict(kind="looped", halting="fixed"),
    "LO": dict(kind="looped", halting="oracle"),
    "LHs": dict(kind="looped", halting="learned", halt_input="state"),
    "LHx": dict(kind="looped", halting="learned", halt_input="index"),
    "LFni": dict(kind="looped", halting="fixed", injection=False),
    "LHsni": dict(kind="looped", halting="learned", halt_input="state", injection=False),
}
# learning rates per arm are filled in from the pilot; these are the pilot's candidates
LR_CANDIDATES = (3e-4, 1e-3, 3e-3)
DEFAULT_LR = {"S2": 1e-3, "S8": 3e-4, "S8pm": 1e-3, "LF": 1e-3, "LO": 1e-3, "LHs": 1e-3, "LHx": 1e-3,
              "LFni": 1e-3, "LHsni": 1e-3}


def run(name, arm, seed, **extra):
    cfg = {"name": name, "arm": arm, "seed": seed, "lr": DEFAULT_LR[arm], **ARMS[arm], **extra}
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    grids = {}
    # pilot: learning rates (3 x 2 seeds per arm) and PonderNet constants for LHs, short budget
    g = []
    for arm in ("S2", "S8", "LF", "LO", "LHs", "LHx"):
        for lr in LR_CANDIDATES:
            for seed in (0, 1):
                g.append(run(f"pilot_{arm}_lr{lr:g}_s{seed}", arm, seed, lr=lr, steps=6000, eval_every=2000))
    for beta in (0.003, 0.01, 0.03):
        for lam_p in (0.1, 0.2, 0.4):
            for seed in (0, 1):
                g.append(run(f"pilot_LHs_b{beta:g}_l{lam_p:g}_s{seed}", "LHs", seed, beta=beta, lam_p=lam_p,
                             steps=6000, eval_every=2000))
    grids["pilot"] = g
    # main cell
    g = []
    for arm, n in (("S8", 29), ("LO", 29), ("LHs", 29), ("S2", 10), ("LF", 10), ("LHx", 10)):
        for seed in range(n):
            g.append(run(f"main_{arm}_s{seed}", arm, seed))
    grids["main"] = g
    # P-pool: deep tail subsampled
    g = []
    for f in (1e-1, 1e-2, 1e-3, 1e-4):
        for arm in ("S8", "LO", "LHs"):
            for seed in range(10):
                g.append(run(f"ppool_f{f:g}_{arm}_s{seed}", arm, seed, tail_factor=f))
    grids["ppool"] = g
    # injection ablation
    grids["injection"] = [run(f"inj_{arm}_s{seed}", arm, seed) for arm in ("LFni", "LHsni") for seed in range(10)]
    # doubling-allowed regime
    grids["doubling"] = [run(f"dbl_{arm}_s{seed}", arm, seed, pair_pair_mask=False, ctx_single_step=False)
                         for arm in ("S8", "LF", "LO", "LHs") for seed in range(10)]
    # parameter-matched static
    grids["pm"] = [run(f"pm_S8pm_s{seed}", "S8pm", seed) for seed in range(10)]
    for name, g in grids.items():
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(g, f, indent=0)
        print(f"{name}: {len(g)} runs")
    print("total (without pilot):", sum(len(g) for n, g in grids.items() if n != "pilot"))


if __name__ == "__main__":
    main()
