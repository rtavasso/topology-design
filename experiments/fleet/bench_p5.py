"""Cost of each Phase-5 feature relative to plain training (run-steps per second over a short horizon).

Usage: PYTHONPATH=src python experiments/fleet/bench_p5.py [--device auto] [--R 64] [--steps 300]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from topology_design import fleet as fl  # noqa: E402
from run_fleet import pick_device  # noqa: E402

FEATURES = {
    "plain": {},
    "branch time constants": dict(branch_lr_mult=(1.0, 1.0), branch_wd_mult=(0.5, 2.0)),
    "unmasked branch": dict(unmask_branch=0),
    "consolidation": dict(consolidate=0.1),
    "route time": dict(route="time"),
    "route rot": dict(route="rot"),
    "route agree (group grads)": dict(route="agree"),
    "signal measurement (5A)": dict(measure_signal=True),
    "arrivals (5E)": dict(arrive_start=348, arrive_n=35, arrive_every=10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    dev = pick_device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device {dev}, R={args.R}, {args.steps} steps, eval every 100 (no dense early evals)")
    base = None
    for name, extra in FEATURES.items():
        cfgs = [fl.FleetConfig(arm="pk012000", branches=((1, 2, 3), (4, 5, 6)), steps=args.steps, eval_every=100,
                               dense_early=False, model_seed=j, data_seed=j, **extra) for j in range(args.R)]
        t = time.time()
        try:
            fl.train_fleet(cfgs, dev)
        except RuntimeError as e:
            print(f"{name:28s} failed: {str(e)[:100]}")
            continue
        if dev == "cuda":
            torch.cuda.synchronize()
        rs = args.R * (args.steps + 1) / (time.time() - t)
        base = base or rs
        print(f"{name:28s} {rs:8.0f} run-steps/s   ({base / rs:4.2f}x the plain cost)", flush=True)


if __name__ == "__main__":
    main()
