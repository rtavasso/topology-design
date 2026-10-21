"""Validate the fleet trainer on a GPU against Study 4's CPU results (experiments/paths/results_data/selection.json).

Re-runs Stage-0 cells of Study 4 (same graphs, lrs, seeds 400-402, horizon 40k, stop 5k after grokking) on the device and
compares the median grokking step per cell with the CPU reference. Trajectories diverge in chaotic phases (float
summation order differs between devices), so the check is statistical: medians should agree within run-to-run spread.

Usage: PYTHONPATH=src python experiments/fleet/validate_gpu.py [--device auto] [--cells pk0000:0.004 pk0120:0.004 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from topology_design import fleet as fl  # noqa: E402
from run_fleet import pick_device  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "..", "paths", "results_data", "selection.json")
DEFAULT_CELLS = ["pk0000:0.004", "pk0000:0.008", "pk0100:0.004", "pk0110:0.008", "pk0120:0.008", "pk0121:0.004"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    args = ap.parse_args()
    dev = pick_device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    ref = json.load(open(REF))
    cfgs = []
    for cell in args.cells:
        arm, lr = cell.split(":")
        for s in (400, 401, 402):
            cfgs.append(fl.FleetConfig(arm=arm, lr=float(lr), weight_decay=1.0, steps=40000, stop_after_grok=5000,
                                       dense_early=True, model_seed=s, data_seed=s, tag="validate"))
    t = time.time()
    res = fl.train_fleet(cfgs, dev)
    print(f"{len(cfgs)} runs on {dev} in {time.time() - t:.0f}s\n")
    print(f"{'cell':16s} {'CPU median t_grok':>18s} {'device t_grok (3 seeds)':>26s} {'device median':>14s}")
    for cell in args.cells:
        arm, lr = cell.split(":")
        tg = [r["summary"]["t_grok"] for r, c in zip(res, cfgs) if c.arm == arm and c.lr == float(lr)]
        med = statistics.median([x if x is not None else float("inf") for x in tg])
        cpu = ref[arm]["table"][str(float(lr))]["median_t_grok"]
        print(f"{cell:16s} {cpu:>18} {str(tg):>26s} {med:>14}")


if __name__ == "__main__":
    main()
