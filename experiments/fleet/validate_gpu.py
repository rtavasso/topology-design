"""Validate the fleet trainer on a device against Study 4's CPU runs, seed by seed. Exits non-zero on failure.

Reference: experiments/paths/results_data/compact.json.gz holds Study 4's Stage-1 runs (10 seeds per graph at each
graph's selected lr, CPU single-run trainer). This script re-runs chosen graphs with the same configuration on the
device and compares, per seed:
  - t_fit (first eval with train accuracy >= 0.99): reached in a non-chaotic phase, so it should match exactly;
  - t_grok: may differ because float summation order differs between devices and chaotic phases amplify it, so the
    criterion is statistical: median |t_grok(device) - t_grok(CPU)| must be within the CPU seed IQR of that graph.

Usage: PYTHONPATH=src python experiments/fleet/validate_gpu.py [--device auto] [--graphs pk0000 pk0100 pk0111]
"""

from __future__ import annotations

import argparse
import gzip
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

REF = os.path.join(os.path.dirname(__file__), "..", "paths", "results_data", "compact.json.gz")


def iqr(v):
    q = statistics.quantiles(v, n=4)
    return q[2] - q[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--graphs", nargs="+", default=["pk0000", "pk0100", "pk0111"])
    ap.add_argument("--t-fit-match", type=float, default=0.8, help="required fraction of exact t_fit matches")
    args = ap.parse_args()
    dev = pick_device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    ref = {(r["arm"], r["seed"]): r for r in json.load(gzip.open(REF))["runs"] if r["arm"] in args.graphs}
    cfgs = [fl.FleetConfig(arm=a, lr=r["lr"], weight_decay=1.0, steps=40000, stop_after_grok=5000, dense_early=True,
                           model_seed=s, data_seed=s, tag="validate") for (a, s), r in sorted(ref.items())]
    t = time.time()
    res = fl.train_fleet(cfgs, dev)
    print(f"{len(cfgs)} runs on {dev} in {time.time() - t:.0f}s\n")
    ok = True
    fits = []
    for g in args.graphs:
        rows = [(c, o) for c, o in zip(cfgs, res) if c.arm == g]
        cpu_g = [ref[(g, c.model_seed)]["summary"]["t_grok"] for c, _ in rows]
        dev_g = [o["summary"]["t_grok"] for _, o in rows]
        fits += [o["summary"]["t_fit"] == ref[(g, c.model_seed)]["summary"]["t_fit"] for c, o in rows]
        pairs = [(a, b) for a, b in zip(dev_g, cpu_g) if a is not None and b is not None]
        miss = sum((a is None) != (b is None) for a, b in zip(dev_g, cpu_g))
        cpu_valid = [x for x in cpu_g if x is not None]
        spread = iqr(cpu_valid) if len(cpu_valid) >= 4 else float("inf")
        med_diff = statistics.median([abs(a - b) for a, b in pairs]) if pairs else float("inf")
        good = med_diff <= max(spread, 0.1 * statistics.median(cpu_valid)) and miss <= 1
        ok &= good
        print(f"{g}: lr {rows[0][0].lr:g}  CPU t_grok {cpu_g}\n{'':8s}device t_grok {dev_g}\n"
              f"{'':8s}median |diff| {med_diff:.0f} vs CPU IQR {spread:.0f}; grok/no-grok disagreements {miss} -> "
              f"{'PASS' if good else 'FAIL'}")
    frac = sum(fits) / len(fits)
    ok &= frac >= args.t_fit_match
    print(f"\nexact t_fit matches: {sum(fits)}/{len(fits)} ({'PASS' if frac >= args.t_fit_match else 'FAIL'})")
    print("VALIDATION", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
