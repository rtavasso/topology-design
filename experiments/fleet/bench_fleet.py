"""Throughput of the fleet trainer vs batch size on a device.

Usage: PYTHONPATH=src python experiments/fleet/bench_fleet.py [--device auto] [--R 1 16 64 256] [--arm pk012000]
Prints ms per training step, run-steps per second, and ms per eval (batched measurement), for each batch size R.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from topology_design import fleet as fl  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_fleet import pick_device  # noqa: E402


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--R", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--arm", default="pk012000")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mixed", action="store_true", help="mix graphs in the batch (exercises per-run parent gather)")
    args = ap.parse_args()
    dev = pick_device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device {dev}, torch {torch.__version__}")
    arms = [args.arm] if not args.mixed else ["pk012000", "pk000000", "pk011000", "pk012300"]
    for R in args.R:
        cfgs = [fl.FleetConfig(arm=arms[j % len(arms)], model_seed=j, data_seed=j,
                               branches=((1, 2, 3), (4, 5, 6)) if len(arms[0]) == 8 else None) for j in range(R)]
        try:
            f = fl.Fleet(cfgs, dev)
        except RuntimeError as e:
            print(f"R={R}: {e}")
            break
        tasks = [fl.build_task(c) for c in cfgs]
        tr = torch.stack([t["tr"] for t in tasks]).to(dev)
        a, b = tasks[0]["a"].to(dev), tasks[0]["b"].to(dev)
        Y = torch.stack([t["y"] for t in tasks]).to(dev).gather(1, tr)
        grid = {"a": a, "b": b, "key": tasks[0]["key"].to(dev), "y": torch.stack([t["y"] for t in tasks]).to(dev)}
        for s in ("train", "test", "exc", "rule_train"):
            grid[s] = torch.stack([t[s] for t in tasks]).to(dev)
        opt = fl.FleetAdamW(f)
        active = torch.ones(R, dtype=torch.bool, device=dev)
        n = tr.shape[1]

        grid["n_key"] = torch.bincount(tasks[0]["key"], minlength=cfgs[0].p).to(dev)

        def step():  # identical work to train_fleet's step (including divergence masking)
            opt.zero_grad()
            lg = f.forward(a[tr], b[tr])
            per = F.cross_entropy(lg.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1)
            torch.where(active & torch.isfinite(per), per, torch.zeros_like(per)).sum().backward()
            opt.step(active)

        try:
            for _ in range(3):
                step()
            fl.measure(f, grid)  # warm-up
            sync(dev)
            t = time.time()
            for _ in range(args.steps):
                step()
            sync(dev)
            ms = (time.time() - t) / args.steps * 1000
            t = time.time()
            fl.measure(f, grid)
            sync(dev)
            ev = (time.time() - t) * 1000
        except RuntimeError as e:  # out of memory at this batch size
            print(f"R={R}: {str(e)[:120]}")
            break
        print(f"R={R:4d}: {ms:8.2f} ms/step  {R * 1000 / ms:9.0f} run-steps/s  "
              f"{ms / R:7.3f} ms per run-step  eval {ev:8.1f} ms ({ev / R:.2f} ms/run)", flush=True)
        del f, opt
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
