"""Run a grid of experiments with the fleet trainer: runs are packed into large batches on one device. Resumable.

Usage:
    PYTHONPATH=src python experiments/fleet/run_fleet.py --grid GRID.json --out results/X [--device auto] [--max-batch N]

Grid format as in experiments/grokking/run_sweep.py: {"base": {...}, "vary": {...}} or {"grids": [...]}, optional
"tie_data_seed": true. Fields are FleetConfig fields (src/topology_design/fleet.py). Runs that share shapes, task family,
horizon and eval schedule are packed together regardless of graph, lr, wd, multipliers or seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from topology_design import fleet as fl  # noqa: E402
from topology_design import grokking as gk  # noqa: E402

TUPLE_FIELDS = ("alphas", "block_lr_mult", "block_wd_mult", "branches")


def _tuplify(kw: dict) -> dict:
    for k in TUPLE_FIELDS:
        if kw.get(k) is not None:
            v = kw[k]
            kw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return kw


def expand(grid: dict) -> list[fl.FleetConfig]:
    if "grids" in grid:
        seen, out = set(), []
        for g in grid["grids"]:
            for c in expand(g):
                if c.run_id not in seen:
                    seen.add(c.run_id)
                    out.append(c)
        return out
    base, vary = grid.get("base", {}), grid.get("vary", {})
    keys = list(vary)
    cfgs = []
    for vals in itertools.product(*[vary[k] for k in keys]):
        kw = {**base, **dict(zip(keys, vals))}
        if grid.get("tie_data_seed"):
            kw["data_seed"] = kw["model_seed"]
        cfgs.append(fl.FleetConfig(**_tuplify(kw)))
    ids = [c.run_id for c in cfgs]
    assert len(ids) == len(set(ids)), "duplicate configurations in grid"
    return cfgs


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batches(cfgs: list[fl.FleetConfig], max_batch: int) -> list[list[fl.FleetConfig]]:
    groups: dict[tuple, list] = {}
    for c in cfgs:
        groups.setdefault(fl.struct_key(c), []).append(c)
    out = []
    for g in groups.values():
        g = sorted(g, key=lambda c: (c.lr, c.weight_decay, c.arm))  # similar run lengths together: less idle compute
        out += [g[i:i + max_batch] for i in range(0, len(g), max_batch)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-batch", type=int, default=0, help="runs per batch (0 = 256 cuda / 64 mps / 16 cpu)")
    ap.add_argument("--allow-tf32", action="store_true", help="faster on Ampere+ GPUs, less exact")
    ap.add_argument("--deterministic", action="store_true", help="bitwise-reproducible CUDA (slower)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    dev = pick_device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
    mb = args.max_batch or {"cuda": 256, "mps": 64}.get(dev, 16)
    cfgs = expand(json.load(open(args.grid)))
    os.makedirs(args.out, exist_ok=True)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(args.out, c.run_id + ".json"))]
    bs = batches(todo, mb)
    print(f"{len(cfgs)} runs in grid, {len(todo)} to do, {len(bs)} batches of up to {mb} on {dev}", flush=True)
    if args.dry:
        for b in bs:
            print(len(b), b[0].run_id, "...")
        return
    t0 = time.time()
    done = 0
    for bi, b in enumerate(bs):
        tb = time.time()
        results = fl.train_fleet(b, dev)
        for res in results:
            path = os.path.join(args.out, res["run_id"] + ".json")
            gk.save(res, path + ".tmp")
            os.replace(path + ".tmp", path)
        done += len(b)
        steps = sum((r["summary"]["steps_run"] or 0) for r in results)
        dt = time.time() - tb
        print(f"[batch {bi + 1}/{len(bs)}] {len(b)} runs, {steps} run-steps in {dt:.0f}s "
              f"({steps / max(dt, 1e-9):.0f} run-steps/s); {done}/{len(todo)} done; elapsed {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
