"""Run a rung-4 grid on the rewrite carrier (experiments/rewrite/DESIGN.md), resumable, sharded.

    PYTHONPATH=src python experiments/rewrite/run_rewrite.py --grid experiments/rewrite/grids/main.json \
        --data data/rewrite --out results/rewrite_main [--shard i/k] [--device cuda]

Each run writes <out>/results/<name>.json.gz; runs with a file are skipped, so a grid resumes. Runs are assigned to shards
by estimated cost (longest first). The data directory holds the files build_dataset.py writes; evaluation sets are loaded
once per process and shared by the runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from topology_design.rewrite import train as T  # noqa: E402

COST = {"EI16": 4.0, "RT-post": 1.3, "RT-examples": 1.5}      # relative per step


def est_cost(cfg):
    return int(cfg.get("steps", 10000)) * COST.get(cfg["arm"], 1.0)


def assign_shards(cfgs, k):
    order = sorted(range(len(cfgs)), key=lambda i: (-est_cost(cfgs[i]), cfgs[i].get("name", "")))
    out, loads = [[] for _ in range(k)], [0.0] * k
    for i in order:
        j = min(range(k), key=lambda j: (loads[j], j))
        out[j].append(cfgs[i]); loads[j] += est_cost(cfgs[i])
    return out, loads


def save(rec, path):
    with gzip.open(path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfgs = json.load(open(args.grid))
    i, k = (int(x) for x in args.shard.split("/"))
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(res_dir, c["name"] + ".json.gz"))]
    shards, loads = assign_shards(todo, k)
    mine = shards[i]
    print(f"shard {i}/{k}: {len(mine)} of {len(todo)} pending runs ({len(cfgs)} in grid), est. load {loads[i]:.0f}; device {args.device}", flush=True)
    sets = T.eval_sets_for({}, args.data)
    print({tag: {w: len(rs) for w, rs in s.items()} for tag, s in sets.items()}, flush=True)
    for cfg in mine:
        path = os.path.join(res_dir, cfg["name"] + ".json.gz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        try:
            rec = T.run(cfg, args.data, device=args.device, log=lambda *a: print(*a, flush=True), sets=sets)
            rec["shard"] = args.shard
            save(rec, path)
            print(f"done {cfg['name']} in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAILED {cfg['name']}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
