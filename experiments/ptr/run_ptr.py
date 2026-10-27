"""Run a rung-0 grid (experiments/ptr/DESIGN.md), one result file per run, resumable.

    PYTHONPATH=src python experiments/ptr/run_ptr.py --grid experiments/ptr/grids/main.json --out results/ptr_main \
        [--shard i/k] [--device cuda] [--max-runs N]

Each run writes <out>/results/<name>.json.gz; runs whose file exists are skipped, so a grid resumes. Shards take every
k-th run (after sorting by name), which lets several processes share one or more GPUs.
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

from topology_design.ptr.train import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    i, k = (int(x) for x in args.shard.split("/"))
    grid = sorted(json.load(open(args.grid)), key=lambda c: c["name"])
    mine = grid[i::k]
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    done = 0
    for cfg in mine:
        path = os.path.join(res_dir, cfg["name"] + ".json.gz")
        if os.path.exists(path):
            continue
        if args.max_runs is not None and done >= args.max_runs:
            break
        t0 = time.time()
        print(f"=== {cfg['name']} on {dev}", flush=True)
        try:
            rec = run({k_: v for k_, v in cfg.items() if k_ not in ("name", "arm")}, dev,
                      log=lambda s: print(f"[{cfg['name']}] {s}", flush=True))
            rec["name"], rec["arm"] = cfg["name"], cfg["arm"]
            with gzip.open(path + ".tmp", "wt") as f:
                json.dump(rec, f)
            os.replace(path + ".tmp", path)
            print(f"=== done {cfg['name']} in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED {cfg['name']}", flush=True)
        done += 1
    print(f"shard {args.shard}: {done} runs this session; {len(mine)} in shard")


if __name__ == "__main__":
    main()
