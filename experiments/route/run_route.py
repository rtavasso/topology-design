"""Run a rung-1 grid (experiments/route/DESIGN.md), resumable, with runs of one arm stacked into one model.

    PYTHONPATH=src python experiments/route/run_route.py --grid experiments/route/grids/main.json --out results/route_main \
        [--shard i/k] [--device cuda] [--max-stack 10]

Runs are grouped by everything except seed and lr; each group (up to --max-stack runs) trains as one stack with the
trainer its `model` field names (router -> train_route, neural -> train_neural). Each run writes
<out>/results/<name>.json.gz; runs with a file are skipped, so a grid resumes. Stacks are assigned to shards by
estimated cost (longest first), one shard per GPU.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from topology_design.route import train_route, train_neural  # noqa: E402

PER_RUN_KEYS = ("seed", "lr")
SEC = {"router": 0.09, "index": 0.02, "static": 0.012, "looped": 0.0113}   # per run-step, pilot-measured (static: per layer; looped: per loop)


def est_cost(cfg):
    steps = int(cfg.get("steps", 6000))
    if cfg.get("model") == "router":
        return steps * SEC["index" if cfg.get("kind") == "index" else "router"]
    if cfg.get("kind") == "static":
        return steps * SEC["static"] * int(cfg.get("layers", 4))
    return steps * SEC["looped"] * int(cfg.get("T", 10))


def assign_shards(chunks, k):
    cost = [sum(est_cost(c) for c in ch) for ch in chunks]
    order = sorted(range(len(chunks)), key=lambda i: (-cost[i], chunks[i][0].get("name", "")))
    out, loads = [[] for _ in range(k)], [0.0] * k
    for i in order:
        j = min(range(k), key=lambda j: (loads[j], j))
        out[j].append(chunks[i]); loads[j] += cost[i]
    return out, loads


def group_key(cfg):
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in PER_RUN_KEYS and k != "name"}, sort_keys=True)


def save(rec, path):
    with gzip.open(path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def run_chunk(chunk, dev, path_of):
    names = [c["name"] for c in chunk]
    logf = lambda s: print(f"[{names[0]}+{len(chunk)-1}] {s}", flush=True)
    trainer = train_route if chunk[0].get("model") == "router" else train_neural
    try:
        recs = trainer.run_stack(chunk, dev, log=logf)
    except Exception:
        traceback.print_exc()
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        if len(chunk) > 1:
            h = len(chunk) // 2
            print(f"=== splitting the stack of {len(chunk)} into {h} + {len(chunk) - h} and retrying", flush=True)
            errs = []
            for part in (chunk[:h], chunk[h:]):                 # both halves are tried even if the first fails
                try:
                    run_chunk(part, dev, path_of)
                except Exception as e:
                    traceback.print_exc(); errs.append(e)
            if errs:
                raise errs[0]
            return
        raise
    for c, rec in zip(chunk, recs):
        save(rec, path_of(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-stack", type=int, default=10)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    i, k = (int(x) for x in args.shard.split("/"))
    grid = sorted(json.load(open(args.grid)), key=lambda c: c["name"])
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    path_of = lambda c: os.path.join(res_dir, c["name"] + ".json.gz")
    todo = [c for c in grid if not os.path.exists(path_of(c))]
    groups = defaultdict(list)
    for c in todo:
        groups[group_key(c)].append(c)
    chunks = []
    for key in sorted(groups):
        g = groups[key]
        for j in range(0, len(g), args.max_stack):
            chunks.append(g[j:j + args.max_stack])
    per_shard, loads = assign_shards(chunks, k)
    mine = per_shard[i]
    print(f"{len(grid)} runs, {len(todo)} to do, {len(chunks)} stacks, {len(mine)} in shard {args.shard} on {dev}, "
          f"est. {loads[i]/3600:.1f} h this shard (all: {', '.join(f'{l/3600:.1f}' for l in loads)} h)", flush=True)
    done = 0
    for chunk in mine:
        if args.max_groups is not None and done >= args.max_groups:
            break
        names = [c["name"] for c in chunk]
        t0 = time.time()
        print(f"=== stack of {len(chunk)}: {names[0]} .. {names[-1]}", flush=True)
        try:
            run_chunk(chunk, dev, path_of)
            print(f"=== done in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED stack {names[0]} (no file written; a later session retries it)", flush=True)
        done += 1
    print(f"shard {args.shard}: {done} stacks this session")


if __name__ == "__main__":
    main()
