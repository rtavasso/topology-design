"""Run a rung-0 grid (experiments/ptr/DESIGN.md), resumable, with runs of one arm stacked into one batched model.

    PYTHONPATH=src python experiments/ptr/run_ptr.py --grid experiments/ptr/grids/main.json --out results/ptr_main \
        [--shard i/k] [--device cuda] [--max-stack 32] [--no-stack]

Runs are grouped by everything except seed, lr and tail_factor; each group (up to --max-stack runs) trains as one
stack (src/topology_design/ptr/train_stacked.py). Each run still writes its own <out>/results/<name>.json.gz, and
runs with a file are skipped, so a grid resumes. Shards take every k-th group (after sorting), one shard per GPU.
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

from topology_design.ptr.train import run  # noqa: E402
from topology_design.ptr.train_stacked import run_stack, PER_RUN_KEYS  # noqa: E402


def group_key(cfg):
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in PER_RUN_KEYS and k not in ("name",)},
                      sort_keys=True)


def save(rec, path):
    with gzip.open(path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-stack", type=int, default=32)
    ap.add_argument("--no-stack", action="store_true")
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
        size = 1 if args.no_stack else args.max_stack
        for j in range(0, len(g), size):
            chunks.append(g[j:j + size])
    mine = chunks[i::k]
    print(f"{len(grid)} runs, {len(todo)} to do, {len(chunks)} stacks, {len(mine)} in shard {args.shard} on {dev}",
          flush=True)
    done = 0
    for chunk in mine:
        if args.max_groups is not None and done >= args.max_groups:
            break
        names = [c["name"] for c in chunk]
        t0 = time.time()
        print(f"=== stack of {len(chunk)}: {names[0]} .. {names[-1]}", flush=True)
        try:
            if len(chunk) == 1 and args.no_stack:
                c = chunk[0]
                rec = run({k_: v for k_, v in c.items() if k_ not in ("name", "arm")}, dev,
                          log=lambda s: print(f"[{c['name']}] {s}", flush=True))
                rec["name"], rec["arm"] = c["name"], c["arm"]
                save(rec, path_of(c))
            else:
                logf = lambda s: print(f"[{names[0]}+{len(chunk)-1}] {s}", flush=True)
                try:
                    recs = run_stack(chunk, dev, log=logf)
                except Exception:
                    traceback.print_exc()
                    print("=== retrying this stack in fp32 (no autocast)", flush=True)
                    recs = run_stack(chunk, dev, log=logf, amp=False)
                for c, rec in zip(chunk, recs):
                    save(rec, path_of(c))
            print(f"=== done in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED stack {names[0]}", flush=True)
        done += 1
    print(f"shard {args.shard}: {done} stacks this session")


if __name__ == "__main__":
    main()
