"""Run a rung-0 grid (experiments/ptr/DESIGN.md), resumable, with runs of one arm stacked into one batched model.

    PYTHONPATH=src python experiments/ptr/run_ptr.py --grid experiments/ptr/grids/main.json --out results/ptr_main \
        [--shard i/k] [--device cuda] [--max-stack 32] [--no-stack]

Runs are grouped by everything except seed, lr and tail_factor; each group (up to --max-stack runs) trains as one
stack (src/topology_design/ptr/train_stacked.py). Each run still writes its own <out>/results/<name>.json.gz, and
runs with a file are skipped, so a grid resumes. Stacks are assigned to shards by estimated cost (longest first,
--shard-mode balanced, the default) or round-robin (--shard-mode modulo); one shard per GPU.
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
from topology_design.ptr.train_stacked import run_stack, PER_RUN_KEYS, assign_shards, est_cost  # noqa: E402


def group_key(cfg):
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in PER_RUN_KEYS and k not in ("name",)},
                      sort_keys=True)


def save(rec, path):
    with gzip.open(path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def run_chunk(chunk, dev, path_of, amp=None):
    """Train one stack and save its records. On failure a stack of more than one run is split in two and each half
    retried (an out-of-memory error is the usual cause; the fp32 fallback needs *more* memory, so it is used only
    for a single run)."""
    names = [c["name"] for c in chunk]
    logf = lambda s: print(f"[{names[0]}+{len(chunk)-1}] {s}", flush=True)
    try:
        recs = run_stack(chunk, dev, log=logf, amp=amp)
    except Exception:
        traceback.print_exc()
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        if len(chunk) > 1:
            h = len(chunk) // 2
            print(f"=== splitting the stack of {len(chunk)} into {h} + {len(chunk) - h} and retrying", flush=True)
            run_chunk(chunk[:h], dev, path_of, amp)
            run_chunk(chunk[h:], dev, path_of, amp)
            return
        if amp is not False:
            print("=== retrying this run in fp32 (no autocast)", flush=True)
            run_chunk(chunk, dev, path_of, amp=False)
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
    ap.add_argument("--max-stack", type=int, default=10)   # 10 runs at T = 10 ≈ 7 GB of activations on a T4; T = 24 needs ≤ 5
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--shard-mode", choices=("balanced", "modulo"), default="balanced")
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
    if args.shard_mode == "balanced":
        per_shard, loads = assign_shards(chunks, k)
        mine = per_shard[i]
        est = f", est. {loads[i]/3600:.1f} h this shard (all shards: {', '.join(f'{l/3600:.1f}' for l in loads)} h)"
    else:
        mine = chunks[i::k]
        est = f", est. {sum(est_cost(c) for ch in mine for c in ch)/3600:.1f} h this shard"
    print(f"{len(grid)} runs, {len(todo)} to do, {len(chunks)} stacks, {len(mine)} in shard {args.shard} on {dev}{est}",
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
                run_chunk(chunk, dev, path_of)
            print(f"=== done in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED stack {names[0]} (no file written; a later session retries it)", flush=True)
        done += 1
    print(f"shard {args.shard}: {done} stacks this session")


if __name__ == "__main__":
    main()
