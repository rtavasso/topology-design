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

# relative cost of a run per step, RT-canon = 1, measured in the pilot (results_data/pilot_report.md §3: a 12,000-step run with
# curves every 250 projected from the per-step and per-evaluation times; EI16's step is its beam search)
COST = {"EI16": 6.9, "RT-examples": 2.5, "PF": 1.3, "RT-post": 1.1}
SECONDS_PER_UNIT = 0.105                                       # RT-canon: 21 min per 12,000 steps on a T4 with two shards (pilot)


def est_cost(cfg):
    return int(cfg.get("steps", 10000)) * COST.get(cfg["arm"], 1.0)


def fits(cfg, left_s, rate=SECONDS_PER_UNIT, margin=1.15):
    """Whether a run can finish inside what is left of the session: a Kaggle session that runs out of time saves nothing,
    so a shard stops starting runs it cannot finish and the kernel ends normally with what it has (the launcher's resume
    then uploads those results and a second session runs the rest). left_s None = no budget."""
    return left_s is None or est_cost(cfg) * rate * margin <= left_s


HEAVY = ("PF",)        # arms whose training stream is several GB in memory (the PF targets): kept in one shard, so never two at once


def assign_shards(cfgs, k):
    """Longest first onto the lightest shard; the memory-heavy arms all go to the shard the first of them landed in (a
    shard process holds about 3 GB, a PF one about 5.5 GB; three PF runs side by side would crowd a 29 GB session)."""
    order = sorted(range(len(cfgs)), key=lambda i: (-est_cost(cfgs[i]), cfgs[i].get("name", "")))
    out, loads = [[] for _ in range(k)], [0.0] * k
    heavy_shard = None
    for i in order:
        j = min(range(k), key=lambda j: (loads[j], j))
        if cfgs[i]["arm"] in HEAVY:
            heavy_shard = j if heavy_shard is None else heavy_shard
            j = heavy_shard
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
    ap.add_argument("--budget-hours", type=float, default=None, help="session budget: a run that cannot finish inside it is not started")
    ap.add_argument("--started", type=float, default=None, help="epoch seconds at which the session started (default: now)")
    args = ap.parse_args()
    started = args.started or time.time()
    cfgs = json.load(open(args.grid))
    i, k = (int(x) for x in args.shard.split("/"))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // k))      # k shard processes share the machine's cores
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(res_dir, c["name"] + ".json.gz"))]
    shards, loads = assign_shards(todo, k)
    mine = shards[i]
    print(f"shard {i}/{k}: {len(mine)} of {len(todo)} pending runs ({len(cfgs)} in grid), est. load {loads[i]:.0f}; device {args.device}", flush=True)
    sets = T.eval_sets_for({}, args.data)
    print({tag: {w: len(rs) for w, rs in s.items()} for tag, s in sets.items()}, flush=True)
    rate, skipped = SECONDS_PER_UNIT, []
    for cfg in mine:
        path = os.path.join(res_dir, cfg["name"] + ".json.gz")
        if os.path.exists(path):
            continue
        left = None if args.budget_hours is None else args.budget_hours * 3600 - (time.time() - started)
        if not fits(cfg, left, rate):
            skipped.append(cfg["name"])
            print(f"budget: not starting {cfg['name']} (needs about {est_cost(cfg) * rate / 60:.0f} min, {left / 60:.0f} min left)", flush=True)
            continue
        t0 = time.time()
        try:
            rec = T.run(cfg, args.data, device=args.device, log=lambda *a: print(*a, flush=True), sets=sets)
            rec["shard"] = args.shard
            save(rec, path)
            rate = max(rate, (time.time() - t0) / est_cost(cfg))      # the slowest pace seen in this shard, never faster than the pilot's
            print(f"done {cfg['name']} in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAILED {cfg['name']}", flush=True)
    print(f"shard done; {len(skipped)} runs left for a resume {skipped}" if skipped else "shard done", flush=True)


if __name__ == "__main__":
    main()
