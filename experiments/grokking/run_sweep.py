"""Run a grid of grokking runs in parallel worker processes, resumably.

Usage:
    PYTHONPATH=src python experiments/grokking/run_sweep.py --grid experiments/grokking/grids/pilot.json \
        --out results/pilot --workers 4

A grid file is JSON: {"base": {RunConfig fields...}, "vary": {field: [values, ...], ...}}.
The Cartesian product of `vary` is run on top of `base`. Finished runs (an existing <run_id>.json)
are skipped, so the command can be re-run after an interruption.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from topology_design import grokking as gk  # noqa: E402


def expand(grid: dict) -> list[gk.RunConfig]:
    if "grids" in grid:  # a list of sub-grids; duplicates (same run_id) are dropped
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
        if grid.get("tie_data_seed"):  # paired design: data seed index = model seed
            kw["data_seed"] = kw["model_seed"]
        cfgs.append(gk.RunConfig(**kw))
    return cfgs


def _work_batch(args):
    cfgs, out, device = args
    from topology_design import grokking_batched as gb
    todo = [c for c in cfgs if not os.path.exists(os.path.join(out, c.run_id + ".json"))]
    if not todo:
        return [(c.run_id, "skip", 0.0) for c in cfgs]
    try:
        results = gb.train_batch(todo, device=device)
    except Exception:
        with open(os.path.join(out, todo[0].run_id + ".batch.err"), "w") as f:
            f.write(traceback.format_exc())
        return [(c.run_id, "ERROR", 0.0) for c in todo]
    msgs = []
    for res in results:
        path = os.path.join(out, res["run_id"] + ".json")
        gk.save(res, path + ".tmp")
        os.replace(path + ".tmp", path)
        s = res["summary"]
        msgs.append((res["run_id"], f"grok={s['t_grok']} fit={s['t_fit']} test={s['final_test_acc']:.3f}", res["seconds"]))
    return msgs


def make_batches(cfgs, size):
    from topology_design import grokking_batched as gb
    groups = {}
    for c in cfgs:
        groups.setdefault(gb.struct_key(c), []).append(c)
    batches = []
    for g in groups.values():
        batches += [g[i : i + size] for i in range(0, len(g), size)]
    return batches


def _work(args):
    cfg, out = args
    path = os.path.join(out, cfg.run_id + ".json")
    if os.path.exists(path):
        return cfg.run_id, "skip", 0.0
    try:
        res = gk.train_run(cfg)
        tmp = path + ".tmp"
        gk.save(res, tmp)
        os.replace(tmp, path)
        s = res["summary"]
        return cfg.run_id, f"grok={s['t_grok']} fit={s['t_fit']} test={s['final_test_acc']:.3f}", res["seconds"]
    except Exception:  # keep the sweep going; record the failure
        with open(path + ".err", "w") as f:
            f.write(traceback.format_exc())
        return cfg.run_id, "ERROR", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--dry", action="store_true", help="list runs and exit")
    ap.add_argument("--batch", type=int, default=1, help="runs per stacked batch (1 = unbatched trainer)")
    ap.add_argument("--device", default="cpu", help="cpu or mps (batched trainer only)")
    args = ap.parse_args()
    with open(args.grid) as f:
        cfgs = expand(json.load(f))
    os.makedirs(args.out, exist_ok=True)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(args.out, c.run_id + ".json"))]
    print(f"{len(cfgs)} runs in grid, {len(todo)} to do, {args.workers} workers", flush=True)
    if args.dry:
        for c in todo:
            print(c.run_id)
        return
    t0 = time.time()
    if args.batch > 1 or args.device != "cpu":
        batches = make_batches(todo, args.batch)
        print(f"{len(batches)} batches of up to {args.batch} on {args.device}", flush=True)
        i = 0
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for msgs in pool.imap_unordered(_work_batch, [(b, args.out, args.device) for b in batches]):
                for rid, msg, secs in msgs:
                    i += 1
                    print(f"[{i}/{len(todo)}] {rid}: {msg} (batch {secs:.0f}s; elapsed {time.time() - t0:.0f}s)", flush=True)
        return
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (rid, msg, secs) in enumerate(pool.imap_unordered(_work, [(c, args.out) for c in todo]), 1):
            print(f"[{i}/{len(todo)}] {rid}: {msg} ({secs:.0f}s; elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
