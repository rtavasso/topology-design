"""Build the rewrite carrier's dataset (DESIGN §1 Instances, §3, §9; review 1 B1/B2/M4/M6).

Training stream: the eight training worlds, size 1–5 at cap 5, layer sampling with K_WEIGHTS, rejection to binding ≥ 30 %
(multi-axiom worlds) and GP1 ≥ 30 % per world where feasible (rates fixed from a warm-up and recorded), n per world.
Evaluation sets (fixed): every one of the fifteen worlds × {size ≤ 5 at cap 5, size 6–7 at cap 7}, binding ≥ 50 % where
feasible, with the secondary per-decision strata; plus {A, C} at k = 7–8, size ≤ 5. PF targets on the first --pf-per-world
training instances of each world (size ≤ 5 only; the PF arm's stream).

Output: <out>/train.jsonl.gz, <out>/eval_<tag>.jsonl.gz, <out>/meta.json (rates, realized tables, summaries, build
provenance). Multiprocess; resumable per (set, world) file.

    PYTHONPATH=src python3 experiments/rewrite/build_dataset.py --out data/rewrite --n-train 10000 --n-eval 600 --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from topology_design.rewrite.data import K_WEIGHTS, accept, acceptance_rates, category, complete_instance, make_candidate, realized_table, summarize  # noqa: E402
from topology_design.rewrite.rules import WORLDS, world_from_name, world_name  # noqa: E402

TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]          # canonical names order the axioms A, C, I, D ({D, I} is "ID")
HELD_OUT = [world_name(W) for W in WORLDS if world_name(W) not in TRAIN_WORLDS]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _job(args):
    """One worker call: generate instances under one world until `need` accepted (or `tries` candidates), with the given
    rejection rates. Returns (accepted records, candidate category counts, n_none)."""
    (wname, seed, need, tries, sizes, cap, radius, k_set, secondary, pf_first, rates, multi) = args
    rng = random.Random(seed)
    W = world_from_name(wname)
    out, counts, n_none, n_cand = [], {}, 0, 0
    while len(out) < need and n_cand < tries:
        cand = make_candidate(rng, W, cap, sizes, radius=radius, k_set=k_set)
        n_cand += 1
        if cand is None:
            n_none += 1
            continue
        rec = cand["rec"]
        c = category(rec)
        counts[c] = counts.get(c, 0) + 1
        if c == "B" and rec["first"]["GP1"]:
            counts["BG"] = counts.get("BG", 0) + 1
        if rates is None:
            continue                                   # warm-up: categories only
        if accept(rec, rates, rng):
            out.append(complete_instance(cand, secondary=secondary, pf=len(out) < pf_first))
    return out, counts, n_none, n_cand


def build_world(pool, wname, seed, n, sizes, cap, radius, k_set, secondary, pf_per_world, target_binding, target_gp1,
                warmup, workers, chunk=50):
    """Warm-up (no rejection) to estimate the category rates, then rejection at the fixed rates until n accepted."""
    multi = 1 < len(wname) < 4                     # singletons bind rarely and the full world never: no binding target there
    tb = target_binding if multi else 0.0
    t0 = time.time()
    # warm-up
    jobs = [(wname, (seed << 20) + i, 10 ** 9, warmup // workers + 1, sizes, cap, radius, k_set, False, 0, None, multi)
            for i in range(workers)]
    counts = {}
    n_none = n_cand = 0
    for recs, c, nn, nc in pool.map(_job, jobs):
        for key, v in c.items():
            counts[key] = counts.get(key, 0) + v
        n_none += nn; n_cand += nc
    rates = acceptance_rates(counts, tb, target_gp1)
    rates.update({"warmup_counts": counts, "warmup_none": n_none, "warmup_candidates": n_cand})
    # main
    recs_all = []
    j = 0
    while len(recs_all) < n:
        need = n - len(recs_all)
        per = max(1, min(chunk, (need + workers - 1) // workers))
        jobs = []
        for i in range(workers):
            pf_first = 0
            if pf_per_world:
                pf_first = per            # PF on every record of these jobs until the per-world quota is met
            jobs.append((wname, (seed << 20) + 1000 + j * workers + i, per, 50 * per, sizes, cap, radius, k_set, secondary, pf_first, rates, multi))
        j += 1
        n_pf = sum(1 for r in recs_all if "pf" in r)
        for recs, c, nn, nc in pool.map(_job, jobs):
            for r in recs:
                if "pf" in r and n_pf >= pf_per_world:
                    del r["pf"]
                elif "pf" in r:
                    n_pf += 1
            recs_all.extend(recs)
        if j > 200:
            break
    recs_all = recs_all[:n]
    rates["seconds"] = time.time() - t0
    rates["accepted"] = len(recs_all)
    return recs_all, rates


def write_jsonl(path, records):
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rewrite"))
    ap.add_argument("--n-train", type=int, default=10000, help="training instances per training world")
    ap.add_argument("--n-eval", type=int, default=600, help="evaluation instances per world and size range")
    ap.add_argument("--pf-per-world", type=int, default=2500, help="training instances per world with PF targets")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--target-binding", type=float, default=0.30)
    ap.add_argument("--target-gp1", type=float, default=0.30)
    ap.add_argument("--eval-binding", type=float, default=0.50)
    ap.add_argument("--only", default=None, help="comma-separated set tags to build: train, eval5, eval7, evalk78")
    ap.add_argument("--worlds", default=None, help="comma-separated worlds (default: all for the set)")
    ap.add_argument("--smoke", action="store_true", help="tiny build for tests")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_eval, args.pf_per_world, args.warmup = 40, 20, 10, 40
    os.makedirs(args.out, exist_ok=True)
    only = set(args.only.split(",")) if args.only else {"train", "eval5", "eval7", "evalk78"}
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    meta_path = os.path.join(args.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.setdefault("builds", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": head, "args": vars(args)})
    meta["k_weights"] = K_WEIGHTS
    meta["train_worlds"] = TRAIN_WORLDS
    meta["held_out_worlds"] = HELD_OUT
    sets = []
    if "train" in only:
        sets.append(("train", TRAIN_WORLDS, range(1, 6), 5, 6, None, False, args.pf_per_world, args.n_train, args.target_binding, args.target_gp1))
    if "eval5" in only:
        sets.append(("eval5", [world_name(W) for W in WORLDS], range(1, 6), 5, 6, None, True, 0, args.n_eval, args.eval_binding, 0.0))
    if "eval7" in only:
        sets.append(("eval7", [world_name(W) for W in WORLDS], range(6, 8), 7, 4, {1, 2, 3, 4}, True, 0, args.n_eval, args.eval_binding, 0.0))
    if "evalk78" in only:
        sets.append(("evalk78", ["AC"], range(1, 6), 5, 8, {7, 8}, True, 0, args.n_eval, 0.0, 0.0))
    worlds_filter = set(args.worlds.split(",")) if args.worlds else None
    with mp.Pool(args.workers) as pool:
        for tag, worlds, sizes, cap, radius, k_set, secondary, pf_per_world, n, tb, tg in sets:
            if worlds_filter:
                worlds = [w for w in [world_name(W) for W in WORLDS] if w in worlds_filter] if tag == "train" else [w for w in worlds if w in worlds_filter]
            for wname in worlds:
                wi = [world_name(W) for W in WORLDS].index(wname)
                path = os.path.join(args.out, f"{tag}_{wname}.jsonl.gz")
                if os.path.exists(path) and not args.smoke:
                    print(f"[{tag} {wname}] exists, skipping", flush=True)
                    continue
                seed = args.seed * 100000 + (["train", "eval5", "eval7", "evalk78"].index(tag) + 1) * 1000 + wi
                recs, rates = build_world(pool, wname, seed, n, list(sizes), cap, radius, k_set, secondary, pf_per_world, tb, tg,
                                          args.warmup, args.workers)
                write_jsonl(path, recs)
                summ = summarize(recs).get(wname, {})
                meta.setdefault("rates", {}).setdefault(tag, {})[wname] = rates
                meta.setdefault("summary", {}).setdefault(tag, {})[wname] = summ
                meta.setdefault("realized", {}).setdefault(tag, {})[wname] = realized_table(recs)
                json.dump(meta, open(meta_path, "w"), indent=1)
                print(f"[{tag} {wname}] n={len(recs)} binding={summ.get('binding', 0):.2f} GP1={summ.get('GP1', 0):.2f} "
                      f"k={ {k: round(v, 2) for k, v in summ.get('k', {}).items()} } ladder={ {d: round(v, 2) for d, v in summ.get('ladder', {}).items()} } "
                      f"feasible={rates['feasible']} q={rates['q_gp1']:.2f} p={rates['p_plain']:.2f} {rates['seconds']:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
