"""Stage 2 of experiments/theory/worlds/DESIGN.md (revision 2): read x block, and the further rows. EXPLORATORY.

Predictions R2a-R2g and the pre-fixed readings (support / the read, not streams / gating, not grouping / null) are in
DESIGN.md sections 6-7, committed before stage 0. Re-stated after stage 0 without change of direction; one change of
plan, made on stage 0's evidence and before any stage-2 run: the coverages are 7 and 9 (the per-token monolith is
already within 3x of its seen error at 11 and 13, so nothing can differ there), and `mono` appears twice - at matched
parameters (width 40, where it under-fits: 0.066 on seen worlds against 0.024) and at matched fit (width 120, 8,000
steps: 0.022; about six times the parameters), coverage 9 only for the wide one.

Tuning: each arm's rate (and softmax or sigmoid scores for the summed-read streams arms) on splits 100-102 at
coverage 9, chosen by seen-world error; every configuration's held-out error is printed. Main: twelve splits (three
teachers), the model's seed equal to the split. Null: an arm against itself with another initialisation.
"""
import json
import os
import random
import statistics
import sys
from stage0 import cfg, cell, regime
from worlds import run

HERE = os.path.dirname(os.path.abspath(__file__))
med = statistics.median
GRID = {"streams_self": [dict(lr=l, score=s) for s in ("softmax", "sigmoid") for l in (0.003, 0.01, 0.03, 0.1)],
        "streams_joint": [dict(lr=l, score=s) for s in ("softmax", "sigmoid") for l in (0.003, 0.01, 0.03, 0.1)],
        "tok_streams": [dict(lr=l) for l in (0.003, 0.01, 0.03, 0.1)],
        "unit": [dict(lr=l) for l in (0.003, 0.01, 0.03, 0.1)],
        "oracle": [dict(lr=l) for l in (0.003, 0.01, 0.03, 0.1)]}
FIXED = {"select": {"mono": dict(lr=0.03), "tok_mono": dict(lr=0.01), "mono_wide": dict(lr=0.01, W=120, steps=8000)},
         "sum": {"mono": dict(lr=0.03), "tok_mono": dict(lr=0.1)}}                # from stage 0


def best_cfg(task, arm):
    base = arm.replace("null_", "").split("_wd")[0].split("_lr")[0]
    if base in FIXED[task]:
        c = dict(FIXED[task][base])
    else:
        c = dict(json.load(open(os.path.join(HERE, f"stage2_cfg_{task}_{base}.json")))["best"])
    if "_wd" in arm:
        c["wd"] = float(arm.split("_wd")[1])
    if "_lr" in arm:                         # another rate whose seen error is within 2x of the best (DESIGN.md section 5)
        c["lr"] = float(arm.split("_lr")[1])
    return ("mono" if base == "mono_wide" else base), c


def load(task, cover, arm):
    f = os.path.join(HERE, f"stage2_{task}_{cover}_{arm}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def boot(v, n=10000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(med([v[rnd.randrange(len(v))] for _ in v]) for _ in range(n))
    return ms[int(0.05 * n)], ms[int(0.95 * n)]


if __name__ == "__main__":
    mode, task = sys.argv[1], sys.argv[2]
    if mode == "tune":                       # tune <task> <arm> [extra rate ...]
        arm = sys.argv[3]
        f = os.path.join(HERE, f"stage2_cfg_{task}_{arm}.json")
        table = json.load(open(f))["table"] if os.path.exists(f) else []
        grid = GRID[arm] if len(sys.argv) == 4 else [dict(g, lr=float(x)) for x in sys.argv[4:] for g in GRID[arm] if g["lr"] == 0.1]
        print("| arm, configuration | seen | held out | regime (runs) |\n|---|---|---|---|", flush=True)
        for g in grid:
            rs = [run(cfg(task, arm, 9, s, **g), s) for s in (100, 101, 102)]
            reg = {}
            for r in rs:
                reg[regime(r, task)] = reg.get(regime(r, task), 0) + 1
            table.append(dict(cfg=g, seen=med([r["seen"] for r in rs]), held=med([r["held"] for r in rs])))
            print(f"| {task} {arm} {g} | {cell([r['seen'] for r in rs])} | {cell([r['held'] for r in rs])} | {', '.join(f'{k} {v}' for k, v in sorted(reg.items()))} |", flush=True)
            json.dump(dict(best=min(table, key=lambda t: t["seen"])["cfg"], table=table), open(f, "w"))
        print("\nbest by seen error:", min(table, key=lambda t: t["seen"])["cfg"], flush=True)
    elif mode == "main":                     # main <task> <cover> <arm> ...
        cover = int(sys.argv[3])
        for arm in sys.argv[4:]:
            a, c = best_cfg(task, arm)
            off = 1000 if arm.startswith("null_") else 0
            rs = [run(cfg(task, a, cover, s, **c), s + off) for s in range(12)]
            json.dump(dict(cfg=c, runs=rs), open(os.path.join(HERE, f"stage2_{task}_{cover}_{arm}.json"), "w"))
            reg = {}
            for r in rs:
                reg[regime(r, task)] = reg.get(regime(r, task), 0) + 1
            print(f"| {task}, coverage {cover}, {arm} {c} | {rs[0]['nparams']} | {cell([r['seen'] for r in rs])} | {cell([r['held'] for r in rs])} | "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(reg.items()))} | {med([r['one_true'] for r in rs]):.2f} | {med([r['one_blend'] for r in rs]):.2f} |", flush=True)
    else:                                    # report <task> <cover>
        cover = int(sys.argv[3])
        pairs = [("C1 streams under the per-token read", "tok_streams", "tok_mono"), ("C2 streams under the summed read", "streams_self", "mono"),
                 ("C2' joint-gate streams under the summed read", "streams_joint", "mono"), ("C3 the read, one block", "tok_mono", "mono"),
                 ("C4 the read, streams", "tok_streams", "streams_self"), ("gating without grouping against streams", "unit", "streams_self"),
                 ("null: tok_mono, another initialisation", "null_tok_mono", "tok_mono"), ("null: mono, another initialisation", "null_mono", "mono")]
        print("| contrast (A / B) | held-out error ratio: median of 12 paired [5 %, 95 % bootstrap] | splits where A is lower | same ratio on seen worlds |\n|---|---|---|---|")
        for label, A, B in pairs:
            a, b = load(task, cover, A), load(task, cover, B)
            if a and b:
                rh = [x["held"] / y["held"] for x, y in zip(a["runs"], b["runs"])]
                rs_ = [x["seen"] / y["seen"] for x, y in zip(a["runs"], b["runs"])]
                lo, hi = boot(rh)
                print(f"| {label}: {A} / {B} | {med(rh):.2f} [{lo:.2f}, {hi:.2f}] | {sum(x < 1 for x in rh)}/12 | {med(rs_):.2f} |")
