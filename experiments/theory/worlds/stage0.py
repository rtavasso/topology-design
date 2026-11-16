"""Stage 0 of experiments/theory/worlds/DESIGN.md (revision 2): the two monoliths, no streams. EXPLORATORY.
Predictions R0a-R0e are in DESIGN.md section 7, committed before this was run."""
import json
import os
import statistics
import sys
from worlds import run

HERE = os.path.dirname(os.path.abspath(__file__))
med = statistics.median
CK = (1000, 2000, 3000)


def cfg(task, arm, cover, split, **kw):
    return dict(task=task, arm=arm, cover=cover, split=split, tseed=10 + split % 3, **kw)


def cell(v):
    return f"{med(v):.2g} [{min(v):.2g}, {max(v):.2g}]"


def regime(r, task):
    """Where a run's held-out error sits against the exact references (DESIGN.md section 2)."""
    h, ref = r["held"], r["ref"]
    bar = 0.5 * (ref["blend"] if task == "select" else min(ref["allmean"], ref["one"]))
    if h <= 3 * r["seen"] and h <= bar:      # full: recombines, and is within 3x of its own error on seen worlds
        return "full"
    if task == "select":
        return ("garbage" if h >= ref["allmean"] else "beyond" if h <= 0.5 * ref["trans"] else "recombines" if h <= 0.5 * ref["blend"]
                else "partial" if h <= ref["blend"] else "worse-than-blend")
    return "garbage" if h >= min(ref["allmean"], ref["one"]) else "recombines" if h <= 0.5 * min(ref["allmean"], ref["one"]) else "partial"


def row(label, rs, task, control=False):
    if rs[0]["nheld"] == 0:
        return f"| {label} | {cell([r['seen'] for r in rs])} | - | - | - | - | - |"
    if control:                              # references are zero when the world is irrelevant: no regime label
        ok = sum(r["held"] <= max(1.5 * r["seen"], 1e-3) for r in rs)
        return f"| {label} | {cell([r['seen'] for r in rs])} | {cell([r['held'] for r in rs])} | - | held out within 1.5x of seen (or below 1e-3): {ok}/{len(rs)} | - | - |"
    reg = {}
    for r in rs:
        reg[regime(r, task)] = reg.get(regime(r, task), 0) + 1
    refs = "; ".join(f"{k} {med([r['ref'][k] for r in rs]):.2f}" for k in rs[0]["ref"] if k != "forced")
    return (f"| {label} | {cell([r['seen'] for r in rs])} | {cell([r['held'] for r in rs])} | {refs} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(reg.items()))} | {med([r['one_true'] for r in rs]):.2f} | {med([r['one_blend'] for r in rs]):.2f} |")


HEAD = "| run | seen | held out | references (median) | regime (runs) | one pass on held-out worlds: distance to the true update | distance to the blend |\n|---|---|---|---|---|---|---|"

if __name__ == "__main__":
    mode, task, arm = sys.argv[1], sys.argv[2], sys.argv[3]
    lrf = lambda W: os.path.join(HERE, f"stage0_lr_{task}_{arm}_{W}.json")
    best_lr = lambda W: (lambda d: float(min(d, key=d.get)))(json.load(open(lrf(W))))
    if mode == "rates":                      # rates <task> <arm> <width> [rate ...]
        W = int(sys.argv[4]); lrfile = lrf(W)
        print(HEAD, flush=True)
        best, rates = {}, [float(x) for x in sys.argv[5:]] or [0.003, 0.01, 0.03, 0.1]
        for lr in rates:
            rs = [run(cfg(task, arm, 9, s, lr=lr, W=W), s) for s in (100, 101, 102)]
            best[lr] = med([r["seen"] for r in rs])
            print(row(f"{task} {arm} width {W} rate {lr:g}", rs, task), flush=True)
        old = json.load(open(lrfile)) if os.path.exists(lrfile) else {}
        old.update({str(k): v for k, v in best.items()})
        json.dump(old, open(lrfile, "w"))
        print("\nbest by seen error:", min(old, key=old.get), flush=True)
    else:
        store, out = {}, os.path.join(HERE, f"stage0_{mode}_{task}_{arm}.json")
        print(HEAD, flush=True)
        if mode == "dial":
            plan = [(f"coverage {c}", dict(cover=c)) for c in ((7, 9, 11, 13, 15) if task == "select" else (7, 9, 15))]
        elif mode == "fit":
            plan = [("coverage 9, 8,000 steps", dict(cover=9, steps=8000))]
            if arm == "mono":
                plan += [("coverage 9, width 120", dict(cover=9, W=120)), ("coverage 9, width 120, 8,000 steps", dict(cover=9, W=120, steps=8000))]
        else:                                # control: all primitives identical, so the world is irrelevant
            plan = [("irrelevant world, coverage 9", dict(cover=9, same=True))]
        for label, kw in plan:
            c = kw.pop("cover")
            lr = best_lr(kw.get("W", 40))
            rs = [run(cfg(task, arm, c, s, lr=lr, ckpts=CK, **kw), s) for s in range(6 if mode != "control" else 3)]
            store[label] = rs
            print(row(f"{task} {arm} rate {lr:g}, {label}", rs, task, control=(mode == "control")), flush=True)
            json.dump(store, open(out, "w"))
