"""Sensitivity of S (and the other thresholds) to criterion choices. usage: python q1_sensitivity.py ROOT"""
import sys, statistics as stt
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from q1_verify import load, ungrok_rows, scratch_rows, n_crit, n_floor, thresh, fmtc, sustained

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/topology-design"
U = ungrok_rows(load("kaggle_q1u"))
Sx = scratch_rows(load("kaggle_q1s"))
for x in U:
    x["lenient"] = x["med5"] >= 0.5
    x["final90"] = x["final"] >= 0.9

cells = [("sym", "narrow"), ("sym", "base"), ("sym", "wide"), ("asym", "base")]


def S_of(nc, nf):
    if nc[0] is None or nf[0] is None:
        return f"cens (nc {fmtc(nc)}, nf {fmtc(nf)})"
    s = nf[0] / nc[0]
    tag = ""
    if nc[1] or nf[1]:
        tag = f" [nc {fmtc(nc)}, nf {fmtc(nf)}]"
    return f"{s:.2f} ({nf[0]}/{nc[0]}){tag}"


variants = []
for g, w in cells:
    row = {}
    nc0 = n_crit(U, g, w)[0]
    nf0 = n_floor(Sx, g, w)[0]
    row["preregistered"] = S_of(nc0, nf0)
    # (a) mismatched/lenient pair
    row["(a) lenient both"] = S_of(n_crit(U, g, w, key="lenient")[0], n_floor(Sx, g, w, key="fc")[0])
    row["(a') lenient ungrok only"] = S_of(n_crit(U, g, w, key="lenient")[0], nf0)
    row["(a'') first-cross scratch only"] = S_of(nc0, n_floor(Sx, g, w, key="fc")[0])
    # (b) n_crit thresholds
    row["(b) n_crit >=3/4"] = S_of(n_crit(U, g, w, k_needed=3)[0], nf0)
    row["(b) n_crit >=1/4"] = S_of(n_crit(U, g, w, k_needed=1)[0], nf0)
    row["(b) n_crit 4/4"] = S_of(n_crit(U, g, w, k_needed=4)[0], nf0)
    # (c) other variants
    row["(c) n_floor 3/3"] = S_of(nc0, n_floor(Sx, g, w, k_needed=3)[0])
    row["(c) n_floor >=1/3"] = S_of(nc0, n_floor(Sx, g, w, k_needed=1)[0])
    base_lr = 0.002 if g == "sym" else 0.001
    sub = [x for x in Sx if x["lr"] == base_lr]
    row["(c) scratch at the ungrok lr only"] = S_of(nc0, n_floor(sub, g, w)[0])
    row["(c) n_floor at half budget"] = S_of(nc0, n_floor(Sx, g, w, key="succ_half")[0])
    row["(c) keep thresh 0.90 in last 2000"] = S_of(n_crit(U, g, w, key="keep90")[0], nf0)
    row["(c) ungrok final>=0.9"] = S_of(n_crit(U, g, w, key="final90")[0], nf0)
    # pooled lrs: >= half of all 12 scratch seeds (min over lr removed)
    lv = defaultdict(list)
    for x in Sx:
        if x["g"] == g and x["w"] == w:
            lv[x["n"]].append(x["succ"])
    row["(c) scratch pooled over lrs >= half"] = S_of(nc0, thresh(lv, frac_needed=0.5))
    # non-monotone: smallest level with >= half, ignoring larger-level failures
    lvu = defaultdict(list)
    for x in U:
        if x["g"] == g and x["w"] == w:
            lvu[x["n"]].append(x["keep"])
    smallest = min(L for L, v in lvu.items() if sum(v) >= 0.5 * len(v))
    row["(c) n_crit without 'every larger level'"] = S_of((smallest, None), nf0)
    variants.append(((g, w), row))

keys = list(variants[0][1].keys())
print("| variant | " + " | ".join(f"{g} {w}" for (g, w), _ in variants) + " |")
print("|---|" + "---|" * len(variants))
for k in keys:
    print(f"| {k} | " + " | ".join(r[k] for _, r in variants) + " |")

# Grid-resolution bracket for sym base: true thresholds lie between adjacent levels
print("\nGrid bracket: n_crit in (prev level, n_crit], n_floor in (prev scratch level, n_floor]")
ul = sorted({x["n"] for x in U})
for g, w in cells:
    nc = n_crit(U, g, w)[0][0]
    nf = n_floor(Sx, g, w)[0][0]
    sl = sorted({x["n"] for x in Sx if x["g"] == g and x["w"] == w})
    pc = ul[ul.index(nc) - 1]
    pf = sl[sl.index(nf) - 1] if sl.index(nf) > 0 else None
    lo = (pf / nc) if pf else None
    hi = nf / pc
    print(f" {g} {w}: n_crit in ({pc},{nc}], n_floor in ({pf},{nf}] -> S in ({lo if lo is None else round(lo,2)}, {hi:.2f})")

# lenient detail for sym base
print("\nsym base: per-level lenient ungrok counts and scratch first-cross per lr")
print(" ungrok lenient:", n_crit(U, "sym", "base", key="lenient")[1])
print(" ungrok med5 values per level:")
for L in sorted({x["n"] for x in U}):
    print("  ", L, [round(x["med5"], 3) for x in U if x["g"] == "sym" and x["w"] == "base" and x["n"] == L])
_, per, counts, best = n_floor(Sx, "sym", "base", key="fc")
print(" scratch first-cross counts:", counts)
# scratch runs that crossed but did not sustain / lost later
print("\nscratch runs that first-crossed but failed sustained criterion, or dipped after:")
for x in Sx:
    if x["fc"] and not x["succ"]:
        print("  crossed-not-sustained", x["g"], x["w"], x["n"], x["lr"], x["seed"], "fc", x["t_fc"], "final", round(x["final"], 3))
nl = sum(1 for x in Sx if x["succ"] and x["min_after"] is not None and x["min_after"] < 0.95)
print(" sustained successes that later dipped below 0.95 in the 5000 extra steps:", nl,
      "of", sum(x["succ"] for x in Sx))
