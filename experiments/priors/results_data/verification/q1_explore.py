"""Semi-grokking, convergence, width seed-level, dropped-pair accuracy, rule fraction / weight norms at the shrink,
and the scratch stop-rule asymmetry. usage: python q1_explore.py ROOT"""
import sys, math, statistics as stt
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from q1_verify import load, ungrok_rows, scratch_rows, n_floor, n_crit, fmtc

U = ungrok_rows(load("kaggle_q1u"))
Sx = scratch_rows(load("kaggle_q1s"))


def wn_total(wn):
    if wn is None:
        return None
    if isinstance(wn, dict):
        vals = list(wn.values())
    else:
        vals = wn
    try:
        return math.sqrt(sum(float(v) ** 2 for v in vals))
    except TypeError:
        return None


print("weight_norms format example:", type(U[0]["wn_shrink"]), str(U[0]["wn_shrink"])[:200])

# ---- 3. semi-grokking + convergence + slope
print("\n## Semi-grokked runs (final test 0.1-0.9) and convergence")
print("cell level seed | final | mean last2500 | mean 15k earlier | diff | unconv | kept")
for x in sorted(U, key=lambda x: (x["g"], x["w"], x["n"], x["seed"])):
    if 0.1 <= x["final"] <= 0.9 or x["unconv"]:
        print(f" {x['g']} {x['w']} {x['n']} {x['seed']} | {x['final']:.3f} | {x['m_end']:.3f} | {x['m_early']:.3f} | "
              f"{x['m_end']-x['m_early']:+.3f} | {x['unconv']} | {x['keep']}")

print("\n## Trajectory of post-shrink test acc (mean over 2,500-step windows ending at +5k,+10k,...,+30k/40k), near-threshold levels")
for x in sorted(U, key=lambda x: (x["g"], x["w"], x["n"], x["seed"])):
    nc = n_crit(U, x["g"], x["w"])[0][0]
    levels = sorted({y["n"] for y in U})
    i = levels.index(nc)
    near = levels[max(0, i - 2): i + 2]
    if x["n"] not in near:
        continue
    t0, end = x["t_shrink"], x["end"]
    ws = []
    for e in range(t0 + 5000, end + 1, 5000):
        a = [r["test_acc"] for r in x["traj"] if e - 2500 < r["step"] <= e]
        ws.append(stt.mean(a))
    print(f" {x['g']:4s} {x['w']:6s} {x['n']:5d} s{x['seed']} keep={int(x['keep'])} min_last2k="
          f"{min(r['test_acc'] for r in x['traj'] if r['step'] >= end - 2000):.3f} :", " ".join(f"{v:.3f}" for v in ws))

# ---- 4. width seed-level
print("\n## Width: per seed, per level, min test acc in last 2000 steps (sym)")
for w in ("narrow", "base", "wide"):
    for L in (435, 522, 696, 870, 1044, 1218):
        xs = sorted([x for x in U if x["g"] == "sym" and x["w"] == w and x["n"] == L], key=lambda x: x["seed"])
        print(f" {w:6s} {L:5d}:", " ".join(
            f"s{x['seed']}={min(r['test_acc'] for r in x['traj'] if r['step'] >= x['end'] - 2000):.3f}" for x in xs))

# ---- 5. dropped-pair accuracy vs test accuracy
print("\n## Dropped-pair vs test accuracy after the shrink (median over runs of |drop - test| at end, and at +2000)")
dd = defaultdict(list)
for x in U:
    end, t0 = x["end"], x["t_shrink"]
    fin = [r for r in x["traj"] if r["step"] == x["traj"][-1]["step"]][0]
    early = min((r for r in x["traj"] if r["step"] >= t0 + 2000), key=lambda r: r["step"])
    post = [r for r in x["traj"] if r["step"] > t0 and r.get("drop_acc") is not None]
    gap_max = max(r["drop_acc"] - r["test_acc"] for r in post) if post else None
    dd[(x["g"], x["w"], x["n"])].append((fin.get("drop_acc"), fin["test_acc"], early.get("drop_acc"), early["test_acc"], gap_max))
print("cell level | final drop_acc (per seed) | final test_acc | max over post-shrink of (drop - test)")
for k in sorted(dd):
    v = dd[k]
    if all(a is None for a, *_ in v):
        continue
    print(f" {k} | " + ",".join(f"{a:.3f}" for a, *_ in v) + " | " + ",".join(f"{b:.3f}" for _, b, *_ in v) +
          " | " + ",".join(f"{g:+.3f}" for *_, g in v))

# rule fraction and weight norm at the shrink vs kept, near threshold
print("\n## At the shrink: steps since t_grok, test acc, rule_frac_test, total weight norm; vs kept (near-threshold levels)")
for g, w in [("sym", "narrow"), ("sym", "base"), ("sym", "wide"), ("asym", "base")]:
    nc = n_crit(U, g, w)[0][0]
    levels = sorted({y["n"] for y in U})
    i = levels.index(nc)
    near = levels[max(0, i - 1): i + 2]
    per_seed = {}
    for x in U:
        if x["g"] == g and x["w"] == w:
            per_seed[x["seed"]] = (x["t_shrink"], x["t_grok1"], x["rf_shrink"], wn_total(x["wn_shrink"]))
    print(f" {g} {w} (near levels {near}):")
    for sd, (ts, tg, rf, wn) in sorted(per_seed.items()):
        kept = {L: int(x["keep"]) for x in U for L in near if x["g"] == g and x["w"] == w and x["seed"] == sd and x["n"] == L}
        print(f"   seed {sd}: shrink {ts} (t_grok {tg}), rule_frac {rf:.3f}, |W| {wn if wn is None else round(wn,1)}, kept {kept}")

# ---- scratch stop-rule asymmetry: would scratch successes pass an "end-of-run" criterion?
print("\n## Scratch: sustained successes that dipped below 0.95 after their sustained window (within the extra steps)")
cnt = defaultdict(lambda: [0, 0])
for x in Sx:
    x["end_ok"] = all(r["test_acc"] >= 0.95 for r in [] )
for r_ in load("kaggle_q1s"):
    pass
for x in Sx:
    pass
# recompute directly
runs = load("kaggle_q1s")
from q1_verify import sustained
endok = []
for r in runs:
    c, tj = r["config"], r["trajectory"]
    t_s = sustained(tj, budget=c["steps"])
    last = tj[-1]["step"]
    end_ok = all(e["test_acc"] >= 0.95 for e in tj if e["step"] >= last - 2000)
    dip = t_s is not None and any(e["test_acc"] < 0.95 for e in tj if e["step"] > t_s + 2000)
    g, w = c["tag"].split("-")[1:]
    cnt[(g, w)][0] += dip
    cnt[(g, w)][1] += t_s is not None
    endok.append(dict(g=g, w=w, lr=c["lr"], n=int(round(c["train_frac"] * 3481)), seed=c["model_seed"],
                      succ=t_s is not None, end_ok=(t_s is not None and end_ok), excluded=False))
print(" dips/successes:", dict(cnt))
for g, w in [("sym", "narrow"), ("sym", "base"), ("sym", "wide"), ("asym", "base")]:
    nf = n_floor(endok, g, w, key="end_ok")
    print(f" {g} {w}: n_floor requiring also >=0.95 over the run's last 2000 steps: {fmtc(nf[0])} best {nf[3]}; counts {nf[2]}")
