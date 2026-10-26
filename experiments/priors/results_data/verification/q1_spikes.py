"""Are near-threshold 'failures' loss of the rule or transient single-eval dips? n_crit under spike-robust criteria,
and the fixed-window (ungrok) vs any-window (scratch) asymmetry. usage: python q1_spikes.py ROOT"""
import sys, statistics as stt
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from q1_verify import load, ungrok_rows, scratch_rows, n_crit, n_floor, fmtc, sustained

U = ungrok_rows(load("kaggle_q1u"))
Sx = scratch_rows(load("kaggle_q1s"))
for x in U:
    end, t0 = x["end"], x["t_shrink"]
    last = [r["test_acc"] for r in x["traj"] if end - 2000 <= r["step"] <= end]
    x["n_last"] = len(last)
    x["n_below"] = sum(a < 0.95 for a in last)
    x["mean2k"] = stt.mean(last) >= 0.95
    x["med2k"] = stt.median(last) >= 0.95
    x["le1"] = x["n_below"] <= 1
    # any sustained window (>=0.95 for 2000 steps) starting in the last 10k / 15k steps of the post-shrink phase
    for W in (5000, 10000, 15000):
        seg = [r for r in x["traj"] if r["step"] >= end - W]
        x[f"any{W}"] = sustained(seg) is not None

print("Near-threshold failures: evals < 0.95 among the last-2000-step evals")
for x in sorted(U, key=lambda x: (x["g"], x["w"], x["n"], x["seed"])):
    if not x["keep"] and x["mean2k"]:
        print(f" {x['g']} {x['w']} {x['n']} s{x['seed']}: {x['n_below']}/{x['n_last']} evals below, mean>=0.95")

cells = [("sym", "narrow"), ("sym", "base"), ("sym", "wide"), ("asym", "base")]
print("\n| ungrok criterion | " + " | ".join(f"{g} {w}" for g, w in cells) + " |")
print("|---|" + "---|" * 4)
nfs = {c: n_floor(Sx, *c)[0][0] for c in cells}
for key, lab in [("keep", "prereg (all evals >= .95, last 2k)"), ("le1", "<= 1 eval below .95 in last 2k"),
                 ("med2k", "median last 2k >= .95"), ("mean2k", "mean last 2k >= .95"),
                 ("any5000", "sustained 2k window within last 5k"), ("any10000", "sustained 2k window within last 10k"),
                 ("any15000", "sustained 2k window within last 15k")]:
    out = []
    for c in cells:
        (nc, cnt) = n_crit(U, *c, key=key)
        s = f"{nfs[c]/nc[0]:.2f}" if nc[0] else "cens"
        out.append(f"n_crit {fmtc(nc)}, S {s} [" + ",".join(f"{k}" for L, (k, n) in cnt.items() if L >= 435) + "]")
    print(f"| {lab} | " + " | ".join(out) + " |")
print("(bracketed: kept counts at 435,522,696,870,1044,1218)")
# P4 under each

print("\nP3 / P4 under ungrok criterion variants (scratch side preregistered)")
for key in ("keep", "le1", "med2k", "mean2k", "any10000", "lenient"):
    if key == "lenient":
        for x in U: x["lenient"] = x["med5"] >= 0.5
    nc = {c: n_crit(U, *c, key=key)[0][0] for c in cells}
    rc = nc[("asym", "base")] / nc[("sym", "base")]
    rf = nfs[("asym", "base")] / nfs[("sym", "base")]
    p3 = "supported" if rc < rf else ("tie" if abs(rc - rf) < 1e-9 else "falsified")
    nn, nb, nw = nc[("sym", "narrow")], nc[("sym", "base")], nc[("sym", "wide")]
    p4 = "supported" if (nw <= nb <= nn and (nw < nb or nb < nn)) else "falsified"
    print(f" {key:9s}: n_crit asym/sym {nc[('asym','base')]}/{nb} = {rc:.2f} vs floors {rf:.2f} -> P3 {p3}; "
          f"narrow {nn}, base {nb}, wide {nw} -> P4 {p4}")
