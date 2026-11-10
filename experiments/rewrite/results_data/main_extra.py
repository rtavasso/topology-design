"""Unscored companions to main_report.md, written after the verdicts were read (post hoc): (a) the curves at a few steps —
is the held-out success still rising, flat, or past a peak at the end; (b) sizes 6–7 broken down by size, by k and by failure
category. Usage: PYTHONPATH=src python3 experiments/rewrite/results_data/main_extra.py <results dir>"""
import sys, collections
sys.path.insert(0, "experiments/rewrite"); sys.path.insert(0, "src")
import analyze as A

runs = A.load_results(sys.argv[1])
ARMS = ["RT-canon", "RT-post", "RT-all", "EI16", "PF", "RT-examples", "RT-shuffled", "RT-absolute"]
HELD = A.PAIRS_HELD + A.TRIPLES + A.FULL
STEPS = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000]

print("## (a) Curves: masked | unmasked success on the curve subsets, seed-median, by step\n")
print("**Caveat (review 6, finding 3):** the last curve point (step 12,000) is on the full eval sets, the earlier ones on the first 100 instances per world, and \"final − max\" is negative for any flat noisy curve (expected −0.034 from binomial noise alone). These columns do not show a late decline; in 1,000-step window means the held-out curves are still rising at the end (`review/review6_scripts/r6_curves.py`).\n")
for label, worlds in (("training worlds", A.TRAIN_WORLDS), ("held-out worlds", HELD)):
    print(f"### {label}\n")
    print("| arm | " + " | ".join(str(s) for s in STEPS) + " | step of the max (per seed, masked) | final − max (per seed, masked) |")
    print("|---|" + "---|" * (len(STEPS) + 2))
    for arm in ARMS:
        rs = A.by_seed(runs, arm)
        cols = []
        for s in STEPS:
            vm, vu = [], []
            for r in rs:
                xs, ym = A.curve_series(r, "eval5", worlds, "ok_m"); _, yu = A.curve_series(r, "eval5", worlds, "ok_u")
                if s in xs:
                    vm.append(ym[xs.index(s)]); vu.append(yu[xs.index(s)])
            cols.append(f"{A._median(vm):.2f} \\| {A._median(vu):.2f}" if vm else "–")
        peaks, drops = [], []
        for r in rs:
            xs, ym = A.curve_series(r, "eval5", worlds, "ok_m")
            i = max(range(len(ym)), key=lambda j: ym[j]); peaks.append(xs[i]); drops.append(ym[-1] - ym[i])
        print(f"| {arm} | " + " | ".join(cols) + f" | {' '.join(str(p) for p in peaks)} | {' '.join(f'{d:+.2f}' for d in drops)} |")
    print()

print("## (b) Sizes 6–7 (eval7), training worlds\n")
print("| arm | size 6 masked | size 6 unmasked | size 7 masked | size 7 unmasked |")
print("|---|---|---|---|---|")
for arm in ["RT-canon", "RT-post", "RT-all", "RT-absolute"]:
    rs = A.by_seed(runs, arm)
    cells = []
    for size in (6, 7):
        for key in ("ok_m", "ok_u"):
            cells.append(A.summarize_seeds([A.rate(A.instances(r, "eval7", world=A.TRAIN_WORLDS, size=size), key) for r in rs]))
    n6 = len(A.instances(rs[0], "eval7", world=A.TRAIN_WORLDS, size=6)); n7 = len(A.instances(rs[0], "eval7", world=A.TRAIN_WORLDS, size=7))
    print(f"| {arm} (n {n6} / {n7}) | " + " | ".join(cells) + " |")
print()
print("| arm | k | n | sizes 6–7 masked | size ≤ 5 masked (same k, training worlds) | greedy (6–7) | lookahead-3 (6–7) |")
print("|---|---|---|---|---|---|---|")
for arm in ["RT-canon", "RT-all", "RT-absolute"]:
    rs = A.by_seed(runs, arm)
    for k in (1, 2, 3, 4):
        r7 = lambda r: A.instances(r, "eval7", world=A.TRAIN_WORLDS, k=k)
        r5 = lambda r: A.instances(r, "eval5", world=A.TRAIN_WORLDS, k=k)
        if not r7(rs[0]):
            continue
        print(f"| {arm} | {k} | {len(r7(rs[0]))} | {A.summarize_seeds([A.rate(r7(r)) for r in rs])} | {A.summarize_seeds([A.rate(r5(r)) for r in rs]) if r5(rs[0]) else '–'} | {A.ladder_rate(r7(rs[0]), 1):.2f} | {A.ladder_rate(r7(rs[0]), 3):.2f} |")
print()


def depth(t):                                    # nesting depth of a term written with parentheses
    d = m = 0
    for ch in str(t):
        if ch == "(":
            d += 1; m = max(m, d)
        elif ch == ")":
            d -= 1
    return m


print("Failure categories at sizes 6–7 by k (training worlds, pooled over seeds): masked | unmasked\n")
print("| arm | k | n (pooled) | masked | unmasked |")
print("|---|---|---|---|---|")
for arm in ["RT-canon", "RT-all", "RT-absolute"]:
    rs = A.by_seed(runs, arm)
    for k in (1, 2, 3, 4):
        cm, cu, n = collections.Counter(), collections.Counter(), 0
        for r in rs:
            for p_ in A.instances(r, "eval7", world=A.TRAIN_WORLDS, k=k):
                cm[p_["cat_m"]] += 1; cu[p_["cat_u"]] += 1; n += 1
        fm = " ".join(f"{c}:{v / n:.2f}" for c, v in cm.most_common()); fu = " ".join(f"{c}:{v / n:.2f}" for c, v in cu.most_common())
        print(f"| {arm} | {k} | {n} | {fm} | {fu} |")
print()
print("Masked success by the nesting depth of the deeper of t and s (the structural features are clamped at depth 4, the range seen at size ≤ 5): rate (n per seed)\n")
print("| arm | set | k | depth 1 | 2 | 3 | 4 | 5 | 6 |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in ["RT-canon", "RT-all"]:
    rs = A.by_seed(runs, arm)
    for tag, ks in (("eval5", (None,)), ("eval7", (1, 2, 3, 4))):
        for k in ks:
            bd = collections.defaultdict(lambda: [0, 0])
            for r in rs:
                rows = A.instances(r, tag, world=A.TRAIN_WORLDS) if k is None else A.instances(r, tag, world=A.TRAIN_WORLDS, k=k)
                for p_ in rows:
                    dd = max(depth(p_["t"]), depth(p_["s"])); bd[dd][0] += p_["ok_m"]; bd[dd][1] += 1
            cells = [f"{bd[d][0] / bd[d][1]:.2f} ({bd[d][1] // len(rs)})" if bd[d][1] else "–" for d in range(1, 7)]
            print(f"| {arm} | {'size ≤ 5' if tag == 'eval5' else 'sizes 6–7'} | {'all' if k is None else k} | " + " | ".join(cells) + " |")
