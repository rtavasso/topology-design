"""Q2's signal on a denser coverage grid, and how its sampling error scales with the number of held-out instances.
Usage: PYTHONPATH=src python3 experiments/belief/power_coverage.py >> experiments/belief/power.md"""
import collections, math, random, sys
import numpy as np
sys.path.insert(0, "experiments/belief")
import power as P

g = np.random.default_rng(1)
CAR = sys.argv[1] if len(sys.argv) > 1 else "H8"
NINST = int(sys.argv[2]) if len(sys.argv) > 2 else 800
GRID = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (8, 16, 32, 64, 128)
nw = (1 << len(P.CARRIERS[CAR])) - 1
order = list(range(1, nw + 1))
random.Random(7).shuffle(order)
nh = (nw + 1) // 4
heldout, pool = set(order[:nh]), order[nh:]
trains = {N: set(pool[:N]) for N in GRID}
data, sec = P.gather(CAR, NINST, 2, heldout=heldout, trains=trains)
print(f"\n## Q2 on a denser coverage grid ({CAR}, {nw} worlds)\n")
print(f"{len(data)} held-out-world instances ({nh} held-out worlds), nested training sets; stand-in noise 0.12; bootstrap over worlds. "
      f"The last columns use the first 200 and all {len(data)} instances, to show how the error shrinks.\n")
print("| training worlds | generalising against memorising: TV > 0.05 / mean TV | beta at gam = 0 / 0.5 / 1 | sd of beta, 200 instances | sd of beta, all instances |")
print("|---|---|---|---|---|")
sg = None
for N in GRID:
    def groups(rows):
        byw = collections.defaultdict(list)
        for ins, tr in rows:
            byw[ins.true] += tr[N]
        return list(byw.values())
    allp = [r for _, tr in data for r in tr[N]]
    if sg is None:
        sg = P.calibrate([r["q"] for r in allp], 0.12, g)
    sep = np.array([P.tv(r["q"], r["qt"]) for r in allp])
    res = [P.stats(groups(data), "qt", "q", gam, sg, g) for gam in (0.0, 0.5, 1.0)]
    small = P.stats(groups(data[:200]), "qt", "q", 0.5, sg, g)
    print(f"| {N} | {np.mean(sep > 0.05):.0%} / {sep.mean():.3f} | " + " / ".join(f"{r[0]:.2f}" for r in res) + f" | {small[2]:.3f} | {res[1][2]:.3f} |", flush=True)
