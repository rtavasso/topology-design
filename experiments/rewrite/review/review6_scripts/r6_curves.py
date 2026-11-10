"""Review 6, part C — the curves read point by point, on a like-for-like denominator.

    python3 experiments/rewrite/review/review6_scripts/r6_curves.py > /tmp/r6_curves.txt

The curve's points at steps 0..11750 are on the first 100 instances per world (1,500 for eval5); the point at 12,000 is
on the full sets (9,000). main_extra.py's "final − max" and its 12,000 column compare the two. Here the 12,000 point is
recomputed on the same first-100-per-world subset from the per-instance records (per_instance is stored in the order the
evaluation iterated: by world, file order), so every point has the same instances."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *

D = load()
M5 = D["meta"]["eval5"]
# first 100 per world, in stored order
sub, seen = [], {}
for i, m in enumerate(M5):
    seen[m["world"]] = seen.get(m["world"], 0) + 1
    if seen[m["world"]] <= 100:
        sub.append(i)
sub = np.array(sub)
def cells_of(ix):
    c = {}
    for i in ix:
        m = M5[i]; k = f"{m['world']}|{int(m['binding'])}|{int(m['GP1'])}|{m['k']}"; c[k] = c.get(k, 0) + 1
    return c
# check that the subset reproduces the curve's cell counts at an earlier step
r0 = D["runs"]["RT-canon_s0"]
print("subset cell counts equal the curve's (step 11750):", cells_of(sub) == {k: v["n"] for k, v in r0["curve"][-2]["eval5"].items()})
print("the stored final point equals the full-set cells:", cells_of(range(len(M5))) == {k: v["n"] for k, v in r0["curve"][-1]["eval5"].items()})

def series(run, W, rd="ok_m"):
    xs, ys = [], []
    for pt in run["curve"][:-1]:
        num = den = 0
        for ck, c in pt["eval5"].items():
            if ck.split("|")[0] in W:
                num += c[rd]; den += c["n"]
        xs.append(pt["step"]); ys.append(num / den)
    ix = np.array([i for i in sub if M5[i]["world"] in W])
    xs.append(run["curve"][-1]["step"]); ys.append(float(run["out"]["eval5"][rd][ix].mean()))
    full = np.array([i for i, m in enumerate(M5) if m["world"] in W])
    return xs, ys, float(run["out"]["eval5"][rd][full].mean()), len(ix)

for label, W in (("training worlds", TRAIN), ("held-out worlds", HELD)):
    print(f"\n## {label}: like-for-like curve (subset at every step, incl. 12,000)")
    for arm in ARMS:
        rows = []
        for r in seeds(D, arm):
            xs, ys, full, n = series(r, W)
            i = int(np.argmax(ys)); last5 = ys[-5:]; prev5 = ys[-10:-5]
            # noise scale: sd of point-to-point differences over the last 12 points / sqrt(2)
            dif = np.diff(ys[-13:]); sd = float(np.std(dif) / np.sqrt(2))
            rows.append((r["cfg"]["seed"], xs[i], ys[i], ys[-1], ys[-1] - ys[i], full, np.mean(last5) - np.mean(prev5), sd, np.mean(ys[-9:]) - np.mean(ys[-17:-9]), max(ys) - min(ys[-9:])))
        print(f"{arm} (subset n={n}; binomial SE at p=0.7 is {np.sqrt(0.7 * 0.3 / n):.3f})")
        for s, px, py, fin, drop, full, tr, sd, tr2, _ in rows:
            print(f"   s{s}: peak {py:.3f} @ {px}; final(subset) {fin:.3f}; final-max {drop:+.3f}; final(full set) {full:.3f}; subset-full {fin - full:+.3f}; mean(last 5 pts)-mean(prev 5) {tr:+.3f}; mean(10k-12k)-mean(8k-9.75k) {tr2:+.3f}; pt noise sd {sd:.3f}")
        print(f"   summary: final-max per seed (like-for-like) {' '.join(f'{x[4]:+.2f}' for x in rows)} | seeds whose peak is the last point: {sum(x[1] == 12000 for x in rows)}/{len(rows)} | median late trend (last 5 vs prev 5) {np.median([x[6] for x in rows]):+.3f}, seeds negative {sum(x[6] < 0 for x in rows)}/{len(rows)} | 2k-window trend median {np.median([x[8] for x in rows]):+.3f}, negative {sum(x[8] < 0 for x in rows)}/{len(rows)}")

# expected max-minus-final of a flat noisy curve: simulate a flat curve with binomial noise at n=700 over the last 17 points... (48 points)
rng = np.random.default_rng(0)
n = 700; p = 0.69
sim = rng.binomial(n, p, size=(20000, 25)) / n      # a plateau of 25 points (6,000 steps)
print(f"\nflat plateau, independent binomial noise n={n}, p={p}, 25 points: E[final - max] = {np.mean(sim[:, -1] - sim.max(1)):+.3f}; P(peak before the end) = {np.mean(sim.argmax(1) != 24):.2f}")

# unusual shapes: biggest single-step drops, collapses, runs still rising at the end
print("\n## events: largest point-to-point drops (masked, like-for-like), any world class")
ev = []
for name, r in D["runs"].items():
    for label, W in (("train", TRAIN), ("held", HELD)):
        xs, ys, _, _ = series(r, W)
        for a in range(4, len(ys) - 1):
            ev.append((ys[a + 1] - ys[a], name, label, xs[a + 1], ys[a], ys[a + 1]))
for e in sorted(ev)[:15]:
    print(f"   {e[1]} {e[2]} step {e[3]}: {e[4]:.3f} -> {e[5]:.3f} ({e[0]:+.3f})")
print("\n## runs still rising at the end (training worlds, masked): mean(last 5) - mean(prev 5) >= 0.01")
for name, r in D["runs"].items():
    xs, ys, _, _ = series(r, TRAIN)
    if np.mean(ys[-5:]) - np.mean(ys[-10:-5]) >= 0.01:
        print(f"   {name}: {np.mean(ys[-10:-5]):.3f} -> {np.mean(ys[-5:]):.3f} final {ys[-1]:.3f}")
print("\n## RT-absolute per seed, training-world curve at 2k/4k/8k/12k and loss; evalk78 final")
for r in seeds(D, "RT-absolute"):
    xs, ys, _, _ = series(r, TRAIN)
    print(f"   s{r['cfg']['seed']}: " + " ".join(f"{ys[xs.index(s)]:.2f}" for s in (1000, 2000, 4000, 8000, 12000)) + f" | loss@end {r['curve'][-1]['loss']:.3f} | k78 {r['out']['evalk78']['ok_m'].mean():.3f}")
print("\n## loss at the end per arm (mean of the last 250 steps)")
for arm in ARMS:
    print(f"   {arm}: " + " ".join(f"{r['curve'][-1]['loss']:.3f}" for r in seeds(D, arm)), "| NaN in losses:", sum(int(np.isnan(np.array(r['losses'], float)).sum()) for r in seeds(D, arm)))
print("\n## per held-out world: RT-canon like-for-like curve, seed-median at 4k/6k/8k/10k/12k and final-max per seed")
for w in HELD:
    cols = []; drops = []
    for r in seeds(D, "RT-canon"):
        xs, ys, _, _ = series(r, [w]); cols.append([ys[xs.index(s)] for s in (4000, 6000, 8000, 10000, 12000)]); drops.append(np.mean(ys[-9:]) - np.mean(ys[-17:-9]))
    print(f"   {w}: " + " ".join(f"{np.median([c[j] for c in cols]):.2f}" for j in range(5)) + f" | 2k-window trend per seed {' '.join(f'{d:+.2f}' for d in drops)}")
