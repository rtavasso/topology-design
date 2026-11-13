"""Recompute the headline numbers of DESIGN §4 / prelim_info.md (H8, rho=0.5, k in {2,3,4}, seed 0)."""
import random, sys, time, pickle
import numpy as np
from core import *
from reftrace import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
order = sys.argv[3] if len(sys.argv) > 3 else "prelim"
rng = random.Random(seed)
rows = []
allrecs = []
t0 = time.time()
for _ in range(n):
    ins = Inst(rng)
    r = dict(k=ins.k, n=ins.n, ncons=int(ins.cons.sum()), Hw=H2(ins.post0), size=int(WSIZE[ins.true]))
    q0 = ins.mix(0, ins.wmask(0, ins.B))
    r["H_hidden"] = H2(q0)
    wm = np.zeros(NW, dtype=bool); wm[ins.true] = True
    r["H_known"] = H2(ins.mix(0, wm))
    e0, e1 = ins.estart[0], ins.estart[1]
    r["rej1"] = float(q0[~HAS[ins.true, ins.er[e0:e1]]].sum())
    for pol in ("oracle", "q", "mask"):
        recs, summ = run(ins, pol, order=order, extra=(pol == "q"))
        r[pol] = summ
        if pol == "q":
            for d in recs:
                d["inst"] = len(rows)
            allrecs += recs
            r["n_dec"] = len(recs)
    rows.append(r)
m = lambda f: float(np.mean([f(r) for r in rows]))
print(f"n={n} seed={seed} order={order}  {(time.time()-t0)/n:.2f}s/inst")
print(f"world entropy {m(lambda r: r['Hw']):.3f} bits; consistent worlds {m(lambda r: r['ncons']):.1f}; true size {m(lambda r: r['size']):.2f}")
print(f"first-step entropy known/hidden {m(lambda r: r['H_known']):.3f} / {m(lambda r: r['H_hidden']):.3f}; P(first rejected) {m(lambda r: r['rej1']):.3f}")
for pol in ("oracle", "q", "mask"):
    print(f"{pol}: solved {m(lambda r: r[pol]['ok']):.3f} calls {m(lambda r: r[pol]['calls']):.2f} rej {m(lambda r: r[pol]['rej']):.2f}")
post = [d for d in allrecs if d["calls"] > 0 and d["m"] is not None]
dropped = sum(1 for d in allrecs if d["calls"] > 0 and d["m"] is None)
sep = np.array([tv(d["q"], d["m"]) for d in post])
print(f"post-verdict decisions {len(post)} ({len(post)/n:.2f}/inst), dropped because mask belief empty {dropped}; "
      f"TV>0.05: {np.mean(sep>0.05):.3f}; >0.10 {np.mean(sep>0.10):.3f}; >0.20 {np.mean(sep>0.20):.3f}; mean TV {sep.mean():.4f}; mean on informative {sep[sep>0.05].mean():.3f}")
print(f"ball nodes {m(lambda r: r['n']):.0f}")
pickle.dump(dict(rows=rows, recs=allrecs), open(f"trace_{order}_{seed}_{n}.pkl", "wb"))
