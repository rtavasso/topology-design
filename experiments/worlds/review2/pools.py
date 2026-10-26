# Pool sizes per length, hold-out arithmetic, and how often the "held-out" pattern is used by minimal programs of
# pool functions (reviewer 2). Needs fn5.npz (isa_counts.py 5) and lcA5.npz / lcB5.npz (lc_bfs.py).
import numpy as np, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from isa_common import key49
d = np.load("fn5.npz"); T, L, nmin, ncA, ncB = d["T"], d["L"], d["nmin"], d["ncA"], d["ncB"]
fk = key49(T)
nz = L >= 1
print("functions by l (1..5):", [int((L == l).sum()) for l in range(1, 6)])
for nm, f, nc in (("A mul>inc", "lcA5.npz", ncA), ("B inc>mul", "lcB5.npz", ncB)):
    e = np.load(f); pos = np.searchsorted(e["fk"], fk); pos[pos >= len(e["fk"])] = 0
    hit = e["fk"][pos] == fk; lc = np.where(hit, e["lc"][pos], 99)
    pool = nz & (lc <= 5)
    print(f"\n{nm}: pool {pool.sum()}")
    for l in range(1, 6):
        s = L == l; ps = pool & s
        some = ps & (nc < nmin)        # at least one minimal program uses P
        allp = ps & (nc == 0)          # every minimal program uses P (l < l_c <= 5)
        print(f"  l={l}: total {s.sum():6d} pool {ps.sum():6d} removed {(s & ~pool).sum():5d} ({(s & ~pool).sum()/s.sum():.1%}); "
              f"pool fns with >=1 P-using minimal program {some.sum():6d} ({some.sum()/max(1,ps.sum()):.1%}); "
              f"pool fns whose every minimal program uses P {allp.sum():5d}; share of minimal programs using P "
              f"{1 - nc[ps].sum()/nmin[ps].sum():.1%}")
    # training-mass view: equal weight per length over pool functions
    w = np.zeros(len(T))
    for l in range(1, 6):
        ps = pool & (L == l); w[ps] = 0.2 / ps.sum()
    print(f"  training mass on functions with >=1 P-using minimal program: {w[pool & (nc < nmin)].sum():.1%}; "
          f"on functions whose every minimal program uses P: {w[pool & (nc == 0)].sum():.1%}")
print("\nhold-out arithmetic (10% per length, rounded):")
for l in range(1, 6):
    n = int((L == l).sum()); print(f"  l={l}: n={n} hold={round(0.1*n)} remain={n-round(0.1*n)}")
n3 = sum(int((L == l).sum()) - round(0.1 * int((L == l).sum())) for l in (1, 2, 3)); print("  l<=3 remaining:", n3)
