# Recompute the DESIGN §3 table rows for mul>inc and inc>mul from the independent BFS (reviewer 2), and cross-check
# l_c > l against the independent minimal-program DP (ncX == 0 <=> every minimal program uses P <=> l_c > l).
import numpy as np, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from isa_common import key49
d = np.load("fn6.npz"); T, L = d["T"], d["L"]; fk = key49(T)
for nm, f, nc in (("mul>inc", "lcA7.npz", d["ncA"]), ("inc>mul", "lcB7.npz", d["ncB"])):
    e = np.load(f); pos = np.searchsorted(e["fk"], fk); pos[pos >= len(e["fk"])] = 0
    hit = e["fk"][pos] == fk; lc = np.where(hit, e["lc"][pos], 99).astype(int)
    lc[(lc == 99) & np.isin(fk, e["extra"])] = 7
    nz = L >= 1
    pool = nz & (lc <= 5); test = nz & (lc > 7); pmin = nz & (L <= 5) & np.isin(lc, [6, 7])
    print(f"{nm}: pool {pool.sum()} test l<=5 {(test & (L <= 5)).sum()} test l=6 {(test & (L == 6)).sum()} "
          f"P-minimal {pmin.sum()} | l_c=6 among l=6 {(nz & (L == 6) & (lc == 6)).sum()}")
    agree = ((lc > L) == (nc == 0))[nz]
    print(f"   DP cross-check (l_c>l iff no compliant minimal program): agree on {agree.mean():.6f} of {nz.sum()}")
    # l=6 compliant comparison group size, and atom divergence quick look: share of P-test fns by l_c bucket
