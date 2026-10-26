# Independent minimal-compliant-length BFS over (state, freshness flags), reviewer 2.
# Usage: python lc_bfs.py PAT LMAX out.npz [LCHECK]   PAT like mul>inc.
# Stores l_c for every function reached compliantly at length <= LMAX. If LCHECK = LMAX+1 is given, additionally
# expands the last frontier once and records which functions appear at LMAX+1 (functions only, no states stored).
import numpy as np, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from isa_common import INS, apply, start, key98, key49
A, B = sys.argv[1].split(">"); LMAX = int(sys.argv[2]); out = sys.argv[3]
LCHECK = int(sys.argv[4]) if len(sys.argv) > 4 else None

def allowed(i, F):  # F (n,2) uint8 freshness
    o, d, a, b = i
    if o != B: return np.ones(len(F), bool)
    ok = F[:, a] == 0
    if b is not None: ok &= F[:, b] == 0
    return ok

def k_sf(S, F):
    return key98(np.concatenate([S.reshape(len(S), 98), F.astype(np.int8)], 1))

S = start.copy(); F = np.zeros((1, 2), np.uint8)
seen = np.sort(k_sf(S, F)); fT = [S[:, 0]]; fL = [np.zeros(1, np.int8)]; t0 = time.time()
for L in range(1, LMAX + 1):
    cs, cf, ck = [], [], []
    for i in INS:
        ok = allowed(i, F)
        if not ok.any(): continue
        nS = apply(S[ok], i); nF = F[ok].copy(); nF[:, i[1]] = 1 if i[0] == A else 0
        k = k_sf(nS, nF); pos = np.searchsorted(seen, k); pos[pos >= len(seen)] = 0; new = seen[pos] != k
        cs.append(nS[new]); cf.append(nF[new]); ck.append(k[new])
    k = np.concatenate(ck); uk, fi = np.unique(k, return_index=True)
    S = np.concatenate(cs)[fi]; F = np.concatenate(cf)[fi]; seen = np.sort(np.concatenate([seen, uk]))
    fT.append(S[:, 0]); fL.append(np.full(len(S), L, np.int8))
    print(f"{A}>{B} L={L} (state,flag) {len(S)} ({time.time()-t0:.0f}s)", flush=True)
T = np.concatenate(fT); Lc = np.concatenate(fL); fk = key49(T)
o = np.lexsort((Lc, fk)); fk, T, Lc = fk[o], T[o], Lc[o]; keep = np.r_[True, fk[1:] != fk[:-1]]
extra = np.zeros(0, fk.dtype)
if LCHECK:
    ex = []
    for i in INS:
        ok = allowed(i, F)
        if not ok.any(): continue
        u = np.unique(key49(apply(S[ok], i)[:, 0]))
        ex.append(u[~np.isin(u, fk[keep])])
    extra = np.unique(np.concatenate(ex))
np.savez(out, fk=fk[keep], lc=Lc[keep], extra=extra)
print("functions reached compliantly by l_c:", np.bincount(Lc[keep]).tolist(), "extra layer functions:", len(extra))
