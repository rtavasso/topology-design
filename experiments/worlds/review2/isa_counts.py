# Independent enumeration (reviewer 2). BFS over full register states to depth LMAX; for every state at its BFS depth:
# number of minimal programs reaching it, and for patterns A=mul>inc, B=inc>mul the number of *compliant* minimal
# programs, split by freshness flags (2 bits). Function level: l, n_min, n_min_compliant_A, n_min_compliant_B.
# Writes fn.npz: T (nf,49) int8 tables, L, nmin, ncA, ncB. Semantics implemented from DESIGN.md §2-3 text.
import numpy as np, sys, time
p = 7; LMAX = int(sys.argv[1]); out = sys.argv[2]
X, Y = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
start = np.stack([X.ravel(), Y.ravel()]).astype(np.int8)[None]
INS = []
for d in (0, 1):
    for o in ("add", "sub", "mul"):
        for a in (0, 1):
            for b in (0, 1): INS.append((o, d, a, b))
    for o in ("neg", "inc"):
        for a in (0, 1): INS.append((o, d, a, None))
PATS = {"A": ("mul", "inc"), "B": ("inc", "mul")}

def apply(S, i):
    o, d, a, b = i; x = S[:, a].astype(np.int16); r = S.copy()
    if o == "add": v = x + S[:, b]
    elif o == "sub": v = x - S[:, b]
    elif o == "mul": v = x * S[:, b]
    elif o == "neg": v = -x
    else: v = x + 1
    r[:, d] = (v % p).astype(np.int8); return r

def flagmap(i, pat):
    """(4,) array: new flag index for each old flag index (bit r = register r fresh A-output), -1 = forbidden."""
    A, B = pat; o, d, a, b = i; res = np.empty(4, np.int64)
    for f in range(4):
        fr = [(f >> 0) & 1, (f >> 1) & 1]
        if o == B and (fr[a] or (b is not None and fr[b])): res[f] = -1; continue
        fr[d] = 1 if o == A else 0
        res[f] = fr[0] | (fr[1] << 1)
    return res
FM = {k: np.stack([flagmap(i, v) for i in INS]) for k, v in PATS.items()}

rng = np.random.default_rng(12345)
W = rng.integers(1, 2**62, size=(2, 98), dtype=np.uint64)
def key(S):
    o = np.empty((len(S), 2), np.uint64)
    for s in range(0, len(S), 300000):
        f = S[s:s+300000].reshape(-1, 98).astype(np.uint64)
        o[s:s+300000, 0] = f @ W[0] if False else (f * W[0]).sum(1)
        o[s:s+300000, 1] = (f * W[1]).sum(1)
    return o.view(np.dtype((np.void, 16))).ravel()

layer = start; cnt = np.ones(1, np.int64)
fc = {k: np.zeros((1, 4), np.int64) for k in PATS}
for k in PATS: fc[k][0, 0] = 1
seen = np.sort(key(start))
Ts, Ls, Ns, CA, CB = [start[:, 0]], [np.zeros(1, np.int8)], [cnt], [fc["A"].sum(1)], [fc["B"].sum(1)]
t0 = time.time()
for L in range(1, LMAX + 1):
    n = len(layer)
    allk, allc, allfa, allfb, allS = [], [], [], [], []
    for ii, i in enumerate(INS):
        ch = apply(layer, i); k = key(ch)
        pos = np.searchsorted(seen, k); pos[pos >= len(seen)] = 0; new = seen[pos] != k
        if not new.any(): continue
        allk.append(k[new]); allc.append(cnt[new]); allS.append(ch[new])
        for nm, acc in (("A", allfa), ("B", allfb)):
            src = fc[nm][new]; dst = np.zeros_like(src); m = FM[nm][ii]
            for f in range(4):
                if m[f] >= 0: dst[:, m[f]] += src[:, f]
            acc.append(dst)
        del ch
    k = np.concatenate(allk); uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
    S = np.concatenate(allS)[first]; del allS
    c = np.zeros(len(uk), np.int64); np.add.at(c, inv, np.concatenate(allc))
    fa = np.zeros((len(uk), 4), np.int64); np.add.at(fa, inv, np.concatenate(allfa))
    fb = np.zeros((len(uk), 4), np.int64); np.add.at(fb, inv, np.concatenate(allfb))
    layer, cnt, fc = S, c, {"A": fa, "B": fb}
    seen = np.sort(np.concatenate([seen, uk]))
    Ts.append(S[:, 0]); Ls.append(np.full(len(S), L, np.int8)); Ns.append(c); CA.append(fa.sum(1)); CB.append(fb.sum(1))
    print(f"L={L} states {len(S)} min-paths {c.sum()} ({time.time()-t0:.0f}s)", flush=True)
T = np.concatenate(Ts); L = np.concatenate(Ls); N = np.concatenate(Ns); A = np.concatenate(CA); B = np.concatenate(CB)
del Ts
fk = key(np.concatenate([T, T], 1).reshape(-1, 2, 49))  # key of the table alone (duplicated to 98)
order = np.lexsort((L, fk)); fk, T, L, N, A, B = fk[order], T[order], L[order], N[order], A[order], B[order]
first = np.r_[True, fk[1:] != fk[:-1]]
grp = np.cumsum(first) - 1
Lmin = L[first]
same = L == Lmin[grp]  # states at the function's minimal length
nf = first.sum()
nmin = np.bincount(grp[same], weights=N[same], minlength=nf).astype(np.int64)
ncA = np.bincount(grp[same], weights=A[same], minlength=nf).astype(np.int64)
ncB = np.bincount(grp[same], weights=B[same], minlength=nf).astype(np.int64)
np.savez_compressed(out, T=T[first], L=Lmin, nmin=nmin, ncA=ncA, ncB=ncB)
print("functions by l:", np.bincount(Lmin).tolist())
print("minimal programs by l:", [int(nmin[Lmin == l].sum()) for l in range(LMAX + 1)])
