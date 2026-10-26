# Design tool for the worlds round (experiments/worlds/DESIGN.md §3). Usage: python certify_patterns.py [LCERT=7]
"""Certify held-out compound patterns (DBCA-style) for the mod-7 two-register instruction set.

A pattern P = (A, B) is a data-dependent bigram: an instruction of op A writes a register, and a later instruction
of op B reads that register while it still holds A's output. A program is *compliant* if it never executes P.

For each candidate pattern, BFS over (register state, freshness flags) through compliant transitions gives, for every
function, its minimal compliant length l_c. With l the unrestricted minimal length:
  pool  = {f : 1 <= l(f) <= 5 and l_c(f) <= 5}           (trainable without P)
  test  = {f : 1 <= l(f) <= 6 and l_c(f) > LCERT}         (certified: every program of length <= LCERT uses P)
  grey  = the rest of l(f) <= 6 with l_c(f) > 5           (excluded from both)
The last BFS layer (LCERT) is not stored: its children are only checked against the candidate test set.
"""
import sys, time
import numpy as np

p = 7
LCERT = int(sys.argv[1]) if len(sys.argv) > 1 else 7
X, Y = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
S0 = np.stack([X.ravel(), Y.ravel()]).astype(np.int8)[None]  # (1, 2, 49)
OPS = ("add", "sub", "mul", "neg", "inc")
INS = [(o, d, a, b) for d in (0, 1) for o in ("add", "sub", "mul") for a in (0, 1) for b in (0, 1)] + \
      [(o, d, a, None) for d in (0, 1) for o in ("neg", "inc") for a in (0, 1)]
assert len(INS) == 32
rng = np.random.default_rng(7)
W1, W2 = (rng.integers(1, 2**62, size=100, dtype=np.uint64) for _ in range(2))


def apply(S, i):
    o, d, a, b = i
    out = S.copy(); x = S[:, a].astype(np.int16)
    v = {"add": lambda: x + S[:, b], "sub": lambda: x - S[:, b], "mul": lambda: x * S[:, b],
         "neg": lambda: -x, "inc": lambda: x + 1}[o]() % p
    out[:, d] = v.astype(np.int8)
    return out


def hkey(M):  # M: (n, m) int8 -> (n,) 16-byte keys
    m = M.shape[1]; out = np.empty((len(M), 2), np.uint64)
    for s in range(0, len(M), 400000):
        f = M[s:s + 400000].astype(np.uint64)
        out[s:s + 400000, 0] = (f * W1[:m]).sum(1); out[s:s + 400000, 1] = (f * W2[:m]).sum(1)
    return out.view(np.dtype((np.void, 16))).ravel()


def bfs(pattern, lmax, test_keys=None):
    """Return dict-like arrays: function key -> minimal compliant length (<= lmax-1 stored, lmax checked only)."""
    A, B = pattern if pattern else (None, None)
    S, F = S0, np.zeros((1, 2), np.int8)  # F[:, r] = 1 if register r holds a fresh output of op A
    seen = np.sort(hkey(np.concatenate([S.reshape(1, -1), F], 1)))
    fkeys, flens = [hkey(S[:, 0])], [np.zeros(1, np.int8)]
    hit_last = set()
    for L in range(1, lmax + 1):
        last = L == lmax and test_keys is not None
        newS, newF, newK = [], [], []
        for i in INS:
            o, d, a, b = i
            ok = np.ones(len(S), bool)
            if o == B:
                ok &= F[:, a] == 0
                if b is not None:
                    ok &= F[:, b] == 0
            if not ok.any():
                continue
            cS = apply(S[ok], i); cF = F[ok].copy(); cF[:, d] = 1 if o == A else 0
            if last:
                k = hkey(cS[:, 0]); hit_last.update(k[np.isin(k, test_keys)].tolist()); continue
            k = hkey(np.concatenate([cS.reshape(len(cS), -1), cF], 1))
            pos = np.searchsorted(seen, k); pos[pos >= len(seen)] = 0
            new = seen[pos] != k
            newS.append(cS[new]); newF.append(cF[new]); newK.append(k[new])
        if last:
            break
        k = np.concatenate(newK); u, first = np.unique(k, return_index=True)
        S, F = np.concatenate(newS)[first], np.concatenate(newF)[first]
        seen = np.sort(np.concatenate([seen, u]))
        fkeys.append(hkey(S[:, 0])); flens.append(np.full(len(S), L, np.int8))
    fk = np.concatenate(fkeys); fl = np.concatenate(flens)
    order = np.lexsort((fl, fk)); fk, fl = fk[order], fl[order]
    keep = np.r_[True, fk[1:] != fk[:-1]]
    return fk[keep], fl[keep], hit_last


t0 = time.time()
fk_all, fl_all, _ = bfs(None, 6)
print(f"unrestricted: {len(fk_all)} functions at l<=6 ({time.time()-t0:.0f}s);",
      "by length:", np.bincount(fl_all).tolist(), flush=True)
CANDIDATES = [("mul", "inc"), ("mul", "add"), ("mul", "sub"), ("add", "mul"), ("inc", "mul"), ("sub", "mul"),
              ("neg", "mul"), ("mul", "mul"), ("mul", "neg"), ("inc", "inc")]
if len(sys.argv) > 2:
    CANDIDATES = [tuple(s.split(">")) for s in sys.argv[2].split(",")]
for P in CANDIDATES:
    t = time.time()
    fk, fl, _ = bfs(P, 6)
    lc = np.full(len(fk_all), 99, np.int16)
    pos = np.searchsorted(fk, fk_all); pos[pos >= len(fk)] = 0; m = fk[pos] == fk_all
    lc[m] = fl[pos[m]]
    cand = (fl_all >= 1) & (lc > 6)
    _, _, hit7 = bfs(P, LCERT, test_keys=fk_all[cand]) if LCERT >= 7 else (None, None, set())
    in7 = np.isin(fk_all, np.array(list(hit7), dtype=fk_all.dtype)) if hit7 else np.zeros(len(fk_all), bool)
    lc[cand & in7] = 7
    pool = (fl_all >= 1) & (fl_all <= 5) & (lc <= 5)
    test = (fl_all >= 1) & (lc > LCERT)
    grey = (fl_all >= 1) & (lc > 5) & ~test
    n5 = ((fl_all >= 1) & (fl_all <= 5)).sum()
    tb = np.bincount(fl_all[test], minlength=7)[1:].tolist()
    print(f"{P[0]}>{P[1]}: pool {pool.sum()} ({pool.sum()/n5:.1%} of l<=5), test by l {tb} "
          f"(l<=5: {sum(tb[:5])}, l=6: {tb[5]}), grey l<=5 {(grey & (fl_all<=5)).sum()} l=6 {(grey & (fl_all==6)).sum()} "
          f"({time.time()-t:.0f}s)", flush=True)
