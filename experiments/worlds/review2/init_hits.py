# Signal at initialization for the sampled-MML estimator (DESIGN §4): paths from a uniform policy over 33 tokens
# (HALT forced after 7 instructions). For targets from the D3 / Nall training priors: P(hit at one query), and
# P(path computes the target exactly | hit) = share of hits that are not spurious. Also hit rates if a sampled path
# were scored on m held-out points of the episode instead of one (reviewer 2).
import numpy as np, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from isa_common import INS, apply, start
d = np.load("fn5.npz"); T, L = d["T"], d["L"]
rng = np.random.default_rng(0); M = 200000
S = np.repeat(start, M, 0); alive = np.ones(M, bool); length = np.zeros(M, int)
for t in range(7):
    tok = rng.integers(0, 33, M); halt = tok == 32
    alive &= ~(halt & alive) if True else alive
    for j, i in enumerate(INS):
        m = alive & (tok == j)
        if m.any(): S[m] = apply(S[m], i)
    length += alive
F = S[:, 0]  # (M, 49) sampled functions
print("path length distribution:", np.bincount(length, minlength=8) / M)
from collections import Counter
fk = [r.tobytes() for r in F]; c = Counter(fk)
print(f"distinct functions among {M} random paths: {len(c)}; top-5 shares:", [round(v / M, 4) for _, v in c.most_common(5)])
tk = {T[i].tobytes(): i for i in range(len(T))}
for cell, lmax in (("D3", 3), ("Nall", 5)):
    hits = []; exact = []; hitm = {1: [], 4: [], 8: []}
    for _ in range(3000):
        l = rng.integers(1, lmax + 1); fi = rng.choice(np.where(L == l)[0]); x = rng.integers(49)
        # use a block of 64 sampled paths per target
        idx = rng.integers(0, M, 64); h = F[idx, x] == T[fi, x]; e = (F[idx] == T[fi]).all(1)
        hits.append(h.mean()); exact.append(e[h].mean() if h.any() else np.nan)
        for m in hitm:
            xs = rng.choice(49, m, replace=False); hitm[m].append((F[idx][:, xs] == T[fi, xs]).all(1).mean())
    print(f"{cell}: P(hit one query) {np.mean(hits):.3f}; P(exact | hit) {np.nanmean(exact):.4f}; "
          + "; ".join(f"P(hit all of {m} points) {np.mean(v):.4f}" for m, v in hitm.items()))
