import numpy as np
p = 7
X, Y = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
start = np.stack([X.ravel(), Y.ravel()]).astype(np.int8)[None]
INS = []
for d in (0, 1):
    for o in ("add", "sub", "mul"):
        for a in (0, 1):
            for b in (0, 1): INS.append((o, d, a, b))
    for o in ("neg", "inc"):
        for a in (0, 1): INS.append((o, d, a, None))

def apply(S, i):
    o, d, a, b = i; x = S[:, a].astype(np.int16); r = S.copy()
    if o == "add": v = x + S[:, b]
    elif o == "sub": v = x - S[:, b]
    elif o == "mul": v = x * S[:, b]
    elif o == "neg": v = -x
    else: v = x + 1
    r[:, d] = (v % p).astype(np.int8); return r

_r = np.random.default_rng(999)
_W = _r.integers(1, 2**62, size=(4, 100), dtype=np.uint64)

def key98(M, salt=0):
    M = M.reshape(len(M), -1); m = M.shape[1]; o = np.empty((len(M), 2), np.uint64)
    for s in range(0, len(M), 300000):
        f = M[s:s+300000].astype(np.uint64)
        o[s:s+300000, 0] = (f * _W[2*salt][:m]).sum(1); o[s:s+300000, 1] = (f * _W[2*salt+1][:m]).sum(1)
    return o.view(np.dtype((np.void, 16))).ravel()

def key49(T):
    return key98(T.reshape(len(T), 49))
