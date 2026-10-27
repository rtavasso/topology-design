# Review check of DESIGN.md §5 planted block: attention (register vs key slot, softmax scale 30) copies the value slot
# into scratch; a linear map writes scratch -> register and clears scratch. Question: is it a no-op once the register
# holds an answer (needed by LF, LR and any over-application), and are pair tokens fixed points?
import numpy as np
rng = np.random.default_rng(1)
NK, NA = 64, 16; NS = NK + NA
d = 400
# slot subspaces (orthogonal by construction)
K, V, R, S, M = slice(0, 80), slice(80, 160), slice(160, 240), slice(240, 320), 320
def sym(s, sl):
    v = np.zeros(d); v[sl.start + s] = 1.0; return v
def make_episode(k, P=48, Q=1):
    keys = rng.permutation(NK)[:P]
    chain = list(keys[:k]); ans = rng.integers(NA)
    pairs = [(chain[i], chain[i + 1]) for i in range(k - 1)] + [(chain[-1], NK + ans)]
    # distractor pairs: remaining keys mapped to random other keys/answers (forest not enforced; irrelevant here)
    rest = keys[k:]
    for i, kk in enumerate(rest):
        tgt = rest[i + 1] if i + 1 < len(rest) else NK + rng.integers(NA)
        pairs.append((kk, tgt))
    rng.shuffle(pairs)
    toks = [sym(a, K) + sym(b, V) for a, b in pairs]
    qtok = np.zeros(d); qtok[M] = 1; qtok += sym(chain[0], R)
    toks.append(qtok)
    return np.stack(toks), NK + ans
def block(H, scale=30.0, gated=False):
    q = H[:, R]; kk = H[:, K]; v = H[:, V]
    sc = scale * q @ kk.T
    n = len(H); sc[np.triu_indices(n, 1)] = -np.inf  # causal
    sc = sc - sc.max(1, keepdims=True); w = np.exp(sc); w /= w.sum(1, keepdims=True)
    att = w @ v  # copies value slot content into scratch
    H2 = H.copy(); H2[:, S] += att
    # linear map: register <- scratch, scratch cleared
    out = H2.copy()
    if gated:  # non-linear fix: overwrite only if scratch holds a clean one-hot (max > 0.5)
        ok = H2[:, S].max(1) > 0.5
        out[ok, R] = H2[ok, S]
    else:
        out[:, R] = H2[:, S]
    out[:, S] = 0
    return out
def readout(h):  # register -> answer id (argmax over answer coordinates), plus the register's max mass
    reg = h[R]; return int(np.argmax(reg[NK:])) + NK, reg.max(), reg[:NK].max()
for gated in (False, True):
    print(f"\n{'GATED overwrite (needs MLP)' if gated else 'LINEAR overwrite as in §5'}")
    for k in (1, 4, 8, 32):
        H, ans = make_episode(k)
        for t in range(1, 34):
            H = block(H, gated=gated)
            if t in (k, k + 1, k + 2, 33):
                a, m, mk = readout(H[-1]); print(f"  k={k:2d} after {t:2d} loops: readout {'OK ' if a == ans else 'BAD'}  reg answer-mass {m:5.2f}  key-mass {mk:5.2f}")
# pair tokens: are their K/V slots fixed under the block?  (register garbage is tolerated)
H, ans = make_episode(4); H0 = H.copy()
for t in range(33): H = block(H)
print("\npair tokens after 33 loops: max |dK|,|dV| =", np.abs(H[:-1, K] - H0[:-1, K]).max(), np.abs(H[:-1, V] - H0[:-1, V]).max(),
      "; max register mass on pair tokens =", H[:-1, R].max())
