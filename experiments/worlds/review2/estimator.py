# Planted check of the best-first evaluation estimator (DESIGN §4 "Evaluating O and E", §11): an oracle that samples
# minimal programs from pi* under the generator prior (b). Best-first enumeration of its paths = taking paths in order
# of probability post(f)/nmin(f). Stop at covered mass >= 0.999 or 4,096 paths; residual delta spread uniformly.
# Reports covered mass and excess code length over exact Bayes(b), by k (reviewer 2).
import numpy as np, sys
d = np.load("fn6.npz"); T, L, nmin = d["T"], d["L"], d["nmin"].astype(float)
nz = L >= 1; T, L, nmin = T[nz], L[nz], nmin[nz]
w = np.zeros(len(T))
for l in range(1, 7): w[L == l] = 1 / 6 / (L == l).sum()
rng = np.random.default_rng(1)
NEP = int(sys.argv[1]) if len(sys.argv) > 1 else 120; MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
cov = np.zeros((7, 16)); exc = np.zeros((7, 16)); nep = np.zeros(7); ex_bayes = np.zeros((7, 16))
for ep in range(NEP):
    lt = 1 + ep % 6; fi = rng.choice(np.where(L == lt)[0]); order = rng.permutation(49)
    cons = np.ones(len(T), bool); nep[lt] += 1
    for k in range(16):
        x = order[k]; z = T[fi, x]
        idx = np.where(cons)[0]; post = w[idx] / w[idx].sum()
        exact = np.bincount(T[idx, x], weights=post, minlength=7)[z]
        pp = post / nmin[idx]; o = np.argsort(-pp)                    # per-path probability, best first
        ncum = np.cumsum(nmin[idx][o]); mcum = np.cumsum(post[o])
        # number of whole functions fully taken before hitting either limit
        stop_m = np.searchsorted(mcum, 0.999) ; stop_p = np.searchsorted(ncum, MAXP)
        j = min(stop_m, stop_p)
        take = np.zeros(len(o)); take[:j] = 1.0
        if j < len(o):  # partial function at the boundary
            prev_n = ncum[j - 1] if j > 0 else 0
            if stop_p <= stop_m: take[j] = (MAXP - prev_n) / nmin[idx][o[j]]
            else: take[j] = 1.0
        cm = post[o] * take
        covered = np.bincount(T[idx[o], x], weights=cm, minlength=7)
        delta = 1 - cm.sum(); qhat = covered[z] + delta / 7
        cov[lt, k] += 1 - delta; exc[lt, k] += np.log(exact) - np.log(qhat); ex_bayes[lt, k] += -np.log(exact)
        cons &= T[:, x] == z
print(f"MAXP={MAXP}; rows: true l; columns k=0..15")
for lt in range(1, 7):
    n = nep[lt]
    print(f"l={lt} covered mass:", " ".join(f"{v:.2f}" for v in cov[lt] / n))
    print(f"     excess nats:  ", " ".join(f"{v:.2f}" for v in exc[lt] / n))
    print(f"     Bayes(b) nats:", " ".join(f"{v:.2f}" for v in ex_bayes[lt] / n))
print("mean excess per position (all l): %.3f nats; Bayes(b) mean per position %.3f" % (exc[1:].sum() / nep.sum() / 16, ex_bayes[1:].sum() / nep.sum() / 16))
