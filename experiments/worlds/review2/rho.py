# Is the memorization index rho (DESIGN §7.2) measuring memorization? For unseen-in-range functions, how often is the
# training-pool version space non-empty at each k, and what is Bayes(a)'s accuracy there (reviewer 2).
import numpy as np
d = np.load("fn5.npz"); T, L = d["T"].astype(np.int8), d["L"]
rng = np.random.default_rng(0)
hold = np.zeros(len(T), bool)
for l in range(1, 6):
    idx = np.where(L == l)[0]; hold[rng.choice(idx, round(0.1 * len(idx)), replace=False)] = True
cells = {"Nall": (~hold) & (L >= 1)}
for N in (100, 400):
    m = (~hold) & (L >= 1) & (L <= 3)
    for l in (4, 5):
        idx = np.where((~hold) & (L == l))[0]; m[rng.choice(idx, N, replace=False)] = True
    cells[f"N{N}"] = m
for cname, pool in cells.items():
    w = np.zeros(len(T))
    for l in range(1, 6):
        s = pool & (L == l); w[s] = 0.2 / s.sum()
    P = np.where(pool)[0]; TP = T[P]; wP = w[P]
    for lt in (4, 5):
        tests = rng.choice(np.where(hold & (L == lt))[0], 300, replace=False)
        ne = np.zeros(16); acc = np.zeros(16); accne = np.zeros(16); nne = np.zeros(16)
        for fi in tests:
            order = rng.permutation(49); cons = np.ones(len(P), bool)
            for k in range(16):
                x = order[k]
                if cons.any():
                    pr = np.bincount(TP[cons, x], weights=wP[cons], minlength=7)
                    c = float(np.argmax(pr) == T[fi, x]); ne[k] += 1; accne[k] += c; acc[k] += c
                else:
                    acc[k] += 1 / 7
                cons &= TP[:, x] == T[fi, x]
        n = len(tests)
        print(f"{cname} unseen l={lt}: P(pool VS non-empty) at k=0,2,4,6,8,10,12,15:",
              " ".join(f"{ne[k]/n:.3f}" for k in (0, 2, 4, 6, 8, 10, 12, 15)),
              "| acc_a at k>=8: %.3f" % (acc[8:].mean() / n))
