# Rung-0 derivation (experiments/ptr/DESIGN.md §4): the distinguishing quantity before any run (lesson L22).
# Chain length k ~ truncated Zipf(alpha) on 1..kmax. A static transformer of depth L reaches k <= 2^(L-2) by doubling
# (Sanford, Hsu & Telgarsky 2024); an oracle-halting loop spends k+1 steps on a query token and 1 on a context token.
import numpy as np, sys
def zipf(alpha, kmax):
    k = np.arange(1, kmax + 1); p = k ** (-alpha); return k, p / p.sum()
print("A. static ceiling P(k <= 2^(L-2)) and compute, by tail alpha (kmax = 32)")
print(f"{'alpha':>6} {'E[k]':>6} {'P(k>8)':>7} {'P(k>16)':>8} | " + " ".join(f"L={L}" for L in (3,4,5,6,7)) + " | ceiling-free loop cost (query) E[k]+1")
for alpha in (0.5, 1.0, 1.5, 2.0):
    k, p = zipf(alpha, 32)
    row = [f"{(p * (k <= 2 ** (L - 2))).sum():.3f}" for L in (3, 4, 5, 6, 7)]
    print(f"{alpha:6.1f} {(p*k).sum():6.2f} {(p*(k>8)).sum():7.3f} {(p*(k>16)).sum():8.3f} | " + " ".join(f"{r:>4}" for r in row) + f" | {(p*k).sum()+1:.2f}")
print("\nB. compute per token over a whole sequence (P context pairs = 2P tokens needing 1 step; Q query tokens needing k+1),")
print("   P = 48 pairs, Q = 8 queries, alpha = 1.5, kmax = 32: static L costs L per token for every token.")
k, p = zipf(1.5, 32); P, Q = 48, 8
toks = 2 * P + 3 * Q
oracle = (2 * P * 1 + 2 * Q * 1 + Q * ((p * k).sum() + 1)) / toks
for L in (4, 6, 8, 16, 32):
    print(f"   static L={L:2d}: {L:5.2f} steps/token   oracle loop: {oracle:5.2f} steps/token   ratio {L/oracle:4.1f}x")
print("\nC. training examples at each depth k, for N = 2,000,000 query tokens, alpha = 1.5, kmax_train = 8 (then k > 8 unseen):")
k, p = zipf(1.5, 8); N = 2_000_000
print("   " + "  ".join(f"k={kk}: {int(N*pp):,}" for kk, pp in zip(k, p)))
k, p = zipf(2.0, 8)
print("   alpha = 2.0: " + "  ".join(f"k={kk}: {int(N*pp):,}" for kk, pp in zip(k, p)))
