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
print("\nB. compute per token over a whole training sequence: 1 BOS + P pair tokens + Qn query tokens + Qn answer tokens;")
print("   context tokens take 1 step, a query takes k (its chain length). P = 48, Qn = 6, alpha = 1.5, kmax = 8 (training).")
k, p = zipf(1.5, 8); P, Q = 48, 6
toks = 1 + P + 2 * Q
oracle = (1 + P + Q + Q * (p * k).sum()) / toks
for L in (2, 4, 8, 16):
    print(f"   static L={L:2d}: {L:5.2f} steps/token   oracle loop: {oracle:5.2f} steps/token   ratio {L/oracle:4.1f}x")
k, p = zipf(1.5, 32)
print(f"   at evaluation with k up to 32 (same tail): oracle {(1 + P + Q + Q * (p * k).sum()) / toks:5.2f} steps/token")
print("\nC. training examples at each depth k for the planned budget N = 20,000 steps x 64 episodes x 6 queries = 7.68M queries,")
print("   alpha = 1.5, kmax = 8; then with the tail (k >= 6) scaled by the P-pool factor:")
k, p = zipf(1.5, 8); N = 20000 * 64 * 6
print("   factor 1:    " + "  ".join(f"k={kk}: {int(N*pp):,}" for kk, pp in zip(k, p)))
for f in (1e-1, 1e-2, 1e-3, 1e-4):
    q = p.copy(); q[k >= 6] *= f; q /= q.sum()
    print(f"   factor {f:<6}" + "  ".join(f"k={kk}: {int(N*pp):,}" for kk, pp in zip(k, q) if kk >= 6))
