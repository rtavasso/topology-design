# ceiling.py section B counts 2P+3Q = 120 tokens (pair = 2 tokens, query = 3); DESIGN §2 says pairs are single
# tokens and the sequence is P + 2Q = 64. Recount with the design's own tokenization and with the §8 training budget.
import numpy as np
def zipf(a, kmax):
    k = np.arange(1, kmax + 1); p = k ** -a; return k, p / p.sum()
P, Q = 48, 8
for a in (1.0, 1.5, 2.0):
    k, p = zipf(a, 32); Ek = (p * k).sum()
    toks = P + 2 * Q
    oracle = (P * 1 + Q * 1 + Q * (Ek + 1)) / toks
    print(f"alpha={a}: oracle {oracle:.2f} steps/token (design says 1.29 at 1.5); S4 ratio {4/oracle:.1f}x (design 3.1x); S8 {8/oracle:.1f}x (design 6.2x)")
# examples per depth at the §8 budget (30k steps x 64 episodes x 8 queries) vs the §4.2 figure (2M)
N = 30_000 * 64 * 8
for a in (1.5, 2.0):
    k, p = zipf(a, 8)
    print(f"alpha={a}: N={N:,} queries; k=8 examples {int(N*p[-1]):,} (design §4.2 says {int(2e6*p[-1]):,} from N=2M)")
# no-hop context prior: the answer must be a terminal answer present in context. Simulate the forest generator
# (8 queried chains Zipf(1.5, kmax 8) + distractor chains to 48 pairs, answers uniform with replacement) and score
# the best guess 'most frequent terminal answer in context, excluding answers already revealed by earlier queries'
rng = np.random.default_rng(0)
def episode(a=1.5):
    k, p = zipf(a, 8)
    while True:
        ks = rng.choice(k, size=Q, p=p)
        if ks.sum() <= P: break
    chains = list(ks)
    left = P - ks.sum()
    while left > 0:
        kk = min(int(rng.choice(k, p=p)), left); chains.append(kk); left -= kk
    ans = rng.integers(16, size=len(chains))
    return ans  # per chain; first Q are queried
hit_first, hit_last, hit_oracleprior = [], [], []
for _ in range(20000):
    ans = episode()
    term = list(ans)  # terminal answers present in context (multiset)
    # query 1: guess most frequent terminal answer
    vals, cnt = np.unique(term, return_counts=True)
    hit_first.append(vals[np.argmax(cnt)] == ans[0])
    # query 8: earlier 7 answers revealed; remove one chain per revealed answer
    rem = list(term)
    for a_ in ans[:7]: rem.remove(a_)
    vals, cnt = np.unique(rem, return_counts=True)
    hit_last.append(vals[np.argmax(cnt)] == ans[7])
print(f"no-hop guess accuracy: query 1 {np.mean(hit_first):.3f}, query 8 {np.mean(hit_last):.3f}  (design's 'chance' = 1/16 = 0.0625)")
