# Review check: what a depth-L static *causal* transformer can reach on the rung-0 tokenization
# (pairs as single tokens, shuffled; queries after all pairs; answers absorbing).
# Model of the best static circuit: every pair token greedily extends its frontier pointer by reading the
# frontier pair's pointer when that pair is EARLIER in the sequence (causal mask); the query (after all pairs)
# extends its pointer by the frontier pair's reach every layer. DESIGN.md claims the Sanford 2^(L-2) ceiling;
# that ceiling includes a previous-token layer that this tokenization does not need.
import numpy as np, sys
rng = np.random.default_rng(0)
def reach_ok(k, L, causal=True):
    # nodes 1..k+1 ; pair j has key node_j, value node_{j+1}; r[j] = how far pair j points (initially 1)
    pos = rng.permutation(k)  # pos[j-1] = position of pair j among the chain's own pairs (relative order is all that matters)
    r = np.ones(k + 1, dtype=int)  # index 1..k
    c = 1  # query holds node c
    for t in range(L):
        # query update (reads layer-t states)
        c_new = c + r[c] if c <= k else c
        # pair updates in parallel
        r_new = r.copy()
        for j in range(1, k + 1):
            m = j + r[j]
            if m <= k and ((not causal) or pos[m - 1] < pos[j - 1]):
                r_new[j] = r[j] + r[m]
        r, c = r_new, c_new
        if c > k:
            return True
    return c > k
def table(causal, trials=2000):
    print(f"\n{'causal' if causal else 'non-causal'} attention: P(static depth-L model reaches the answer), 'greedy doubling' circuit")
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
    print("  L  | " + " ".join(f"k={k:>2}" for k in ks))
    for L in (2, 3, 4, 5, 6, 8):
        row = []
        for k in ks:
            row.append(np.mean([reach_ok(k, L, causal) for _ in range(trials)]))
        print(f"  {L:>2} | " + " ".join(f"{p:5.2f}" for p in row))
table(False, 200)
table(True, 2000)
print("\nSerial-only circuit (query hops once per layer, pairs do nothing): reach k <= L exactly.")
print("DESIGN.md's 2^(L-2): L=4 -> 4, L=8 -> 64.  Greedy-doubling non-causal: 2^L - 1 (L=4 -> 15).")
