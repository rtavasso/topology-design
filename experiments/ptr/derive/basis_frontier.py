# Rung-0 / Entry 51 level-1 derivation: a small basis covers the full transformation monoid T_n (all maps of an n-set
# to itself = all token maps on an alphabet of size n) at the price of long words. BFS over the Cayley graph gives,
# for each generating set: coverage of T_n, the distribution of minimal word lengths, the diameter, and redundancy
# (number of minimal words per map). Usage: python basis_frontier.py [nmax=6]
import numpy as np, sys, itertools, time
nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
def compose(f, g):  # (f∘g)(x) = f(g(x)); maps as tuples
    return tuple(f[g[x]] for x in range(len(g)))
def bfs(n, gens):
    ident = tuple(range(n)); dist = {ident: 0}; nwords = {ident: 1}; frontier = [ident]; L = 0
    while frontier:
        L += 1; nxt = []
        for f in frontier:
            for g in gens:
                h = compose(g, f)
                if h not in dist:
                    dist[h] = L; nwords[h] = nwords[f]; nxt.append(h)
                elif dist[h] == L:
                    nwords[h] += nwords[f]
        frontier = nxt
    return dist, nwords
def gensets(n):
    cyc = tuple((i + 1) % n for i in range(n)); tr = (1, 0) + tuple(range(2, n)); col = (0, 0) + tuple(range(2, n))
    allt = [tuple(j if x == i else i if x == j else x for x in range(n)) for i in range(n) for j in range(i + 1, n)]
    allc = [tuple(j if x == i else x for x in range(n)) for i in range(n) for j in range(n) if i != j]
    return {"3 gens {cycle, transposition, collapse}": [cyc, tr, col],
            "2 gens {cycle, transposition}: permutations only": [cyc, tr],
            f"all transpositions + all collapses ({len(allt)+len(allc)} gens)": allt + allc,
            f"transpositions+collapses+cycle ({len(allt)+len(allc)+1} gens)": allt + allc + [cyc]}
for n in range(3, nmax + 1):
    print(f"\nn = {n}: |T_n| = n^n = {n**n:,}")
    for name, gens in gensets(n).items():
        t = time.time(); dist, nw = bfs(n, gens)
        d = np.array(list(dist.values())); w = np.array([nw[f] for f in dist])
        cov = len(dist) / n ** n
        hist = np.bincount(d)
        print(f"  {name:58s} coverage {cov:6.3f}  mean len {d.mean():5.2f}  diameter {d.max():3d}  "
              f"mean minimal words/map {w.mean():7.1f}  median {np.median(w):5.0f}  ({time.time()-t:.1f}s)")
        if cov == 1.0 and len(gens) <= 3:
            print("     length histogram:", " ".join(f"{L}:{c}" for L, c in enumerate(hist) if c))
