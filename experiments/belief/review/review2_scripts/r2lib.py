"""Review 2 — reviewer's own implementation of the four mechanisms, the probes and the regressions of DESIGN.md rev 3.

Imports only the carrier library and review 1's instance generator (review1_scripts/core.py: ball, distances for all 255
worlds). Does NOT import the author's refs3.py / power.py. World ids are 1..255 (bitmask over the eight rules); arrays are
indexed by id - 1.
"""
import os, sys, math, random, collections
import numpy as np

REPO = os.environ.get("REPO", "/home/claude/topology-design")
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "review1_scripts"), os.path.join(REPO, "experiments/belief/review/review1_scripts")):
    if os.path.exists(os.path.join(_cand, "core.py")):          # review 1's instance generator (ball, distances for 255 worlds)
        sys.path.insert(0, _cand)
        break
from core import Inst, HAS, WORLDS, NW, NR, WSIZE, prior_vec      # noqa: E402

GRID = (4, 8, 16, 32, 64)


def draw_sets(draw):
    """The design's draw convention (shuffle the 255 world ids with Random(draw); first 64 held out; nested training sets
    from the rest). Returned as bool[255] masks."""
    order = list(range(1, 256))
    random.Random(draw).shuffle(order)
    held = np.zeros(NW, bool); held[[w - 1 for w in order[:64]]] = True
    pool = order[64:]
    trains = {}
    for N in GRID:
        m = np.zeros(NW, bool); m[[w - 1 for w in pool[:N]]] = True
        trains[N] = m
    return held, trains, pool


class Dec:
    """A decision: state v, remaining budget b, known-present mask P, known-absent mask A."""

    def __init__(self, ins, v, b, P, A):
        self.ins, self.v, self.b, self.P, self.A = ins, v, b, P, A
        e0, e1 = ins.estart[v], ins.estart[v + 1]
        self.e0, self.e1 = e0, e1
        self.rules = ins.er[e0:e1]
        self.dst = ins.ed[e0:e1]
        self.ok = ins.optmat(v)                     # [255, E] edge optimal in world
        self.cnt = self.ok.sum(1)
        self.Dv = ins.D[:, v]
        self.alive = ins.cons & (self.Dv <= b)      # no evidence used
        self.live = self.fit(P, A)
        self.nE = e1 - e0

    def fit(self, P, A):
        m = self.alive.copy()
        if P:
            m &= (WORLDS & P) == P
        if A:
            m &= (WORLDS & A) == 0
        return m

    def mix(self, wm, weights=None):
        good = wm & (self.cnt > 0)
        if not good.any():
            return None
        w = np.ones(NW) if weights is None else weights
        v = (self.ok[good] * (w[good] / self.cnt[good])[:, None]).sum(0)
        z = v.sum()
        return v / z if z > 0 else None

    def single(self, wid):
        """Uniform over the optimal actions of world wid (1..255) if it can reach the goal within b; else None."""
        if wid <= 0 or self.Dv[wid - 1] > self.b or self.cnt[wid - 1] == 0:
            return None
        v = self.ok[wid - 1].astype(float)
        return v / v.sum()

    def rm(self, p, r):
        if p is None:
            return None
        x = np.where(self.rules == r, 0.0, p)
        z = x.sum()
        return x / z if z > 1e-12 else None

    def rmA(self, p, A):
        if p is None:
            return None
        x = np.where((A >> self.rules) & 1 == 1, 0.0, p)
        z = x.sum()
        return x / z if z > 1e-12 else None

    def uniform(self, A):
        x = ((A >> self.rules) & 1 == 0).astype(float)
        z = x.sum()
        return x / z if z > 0 else None

    def probes(self, pool=None):
        """Rules r not yet tested, with an action here, that some fitting world lacks (and some has)."""
        base = self.live if pool is None else self.live & pool
        out = []
        for r in sorted(set(int(x) for x in self.rules)):
            if (self.P | self.A) >> r & 1:
                continue
            lack = base & ~HAS[:, r]
            if not lack.any() or not (base & HAS[:, r]).any():
                continue
            out.append(r)
        return out


def argmax_first(ins, v, vec, tol=0.0):
    """First maximiser in the fixed action order (rule index, target node, path) among actions with positive mass.
    tol = 0 is the author's raw float comparison."""
    e0 = ins.estart[v]
    idxs = [j for j in range(len(vec)) if vec[j] > 0]
    idxs.sort(key=lambda j: (2 * int(ins.er[e0 + j]) + (1 if ins.ep[e0 + j] else 0), int(ins.ed[e0 + j]), ins.ep[e0 + j]))
    best = None
    for j in idxs:
        if best is None or vec[j] > vec[best] + tol:
            best = j
    return best


def walk(ins, true, policy, on_decision, tol=0.0, maxcalls=40):
    """Greedy trajectory under `policy(dec) -> vector` in the true world (index 0..254); ends at the goal, when the budget
    is used, or when the true world can no longer finish within the remaining budget."""
    v, b, P, A, calls = 0, ins.B, 0, 0, 0
    while v != ins.s and b > 0 and calls < ins.B + maxcalls:
        if ins.D[true, v] > b:
            break
        d = Dec(ins, v, b, P, A)
        p = policy(d)
        if p is None:
            break
        on_decision(d, calls)
        j = argmax_first(ins, v, p, tol)
        calls += 1
        r = int(d.rules[j])
        if HAS[true, r]:
            v, b, P = int(d.dst[j]), b - 1, P | (1 << r)
        else:
            A |= 1 << r


def shortest_path_counts(ins, wid):
    """Number of shortest action-paths to the goal from every node of the ball in world wid (0 where unreachable <= B)."""
    D = ins.D[wid - 1].astype(np.int64)
    ok = HAS[wid - 1][ins.er] & (D[ins.es] < 99) & (D[ins.ed] == D[ins.es] - 1)
    es, ed = ins.es[ok], ins.ed[ok]
    lvl = D[es]
    cnt = np.zeros(ins.n)
    cnt[ins.s] = 1.0
    for d in range(1, ins.B + 1):
        sel = lvl == d
        if sel.any():
            np.add.at(cnt, es[sel], cnt[ed[sel]])
    return cnt


class Reg:
    """Pooled least squares of model vectors on reference directions; sums kept per cluster (held-out world) for a
    bootstrap. A direction that is None at a probe is treated as zero there (the author's convention)."""

    def __init__(self, k):
        self.k = k
        self.G = collections.defaultdict(lambda: np.zeros((k, k)))
        self.b = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(k)))
        self.e = collections.defaultdict(lambda: collections.defaultdict(float))
        self.n = 0
        self.nw = collections.Counter()

    def add(self, cl, dirs, models):
        k = self.k
        E = next(len(d) for d in dirs if d is not None)
        Dm = np.stack([np.zeros(E) if d is None else d for d in dirs])
        self.G[cl] += Dm @ Dm.T
        names = [n for n, dp in models.items() if dp is not None]
        if names:
            Mm = np.stack([models[n] for n in names])
            B = Mm @ Dm.T
            en = (Mm * Mm).sum(1)
            bc, ec = self.b[cl], self.e[cl]
            for i, n in enumerate(names):
                bc[n] += B[i]
                ec[n] += float(en[i])
        self.n += 1
        self.nw[cl] += 1

    def _solve(self, cls, name):
        G = sum((self.G[c] for c in cls), np.zeros((self.k, self.k)))
        b = sum((self.b[c][name] for c in cls if name in self.b[c]), np.zeros(self.k))
        if G[0, 0] < 1e-9:
            return np.full(self.k, np.nan)
        return np.linalg.lstsq(G, b, rcond=1e-9)[0]

    def coef(self, name):
        return self._solve(list(self.G), name) if self.G else np.full(self.k, np.nan)

    def gram(self):
        return sum(self.G.values(), np.zeros((self.k, self.k)))

    def energy(self, name):
        return sum(self.e[c].get(name, 0.0) for c in self.G)

    def resid_share(self, name):
        c = self.coef(name)
        b = sum((self.b[cl][name] for cl in self.G if name in self.b[cl]), np.zeros(self.k))
        E = self.energy(name)
        return float("nan") if E <= 1e-12 or np.isnan(c).any() else 1.0 - float(c @ b) / E

    def boot(self, name, g, nb=400, f=None):
        """Bootstrap over clusters: returns array [nb, k] (or f applied)."""
        W = list(self.G)
        if len(W) < 2:
            return np.full((nb, self.k), np.nan)
        Gs = np.stack([self.G[c] for c in W])
        bs = np.stack([self.b[c][name] if name in self.b[c] else np.zeros(self.k) for c in W])
        out = np.full((nb, self.k), np.nan)
        for t in range(nb):
            idx = g.integers(0, len(W), len(W))
            G = Gs[idx].sum(0)
            if G[0, 0] < 1e-9:
                continue
            out[t] = np.linalg.lstsq(G, bs[idx].sum(0), rcond=1e-9)[0]
        return out

    def corr(self, i=0, j=1):
        G = self.gram()
        return G[i, j] / math.sqrt(G[i, i] * G[j, j]) if G[i, i] > 0 and G[j, j] > 0 else float("nan")

    def neff(self, i=0):
        """Effective number of clusters carrying direction i's energy."""
        e = np.array([G[i, i] for G in self.G.values()])
        return float(e.sum() ** 2 / (e ** 2).sum()) if e.sum() > 0 else 0.0


def classify(lo, hi, zero_name):
    """DESIGN §8 class from the 95 % bounds of the two scored coefficients (index 0 = G direction, 1 = O direction)."""
    if np.isnan(lo).any() or np.isnan(hi).any():
        return "n/a"
    if hi[0] <= 0.25 and hi[1] <= 0.25:
        return zero_name
    if lo[0] >= 0.5:
        return "G"
    if lo[1] >= 0.5:
        return "O"
    return "mixed"


def noisy(p, sigma, g):
    w = (p + 1e-3) * np.exp(sigma * g.standard_normal(p.shape))
    return w / w.sum()


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
