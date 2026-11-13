"""Reviewer's independent implementation of the hidden-world carrier (DESIGN §3-§4).

Does NOT import experiments/belief/power.py or prelim_info.py. Uses only the repo's executor (rules.valid_actions,
data.sample_term). Distances d_W are computed with a numpy boolean (world x node) reachability DP, a different algorithm
from the author's per-node python-int bitsets; `brute_dw` is a third, direct BFS per world used as a cross-check.

The RNG call order (sample_term -> choice(KS) -> choice(sphere) -> random()) is the same as the author's, so that with the
same seed the same instances are drawn and numbers can be compared one to one.
"""
import math, random
from collections import deque
import numpy as np
from topology_design.rewrite.rules import RULES, valid_actions, apply_local
from topology_design.rewrite.search import FULL
from topology_design.rewrite.data import sample_term
from topology_design.rewrite.term import nodes as term_nodes, replace_at, size

CAP = 5
NR = len(RULES)
RIDX = {r: i for i, r in enumerate(RULES)}
NW = (1 << NR) - 1
WORLDS = np.arange(1, NW + 1)                                   # world id w (bitmask over rules), row index w-1
HAS = ((WORLDS[:, None] >> np.arange(NR)[None, :]) & 1).astype(bool)   # [255, 8]
WSIZE = HAS.sum(1)


def prior_vec(rho):
    return rho ** WSIZE * (1 - rho) ** (NR - WSIZE)


class Inst:
    def __init__(self, rng, ks=(2, 3, 4), delta=2, rho=0.5, sizes=(2, 3, 4, 5), draw_world=True):
        while True:
            u = sample_term(rng, sizes)
            self.k = rng.choice(ks)
            self.B = self.k + delta
            self._ball(u, self.B)
            sphere = [i for i, d in enumerate(self.fdist) if d == self.k]
            if sphere:
                break
        self.u = u
        self.s = rng.choice(sphere)
        self.n = len(self.terms)
        self._dist()
        self.rho = rho
        self.prior = prior_vec(rho)
        self.cons = self.D[:, 0] <= self.B                       # bool[255]
        p = self.prior * self.cons
        self.post0 = p / p.sum()
        if draw_world:
            x = rng.random(); acc = 0.0
            w = None
            for wi in np.nonzero(self.cons)[0]:
                acc += self.post0[wi]
                w = wi
                if x <= acc:
                    break
            self.true = int(w)                                   # row index (world id - 1)

    # ---- ball of radius B around u in the full world
    def _ball(self, u, radius):
        idx, terms, fdist = {u: 0}, [u], [0]
        es, er, ed, ep = [], [], [], []
        q = deque([u])
        while q:
            v = q.popleft()
            i = idx[v]
            if fdist[i] >= radius:
                continue
            for rule, path, x in valid_actions(v, FULL, CAP):
                if x not in idx:
                    idx[x] = len(terms); terms.append(x); fdist.append(fdist[i] + 1); q.append(x)
                es.append(i); er.append(RIDX[rule]); ed.append(idx[x]); ep.append(path)
        self.terms, self.fdist = terms, fdist
        self.es, self.er, self.ed = np.array(es, dtype=np.int64), np.array(er, dtype=np.int64), np.array(ed, dtype=np.int64)
        self.ep = ep
        # edges are appended in node order, so they are sorted by source
        n = len(terms)
        self.estart = np.searchsorted(self.es, np.arange(n + 1))

    # ---- D[w, i] = d_W(i -> s) inside the ball (99 if > B)
    def _dist(self):
        n, B = self.n, self.B
        reach = np.zeros((NW, n), dtype=bool)
        reach[:, self.s] = True
        D = np.full((NW, n), 99, dtype=np.int8)
        D[:, self.s] = 0
        per_rule = []
        for r in range(NR):
            sel = np.nonzero(self.er == r)[0]
            if len(sel) == 0:
                per_rule.append(None); continue
            src, dst = self.es[sel], self.ed[sel]                # src sorted
            uniq, starts = np.unique(src, return_index=True)
            per_rule.append((dst, uniq, starts, np.nonzero(HAS[:, r])[0]))
        for j in range(1, B + 1):
            new = reach.copy()
            for r in range(NR):
                if per_rule[r] is None:
                    continue
                dst, uniq, starts, rows = per_rule[r]
                t = reach[np.ix_(rows, dst)]                     # [128, E_r]
                red = np.logical_or.reduceat(t, starts, axis=1)  # [128, n_src]
                sub = new[np.ix_(rows, uniq)]
                new[np.ix_(rows, uniq)] = sub | red
            newly = new & ~reach
            D[newly] = j
            reach = new
        self.D = D

    # ---- helpers
    def edges(self, i):
        return range(self.estart[i], self.estart[i + 1])

    def wmask(self, i, b, P=0, A=0, base=None, live=True):
        """Worlds: base (default: consistent with (u, s)), agreeing with evidence (P subset of W, A disjoint), and (if live)
        able to finish from i within b accepted steps."""
        m = (self.cons if base is None else base).copy()
        if live:
            m &= self.D[:, i] <= b
        if P:
            m &= (WORLDS & P) == P
        if A:
            m &= (WORLDS & A) == 0
        return m

    def optmat(self, i):
        """ok[w, e]: edge e of node i is optimal in world w (rule in w and d_W decreases by one). Rows of worlds that cannot
        reach s from i within B are all False."""
        e0, e1 = self.estart[i], self.estart[i + 1]
        r, x = self.er[e0:e1], self.ed[e0:e1]
        Di = self.D[:, i].astype(np.int16)
        ok = HAS[:, r] & (self.D[:, x].astype(np.int16) == (Di[:, None] - 1)) & (Di[:, None] < 99)
        return ok

    def mix(self, i, wm, weights=None):
        """Bayes mixture over worlds in wm of the uniform-over-optimal-actions label. Vector over the edges of node i.
        Returns None when wm is empty (or no world in it has an optimal action)."""
        if weights is None:
            weights = self.prior
        ok = self.optmat(i)[wm]
        cnt = ok.sum(1)
        w = weights[wm]
        good = cnt > 0
        if not good.any():
            return None
        v = (ok[good] * (w[good] / cnt[good])[:, None]).sum(0)
        z = v.sum()
        return v / z if z > 0 else None

    def post(self, wm, weights=None):
        w = (self.prior if weights is None else weights) * wm
        return w / w.sum()


def renorm(v):
    z = v.sum()
    return v / z if z > 0 else None


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def H2(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def brute_dw(inst, w_id, limit):
    """Direct BFS in world w from u to s using the rules themselves (not the ball); distance or None if > limit."""
    rules = [RULES[j] for j in range(NR) if w_id >> j & 1]
    u, s = inst.terms[0], inst.terms[inst.s]
    if u == s:
        return 0
    dist = {u: 0}
    q = deque([u])
    while q:
        v = q.popleft()
        if dist[v] >= limit:
            continue
        nds = term_nodes(v)
        for rule in rules:
            for path, sub, _ in nds:
                y = apply_local(rule, sub)
                if y is None:
                    continue
                new = replace_at(v, path, y)
                if size(new) > CAP or new == v or new in dist:
                    continue
                dist[new] = dist[v] + 1
                if new == s:
                    return dist[new]
                q.append(new)
    return None
