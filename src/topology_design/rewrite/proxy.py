"""Syntactic progress proxies, per-decision difficulty (GP1, D*_sep, D*_any) and the symbolic lookahead ladder
(DESIGN §1, §4, review 1 B3/B5).

The proxies are proxy-relative by construction: tree-edit distance (Zhang–Shasha, unit costs) is the primary σ, string-edit
distance over prefix tokens the sensitivity proxy. Lookahead values are ordered by (σ, steps) so that they stay defined
past the remaining distance (every neighbour reaches σ = 0 within d − 1 steps by backtracking once d exceeds it; the step
count then separates them)."""

from __future__ import annotations

from functools import lru_cache

from .rules import RULES, successors, valid_actions
from .term import is_leaf, prefix

RULE_INDEX = {r: i for i, r in enumerate(RULES)}


# ---------------- proxies ----------------

def _postorder(t):
    nodes = []          # [label, leftmost-leaf index]
    def rec(u):
        if is_leaf(u):
            nodes.append([u, None]); i = len(nodes) - 1; nodes[i][1] = i
            return i
        lf = rec(u[0]); rec(u[1])
        nodes.append(["+", lf])
        return lf
    rec(t)
    return nodes


@lru_cache(maxsize=1 << 18)
def ted(t1, t2) -> int:
    """Zhang–Shasha ordered tree edit distance with unit costs (rename when labels differ, insert, delete)."""
    A, B = _postorder(t1), _postorder(t2)
    la, lb = [n[0] for n in A], [n[0] for n in B]
    lA, lB = [n[1] for n in A], [n[1] for n in B]
    m, n = len(A), len(B)
    def keyroots(l):
        kr = {}
        for i, v in enumerate(l):
            kr[v] = i
        return sorted(kr.values())
    KA, KB = keyroots(lA), keyroots(lB)
    TD = [[0] * n for _ in range(m)]
    for i in KA:
        for j in KB:
            li, lj = lA[i], lB[j]
            fd = [[0] * (j - lj + 2) for _ in range(i - li + 2)]
            for x in range(1, i - li + 2):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, j - lj + 2):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, i - li + 2):
                for y in range(1, j - lj + 2):
                    ii, jj = li + x - 1, lj + y - 1
                    if lA[ii] == li and lB[jj] == lj:
                        c = 0 if la[ii] == lb[jj] else 1
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[x - 1][y - 1] + c)
                        TD[ii][jj] = fd[x][y]
                    else:
                        p, q = lA[ii] - li, lB[jj] - lj
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[p][q] + TD[ii][jj])
    return TD[m - 1][n - 1]


@lru_cache(maxsize=1 << 18)
def sed(t1, t2) -> int:
    a, b = prefix(t1), prefix(t2)
    D = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, D[0] = D[0], i
        for j in range(1, len(b) + 1):
            cur = D[j]
            D[j] = min(D[j] + 1, D[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return D[len(b)]


PROXIES = {"ted": ted, "sed": sed}


# ---------------- lookahead values ----------------

def lookahead_value(v, s, W, cap, d, sigma):
    """(σ, steps) of the best state reachable from v within d − 1 in-world steps (v itself at 0 steps): the value of
    choosing the action that leads to v, under a lookahead of depth d. Ordered lexicographically."""
    best = (sigma(v, s), 0)
    frontier, seen = {v}, {v}
    for k in range(1, d):
        nxt = set()
        for u in frontier:
            for w in successors(u, W, cap):
                if w not in seen:
                    seen.add(w); nxt.add(w)
                    val = (sigma(w, s), k)
                    if val < best:
                        best = val
        frontier = nxt
        if not frontier:
            break
    return best


def lookahead_profile(v, s, W, cap, dmax, sigma):
    """[V_1(v), ..., V_dmax(v)]: lookahead_value(v, ..., d, sigma) for every d ≤ dmax from one BFS (the prefix minimum of
    (σ, steps) over the layers 0 .. d − 1)."""
    best = (sigma(v, s), 0)
    out = [best]
    frontier, seen = {v}, {v}
    for k in range(1, dmax):
        nxt = set()
        for u in frontier:
            for w in successors(u, W, cap):
                if w not in seen:
                    seen.add(w); nxt.add(w)
                    val = (sigma(w, s), k)
                    if val < best:
                        best = val
        frontier = nxt
        out.append(best)
        if not frontier:
            out.extend([best] * (dmax - 1 - k))
            break
    return out


# ---------------- per-decision difficulty ----------------

def decision(t, s, W, cap, dgoal, sigma=ted, dmax=None, depth: bool = True) -> dict:
    """At state t with goal s under W: the candidate set (valid state-changing successors), the correct ones (distance
    falls by one; dgoal gives the exact distance to s), and
      B1     the number of candidates in the one-step σ-argmin,
      GP1    strict garden path at depth 1: min σ over correct > min σ over incorrect,
      tie1   a σ-tie at depth 1 between a correct and an incorrect candidate,
      Dsep   the smallest d with argmin_(σ, steps) V_d ⊆ correct,
      Dany   the smallest d with argmin ∩ correct ≠ ∅.
    dgoal must cover every candidate (a BFS from t to radius ≥ 1 with reverse distances, or the full graph).
    depth=False stops after the depth-1 quantities (Dsep and Dany are then None): the cheap phase of generation."""
    dt = dgoal[t]
    cands = {v: acts for v, acts in successors(t, W, cap).items()}
    correct = {v for v in cands if dgoal.get(v) == dt - 1}
    assert correct, "no correct action at a non-goal state"
    sig1 = {v: sigma(v, s) for v in cands}
    mn = min(sig1.values())
    argmin1 = {v for v in cands if sig1[v] == mn}
    min_c = min(sig1[v] for v in correct)
    min_i = min((sig1[v] for v in cands if v not in correct), default=None)
    gp1 = min_i is not None and min_c > min_i
    tie1 = min_i is not None and min_c == min_i
    if not depth:
        return {"d": dt, "n_cand": len(cands), "n_correct": len(correct), "B1": len(argmin1), "GP1": bool(gp1), "tie1": bool(tie1),
                "Dsep": None, "Dany": None}
    limit = max(dmax if dmax is not None else dt, 1)
    # one BFS per candidate gives V_d for every d at once: V_d(v) = prefix minimum of (σ, steps) over the layers ≤ d − 1
    profiles = {v: lookahead_profile(v, s, W, cap, limit, sigma) for v in cands}
    dsep = dany = None
    for d in range(1, limit + 1):
        m = min(profiles[v][d - 1] for v in cands)
        arg = {v for v in cands if profiles[v][d - 1] == m}
        if dany is None and arg & correct:
            dany = d
        if arg <= correct:
            dsep = d
            break
    if dsep is None:
        dsep = limit + 1                       # not separated within the remaining distance (should not happen: at d = dt
    if dany is None:                           # the correct actions reach σ = 0 in dt − 1 steps and no incorrect one can)
        dany = limit + 1
    return {"d": dt, "n_cand": len(cands), "n_correct": len(correct), "B1": len(argmin1), "GP1": bool(gp1), "tie1": bool(tie1),
            "Dsep": dsep, "Dany": dany}


# ---------------- the symbolic ladder ----------------

def ladder_policy(t, s, W, cap, d, sigma=ted, tie="lex", rng=None):
    """One decision of the lookahead-d rung: the valid action whose successor has the best (σ, steps) value; ties broken
    lexicographically by (rule, path) or at random. Returns (rule, path, new) or None when no valid action exists."""
    acts = valid_actions(t, W, cap)
    if not acts:
        return None
    vals = {}
    for rule, path, new in acts:
        if new not in vals:
            vals[new] = lookahead_value(new, s, W, cap, d, sigma)
    m = min(vals.values())
    best = [(rule, path, new) for rule, path, new in acts if vals[new] == m]
    if tie == "lex":
        best.sort(key=lambda a: (RULE_INDEX[a[0]], a[1]))
        return best[0]
    return rng.choice(best)


def ladder_rollout(t, s, W, cap, d, T, sigma=ted, tie="lex", rng=None) -> dict:
    """Run the lookahead-d rung from t for at most T actions; halt when the term equals s. Returns success and steps."""
    u = t
    for i in range(T):
        if u == s:
            return {"ok": True, "steps": i}
        a = ladder_policy(u, s, W, cap, d, sigma, tie, rng)
        if a is None:
            return {"ok": False, "steps": i, "reason": "no valid action"}
        u = a[2]
    return {"ok": u == s, "steps": T, "reason": None if u == s else "horizon"}


# ---------------- the training-world reference (DESIGN §17: P2's non-composing comparator) ----------------

def goal_distances(s, U, cap, actions=None) -> dict:
    """Distance to s inside world U for every term that can reach it (the rule set is closed under inverses, so the
    rewrite graph is symmetric and a BFS from s gives distances to s). `actions(u)` may supply cached full-world actions
    [(rule, path, new)]; they are filtered to U here."""
    from collections import deque
    from .rules import RULE_AX
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        acts = [a for a in actions(u) if RULE_AX[a[0]] in U] if actions is not None else valid_actions(u, U, cap)
        for _, _, v in acts:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def trainworld_policy(u, s, W, cap, dist: dict):
    """One decision of the training-world reference: among the actions valid in W (the mask), take one that lies on a
    shortest proof to s inside some *training* world — the smallest such world first, then the shorter proof, then the
    world's name; if no valid action does, the one whose successor is nearest to s in any training world's metric; if no
    successor can reach s inside a training world, the first by (rule, path). `dist` = {training world name: distances to
    s inside it}. It never plans with a set of axioms it was not 'trained' on and never reads which world it is in — it
    is what a router scores that acts at every state as some training world would and lets the executor's mask choose."""
    from .rules import RULE_AX
    cand = valid_actions(u, W, cap)
    if not cand:
        return None

    def key(a):
        rule, path, v = a
        best = fb = None
        for name, d in dist.items():
            if RULE_AX[rule] not in name:
                continue
            if u in d and d.get(v) == d[u] - 1:
                k_ = (len(name), d[u], name)
                if best is None or k_ < best:
                    best = k_
            if v in d:
                f_ = (d[v], len(name), name)
                if fb is None or f_ < fb:
                    fb = f_
        if best is not None:
            return (0, best, RULE_INDEX[rule], path)
        if fb is not None:
            return (1, fb, RULE_INDEX[rule], path)
        return (2, (0, 0, ""), RULE_INDEX[rule], path)

    return min(cand, key=key)


def trainworld_rollout(t, s, W, cap, T, dist: dict) -> bool:
    u = t
    for _ in range(T):
        if u == s:
            return True
        a = trainworld_policy(u, s, W, cap, dist)
        if a is None:
            return False
        u = a[2]
    return u == s
