"""Exact posteriors over proofs (DESIGN §1, §4; review 1 S1–S2).

shortest_posterior — the RT-post target: uniform over shortest proofs as state sequences; every state on a shortest proof is
    a teacher-forced state with weight = (#shortest proofs through it) / (#shortest proofs); its action target spreads the
    next-state probability to_goal[v] / to_goal[u] uniformly over the actions that produce v.
pf_posterior — the fixed-prior posterior: a flat prior over routes (uniform over valid state-changing actions ∪ HALT at every
    state; HALT absorbs) conditioned on halting at the goal within T steps; per-state action marginals and visitation
    weights by forward–backward over the reachable graph. Precomputed with the dataset (router-independent)."""

from __future__ import annotations

import numpy as np

from .rules import HALT
from .search import bfs, paths_through, reverse_distances, shortest_dag


def shortest_posterior(t, s, W, cap: int, succ=None, dgoal=None) -> dict:
    """{state: {"weight": share of shortest proofs through the state, "target": {action: prob}}} over the shortest DAG,
    with action = (rule, path) or HALT (at the goal, HALT with probability 1)."""
    if succ is None or dgoal is None:
        _, succ, complete = bfs(t, W, cap)
        assert complete
        dgoal = reverse_distances(succ, s)
    assert t in dgoal, "goal unreachable"
    dag = shortest_dag(succ, dgoal, t)
    from_start, to_goal = paths_through(dag, dgoal, t, s)
    total = to_goal[t]
    out = {}
    for u in dag:
        if u == s:
            out[u] = {"weight": from_start.get(u, 0) / total, "target": {HALT: 1.0}}
            continue
        tgt = {}
        for v, acts in dag[u].items():
            p = to_goal[v] / to_goal[u]
            for a in acts:
                tgt[a] = tgt.get(a, 0.0) + p / len(acts)
        out[u] = {"weight": from_start.get(u, 0) * to_goal[u] / total, "target": tgt}
    return out


def pf_posterior(t, s, W, cap: int, T: int, succ=None) -> dict:
    """Flat-prior posterior over routes of at most T actions that halt at s.
    Returns {"p_success": prior probability of success, "states": {u: {"visits": expected visits under the posterior,
    "target": {action: prob}}}} for every state with non-zero posterior visitation."""
    if succ is None:
        _, succ, complete = bfs(t, W, cap)
        assert complete, "reachable graph too large for an exact posterior"
    nodes_ = list(succ.keys())
    idx = {u: i for i, u in enumerate(nodes_)}
    N = len(nodes_)
    # adjacency with action multiplicity: from i, to j, with m actions (prior prob m / (n_i + 1) for the move, 1/(n_i+1) HALT)
    nact = np.zeros(N)
    edges = []                                                # (i, j, m)
    for u, s_ in succ.items():
        i = idx[u]
        n = sum(len(a) for a in s_.values())
        nact[i] = n
        for v, acts in s_.items():
            if v in idx:
                edges.append((i, idx[v], len(acts)))
    src = np.array([e[0] for e in edges], dtype=np.int64) if edges else np.zeros(0, dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64) if edges else np.zeros(0, dtype=np.int64)
    mult = np.array([e[2] for e in edges], dtype=float) if edges else np.zeros(0)
    denom = nact + 1.0
    goal = np.zeros(N); goal[idx[s]] = 1.0
    # backward: beta[k][i] = P(success from i with T - k actions left)
    beta = np.zeros((T + 1, N))
    beta[T] = goal                                            # forced halt at T: success iff at the goal
    for k in range(T - 1, -1, -1):
        move = np.zeros(N)
        if len(edges):
            np.add.at(move, src, mult * beta[k + 1][dst])
        beta[k] = (goal + move) / denom
    # forward: alpha[k][i] = prior mass alive at i after k actions
    alpha = np.zeros((T + 1, N)); alpha[0][idx[t]] = 1.0
    for k in range(T):
        nxt = np.zeros(N)
        if len(edges):
            np.add.at(nxt, dst, alpha[k][src] * mult / denom[src])
        alpha[k + 1] = nxt
    p_succ = beta[0][idx[t]]
    assert p_succ > 0, "goal unreachable within T"
    # posterior mass of (state i, action) summed over steps
    visits = np.zeros(N)
    halt_mass = np.zeros(N)
    edge_mass = np.zeros(len(edges))
    for k in range(T):
        h = alpha[k] * goal / denom                           # halt at k (only at the goal contributes to success)
        halt_mass += h
        if len(edges):
            em = alpha[k][src] * mult / denom[src] * beta[k + 1][dst]
            edge_mass += em
        visits += alpha[k] * beta[k]
    halt_mass += alpha[T] * goal                              # the forced halt at T
    visits += alpha[T] * goal
    states = {}
    per_state_edges = {}
    for e_i, (i, j, m) in enumerate(edges):
        per_state_edges.setdefault(i, []).append((j, m, edge_mass[e_i]))
    for u, i in idx.items():
        tot = halt_mass[i] + sum(em for _, _, em in per_state_edges.get(i, []))
        if tot <= 1e-15:
            continue
        tgt = {}
        if halt_mass[i] > 0:
            tgt[HALT] = halt_mass[i] / tot
        for j, m, em in per_state_edges.get(i, []):
            v = nodes_[j]
            for a in succ[u][v]:
                tgt[a] = tgt.get(a, 0.0) + em / tot / m
        states[u] = {"visits": visits[i] / p_succ, "target": tgt}
    return {"p_success": float(p_succ), "states": states}
