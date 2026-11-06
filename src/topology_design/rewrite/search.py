"""Exact search over the rewrite graph: BFS within a size cap and a radius, reverse distances to a goal, shortest-proof
counting and enumeration, the binding test, and strategy (trace) classes modulo commuting moves (DESIGN §1, Q6)."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from .rules import RULES, successors, world_from_name
from .term import paths_overlap

RULE_INDEX = {r: i for i, r in enumerate(RULES)}
FULL = frozenset("ACID")


def bfs(t, W, cap: int, radius: int | None = None, max_nodes: int | None = None, stop_at=None):
    """BFS from t under W. Returns (dist, succ, complete): dist {term: d}, succ {term: {new: [(rule, path)]}} for every
    expanded node, complete False when max_nodes was hit. With stop_at, the search ends when that term's layer is complete
    (distances up to and including stop_at's layer are then exact)."""
    dist = {t: 0}
    succ = {}
    q = deque([t])
    stop_layer = None
    while q:
        u = q.popleft()
        if stop_layer is not None and dist[u] >= stop_layer:
            break
        if radius is not None and dist[u] >= radius:
            succ[u] = {}
            continue
        s_ = successors(u, W, cap)
        succ[u] = s_
        for v in s_:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
                if v == stop_at and stop_layer is None:
                    stop_layer = dist[v]
                if max_nodes is not None and len(dist) > max_nodes:
                    return dist, succ, False
    return dist, succ, True


def distance(t, s, W, cap: int, radius: int = 12):
    """Exact distance t -> s under W within the radius (None if unreachable within it)."""
    if t == s:
        return 0
    dist, _, _ = bfs(t, W, cap, radius=radius, stop_at=s)
    return dist.get(s)


def reverse_distances(succ: dict, goal) -> dict:
    """Distance from every node of the expanded graph to goal, along the graph's edges."""
    pred = {}
    for u, s_ in succ.items():
        for v in s_:
            pred.setdefault(v, []).append(u)
    d = {goal: 0}
    q = deque([goal])
    while q:
        v = q.popleft()
        for u in pred.get(v, []):
            if u not in d:
                d[u] = d[v] + 1
                q.append(u)
    return d


def shortest_dag(succ: dict, dgoal: dict, start):
    """Edges on shortest paths from start to the goal: {u: {v: actions}} restricted to d(v) = d(u) - 1."""
    out = {}
    stack = [start]
    seen = {start}
    while stack:
        u = stack.pop()
        du = dgoal.get(u)
        if du is None or du == 0:
            out.setdefault(u, {})
            continue
        edges = {v: acts for v, acts in succ.get(u, {}).items() if dgoal.get(v) == du - 1}
        out[u] = edges
        for v in edges:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return out


def count_shortest(dag: dict, start, goal) -> tuple:
    """(state sequences, action sequences) counts of shortest proofs."""
    @lru_cache(maxsize=None)
    def f(u):
        if u == goal:
            return (1, 1)
        ns = na = 0
        for v, acts in dag.get(u, {}).items():
            s_, a_ = f(v)
            ns += s_
            na += a_ * len(acts)
        return (ns, na)
    return f(start)


def paths_through(dag: dict, dgoal: dict, start, goal) -> tuple:
    """For every node on the shortest DAG: (#shortest state sequences start -> node, #node -> goal). The product is the
    number of shortest proofs through the node (the RT-post visitation weight)."""
    @lru_cache(maxsize=None)
    def to_goal(u):
        if u == goal:
            return 1
        return sum(to_goal(v) for v in dag.get(u, {}))
    nodes_ = list(dag.keys())
    nodes_.sort(key=lambda u: -dgoal[u])             # farthest from the goal first: edges go one layer down
    from_start = {start: 1}
    for u in nodes_:
        for v in dag.get(u, {}):
            from_start[v] = from_start.get(v, 0) + from_start.get(u, 0)
    return from_start, {u: to_goal(u) for u in nodes_}


def enumerate_shortest(dag: dict, start, goal, limit: int = 5000, actions: bool = False) -> list:
    """Shortest proofs as state sequences [(t0, t1, ..., goal)], or as action sequences when actions=True; at most limit."""
    out = []

    def rec(u, acc):
        if len(out) >= limit:
            return
        if u == goal:
            out.append(tuple(acc))
            return
        for v, acts in dag.get(u, {}).items():
            if actions:
                for a in acts:
                    rec(v, acc + [a])
                    if len(out) >= limit:
                        return
            else:
                rec(v, acc + [v])
    rec(start, [] if actions else [start])
    return out


def binding(t, s, W, cap: int, d_W: int) -> bool:
    """The world binds when a missing axiom lengthens the proof: d_full(t, s) < d_W(t, s)."""
    if W == FULL:
        return False
    d_full = distance(t, s, FULL, cap, radius=d_W)
    return d_full is not None and d_full < d_W


# ---- strategy classes: Mazurkiewicz traces (proofs equivalent modulo swapping adjacent moves at non-overlapping nodes)

def foata_form(actions) -> tuple:
    """Canonical representative of a proof's trace class: the Foata normal form, as a tuple of sorted steps (tuples of
    independent actions). Two action sequences are equivalent iff their Foata forms are equal. Dependence = overlapping
    paths (one node an ancestor of the other, or equal); independent actions commute because a rewrite at a node leaves
    every disjoint subtree unchanged."""
    acts = list(actions)
    n = len(acts)
    done = [False] * n
    steps = []
    remaining = n
    while remaining:
        ready = []
        for j in range(n):
            if done[j]:
                continue
            if all(done[i] or not paths_overlap(acts[i][1], acts[j][1]) for i in range(j)):
                ready.append(j)
        step = tuple(sorted((acts[j] for j in ready), key=lambda a: (RULE_INDEX[a[0]], a[1])))
        for j in ready:
            done[j] = True
        remaining -= len(ready)
        steps.append(step)
    return tuple(steps)


def trace_classes(action_sequences) -> dict:
    """{foata form: number of action sequences in that class}."""
    out = {}
    for seq in action_sequences:
        f = foata_form(seq)
        out[f] = out.get(f, 0) + 1
    return out


def world(name: str):
    return world_from_name(name)
