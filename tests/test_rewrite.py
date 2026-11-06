"""Tests for the rewrite carrier (experiments/rewrite/DESIGN.md §10, G1)."""

import itertools
import random

import pytest

from topology_design.rewrite.term import parse, to_str, prefix, nodes, replace_at, subterm_at, size, nnodes, iter_terms, paths_overlap, structural_features
from topology_design.rewrite.rules import RULES, HALT, step, valid_actions, successors, verify, apply_local, world_from_name, WORLDS
from topology_design.rewrite.search import (bfs, distance, reverse_distances, shortest_dag, count_shortest, enumerate_shortest,
                                            binding, foata_form, trace_classes, paths_through, FULL)

W_ALL = FULL
P = parse


def test_terms_roundtrip_and_positions():
    t = P("((a+b)+0)")
    assert to_str(t) == "((a+b)+0)" and prefix(t) == ["+", "+", "a", "b", "0"]
    assert size(t) == 3 and nnodes(t) == 5
    paths = [p for p, _, _ in nodes(t)]
    assert paths == ["", "L", "LL", "LR", "R"]
    assert subterm_at(t, "LR") == "b" and replace_at(t, "L", "c") == P("(c+0)")
    assert paths_overlap("L", "LR") and not paths_overlap("L", "R") and paths_overlap("", "R")
    feats = structural_features(t)
    assert feats[0] == (0, "", 3, False) and feats[2] == (2, "LL", 1, True)


def test_rules_on_hand_built_terms():
    assert apply_local("Af", P("((a+b)+c)")) == P("(a+(b+c))")
    assert apply_local("Ab", P("(a+(b+c))")) == P("((a+b)+c)")
    assert apply_local("Af", P("(a+(b+c))")) is None
    assert apply_local("C", P("(a+b)")) == P("(b+a)") and apply_local("C", "a") is None
    assert apply_local("If", P("(a+0)")) == "a" and apply_local("If", P("(0+a)")) == "a" and apply_local("If", P("(0+0)")) == "0"
    assert apply_local("If", P("(a+b)")) is None
    assert apply_local("IbR", "a") == P("(a+0)") and apply_local("IbL", "a") == P("(0+a)")
    assert apply_local("Df", P("(a+a)")) == "a" and apply_local("Df", P("(a+b)")) is None
    assert apply_local("Db", "a") == P("(a+a)")
    # identity is invertible both ways (review 1 S9)
    for u in (P("(a+b)"), "c"):
        assert apply_local("If", apply_local("IbR", u)) == u and apply_local("If", apply_local("IbL", u)) == u


def test_executor_validity_out_of_world_and_cap():
    t = P("((a+b)+0)")
    assert step(t, "If", "", W_ALL, 5) == P("(a+b)")
    assert step(t, "If", "", world_from_name("AC"), 5) is None            # out of world
    assert step(t, "Af", "R", W_ALL, 5) is None                          # no match at a leaf
    assert step(t, "C", "LRX", W_ALL, 5) is None                         # no such node
    assert step(t, "Db", "", W_ALL, 5) is None and step(t, "Db", "", W_ALL, 6) is not None   # cap
    # self-loops are excluded from the candidate set
    acts = valid_actions(P("(a+a)"), W_ALL, 5)
    assert all(new != P("(a+a)") for _, _, new in acts)
    assert ("C", "") not in [(r, p) for r, p, _ in acts]


def test_verify_routes():
    t, s = P("((a+b)+0)"), P("(b+a)")
    good = [("If", ""), ("C", ""), HALT]
    assert verify(t, s, good, W_ALL, 5)["ok"]
    r = verify(t, s, [("If", ""), ("C", "")], W_ALL, 5)
    assert r["ok"] and r["steps"] == 2                                   # halt implied at the end
    r = verify(t, s, [("If", ""), HALT], W_ALL, 5)
    assert not r["ok"] and r["reason"] == "wrong halt"
    r = verify(t, s, [("C", ""), ("If", "")], world_from_name("I"), 5)
    assert not r["ok"] and r["reason"] == "out of world" and r["first_invalid"] == 0
    r = verify(t, s, [("Af", "R")], W_ALL, 5)
    assert not r["ok"] and r["reason"] == "no match"


def brute_force_distance(t, s, W, cap, radius=8):
    """Breadth-first over the executor itself, independent of search.bfs."""
    frontier, seen = {t}, {t}
    for d in range(radius + 1):
        if s in frontier:
            return d
        nxt = set()
        for u in frontier:
            for _, _, v in valid_actions(u, W, cap):
                if v not in seen:
                    seen.add(v); nxt.add(v)
        frontier = nxt
    return None


def test_bfs_matches_brute_force_on_small_terms():
    rng = random.Random(0)
    terms = list(iter_terms(3))
    for W in (world_from_name("AC"), world_from_name("CI"), W_ALL):
        for _ in range(30):
            t, s = rng.choice(terms), rng.choice(terms)
            assert distance(t, s, W, 4, radius=8) == brute_force_distance(t, s, W, 4)


def test_reverse_distances_shortest_dag_and_counts():
    W = world_from_name("AC")
    t = P("((a+b)+c)")
    dist, succ, complete = bfs(t, W, 5)
    assert complete
    s = P("(c+(b+a))")
    dgoal = reverse_distances(succ, s)
    assert dgoal[t] == dist[s]
    dag = shortest_dag(succ, dgoal, t)
    ns, na = count_shortest(dag, t, s)
    seqs = enumerate_shortest(dag, t, s)
    assert ns == len(seqs) and all(len(q) == dgoal[t] + 1 and q[-1] == s for q in seqs)
    acts = enumerate_shortest(dag, t, s, actions=True)
    assert na == len(acts)
    for a in acts:
        assert verify(t, s, list(a), W, 5)["ok"]
    fs, tg = paths_through(dag, dgoal, t, s)
    assert fs[s] == ns and tg[t] == ns


def test_binding_test():
    # under {A, C} the identity cannot be removed, so (a+0) -> a binds when I is missing; under I alone it does not
    assert binding(P("(a+0)"), "a", world_from_name("I"), 5, d_W=1) is False
    t, s = P("((a+0)+b)"), P("(a+b)")
    dW = distance(t, s, world_from_name("ACD"), 5)
    assert dW is None or dW > 1
    # a world where the detour exists and is longer than the full-world proof
    W = world_from_name("CI")
    t, s = P("((a+b)+c)"), P("(c+(a+b))")      # C at root: distance 1 in CI; full world also 1 -> not binding
    assert binding(t, s, W, 5, d_W=distance(t, s, W, 5)) is False
    # a real binding instance: under {A, C} the 0 must be carried around the tree (3 moves); with I it is removed and
    # reinserted (2 moves)
    W = world_from_name("AC")
    t, s = P("(a+(b+0))"), P("((0+a)+b)")
    dW, dF = distance(t, s, W, 5), distance(t, s, FULL, 5)
    assert (dW, dF) == (3, 2) and binding(t, s, W, 5, d_W=dW) is True
    t, s = P("((a+0)+(b+c))"), P("((a+b)+(c+0))")
    dW, dF = distance(t, s, W, 5), distance(t, s, FULL, 5)
    assert (dW, dF) == (4, 3) and binding(t, s, W, 5, d_W=dW) is True


def test_trace_classes_commute_only_disjoint_moves():
    # C at L and C at R commute; C at root and C at L do not
    a = [("C", "L"), ("C", "R")]
    b = [("C", "R"), ("C", "L")]
    assert foata_form(a) == foata_form(b)
    c = [("C", ""), ("C", "L")]
    d = [("C", "L"), ("C", "")]
    assert foata_form(c) != foata_form(d)
    assert len(trace_classes([a, b, c, d])) == 3
    # equivalence is respected by the executor: both orders of commuting moves give the same term
    t = P("((a+b)+(c+d))")
    ua = verify(t, P("((b+a)+(d+c))"), a, W_ALL, 5); ub = verify(t, P("((b+a)+(d+c))"), b, W_ALL, 5)
    assert ua["ok"] and ub["ok"]


def test_worlds_enumeration():
    assert len(WORLDS) == 15 and WORLDS[0] == frozenset("A") and WORLDS[-1] == FULL


# ---------------- proxies, per-decision difficulty, the ladder ----------------

from topology_design.rewrite.proxy import ted, sed, decision, lookahead_value, ladder_policy, ladder_rollout  # noqa: E402


def test_proxies_on_hand_built_pairs():
    assert ted(P("(a+b)"), P("(a+b)")) == 0 and sed(P("(a+b)"), P("(a+b)")) == 0
    assert ted(P("(a+b)"), P("(a+c)")) == 1                       # one rename
    assert ted("a", P("(a+0)")) == 2                               # insert a node and a leaf
    assert ted(P("(a+b)"), P("(b+a)")) == 2 and sed(P("(a+b)"), P("(b+a)")) == 2
    assert ted(P("((a+b)+c)"), P("(a+(b+c))")) >= 1


def _instance_with_graph(t, s, W, cap):
    dist, succ, complete = bfs(t, W, cap)
    assert complete and s in dist
    dgoal = reverse_distances(succ, s)
    return dist, succ, dgoal


def test_decision_invariants_and_oracle_rung():
    rng = random.Random(1)
    W, cap = world_from_name("AC"), 5
    terms = [u for u in iter_terms(4) if size(u) >= 3]
    checked = 0
    for _ in range(60):
        t = rng.choice(terms)
        dist, succ, complete = bfs(t, W, cap, radius=5)
        far = [u for u, d in dist.items() if 2 <= d <= 4]
        if not far:
            continue
        s = rng.choice(far)
        dgoal = reverse_distances(succ, s)
        if t not in dgoal:
            continue
        info = decision(t, s, W, cap, dgoal)
        assert 1 <= info["Dany"] <= info["Dsep"] <= info["d"], info
        if info["GP1"]:
            assert info["Dsep"] >= 2 and not info["tie1"]
        # the rung whose lookahead equals the remaining distance always succeeds; the greedy rung may not
        assert ladder_rollout(t, s, W, cap, d=info["d"], T=2 * info["d"] + 2)["ok"]
        checked += 1
    assert checked >= 20


def test_garden_paths_exist_and_greedy_fails_on_some():
    rng = random.Random(2)
    W, cap = world_from_name("AC"), 5
    terms = [u for u in iter_terms(5) if size(u) >= 4]
    gp, greedy_fail, n = 0, 0, 0
    for _ in range(150):
        t = rng.choice(terms)
        dist, succ, _ = bfs(t, W, cap, radius=5)
        far = [u for u, d in dist.items() if d == 4]
        if not far:
            continue
        s = rng.choice(far)
        dgoal = reverse_distances(succ, s)
        info = decision(t, s, W, cap, dgoal)
        gp += info["GP1"]
        greedy_fail += not ladder_rollout(t, s, W, cap, d=1, T=10)["ok"]
        n += 1
        if n >= 40:
            break
    assert n >= 20 and gp >= 1 and greedy_fail >= 1, (n, gp, greedy_fail)


def test_lookahead_value_ordering_past_remaining_distance():
    # from a state one step from the goal, a d = 3 lookahead must still prefer the direct move (steps break the σ tie)
    W, cap = world_from_name("AC"), 5
    s = P("(b+a)")
    t = P("(a+b)")
    acts = valid_actions(t, W, cap)
    direct = [a for a in acts if a[2] == s][0]
    vals = {new: lookahead_value(new, s, W, cap, 3, ted) for _, _, new in acts}
    assert vals[direct[2]] == (0, 0) and all(vals[new] >= (0, 0) for new in vals)
    assert ladder_policy(t, s, W, cap, d=3)[2] == s


# ---------------- exact posteriors (RT-post, PF) ----------------

from topology_design.rewrite.posterior import shortest_posterior, pf_posterior  # noqa: E402


def _brute_pf(t, s, W, cap, T):
    """Enumerate every route of at most T actions under the flat prior (uniform over valid state-changing actions ∪ HALT,
    forced halt at T) and accumulate the success-conditioned mass per (state, action) and per state visit."""
    mass, visits, total = {}, {}, [0.0]

    def credit(trail, u, q):
        total[0] += q
        for x, a in trail:
            mx = mass.setdefault(x, {})
            mx[a] = mx.get(a, 0.0) + q
        mu = mass.setdefault(u, {})
        mu[HALT] = mu.get(HALT, 0.0) + q
        for x in [x for x, _ in trail] + [u]:
            visits[x] = visits.get(x, 0.0) + q

    def rec(u, k, p, trail):
        acts = valid_actions(u, W, cap)
        n = len(acts) + 1
        if k == T:
            if u == s:
                credit(trail, u, p)
            return
        if u == s:
            credit(trail, u, p / n)
        for rule, path, new in acts:
            rec(new, k + 1, p / n, trail + [(u, (rule, path))])

    rec(t, 0, 1.0, [])
    return total[0], mass, visits


@pytest.mark.parametrize("W_name,t,s,T", [("AC", "((a+b)+c)", "(c+(b+a))", 3), ("CI", "(a+0)", "(0+a)", 3), ("AC", "(a+(b+0))", "((0+a)+b)", 4)])
def test_pf_posterior_matches_route_enumeration(W_name, t, s, T):
    W, cap = world_from_name(W_name), 4
    t, s = P(t), P(s)
    r = pf_posterior(t, s, W, cap, T)
    p_succ, mass, visits = _brute_pf(t, s, W, cap, T)
    assert abs(r["p_success"] - p_succ) < 1e-12 and p_succ > 0
    assert set(r["states"]) == set(mass)
    for u, info in r["states"].items():
        tgt = info["target"]
        assert abs(sum(tgt.values()) - 1) < 1e-9
        tot = sum(mass[u].values())
        for a in set(tgt) | set(mass[u]):
            assert abs(tgt.get(a, 0.0) - mass[u].get(a, 0.0) / tot) < 1e-9, (u, a)
        assert abs(info["visits"] - visits[u] / p_succ) < 1e-9
    # the start is visited at least once on every successful route; the goal's target keeps halting mass
    assert r["states"][t]["visits"] >= 1 - 1e-9
    assert r["states"][s]["target"][HALT] > 0


def test_shortest_posterior_is_uniform_over_shortest_proofs():
    W, cap = world_from_name("AC"), 5
    t, s = P("((a+b)+c)"), P("(c+(b+a))")
    sp = shortest_posterior(t, s, W, cap)
    dist, succ, _ = bfs(t, W, cap)
    dgoal = reverse_distances(succ, s)
    dag = shortest_dag(succ, dgoal, t)
    seqs = enumerate_shortest(dag, t, s)
    d = dgoal[t]
    # weight = share of shortest state sequences through the state; the weights sum to the number of states per proof
    for u, info in sp.items():
        share = sum(u in q for q in seqs) / len(seqs)
        assert abs(info["weight"] - share) < 1e-12
        assert abs(sum(info["target"].values()) - 1) < 1e-12
        assert set(info["target"]) == ({HALT} if u == s else {a for v, acts in dag[u].items() for a in acts})
    assert abs(sum(info["weight"] for info in sp.values()) - (d + 1)) < 1e-12
    assert sp[t]["weight"] == 1.0 and sp[s]["weight"] == 1.0
    # the action target at a state is the next-state distribution under a uniform draw of a shortest proof through it
    for u, info in sp.items():
        if u == s:
            continue
        through = [q for q in seqs if u in q]
        nxt = {}
        for q in through:
            v = q[q.index(u) + 1]
            nxt[v] = nxt.get(v, 0) + 1 / len(through)
        for v, acts in dag[u].items():
            assert abs(sum(info["target"][a] for a in acts) - nxt.get(v, 0.0)) < 1e-12


def test_shortest_posterior_weights_count_paths_not_actions():
    # two actions producing the same successor (C at a symmetric node) do not double the state's weight
    W, cap = world_from_name("CI"), 5
    t, s = P("(a+0)"), P("(0+a)")              # C at root: one state sequence; IbL/If routes are longer
    sp = shortest_posterior(t, s, W, cap)
    assert set(sp) == {t, s} and sp[t]["target"] == {("C", ""): 1.0}
