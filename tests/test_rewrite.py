"""Tests for the rewrite carrier (experiments/rewrite/DESIGN.md §10, G1)."""

import itertools
import json
import os
import math
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


# ---------------- generation: layer sampling, records, rejection ----------------

from topology_design.rewrite.data import (make_instance, make_candidate, complete_instance, sample_term, sample_shape, acceptance_rates,  # noqa: E402
                                          category, dec_action, enc_action, canonical_proof, CATALAN)


def test_sample_shape_is_uniform_over_bracketings():
    rng = random.Random(0)
    counts = {}
    for _ in range(4200):
        counts[sample_shape(rng, 4)] = counts.get(sample_shape(rng, 4), 0) + 1
    assert len(counts) == CATALAN[3] == 5
    assert all(abs(c / 4200 - 0.2) < 0.03 for c in counts.values()), counts


def test_instance_record_is_consistent():
    rng = random.Random(5)
    W = world_from_name("AC")
    n = 0
    while n < 15:
        rec = make_instance(rng, W, 5, range(3, 6), pf=True, secondary=True)
        if rec is None:
            continue
        n += 1
        t, s = P(rec["t"]), P(rec["s"])
        acts = [dec_action(a) for a in rec["canon"]]
        v = verify(t, s, acts, W, 5)
        assert v["ok"] and v["steps"] == rec["k"] == distance(t, s, W, 5)          # canonical proof is a shortest proof
        assert acts[-1] == HALT and len(acts) == rec["k"] + 1
        # the lexicographic choice at the first step: no shortest-path action is smaller
        dist, succ, _ = bfs(t, W, 5)
        dgoal = reverse_distances(succ, s)
        dag = shortest_dag(succ, dgoal, t)
        first_ok = [a for v_, aa in dag[t].items() for a in aa]
        assert min(first_ok, key=lambda a: (RULES.index(a[0]), a[1])) == acts[0]
        # posterior states include t and s with weight 1 and targets summing to one
        states = {p["state"]: p for p in rec["post"]}
        assert states[rec["t"]]["weight"] == 1.0 and states[rec["s"]]["weight"] == 1.0
        assert all(abs(sum(p["target"].values()) - 1) < 1e-9 for p in rec["post"])
        assert states[rec["s"]]["target"] == {HALT: 1.0}
        # binding is consistent with the distances
        dF = distance(t, s, W_ALL, 5)
        assert rec["binding"] == (dF < rec["k"])
        # first-decision quantities are in range; the secondary strata bound the first decision
        f = rec["first"]
        assert 1 <= f["Dany"] <= f["Dsep"] <= rec["k"] and f["n_correct"] >= 1
        assert rec["secondary"]["canon_Dany_max"] >= f["Dany"] and rec["secondary"]["min_over_proofs_Dany_max"] >= 1
        # the ladder at d = k succeeds (oracle); the stored rungs agree with a recomputation
        assert ladder_rollout(t, s, W, 5, d=rec["k"], T=2 * rec["k"] + 2)["ok"]
        assert rec["ladder"]["1"] == ladder_rollout(t, s, W, 5, d=1, T=2 * rec["k"] + 2)["ok"]
        # PF targets sum to one; the kept states carry nearly all the visitation mass
        pf = rec["pf"]
        assert all(abs(sum(p["target"].values()) - 1) < 1e-9 for p in pf["states"])
        assert sum(p["visits"] for p in pf["states"]) >= 0.99 * pf["visits_total"] or len(pf["states"]) == 256
        assert 0 < pf["p_success"] <= 1


def test_candidate_then_completion_matches_make_instance():
    rng1, rng2 = random.Random(9), random.Random(9)
    W = world_from_name("CI")
    a = make_instance(rng1, W, 5, range(2, 6))
    c = make_candidate(rng2, W, 5, range(2, 6))
    b = complete_instance(c) if c is not None else None
    assert a == b


def test_acceptance_rates_hit_targets_when_feasible():
    r = acceptance_rates({"B": 20, "BG": 5, "G": 30, "P": 50}, 0.3, 0.3)
    assert r["feasible"] and r["expected_binding"] >= 0.3 - 1e-9 and r["expected_gp1"] >= 0.3 - 1e-9
    # all-plain warm-up: nothing to do
    r = acceptance_rates({"P": 50}, 0.0, 0.3)
    assert not r["feasible"] and r["p_plain"] == 1.0
    # binding target of zero with plenty of GP1: accept everything that keeps GP1 >= 0.3
    r = acceptance_rates({"G": 40, "P": 60}, 0.0, 0.3)
    assert r["feasible"] and r["q_gp1"] == 1.0 and abs(r["expected_gp1"] - 40 / (40 + r["p_plain"] * 60)) < 1e-9


def test_generator_populates_strata_at_cap5():
    # layer sampling in {A, C}: k >= 3 is the majority, binding and GP1 instances both appear (G1)
    rng = random.Random(11)
    W = world_from_name("AC")
    recs = []
    while len(recs) < 60:
        c = make_candidate(rng, W, 5, range(1, 6))
        if c is not None:
            recs.append(c["rec"])
    ks = [r["k"] for r in recs]
    cats = [category(r) for r in recs]
    assert sum(k >= 3 for k in ks) / len(ks) >= 0.5, ks
    assert "B" in cats and ("G" in cats or any(r["first"]["GP1"] for r in recs)), cats
    assert all(enc_action(dec_action(a)) == a for a in ("Af@", "C@LR", "IbL@R", HALT))


# ---------------- model, rollouts, training harness ----------------

import torch  # noqa: E402

from topology_design.rewrite.model import (encode, collate, Router, matched_configs, action_slot, slot_action, HALT_INDEX,  # noqa: E402
                                           N_ACTIONS, VOCAB, SEG, N_RULES, RULE_INDEX)
from topology_design.rewrite import train as T  # noqa: E402


def test_matched_configs_within_one_percent_and_pointer_addresses_size7():
    cfg_s, cfg_a, n_s, n_a = matched_configs()
    assert abs(n_s - n_a) / n_a <= 0.01
    enc = encode("ACID", "(((a+b)+c)+((d+0)+(a+b)))", "((a+(b+(c+d)))+(0+(a+b)))")
    assert len(enc["term_paths"]) == 13
    b = collate([enc])
    out = Router(cfg_s)(b)
    assert out.shape == (1, N_ACTIONS) and torch.isfinite(out).all()            # every slot addressable at size 7
    small = collate([encode("AC", "(a+b)", "(b+a)")])
    out = Router(cfg_s)(small)
    assert torch.isfinite(out[0, :3 * N_RULES]).all() and not torch.isfinite(out[0, 3 * N_RULES:HALT_INDEX]).any() and torch.isfinite(out[0, HALT_INDEX])
    assert torch.isfinite(Router(cfg_a)(small)).all()                              # the categorical head addresses everything
    paths = enc["term_paths"]
    assert slot_action(action_slot(("C", "RL"), {p: i for i, p in enumerate(paths)}), paths) == ("C", "RL")
    assert slot_action(HALT_INDEX, paths) == HALT and slot_action(12 * N_RULES + 7, ["", "L", "R"])[1] is None


_INV = {v: k for k, v in VOCAB.items()}


def _decode_tree(toks):
    """Prefix tokens -> term."""
    it = iter(toks)

    def rec():
        x = next(it)
        if x == "+":
            return (rec(), rec())
        return x
    return rec()


class OracleRouter:
    """A planted router for the harness: reads the batch back into (world, s, t) and emits the lexicographically first
    shortest action (HALT at the goal). Acts through the same slots as the learned router."""
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, b):
        B = b["tok"].shape[0]
        out = torch.full((B, N_ACTIONS), -10.0)
        for i in range(B):
            toks = [_INV[int(x)] for x, m in zip(b["tok"][i], b["mask"][i]) if m]
            segs = [int(x) for x, m in zip(b["seg"][i], b["mask"][i]) if m]
            world = "".join(tk[0] for tk in toks[:4] if tk.endswith("+"))
            goal = _decode_tree([tk for tk, sg in zip(toks, segs) if sg == SEG["goal"]])
            term = _decode_tree([tk for tk, sg in zip(toks, segs) if sg == SEG["term"]])
            W = world_from_name(world)
            if term == goal:
                out[i, HALT_INDEX] = 10.0
                continue
            dist, succ, _ = bfs(term, W, 7)
            dgoal = reverse_distances(succ, goal)
            dag = shortest_dag(succ, dgoal, term)
            a = canonical_proof(dag, term, goal)[0]
            pi = {p: j for j, (p, _, _) in enumerate(nodes(term))}
            out[i, action_slot(a, pi)] = 10.0
        return out


def _tiny_dataset(tmp_path, n=6):
    import gzip, json
    rng = random.Random(21)
    recs_by = {}
    for wn in ("AC", "CI"):
        W = world_from_name(wn)
        recs = []
        while len(recs) < n:
            r = make_instance(rng, W, 5, range(2, 6), pf=True)
            if r is not None and r["k"] >= 2:
                recs.append(r)
        recs_by[wn] = recs
        for tag in ("train", "eval5"):
            with gzip.open(tmp_path / f"{tag}_{wn}.jsonl.gz", "wt") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
    return recs_by


def test_oracle_router_solves_every_instance_in_the_harness(tmp_path):
    recs_by = _tiny_dataset(tmp_path)
    recs = [T.prepare(r) for rs in recs_by.values() for r in rs]
    for masked in (True, False):
        outs = T.rollout(OracleRouter(), recs, "cpu", masked=masked)
        assert all(o["ok"] and o["first_ok"] and o["category"] == "ok" and o["steps"] == r["k"] for o, r in zip(outs, recs))
    # a router that always halts fails every instance with "wrong halt"; one that picks an out-of-world rule loops
    class Halter(OracleRouter):
        def __call__(self, b):
            out = torch.full((b["tok"].shape[0], N_ACTIONS), -10.0); out[:, HALT_INDEX] = 10.0; return out
    outs = T.rollout(Halter(), recs, "cpu", masked=False)
    assert all(o["category"] == "wrong halt" and not o["ok"] for o in outs)

    class OutOfWorld(OracleRouter):
        def __call__(self, b):
            out = torch.full((b["tok"].shape[0], N_ACTIONS), -10.0); out[:, RULE_INDEX["Db"]] = 10.0; return out   # Db at the root
    outs = T.rollout(OutOfWorld(), [r for r in recs if r["world"] in ("AC", "CI")], "cpu", masked=False)
    assert all(o["category"] == "invalid: out of world" for o in outs)
    # masked: the same router cannot choose Db; it takes the best valid slot instead (some valid action), no invalid category
    outs = T.rollout(OutOfWorld(), recs, "cpu", masked=True)
    assert not any(o["category"].startswith("invalid") for o in outs)


def test_training_rows_and_a_few_steps_of_every_arm(tmp_path):
    recs_by = _tiny_dataset(tmp_path)
    sets = {"eval5": T.load_set(str(tmp_path), "eval5", ["AC", "CI"])}
    rng = random.Random(0)
    rec = T.prepare(recs_by["AC"][0])
    rows = T.rows_rt(rec, rec["world"], None, rng)
    assert len(rows) == rec["k"] + 1 and all(sum(r[4].values()) == 1.0 for r in rows) and list(rows[-1][4]) == [HALT_INDEX]
    rows = T.rows_post(rec, rec["world"], None, rng)
    assert abs(sum(r[5] for r in rows) - (rec["k"] + 1)) < 1e-9 or len(rows) == 24
    rows = T.rows_pf(rec, rec["world"], None, rng)
    assert len(rows) == rec["k"] + 1 and abs(sum(r[5] for r in rows) - (rec["k"] + 1)) < 1e-9
    for arm in ("RT-canon", "RT-post", "PF", "RT-absolute", "RT-examples", "RT-shuffled", "EI16"):
        cfg = {"arm": arm, "seed": 1, "lr": 1e-3, "steps": 2, "rows": 16, "inst": 4, "eval_every": 2, "warmup": 1, "worlds": ["AC", "CI"]}
        res = T.run(cfg, str(tmp_path), device="cpu", log=lambda *a: None, sets=sets)
        assert len(res["curve"]) == 2 and "per_instance" in res["final"]["eval5"]
        assert all(math.isfinite(l) for l in res["losses"]) or arm == "EI16"


def test_beam_search_finds_the_oracle_route_under_a_planted_router(tmp_path):
    recs_by = _tiny_dataset(tmp_path)
    recs = [T.prepare(r) for r in recs_by["AC"]]
    found = T.beam_search(OracleRouter(), recs, "cpu", W=4, eps=0.0)
    for rec, route in zip(recs, found):
        assert route is not None
        acts = [a for _, a in route]
        v = verify(P(rec["t"]), P(rec["s"]), acts, world_from_name(rec["world"]), rec["cap"])
        assert v["ok"] and v["steps"] == rec["k"]


# ---------------- analysis on fixtures cut from real run records (L33) ----------------

import importlib.util  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "rewrite")


def _analyze():
    spec = importlib.util.spec_from_file_location("rw_analyze", os.path.join(os.path.dirname(__file__), "..", "experiments", "rewrite", "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analysis_on_real_fixtures_is_consistent():
    A = _analyze()
    runs = A.load_results(FIX)
    assert {A.arm_of(r) for r in runs} == {"RT-canon", "RT-post", "PF", "EI16", "RT-all", "RT-absolute", "RT-examples", "RT-shuffled"}
    for r in runs:
        # the final cells' n equal the per-instance counts, and the cells' successes equal the per-instance successes
        for tag in ("eval5", "eval7", "evalk78"):
            cells = r["final"][tag]["cells"]
            pis = A.instances(r, tag)
            assert sum(c["n"] for c in cells.values()) == len(pis)
            assert sum(c["ok_m"] for c in cells.values()) == sum(pi["ok_m"] for pi in pis)
            for key, c in cells.items():
                w, b, g, k = key.split("|")
                sel = A.instances(r, tag, world=w, binding=bool(int(b)), GP1=bool(int(g)), k=int(k))
                assert len(sel) == c["n"] and sum(pi["ok_m"] for pi in sel) == c["ok_m"]
        # the curve's last point is the final evaluation (full sets); the per-instance record carries the P6 fields
        xs, ys = A.curve_series(r, "eval5", A.TRAIN_WORLDS)
        assert xs[-1] == r["cfg"]["steps"] and all(0 <= y <= 1 for y in ys)
        pi = A.instances(r, "eval5")[0]
        assert abs(sum(pi["target_first"].values()) - 1) < 1e-9 and set(pi["p_first"]) == set(pi["target_first"])
        assert "visited_goal_m" in pi and "first_canon" in pi
    derived = json.load(open(os.path.join(FIX, "derived.json")))
    md = A.report(runs, derived)
    for sec in ("## P2", "## P3", "## P4", "## P5", "## P6", "## P7", "## P8", "Failure categories", "## P1"):
        assert sec in md, sec
    # every P3 row of RT-absolute carries the addressability split from the derived fixture
    p3 = md[md.index("## P3"):md.index("## P4")]
    assert "addressable" in [l for l in p3.splitlines() if l.startswith("| RT-absolute | train")][0]
    # ladder rates on a selection are read from the instances themselves
    sel = A.instances(runs[0], "eval5", world=["AC"], binding=True)
    assert sel and 0 <= A.ladder_rate(sel, 1) <= 1


def test_analysis_planted_outcomes():
    A = _analyze()
    runs = A.load_results(FIX)
    r = [x for x in runs if A.arm_of(x) == "RT-canon"][0]
    # plant: every instance solved -> rate 1; P2's joint condition becomes 1/1
    for pi in r["final"]["eval5"]["per_instance"]:
        pi["ok_m"] = True
    assert A.rate(A.instances(r, "eval5")) == 1.0
    md = A.report([r], None)
    p2 = md[md.index("## P2"):md.index("## P3")]
    row = [l for l in p2.splitlines() if l.startswith("| RT-canon | pair |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[7] == "1/1" and cells[8] == "1/1"
    # P4 at zero success never passes: with the fixture's untrained router (success 0) the strict rule gives 0/1 or
    # "rungs not separated", never a pass (review 2, B5)
    r0 = A.load_results(FIX)
    r0 = [x for x in r0 if A.arm_of(x) == "RT-canon"][0]
    md0 = A.report([r0], None)
    p4 = md0[md0.index("## P4"):md0.index("## P5") if "## P5" in md0 else md0.index("## P6")]
    p4_rows = [l for l in p4.splitlines() if l.startswith("| RT-canon |")]
    assert len(p4_rows) == 3                                   # the §16 cells, the §6 (GP1-only) cells, the upper-only sensitivity set
    for row in p4_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[9] in ("0/1", "rungs not separated")
    # classification from curves cut from the real curve: a collapse and a learned run
    assert A.classify([0.0, 0.3, 0.8, 0.9, 0.9, 0.6]) == "learned-then-collapsed"
    assert A.classify([0.0, 0.3, 0.8, 0.9, 0.92, 0.91]) == "learned"
    assert A.classify([0.0, 0.02, 0.05, 0.1, 0.18, 0.3]) == "slow-rising"
    assert A.classify([0.0, 0.01, 0.0, 0.02, 0.01, 0.0]) == "flat"


def test_matched_p3_ratio_and_separated_cells_on_planted_fixture():
    A = _analyze()
    r = [x for x in A.load_results(FIX) if A.arm_of(x) == "RT-canon"][0]
    # plant: size-5 success 1 on every cell and sizes 6-7 success 1 on half the instances -> matched ratio 0.5 (cells all covered)
    for pi in r["final"]["eval5"]["per_instance"]:
        pi["ok_m"] = True
    e7 = r["final"]["eval7"]["per_instance"]
    for i, pi in enumerate(e7):
        pi["ok_m"] = (i % 2 == 0)
    md = A.report([r], None)
    p3 = md[md.index("## P3"):md.index("## P4")]
    row = [l for l in p3.splitlines() if l.startswith("| RT-canon | train")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    med = float(cells[4].split()[0])                      # columns: arm, worlds, size<=5, sizes 6-7, matched ratio, ...
    assert 0.35 <= med <= 0.65, cells


def test_p4_amended_cells_on_planted_fixture():
    """P4's pass reading (DESIGN §16): a (world, binding, GP1, k) cell of n ≥ 10 whose rungs separate; a router strictly
    between the d = 2 and d = 3 rungs passes, one at the d = 3 rung does not; the §6 (GP1-only) row stays unreadable
    when the separated cell is a non-GP1 one. Built from real records of the fixture with planted ladder outcomes."""
    import copy
    A = _analyze()
    r = [x for x in A.load_results(FIX) if A.arm_of(x) == "RT-canon"][0]
    base = [pi for pi in r["final"]["eval5"]["per_instance"] if pi["world"] == "ID" and pi["k"] == 4 and pi["binding"] and not pi["GP1"]]
    assert base, "the fixture has an ID binding non-GP1 k=4 instance"
    cell = []
    for i in range(10):                                       # rungs 0.2 / 0.5 / 0.9 on ten copies: separated
        pi = copy.deepcopy(base[0])
        pi["ladder"] = {"1": i < 2, "2": i < 5, "3": i < 9}
        pi["ok_m"] = i < 7                                    # 0.7: strictly between d = 2 and d = 3
        cell.append(pi)
    others = [pi for pi in r["final"]["eval5"]["per_instance"] if not (pi["world"] == "ID" and pi["k"] == 4)]
    r["final"]["eval5"]["per_instance"] = others + cell
    md = A.report([r], None)
    p4 = md[md.index("## P4"):md.index("## P5") if "## P5" in md else md.index("## P6")]
    rows = {l.split("|")[2].strip(): [c.strip() for c in l.strip("|").split("|")] for l in p4.splitlines() if l.startswith("| RT-canon |")}
    assert rows["§16 cells"][2] == "10" and rows["§16 cells"][9] == "1/1", rows["§16 cells"]
    assert rows["§6 cells (GP1)"][9] == "rungs not separated"
    for pi in cell:                                           # at the d = 3 rung: not strictly between
        pi["ok_m"] = True
    md = A.report([r], None)
    p4 = md[md.index("## P4"):md.index("## P5") if "## P5" in md else md.index("## P6")]
    row = [c.strip() for c in [l for l in p4.splitlines() if l.startswith("| RT-canon | §16")][0].strip("|").split("|")]
    assert row[9] == "0/1" and row[10] == "1/1", row
    # the derive step's cell set is checked against the records: a matching set is acknowledged, a different one flagged
    md = A.report([r], {"p4_amended_pooled": {"cells": ["ID|1|0|4"], "d1r": 0.2, "d2r": 0.5, "d3r": 0.9, "cells_lower_separation_fails_random": []}})
    assert "recomputed from the run records equal the derive step's (1 cells)" in md and "Random-tie-break rungs" in md
    md = A.report([r], {"p4_amended_pooled": {"cells": ["AC|1|0|4"]}})
    assert "P4 CELL SET MISMATCH" in md


def _load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "rewrite", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not part of this bundle")
    import sys
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("rw_" + name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_session_budget_guard_and_shard_balance():
    """A run that cannot finish inside what is left of the session is not started (a Kaggle session that hits its limit
    saves nothing); the main grid's four shards are balanced under the costs measured in the pilot, EI16 first."""
    R = _load_script("run_rewrite.py")
    ei, rt = {"arm": "EI16", "steps": 12000, "name": "e"}, {"arm": "RT-canon", "steps": 12000, "name": "r"}
    assert R.fits(ei, None) and R.fits(rt, None)                       # no budget
    assert not R.fits(ei, 3600) and R.fits(rt, 3600)                   # EI16 needs about 2.4 h, RT-canon about 21 min
    assert not R.fits(rt, 3600, rate=3 * R.SECONDS_PER_UNIT)           # a shard that has run three times slower than the pilot
    seeds = {"RT-canon": 10, "RT-post": 10, "RT-all": 3, "RT-absolute": 5, "EI16": 5, "PF": 3, "RT-examples": 3, "RT-shuffled": 2}
    grid = [{"arm": a, "steps": 12000, "name": f"{a}_s{s}"} for a, n in seeds.items() for s in range(n)]
    shards, loads = R.assign_shards(grid, 4)
    assert sorted(c["name"] for s in shards for c in s) == sorted(c["name"] for c in grid)
    assert max(loads) / min(loads) < 1.1
    assert all(s[0]["arm"] == "EI16" for s in shards)                  # the long runs start first
    assert sum(1 for s in shards if any(c["arm"] == "PF" for c in s)) == 1   # the memory-heavy runs never side by side
    assert 20 < sum(loads) * R.SECONDS_PER_UNIT / 3600 < 35            # the pilot's projection: about 27 process-hours


def test_pilot_read_lr_rule_on_planted_records():
    """The pilot's lr rule (PILOT.md item 1): the lr with the highest training-world eval5 masked success at the last
    step, ties by the lower loss — on records cut from the real fixtures with planted outcomes; held-out rows do not count."""
    import copy
    P = _load_script("pilot_read.py")
    A = _analyze()
    runs = A.load_results(FIX)
    md, chosen, cost = P.report(runs)
    assert set(chosen) == {A.arm_of(r) for r in runs} and "**holds**" in md
    base = [r for r in runs if A.arm_of(r) == "RT-canon"][0]

    def variant(lr, train_ok, held_ok, loss):
        r = copy.deepcopy(base)
        r["cfg"]["lr"] = lr
        for key, c in r["final"]["eval5"]["cells"].items():
            c["ok_m"] = int(round(c["n"] * (train_ok if key.split("|")[0] in A.TRAIN_WORLDS else held_ok)))
            c["ok_u"] = min(c["ok_u"], c["ok_m"])
        r["curve"][-1]["loss"] = loss
        return r

    others = [r for r in runs if A.arm_of(r) != "RT-canon"]
    # the higher training-world success wins even when its held-out rows are worse
    _, chosen, _ = P.report(others + [variant(3e-4, 0.5, 1.0, 0.1), variant(3e-3, 1.0, 0.0, 0.9)])
    assert chosen["RT-canon"] == 3e-3
    # a tie on success goes to the lower loss
    _, chosen, _ = P.report(others + [variant(3e-4, 1.0, 0.0, 0.2), variant(3e-3, 1.0, 0.0, 0.9)])
    assert chosen["RT-canon"] == 3e-4
