"""Tests for the rewrite carrier (experiments/rewrite/DESIGN.md §10, G1)."""

import itertools
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
