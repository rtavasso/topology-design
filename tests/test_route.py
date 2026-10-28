"""Tests for rung 1 (experiments/route/DESIGN.md §11, gates G1–G2): the generator, the exact executor and route
marginal, the planted router and the planted failures, and the trainers' run independence."""

import itertools

import numpy as np
import pytest
import torch

from topology_design.route.data import RouteConfig, generate, check_episode, word_class, follow, forced_word, H, P
from topology_design.route.exec import (tables, maps, marginal, committed, oracle_route_ok, readout, remaining_distance,
                                        HOP, PREV, HALT)
from topology_design.route.router import StackedRouter, RouterConfig, plant
from topology_design.route.pool import to_dict, EpisodePool, stack_batches, strata_episodes, STRATA

DC = RouteConfig()


def _batch(ep, R=1):
    d = to_dict(ep, "cpu")
    return {k: v.unsqueeze(0).expand(R, *v.shape).clone() for k, v in d.items()}


def test_generator_chains_words_and_adjacency():
    rng = np.random.default_rng(0)
    ep = generate(DC, 32, rng)
    check_episode(ep, DC)
    # the prev slot is the previous pair's key; the first pair's prev is DEAD
    assert (ep.pair_prev[:, 1:] == ep.pair_key[:, :-1]).all() and (ep.pair_prev[:, 0] == DC.DEAD).all()
    # training words: no two consecutive P's, last step H
    for b in range(32):
        for j in range(DC.Qn):
            w = ep.q_word[b, j, :ep.q_k[b, j]]
            assert w[-1] == H and word_class(w)["max_prun"] <= 1
    # forced strata produce their classes
    for kind, k, test in (("pp", 8, lambda c: c["max_prun"] >= 2), ("alt", 8, lambda c: c["nP"] == 4),
                          ("deepP", 8, lambda c: c["deepest_P"] >= 6), ("none", 4, lambda c: c["nP"] == 0),
                          ("train", 32, lambda c: c["max_prun"] <= 1 and c["k"] == 32)):
        e = generate(DC, 6, rng, forced_k=k, forced_kind=kind, Qn=3)
        check_episode(e, DC)
        assert all(test(word_class(e.q_word[b, 0, :k])) for b in range(6)), kind


def test_executor_oracle_rule_and_remaining_distance():
    rng = np.random.default_rng(1)
    ep = generate(DC, 6, rng, forced_k=7, forced_kind="pp", Qn=3)
    tb = tables(ep, DC, "cpu")
    B, S = 6, DC.n_sym
    # the written rule as a Markov policy: HALT on answers, PREV where the value is ARROW, HOP otherwise
    sym = torch.arange(S).unsqueeze(0).expand(B, S)
    is_key = sym < DC.n_keys; is_ans = (sym >= DC.n_keys) & (sym < DC.n_keys + DC.n_ans)
    arrow = is_key & (tb["hop"] == DC.DEAD)
    pi = torch.stack([(is_key & ~arrow).float(), arrow.float(), (~is_key).float()], -1)
    R = readout(DC, 0.02, "cpu")
    q, es, prof, alive = marginal(pi, tb["hop"], tb["prev"], tb["q_start"], 40, R)
    assert torch.allclose(q.gather(-1, (tb["q_ans"] - DC.n_keys).unsqueeze(-1)).squeeze(-1), torch.full((B, 3), 0.98), atol=1e-5)
    assert torch.allclose(es, tb["q_k"].float()) and float(alive.max()) == 0
    s, steps, acts, states = committed(pi, tb["hop"], tb["prev"], tb["q_start"], 40, R)
    assert oracle_route_ok(acts, steps, tb["q_word"], tb["q_k"]).all() and (s == tb["q_ans"]).all()
    d = remaining_distance(states, tb["hop"], tb["prev"], tb["q_word"], tb["q_k"], DC)
    for b in range(B):
        for j in range(3):
            k = int(tb["q_k"][b, j])
            assert d[b, j, :k + 1].tolist() == list(range(k, -1, -1))
    # the numpy reference executor agrees
    for b in range(B):
        assert follow(ep, b, 0, DC)[-1] == ep.q_ans[b, 0]


def test_marginal_equals_brute_force_enumeration():
    rng = np.random.default_rng(2)
    ep = generate(DC, 3, rng, Qn=2)
    tb = tables(ep, DC, "cpu")
    torch.manual_seed(0)
    pi = torch.softmax(torch.randn(3, DC.n_sym, 3) * 1.5, -1)
    R = readout(DC, 0.02, "cpu")
    T = 5
    q, es, _, _ = marginal(pi, tb["hop"], tb["prev"], tb["q_start"], T, R)
    for b in range(3):
        for j in range(2):
            tot = torch.zeros(DC.n_ans); est = 0.0
            for L in range(T + 1):
                for route in itertools.product([HOP, PREV], repeat=L):
                    s = int(tb["q_start"][b, j]); p = 1.0
                    for a in route:
                        p *= float(pi[b, s, a]); s = int(tb["hop"][b, s] if a == HOP else tb["prev"][b, s])
                    ph = float(pi[b, s, HALT]) if L < T else 1.0
                    tot += p * ph * R[s]; est += p * ph * L
            assert torch.allclose(tot, q[b, j], atol=1e-6) and abs(est - float(es[b, j])) < 1e-5
    # the step-indexed form agrees with its own enumeration
    pit = torch.softmax(torch.randn(T, 3, 1, 3), -1)
    q2, _, _, _ = marginal(pit, tb["hop"], tb["prev"], tb["q_start"], T, R)
    b, j = 1, 0
    tot = torch.zeros(DC.n_ans)
    for L in range(T + 1):
        for route in itertools.product([HOP, PREV], repeat=L):
            s = int(tb["q_start"][b, j]); p = 1.0
            for t, a in enumerate(route):
                p *= float(pit[t, b, 0, a]); s = int(tb["hop"][b, s] if a == HOP else tb["prev"][b, s])
            ph = float(pit[L, b, 0, HALT]) if L < T else 1.0
            tot += p * ph * R[s]
    assert torch.allclose(tot, q2[b, j], atol=1e-6)


def test_planted_router_reproduces_the_oracle_route_everywhere():
    rng = np.random.default_rng(3)
    m = plant(StackedRouter(2, RouterConfig(), DC))
    sharp = plant(StackedRouter(1, RouterConfig(), DC), gain=6.0, hop_bias=8.0)
    R = readout(DC, 0.02, "cpu")
    with torch.no_grad():
        for name, (k, kind) in STRATA:
            ep = generate(DC, 8, rng, forced_k=k, forced_kind=kind, Qn=3)
            tb = tables(ep, DC, "cpu")
            pi = torch.softmax(m.logits(_batch(ep, 2)), -1)
            s, steps, acts, states = committed(pi[1], tb["hop"], tb["prev"], tb["q_start"], 40, R)
            assert oracle_route_ok(acts, steps, tb["q_word"], tb["q_k"]).all(), name
            pis = torch.softmax(sharp.logits(_batch(ep, 1)), -1)
            q, es, _, _ = marginal(pis[0], tb["hop"], tb["prev"], tb["q_start"], 40, R)
            pc = q.gather(-1, (tb["q_ans"] - DC.n_keys).unsqueeze(-1)).squeeze(-1)
            assert float(pc.min()) > 0.95 and float((es - tb["q_k"]).abs().max()) < 0.25, name
    # the mild planted head is a few units from the init (L24)
    assert float(m.head.weight[0].norm(dim=0).max()) <= 4.5 and float(m.head.bias.abs().max()) <= 4.0


def test_planted_failures_fail_where_stated():
    rng = np.random.default_rng(4)
    R = readout(DC, 0.02, "cpu")
    ep = generate(DC, 16, rng, forced_k=6, forced_kind="pp", Qn=3)
    tb = tables(ep, DC, "cpu")
    B, S = 16, DC.n_sym
    # the best fixed program (all-H, DESIGN.md §5) as a step-indexed policy: fails every word with a P
    pit = torch.zeros(40, B, 1, 3); pit[..., HOP] = 1.0
    s, steps, acts, _ = committed(pit, tb["hop"], tb["prev"], tb["q_start"], 40, R)
    assert not (s[:, 0] == tb["q_ans"][:, 0]).any()                      # the forced query has a PP run
    # a router that never halts: forced halt at T; wrong on every query under the rule-following policy with HALT off
    sym = torch.arange(S).unsqueeze(0).expand(B, S)
    is_key = sym < DC.n_keys; arrow = is_key & (tb["hop"] == DC.DEAD)
    pi = torch.stack([(~arrow).float(), arrow.float(), torch.zeros(B, S)], -1)
    q, es, _, alive = marginal(pi, tb["hop"], tb["prev"], tb["q_start"], 40, R)
    assert torch.allclose(es, torch.full_like(es, 40.0)) and float(alive.min()) > 0.99   # all mass halts by force at T
    # (answers absorb, so the answer is still right at T: the failure is cost, which the step term measures)
    # a myopic router that halts whenever the readout would not change (i.e. on P-states): fails every word with a P
    pi_my = torch.stack([(~arrow & is_key).float(), torch.zeros(B, S), (arrow | ~is_key).float()], -1)
    s2, _, _, _ = committed(pi_my, tb["hop"], tb["prev"], tb["q_start"], 40, R)
    assert not (s2[:, 0] == tb["q_ans"][:, 0]).any()


def test_pool_and_strata_shapes():
    pool = EpisodePool(DC, 50, seed=5)
    b = stack_batches([pool.sample(4), EpisodePool(DC, 30, seed=6).sample(4)])
    assert b["pair_key"].shape == (2, 4, DC.P) and b["q_word"].shape[:3] == (2, 4, DC.Qn)
    st = strata_episodes(DC, 3, seed=7)
    assert set(st) == {n for n, _ in STRATA} and st["alt32"].q_k[:, 0].tolist() == [32, 32, 32]


def test_routed_runs_are_independent_and_teacher_loss_is_zero_on_the_planted_router():
    from topology_design.route.train_route import teacher_loss, policy, flat
    rng = np.random.default_rng(8)
    ep = generate(DC, 4, rng, Qn=3)
    b = _batch(ep, 2)
    m = StackedRouter(2, RouterConfig(), DC)
    fb = flat(b); mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], DC)
    pi = policy(m, b, 12)
    q, es, _, _ = marginal(pi, mp["hop"], mp["prev"], fb["q_start"], 12, readout(DC, 0.02, "cpu"))
    loss0 = -torch.log(q[:4].gather(-1, (fb["q_ans"][:4] - DC.n_keys).unsqueeze(-1)) + 1e-9).mean()    # run 0 only
    loss0.backward()
    assert m.head.weight.grad[0].abs().max() > 0 and m.head.weight.grad[1].abs().max() == 0
    pm = plant(StackedRouter(1, RouterConfig(), DC), gain=6.0, hop_bias=8.0)
    with torch.no_grad():
        pi1 = policy(pm, _batch(ep, 1), 12)
        fb1 = flat(_batch(ep, 1)); mp1 = maps(fb1["pair_key"], fb1["pair_val"], fb1["pair_prev"], DC)
        tl = teacher_loss(pi1, fb1["q_start"], fb1["q_word"], fb1["q_k"], mp1["hop"], mp1["prev"], 12)
        assert float(tl.max()) < 0.1
        uni = torch.full_like(pi1, 1 / 3)
        tlu = teacher_loss(uni, fb1["q_start"], fb1["q_word"], fb1["q_k"], mp1["hop"], mp1["prev"], 12)
        assert torch.allclose(tlu, (fb1["q_k"].float() + 1) * np.log(3), atol=1e-4)


def test_neural_baseline_runs_on_route_tokens():
    from topology_design.ptr.model import ModelConfig
    from topology_design.ptr.stacked import StackedPtrModel
    rng = np.random.default_rng(9)
    ep = generate(DC, 3, rng)
    b = {k: v for k, v in _batch(ep, 2).items() if k != "q_word"}
    for mc in (ModelConfig(kind="static", layers=2, slots=5, d=344), ModelConfig(kind="looped", halting="fixed", T=4, slots=5, d=344)):
        m = StackedPtrModel(2, mc, DC)
        assert m.sym.exact and hasattr(m.sym, "prev_slot")
        out = m(b, mode="train")
        loss, per_run, _ = m.loss(out)
        loss.backward()
        assert torch.isfinite(loss) and out["logits"].shape == (2, 3, DC.Qn, DC.n_ans)
