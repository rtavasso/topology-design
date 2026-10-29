"""Tests for rung 2 (experiments/routefind/DESIGN.md §8, G1): the route-finding estimators against exact gradients,
the beam against the oracle and brute force, replay, and the trainer for every estimator."""

import itertools
import math

import numpy as np
import pytest
import torch

from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, marginal, readout, oracle, oracle_policy, route_ok, sid, PER_SYM, HOP, PREV, SET, HALT
from topology_design.route.estimators import (sample_routes, route_logprob, readout_logp, iw_loss, rf_loss, rl_loss,
                                              beam_search, replay, best_correct_sample)

DC = RouteConfig()
NS = DC.n_sym * PER_SYM


def _small(seed=0, B=2, k=2):
    ep = generate(DC, B, np.random.default_rng(seed), Qn=1, forced_k=k, forced_kind="train")
    return tables(ep, DC, "cpu")


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def _est_grad(theta, tb, T, K, loss_fn, reps, lam=0.02, eps=0.02):
    g_all = []
    for _ in range(reps):
        pi = torch.softmax(theta, -1)
        sym, steps, acts, states = sample_routes(pi, tb, tb["q_start"], T, K)
        lpr = route_logprob(pi, acts, states)
        lp = readout_logp(sym, tb["q_ans"].unsqueeze(-1), DC, eps)
        sur, _ = loss_fn(lp, steps, lpr, lam)
        g, = torch.autograd.grad(sur.sum(), theta)
        g_all.append(g)
    return torch.stack(g_all).mean(0)


def test_rloo_matches_the_exact_gradient_of_the_expected_log_likelihood():
    torch.manual_seed(0)
    tb = _small(0)
    T, lam = 5, 0.02
    theta = torch.nn.Parameter(torch.randn(2, NS, 4) * 0.7)
    pi = torch.softmax(theta, -1)
    logR = torch.log(readout(DC, 0.02, "cpu"))
    q, es, _, _ = marginal(pi, tb, tb["q_start"], T, logR)                      # E[log p(y | r)] by the recursion
    val = -q.gather(-1, (tb["q_ans"] - DC.n_keys).unsqueeze(-1)).squeeze(-1) + lam * es
    g_exact, = torch.autograd.grad(val.sum(), theta)
    g_est = _est_grad(theta, tb, T, 64, rf_loss, 20)
    assert _cos(g_est, g_exact) > 0.97 and 0.8 < float(g_est.norm() / g_exact.norm()) < 1.25


def test_vimco_at_large_K_matches_the_exact_marginal_gradient():
    torch.manual_seed(1)
    tb = _small(1)
    T, lam = 5, 0.02
    theta = torch.nn.Parameter(torch.randn(2, NS, 4) * 0.7)
    pi = torch.softmax(theta, -1)
    q, es, _, _ = marginal(pi, tb, tb["q_start"], T, readout(DC, 0.02, "cpu"))
    val = -torch.log(q.gather(-1, (tb["q_ans"] - DC.n_keys).unsqueeze(-1)).squeeze(-1)) + lam * es
    g_exact, = torch.autograd.grad(val.sum(), theta)
    g_est = _est_grad(theta, tb, T, 256, iw_loss, 20)
    assert _cos(g_est, g_exact) > 0.97 and 0.85 < float(g_est.norm() / g_exact.norm()) < 1.15


def test_route_logprob_and_readout_on_the_oracle_route():
    ep = generate(DC, 8, np.random.default_rng(2), Qn=3)
    tb = tables(ep, DC, "cpu")
    oa, osteps, ost = oracle(tb, tb["q_start"], 14, DC)
    pio = oracle_policy(tb, DC) * 0.96 + 0.01                                          # sharp but not one-hot
    lp = route_logprob(pio, oa, ost)
    n_act = (oa >= 0).sum(-1).float()
    assert torch.allclose(lp, n_act * math.log(0.97), atol=1e-4)
    sym = ost[:, :, -1] // PER_SYM
    assert torch.allclose(readout_logp(sym, tb["q_ans"], DC, 0.02), torch.full_like(lp, math.log(0.98)))
    assert float(readout_logp(torch.full_like(sym, DC.DEAD), tb["q_ans"], DC, 0.02)[0, 0]) == pytest.approx(math.log(1 / 16))


@pytest.mark.parametrize("deferral", [1, 0])
def test_beam_returns_the_oracle_route_under_the_oracle_policy_and_replay_agrees(deferral):
    cfg = RouteConfig(deferral=deferral)
    ep = generate(cfg, 16, np.random.default_rng(3), Qn=3)
    tb = tables(ep, cfg, "cpu")
    oa, osteps, ost = oracle(tb, tb["q_start"], 14, cfg)
    pio = oracle_policy(tb, cfg) * 0.96 + 0.01
    found, acts, steps = beam_search(pio, tb, tb["q_start"], tb["q_ans"], 14, 4, cfg, eps_mix=0.0)
    assert found.all() and route_ok(acts, steps, oa, osteps).all()
    assert (replay(tb, tb["q_start"], oa) == ost).all()


def test_beam_finds_the_map_correct_route_of_brute_force():
    """With W at least the number of states reachable within T, the beam's kept route maximizes log pi - lam * moves
    among the routes of length <= T that end on the label in the PRUNED space (no no-op moves, no moves out of answers
    or dead states, halting at the first arrival); merging identical states loses nothing: pi is Markov."""
    torch.manual_seed(4)
    tb = _small(4, B=3, k=2)
    T, lam = 4, 0.02
    pi = torch.softmax(torch.randn(3, NS, 4) * 1.5, -1)
    found, acts, steps = beam_search(pi, tb, tb["q_start"], tb["q_ans"], T, 81, DC, eps_mix=0.0, lam=lam)
    logpi = torch.log(pi)
    states_all = replay(tb, tb["q_start"], acts)
    tabs = {HOP: tb["hop"], PREV: tb["prev"], SET: tb["set"]}
    is_ans = lambda x: DC.n_keys <= x // PER_SYM < DC.DEAD
    for b in range(3):
        best = -float("inf")
        for L in range(T + 1):
            for route in itertools.product([HOP, PREV, SET], repeat=L):
                s = int(sid(tb["q_start"][b, 0], 0, 0)); lp = 0.0; ok = True
                for a in route:
                    if is_ans(s) or s // PER_SYM == DC.DEAD:
                        ok = False; break
                    nxt = int(tabs[a][b, s])
                    if nxt == s or nxt // PER_SYM == DC.DEAD:
                        ok = False; break
                    lp += float(logpi[b, s, a]); s = nxt
                if not ok or s // PER_SYM != int(tb["q_ans"][b, 0]):
                    continue
                score = lp + (float(logpi[b, s, HALT]) if L < T else 0.0) - lam * L
                best = max(best, score)
        if best > -float("inf"):
            assert bool(found[b, 0])
            a = acts[b, 0]; n = int(steps[b, 0])
            st = states_all[b, 0]
            got = sum(float(logpi[b, int(st[t]), int(a[t])]) for t in range(len(a)) if a[t] >= 0) - lam * n
            assert got == pytest.approx(best, abs=1e-4)


@pytest.mark.parametrize("est", ["iw", "ell", "rl", "si", "ei"])
def test_trainer_runs_every_estimator_and_records_search_stats(est):
    from topology_design.route.train_route import run_stack
    base = dict(steps=2, batch=4, pool_n=40, eval_n=2, eval_every=2, crit_n=6, crit_every=1, warmup=1, T_eval=20,
                K=4, W=4, estimator=est, per_run_init=True)
    recs = run_stack([{**base, "seed": 0}, {**base, "seed": 1}], dev="cpu", log=lambda s: None)
    c = recs[0]["curve"][-1]
    assert {"search", "crit_obj", "clip_frac", "estimator_now"} <= set(c) and "search_init" in recs[0]
    st = c["search"]
    assert 0.0 <= st["oracle_kept"] <= st["hit"] <= 1.0 and "hit_P_by_k" in st and recs[0]["train"]["rollouts_per_query"] == 4
    assert recs[0]["train"]["stack_seeds"] == [0, 1]


def test_per_run_init_and_sampling_do_not_depend_on_the_stack_and_warm_start_switches():
    from topology_design.route.train_route import run_stack
    base = dict(steps=2, batch=4, pool_n=40, eval_n=2, eval_every=2, crit_n=6, crit_every=1, warmup=1, T_eval=20,
                K=4, estimator="iw", per_run_init=True)
    a = run_stack([{**base, "seed": 0}], dev="cpu", log=lambda s: None)[0]
    b = run_stack([{**base, "seed": 0}, {**base, "seed": 3}], dev="cpu", log=lambda s: None)[0]
    assert a["search_init"] == b["search_init"]
    assert a["curve"][-1]["loss"] == pytest.approx(b["curve"][-1]["loss"], rel=1e-4)
    w = run_stack([{**base, "seed": 0, "warm_steps": 1}], dev="cpu", log=lambda s: None)[0]
    assert [c["estimator_now"] for c in w["curve"]] == ["exact", "iw"]


def _enumerate_routes(tb, T):
    """Every route of length <= T from query 0 of episode 0: moves from {HOP, PREV, SET}, then HALT (none at T)."""
    rows = []
    for L in range(T + 1):
        for mv in itertools.product([HOP, PREV, SET], repeat=L):
            a = list(mv) + ([HALT] if L < T else [])
            rows.append(a + [-1] * (T - len(a)))
    acts = torch.tensor(rows).unsqueeze(0)                                               # (1, N, T)
    qs = tb["q_start"][:1, :1].expand(1, acts.shape[1])
    states = replay({k: v[:1] for k, v in tb.items() if k in ("hop", "prev", "set")}, qs, acts)
    steps = torch.tensor([sum(1 for x in r if x in (HOP, PREV, SET)) for r in rows]).unsqueeze(0)
    return acts, states, steps


@pytest.mark.parametrize("name", ["iw", "ell", "rl"])
def test_estimators_are_exactly_unbiased_at_K2_by_enumeration(name):
    """E over all K = 2 route pairs of the surrogate's gradient equals the exact gradient of the stated objective:
    iw: -E[log mean(p1, p2)] + lam E[steps]; ell: -E[log p] + lam E[steps]; rl: -E[p] + lam E[steps] (code review 1, 5)."""
    torch.manual_seed(5)
    tb = _small(5, B=1, k=2)
    T, lam = 3, 0.05
    theta = torch.nn.Parameter(torch.randn(1, NS, 4))
    pi = torch.softmax(theta, -1)
    acts, states, steps = _enumerate_routes(tb, T)
    lpr = route_logprob(pi, acts, states)[0]                                             # (N,)
    assert float(lpr.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    lpy = readout_logp(states[0, :, -1] // PER_SYM, tb["q_ans"][0, 0].expand(acts.shape[1]), DC, 0.02)
    N = acts.shape[1]
    i1, i2 = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    i1, i2 = i1.reshape(-1), i2.reshape(-1)
    pair = lambda x: torch.stack([x[i1], x[i2]], -1).unsqueeze(0)                         # (1, N*N, 2)
    fn = {"iw": iw_loss, "ell": rf_loss, "rl": rl_loss}[name]
    sur, _ = fn(pair(lpy), pair(steps[0].float()), pair(lpr), lam)
    w = (lpr[i1] + lpr[i2]).exp().detach()
    g_est, = torch.autograd.grad((w * sur[0]).sum(), theta, retain_graph=True)
    pr = lpr.exp(); es = (pr * steps[0]).sum()
    if name == "iw":
        val = -((pr[i1] * pr[i2]) * (torch.logaddexp(lpy[i1], lpy[i2]) - math.log(2))).sum() + lam * es
    elif name == "ell":
        val = -(pr * lpy).sum() + lam * es
    else:
        val = -(pr * lpy.exp()).sum() + lam * es
    g_exact, = torch.autograd.grad(val, theta)
    assert torch.allclose(g_est, g_exact, atol=1e-5 * float(g_exact.abs().max()) + 1e-7)


def test_best_correct_sample_picks_the_most_probable_correct_route():
    sym = torch.tensor([[[50, 52, 50]]]); q_ans = torch.tensor([[50]])
    lp = torch.tensor([[[-3.0, -0.1, -2.0]]]); steps = torch.tensor([[[2, 1, 2]]])
    ok, idx = best_correct_sample(sym, q_ans, lp, steps, 0.02)
    assert bool(ok[0, 0]) and int(idx[0, 0]) == 2
    ok2, _ = best_correct_sample(sym, torch.tensor([[60]]), lp, steps, 0.02)
    assert not bool(ok2[0, 0])
