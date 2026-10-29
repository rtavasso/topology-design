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


# ---------- revision 3: fixed-prior posterior, trajectory balance, actor-critic ----------
from topology_design.route.estimators import posterior_counts, flat_prior, answer_likelihood, pf_loss, tb_loss, ac_loss


@pytest.mark.parametrize("deferral", [1, 0])
def test_fixed_prior_posterior_matches_brute_force(deferral):
    cfg = RouteConfig(deferral=deferral)
    ep = generate(cfg, 2, np.random.default_rng(1), Qn=1, forced_k=3, forced_kind="alt")
    tb = tables(ep, cfg, "cpu")
    T, lam, eps = 5, 0.02, 0.02
    pi0 = flat_prior(tb, cfg)
    counts, Z = posterior_counts(pi0, tb, tb["q_start"], tb["q_ans"], T, cfg, lam, eps)
    tabs = {HOP: tb["hop"], PREV: tb["prev"], SET: tb["set"]}
    for b in range(2):
        Zb = 0.0; cb = torch.zeros(cfg.n_sym * PER_SYM, 4, dtype=torch.float64)
        for L in range(T + 1):
            for mv in itertools.product([HOP, PREV, SET], repeat=L):
                s = int(sid(tb["q_start"][b, 0], 0, 0)); w = 1.0; visits = []
                for a in mv:
                    w *= float(pi0[b, s, a]) * math.exp(-lam); visits.append((s, a)); s = int(tabs[a][b, s])
                if w == 0:
                    continue
                if L < T:
                    w *= float(pi0[b, s, HALT]); visits.append((s, HALT))
                w *= float(answer_likelihood(torch.tensor(s // PER_SYM), tb["q_ans"][b, 0], cfg, eps))
                Zb += w
                for x, a in visits:
                    cb[x, a] += w
        assert float(Z[b, 0]) == pytest.approx(Zb, rel=1e-9)
        assert torch.allclose(counts[b, 0], cb / Zb, atol=1e-12)


def test_pf_gradient_is_the_fixed_prior_posterior_expectation_of_grad_log_pi():
    torch.manual_seed(2)
    tb = _small(2, B=1, k=2)
    T, lam = 3, 0.02
    theta = torch.nn.Parameter(torch.randn(1, NS, 4))
    pi = torch.softmax(theta, -1)
    counts, Z = posterior_counts(flat_prior(tb, DC), tb, tb["q_start"], tb["q_ans"], T, DC, lam, 0.02)
    g_pf, = torch.autograd.grad(pf_loss(pi, counts).sum(), theta)
    acts, states, steps = _enumerate_routes(tb, T)
    lp0 = route_logprob(flat_prior(tb, DC), acts, states)[0].double()
    Ly = answer_likelihood(states[0, :, -1] // PER_SYM, tb["q_ans"][0, 0].expand(acts.shape[1]), DC, 0.02)
    w = lp0.exp() * torch.exp(-lam * steps[0].double()) * Ly
    post = (w / w.sum()).float()
    lpr = route_logprob(torch.softmax(theta, -1), acts, states)[0]
    g_ref, = torch.autograd.grad(-(post * lpr).sum(), theta)
    assert torch.allclose(g_pf, g_ref, atol=1e-5)


def test_trajectory_balance_is_zero_iff_log_ratios_agree():
    lp = torch.tensor([[[-2.0, -3.0, -1.0]]]); steps = torch.tensor([[[2, 3, 1]]]); lLy = torch.zeros(1, 1, 3)
    lp0 = torch.tensor([[[-1.0, -2.0, 0.0]]])
    loss, _ = tb_loss(lLy, steps, lp, lp0 - 0.02 * steps.float() * 0, 0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)                                 # log pi - log R = -1 for all three
    loss2, _ = tb_loss(lLy, steps, lp + torch.tensor([[[0.0, 0.5, 0.0]]]), lp0, 0.0)
    assert float(loss2) > 0


def test_actor_critic_targets_and_advantages_on_a_hand_built_route():
    """One route of 2 moves then HALT, reward 1 at the end, lam = 0.1: TD(0) targets are -0.1 + V(next) for moves and
    1 at the HALT; with gae = 1 the advantage is the Monte-Carlo return minus V."""
    NSs = 5
    pi = torch.full((1, NSs, 4), 0.25, requires_grad=True)
    V = torch.tensor([[0.3, 0.5, 0.7, 0.0, 0.0]], requires_grad=True)
    acts = torch.tensor([[[[HOP, HOP, HALT, -1]]]]); states = torch.tensor([[[[0, 1, 2, 2, 2]]]])
    reward = torch.tensor([[[1.0]]])
    loss0, _ = ac_loss(pi, V, acts, states, reward, 0.1, gae=0.0)
    gV, = torch.autograd.grad(loss0.sum(), V, retain_graph=True)
    # critic gradient (V - target): t0: 0.3 - (-0.1 + 0.5) = -0.1; t1: 0.5 - (-0.1 + 0.7) = -0.1; t2: 0.7 - 1 = -0.3
    assert torch.allclose(gV[0, :3], torch.tensor([-0.1, -0.1, -0.3]), atol=1e-6)
    loss1, ret = ac_loss(pi, V, acts, states, reward, 0.1, gae=1.0)
    gpi, = torch.autograd.grad(loss1.sum(), pi)
    # MC return from t0 = -0.1 - 0.1 + 1 = 0.8: advantage 0.8 - 0.3 = 0.5 on HOP at state 0 (d(-A log pi)/d pi = -A / pi)
    assert float(gpi[0, 0, HOP]) == pytest.approx(-0.5 / 0.25, rel=1e-5)
    assert float(ret) == pytest.approx(-0.8, rel=1e-6)


# ---------- revision 4 (design review 3, code review 3) ----------
from topology_design.route.estimators import canonical_routes, pfis_loss


@pytest.mark.parametrize("est", ["pf", "exal", "tb", "pfis", "ac"])
def test_trainer_runs_the_revision_4_signals(est):
    from topology_design.route.train_route import run_stack
    base = dict(steps=2, batch=4, pool_n=40, eval_n=2, eval_every=2, crit_n=6, crit_every=1, warmup=1, T_eval=20,
                K=4, W=4, estimator=est, per_run_init=True, value_head=(est == "ac"), ans_pool=4)
    recs = run_stack([{**base, "seed": 0}, {**base, "seed": 1}], dev="cpu", log=lambda s: None)
    c = recs[0]["curve"][-1]
    assert math.isfinite(c["loss"]) and "set_other" in c
    assert "samp_min_acc" in recs[0]["evals"][-1]["eval"]["in4"]
    if est in ("ac", "pfis", "tb"):
        assert set(c["diag"]) == {"ac": {"td_abs", "ev", "distinct_P"}, "pfis": {"ess", "any_ans", "distinct_P"}, "tb": {"distinct_P"}}[est]


def test_value_head_is_zero_at_init_detached_and_leaves_the_policy_init_unchanged():
    from topology_design.route.router import StackedRouter, RouterConfig
    from topology_design.route.train_route import per_run_init_
    dc = RouteConfig()
    ms = []
    for vh in (False, True):
        rc = RouterConfig(value_head=vh)
        torch.manual_seed(0); m = StackedRouter(1, rc, dc); per_run_init_(m, [3], rc, dc); ms.append(m)
    pa = dict(ms[0].named_parameters()); pb = dict(ms[1].named_parameters())
    assert all(torch.equal(pa[n], pb[n]) for n in pa)                        # same policy parameters at a seed
    b = {k: torch.as_tensor(v).unsqueeze(0) for k, v in
         dict(pair_key=generate(dc, 2, np.random.default_rng(0)).pair_key, pair_val=generate(dc, 2, np.random.default_rng(0)).pair_val,
              pair_prev=generate(dc, 2, np.random.default_rng(0)).pair_prev, pair_flag=generate(dc, 2, np.random.default_rng(0)).pair_flag).items()}
    lg, V = ms[1].logits_and_values(b)
    assert torch.equal(V, torch.zeros_like(V))
    with torch.no_grad():
        ms[1].v2.weight.normal_()
    _, V = ms[1].logits_and_values(b)
    V.sum().backward()
    assert ms[1].head.weight.grad is None and ms[1].q.weight.grad is None and ms[1].v1.weight.grad is not None


def test_canonical_routes_cut_at_the_first_arrival_and_tb_ignores_no_op_moves():
    cfg = RouteConfig()
    ans = sid(torch.tensor(cfg.n_keys), torch.tensor(0), torch.tensor(0)).item()
    k0, k1 = sid(torch.tensor(0), torch.tensor(0), torch.tensor(0)).item(), sid(torch.tensor(1), torch.tensor(0), torch.tensor(0)).item()
    # clean: HOP, HOP(->answer), HALT ; looped: HOP, HOP(->answer), HOP (no-op at the answer), SET (no-op), HALT
    acts = torch.tensor([[[[HOP, HOP, HALT, -1, -1, -1], [HOP, HOP, HOP, SET, HALT, -1]]]])
    states = torch.tensor([[[[k0, k1, ans, ans, ans, ans, ans], [k0, k1, ans, ans, ans, ans, ans]]]])
    a_c, mv, arr = canonical_routes(acts, states, cfg)
    assert torch.equal(a_c[0, 0, 0], a_c[0, 0, 1]) and mv.tolist() == [[[2, 2]]]
    assert a_c[0, 0, 0].tolist() == [HOP, HOP, -1, -1, -1, -1] and arr.tolist() == [[[ans, ans]]]   # no HALT term on arrival
    # a route that halts on a key keeps its HALT and has no arrival
    a2, mv2, arr2 = canonical_routes(torch.tensor([[[[HOP, HALT, -1]]]]), torch.tensor([[[[k0, k1, k1, k1]]]]), cfg)
    assert a2.tolist() == [[[[HOP, HALT, -1]]]] and arr2.tolist() == [[[-1]]] and mv2.tolist() == [[[1]]]
    NS = cfg.n_sym * PER_SYM
    pi = torch.full((1, NS, 4), 0.25)
    pi0 = torch.full((1, NS, 4), 0.25); pi0[0, cfg.n_keys * PER_SYM:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    lpr, lp0 = route_logprob(pi, a_c, states), route_logprob(pi0, a_c, states)
    assert torch.isfinite(lp0).all() and float(lp0[0, 0, 0]) == pytest.approx(2 * math.log(0.25))
    from topology_design.route.estimators import arrival_halt_loss
    pi_h = pi.clone(); pi_h[0, ans] = torch.tensor([0.1, 0.1, 0.1, 0.7])
    assert float(arrival_halt_loss(pi_h, arr)) == pytest.approx(-math.log(0.7), rel=1e-5)
    loss, _ = tb_loss(torch.zeros(1, 1, 2), mv, lpr, lp0, 0.02)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)                      # the two samples are the same canonical route


def test_pfis_weights_are_the_self_normalized_fixed_prior_posterior_ratios():
    lpy = torch.log(torch.tensor([[[0.98, 0.98, 0.0, 0.001]]]))
    steps = torch.tensor([[[2, 3, 1, 2]]])
    lpi = torch.log(torch.tensor([[[0.1, 0.2, 0.3, 0.05]]])).requires_grad_(True)
    lp0 = torch.log(torch.tensor([[[0.05, 0.01, 0.2, 0.05]]]))
    loss, ess = pfis_loss(lpy, steps, lpi, lp0, 0.02)
    g, = torch.autograd.grad(loss.sum(), lpi)
    w = torch.exp(lp0 - 0.02 * steps.float() + lpy - lpi.detach()); w = w / w.sum()
    assert torch.allclose(-g, w, atol=1e-6) and float(-g[0, 0, 2]) == 0.0
    assert float(ess) == pytest.approx(float(1 / (w ** 2).sum() / 4), rel=1e-5)
    l2, e2 = pfis_loss(torch.full((1, 1, 2), -math.inf), torch.ones(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), 0.02)
    assert float(l2) == 0.0 and float(e2) == 0.0


def test_program_stats_classifies_early_late_and_other_sets_on_shortest_correct_routes():
    from topology_design.route.train_route import program_stats
    cfg = RouteConfig()
    S = cfg.n_sym
    s_ = lambda sym, c=0, h=0: sid(torch.tensor(sym), torch.tensor(c), torch.tensor(h)).item()
    mp = {"flag": torch.zeros(1, S, dtype=torch.long)}
    A = cfg.n_keys
    # oracle: 0 -HOP-> 1 -PREV-> 5 -HOP-> answer (symbol 1 is P-leaving); the oracle takes 4 moves incl. one SET
    oa = torch.tensor([[[SET, HOP, PREV, HOP, HALT, -1]]])
    ost = torch.tensor([[[s_(0), s_(0, 1, 1), s_(1, 1), s_(5), s_(A), s_(A), s_(A)]]])
    E_ = torch.tensor([[[SET, HOP, PREV, HOP, HALT, -1]]]); E_st = ost.clone()
    L_ = torch.tensor([[[HOP, SET, PREV, HOP, HALT, -1]]]); L_st = torch.tensor([[[s_(0), s_(1), s_(1, 1, 1), s_(5), s_(A), s_(A), s_(A)]]])
    O_ = torch.tensor([[[HOP, PREV, SET, HOP, HALT, -1]]]); O_st = torch.tensor([[[s_(0), s_(1), s_(5), s_(5, 1, 1), s_(A), s_(A), s_(A)]]])
    q_ans, osteps, fin = torch.tensor([[A]]), torch.tensor([[4]]), torch.tensor([[A]])
    out = [program_stats(a, st, osteps, fin, q_ans, osteps, mp, oa, ost) for a, st in ((E_, E_st), (L_, L_st), (O_, O_st))]
    assert [float(o[1]) for o in out] == [0.0, 1.0, 0.0] and [float(o[2]) for o in out] == [0.0, 0.0, 1.0]
    ok, late, oth = program_stats(L_, L_st, torch.tensor([[5]]), fin, q_ans, osteps, mp, oa, ost)
    assert not bool(ok) and math.isnan(float(late))                          # not shortest: not classified


def test_answer_pool_limits_distinct_answers_and_leaves_the_default_stream_unchanged():
    from topology_design.route.train_route import answer_only_readout
    a = generate(RouteConfig(), 5, np.random.default_rng(3)); b = generate(RouteConfig(ans_pool=0), 5, np.random.default_rng(3))
    assert np.array_equal(a.pair_val, b.pair_val) and np.array_equal(a.q_ans, b.q_ans)
    cfg = RouteConfig(ans_pool=4)
    ep = generate(cfg, 20, np.random.default_rng(3))
    for i in range(20):
        vals = ep.pair_val[i]
        assert len(set(v for v in vals.tolist() if cfg.n_keys <= v < cfg.DEAD)) <= 4
    Rd = answer_only_readout(readout(cfg, 0.02, "cpu"), cfg)
    sym = torch.arange(Rd.shape[0]) // PER_SYM
    assert float(Rd[sym < cfg.n_keys].abs().sum()) == 0.0 and float(Rd[sym == cfg.DEAD].abs().sum()) == 0.0
    assert torch.allclose(Rd[sym == cfg.n_keys + 2][0], readout(cfg, 0.02, "cpu")[(cfg.n_keys + 2) * PER_SYM])


def test_critic_warmup_switches_the_actor_off():
    NSs = 5
    pi = torch.full((1, NSs, 4), 0.25, requires_grad=True)
    V = torch.tensor([[0.3, 0.5, 0.7, 0.0, 0.0]], requires_grad=True)
    acts = torch.tensor([[[[HOP, HOP, HALT, -1]]]]); states = torch.tensor([[[[0, 1, 2, 2, 2]]]])
    loss, _ = ac_loss(pi, V, acts, states, torch.tensor([[[1.0]]]), 0.1, gae=0.0, actor_scale=0.0)
    gpi, gV = torch.autograd.grad(loss.sum(), (pi, V))
    assert float(gpi.abs().sum()) == 0.0 and float(gV.abs().sum()) > 0.0


def test_distinct_routes_counts_unique_sampled_routes_on_P_words():
    from topology_design.route.train_route import distinct_routes_P
    a = torch.tensor([[[[HOP, HALT, -1], [HOP, HALT, -1], [SET, HOP, HALT], [HOP, HOP, HALT]],
                       [[HOP, HALT, -1], [HOP, HALT, -1], [HOP, HALT, -1], [HOP, HALT, -1]]]])     # (1, 2, 4, 3)
    w = torch.tensor([[[0, 1, 0], [0, 0, 0]]])                                                     # only query 0 has a P
    assert distinct_routes_P(a, w, 1).tolist() == [3.0]
    w2 = torch.tensor([[[0, 1, 0], [0, 1, 0]]])
    assert distinct_routes_P(a, w2, 1).tolist() == [2.0]
