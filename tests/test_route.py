"""Tests for rung 1 (experiments/route/DESIGN.md §11, gates G1–G2, revision 2): the generator (deferred flags, decoys),
the exact executor over the product state and the route marginal, the planted router and the planted failures (the
all-H program, the symbol-only router), and the trainers' run independence."""

import itertools

import numpy as np
import pytest
import torch

from topology_design.route.data import RouteConfig, generate, check_episode, word_class, oracle_actions, H, P
from topology_design.route.exec import (tables, maps, marginal, committed, oracle, oracle_policy, route_ok, readout,
                                        remaining_distance, sid, PER_SYM, HOP, PREV, SET, HALT)
from topology_design.route.router import StackedRouter, RouterConfig, plant
from topology_design.route.pool import to_dict, EpisodePool, stack_batches, strata_episodes, STRATA, KEYS

DC = RouteConfig()            # deferral 1 (the thesis setting)
DC0 = RouteConfig(deferral=0)


def _batch(ep, R=1):
    d = to_dict(ep, "cpu")
    return {k: v.unsqueeze(0).expand(R, *v.shape).clone() for k, v in d.items()}


@pytest.mark.parametrize("cfg", [DC, DC0], ids=["deferral1", "deferral0"])
def test_generator_chains_words_flags_and_decoys(cfg):
    rng = np.random.default_rng(0)
    ep = generate(cfg, 32, rng)
    check_episode(ep, cfg)
    assert (ep.pair_prev[:, 1:] == ep.pair_key[:, :-1]).all() and (ep.pair_prev[:, 0] == cfg.DEAD).all()
    assert ep.pair_flag.min() == 0 and ep.pair_flag.max() == 1
    for b in range(32):
        pos = {int(k): i for i, k in enumerate(ep.pair_key[b])}
        flagged = set()
        for j in range(cfg.Qn):
            k = int(ep.q_k[b, j]); w = ep.q_word[b, j, :k]
            assert w[0] == H and w[-1] == H and word_class(w)["max_prun"] <= 1
            states, acts = oracle_actions(ep, b, j, cfg)
            syms = [s for s, c, h in states]
            chain = []                                          # s_0 .. s_k from the oracle route's moves
            for (s, c, h), a in zip(states, acts):
                if a in (HOP, PREV):
                    chain.append(s)
            chain.append(syms[-1])
            assert len(chain) == k + 1 and chain[-1] == ep.q_ans[b, j]
            for i in range(1, k + 1):                           # step i leaves s_{i-1}
                if w[i - 1] == P:
                    assert ep.pair_key[b, pos[chain[i - 1]] - 1] == chain[i]            # pair(s_i) right before pair(s_{i-1})
                    assert ep.pair_val[b, pos[chain[i - 1]]] != chain[i]                 # a decoy, not the next state
                    assert ep.pair_val[b, pos[chain[i - 1]]] < cfg.n_keys               # the decoy is a key
                    flagged.add(chain[max(i - 1 - cfg.deferral, 0)])
                else:
                    assert ep.pair_val[b, pos[chain[i - 1]]] == chain[i]
        # every flag the queried chains need is set; flags on other chains are the unqueried chains' own
        assert all(ep.pair_flag[b, pos[s]] == 1 for s in flagged)
    for kind, k, test in (("pp", 8, lambda c: 2 <= c["max_prun"] <= 3), ("alt", 8, lambda c: c["nP"] == 3),
                          ("deepP", 8, lambda c: c["deepest_P"] >= 6), ("none", 4, lambda c: c["nP"] == 0),
                          ("train", 32, lambda c: c["max_prun"] <= 1 and c["k"] == 32), ("pp", 4, lambda c: c["max_prun"] == 2)):
        e = generate(cfg, 6, rng, forced_k=k, forced_kind=kind, Qn=3)
        check_episode(e, cfg)
        assert all(test(word_class(e.q_word[b, 0, :k])) for b in range(6)), kind


@pytest.mark.parametrize("cfg", [DC, DC0], ids=["deferral1", "deferral0"])
def test_executor_matches_the_numpy_oracle_and_counts_distance(cfg):
    rng = np.random.default_rng(1)
    for kind, k in (("pp", 7), ("train", 8), ("alt", 6)):
        ep = generate(cfg, 6, rng, forced_k=k, forced_kind=kind, Qn=3)
        tb = tables(ep, cfg, "cpu")
        B = 6
        pi = oracle_policy(tb, cfg)
        assert pi.shape == (B, cfg.n_sym * PER_SYM, 4) and torch.allclose(pi.sum(-1), torch.ones(B, pi.shape[1], dtype=pi.dtype))
        R = readout(cfg, 0.02, "cpu")
        T = 48
        q, es, prof, alive = marginal(pi, tb, tb["q_start"], T, R)
        pc = q.gather(-1, (tb["q_ans"] - cfg.n_keys).unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(pc, torch.full((B, 3), 0.98, dtype=q.dtype), atol=1e-5) and float(alive.max()) == 0
        s, steps, acts, states = committed(pi, tb, tb["q_start"], T)
        oa, osteps, ostates = oracle(tb, tb["q_start"], T, cfg)
        assert (acts == oa).all() and (steps == osteps).all() and route_ok(acts, steps, oa, osteps).all()
        assert (s == tb["q_ans"]).all() and torch.allclose(es, osteps.to(es.dtype))
        d = remaining_distance(states, tb, cfg)
        for b in range(B):
            for j in range(3):
                st_np, acts_np = oracle_actions(ep, b, j, cfg)
                n = len(acts_np) - 1                              # moves (HOP / PREV / SET) before HALT
                assert int(osteps[b, j]) == n
                assert acts[b, j, :n + 1].tolist() == acts_np and acts[b, j, n + 1:].eq(-1).all()
                assert [sid(*x) for x in st_np] == states[b, j, :n + 1].tolist()
                assert d[b, j, :n + 1].tolist() == list(range(n, -1, -1))
                nP = int((ep.q_word[b, j, :ep.q_k[b, j]] == P).sum())
                assert n == int(ep.q_k[b, j]) + nP                # one SET per P-step
        if kind == "pp" and cfg.deferral == 1:                   # the pending count reaches 2 only in PP runs under deferral 1
            assert ((states % PER_SYM) // 2 == 2).any()


def test_marginal_equals_brute_force_enumeration():
    rng = np.random.default_rng(2)
    ep = generate(DC, 3, rng, Qn=2)
    tb = tables(ep, DC, "cpu")
    torch.manual_seed(0)
    NS = DC.n_sym * PER_SYM
    pi = torch.softmax(torch.randn(3, NS, 4) * 1.5, -1).float()
    R = readout(DC, 0.02, "cpu")
    T = 5
    q, es, _, _ = marginal(pi, tb, tb["q_start"], T, R)
    tabs = {HOP: tb["hop"], PREV: tb["prev"], SET: tb["set"]}
    for b in range(3):
        for j in range(2):
            tot = torch.zeros(DC.n_ans); est = 0.0
            for L in range(T + 1):
                for route in itertools.product([HOP, PREV, SET], repeat=L):
                    s = int(sid(tb["q_start"][b, j], 0, 0)); p = 1.0
                    for a in route:
                        p *= float(pi[b, s, a]); s = int(tabs[a][b, s])
                    ph = float(pi[b, s, HALT]) if L < T else 1.0
                    tot += p * ph * R[s]; est += p * ph * L
            assert torch.allclose(tot.to(q.dtype), q[b, j], atol=1e-6) and abs(est - float(es[b, j])) < 1e-5
    pit = torch.softmax(torch.randn(T, 3, 1, 4), -1).float()
    q2, _, _, _ = marginal(pit, tb, tb["q_start"], T, R)
    b, j = 1, 0
    tot = torch.zeros(DC.n_ans)
    for L in range(T + 1):
        for route in itertools.product([HOP, PREV, SET], repeat=L):
            s = int(sid(tb["q_start"][b, j], 0, 0)); p = 1.0
            for t, a in enumerate(route):
                p *= float(pit[t, b, 0, a]); s = int(tabs[a][b, s])
            ph = float(pit[L, b, 0, HALT]) if L < T else 1.0
            tot += p * ph * R[s]
    assert torch.allclose(tot.to(q2.dtype), q2[b, j], atol=1e-6)


@pytest.mark.parametrize("cfg", [DC, DC0], ids=["deferral1", "deferral0"])
def test_planted_router_reproduces_the_oracle_route_everywhere(cfg):
    rng = np.random.default_rng(3)
    m = plant(StackedRouter(2, RouterConfig(), cfg))
    sharp = plant(StackedRouter(1, RouterConfig(), cfg), gain=6.0, hop_bias=8.0)
    R = readout(cfg, 0.02, "cpu")
    with torch.no_grad():
        for name, (k, kind) in STRATA:
            ep = generate(cfg, 6, rng, forced_k=k, forced_kind=kind, Qn=3)
            tb = tables(ep, cfg, "cpu")
            pi = torch.softmax(m.logits(_batch(ep, 2)), -1)
            s, steps, acts, states = committed(pi[1], tb, tb["q_start"], 48)
            oa, osteps, _ = oracle(tb, tb["q_start"], 48, cfg)
            assert route_ok(acts, steps, oa, osteps).all(), name
            pis = torch.softmax(sharp.logits(_batch(ep, 1)), -1)
            q, es, _, _ = marginal(pis[0], tb, tb["q_start"], 48, R)
            pc = q.gather(-1, (tb["q_ans"] - cfg.n_keys).unsqueeze(-1)).squeeze(-1)
            assert float(pc.min()) > 0.95 and float((es - osteps).abs().max()) < 0.5, name
    assert float(m.head.weight[0].norm(dim=0).max()) <= 8.5 and float(m.head.bias.abs().max()) <= 4.0


def test_planted_failures_fail_where_stated():
    rng = np.random.default_rng(4)
    ep = generate(DC, 16, rng, forced_k=6, forced_kind="pp", Qn=3)
    tb = tables(ep, DC, "cpu")
    B = 16
    # the best fixed program (all-H, DESIGN.md §5) as a step-indexed policy: never the oracle route on a word with
    # a P; HOP at the P-step follows the decoy into another chain, so the answer is right only by chance (<= 1/16)
    oa, osteps, _ = oracle(tb, tb["q_start"], 48, DC)
    hasP = (tb["q_word"] == P).any(-1)
    pit = torch.zeros(48, B, 1, 4, dtype=torch.float32); pit[..., HOP] = 1.0
    s, steps, acts, _ = committed(pit, tb, tb["q_start"], 48)
    assert not route_ok(acts, steps, oa, osteps)[hasP].any() and float((s[:, 0] == tb["q_ans"][:, 0]).float().mean()) < 0.3
    # a symbol-only router (the F4 null): the same action at every (pending, set-here) of a symbol. The oracle route
    # takes SET and then a move at the same symbol, so no symbol-only committed route is the oracle route on a word
    # with a P; without SET the pending count stays 0, PREV is unavailable and HOP at a P-step follows the decoy.
    for seed in range(3):
        torch.manual_seed(seed)
        m = StackedRouter(1, RouterConfig(state_input="symbol"), DC)
        with torch.no_grad():
            lg = m.logits(_batch(ep, 1))[0]
            assert torch.allclose(lg.view(B, DC.n_sym, PER_SYM, 4)[:, :, :1], lg.view(B, DC.n_sym, PER_SYM, 4))
            pi = torch.softmax(lg * 50, -1)                                                 # sharpened: a committed policy
        s2, steps2, acts2, _ = committed(pi, tb, tb["q_start"], 48)
        assert not route_ok(acts2, steps2, oa, osteps)[hasP].any()
    # a router that never halts: all mass alive at T, E[steps] = T; the answer is still right at T (answers absorb)
    pi_nh = oracle_policy(tb, DC).clone(); pi_nh[..., HALT] = 0; pi_nh[..., HOP] += 1 - pi_nh.sum(-1)
    R = readout(DC, 0.02, "cpu")
    q, es, _, alive = marginal(pi_nh, tb, tb["q_start"], 48, R)
    assert torch.allclose(es, torch.full_like(es, 48.0)) and float(alive.min()) > 0.99
    assert (q.argmax(-1) == tb["q_ans"] - DC.n_keys).all()


def test_pool_and_strata_shapes():
    pool = EpisodePool(DC, 50, seed=5)
    b = stack_batches([pool.sample(4), EpisodePool(DC, 30, seed=6).sample(4)])
    assert set(b) == set(KEYS) and "pair_flag" in b
    assert b["pair_key"].shape == (2, 4, DC.P) and b["q_word"].shape[:3] == (2, 4, DC.Qn)
    st = strata_episodes(DC, 3, seed=7)
    assert set(st) == {n for n, _ in STRATA} and st["alt32"].q_k[:, 0].tolist() == [32, 32, 32]


def test_routed_runs_are_independent_and_teacher_loss_is_zero_on_the_planted_router():
    from topology_design.route.train_route import teacher_loss, policy, flat, action_accuracy
    rng = np.random.default_rng(8)
    ep = generate(DC, 4, rng, Qn=3)
    b = _batch(ep, 2)
    m = StackedRouter(2, RouterConfig(), DC)
    fb = flat(b); mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], DC)
    pi = policy(m, b, 14)
    q, es, _, _ = marginal(pi, mp, fb["q_start"], 14, readout(DC, 0.02, "cpu"))
    loss0 = -torch.log(q[:4].gather(-1, (fb["q_ans"][:4] - DC.n_keys).unsqueeze(-1)) + 1e-9).mean()    # run 0 only
    loss0.backward()
    assert m.head.weight.grad[0].abs().max() > 0 and m.head.weight.grad[1].abs().max() == 0
    pm = plant(StackedRouter(1, RouterConfig(), DC), gain=6.0, hop_bias=8.0)
    with torch.no_grad():
        b1 = _batch(ep, 1); fb1 = flat(b1)
        mp1 = maps(fb1["pair_key"], fb1["pair_val"], fb1["pair_prev"], fb1["pair_flag"], DC)
        pi1 = policy(pm, b1, 14)
        oa, osteps, ostates = oracle(mp1, fb1["q_start"], 14, DC)
        tl = teacher_loss(pi1, oa, ostates)
        assert float(tl.max()) < 0.1
        h, c = action_accuracy(pi1, oa, ostates)
        assert (h == c).all() and c[SET] > 0 and c[PREV] > 0
        uni = torch.full_like(pi1, 1 / 4)
        tlu = teacher_loss(uni, oa, ostates)
        assert torch.allclose(tlu, ((osteps.float() + 1) * np.log(4)).to(tlu.dtype), atol=1e-4)
        # the chunked evaluation equals the whole
        assert torch.allclose(policy(pm, b1, 14, chunk=2), pi1)


def test_neural_baseline_reads_the_flag():
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
        e0, *_ = m.embed(b)
        b2 = {**b, "pair_flag": 1 - b["pair_flag"]}
        e1, *_ = m.embed(b2)
        diff = (e1 - e0)[:, :, 1:1 + DC.P]
        assert torch.allclose(diff.norm(dim=-1), torch.ones_like(diff.norm(dim=-1)), atol=1e-5)     # one unit along the flag direction


def test_trainers_smoke_and_record_fields():
    from topology_design.route import train_route, train_neural
    torch.manual_seed(0)
    base = dict(steps=2, batch=4, pool_n=40, eval_n=2, eval_every=2, crit_n=6, crit_every=1, warmup=1, T_eval=20)
    recs = train_route.run_stack([{**base, "seed": 0}, {**base, "seed": 1}], dev="cpu", log=lambda s: None)
    c, e = recs[0]["curve"][-1], recs[0]["evals"][-1]["eval"]
    assert {"acc_by_k", "ans_acc_by_k", "acc_by_Pdepth", "act_acc", "train_acc_by_k"} <= set(c)
    assert {"route_acc", "acc", "marg_acc", "q_steps", "oracle_steps", "e_steps", "act_acc"} <= set(e["in4"])
    recs = train_route.run_stack([{**base, "kind": "index", "seed": 0}], dev="cpu", log=lambda s: None)
    assert "table" in recs[0]["head"]
    recs = train_neural.run_stack([{**base, "kind": "looped", "halting": "fixed", "T": 3, "T_eval": 6, "seed": 0}], dev="cpu", log=lambda s: None)
    e = recs[0]["evals"][-1]["eval"]["in4"]
    assert {"acc_at_k", "acc_at_k1", "acc_at_kP", "acc_any", "kl_exit"} <= set(e) and "eval_T_train" in recs[0]
