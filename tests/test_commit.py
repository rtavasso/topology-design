"""Rung 3 (experiments/commit/DESIGN.md §8, G1): checkpoints, the constant schedule, the beam's completions and the
forced-breadth losses, the entropy bonus, program identity on isolated-P words, PP set placement, the gate."""
import math
import os

import numpy as np
import pytest
import torch

from topology_design.route.data import RouteConfig, generate, P
from topology_design.route.exec import tables, oracle, route_ok, sid, PER_SYM, HOP, PREV, SET, HALT
from topology_design.route import estimators as E
from topology_design.route.router import StackedRouter, RouterConfig
from topology_design.route.train_route import (run_stack, router_state, load_router_state_, per_run_init_, policy, flat,
                                               isolated_P, pp_set_placement, program_stats)
from topology_design.route.pool import to_dict

DC = RouteConfig()
SMALL = dict(steps=2, batch=4, pool_n=40, eval_n=2, eval_every=2, crit_n=6, crit_every=1, warmup=1, T_eval=20, K=4, W=4, per_run_init=True)


def _batch(dc, n=2, seed=0):
    ep = generate(dc, n, np.random.default_rng(seed))
    return {k: v.unsqueeze(0) for k, v in to_dict(ep, "cpu").items()}, ep


def test_checkpoint_round_trip_across_stack_sizes_and_through_the_trainer(tmp_path):
    rc = RouterConfig()
    torch.manual_seed(0); m3 = StackedRouter(3, rc, DC); per_run_init_(m3, [0, 1, 2], rc, DC)
    st = router_state(m3, 1)
    assert all(not v.requires_grad and v.device.type == "cpu" for v in st.values())
    torch.manual_seed(1); m1 = StackedRouter(1, rc, DC)
    load_router_state_(m1, 0, st)
    b, _ = _batch(DC)
    lg3 = m3.logits(b.__class__((k, v.expand(3, *v.shape[1:])) for k, v in b.items()))[1]
    lg1 = m1.logits(b)[0]
    assert torch.allclose(lg3, lg1, atol=1e-6)
    # through the trainer: save_state then init_from, with a different data config (shift A) and a different stack
    recs = run_stack([{**SMALL, "seed": 5, "save_state": True, "estimator": "exact"}], dev="cpu", log=lambda s: None)
    assert "state" in recs[0] and set(recs[0]["state"]["state"]) == {n for n, _ in m1.named_parameters()}
    p = tmp_path / "r.pt"; torch.save(recs[0]["state"], p)
    m_saved = StackedRouter(1, rc, DC); load_router_state_(m_saved, 0, recs[0]["state"]["state"])
    recs2 = run_stack([{**SMALL, "seed": 7, "deferral": 0, "init_from": str(p), "estimator": "pf", "steps": 1, "crit_every": 1},
                       {**SMALL, "seed": 8, "deferral": 0, "init_from": str(p), "estimator": "pf", "steps": 1, "crit_every": 1}],
                      dev="cpu", log=lambda s: None)
    assert recs2[0]["train"]["init_from"] == "r.pt" and recs2[1]["train"]["init_from"] == "r.pt"
    with pytest.raises(AssertionError):
        run_stack([{**SMALL, "seed": 7, "init_from": str(p), "d": 344, "heads": 1}], dev="cpu", log=lambda s: None)


def test_constant_schedule_and_short_warmup():
    recs = run_stack([{**SMALL, "seed": 0, "schedule": "constant", "warmup": 1, "steps": 3, "crit_every": 1}], dev="cpu", log=lambda s: None)
    assert recs[0]["train"]["schedule"] == "constant" and len(recs[0]["curve"]) == 3


def test_beam_completions_contain_the_beam_search_route_and_are_valid_routes():
    dc = RouteConfig()
    for deferral in (1, 0):
        dc = RouteConfig(deferral=deferral)
        ep = generate(dc, 3, np.random.default_rng(1)); tb = tables(ep, dc, "cpu")
        NS = dc.n_sym * PER_SYM
        torch.manual_seed(0)
        pi = torch.softmax(torch.randn(3, NS, 4), -1)
        comp = E.beam_completions(pi, tb, tb["q_start"], 14, 8, dc, eps_mix=0.1, lam=0.02)
        B, Q, M, T = comp["acts"].shape
        assert M == 8 * 15
        # every valid completion: replay reaches its recorded end state with its recorded number of moves
        xs = E.replay(tb, tb["q_start"], comp["acts"])
        for b in range(B):
            for q in range(Q):
                for m in range(M):
                    if not comp["valid"][b, q, m]:
                        continue
                    a = comp["acts"][b, q, m]
                    n_act = int((a >= 0).sum()); mv = int(((a >= 0) & (a != HALT)).sum())
                    assert mv == int(comp["moves"][b, q, m])
                    assert int(xs[b, q, m, n_act]) == int(comp["end"][b, q, m])
        # the best correct completion under log pi~ - lam moves equals beam_search's route
        found, a_b, st_b = E.beam_search(pi, tb, tb["q_start"], tb["q_ans"], 14, 8, dc, eps_mix=0.1, lam=0.02)
        right = (comp["end"] // PER_SYM == tb["q_ans"].unsqueeze(-1)) & comp["valid"]
        key = torch.where(right, comp["logp_mix"].double() - 0.02 * comp["moves"].double(), torch.full(right.shape, -math.inf, dtype=torch.float64))
        best = key.max(-1)
        assert torch.equal(torch.isfinite(best.values), found)
        for b in range(B):
            for q in range(Q):
                if found[b, q]:
                    assert torch.equal(comp["acts"][b, q, best.indices[b, q]], a_b[b, q]), (b, q)


def test_pfb_weights_are_the_fixed_prior_posterior_on_the_support_and_rlb_gradient_is_the_weighted_policy_gradient():
    dc = RouteConfig()
    ep = generate(dc, 2, np.random.default_rng(2)); tb = tables(ep, dc, "cpu")
    NS = dc.n_sym * PER_SYM
    torch.manual_seed(1)
    logits = torch.randn(2, NS, 4, requires_grad=True)
    pi = torch.softmax(logits, -1)
    comp = E.beam_completions(pi.detach(), tb, tb["q_start"], 14, 4, dc, eps_mix=0.1, lam=0.02)
    xs = E.replay(tb, tb["q_start"], comp["acts"])
    loss, ok = E.pfb_loss(pi, comp, tb["q_ans"], dc, 0.02, 0.02, xs)
    # brute force the weights for query (0, 0)
    lp0 = E.route_logprob(E.flat_prior_like(pi, dc), comp["acts"], xs)[0, 0]
    sym = comp["end"][0, 0] // PER_SYM
    Ly = E.answer_likelihood(sym, tb["q_ans"][0, 0].expand_as(sym), dc, 0.02)
    w = torch.where(comp["valid"][0, 0], (lp0 - 0.02 * comp["moves"][0, 0].double() + torch.log(Ly)), torch.full_like(lp0, -math.inf))
    w = torch.softmax(w, -1)
    lpr = E.route_logprob(pi, comp["acts"], xs)[0, 0]
    ref = -(w.float() * lpr).sum()
    assert float(loss[0, 0]) == pytest.approx(float(ref), rel=1e-5) and float(ok[0, 0]) == 1.0
    # rlb: gradient of -sum_r w_r (R_r - b) log pi(r), w self-normalized under pi
    reward = E.readout_logp(comp["end"] // PER_SYM, tb["q_ans"].unsqueeze(-1), dc, 0.02).exp()
    sur, obj, ess = E.rlb_loss(pi, comp, reward, 0.02, xs)
    g, = torch.autograd.grad(sur[0, 0], logits, retain_graph=True)
    ws = torch.softmax(torch.where(comp["valid"][0, 0], lpr.detach(), torch.full_like(lpr, -math.inf)), -1)
    R = torch.where(comp["valid"][0, 0], reward[0, 0] - 0.02 * comp["moves"][0, 0].float(), torch.zeros_like(reward[0, 0]))
    b = (ws * R).sum()
    ref = -(ws * (R - b) * lpr).sum()
    g2, = torch.autograd.grad(ref, logits)
    assert torch.allclose(g, g2, atol=1e-6) and float(obj[0, 0]) == pytest.approx(-float(b), rel=1e-5)
    assert 0 < float(ess[0, 0]) <= 1


def test_entropy_bonus_pushes_toward_uniform_and_route_entropy_is_the_visited_mean():
    pi = torch.tensor([[[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.97, 0.01, 0.01, 0.01]]], requires_grad=True)
    acts = torch.tensor([[[[HOP, HOP, HALT, -1]]]]); states = torch.tensor([[[[0, 2, 1, 1, 1]]]])
    H = E.route_entropy(pi, acts, states)
    h = lambda p: -sum(x * math.log(x) for x in p)
    assert float(H) == pytest.approx((h([0.7, 0.1, 0.1, 0.1]) + h([0.97, 0.01, 0.01, 0.01]) + h([0.25] * 4)) / 3, rel=1e-5)
    lp = torch.zeros(1, 1, 2); steps = torch.ones(1, 1, 2); lpr = torch.zeros(1, 1, 2, requires_grad=True)
    ent = torch.stack([H, H], -1).view(1, 1, 2)
    sur, _ = E.rl_loss(lp, steps, lpr, 0.02, entropy=ent, beta=0.5)
    g, = torch.autograd.grad(sur.sum(), pi)
    assert g[0, 0, 0] > 0 and g[0, 0, 1] < 0          # descending the surrogate lowers the dominant prob, raises the others


def test_isolated_P_and_pp_set_placement_on_hand_built_routes():
    w = torch.tensor([[[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])
    assert isolated_P(w).tolist() == [[True, False, False]]
    cfg = RouteConfig(); A = cfg.n_keys
    s_ = lambda sym, c=0, h=0: sid(torch.tensor(sym), torch.tensor(c), torch.tensor(h)).item()
    # a word H P P H (k = 4): symbols 0 -H-> 1 ; P-steps leave 1 and 5 ; oracle (E under delta 1):
    # 0 -SET-> 0 -HOP-> 1 -SET-> 1 -PREV-> 5 -PREV-> 9 -HOP-> answer. P-leaving symbols 1 (first), 5 (second); predecessor 0.
    oa = torch.tensor([[[SET, HOP, SET, PREV, PREV, HOP, HALT, -1]]])
    ost = torch.tensor([[[s_(0), s_(0, 1, 1), s_(1, 1), s_(1, 2, 1), s_(5, 1), s_(9), s_(A), s_(A), s_(A)]]])
    qw = torch.tensor([[[0, 1, 1, 0]]])
    ok = torch.tensor([[True]])
    def place(acts, syms):
        st = torch.tensor([[[s_(x) for x in syms]]])            # (c, h) do not matter to the classifier
        out, L = pp_set_placement(acts, st, ok, oa, ost, qw)
        return out[0, 0].tolist(), int(L[0, 0])
    assert place(oa, [0, 0, 1, 1, 5, 9, A, A, A]) == ([1, 1, 0, 0, 0], 2)                       # E-E
    assert place(torch.tensor([[[HOP, SET, PREV, SET, PREV, HOP, HALT, -1]]]), [0, 1, 1, 5, 5, 9, A, A, A]) == ([0, 1, 1, 0, 0], 2)   # L-L
    assert place(torch.tensor([[[SET, HOP, PREV, SET, PREV, HOP, HALT, -1]]]), [0, 0, 1, 5, 5, 9, A, A, A]) == ([1, 0, 1, 0, 0], 2)   # E-L
    assert place(torch.tensor([[[HOP, SET, SET, PREV, PREV, HOP, HALT, -1]]]), [0, 1, 1, 1, 5, 9, A, A, A]) == ([0, 2, 0, 0, 0], 2)   # L-double
    assert place(torch.tensor([[[SET, SET, HOP, PREV, PREV, HOP, HALT, -1]]]), [0, 0, 0, 1, 5, 9, A, A, A]) == ([2, 0, 0, 0, 0], 2)   # E-double
    # not shortest-correct: not classified
    out, _ = pp_set_placement(oa, ost, torch.tensor([[False]]), oa, ost, qw)
    assert out[0, 0].tolist() == [-1] * 5
    # an isolated P before the run: its SET is attributed to that step and not counted. Word H P H P P H (k = 6):
    # 0 -H-> 1 -P-> 2 -H-> 3 -P-> 4 -P-> 5 ; oracle E: SET@0 HOP PREV(1->2) HOP SET@2.. wait: E sets at the predecessor of 3,
    # which is 2: 0 SET, HOP(1), PREV(->2), SET@2, HOP(3), SET@3, PREV(->4), PREV(->5), HOP(answer)
    oa2 = torch.tensor([[[SET, HOP, PREV, SET, HOP, SET, PREV, PREV, HOP, HALT, -1, -1]]])
    ost2 = torch.tensor([[[s_(0), s_(0, 1, 1), s_(1, 1), s_(2), s_(2, 1, 1), s_(3, 1), s_(3, 2, 1), s_(4, 1), s_(5), s_(A), s_(A), s_(A), s_(A)]]])
    qw2 = torch.tensor([[[0, 1, 0, 1, 1, 0]]])
    out, L = pp_set_placement(oa2, ost2, ok, oa2, ost2, qw2)
    assert out[0, 0].tolist() == [1, 1, 0, 0, 0] and int(L[0, 0]) == 2                             # the first SET (for the isolated P) is ignored


@pytest.mark.parametrize("est", ["rlb", "acb", "pfb", "rl"])
def test_trainer_runs_the_rung_3_arms(est):
    extra = {"value_head": True, "gae": 0.9, "critic_warmup": 1} if est == "acb" else ({"beta": 0.03} if est == "rl" else {})
    recs = run_stack([{**SMALL, "seed": 0, "estimator": est, "schedule": "constant", "pp_frac": 0.5, **extra},
                      {**SMALL, "seed": 1, "estimator": est, "schedule": "constant", "pp_frac": 0.5, **extra}], dev="cpu", log=lambda s: None)
    c = recs[0]["curve"][-1]
    assert math.isfinite(c["loss"]) and "set_late_iso" in c
    ev = recs[0]["evals"][-1]
    assert "clean" in ev and "pp_set_placement" in ev["eval"]["pp4"]
    if est in ("rlb", "acb", "pfb"):
        assert "beam_hit" in c["diag"]
    if est == "rl":
        assert "entropy" in c["diag"]


def test_dose_cells_are_evaluated_and_judged_on_the_standard_configuration():
    """Stage-2 review B2: a p_P = 0.08 run's strata hold a third of the P-words of the standard stream, so the movement
    endpoint (>= 0.95 at in4 and in8) and the criterion curve would get noisier exactly along the dose. The clean
    evaluation and the criterion set are generated on the standard configuration (p_P 0.3, pp_frac 0) instead."""
    cfg = {**SMALL, "seed": 3, "deferral": 0, "steps": 1, "crit_every": 1, "eval_every": 1, "eval_n": 12, "crit_n": 40}
    dose = run_stack([{**cfg, "p_P": 0.08}], dev="cpu", log=lambda s: None)[0]
    std = run_stack([{**cfg, "p_P": 0.3}], dev="cpu", log=lambda s: None)[0]
    ev_d, ev_s = dose["evals"][-1], std["evals"][-1]
    assert "clean" in ev_d and "clean" not in ev_s
    assert ev_d["eval"]["in8"]["n_P"] < ev_s["eval"]["in8"]["n_P"]                       # the training-stream strata lose P-words
    assert ev_d["clean"]["in8"]["n_P"] == ev_s["eval"]["in8"]["n_P"]                       # the clean strata are the standard ones
    assert dose["curve"][-1]["n_P"] == std["curve"][-1]["n_P"]                            # and so is the criterion set
    # shift B keeps its criterion set on the training stream (it carries min_acc_pp), and still gets the clean evaluation
    ppb = run_stack([{**cfg, "pp_frac": 1.0}], dev="cpu", log=lambda s: None)[0]
    assert "clean" in ppb["evals"][-1] and ppb["evals"][-1]["clean"]["in8"]["n_P"] == ev_s["eval"]["in8"]["n_P"]


def test_gate_logic_on_synthetic_records():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gate", os.path.join(os.path.dirname(__file__), "..", "experiments", "commit", "gate.py"))
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    def rec(late, mp=1.0, samp=1.0, plus=0.0):
        e = {st: {"min_acc_P": mp, "set_late_P": late, "set_other_P": 0.0, "samp_min_acc_P": samp, "q_steps": 5 + plus, "oracle_steps": 5.0, "acc": 1.0}
             for st in ("in4", "in8", "none8")}
        return {"evals": [{"eval": e}]}
    assert G.gate(rec(0.1), "E")[0] and not G.gate(rec(0.1), "L")[0]
    assert G.gate(rec(0.9), "L")[0] and not G.gate(rec(0.5), "L")[0]
    assert not G.gate(rec(0.1, mp=0.9), "E")[0] and not G.gate(rec(0.1, samp=0.5), "E")[0]
    ok, reasons, _ = G.gate(rec(0.1, plus=1.0), "E")
    assert not ok and any("R" in r for r in reasons)


def test_phase2_outcomes_on_synthetic_records():
    """moved / moved~ / moved-R / locked / drifted / halted / other under shift A, conditioned on the start (the same
    first-deviation key is the intact L program from L and the lost SET from E), and held / to-X / degraded / lost on the
    unchanged delta-1 stream; the first-deviation mass is over the stratum's queries, so a router right on 93 % of words
    that halts on the rest is not 'halted'."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "commit", "analyze.py")
    if not os.path.exists(path):
        pytest.skip("analysis code is not shipped in the Kaggle bundle (it runs locally); the test runs in the repo")
    spec = importlib.util.spec_from_file_location("cm_analyze", path)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    def rec(mp, fd, start="E", deferral=0, acc=1.0, plus=0.0, p_prev=1.0, late=1.0, curve_max=None, clean=True):
        e = {st: {"min_acc_P": mp, "n": 250, "n_P": 217, "first_dev": fd, "acc_P": acc, "acc": acc, "q_steps": 8.0 + plus, "oracle_steps": 8.0,
                  "set_late_P": late, "set_other_P": 0.0} for st in ("in4", "in8", "none8", "pp8")}
        ev = {"eval": {st: {**v, "min_acc_P": 0.0} for st, v in e.items()}}      # the mixture evaluation disagrees on purpose
        if clean:
            ev["clean"] = e
        cm = mp if curve_max is None else curve_max
        curve = [{"step": 100 + 100 * i, "min_acc_P": cm, "p_prev_class": p_prev, "p_set_class": 0.5} for i in range(3)]
        return {"evals": [ev], "train": {"warmup": 100, "steps": 3000, "init_from": f"cm1_{'EX' if start == 'E' else 'EI16'}_p1_s0.pt"},
                "data": {"deferral": deferral, "p_P": 0.3, "pp_frac": 0.0}, "curve": curve}
    assert M.outcome(rec(0.99, {"PREV>HOP": 2})) == "moved"
    assert M.outcome(rec(0.0, {"PREV>HOP": 215, "SET>HALT": 2}, acc=0.0)) == "locked"            # E intact under delta 0
    assert M.outcome(rec(0.0, {"SET>HOP": 220}, acc=0.0)) == "drifted"                          # E lost its SET: the all-H route
    assert M.outcome(rec(0.0, {"SET>HOP": 220}, start="L", acc=0.0, p_prev=1.0)) == "locked"     # L intact (PREV still in place)
    assert M.outcome(rec(0.0, {"SET>HOP": 220}, start="L", acc=0.0, p_prev=0.0)) == "drifted"    # L with PREV lost
    assert M.outcome(rec(0.0, {"SET>HALT": 216}, acc=0.0)) == "halted"
    assert M.outcome(rec(0.93, {"SET>HALT": 16}, curve_max=0.93)) == "other"                     # 16 / 250 halts is not a collapse
    assert M.outcome(rec(0.93, {"SET>HOP": 16}, curve_max=0.99)) == "moved~"                     # reached the criterion, ended at 0.93
    assert M.outcome(rec(0.0, {"HOP>SET": 250}, acc=0.98, plus=1.0)) == "moved-R"                # program R: correct, one step longer
    assert M.outcome(rec(0.0, {"PREV>HOP": 215}, acc=0.0, clean=False)) == "locked"             # no clean eval: the mixture one
    # delta 1: the unchanged stream and shift B
    assert M.outcome(rec(0.99, {}, start="L", deferral=1, late=1.0)) == "held-L"
    assert M.outcome(rec(0.99, {"SET>HOP": 30}, start="L", deferral=1, late=0.0)) == "to-E"
    assert M.outcome(rec(0.70, {"SET>HOP": 190}, start="L", deferral=1)) == "degraded"
    assert M.outcome(rec(0.01, {"SET>HOP": 210}, start="L", deferral=1, acc=0.0)) == "lost"
    r = rec(0.99, {}); r["evals"][0]["clean"]["pp8"]["min_acc"] = 0.92
    assert M.filled(r) and not M.filled(rec(0.99, {}))
