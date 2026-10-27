"""Tests for rung 0, pointer chasing (experiments/ptr/DESIGN.md §10, gate G1): generator, exact executor, the planted
hop block, halting semantics and losses."""

import numpy as np
import pytest
import torch

from topology_design.ptr.data import PtrConfig, generate, follow, remaining_hops, check_forest, zipf_pmf
from topology_design.ptr.model import ModelConfig, PtrModel, oracle_steps, remaining_after
from topology_design.ptr.planted import plant

DC = PtrConfig()


def test_generator_is_a_forest_with_correct_lengths():
    rng = np.random.default_rng(0)
    ep = generate(DC, 64, rng)
    check_forest(ep, DC)
    # answers never appear as keys; every value is a key or an answer
    assert (ep.pair_key < DC.n_keys).all()
    assert (ep.pair_val < DC.n_sym).all()
    assert ep.pair_key.max() < DC.n_keys_train and ep.q_ans.max() < DC.n_keys + DC.n_ans_train


def test_length_distribution_matches_zipf():
    rng = np.random.default_rng(1)
    ep = generate(DC, 2000, rng)
    ks = ep.q_k.ravel()
    emp = np.bincount(ks, minlength=DC.kmax + 1)[1:] / ks.size
    assert np.abs(emp - zipf_pmf(DC.alpha, DC.kmax)).max() < 0.02


def test_forced_k_and_tail_factor():
    rng = np.random.default_rng(2)
    ep = generate(DC, 8, rng, forced_k=32, Qn=3)
    check_forest(ep, DC)
    assert (ep.q_k == 32).sum(1).min() == 1
    rare = PtrConfig(tail_factor=0.01)
    ep2 = generate(rare, 500, rng)
    check_forest(ep2, rare)
    assert (ep2.q_k >= rare.k_deep).mean() < 0.01


def test_executor_and_branch_value():
    rng = np.random.default_rng(3)
    ep = generate(DC, 4, rng)
    for b in range(4):
        for s, k in zip(ep.q_start[b], ep.q_k[b]):
            for t in range(int(k) + 1):
                assert remaining_hops(ep, b, follow(ep, b, int(s), t), DC.n_keys) == int(k) - t


@pytest.mark.parametrize("mask", [False, True])
def test_planted_block_is_an_exact_hop(mask):
    rng = np.random.default_rng(4)
    for halting, T in (("oracle", 40), ("fixed", 40)):
        for single in (True, False):
            m = plant(PtrModel(ModelConfig(kind="looped", halting=halting, injection=False, pair_pair_mask=mask,
                                           ctx_single_step=single, T=T), DC))
            with torch.no_grad():
                for k in (1, 3, 8, 32):
                    ep = generate(DC, 8, rng, forced_k=k, Qn=3)
                    out = m(ep, "cpu", steps=oracle_steps(ep, DC, "cpu"), mode="eval")
                    assert (out["logits"].argmax(-1) == out["targets"]).all()
                    # answers are absorbing and context tokens are fixed points: states stop changing once done
                    hs = out["hs"]
                    assert torch.allclose(hs[-1][:, 1:DC.P + 1], hs[0][:, 1:DC.P + 1], atol=1e-4)
                    if halting == "fixed":
                        assert torch.allclose(hs[-1][:, out["qpos"]], hs[33][:, out["qpos"]], atol=1e-4)
    # too few loops: the long chains fail, the short ones still succeed
    m = plant(PtrModel(ModelConfig(kind="looped", halting="fixed", injection=False, T=4), DC))
    with torch.no_grad():
        ep = generate(DC, 8, rng, forced_k=16, Qn=3)
        out = m(ep, "cpu", mode="eval")
        ok = (out["logits"].argmax(-1) == out["targets"]).numpy()
        assert ok[ep.q_k <= 4].all() and not ok[ep.q_k == 16].any()


def test_oracle_halting_counts_and_freezes():
    rng = np.random.default_rng(5)
    ep = generate(DC, 4, rng)
    st = oracle_steps(ep, DC, "cpu")
    m = PtrModel(ModelConfig(kind="looped", halting="oracle", T=10), DC)
    with torch.no_grad():
        out = m(ep, "cpu", steps=st, mode="eval")
    assert torch.equal(out["steps"], st)
    # a token frozen at step s has the same state at every later step
    hs = out["hs"]
    for b in range(4):
        for n in range(st.shape[1]):
            s = int(st[b, n])
            for t in range(s, len(hs)):
                assert torch.allclose(hs[t][b, n], hs[s - 1][b, n])


def test_pondernet_halting_distribution_sums_to_one_and_losses_are_finite():
    rng = np.random.default_rng(6)
    ep = generate(DC, 3, rng)
    for halt_input in ("state", "state+index", "index"):
        m = PtrModel(ModelConfig(kind="looped", halting="learned", halt_input=halt_input, T=6), DC)
        out = m(ep, "cpu", mode="ponder_train")
        assert torch.allclose(out["p_halt"].sum(0), torch.ones_like(out["p_halt"][0]), atol=1e-5)
        # context tokens run exactly one step in training as at test
        assert (out["steps"][:, 1:DC.P + 1] == 1).all() and (out["steps"][:, out["qpos"]] == 6).all()
        loss, extra = m.loss(out, ep, "cpu")
        loss.backward()
        assert torch.isfinite(loss) and extra["kl"] >= 0
        with torch.no_grad():
            ev = m(ep, "cpu", mode="eval")
        assert (ev["steps"] >= 1).all() and (ev["steps"] <= 6).all() and (ev["steps"][:, 1:DC.P + 1] == 1).all()


def test_value_arm_and_static_arm():
    rng = np.random.default_rng(7)
    ep = generate(DC, 3, rng)
    m = PtrModel(ModelConfig(kind="looped", halting="value", T=5), DC)
    out = m(ep, "cpu", mode="train")
    loss, extra = m.loss(out, ep, "cpu", remaining_hops=remaining_after(ep, DC, 5, "cpu"))
    loss.backward()
    assert torch.isfinite(loss) and "value" in extra
    s = PtrModel(ModelConfig(kind="static", layers=8), DC)
    out = s(ep, "cpu")
    loss, _ = s.loss(out, ep, "cpu")
    loss.backward()
    assert torch.isfinite(loss) and out["logits"].shape == (3, DC.Qn, DC.n_ans)


def test_parameter_matched_static_width():
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    s8 = n(PtrModel(ModelConfig(kind="static", d=96, layers=8), DC))
    lp = n(PtrModel(ModelConfig(kind="looped", halting="learned"), DC))
    assert abs(s8 - lp) / lp < 0.1
