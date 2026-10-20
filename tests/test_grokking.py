"""Tests for the grokking testbed (src/topology_design/grokking.py)."""

import pytest
import torch

from topology_design import grokking as gk


@pytest.mark.parametrize("arm", ["seq", "par", "sink_0.25", "sink_4"])
@pytest.mark.parametrize("beta", [1.0, 0.7])
def test_branch_decomposition_is_exact(arm, beta):
    cfg = gk.RunConfig(arm=arm, p=11, d=16, hidden=32, beta=beta, out_scale=1.7, init_scale=2.0)
    model = gk.build_model(cfg)
    a, b, _ = gk.full_grid(11)
    logits, parts = model(a, b, return_parts=True)
    torch.testing.assert_close(parts.sum(0), logits, atol=1e-5, rtol=1e-5)


def test_all_arms_have_identical_parameters_and_init():
    cfgs = [gk.RunConfig(arm=a, p=11, d=16, hidden=32) for a in ["seq", "par", "sink_0.5", "sink_2"]]
    models = [gk.build_model(c) for c in cfgs]
    counts = {gk.n_params(m) for m in models}
    assert len(counts) == 1
    for m in models[1:]:
        for p0, p1 in zip(models[0].parameters(), m.parameters()):
            assert torch.equal(p0, p1)  # same seed -> bit-identical initial weights across arms


def test_ablation_matches_model_without_block():
    cfg = gk.RunConfig(arm="sink_1", p=11, d=16, hidden=32)
    model = gk.build_model(cfg)
    a, b, _ = gk.full_grid(11)
    al = model.alphas.clone()
    al[2] = 0.0
    lg_abl = model(a, b, alphas=al)
    # sink (block 3) reads h0 and writes last: removing it leaves embedding path + chain
    _, parts = model(a, b, return_parts=True)
    torch.testing.assert_close(lg_abl, parts[0] + parts[1] + parts[2], atol=1e-5, rtol=1e-5)


def test_rule_projection_properties():
    p = 13
    a, b, y = gk.full_grid(p)
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(p * p, p, generator=g)
    R = gk.rule_projection(Z, p)
    torch.testing.assert_close(gk.rule_projection(R, p), R)  # idempotent
    f = torch.randn(p, p, generator=g)
    Zrule = f[(a + b) % p]  # any function of (a+b, c)
    torch.testing.assert_close(gk.rule_projection(Zrule, p), Zrule)  # lives entirely in the rule subspace
    # a perfect modular-addition solution has rule_frac = 1 and restricted acc = 1
    perfect = 5.0 * torch.nn.functional.one_hot(y, p).float()
    Rp = gk.rule_projection(gk.centered(perfect), p)
    assert gk.acc(Rp, y) == 1.0
    # random logits put only ~1/p of their energy in the rule subspace
    frac = (R.pow(2).sum() / Z.pow(2).sum()).item()
    assert 0.3 / p < frac < 3.0 / p


def test_split_is_deterministic_disjoint_and_complete():
    tr, te = gk.split(23, 0.4, data_seed=3)
    tr2, te2 = gk.split(23, 0.4, data_seed=3)
    assert torch.equal(tr, tr2) and torch.equal(te, te2)
    assert len(set(tr.tolist()) & set(te.tolist())) == 0
    assert len(tr) + len(te) == 23 * 23
    assert not torch.equal(tr, gk.split(23, 0.4, data_seed=4)[0])


def test_invalid_parents_rejected():
    with pytest.raises(ValueError):
        gk.RoutedMLP(5, 4, 8, parents=(0, 2), alphas=(1.0, 1.0))


def test_smoke_training_reduces_loss_and_records_measurements():
    cfg = gk.RunConfig(arm="sink_0.5", p=11, d=16, hidden=32, steps=60, eval_every=20, stop_after_grok=0)
    res = gk.train_run(cfg)
    traj = res["trajectory"]
    assert traj[-1]["train_loss"] < traj[0]["train_loss"]
    rec = traj[-1]
    assert len(rec["branches"]) == 4 and len(rec["ablate"]) == 3
    assert set(rec["grad_norms"]) == {"embed", "unembed", "block1", "block2", "block3"}
    assert 0.0 <= rec["rule_frac"] <= 1.0
    assert set(res["summary"]) >= {"t_fit", "t_grok", "grok_delay", "peak_excluded_train_acc"}
