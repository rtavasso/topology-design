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


def test_parent_vector_arms_and_depths():
    assert gk.arm_spec("pk0102") == ((0, 1, 0, 2), (1.0, 1.0, 1.0, 1.0))
    assert gk.block_depths((0, 1, 2, 3)) == [1, 2, 3, 4]
    assert gk.block_depths((0, 0, 0, 0)) == [1, 1, 1, 1]
    assert gk.block_depths((0, 1, 0, 2)) == [1, 2, 1, 3]  # block 4 reads h_2, deepest contributor = block 2
    assert gk.block_depths((0, 1, 1, 3)) == [1, 2, 2, 3]
    assert gk.root_readers((0, 1, 0, 2)) == 2
    from topology_design import linear_family as lf
    chains = {gk.longest_chain(k) for k in lf.all_parent_vectors(4)}
    assert chains == {1, 2, 3, 4}


def test_divergence_is_caught_and_flagged():
    cfg = gk.RunConfig(arm="pk0123", p=11, d=16, hidden=32, steps=200, eval_every=10, stop_after_grok=0,
                       lr=50.0, weight_decay=0.0, init_scale=5.0)
    res = gk.train_run(cfg)
    s = res["summary"]
    assert s["stable"] is False
    assert s["diverged_at"] is not None or s["post_fit_min_train_acc"] < 0.9


def test_stable_flag_on_benign_run():
    cfg = gk.RunConfig(arm="pk0000", p=11, d=16, hidden=64, steps=400, eval_every=20, stop_after_grok=0,
                       lr=3e-3, weight_decay=0.1, train_frac=0.8)
    s = gk.train_run(cfg)["summary"]
    assert s["diverged_at"] is None and s["post_fit_min_train_acc"] >= 0.0


# ---------------------------------------------------------------------------
# path_measure (experiments/paths): exactness of the function-space accounting
# ---------------------------------------------------------------------------


def _trained(arm, steps=60, p=11):
    cfg = gk.RunConfig(arm=arm, p=p, d=16, hidden=32, lr=3e-3, steps=steps, eval_every=steps, stop_after_grok=0)
    model = gk.build_model(cfg)
    tr, te = gk.split(p, 0.4, 0)
    a, b, y = gk.full_grid(p)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1.0)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(a[tr], b[tr]), y[tr]).backward()
        opt.step()
    return model, tr, te


def test_path_measure_is_an_exact_additive_decomposition():
    for arm in ("pk0000", "pk0120", "pk0123"):
        model, tr, te = _trained(arm)
        pm = gk.path_measure(model, tr, te)
        a, b, y = gk.full_grid(model.p)
        Zc = gk.centered(model(a, b))
        g = Zc[te, y[te]].mean().item()
        assert abs(sum(pm["gen"]) - g) < 1e-4 and abs(pm["gen_total"] - g) < 1e-4
        assert abs(sum(pm["mem"]) - (Zc[tr, y[tr]].mean().item() - g)) < 1e-4
        assert abs(sum(pm["shap_rule"]) - pm["rule_energy_total"]) < 1e-4 * max(1.0, pm["rule_energy_total"])
        assert abs(sum(pm["shap_mem"]) - pm["mem_energy_total"]) < 1e-4 * max(1.0, pm["mem_energy_total"])
        # total-effect Shapley sums to (full network) - (embedding path only)
        assert abs(sum(pm["gen_tshap"]) - (pm["gen_total"] - pm["abl_sets"]["1111"]["gen"])) < 1e-4
        assert abs(sum(pm["mem_tshap"]) - (pm["mem_total"] - pm["abl_sets"]["1111"]["mem"])) < 1e-4
        assert len(pm["gen"]) == model.L + 1 and pm["depth"] == gk.block_depths(model.parents)


def test_support_does_not_credit_memorization_as_generalization():
    p = 11
    a, b, y = gk.full_grid(p)
    tr, te = gk.split(p, 0.4, 0)
    lookup = torch.zeros(p * p, p)
    lookup[tr, y[tr]] = 5.0  # boosts train labels only
    g, m = gk.support(gk.centered(lookup), tr, te, y)
    assert abs(g) < 1e-5 and m > 1.0
    rule = torch.nn.functional.one_hot(y, p).float() * 3.0  # a pure function of the rule
    g, m = gk.support(gk.centered(rule), tr, te, y)
    assert g > 1.0 and abs(m) < 1e-5
    assert gk.rule_projection_test(gk.centered(lookup), p, te)[te].abs().max() < 1e-5 + 5.0 / p


def test_subset_shapley_additive_game():
    v = {"".join(str(m >> i & 1) for i in range(3)): float(sum((m >> i & 1) * (i + 1) for i in range(3))) for m in range(8)}
    assert gk.subset_shapley(v, 3) == pytest.approx([1.0, 2.0, 3.0])


def test_dense_early_eval_schedule():
    cfg = gk.RunConfig(dense_early=True, eval_every=100)
    steps = [s for s in range(1200) if gk.is_eval_step(s, cfg)]
    assert steps[:3] == [0, 10, 20] and 200 in steps and 225 in steps and 1100 in steps and 1050 not in steps


def test_composition_dependence_zero_for_root_readers_and_positive_in_chain():
    model, tr, te = _trained("pk0000")
    pm = gk.path_measure(model, tr, te)
    assert pm["comp_dep_state"] == [0.0] * 4 and pm["abl_long"] is None and pm["abl_short"] is not None
    model, tr, te = _trained("pk0123")
    pm = gk.path_measure(model, tr, te)
    assert pm["comp_dep_state"][0] == 0.0 and all(v > 0 for v in pm["comp_dep_state"][1:])
    assert pm["abl_long"] is not None


def test_group_ablation_matches_manual_alpha_zeroing():
    model, tr, te = _trained("pk0120")  # depths 1,2,3,1: short = blocks 1 and 4
    pm = gk.path_measure(model, tr, te)
    a, b, y = gk.full_grid(model.p)
    al = model.alphas.clone()
    al[[0, 3]] = 0.0
    lg = model(a, b, alphas=al)
    assert pm["abl_short"]["test_acc"] == gk.acc(lg[te], y[te])
    al = model.alphas.clone()
    al[1] = 0.0
    assert pm["abl_block"][1]["test_acc"] == gk.acc(model(a, b, alphas=al)[te], y[te]) and len(pm["abl_block"]) == 4
    assert len(pm["abl_sets"]) == 15 and pm["abl_sets"]["1001"] == pm["abl_short"] and pm["abl_sets"]["0110"] == pm["abl_long"]
    al = model.alphas.clone()
    al[[1, 2]] = 0.0
    lg = model(a, b, alphas=al)
    Zc = gk.centered(lg)
    assert abs(pm["abl_sets"]["0110"]["gen"] - Zc[te, y[te]].mean().item()) < 1e-5
    assert "paths" in gk.measure(model, tr, te)
