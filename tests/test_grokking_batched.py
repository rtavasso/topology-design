"""The batched trainer must reproduce single-run training and keep runs isolated from each other."""

import pytest
import torch

from topology_design import grokking as gk
from topology_design import grokking_batched as gb

SMALL = dict(p=13, d=16, hidden=32, steps=40, eval_every=10, stop_after_grok=0)


def _single_params_after(cfg, steps):
    torch.manual_seed(0)
    model = gk.build_model(cfg)
    tr, _ = gk.split(cfg.p, cfg.train_frac, cfg.data_seed)
    a, b, y = gk.full_grid(cfg.p)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(a[tr], b[tr]), y[tr]).backward()
        opt.step()
    return model


def _batched_params_after(cfgs, steps):
    bm = gb.BatchedRoutedMLP(cfgs)
    a, b, y = gk.full_grid(cfgs[0].p)
    splits = [gk.split(c.p, c.train_frac, c.data_seed) for c in cfgs]
    tr = torch.stack([s[0] for s in splits])
    A, B, Y = a[tr], b[tr], y[tr]
    lr = torch.tensor([c.lr for c in cfgs])
    wd = torch.tensor([c.weight_decay for c in cfgs])
    opt = gb.BatchedAdamW(bm.params, lr, wd)
    R, n = A.shape
    for _ in range(steps):
        opt.zero_grad()
        lg = bm.forward(A, B)
        torch.nn.functional.cross_entropy(lg.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1).sum().backward()
        opt.step()
    return bm


def test_mixed_batch_matches_single_runs():
    # one batch mixing sink multipliers, seeds, data seeds, lrs and weight decays (all share parents (0,1,0))
    cfgs = [
        gk.RunConfig(arm="sink_0.5", model_seed=0, data_seed=0, lr=1e-3, weight_decay=1.0, **SMALL),
        gk.RunConfig(arm="sink_2", model_seed=1, data_seed=3, lr=3e-3, weight_decay=0.0, **SMALL),
        gk.RunConfig(arm="sink_1", model_seed=2, data_seed=1, lr=3e-4, weight_decay=0.5, **SMALL),
    ]
    steps = 25
    bm = _batched_params_after(cfgs, steps)
    for j, cfg in enumerate(cfgs):
        single = _single_params_after(cfg, steps)
        batched = bm.single(j)
        for (name, ps), pb in zip(single.named_parameters(), batched.parameters()):
            torch.testing.assert_close(pb, ps, atol=2e-5, rtol=2e-4, msg=f"run {j} param {name}")


def test_runs_in_a_batch_are_isolated():
    base = [gk.RunConfig(arm="seq", model_seed=s, data_seed=s, **SMALL) for s in range(3)]
    changed = list(base)
    changed[2] = gk.RunConfig(arm="seq", model_seed=7, data_seed=9, lr=3e-3, **SMALL)  # perturb run 2 only
    b1 = _batched_params_after(base, 15)
    b2 = _batched_params_after(changed, 15)
    for j in (0, 1):  # runs 0 and 1 must be bit-identical regardless of what run 2 does
        for k in b1.params:
            assert torch.equal(b1.params[k][j], b2.params[k][j]), (j, k)


def test_batch_rejects_mixed_structures():
    with pytest.raises(ValueError):
        gb.BatchedRoutedMLP([gk.RunConfig(arm="seq", **SMALL), gk.RunConfig(arm="par", **SMALL)])


def test_train_batch_output_matches_single_format_and_early_trajectory():
    cfgs = [gk.RunConfig(arm="par", model_seed=s, data_seed=s, **SMALL) for s in range(2)]
    outs = gb.train_batch(cfgs)
    for cfg, out in zip(cfgs, outs):
        single = gk.train_run(cfg)
        assert out["run_id"] == single["run_id"]
        assert set(out["summary"]) == set(single["summary"])
        assert len(out["trajectory"]) == len(single["trajectory"])
        for rb, rs in zip(out["trajectory"][:3], single["trajectory"][:3]):  # early checkpoints agree closely
            assert rb["train_loss"] == pytest.approx(rs["train_loss"], rel=1e-3, abs=1e-5)
            assert rb["test_acc"] == pytest.approx(rs["test_acc"], abs=1e-6)
            assert rb["residual_norm"] == pytest.approx(rs["residual_norm"], rel=1e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU (MPS)")
def test_mps_batch_matches_cpu_single_runs():
    cfgs = [
        gk.RunConfig(arm="sink_0.5", model_seed=0, data_seed=0, lr=1e-3, weight_decay=1.0, **SMALL),
        gk.RunConfig(arm="sink_2", model_seed=1, data_seed=3, lr=3e-3, weight_decay=0.0, **SMALL),
    ]
    steps = 25
    bm = gb.BatchedRoutedMLP(cfgs, device="mps")
    a, b, y = gk.full_grid(cfgs[0].p)
    tr = torch.stack([gk.split(c.p, c.train_frac, c.data_seed)[0] for c in cfgs])
    A, B, Y = a[tr].to("mps"), b[tr].to("mps"), y[tr].to("mps")
    opt = gb.BatchedAdamW(bm.params, torch.tensor([c.lr for c in cfgs], device="mps"),
                          torch.tensor([c.weight_decay for c in cfgs], device="mps"))
    R, n = A.shape
    for _ in range(steps):
        opt.zero_grad()
        lg = bm.forward(A, B)
        torch.nn.functional.cross_entropy(lg.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1).sum().backward()
        opt.step()
    for j, cfg in enumerate(cfgs):
        single = _single_params_after(cfg, steps)
        for (name, ps), pb in zip(single.named_parameters(), bm.single(j).parameters()):
            torch.testing.assert_close(pb, ps, atol=2e-4, rtol=2e-3, msg=f"run {j} param {name}")
    outs = gb.train_batch(cfgs, device="mps")  # full path, including measurements, runs on MPS
    assert all(len(o["trajectory"]) == len(gk.train_run(c)["trajectory"]) for o, c in zip(outs, cfgs))
