"""The fleet trainer must reproduce single-run training and measurement, and keep runs isolated."""

import pytest
import torch

from topology_design import fleet as fl
from topology_design import grokking as gk

SMALL = dict(p=13, d=16, hidden=32, steps=40, eval_every=10, dense_early=False)


def _single_after(cfg: fl.FleetConfig, steps: int, groups=None):
    """Reference: grokking.build_model + torch.optim.AdamW, on the same split and labels."""
    m = gk.build_model(gk.RunConfig(arm=cfg.arm, p=cfg.p, d=cfg.d, hidden=cfg.hidden, beta=cfg.beta,
                                    out_scale=cfg.out_scale, model_seed=cfg.model_seed, init_scale=cfg.init_scale))
    if cfg.alphas:
        m.alphas.copy_(torch.tensor(cfg.alphas))
    t = fl.build_task(cfg)
    a, b, y, tr = t["a"], t["b"], t["y"], t["tr"]
    if groups is None:
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    else:
        opt = torch.optim.AdamW(groups(m), betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(a[tr], b[tr]), y[tr]).backward()
        opt.step()
    return m


def _fleet_after(cfgs, steps):
    f = fl.Fleet(cfgs)
    tasks = [fl.build_task(c) for c in cfgs]
    tr = torch.stack([t["tr"] for t in tasks])
    a, b = tasks[0]["a"], tasks[0]["b"]
    Y = torch.stack([t["y"] for t in tasks]).gather(1, tr)
    opt = fl.FleetAdamW(f)
    R, n = tr.shape
    active = torch.ones(R, dtype=torch.bool)
    for _ in range(steps):
        opt.zero_grad()
        lg = f.forward(a[tr], b[tr])
        torch.nn.functional.cross_entropy(lg.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1).sum().backward()
        opt.step(active)
    return f


def _assert_same(f, j, single, atol=2e-5, rtol=2e-4):
    for (name, ps), pb in zip(single.named_parameters(), f.single(j).parameters()):
        torch.testing.assert_close(pb, ps, atol=atol, rtol=rtol, msg=f"run {j} param {name}")


def test_mixed_graphs_hyperparams_and_exceptions_match_single_runs():
    cfgs = [
        fl.FleetConfig(arm="pk0000", model_seed=0, data_seed=0, lr=1e-3, weight_decay=1.0, **SMALL),
        fl.FleetConfig(arm="pk0120", model_seed=1, data_seed=3, lr=3e-3, weight_decay=0.3, **SMALL),
        fl.FleetConfig(arm="pk0123", model_seed=2, data_seed=1, lr=2e-3, weight_decay=0.0, alphas=(1.0, 0.5, 2.0, 1.0),
                       out_scale=2.0, **SMALL),
    ]
    f = _fleet_after(cfgs, 25)
    for j, c in enumerate(cfgs):
        _assert_same(f, j, _single_after(c, 25))
    # the same with the exceptions task (labels differ per run)
    cfgs = [fl.FleetConfig(arm="pk0120", task="modadd_exc", exc_frac=0.2, model_seed=s, data_seed=s, **SMALL)
            for s in range(2)]
    f = _fleet_after(cfgs, 20)
    for j, c in enumerate(cfgs):
        _assert_same(f, j, _single_after(c, 20))


def test_per_block_lr_and_wd_multipliers_match_param_groups():
    cfg = fl.FleetConfig(arm="pk0120", lr=2e-3, weight_decay=1.0, block_lr_mult=(1.0, 2.0, 0.5, 4.0),
                         block_wd_mult=(1.0, 0.5, 2.0, 0.25), model_seed=3, data_seed=3, **SMALL)

    def groups(m):
        g = [{"params": [m.embed_a.weight, m.embed_b.weight, m.unembed.weight], "lr": cfg.lr,
              "weight_decay": cfg.weight_decay}]
        for i in range(4):
            g.append({"params": [m.w1[i].weight, m.w2[i].weight], "lr": cfg.lr * cfg.block_lr_mult[i],
                      "weight_decay": cfg.weight_decay * cfg.block_wd_mult[i]})
        return g

    # compared at 10 steps: identical to float tolerance; by ~25 steps bmm-vs-mm summation order drifts ~3e-5
    f = _fleet_after([cfg, fl.FleetConfig(arm="pk0120", model_seed=4, data_seed=4, **SMALL)], 10)
    _assert_same(f, 0, _single_after(cfg, 10, groups))


def test_runs_are_isolated():
    base = [fl.FleetConfig(arm="pk0120", model_seed=s, data_seed=s, **SMALL) for s in range(3)]
    changed = list(base)
    changed[2] = fl.FleetConfig(arm="pk0000", model_seed=9, data_seed=7, lr=1e-2, **SMALL)
    f1, f2 = _fleet_after(base, 15), _fleet_after(changed, 15)
    for j in (0, 1):
        for k in f1.params:
            assert torch.equal(f1.params[k][j], f2.params[k][j]), (j, k)


def test_measurement_matches_single_run_measures_and_is_chunk_invariant():
    cfgs = [fl.FleetConfig(arm=a, model_seed=s, data_seed=s, branches=((1, 2), (3, 4)), **SMALL)
            for s, a in enumerate(("pk0120", "pk0000", "pk0123"))]
    f = _fleet_after(cfgs, 30)
    tasks = [fl.build_task(c) for c in cfgs]
    grid = {"a": tasks[0]["a"], "b": tasks[0]["b"], "key": tasks[0]["key"], "y": torch.stack([t["y"] for t in tasks])}
    for s in ("train", "test", "exc", "rule_train"):
        grid[s] = torch.stack([t[s] for t in tasks])
    m_big, m_small = fl.measure(f, grid, chunk=4096), fl.measure(f, grid, chunk=7)
    for k in m_big:
        torch.testing.assert_close(m_big[k], m_small[k], atol=1e-4, rtol=1e-4, msg=k)
    for j, t in enumerate(tasks):
        single = f.single(j)
        pm = gk.path_measure(single, t["tr"], t["te"])
        torch.testing.assert_close(m_big["gen"][j], torch.tensor(pm["gen"]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(m_big["mem"][j], torch.tensor(pm["mem"]), atol=1e-4, rtol=1e-4)
        ref = gk.measure(single, t["tr"], t["te"], paths=False)
        assert abs(float(m_big["test_acc"][j]) - ref["test_acc"]) < 1e-6
        assert abs(float(m_big["train_loss"][j]) - ref["train_loss"]) < 1e-4
        for i in range(4):
            assert abs(float(m_big["abl_block_test_acc"][j, i]) - pm["abl_block"][i]["test_acc"]) < 1e-6
            assert abs(float(m_big["abl_block_gen"][j, i]) - pm["abl_block"][i]["gen"]) < 1e-4
        # branch ablations: blocks {1,2}, {3,4}, all
        a, b, y = t["a"], t["b"], t["y"]
        for q, blk in enumerate(([0, 1], [2, 3], [0, 1, 2, 3])):
            al = single.alphas.clone()
            al[blk] = 0.0
            lg = single(a, b, alphas=al)
            assert abs(float(m_big["abl_branch_test_acc"][j, q]) - gk.acc(lg[t["te"]], y[t["te"]])) < 1e-6
        # test-only rule fraction
        Zc = gk.centered(single(a, b))
        Rp = gk.rule_projection_test(Zc, single.p, t["te"])
        assert abs(float(m_big["rule_frac_test"][j]) - float((Rp.pow(2).sum() / Zc.pow(2).sum()).detach())) < 1e-4


def test_exception_task_and_its_measures():
    c = fl.FleetConfig(arm="pk0000", task="modadd_exc", exc_frac=0.25, model_seed=0, data_seed=5, **SMALL)
    t = fl.build_task(c)
    rule = (t["a"] + t["b"]) % c.p
    assert int(t["exc"].sum()) == round(0.25 * len(t["tr"]))
    assert bool((t["y"][t["exc"]] != rule[t["exc"]]).all()) and bool((t["y"][~t["exc"]] == rule[~t["exc"]]).all())
    assert not bool((t["exc"] & t["test"]).any())
    res = fl.train_fleet([c])
    rec = res[0]["trajectory"][-1]
    assert 0.0 <= rec["exc_acc"] <= 1.0 and len(rec["exc_support"]) == 5


def test_train_fleet_matches_single_trainer_early_and_has_summary():
    cfgs = [fl.FleetConfig(arm=a, model_seed=s, data_seed=s, **SMALL) for s, a in ((0, "pk0120"), (1, "pk0000"))]
    outs = fl.train_fleet(cfgs)
    for c, o in zip(cfgs, outs):
        single = gk.train_run(gk.RunConfig(arm=c.arm, p=c.p, d=c.d, hidden=c.hidden, steps=c.steps,
                                           eval_every=c.eval_every, model_seed=c.model_seed, data_seed=c.data_seed,
                                           lr=c.lr, weight_decay=c.weight_decay, stop_after_grok=0, measure_paths=False))
        assert [r["step"] for r in o["trajectory"]] == [r["step"] for r in single["trajectory"]]
        for rb, rs in zip(o["trajectory"][:3], single["trajectory"][:3]):
            assert rb["train_loss"] == pytest.approx(rs["train_loss"], rel=1e-3, abs=1e-5)
            assert rb["test_acc"] == pytest.approx(rs["test_acc"], abs=1e-6)
        assert {"t_fit", "t_grok", "stable", "n_collapses", "post_fit_min_train_acc"} <= set(o["summary"])


def test_divergent_run_does_not_contaminate_batch():
    small = dict(p=11, d=16, hidden=32, steps=60, eval_every=10, dense_early=False)
    cfgs = [fl.FleetConfig(arm="pk0123", model_seed=0, data_seed=0, lr=3e-3, weight_decay=0.1, **small),
            fl.FleetConfig(arm="pk0123", model_seed=1, data_seed=1, lr=500.0, weight_decay=0.0, **small)]
    outs = fl.train_fleet(cfgs)
    alone = fl.train_fleet(cfgs[:1])
    assert outs[1]["summary"]["stable"] is False
    assert [r["train_loss"] for r in outs[0]["trajectory"]] == [r["train_loss"] for r in alone[0]["trajectory"]]


def test_stop_after_grok_and_struct_key():
    with pytest.raises(ValueError):
        fl.Fleet([fl.FleetConfig(p=13, d=16, hidden=32), fl.FleetConfig(p=17, d=16, hidden=32)])
    # different graphs, lrs and seeds share a batch
    assert fl.struct_key(fl.FleetConfig(arm="pk0000", lr=1e-3)) == fl.struct_key(fl.FleetConfig(arm="pk0123", lr=4e-3))
    assert fl.struct_key(fl.FleetConfig(arm="pk0000")) != fl.struct_key(fl.FleetConfig(arm="pk000000"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU (MPS)")
def test_mps_matches_cpu():
    cfgs = [fl.FleetConfig(arm=a, model_seed=s, data_seed=s, **SMALL) for s, a in ((0, "pk0120"), (1, "pk0000"))]
    cpu, mps = fl.train_fleet(cfgs, "cpu"), fl.train_fleet(cfgs, "mps")
    for c, m in zip(cpu, mps):
        for rc, rm in zip(c["trajectory"][:3], m["trajectory"][:3]):
            assert rm["train_loss"] == pytest.approx(rc["train_loss"], rel=1e-3, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfgs = [fl.FleetConfig(arm=a, model_seed=s, data_seed=s, **SMALL) for s, a in ((0, "pk0120"), (1, "pk0000"))]
    cpu, gpu = fl.train_fleet(cfgs, "cpu"), fl.train_fleet(cfgs, "cuda")
    for c, g in zip(cpu, gpu):
        for rc, rg in zip(c["trajectory"][:3], g["trajectory"][:3]):
            assert rg["train_loss"] == pytest.approx(rc["train_loss"], rel=1e-3, abs=1e-4)
