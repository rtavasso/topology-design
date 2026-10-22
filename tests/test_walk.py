"""Instruments for "diagnose the walk" (experiments/walk/PREREGISTRATION.md), each checked on a case with a known
answer: prequential accounting, closed-form readout refit, drift extrapolation, readout swap, the program-prior
reference learner, and the grids."""

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from topology_design import fleet as fl
from topology_design import fleet_horizon as fh
from topology_design import program_prior as pp

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "walk"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "fleet"))

SMALL = dict(p=11, d=16, hidden=32, steps=30, eval_every=10, dense_early=False, train_frac=0.5)
BR = ((1, 2, 3), (4, 5, 6))


def _grid(cfgs, dev="cpu"):
    t = [fl.build_task(c) for c in cfgs]
    g = {"a": t[0]["a"], "b": t[0]["b"], "key": t[0]["key"], "n_key": torch.bincount(t[0]["key"], minlength=cfgs[0].p),
         "y": torch.stack([x["y"] for x in t])}
    for s in ("train", "test", "exc", "rule_train"):
        g[s] = torch.stack([x[s] for x in t])
    return g, t


def test_prequential_equals_initial_model_ce_when_frozen():
    """lr = 0: the model never changes, so every pair's surprise at arrival is its CE under the initial model."""
    cfg = fl.FleetConfig(arm="pk012000", branches=BR, lr=0.0, arrive_start=10, arrive_n=7, arrive_every=3,
                         **{**SMALL, "steps": 40})
    out = fl.train_fleet([cfg])[0]
    pq = out["summary"]["prequential"]
    t = fl.build_task(cfg)
    n = len(t["tr"])
    arr, inc, K = fl._arrivals([cfg], n, "cpu")
    assert sum(pq["n"]) == n and len(pq["nats"]) == K + 1
    m = fl.Fleet([cfg]).single(0)
    with torch.no_grad():
        ce = F.cross_entropy(m(t["a"][t["tr"]], t["b"][t["tr"]]), t["y"][t["tr"]], reduction="none")
    for k in range(K + 1):
        sel = inc[0] == k
        assert abs(pq["nats"][k] - float(ce[sel].sum())) < 1e-3 * max(1.0, float(ce[sel].sum()))
        assert int(pq["n"][k]) == int(sel.sum())


class _P:
    def __init__(self, p):
        self.p = p


def test_refit_recovers_a_linear_rule_and_not_a_lookup_table():
    p, N = 11, 121
    a, b = torch.arange(p).repeat_interleave(p), torch.arange(p).repeat(p)
    y = ((a + b) % p).unsqueeze(0)
    train = (torch.rand(1, N, generator=torch.Generator().manual_seed(0)) < 0.5)
    grid = {"y": y, "train": train, "test": ~train}
    val = fh.holdout_mask([fl.FleetConfig(data_seed=0)], N, "cpu")
    rule = F.one_hot(y, p).float() + 0.05 * torch.randn(1, N, p, generator=torch.Generator().manual_seed(1))
    r = fh.refit_readout(_P(p), grid, val, X=torch.cat([rule, torch.randn(1, N, 5)], -1))
    assert float(r["refit_test_acc"][0]) > 0.95
    lookup = torch.randn(1, N, 64, generator=torch.Generator().manual_seed(2))  # an independent code per pair
    r = fh.refit_readout(_P(p), grid, val, X=lookup)
    assert float(r["refit_train_acc"][0]) > 0.9 and float(r["refit_test_acc"][0]) < 0.3


def test_refit_matches_numpy_ridge_on_a_network():
    cfg = fl.FleetConfig(arm="pk012000", branches=BR, **SMALL)
    f = fl.Fleet([cfg])
    grid, _ = _grid([cfg])
    val = fh.holdout_mask([cfg], grid["a"].shape[0], "cpu")
    r = fh.refit_readout(f, grid, val)
    r = {**r, "W": r["ridge_W"], "refit_lam_c": r["ridge_lam_c"]}
    X = r["X"][0].double().numpy()
    tr = grid["train"][0].numpy()
    T = F.one_hot(grid["y"][0], cfg.p).double().numpy() - 1.0 / cfg.p
    G = X[tr].T @ X[tr]
    lam = float(r["refit_lam_c"][0]) * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), X[tr].T @ T[tr])
    np.testing.assert_allclose(r["W"][0].double().numpy(), W, rtol=1e-3, atol=1e-5)


def test_logistic_refit_recovers_a_realisable_linear_readout():
    """Labels produced by a linear readout of the features: the refit must find a readout that generalizes."""
    p, N, d = 11, 2000, 24
    g = torch.Generator().manual_seed(3)
    X = torch.randn(1, N, d, generator=g)
    y = torch.einsum("rnd,dp->rnp", X, torch.randn(d, p, generator=g)).argmax(-1)
    train = torch.rand(1, N, generator=g) < 0.6
    grid = {"y": y, "train": train, "test": ~train}
    val = fh.holdout_mask([fl.FleetConfig(data_seed=0)], N, "cpu")
    r = fh.refit_readout(_P(p), grid, val, X=X)
    assert float(r["refit_test_acc"][0]) > 0.85
    assert float(r["refit_test_acc"][0]) >= float(r["ridge_test_acc"][0]) - 0.05


def test_extrapolation_zero_horizon_and_exact_jump():
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, model_seed=s, data_seed=0, **SMALL) for s in (0, 1)]
    f = fl.Fleet(cfgs)
    grid, _ = _grid(cfgs)
    vel = {k: torch.zeros_like(v) for k, v in f.params.items()}
    cur = fh.test_acc(f, grid)
    out = fh.extrapolated_acc(f, grid, vel, 10, (0, 100))
    assert torch.equal(out["extrap_acc_0"], cur) and torch.equal(out["extrap_acc_100"], cur)
    # drift pointing exactly from run 0's params to run 1's: a jump of H lands on run 1
    H, t = 50, 10
    corr = 1 - fh.DRIFT_BETA ** t
    vel = {k: torch.stack([(v[1] - v[0]).detach(), torch.zeros_like(v[1])]) * corr / H for k, v in f.params.items()}
    out = fh.extrapolated_acc(f, grid, vel, t, (H,))
    assert abs(float(out[f"extrap_acc_{H}"][0]) - float(cur[1])) < 1e-6
    for k, v in f.params.items():  # parameters restored
        assert v.requires_grad


def test_swap_replaces_only_chosen_runs_and_resets_their_moments():
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, model_seed=s, data_seed=s, **SMALL) for s in (0, 1)]
    f = fl.Fleet(cfgs)
    opt = fl.FleetAdamW(f)
    grid, t = _grid(cfgs)
    tr = torch.stack([x["tr"] for x in t])
    A, B, Y = grid["a"][tr], grid["b"][tr], grid["y"].gather(1, tr)
    act = torch.ones(2, dtype=torch.bool)
    for _ in range(3):
        opt.zero_grad()
        fl._loss_and_grads(f, opt, A, B, Y, None, act, {})
        opt.step(act)
    U0 = f.params["unembed"].detach().clone()
    m1 = opt.m["unembed"][1].clone()
    val = fh.holdout_mask(cfgs, grid["a"].shape[0], "cpu")
    runs = torch.tensor([True, False])
    sw = fh.swap_readout(f, opt, grid, val, runs, A, B, Y, None)
    assert torch.equal(f.params["unembed"][1], U0[1]) and not torch.equal(f.params["unembed"][0], U0[0])
    assert float(opt.m["unembed"][0].abs().max()) == 0 and torch.equal(opt.m["unembed"][1], m1)
    with torch.no_grad():
        lg = f.forward(A, B)
        ce = F.cross_entropy(lg[0], Y[0])
    assert abs(float(ce) - float(sw["train_ce"][0])) < 1e-4
    assert f.params["unembed"].requires_grad


def test_program_prior_finds_the_rule_and_falls_back_to_uniform():
    p = 13
    T, logm, ex = pp.enumerate_functions(p, 5)
    a, b = np.repeat(np.arange(p), p), np.tile(np.arange(p), p)
    y = (a + b) % p
    assert any((T == y).all(1))
    order = np.random.default_rng(0).permutation(p * p)
    xb = [order[i:i + 1] for i in range(40)]
    r = pp.prequential(T, logm, xb, [y[x] for x in xb], p, true_fn=y)
    k_star = next(i + 1 for i, v in enumerate(r["post_true"]) if v >= 0.99)
    assert k_star <= 10 and sum(r["nats"][20:]) < 0.1
    rnd = np.random.default_rng(1).integers(0, p, p * p)  # labels with no rule
    r = pp.prequential(T, logm, xb, [rnd[x] for x in xb], p)
    uni = 40 * math.log(p)
    assert 0.95 * uni < sum(r["nats"]) < uni + 3.0


def _walk_grids():
    import importlib.util
    spec = importlib.util.spec_from_file_location("walk_make_grids", os.path.join(ROOT, "experiments", "walk", "make_grids.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_block_depths_and_grid_sizes():
    mg = _walk_grids()
    from run_fleet import expand
    assert mg.block_depths("pk0123") == [1, 2, 3, 4] and mg.block_depths("pk0120") == [1, 2, 3, 1]
    assert mg.block_depths("pk012000") == [1, 2, 3, 1, 1, 1]
    sizes = {"e1": 60, "e2": 120, "e3": 100, "e4": 648}
    for name, n in sizes.items():
        cfgs = expand(getattr(mg, name)())
        assert len(cfgs) == n and len({c.run_id for c in cfgs}) == n, name
    e3 = expand(mg.e3())
    c = next(c for c in e3 if c.tag == "wE3-sym-rho4")
    assert math.isclose(c.branch_lr_mult[0] / c.branch_lr_mult[1], 4.0)
    assert math.isclose(c.branch_lr_mult[0] * c.branch_wd_mult[0], 1.0)  # lr*wd fixed per branch
    e2 = expand(mg.e2())
    assert len({fl.struct_key(c) for c in e2}) == 1  # interventions batch with their paired base runs


def test_end_to_end_horizon_swap_and_stream():
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, measure_horizon=True, horizon_extrap=(10, 40),
                           readout_swap_at=sw, arrive_start=20, arrive_n=10, arrive_every=5, model_seed=0, data_seed=0,
                           **SMALL) for sw in (None, 12)]
    out = fl.train_fleet(cfgs)
    rec = out[0]["trajectory"][-1]
    for k in ("refit_test_acc", "refit_train_acc", "refit_lam", "ridge_test_acc", "extrap_acc_10", "extrap_acc_40"):
        assert k in rec and rec[k] is not None
    assert "swap" not in out[0]["summary"] and out[1]["summary"]["swap"]["step"] == 12
    assert out[1]["summary"]["swap"]["refit_test_acc"] is not None
    assert len(out[0]["summary"]["prequential"]["nats"]) == len(out[0]["summary"]["latency"])


def test_drift_jump_moves_only_chosen_runs_by_h_times_drift():
    base = dict(arm="pk012000", branches=BR, measure_horizon=True, horizon_extrap=(10,), model_seed=0, data_seed=0,
                **{**SMALL, "steps": 12, "eval_every": 4})
    a = fl.train_fleet([fl.FleetConfig(**base), fl.FleetConfig(**base, drift_jump_at=8, drift_jump_h=0)])
    # H = 0: the jump is a no-op, so both runs are identical
    assert [r["train_loss"] for r in a[0]["trajectory"]] == [r["train_loss"] for r in a[1]["trajectory"]]
    assert a[1]["summary"]["jump"]["test_acc_before"] == a[1]["summary"]["jump"]["test_acc_after"]
    b = fl.train_fleet([fl.FleetConfig(**base), fl.FleetConfig(**base, drift_jump_at=8, drift_jump_h=500)])
    assert b[1]["summary"]["jump"]["step"] == 8 and b[1]["summary"]["final_train_acc"] is not None
    assert b[0]["trajectory"][-1]["train_loss"] == a[0]["trajectory"][-1]["train_loss"]  # unjumped run unaffected
    assert b[1]["trajectory"][-1]["train_loss"] != a[1]["trajectory"][-1]["train_loss"]
