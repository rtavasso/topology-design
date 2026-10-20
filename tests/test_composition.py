"""Tests for the Study-3 composition testbed (src/topology_design/composition.py)."""

import torch

from topology_design import composition as cp
from topology_design import grokking as gk


def test_task_definitions():
    a, b, c, s, y = cp.full_grid(7, "addmul")
    assert len(y) == 7 ** 3
    i = 3 * 49 + 5 * 7 + 4  # a=3, b=5, c=4
    assert (a[i], b[i], c[i], s[i], y[i]) == (3, 5, 4, (3 + 5) % 7, ((3 + 5) % 7) * 4 % 7)
    _, _, _, _, y3 = cp.full_grid(7, "add3")
    assert y3[i] == (3 + 5 + 4) % 7


def test_projections_identify_rule_and_intermediate():
    p = 7
    a, b, c, s, y = cp.full_grid(p, "addmul")
    g = torch.Generator().manual_seed(0)
    f_y = torch.randn(p, p, generator=g)[y]  # function of the target only
    f_sc = torch.randn(p * p, p, generator=g)[s * p + c]  # function of (s, c)
    torch.testing.assert_close(cp.group_projection(f_y, y, p), f_y)
    torch.testing.assert_close(cp.group_projection(f_sc, s * p + c, p * p), f_sc)
    # a function of y is also a function of (s, c), but not conversely
    torch.testing.assert_close(cp.group_projection(f_y, s * p + c, p * p), f_y)
    assert (cp.group_projection(f_sc, y, p) - f_sc).abs().max() > 0.1
    Z = torch.randn(p ** 3, p, generator=g)
    P = cp.group_projection(Z, y, p)
    torch.testing.assert_close(cp.group_projection(P, y, p), P)  # idempotent


def test_branch_decomposition_exact_and_identical_init():
    cfgs = [cp.CompConfig(arm=a, p=5, d=16, hidden=32, beta=0.8) for a in ("pk012", "pk000", "pk010")]
    models = [cp.build_model(c) for c in cfgs]
    a, b, c, s, y = cp.full_grid(5, "addmul")
    for m in models:
        lg, parts, _ = m(a, b, c, return_parts=True)
        torch.testing.assert_close(parts.sum(0), lg, atol=1e-5, rtol=1e-5)
    for m in models[1:]:
        assert all(torch.equal(p0, p1) for p0, p1 in zip(models[0].parameters(), m.parameters()))


def test_ridge_decoder_recovers_planted_feature():
    p = 7
    a, b, c, s, y = cp.full_grid(p, "addmul")
    H = torch.nn.functional.one_hot(s, p).float() + 0.01 * torch.randn(len(s), p)
    tr, te = cp.split(p, 0.5, 0)
    assert cp.ridge_decode_acc(H, s, p, tr, te) > 0.99
    H_noise = torch.randn(len(s), p)
    assert cp.ridge_decode_acc(H_noise, s, p, tr, te) < 0.4


def test_smoke_training_and_summary_keys():
    cfg = cp.CompConfig(arm="pk012", p=5, d=16, hidden=32, steps=40, eval_every=20, stop_after_grok=0)
    res = cp.train_run(cfg)
    rec = res["trajectory"][-1]
    assert rec["train_loss"] < res["trajectory"][0]["train_loss"]
    assert len(rec["decode_s"]) == 4 and len(rec["branches"]) == 4 and len(rec["ablate"]) == 3
    assert 0 <= rec["inter_frac"] <= 1 and rec["rule_frac"] <= rec["inter_frac"] + 1e-6
    assert {"stable", "grok_delay", "final_decode_s", "peak_inter_frac"} <= set(res["summary"])
