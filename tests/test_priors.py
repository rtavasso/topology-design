"""Instruments for the "priors" round (experiments/priors/PREREGISTRATION.md): the shrink (ungrokking) protocol of Q1
and the matched-norm random-direction jump of Q2, checked on cases with known answers."""

import os
import sys

import pytest
import torch

from topology_design import fleet as fl
from topology_design import grokking as gk

ROOT = os.path.join(os.path.dirname(__file__), "..")
# the grid and analysis tests need the experiment folders, which the Kaggle upload leaves out
needs_experiments = pytest.mark.skipif(
    not (os.path.exists(os.path.join(ROOT, "experiments", "priors")) and
         os.path.exists(os.path.join(ROOT, "experiments", "walk", "analyze.py"))),
    reason="experiment folders not present (e.g. on Kaggle)")
SMALL = dict(p=11, d=16, hidden=32, eval_every=10, dense_early=False, train_frac=0.5, arm="pk000000")


def test_split_is_nested_so_the_kept_set_is_the_scratch_training_set():
    """gk.split takes a prefix of one permutation, so the shrink subset at fraction f is exactly the training set a
    scratch run at train_frac = f sees (Q1 pairs ungrokking and scratch runs on identical examples)."""
    for seed in (0, 3):
        big = set(gk.split(59, 0.4, seed)[0].tolist())
        for f in (0.05, 0.15, 0.3, 0.35):
            small = set(gk.split(59, f, seed)[0].tolist())
            assert small <= big and len(small) == round(f * 59 * 59)


def test_shrink_bookkeeping_and_stop():
    """grok_thresh 0 makes every eval count as grokked, so the shrink happens at the first eval allowed. Afterwards
    the measured train set is the kept subset, the dropped pairs are measured separately, and the run stops
    shrink_steps later."""
    cfg = fl.FleetConfig(**SMALL, steps=200, grok_thresh=0.0, shrink_frac=0.2, shrink_after_grok=20, shrink_steps=50,
                         lr=1e-3)
    out = fl.train_fleet([cfg])[0]
    sh = out["summary"]["shrink"]
    assert sh["step"] == 20 and sh["n_kept"] == round(0.2 * 121)
    traj = out["trajectory"]
    assert traj[-1]["step"] == 70 and out["summary"]["steps_run"] == 70
    before = [r for r in traj if r["step"] <= 20]
    after = [r for r in traj if r["step"] > 20]
    assert all(r["drop_acc"] is None for r in before)
    assert all(r["drop_acc"] is not None for r in after)


def test_no_shrink_trigger_leaves_training_bit_identical():
    """If the run never groks (threshold above 1), the shrink machinery must not change anything."""
    base = fl.FleetConfig(**SMALL, steps=60, lr=2e-3)
    arm = fl.FleetConfig(**SMALL, steps=60, lr=2e-3, grok_thresh=1.5, shrink_frac=0.2, shrink_steps=50)
    a = fl.train_fleet([base])[0]["trajectory"]
    b = fl.train_fleet([arm])[0]["trajectory"]
    for ra, rb in zip(a, b):
        assert ra["train_loss"] == rb["train_loss"] and ra["test_acc"] == rb["test_acc"]


def test_shrunk_run_trains_on_kept_pairs_only():
    """After a shrink at the first eval, the loss is the mean CE over kept pairs only. Without weight decay and with 12
    kept pairs out of 121 the network memorizes them and does not generalize, so the dropped pairs (trained on for a
    single step) stay unfitted: their loss stays far above the kept pairs'."""
    cfg = fl.FleetConfig(**SMALL, steps=600, grok_thresh=0.0, shrink_frac=0.1, shrink_after_grok=0, shrink_steps=590,
                         lr=1e-2, weight_decay=0.0)
    out = fl.train_fleet([cfg])[0]
    assert out["summary"]["shrink"]["step"] == 0
    last = out["trajectory"][-1]
    assert last["train_acc"] == 1.0
    assert last["drop_acc"] < 0.5 and last["drop_loss"] > 5 * last["train_loss"]


def test_null_jumps_match_norms_and_radial_parts():
    """random: same norm per tensor, new direction. radial: same component along theta and same orthogonal norm,
    new orthogonal direction. rescale: only the component along theta. drift (mode 0): untouched."""
    cfgs = [fl.FleetConfig(**SMALL, steps=10, model_seed=s) for s in range(4)]
    fleet = fl.Fleet(cfgs)
    target = {k: v.detach() * 1.3 + 0.1 * torch.randn_like(v) for k, v in fleet.params.items()}
    seeds = torch.tensor([80_000, 80_001, 80_002, 80_003])
    mode = torch.tensor([1, 2, 3, 0])
    out = fl.null_jump(fleet, target, seeds, mode)
    for k, v in fleet.params.items():
        x = v.detach().flatten(1)
        dd = (target[k] - v.detach()).flatten(1)
        do = (out[k] - v.detach()).flatten(1)
        u = x / x.norm(dim=1, keepdim=True)
        rad_d, rad_o = (dd * u).sum(1), (do * u).sum(1)
        # random: same total norm, direction far from the drift's
        assert torch.allclose(do[0].norm(), dd[0].norm(), rtol=1e-5)
        assert abs(float(do[0] @ dd[0] / (do[0].norm() * dd[0].norm()))) < 0.5
        # radial: same radial component and same orthogonal norm, different orthogonal direction
        assert torch.allclose(rad_o[1], rad_d[1], rtol=1e-4, atol=1e-6)
        oo, od = do[1] - rad_o[1] * u[1], dd[1] - rad_d[1] * u[1]
        assert torch.allclose(oo.norm(), od.norm(), rtol=1e-4)
        assert abs(float(oo @ od / (oo.norm() * od.norm()))) < 0.5
        # rescale: only the radial part
        assert torch.allclose(do[2], rad_d[2] * u[2], rtol=1e-4, atol=1e-6)
        assert torch.equal(out[k][3], target[k][3])
    again = fl.null_jump(fleet, target, seeds, mode)
    assert all(torch.equal(again[k], out[k]) for k in out)


def test_null_jumps_run_end_to_end():
    kw = dict(**SMALL, steps=60, lr=2e-3, measure_horizon=True, measure_refit=False, drift_jump_at=30, drift_jump_h=200)
    dirs = ("drift", "random", "radial", "rescale")
    d = fl.train_fleet([fl.FleetConfig(**kw, drift_jump_dir=x) for x in dirs])
    assert [r["summary"]["jump"]["dir"] for r in d] == list(dirs)
    assert len({r["summary"]["jump"]["test_acc_before"] for r in d}) == 1


def test_shrink_by_stops_runs_that_never_grok():
    cfg = fl.FleetConfig(**SMALL, steps=200, grok_thresh=1.5, shrink_frac=0.2, shrink_steps=50, shrink_by=80)
    out = fl.train_fleet([cfg])[0]
    assert out["summary"]["shrink"]["step"] is None and out["summary"]["steps_run"] == 80


def test_dense_until_extends_dense_evals():
    c = fl.FleetConfig(**{**SMALL, "dense_early": True}, steps=10, dense_until=3000)
    assert fl.is_eval_step(2025, c) and not fl.is_eval_step(3025, c) and fl.is_eval_step(3100, c)


@pytest.mark.parametrize("bad", [dict(shrink_frac=0.6), dict(shrink_frac=0.2, shrink_steps=0),
                                 dict(drift_jump_dir="up"), dict(shrink_frac=0.2, shrink_steps=10, stop_after_grok=5)])
def test_validation(bad):
    with pytest.raises(ValueError):
        fl.validate_config(fl.FleetConfig(**{**SMALL, "steps": 10, **bad}))


@needs_experiments
def test_priors_grids_expand():
    sys.path.insert(0, os.path.join(ROOT, "experiments", "fleet"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("priors_make_grids",
                                                  os.path.join(ROOT, "experiments", "priors", "make_grids.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    from run_fleet import expand
    for name, fn in mg.GRID_FNS.items():
        cfgs = expand(fn())
        assert len(cfgs) == mg.EXPECTED[name], name
        for c in cfgs:
            fl.validate_config(c)


def _load_analyze():
    import importlib.util
    spec = importlib.util.spec_from_file_location("priors_analyze", os.path.join(ROOT, "experiments", "priors", "analyze.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_ungrok(graph, width, frac, seed, final, shrink_at=5000, steps=30000):
    traj = [{"step": t, "test_acc": 1.0, "train_acc": 1.0, "drop_acc": None, "weight_norms": [1.0]}
            for t in range(0, shrink_at + 1, 100)]
    traj += [{"step": t, "test_acc": final, "train_acc": 1.0, "drop_acc": final, "weight_norms": [1.0]}
             for t in range(shrink_at + 100, shrink_at + steps + 1, 100)]
    return {"run_id": f"u{graph}{width}{frac}{seed}", "config": {"tag": f"pQ1u-{graph}-{width}", "shrink_frac": frac,
            "model_seed": seed, "shrink_steps": steps},
            "summary": {"shrink": {"step": shrink_at}}, "trajectory": traj}


def _fake_scratch(graph, width, frac, lr, seed, grokked):
    traj = [{"step": t, "test_acc": 1.0 if grokked and t >= 1000 else 0.0} for t in range(0, 3001, 100)]
    return {"run_id": f"s{graph}{width}{frac}{lr}{seed}", "config": {"tag": f"pQ1s-{graph}-{width}", "lr": lr,
            "train_frac": frac, "steps": 60000}, "summary": {"grokked": grokked}, "trajectory": traj}


@needs_experiments
def test_q1_analysis_recovers_planted_thresholds(tmp_path):
    """Planted case: sym base keeps the rule from 0.15 up (n_crit = 522) and groks from scratch from 0.3 up at the
    best lr (n_floor = 1,044), so S = 2 and the search share is log 2 / log(1044 / 3)."""
    import json
    import math
    an = _load_analyze()
    d = tmp_path / "res"
    d.mkdir()
    runs = []
    for f in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        for s in range(4):
            runs.append(_fake_ungrok("sym", "base", f, s, 1.0 if f >= 0.15 else 0.02))
    for f in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        for lr in (0.001, 0.002, 0.004):
            for s in range(3):
                g = (f >= 0.3 and lr == 0.002) or (f >= 0.35)
                runs.append(_fake_scratch("sym", "base", f, lr, s, g))
    for r in runs:
        (d / (r["run_id"] + ".json")).write_text(json.dumps(r))
    lines, js = an.q1([str(d)])
    dec = js["decomposition"]["sym|base"]
    assert dec["n_crit"] == 522 and dec["n_floor"] == 1044 and dec["floor_lr"] == 0.002
    assert abs(dec["S"] - 1044 / 522) < 1e-9
    assert abs(dec["share"] - math.log(2) / math.log(1044 / 3)) < 1e-9
    assert js["P1"] and js["P2"] == "SUPPORTED"
    assert dec["flags"] == ""


def _fake_q2(tag, seed, t_grok, at=None, h=None, dirn=None):
    traj = [{"step": t, "test_acc": 1.0 if t >= t_grok else 0.0} for t in range(0, 15001, 100)]
    c = {"tag": tag, "model_seed": seed, "steps": 15000}
    if at is not None:
        c.update({"drift_jump_at": at, "drift_jump_h": h, "drift_jump_dir": dirn})
    return {"run_id": f"{tag}{seed}{at}{h}{dirn}", "config": c, "summary": {}, "trajectory": traj}


@needs_experiments
def test_q2_analysis_planted(tmp_path):
    """Planted: drift jumps save 400 steps and null jumps save nothing, in every cell and seed -> P5 supported; the
    reverse plant (nulls better) -> falsified."""
    import json
    an = _load_analyze()
    for drift_gain, expect in ((400, True), (0, False)):
        d = tmp_path / f"res{drift_gain}"
        d.mkdir()
        runs = []
        for s in range(10):
            runs.append(_fake_q2("pQ2base", s, 3000))
            for at in (1000, 2000):
                for h in (250, 1000):
                    runs.append(_fake_q2("pQ2jump", s, 3000 - drift_gain, at, h, "drift"))
                    for null in ("radial", "random", "rescale"):
                        runs.append(_fake_q2("pQ2jump", s, 3000 - (0 if drift_gain else 400), at, h, null))
        for r in runs:
            (d / (r["run_id"] + ".json")).write_text(json.dumps(r))
        _, js = an.q2([str(d)])
        assert (js["P5"] == "SUPPORTED") is expect
        if not expect:
            assert js["P5"] == "FALSIFIED"


@needs_experiments
def test_q1_width_order_respects_censoring_and_flags_edge_lr(tmp_path):
    """narrow never keeps the rule (censored above), base keeps it from 522 pairs, wide from 348: the order is C
    even with censoring. Scratch groks only at the top learning rate -> best_lr_at_edge flag."""
    import json
    an = _load_analyze()
    d = tmp_path / "res"
    d.mkdir()
    start = {"narrow": 9.0, "base": 0.15, "wide": 0.1}
    runs = []
    for w, f0 in start.items():
        for f in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
            for s in range(4):
                runs.append(_fake_ungrok("sym", w, f, s, 1.0 if f >= f0 else 0.02))
        for f in (0.1, 0.2, 0.3, 0.4):
            for lr in (0.002, 0.004, 0.008):
                for s in range(3):
                    runs.append(_fake_scratch("sym", w, f, lr, s, f >= 0.3 and lr == 0.008))
    for r in runs:
        (d / (r["run_id"] + ".json")).write_text(json.dumps(r))
    _, js = an.q1([str(d)])
    assert js["P4_order"].startswith("C")
    assert js["flags"]["sym|base"]["best_lr_at_edge"]
    assert "best_lr_at_edge" in js["decomposition"]["sym|base"]["flags"]
