"""Phase 5 grids expand to the arms and seed counts of experiments/fastslow/DESIGN.md, and branch_rms is right."""

import os
import sys

import torch

from topology_design import fleet as fl

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "fastslow"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "fleet"))

import make_grids as mg  # noqa: E402
from run_fleet import expand  # noqa: E402


def _count(cfgs, tag):
    return sum(1 for c in cfgs if c.tag == tag)


def test_pilot_grid():
    cfgs = expand(mg.pilot())
    assert len(cfgs) == 36 and len({c.run_id for c in cfgs}) == 36
    assert all(c.steps == 40000 and c.stop_after_grok == 0 and c.branches == ((1, 2, 3), (4, 5, 6)) for c in cfgs)
    assert _count(cfgs, "p5pilot-mem4") == 9 and all(c.branch_wd_mult == (0.5, 2.0) for c in cfgs if c.tag == "p5pilot-mem4")
    assert len({fl.struct_key(c) for c in cfgs}) == 1  # one batch on the GPU


def test_stage1_grids_match_design():
    a = expand(mg.stage1_5A(0.002))
    assert len(a) == 50 and all(c.measure_signal and c.lr == 0.002 and c.model_seed == c.data_seed for c in a)
    b = expand(mg.stage1_5B(0.002, [1.1, 0.9], [0.8, 1.2]))
    assert len(b) == 180 and len({c.run_id for c in b}) == 180
    for arm, n in (("base", 20), ("mem4", 20), ("mem14", 20), ("mem4s", 20), ("mem14s", 20), ("speed4", 10),
                   ("speed14", 10), ("unmaskedL", 10)):
        assert _count(b, f"p5B-asym-{arm}") == n, arm
    for arm in ("base", "mem4", "mem14", "speed4", "speed14"):
        assert _count(b, f"p5B-sym-{arm}") == 10, arm
    s4 = next(c for c in b if c.tag == "p5B-asym-speed4")
    assert s4.branch_lr_mult == (2.0, 0.5) and s4.branch_wd_mult == (0.5, 2.0)  # equal tau = 1/(lr wd)
    assert next(c for c in b if c.tag == "p5B-asym-mem4s").branch_alpha == (1.1, 0.9)
    assert all(c.steps == 50000 and c.stop_after_grok == 0 and c.measure_path_eff for c in b)


def test_branch_rms_is_rms_of_summed_centered_branch_logits():
    cfg = fl.FleetConfig(arm="pk012000", branches=((1, 2, 3), (4, 5, 6)), p=11, d=16, hidden=32, train_frac=0.5)
    f = fl.Fleet([cfg])
    t = fl.build_task(cfg)
    grid = {"a": t["a"], "b": t["b"], "key": t["key"], "n_key": torch.bincount(t["key"], minlength=11),
            "y": t["y"].unsqueeze(0)}
    for s in ("train", "test", "exc", "rule_train"):
        grid[s] = t[s].unsqueeze(0)
    out = fl.measure(f, grid, chunk=17)
    with torch.no_grad():
        _, parts = f.single(0)(t["a"], t["b"], return_parts=True)
    for q, idx in enumerate(((1, 2, 3), (4, 5, 6))):
        z = sum(parts[i] for i in idx)
        z = z - z.mean(-1, keepdim=True)
        assert abs(float(out["branch_rms"][0, q]) - float(z.pow(2).mean().sqrt())) < 1e-5
