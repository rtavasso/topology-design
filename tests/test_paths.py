"""Study 4 (experiments/paths): the preregistered lr-selection rule and the analysis pipeline."""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sel = _load("paths_select", "experiments/paths/select_lr.py")
an = _load("paths_analyze", "experiments/paths/analyze.py")


def _r(t_grok, stable, test=0.5):
    return {"t_grok": t_grok, "stable": stable, "final_test_acc": test}


def test_selection_prefers_stable_fastest_then_unstable_then_best_test():
    cells = {0.001: [_r(9000, True)] * 3, 0.002: [_r(5000, True), _r(6000, True), _r(None, False)],
             0.004: [_r(2000, False)] * 3}
    c = sel.choose(cells)
    assert c["lr"] == 0.002 and c["tier"] == "stable"  # 0.004 is faster but unstable
    c = sel.choose({0.001: [_r(None, True)] * 3, 0.004: [_r(3000, False)] * 2 + [_r(None, False)]})
    assert c["lr"] == 0.004 and c["tier"] == "unstable-at-best"
    c = sel.choose({0.001: [_r(None, True, 0.4)] * 3, 0.008: [_r(None, False, 0.7)] * 3})
    assert c["lr"] == 0.008 and c["tier"] == "no-grok" and c["grid_edge"]
    c = sel.choose({0.001: [_r(4000, True)] * 3, 0.002: [_r(4000, True)] * 3})
    assert c["lr"] == 0.001  # tie -> smaller lr


def test_holm_and_kendall():
    h = an.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
    assert an.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert an.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_mask():
    assert an.mask([2, 3]) == "0110" and an.mask([1]) == "1000"


@pytest.mark.slow
def test_pipeline_end_to_end_on_tiny_runs(tmp_path):
    """Tiny Stage 0 -> selection -> Stage 1 -> analysis, through the real sweep runner."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from topology_design import grokking as gk

    small = dict(p=11, d=16, hidden=32, steps=300, eval_every=50, train_frac=0.5)
    s0 = tmp_path / "stage0"
    s0.mkdir()
    for arm in an.ALL:
        for lr in (0.003, 0.01):
            for seed in (400, 401):
                cfg = gk.RunConfig(arm=arm, lr=lr, model_seed=seed, data_seed=seed, stop_after_grok=50,
                                   measure_paths=False, **small)
                gk.save(gk.train_run(cfg), str(s0 / (cfg.run_id + ".json")))
    sys.argv = ["select", "--runs", str(s0), "--out", str(tmp_path)]
    sel.main()
    choice = json.load(open(tmp_path / "selection.json"))
    assert set(choice) == set(an.ALL)
    s1 = tmp_path / "stage1"
    s1.mkdir()
    for arm, v in choice.items():
        for seed in (500, 501, 502):
            cfg = gk.RunConfig(arm=arm, lr=v["lr"], model_seed=seed, data_seed=seed, stop_after_grok=50, **small)
            gk.save(gk.train_run(cfg), str(s1 / (cfg.run_id + ".json")))
    sys.argv = ["analyze", "--runs", str(s1), "--selection", str(tmp_path / "selection.json"), "--out",
                str(tmp_path / "report")]
    an.main()
    out = json.load(open(tmp_path / "report" / "compact.json"))
    assert {"H1", "H2", "H3", "holm"} <= set(out["results"]) and len(out["runs"]) == 18
    assert os.path.exists(tmp_path / "report" / "summary.md")
