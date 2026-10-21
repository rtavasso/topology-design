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


def _r(t_grok, stable, test=0.5, pfm=None):
    return {"t_grok": t_grok, "stable": stable, "final_test_acc": test,
            "post_fit_min": (1.0 if stable else 0.5) if pfm is None else pfm}


def test_selection_prefers_stable_fastest_then_unstable_then_best_test():
    cells = {0.001: [_r(9000, True)] * 3, 0.002: [_r(5000, True), _r(6000, True), _r(None, False)],
             0.004: [_r(2000, False)] * 3}
    c = sel.choose(cells)
    assert c["lr"] == 0.002 and c["tier"] == "stable"  # 0.004 is faster but unstable
    c = sel.choose({0.001: [_r(None, True)] * 3, 0.004: [_r(3000, False)] * 2 + [_r(None, False)]})
    assert c["lr"] == 0.004 and c["tier"] == "unstable-at-best"
    # unstable tier ranks by the continuous stability endpoint before speed
    c = sel.choose({0.002: [_r(8000, False, pfm=0.8)] * 3, 0.008: [_r(2000, False, pfm=0.3)] * 3})
    assert c["lr"] == 0.002 and c["tier"] == "unstable-at-best"
    c = sel.choose({0.001: [_r(None, True, 0.4)] * 3, 0.008: [_r(None, False, 0.7)] * 3})
    assert c["lr"] == 0.008 and c["tier"] == "no-grok" and not c["grid_edge"]
    c = sel.choose({0.016: [_r(None, False, 0.7)] * 3, 0.001: [_r(None, True, 0.4)] * 3})
    assert c["grid_edge"]
    c = sel.choose({0.001: [_r(4000, True)] * 3, 0.002: [_r(4000, True)] * 3})
    assert c["lr"] == 0.001  # tie -> smaller lr


def test_holm_and_kendall():
    h = an.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
    assert an.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert an.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def _fake_run(arm, seed, gen_blocks, mem_blocks, abl_gen=None, grok=True, t_grok=500):
    """A run whose every eval has the given direct gen/mem per block (embedding component 0), peaked mem at step 100."""
    import topology_design.grokking as gk
    parents, _ = gk.arm_spec(arm)
    traj = []
    for step, scale in ((0, 0.1), (100, 1.0), (200, 0.5), (300, 0.5), (400, 0.5), (500, 0.5), (600, 0.5)):
        mem = [0.0] + [m * scale for m in mem_blocks]
        gen = [0.0] + list(gen_blocks)
        sets = {an.mask([i + 1 for i in range(4) if mk >> i & 1]): {"gen": (abl_gen or {}).get(mk, sum(gen) / 2),
                                                                     "mem": 0.0, "train_acc": 1.0, "test_acc": 0.5}
                for mk in range(1, 16)}
        traj.append({"step": step, "train_acc": 1.0, "test_acc": 0.99 if step >= t_grok else 0.5, "train_loss": 0.01,
                     "rule_frac": 0.5, "excluded_train_acc": 0.1,
                     "paths": {"gen": gen, "mem": mem, "gen_total": sum(gen), "mem_total": sum(mem),
                               "gen_tshap": gen[1:], "mem_tshap": mem[1:], "abl_sets": sets}})
    return {"arm": arm, "seed": seed, "lr": 1e-3, "wd": 1.0, "depth": gk.block_depths(parents),
            "C": gk.longest_chain(parents), "Q": gk.root_readers(parents), "traj": traj,
            "summary": {"t_grok": t_grok if grok else None, "t_fit": 100}}


def test_h1_detects_short_paths_hosting_gen_and_long_paths_hosting_mem():
    runs = []
    for seed in range(6):
        for g in an.MIXED:
            short = [j for j, d in enumerate(_fake_run(g, 0, [1] * 4, [1] * 4)["depth"]) if d == 1]
            gen = [2.0 if j in short else 0.5 for j in range(4)]  # short blocks carry most gen
            mem = [0.2 if j in short else 1.0 for j in range(4)]  # long blocks carry most mem
            runs.append(_fake_run(g, seed, gen, mem))
    res = an.h1_both(runs, an.MIXED)
    assert res["ci_both_above_0"] and res["direct"]["pooled"]["mean"] > 0.3
    # swapping roles flips the sign
    runs2 = [dict(r, traj=[dict(x, paths=dict(x["paths"], gen=x["paths"]["mem"], mem=x["paths"]["gen"],
                                               gen_tshap=x["paths"]["mem"][1:], mem_tshap=x["paths"]["gen"][1:]))
                           for x in r["traj"]]) for r in runs]
    assert an.h1_both(runs2, an.MIXED)["direct"]["pooled"]["mean"] < 0
    # non-grokked runs are excluded and counted
    runs3 = runs + [_fake_run("pk0100", 99, [1, 1, 1, 1], [1, 1, 1, 1], grok=False)]
    assert an.h1(runs3, an.MIXED, "direct")["not_grokked"]["pk0100"] == 1


def test_h2_pairs_by_seed_and_uses_the_right_ablation():
    runs = []
    for seed in range(5):
        for arm, u in (("pk0120", 0.2), ("pk0121", 0.6), ("pk0100", 0.3), ("pk0110", 0.5)):
            blocks = [2, 3] if arm in ("pk0120", "pk0121") else [2]
            mk = sum(1 << (b - 1) for b in blocks)
            gen = [1.0, 1.0, 1.0, 1.0]
            runs.append(_fake_run(arm, seed, gen, [0.1] * 4, abl_gen={mk: 4.0 * (1 - u)}))
    res = an.h2(runs)
    assert res["H2a"]["mean"] == pytest.approx(0.2 - 0.6) and res["H2b"]["mean"] == pytest.approx(0.3 - 0.5)
    assert res["ci_both_below_0"] and res["H2a"]["n"] == 5


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
        for lr in (0.003, 0.03):
            for seed in (400, 401):
                cfg = gk.RunConfig(arm=arm, lr=lr, model_seed=seed, data_seed=seed, stop_after_grok=50,
                                   dense_early=True, **small)
                gk.save(gk.train_run(cfg), str(s0 / (cfg.run_id + ".json")))
    sys.argv = ["select", "--runs", str(s0), "--out", str(tmp_path), "--expect", "999"]
    with pytest.raises(SystemExit):  # refuses to select on incomplete Stage 0
        sel.main()
    sys.argv = ["select", "--runs", str(s0), "--out", str(tmp_path), "--expect", str(len(an.ALL) * 2 * 2)]
    sel.main()
    choice = json.load(open(tmp_path / "selection.json"))
    assert set(choice) == set(an.ALL)
    s1 = tmp_path / "stage1"
    s1.mkdir()
    for arm, v in choice.items():
        for seed in (500, 501, 502):
            cfg = gk.RunConfig(arm=arm, lr=v["lr"], model_seed=seed, data_seed=seed, stop_after_grok=50,
                               dense_early=True, **small)
            gk.save(gk.train_run(cfg), str(s1 / (cfg.run_id + ".json")))
    sys.argv = ["analyze", "--runs", str(s1), "--stage0", str(s0), "--selection", str(tmp_path / "selection.json"),
                "--out", str(tmp_path / "report")]
    an.main()
    out = json.load(open(tmp_path / "report" / "compact.json"))
    assert {"H1", "H2", "H3", "holm", "verdicts"} <= set(out["results"]) and len(out["runs"]) == 3 * len(an.ALL)
    assert out["shared_lr"] is not None
    assert os.path.exists(tmp_path / "report" / "summary.md")
