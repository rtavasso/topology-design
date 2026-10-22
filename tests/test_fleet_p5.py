"""Phase 5 extensions of the fleet trainer (experiments/fastslow/DESIGN.md): every mechanism checked against a direct
computation on a single model, plus end-to-end smoke runs."""

import math

import pytest
import torch
import torch.nn.functional as F

from topology_design import fleet as fl
from topology_design import fleet_signal as fs
from topology_design import grokking as gk

SMALL = dict(p=11, d=16, hidden=32, steps=30, eval_every=10, dense_early=False, train_frac=0.5)
BR = ((1, 2, 3), (4, 5, 6))  # pk012000: long branch L = blocks 1-3, short branch S = blocks 4-6


def _setup(cfgs):
    f = fl.Fleet(cfgs)
    opt = fl.FleetAdamW(f)
    t = [fl.build_task(c) for c in cfgs]
    tr = torch.stack([x["tr"] for x in t])
    a, b = t[0]["a"], t[0]["b"]
    Y = torch.stack([x["y"] for x in t]).gather(1, tr)
    w = torch.ones(tr.shape, dtype=f.dtype)
    grid = {"a": a, "b": b, "key": t[0]["key"], "n_key": torch.bincount(t[0]["key"], minlength=cfgs[0].p),
            "y": torch.stack([x["y"] for x in t])}
    for s in ("train", "test", "exc", "rule_train"):
        grid[s] = torch.stack([x[s] for x in t])
    return f, opt, a[tr], b[tr], Y, w, grid, tr


def _single_parts(f, j, A, B):
    m = f.single(j)
    return m, m(A[j], B[j], return_parts=True)


# ---------------------------------------------------------------------------
# 5B: per-branch multipliers and the unmasked arm
# ---------------------------------------------------------------------------


def test_branch_multipliers_expand_to_blocks():
    a = fl.FleetConfig(arm="pk012000", branches=BR, branch_lr_mult=(2.0, 0.5), branch_wd_mult=(0.5, 2.0),
                       branch_alpha=(1.5, 0.75), **SMALL)
    b = fl.FleetConfig(arm="pk012000", branches=BR, block_lr_mult=(2, 2, 2, .5, .5, .5),
                       block_wd_mult=(.5, .5, .5, 2, 2, 2), alphas=(1.5, 1.5, 1.5, .75, .75, .75), **SMALL)
    fa, fb = fl.Fleet([a]), fl.Fleet([b])
    assert torch.equal(fa.alphas, fb.alphas)
    for k in fa.params:
        assert torch.equal(fa.lr_t[k], fb.lr_t[k]) and torch.equal(fa.wd_t[k], fb.wd_t[k])
    with pytest.raises(ValueError):
        fl.Fleet([fl.FleetConfig(arm="pk012000", branches=BR, branch_lr_mult=(1.0,), **SMALL)])


def test_unmasked_branch_gradients():
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, unmask_branch=0, model_seed=s, data_seed=s, **SMALL)
            for s in range(2)]
    f, opt, A, B, Y, w, grid, tr = _setup(cfgs)
    active = torch.ones(2, dtype=torch.bool)
    fl._loss_and_grads(f, opt, A, B, Y, w, active, {})
    for j in range(2):
        m, (lg, parts) = _single_parts(f, j, A, B)
        full = F.cross_entropy(lg, Y[j])
        g_full = torch.autograd.grad(full, list(m.parameters()), retain_graph=True)
        partial = F.cross_entropy(parts[0] + parts[1] + parts[2] + parts[3], Y[j])
        L_params = [m.w1[i].weight for i in range(3)] + [m.w2[i].weight for i in range(3)]
        g_part = dict(zip([id(p) for p in L_params], torch.autograd.grad(partial, L_params)))
        names = dict(m.named_parameters())
        for (name, p_), gf in zip(names.items(), g_full):
            fname = {"embed_a.weight": "embed_a", "embed_b.weight": "embed_b", "unembed.weight": "unembed"}.get(name)
            if fname is None:
                kind, i = name.split(".")[0], int(name.split(".")[1])
                fname = f"{kind}_{i}"
            ref = g_part[id(p_)] if id(p_) in g_part else gf
            torch.testing.assert_close(f.params[fname].grad[j], ref, atol=1e-6, rtol=1e-5, msg=name)


# ---------------------------------------------------------------------------
# 5C: routing
# ---------------------------------------------------------------------------


def test_group_gradient_equals_full_gradient_and_identity_routing_matches_plain():
    base = [fl.FleetConfig(arm="pk012000", branches=BR, model_seed=s, data_seed=s, **SMALL) for s in range(2)]
    grp = [fl.FleetConfig(arm="pk012000", branches=BR, route="agree", route_tau=0.0, model_seed=s, data_seed=s, **SMALL)
           for s in range(2)]
    f1, o1, A, B, Y, w, _, _ = _setup(base)
    f2, o2, *_ = _setup(grp)
    act = torch.ones(2, dtype=torch.bool)
    fl._loss_and_grads(f1, o1, A, B, Y, w, act, {})
    fl._loss_and_grads(f2, o2, A, B, Y, w, act, {})
    for k in f1.params:
        torch.testing.assert_close(f2.params[k].grad, f1.params[k].grad, atol=1e-6, rtol=1e-5)
    assert set(o2.sign_sum) == {k for k in f2.params if k.startswith("w")}
    # tau = 0 -> the agreement mask keeps everything -> routing is the identity
    for _ in range(5):
        for f, o in ((f1, o1), (f2, o2)):
            o.zero_grad()
            fl._loss_and_grads(f, o, A, B, Y, w, act, {})
            o.step(act)
    for k in f1.params:
        torch.testing.assert_close(f2.params[k], f1.params[k], atol=1e-5, rtol=1e-4)
    assert float(o2.take_angle().max()) < 0.1  # arccos near 1 in float32


def _route_case(mode, R=2):
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, route=mode, route_branch=0, model_seed=s, data_seed=s,
                           **SMALL) for s in range(R)]
    f = fl.Fleet(cfgs)
    o = fl.FleetAdamW(f)
    o.t = 10
    g = torch.Generator().manual_seed(0)
    U = {k: torch.randn(v.shape, generator=g) for k, v in f.params.items()}
    if o.ema_u is not None:
        o.ema_u = {k: torch.randn(v.shape, generator=g) for k, v in f.params.items()}
    o.sign_sum = {k: torch.randint(-8, 9, v.shape, generator=g).float() for k, v in f.params.items() if k.startswith("w")}
    return f, o, U


def _recv_names(L_blocks=(0, 1, 2)):
    return [f"w1_{i}" for i in L_blocks] + [f"w2_{i}" for i in L_blocks]


@pytest.mark.parametrize("mode", ["time", "agree", "rand", "topk", "rot"])
def test_routing_is_norm_matched_one_sided_and_well_defined(mode):
    f, o, U = _route_case(mode)
    out = o._route(dict(U))
    rn, sn = _recv_names(), _recv_names((3, 4, 5))
    for k in sn + ["embed_a", "embed_b", "unembed"]:
        assert torch.equal(out[k], U[k])  # other branch and shared params untouched
    nu = sum(U[k].flatten(1).pow(2).sum(1) for k in rn).sqrt()
    no = sum(out[k].flatten(1).pow(2).sum(1) for k in rn).sqrt()
    torch.testing.assert_close(no, nu, atol=1e-5, rtol=1e-5)
    ang = o.take_angle()
    assert bool((ang > 1.0).all()) and bool(torch.isfinite(ang).all())


def test_agreement_topk_and_random_masks_have_matched_density():
    for mode in ("agree", "topk", "rand"):
        f, o, U = _route_case(mode, R=1)
        out = o._route(dict(U))
        rn = _recv_names()
        amask_count = sum(((o.sign_sum[k].abs() / 8) >= 0.75).sum() for k in rn)
        nz = sum((out[k] != 0).sum() for k in rn)
        total = sum(U[k].numel() for k in rn)
        if mode in ("agree", "topk"):
            assert int(nz) == int(amask_count), mode
        else:
            assert abs(int(nz) - int(amask_count)) < 4 * math.sqrt(total * 0.25), mode


def test_rotation_null_matches_angle_of_time_routing():
    f, o, U = _route_case("rot", R=1)
    rn = _recv_names()
    E = {k: o.ema_u[k] / (1 - o.route_alpha[0] ** o.t) for k in rn}
    cos = sum((U[k] * E[k]).sum() for k in rn) / (sum(U[k].pow(2).sum() for k in rn).sqrt() *
                                                  sum(E[k].pow(2).sum() for k in rn).sqrt())
    expect = math.degrees(math.acos(float(cos)))
    o._route(dict(U))
    assert abs(float(o.take_angle()[0]) - expect) < 0.05


# ---------------------------------------------------------------------------
# 5D: consolidation
# ---------------------------------------------------------------------------


def test_consolidation_gradient():
    k_ = 0.3
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, consolidate=k_, model_seed=0, data_seed=0, **SMALL),
            fl.FleetConfig(arm="pk012000", branches=BR, model_seed=1, data_seed=1, **SMALL)]
    f, opt, A, B, Y, w, grid, tr = _setup(cfgs)
    n = tr.shape[1]
    st = {"kappa": torch.tensor([k_, 0.0]), "cons_beta": torch.tensor([0.99, 0.99]),
          "cons_perm": torch.stack([torch.arange(n), torch.arange(n)]), "cons_ema": None}
    act = torch.ones(2, dtype=torch.bool)
    fl._loss_and_grads(f, opt, A, B, Y, w, act, st)
    m, (lg, parts) = _single_parts(f, 0, A, B)
    zt = parts[1] + parts[2] + parts[3]
    zs = (parts[4] + parts[5] + parts[6]).detach()
    zt, zs = zt - zt.mean(-1, keepdim=True), zs - zs.mean(-1, keepdim=True)
    loss = F.cross_entropy(lg, Y[0]) + k_ * ((zt - zs) ** 2).mean(-1).mean()
    grads = dict(zip([n_ for n_, _ in m.named_parameters()], torch.autograd.grad(loss, list(m.parameters()))))
    torch.testing.assert_close(f.params["w1_0"].grad[0], grads["w1.0.weight"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(f.params["w2_2"].grad[0], grads["w2.2.weight"], atol=1e-6, rtol=1e-5)
    # source branch and shared parameters see only the task loss
    plain = dict(zip([n_ for n_, _ in m.named_parameters()],
                     torch.autograd.grad(F.cross_entropy(m(A[0], B[0]), Y[0]), list(m.parameters()))))
    torch.testing.assert_close(f.params["w1_4"].grad[0], plain["w1.4.weight"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(f.params["unembed"].grad[0], plain["unembed.weight"], atol=1e-6, rtol=1e-5)
    assert st["cons_ema"] is not None and st["cons_ema"].shape == (2, n, cfgs[0].p)


# ---------------------------------------------------------------------------
# 5E: arrivals and probes
# ---------------------------------------------------------------------------


def test_arrival_schedule():
    cfgs = [fl.FleetConfig(arrive_start=20, arrive_n=7, arrive_every=5, model_seed=s, data_seed=s, **SMALL)
            for s in range(2)]
    n = fl.build_task(cfgs[0])["tr"].numel()
    arr, inc, K = fl._arrivals(cfgs, n, "cpu")
    assert K == math.ceil((n - 20) / 7)
    assert int((arr[0] == 0).sum()) == 20 and int((inc[0] == 1).sum()) == 7 and int(arr.max()) == K * 5
    assert not torch.equal(arr[0], arr[1])  # per-seed order
    out = fl.train_fleet(cfgs)
    seen = [r["n_seen"] for r in out[0]["trajectory"]]
    assert seen == sorted(seen) and seen[0] == 20
    assert len(out[0]["summary"]["latency"]) == K + 1
    # no arrivals configured -> identical to the plain trainer
    plain = fl.FleetConfig(model_seed=0, data_seed=0, **SMALL)
    allin = fl.FleetConfig(model_seed=0, data_seed=0, arrive_start=0, **SMALL)
    a, b = fl.train_fleet([plain]), fl.train_fleet([allin])
    assert [r["train_loss"] for r in a[0]["trajectory"]] == [r["train_loss"] for r in b[0]["trajectory"]]


def test_probe_is_additive_and_matches_single_model():
    cfg = fl.FleetConfig(arm="pk012000", branches=BR, model_seed=0, data_seed=0, **SMALL)
    f, opt, A, B, Y, w, grid, tr = _setup([cfg])
    te = fl.build_task(cfg)["te"]
    pairs = te[:3].unsqueeze(0)
    pr = fs.probe(f, grid, pairs)
    m = f.single(0)
    a, b, y = grid["a"], grid["b"], grid["y"][0]
    for j in range(3):
        x = int(pairs[0, j])
        lg = m(a[x:x + 1], b[x:x + 1])
        ce = F.cross_entropy(lg, y[x:x + 1])
        lab = lg[0, y[x]] - lg[0].mean()
        gce = torch.autograd.grad(ce, list(m.parameters()), retain_graph=True)
        glab = torch.autograd.grad(lab, list(m.parameters()))
        imp = sum((gl * -gc).sum() for gl, gc in zip(glab, gce))
        assert abs(float(pr["imprint"][0, j].sum()) - float(imp)) < 1e-4
        # transfer: gen over the test set without x
        Z = m(a, b)
        labs = Z.gather(1, y.unsqueeze(1)).squeeze(1) - Z.mean(1)
        mask = grid["test"][0].clone()
        mask[x] = False
        gen_ex = labs[mask].mean()
        gg = torch.autograd.grad(gen_ex, list(m.parameters()))
        tra = sum((g_ * -gc).sum() for g_, gc in zip(gg, gce))
        assert abs(float(pr["transfer"][0, j].sum()) - float(tra)) < 1e-4


# ---------------------------------------------------------------------------
# 5A: signal measurement
# ---------------------------------------------------------------------------


def test_signal_measure_leaves_gradients_untouched_and_is_well_formed():
    cfgs = [fl.FleetConfig(arm="pk012000", branches=BR, measure_signal=True, model_seed=s, data_seed=s, **SMALL)
            for s in range(2)]
    f, opt, A, B, Y, w, grid, tr = _setup(cfgs)
    act = torch.ones(2, dtype=torch.bool)
    for _ in range(3):
        opt.zero_grad()
        fl._loss_and_grads(f, opt, A, B, Y, w, act, {})
        opt.step(act)
    opt.zero_grad()
    fl._loss_and_grads(f, opt, A, B, Y, w, act, {})
    before = {k: v.grad.clone() for k, v in f.params.items()}
    sig = fs.signal_measure(f, opt, grid, {"A": A, "B": B, "Y": Y, "w": w}, act)
    for k, v in f.params.items():
        assert torch.equal(v.grad, before[k])  # leakage guard: test-set gradients never reach p.grad
    for k, v in sig.items():
        assert v.shape[0] == 2 and bool(torch.isfinite(v).all()), k
    assert sig["S_time98"].shape[1] == 3 and sig["mask_M"].shape[1] == 2  # 2 branches + shared; 2 branches
    torch.testing.assert_close(sig["gen_sq"].sum(1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_separation_statistic_properties():
    g = torch.Generator().manual_seed(1)
    shape = (2, 5, 4)
    x = {"a": torch.randn(shape, generator=g)}
    Ggen = {"a": torch.randn(shape, generator=g)}
    Gmem = {"a": torch.randn(shape, generator=g)}
    zero = fs._sep_stats(x, x, Ggen, Gmem, ["a"])  # G parallel to x -> 0
    torch.testing.assert_close(zero, torch.zeros(2), atol=1e-6, rtol=0)
    # explicit computation
    G = {"a": torch.randn(shape, generator=g)}
    s = fs._sep_stats(x, G, Ggen, Gmem, ["a"])
    for r in range(2):
        xv, Gv, gv, mv = x["a"][r].flatten(), G["a"][r].flatten(), Ggen["a"][r].flatten(), Gmem["a"][r].flatten()
        Gp = Gv - (Gv @ xv) / (xv @ xv) * xv
        mp = mv - (mv @ gv) / (gv @ gv) * gv
        d = gv / gv.norm() - mp / mp.norm()
        assert abs(float(-(Gp @ d) / xv.norm()) - float(s[r])) < 1e-5


def test_support_gradient_matches_single_model():
    cfg = fl.FleetConfig(arm="pk012000", branches=BR, model_seed=2, data_seed=2, **SMALL)
    f, opt, A, B, Y, w, grid, tr = _setup([cfg])
    Ggen, Gmem = fs._support_grads(f, grid, chunk=37)
    m = f.single(0)
    a, b, y = grid["a"], grid["b"], grid["y"][0]
    Z = m(a, b)
    labs = Z.gather(1, y.unsqueeze(1)).squeeze(1) - Z.mean(1)
    gen = labs[grid["test"][0]].mean()
    gref = dict(zip([n for n, _ in m.named_parameters()], torch.autograd.grad(gen, list(m.parameters()))))
    torch.testing.assert_close(Ggen["w1_0"][0], gref["w1.0.weight"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(Ggen["unembed"][0], gref["unembed.weight"], atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [
    dict(route="time"), dict(route="rot"), dict(route="agree"), dict(route="rand"), dict(route="topk"),
    dict(unmask_branch=0), dict(consolidate=0.1), dict(consolidate=0.1, consolidate_null=True),
    dict(measure_signal=True), dict(probe_steps=(10, 20), probe_pairs=4),
    dict(branch_lr_mult=(0.5, 2.0), branch_wd_mult=(2.0, 0.5)),
])
def test_end_to_end_smoke(extra):
    cfgs = [fl.FleetConfig(arm=a, branches=BR, model_seed=s, data_seed=s, **SMALL, **extra)
            for s, a in enumerate(("pk012000", "pk000000"))]
    out = fl.train_fleet(cfgs)
    rec = out[0]["trajectory"][-1]
    assert out[0]["summary"]["steps_run"] == SMALL["steps"]
    if "route" in extra:
        assert rec["route_angle"] is not None
    if extra.get("measure_signal"):
        assert "sig_S_agree" in rec and "sig_mask_M" in rec
    if "probe_steps" in extra:
        assert [p["step"] for p in out[0]["probes"]] == [10, 20] and len(out[0]["probes"][0]["imprint"]) == 4
