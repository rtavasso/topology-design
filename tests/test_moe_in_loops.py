"""Exact statements behind docs/moe_in_loops.md (A1-A7). The architecture rounds themselves live in
experiments/theory/streams/ and are not tests."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "streams"))
from loopmoe import DEF, LMAX, Student, rot_minus_id, sinus_keys  # noqa: E402

torch.set_default_dtype(torch.float64)


def _student(seed=0, **over):
    cfg = dict(DEF); cfg.update(over)
    return Student(cfg, seed), cfg


def _data_path_product(S, cfg, progs, x):
    """Product over passes of I + eps * sum_k g_k V_k diag(1 - tanh^2(B_k h)) B_k, with the gates held fixed."""
    _, aux = S.forward(progs, x, record=True)
    J = torch.eye(cfg["d"])
    for h, g in zip(aux["states"], aux["gates"]):
        h, g = h[0, 0].detach(), g[0, 0]
        Jt = torch.eye(cfg["d"])
        for k in range(cfg["K"]):
            D = torch.diag(1 - torch.tanh(S.p["B"][k].detach() @ h) ** 2)
            Jt = Jt + cfg["eps"] * g[k] * S.p["V"][k].detach() @ D @ S.p["B"][k].detach()
        J = Jt @ J
    return J


def test_A1_a_tied_gate_changes_across_passes_only_through_the_state():
    progs = torch.tensor([[0, 1, 2, 1]])
    x = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(0))
    for over in (dict(progress="none"), dict(progress="reg0")):
        S, _ = _student(**over)
        if "M" in S.p:
            S.p["M"].data.zero_()                                    # a register that does not move
        _, aux = S.forward(progs, x, record=True)
        G = torch.stack(aux["gates"])
        assert (G - G[:1]).abs().max().item() < 1e-14                # same gate on every pass, whatever the program
    S, _ = _student(progress="regrot")
    _, aux = S.forward(progs, x, record=True)
    G = torch.stack(aux["gates"])
    assert (G - G[:1]).abs().max().item() > 1e-4                     # a moving register changes it


def test_A2_gate_that_reads_only_context_and_register_adds_no_term_to_the_data_jacobian():
    progs = torch.tensor([[0, 2, 1, 1]])
    x = torch.randn(1, 1, 6, generator=torch.Generator().manual_seed(1))
    for gate_h, same in ((False, True), (True, False)):
        S, cfg = _student(progress="regrot", gate_h=gate_h)
        S.p["V"].data.mul_(10.0)                                     # make the streams non-negligible
        if gate_h:
            S.p["W2"].data.mul_(30.0)
        J = torch.autograd.functional.jacobian(lambda v: S.forward(progs, v.reshape(1, 1, -1))[0].reshape(-1), x.reshape(-1))
        gap = (J - _data_path_product(S, cfg, progs, x)).abs().max().item()
        assert (gap < 1e-12) if same else (gap > 1e-4), (gate_h, gap)


def _one_pass(score, topk, renorm, seed=2):
    S, cfg = _student(seed, progress="clock", score=score, topk=topk, renorm=renorm, n=1)
    S.p["V"].data.mul_(10.0); S.p["G"].data.mul_(10.0)
    g = torch.Generator().manual_seed(seed)
    progs, x, y = torch.tensor([[1]]), torch.randn(1, 1, 6, generator=g), torch.randn(6, generator=g)
    h, aux = S.forward(progs, x, record=True)
    s = aux["logits"][0]; s.retain_grad()
    (0.5 * ((h[0, 0] - y) ** 2).sum()).backward()
    lam = (h[0, 0] - y).detach()
    with torch.no_grad():
        f = torch.einsum("kiw,kw->ki", S.p["V"], torch.tanh(S.p["B"] @ x[0, 0]))
    return S, cfg, s.detach()[0, 0], s.grad[0, 0], aux["gates"][0][0, 0], lam, f


def test_A3_gating_before_the_streams_what_reaches_streams_and_logits():
    # sigmoid scores, top-2, no renormalisation: nothing reaches an unrun stream or its logit; a run stream's logit
    # gets an absolute signal sigma'(s_j) * eps * <lambda, f_j> that does not involve the other streams.
    S, cfg, s, ds, g, lam, f = _one_pass("sigmoid", 2, False)
    ran = g > 0
    assert ran.sum() == 2
    assert S.p["B"].grad[~ran].abs().max().item() == 0 and S.p["V"].grad[~ran].abs().max().item() == 0
    assert ds[~ran].abs().max().item() == 0
    sig = torch.sigmoid(s)
    assert torch.allclose(ds[ran], (sig * (1 - sig) * cfg["eps"] * (f @ lam))[ran], atol=1e-12)
    # softmax over all streams, top-2, not renormalised: an unrun stream's logit does get a gradient, but it is
    # -p_j * eps * sum over the run streams of p_k <lambda, f_k>: it contains nothing about stream j itself.
    S, cfg, s, ds, g, lam, f = _one_pass("softmax", 2, False)
    ran = g > 0; p = torch.softmax(s, 0)
    common = cfg["eps"] * (p[ran] * (f[ran] @ lam)).sum()
    assert torch.allclose(ds[~ran], -p[~ran] * common, atol=1e-12)
    assert ds[~ran].abs().min().item() > 0
    # softmax renormalised over the run streams, top-1: the gate is identically 1 and the logits get nothing at all.
    S, cfg, s, ds, g, lam, f = _one_pass("softmax", 1, True)
    assert ds.abs().max().item() == 0 and abs(g.max().item() - 1) < 1e-15


def test_A4_one_pass_gain_bound_scales_with_the_sum_of_gates():
    gen = torch.Generator().manual_seed(3)
    bounds = {}
    for name, over in (("softmax", dict(score="softmax")), ("sig_dense", dict(score="sigmoid")), ("sig_top1", dict(score="sigmoid", topk=1))):
        S, cfg = _student(4, progress="clock", n=1, **over)
        S.p["V"].data.mul_(10.0); S.p["G"].data.add_(3.0)             # push sigmoid scores towards 1
        lip = torch.stack([torch.linalg.matrix_norm(S.p["V"][k], 2) * torch.linalg.matrix_norm(S.p["B"][k], 2) for k in range(cfg["K"])]).detach()
        worst = 0.0
        for _ in range(20):
            x = torch.randn(1, 1, 6, generator=gen)
            J = torch.autograd.functional.jacobian(lambda v: S.forward(torch.tensor([[0]]), v.reshape(1, 1, -1))[0].reshape(-1), x.reshape(-1))
            _, aux = S.forward(torch.tensor([[0]]), x, record=True)
            g = aux["gates"][0][0, 0]
            assert torch.linalg.matrix_norm(J, 2).item() <= 1 + cfg["eps"] * (g * lip).sum().item() + 1e-12
            worst = max(worst, g.sum().item())
        bounds[name] = worst
    assert bounds["softmax"] <= 1 + 1e-12                             # a partition of unity: gates sum to one
    assert bounds["sig_top1"] <= 1 + 1e-12 < bounds["sig_dense"]      # sigmoid gates sum up to K dense, up to k sparse


def test_A5_rotating_register_reads_one_position_further_each_pass_at_any_length():
    mc = 8
    keys = sinus_keys(mc)
    R = rot_minus_id([2 * math.pi * j / LMAX for j in range(1, mc // 2 + 1)]) + torch.eye(mc)
    c = keys[0].clone()
    for t in range(LMAX):
        assert torch.allclose(c, keys[t], atol=1e-12)                 # the register IS the key of position t
        assert int((keys @ c).argmax()) == t
        c = R @ c
    S, _ = _student(progress="regsin", keys="sinus", n=12)
    progs = torch.zeros(1, 12, dtype=torch.long)
    _, aux = S.forward(progs, torch.randn(1, 1, 6, generator=torch.Generator().manual_seed(0)), record=True)
    assert [int(a[0, 0].argmax()) for a in aux["attn"]] == list(range(12))


def test_A6_default_vector_is_zero_when_stream_outputs_average_to_zero():
    # Streams here are odd functions of the state; over a state distribution symmetric about zero their mean output
    # is zero, so a running-mean stand-in for an unrun stream is zero and carries no signal to the gate.
    S, cfg = _student(5, progress="clock", score="softmax", topk=1, fill="ema", n=1)
    S.p["V"].data.mul_(10.0)
    half = torch.randn(4, 8, 6, generator=torch.Generator().manual_seed(6))
    x = torch.cat([half, -half], 1)
    progs = torch.tensor([[0], [1], [2], [0]])
    for _ in range(5):
        S.forward(progs, x, train=True)
    assert S.ema.abs().max().item() < 1e-12


def test_A7_once_the_gate_has_decided_running_only_the_top_stream_changes_little():
    # Softmax over all streams, not renormalised. Dense and top-1 outputs of one pass differ by at most
    # eps * (1 - g_top) * max_k |f_k|, and a stream's gradient in the dense pass is proportional to its gate.
    gen = torch.Generator().manual_seed(7)
    for scale in (1.0, 30.0, 300.0):                                  # sharper and sharper gates
        Sd, cfg = _student(8, progress="clock", n=1)
        St, _ = _student(8, progress="clock", n=1, topk=1)
        for S in (Sd, St):
            S.p["V"].data.mul_(10.0); S.p["G"].data.mul_(scale)
        x = torch.randn(1, 1, 6, generator=gen); progs = torch.tensor([[2]])
        hd, aux = Sd.forward(progs, x, record=True)
        ht, _ = St.forward(progs, x)
        g = aux["gates"][0][0, 0]
        with torch.no_grad():
            f = torch.einsum("kiw,kw->ki", Sd.p["V"], torch.tanh(Sd.p["B"] @ x[0, 0]))
        assert (hd - ht).norm().item() <= cfg["eps"] * (1 - g.max().item()) * f.norm(dim=1).max().item() + 1e-12
        hd.sum().backward()
        gV = Sd.p["V"].grad.flatten(1).norm(dim=1)
        ref = torch.stack([torch.autograd.grad((Sd.p["V"][k] @ torch.tanh(Sd.p["B"][k].detach() @ x[0, 0])).sum(), Sd.p["V"])[0][k].norm() for k in range(cfg["K"])])
        assert torch.allclose(gV, cfg["eps"] * g * ref, atol=1e-12)


def test_round_M_monoliths_are_parameter_matched_and_test_sets_are_clean():
    from loopmoe import make_programs
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "streams"))
    import arch_m
    n = {}
    for label, units, over in arch_m.ROWS:
        if over is None:
            continue
        cfg = dict(DEF); cfg.update(over)
        S = Student(cfg, 0)
        n[label.split()[0]] = sum(v.numel() for v in S.params())
        progs = torch.zeros(2, 5, dtype=torch.long)
        h, _ = S.forward(progs, torch.randn(2, 3, cfg["d"], generator=torch.Generator().manual_seed(0)))
        assert h.shape == (2, 3, cfg["d"]) and torch.isfinite(h).all()
        tr, te = make_programs(cfg, 3)
        assert not ({tuple(r) for r in te[4].tolist()} & {tuple(r) for r in tr[4].tolist()})
    for cand, monos in (("M1", ("M2", "M3", "M4", "M7")), ("M8", ("M9", "M10")), ("M12", ("M13", "M15"))):
        for m in monos:
            assert abs(n[m] - n[cand]) / n[cand] < 0.03, (cand, m, n[cand], n[m])
    assert n["M5"] < n["M1"] and n["M11"] < n["M8"] and n["M14"] < n["M12"]      # compute-matched blocks are smaller
