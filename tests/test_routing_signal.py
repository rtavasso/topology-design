"""Numerical checks for docs/routing_signal_stability.md. Each test names the statement it checks (M = mixture of experts,
L = looped block, H = hard choice of a learned block per pass)."""
import itertools
import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- mixture of experts

def test_M1_M2_router_and_expert_gradients():
    g_ = torch.Generator().manual_seed(0)
    d, K, sel = 6, 5, [0, 2, 3]
    h, y = torch.randn(d, generator=g_), torch.randn(d, generator=g_)
    W = [torch.randn(d, d, generator=g_, requires_grad=True) for _ in range(K)]
    s = torch.randn(K, generator=g_, requires_grad=True)
    f = [torch.tanh(W[k] @ h) for k in range(K)]
    gate = torch.softmax(s[sel], 0)
    out = h + sum(gate[i] * f[k] for i, k in enumerate(sel))
    loss = 0.5 * ((out - y) ** 2).sum()
    loss.backward()
    delta = (out - y).detach()
    proj = torch.stack([delta @ f[k].detach() for k in sel])
    mean = (gate.detach() * proj).sum()
    for i, k in enumerate(sel):                                           # M2
        assert torch.allclose(s.grad[k], gate[i].detach() * (proj[i] - mean), atol=1e-12)
    for k in range(K):
        if k not in sel:
            assert s.grad[k] == 0 and W[k].grad is None                  # experts and logits not selected get nothing
    for i, k in enumerate(sel):                                           # M1
        Wk = W[k].detach().clone().requires_grad_(True)
        (delta @ torch.tanh(Wk @ h)).backward()
        assert torch.allclose(W[k].grad, gate[i].detach() * Wk.grad, atol=1e-12)


def _moe_flow(K, y, eps=1e-7, dt=2e-3):
    s, w = np.zeros(K), np.zeros(K)
    s[0] += eps
    s -= s.mean()
    sigma0 = s[0]
    losses = []
    for _ in range(int(45 * K / dt)):
        g = np.exp(s - s.max()); g /= g.sum()
        yh = g @ w
        r = yh - y
        losses.append(0.5 * r * r)
        w, s = w - dt * g * r, s - dt * g * r * (w - yh)
    return (s[0] - s.mean()) / sigma0, r, losses


def test_M3_M4_assignment_asymmetry_is_amplified_by_cosh_of_the_initial_residual():
    for K in (2, 4):
        for y in (1.0, 3.0):
            amp, r_end, losses = _moe_flow(K, y)
            assert abs(r_end) < 1e-6
            assert all(b <= a + 1e-15 for a, b in zip(losses, losses[1:]))            # M3: the loss only goes down
            assert abs(amp / math.cosh(abs(y)) - 1) < 0.01, (K, y, amp)              # M4


def test_M5_mixture_inside_and_outside_differ_by_the_curvature_weighted_variance():
    g_ = torch.Generator().manual_seed(1)
    d, K = 5, 4
    A = torch.randn(d, d, generator=g_)
    H, b, h = A @ A.T, torch.randn(d, generator=g_), torch.randn(d, generator=g_)
    f = torch.randn(K, d, generator=g_)
    gate = torch.softmax(torch.randn(K, generator=g_), 0)
    L = lambda v: 0.5 * v @ H @ v + b @ v
    fbar = gate @ f
    outside = sum(gate[k] * L(h + f[k]) for k in range(K))
    inside = L(h + fbar)
    gap = 0.5 * sum(gate[k] * (f[k] - fbar) @ H @ (f[k] - fbar) for k in range(K))
    assert torch.allclose(outside - inside, gap, atol=1e-10)


# ---------------------------------------------------------------- looped block

def test_L1_L2_gradient_is_a_sum_over_passes_and_sensitivity_is_a_sum_of_jacobian_products():
    g_ = torch.Generator().manual_seed(2)
    d, n = 4, 5
    W = (0.5 * torch.randn(d, d, generator=g_)).requires_grad_(True)
    U, x, y = torch.randn(d, d, generator=g_), torch.randn(d, generator=g_), torch.randn(d, generator=g_)
    hs = [torch.zeros(d)]
    for _ in range(n):
        hs.append(torch.tanh(W @ hs[-1] + U @ x))
    loss = 0.5 * ((hs[-1] - y) ** 2).sum()
    loss.backward()
    lam = (hs[-1] - y).detach()                                           # lambda_n
    total = torch.zeros(d, d)
    sens = torch.zeros(d, d * d)                                          # d h_n / d vec(W)
    Phi = torch.eye(d)                                                    # product of Jacobians from pass t+1 to n
    for t in reversed(range(n)):
        D = torch.diag(1 - hs[t + 1].detach() ** 2)                       # derivative of tanh at pass t
        total += torch.outer(D @ lam, hs[t].detach())                     # G_t^T lambda_{t+1}
        G = torch.kron(D, hs[t].detach().unsqueeze(0))                    # d F / d vec(W) at pass t
        sens += Phi @ G
        lam = W.detach().T @ (D @ lam)                                    # lambda_t = J_t^T lambda_{t+1}
        Phi = Phi @ (D @ W.detach())
    assert torch.allclose(W.grad, total, atol=1e-10)
    jac = torch.autograd.functional.jacobian(
        lambda w: _loop_out(w.view(d, d), U, x, n), W.detach().reshape(-1))
    assert torch.allclose(jac, sens, atol=1e-10)


def _loop_out(W, U, x, n):
    h = torch.zeros(W.shape[0])
    for _ in range(n):
        h = torch.tanh(W @ h + U @ x)
    return h


def test_L3_scalar_loop_sensitivity():
    x = 0.7
    for a, n in ((0.5, 6), (1.0, 6), (1.0, 12), (1.2, 8)):
        hs = [0.0]
        for _ in range(n):
            hs.append(a * hs[-1] + x)
        formula = sum(a ** (n - 1 - t) * hs[t] for t in range(n))
        at = torch.tensor(a, requires_grad=True)
        h = torch.tensor(0.0)
        for _ in range(n):
            h = at * h + x
        h.backward()
        assert abs(at.grad.item() - formula) < 1e-9
    # at a = 1 the sensitivity is x n (n - 1) / 2: it grows as the square of the loop count without any expansion
    assert abs(sum(1.0 * (t * x) for t in range(12)) - x * 12 * 11 / 2) < 1e-12


# ---------------------------------------------------------------- hard choice of a learned block per pass

def _p(S, Th):
    pi, q = torch.softmax(S, 1), torch.softmax(Th, 1)
    return ((pi * q.T).sum(1)).prod()


def _enumerate(S, Th):
    n, K = S.shape
    pi, q = torch.softmax(S, 1), torch.softmax(Th, 1)
    routes = list(itertools.product(range(K), repeat=n))
    prob = torch.stack([torch.stack([pi[t, z[t]] for t in range(n)]).prod() for z in routes])
    succ = torch.stack([torch.stack([q[z[t], t] for t in range(n)]).prod() for z in routes])
    return routes, prob, succ


def test_H1_policy_gradient_identity_by_enumeration():
    g_ = torch.Generator().manual_seed(3)
    n = K = 3
    S = torch.randn(n, K, generator=g_, requires_grad=True)
    Th = torch.randn(K, n, generator=g_)
    routes, prob, succ = _enumerate(S, Th)
    p = (prob * succ).sum()
    assert torch.allclose(p, _p(S, Th))
    p.backward()
    pi = torch.softmax(S.detach(), 1)
    for t in range(n):
        for j in range(K):
            mask = torch.tensor([z[t] == j for z in routes])
            Q = (prob.detach() * succ.detach())[mask].sum() / prob.detach()[mask].sum()
            assert torch.allclose(S.grad[t, j], pi[t, j] * (Q - p.detach()), atol=1e-12)


def _flat(f, n, K):
    return lambda v: f(v[: n * K].view(n, K), v[n * K:].view(K, n))


def test_H2_H4_likelihood_and_expected_loss_differ_by_the_factor_p():
    g_ = torch.Generator().manual_seed(4)
    v = torch.randn(18, generator=g_)                                    # a generic point, n = K = 3
    g_p = torch.autograd.functional.jacobian(_flat(_p, 3, 3), v)
    g_l = torch.autograd.functional.jacobian(_flat(lambda S, T: -torch.log(_p(S, T)), 3, 3), v)
    assert torch.allclose(g_l, -g_p / _flat(_p, 3, 3)(v), atol=1e-12)                 # H2, everywhere
    for n, K in ((2, 2), (3, 3), (4, 4), (2, 4), (3, 6)):                 # the unspecialised point of the toy
        v0 = torch.zeros(2 * n * K)
        p0 = _flat(_p, n, K)(v0)
        assert abs(p0.item() - n ** -n) < 1e-12
        assert torch.autograd.functional.jacobian(_flat(_p, n, K), v0).abs().max() < 1e-12      # stationary
        e_r = torch.linalg.eigvalsh(torch.autograd.functional.hessian(_flat(lambda S, T: 1 - _p(S, T), n, K), v0))
        e_l = torch.linalg.eigvalsh(torch.autograd.functional.hessian(_flat(lambda S, T: -torch.log(_p(S, T)), n, K), v0))
        assert torch.allclose(e_r, p0 * e_l, atol=1e-12)                  # H2 at a stationary point: Hessians differ by p
        assert abs(e_l.min().item() + 1 / K) < 1e-10                      # H4: escape rate 1/K for the likelihood
        assert abs(e_r.min().item() + p0.item() / K) < 1e-12              #     and p/K for the expected loss


def test_H3_reinforce_signal_to_noise_for_one_correct_route():
    for K, n in ((3, 3), (3, 6)):
        t, j = 1, 0
        routes = list(itertools.product(range(K), repeat=n))
        star = (0,) * n
        pr = K ** -n
        for b, kind in ((0.0, "reward"), (1.0, "loss")):                  # b = 1 for the reward is the loss with no baseline
            est = [((1.0 if z == star else 0.0) - b) * ((1.0 if z[t] == j else 0.0) - 1 / K) for z in routes]
            mean = sum(est) * pr
            var = sum(e * e for e in est) * pr - mean ** 2
            assert abs(mean - K ** -n * (1 - 1 / K)) < 1e-12
            snr = abs(mean) / math.sqrt(var)
            if kind == "reward":
                assert abs(snr - K ** (-n / 2) / math.sqrt(1 - K ** -n)) < 1e-9
            else:
                second = (1 / K - K ** -n) * (1 - 1 / K) ** 2 + (1 - 1 / K) / K ** 2
                assert abs(var - (second - mean ** 2)) < 1e-12
                if n >= 6:
                    assert abs(snr / (math.sqrt(K - 1) * K ** -n) - 1) < 0.01     # to leading order in K^-n


def test_H5_a_route_that_reuses_a_block_is_a_dead_end_for_exact_gradient_ascent():
    n = K = 3
    big = 8.0
    S = torch.tensor([[big, 0, 0], [big, 0, 0], [0, big, 0]])            # passes 1, 2 use block 0; pass 3 uses block 1
    Th = torch.tensor([[big, big, -big], [-big, -big, big], [0.0, 0.0, 0.0]])  # block 0 half-good at two jobs; block 2 idle
    S.requires_grad_(True); Th.requires_grad_(True)
    p = _p(S, Th)
    assert abs(p.item() - 0.25) < 0.01
    p.backward()
    assert S.grad[0, 2] < 0 and S.grad[1, 2] < 0                         # the router is pushed away from the idle block
    assert Th.grad[2].abs().max() < 1e-3                                 # the idle block is barely trained ...
    assert abs((Th.grad[2, 0] - Th.grad[2, 1]).item()) < 1e-12           # ... and not toward either job in particular
    S, Th = S.detach().clone(), Th.detach().clone()
    for _ in range(3000):                                                # exact gradient ascent on p does not leave
        S.requires_grad_(True); Th.requires_grad_(True)
        p = _p(S, Th)
        gS, gT = torch.autograd.grad(p, (S, Th))
        S, Th = (S + 0.5 * gS).detach(), (Th + 0.5 * gT).detach()
    assert _p(S, Th).item() < 0.26


def test_H6_fixed_uniform_prior_is_flat_and_every_deterministic_router_is_a_fixed_point_of_its_own_posterior():
    n = K = 3
    uniform = torch.full((K, n), 1 / n)
    perm = torch.eye(n)
    ds = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])           # doubly stochastic
    marg = lambda q: (q.sum(0) / K).prod()                                # marginal success under a uniform prior on routes
    assert abs(marg(uniform) - marg(perm)) < 1e-15 and abs(marg(ds) - marg(perm)) < 1e-15
    q = torch.softmax(torch.tensor([[1.0, 0.2, -0.3], [0.1, 0.4, 0.0], [-0.5, 0.3, 0.9]]), 1)
    pi = torch.softmax(torch.tensor([[30.0, 0, 0], [30.0, 0, 0], [0, 30.0, 0]]), 1)   # a deterministic router
    own = pi * q.T / (pi * q.T).sum(1, keepdim=True)                      # its own posterior, pass by pass
    fixed = q.T / q.T.sum(1, keepdim=True)                                # posterior under the fixed uniform prior
    assert (own - pi).abs().max() < 1e-9
    assert (fixed - pi).abs().max() > 0.3
