"""Exact scalar/vector fixtures from the v0 math doc (archive/v0/docs/03_first_principles_math.md, 04_gradient_dynamics.md).

These were listed as "math oracles required in CI" in v0 but never implemented.
"""

import math

import pytest
import torch

torch.set_default_dtype(torch.float64)


def graph_A(a, b, c, x):  # u=x+ax; v=bu; w=cv; y=x+w   ->  (1+bc+abc)x
    u = x + a * x
    return x + c * (b * u)


def graph_B(a, b, c, x):  # u=ax; v=bu; w=x+v; y=x+cw   ->  (1+c+abc)x
    w = x + b * (a * x)
    return x + c * w


def test_scalar_dag_polynomials_and_zero_gradients():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a, b, c, x = torch.randn(4, generator=g)
        assert graph_A(a, b, c, x) == pytest.approx(((1 + b * c + a * b * c) * x).item())
        assert graph_B(a, b, c, x) == pytest.approx(((1 + c + a * b * c) * x).item())
    x = torch.tensor(1.7)
    zero = torch.zeros(3, requires_grad=True)
    gA = torch.autograd.grad(graph_A(*zero, x), zero)[0]
    gB = torch.autograd.grad(graph_B(*zero, x), zero)[0]
    torch.testing.assert_close(gA, torch.zeros(3))
    torch.testing.assert_close(gB, torch.tensor([0.0, 0.0, 1.7]))


@pytest.mark.parametrize("eps", [1e-3, -1e-3, 0.05, -0.2])
def test_minimum_displacement(eps):
    # A realizes slope 1+eps with a=0, bc=eps: min ||(b,c)|| = sqrt(2|eps|) at |b|=|c|=sqrt|eps|
    b = math.sqrt(abs(eps))
    c = math.copysign(b, eps)
    assert b * c == pytest.approx(eps)
    assert math.hypot(b, c) == pytest.approx(math.sqrt(2 * abs(eps)))
    # grid check that nothing on the constraint is shorter
    bs = torch.linspace(1e-4, 3, 20001)
    assert torch.sqrt(bs**2 + (eps / bs) ** 2).min().item() >= math.sqrt(2 * abs(eps)) - 1e-6


def test_residual_product_hessian_is_indefinite():
    loss = lambda t: 0.5 * (1 + t[0]) ** 2 * (1 + t[1]) ** 2
    H = torch.autograd.functional.hessian(loss, torch.zeros(2))
    torch.testing.assert_close(H, torch.tensor([[1.0, 2.0], [2.0, 1.0]]))
    torch.testing.assert_close(torch.linalg.eigvalsh(H), torch.tensor([-1.0, 3.0]))


def test_relu_residual_abs_and_plain_halfline():
    x = torch.linspace(-3, 3, 61)
    torch.testing.assert_close(x + 2 * torch.relu(-x), x.abs())
    g = torch.Generator().manual_seed(1)
    for v, w in torch.randn(100, 2, generator=g):
        f = lambda t: v * torch.relu(w * t)
        # a bias-free single ReLU is identically zero on one half-line, so it can't equal |x|
        assert f(torch.tensor(1.0)) == 0 or f(torch.tensor(-1.0)) == 0


def test_bottleneck_rank():
    g = torch.Generator().manual_seed(2)
    d, m = 6, 2
    W1, W2 = torch.randn(m, d, generator=g), torch.randn(d, m, generator=g)
    D = torch.diag(torch.rand(m, generator=g) + 0.5)
    assert torch.linalg.matrix_rank(W2 @ D @ W1) <= m
    assert torch.linalg.matrix_rank(torch.eye(d) + W2 @ D @ W1) == d


def test_local_injectivity_bound():
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        J = torch.randn(5, 5, generator=g)
        J = 0.9 * J / torch.linalg.matrix_norm(J, 2)
        smin = torch.linalg.svdvals(torch.eye(5) + J).min()
        assert smin >= 1 - torch.linalg.matrix_norm(J, 2) - 1e-12


def test_one_step_projection_erases_only_x():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(5, generator=g)
    x = x / x.norm()
    W = torch.zeros(5, 5, requires_grad=True)
    loss = 0.5 * ((x + W @ x) ** 2).sum()
    (gW,) = torch.autograd.grad(loss, W)
    W1 = -gW  # eta = 1
    torch.testing.assert_close(W1, -torch.outer(x, x))
    f = lambda z: z + W1 @ z
    torch.testing.assert_close(f(x), torch.zeros(5), atol=1e-12, rtol=0)
    z = torch.randn(5, generator=g)
    z_perp = z - (z @ x) * x
    torch.testing.assert_close(f(z_perp), z_perp)


def test_softmax_gauge():
    g = torch.Generator().manual_seed(5)
    z = torch.randn(8, 4, generator=g)
    y = torch.randint(0, 4, (8,), generator=g)
    ce = torch.nn.functional.cross_entropy
    torch.testing.assert_close(ce(z, y), ce(z + 3.7, y))
    cz = lambda t: t - t.mean(1, keepdim=True)
    torch.testing.assert_close(cz(z), cz(z + torch.randn(8, 1, generator=g)))


def test_two_logit_ce_matches_logistic_margin():
    g = torch.Generator().manual_seed(6)
    z = torch.randn(10, 2, generator=g)
    y = torch.randint(0, 2, (10,), generator=g)
    sign = 2.0 * y - 1.0
    margin = sign * (z[:, 1] - z[:, 0])
    torch.testing.assert_close(
        torch.nn.functional.cross_entropy(z, y, reduction="none"), torch.nn.functional.softplus(-margin)
    )


def test_residual_norm_bound():
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        p = torch.softmax(3 * torch.randn(6, generator=g), 0)
        y = torch.randint(0, 6, (1,), generator=g).item()
        r = p.clone()
        r[y] -= 1
        assert r.norm() <= math.sqrt(2) * (1 - p[y]) + 1e-12


@pytest.mark.parametrize("m", [-3.0, 0.0, 2.0, 5.0])
def test_shared_ce_cross_hessian(m):
    loss = lambda t: torch.nn.functional.softplus(-(t[0] + t[1]))
    H = torch.autograd.functional.hessian(loss, torch.tensor([m / 2, m / 2]))
    s = torch.sigmoid(torch.tensor(m))
    assert H[0, 1].item() == pytest.approx((s * (1 - s)).item())


def test_shortcut_frustration_toy():
    # x1 balanced, x2=+1 w.p. .9, y = x1*x2; margin m = a*x1 + d*x1*x2
    L = lambda t: 0.9 * torch.nn.functional.softplus(-(t[0] + t[1])) + 0.1 * torch.nn.functional.softplus(t[0] - t[1])
    t = torch.tensor([math.log(9), 0.0], requires_grad=True)
    ga, gd = torch.autograd.grad(L(t), t)[0]
    assert ga.item() == pytest.approx(0.0, abs=1e-12)
    assert gd.item() == pytest.approx(-0.18)


def test_product_path_starvation():
    L = lambda t: 0.9 * torch.nn.functional.softplus(-(t[0] + t[1] * t[2])) + 0.1 * torch.nn.functional.softplus(
        t[0] - t[1] * t[2]
    )
    t = torch.tensor([math.log(9), 0.0, 0.0], requires_grad=True)
    g = torch.autograd.grad(L(t), t)[0]
    torch.testing.assert_close(g, torch.zeros(3), atol=1e-12, rtol=0)  # a strict saddle in (b, c)
    t2 = torch.tensor([math.log(9), 0.0, 0.5], requires_grad=True)
    g2 = torch.autograd.grad(L(t2), t2)[0]
    assert g2[1].item() == pytest.approx(0.5 * -0.18)  # escape direction appears once c != 0
