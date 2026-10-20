"""Tests for every numbered result in docs/phase0_linear_theory.md."""

import itertools

import pytest
import torch

from topology_design import linear_family as lf

torch.set_default_dtype(torch.float64)


def _single_example_first_step(parents, task, eta, i=0):
    """One SGD step on example i from theta=0. Returns end-to-end W after the step."""
    L, d = len(parents), task.X.shape[1]
    grads = lf.param_grads(lf.zeros(L, d), parents, task, idx=slice(i, i + 1))
    return lf.end_to_end([-eta * g for g in grads], parents)


def _balanced_trained(L, task, steps=1500, lr=0.02):
    return lf.train(lf.zeros(L, task.X.shape[1]), lf.sequential(L), task, steps=steps, lr=lr)


@pytest.fixture(scope="module")
def task():
    return lf.make_teacher_task(d=5, V=3, n=120, teacher_scale=1.5, seed=1)


# --- Prop. 1 -----------------------------------------------------------------


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_first_order_universal(L):
    d = 4
    base = lf.small_random(L, d, 1.0, seed=L)
    for parents in lf.all_parent_vectors(L):
        def resid(eps):
            Ws = [eps * W for W in base]
            return (lf.end_to_end(Ws, parents) - torch.eye(d) - sum(Ws)).norm().item()

        r1, r2 = resid(1e-2), resid(1e-3)
        assert r2 <= r1 / 100 * 1.2 + 1e-15, (parents, r1, r2)  # residual is O(eps^2)


def test_all_parent_vectors_count():
    assert [len(lf.all_parent_vectors(L)) for L in range(1, 6)] == [1, 2, 6, 24, 120]


def test_identical_ntk_at_zero_for_every_topology(task):
    L, d = 3, task.X.shape[1]
    x, xp = task.X[0], task.X[1]
    expected = L * (task.R @ task.R.T) * (x @ xp)
    for parents in lf.all_parent_vectors(L):
        K = lf.ntk_empirical(lf.zeros(L, d), parents, task.R, x, xp)
        torch.testing.assert_close(K, expected)


# --- Prop. 2 -----------------------------------------------------------------


def _check_parallel_conservation(task, optimizer, lr):
    L, d = 3, task.X.shape[1]
    W0 = lf.small_random(L, d, 0.1, seed=7)
    W1 = lf.train(W0, lf.parallel(L), task, steps=300, lr=lr, optimizer=optimizer)
    assert (W1[0] - W0[0]).norm() > 0.1  # it actually trained
    for i, j in itertools.combinations(range(L), 2):
        torch.testing.assert_close(W1[i] - W1[j], W0[i] - W0[j], atol=1e-11, rtol=0)


def test_parallel_conservation_sgd(task):
    _check_parallel_conservation(task, "sgd", 0.5)


def test_parallel_conservation_adam(task):
    _check_parallel_conservation(task, "adam", 0.01)


# --- Prop. 3 -----------------------------------------------------------------


def test_balancedness_drift_scales_with_eta(task):
    """Fixed training time t = eta*steps: drift from the conserved value shrinks ~linearly in eta."""
    def drift(eta):
        W = lf.train(lf.zeros(2, task.X.shape[1]), lf.sequential(2), task, steps=int(4.0 / eta), lr=eta)
        return lf.balancedness_residuals(W)[0].norm().item()

    d1, d2, d3 = drift(0.04), drift(0.02), drift(0.01)
    assert d1 > d2 > d3 > 0
    for a, b in [(d1, d2), (d2, d3)]:
        assert 1.6 < a / b < 2.5, (d1, d2, d3)  # halving eta roughly halves the drift


@pytest.mark.parametrize("L", [2, 3])
def test_end_to_end_flow_matches_arora(task, L):
    Ws = _balanced_trained(L, task)
    W = lf.end_to_end(Ws, lf.sequential(L))
    eta = 1e-7
    grads = lf.param_grads(Ws, lf.sequential(L), task)
    W_next = lf.end_to_end([w - eta * g for w, g in zip(Ws, grads)], lf.sequential(L))
    velocity = (W_next - W) / eta
    predicted = lf.arora_flow_direction(W, lf.grad_W(W, task), L)
    rel = (velocity - predicted).norm() / predicted.norm()
    assert rel < 5e-3, rel.item()
    # and it is NOT the parallel velocity once W has moved away from I
    rel_par = (velocity + L * lf.grad_W(W, task)).norm() / velocity.norm()
    assert rel_par > 10 * rel


def test_modewise_preconditioner():
    g = torch.Generator().manual_seed(3)
    W = torch.eye(4) + 0.8 * torch.randn(4, 4, generator=g)
    G = torch.randn(4, 4, generator=g)
    U, s, Vh = torch.linalg.svd(W)
    for L in (2, 3, 4):
        lhs = U.T @ lf.arora_flow_direction(W, G, L) @ Vh.T
        rhs = -L * lf.modewise_factor(s, L) * (U.T @ G @ Vh.T)
        torch.testing.assert_close(lhs, rhs, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(lf.modewise_factor(torch.ones(4), 3), torch.ones(4, 4))


# --- Prop. 4 -----------------------------------------------------------------


@pytest.mark.parametrize("L", [2, 3])
def test_closed_form_ntk(task, L):
    Ws = _balanced_trained(L, task)
    W = lf.end_to_end(Ws, lf.sequential(L))
    assert (W - torch.eye(W.shape[0])).norm() > 0.5  # far enough from init to be a real test
    for a, b in [(0, 1), (2, 5), (3, 3)]:
        x, xp = task.X[a], task.X[b]
        K_emp = lf.ntk_empirical(Ws, lf.sequential(L), task.R, x, xp)
        K_cf = lf.ntk_closed_form_chain(W, task.R, x, xp, L)
        K_par = L * (task.R @ task.R.T) * (x @ xp)
        assert ((K_emp - K_cf).norm() / K_emp.norm()) < 5e-3
        assert ((K_emp - K_par).norm() / K_emp.norm()) > 0.2  # kernel really has moved


def test_parallel_ntk_is_constant(task):
    L, d = 3, task.X.shape[1]
    Ws = lf.small_random(L, d, 2.0, seed=5)
    x, xp = task.X[0], task.X[1]
    K = lf.ntk_empirical(Ws, lf.parallel(L), task.R, x, xp)
    torch.testing.assert_close(K, L * (task.R @ task.R.T) * (x @ xp))


# --- Prop. 5 -----------------------------------------------------------------


@pytest.mark.parametrize("L", [2, 3, 4])
def test_first_step_is_scalar_rescale(task, L):
    eta = 0.7
    d = task.X.shape[1]
    W_par = _single_example_first_step(lf.parallel(L), task, eta)
    dz_par = lf.logits(W_par, task) - lf.logits(torch.eye(d), task)
    for parents in lf.all_parent_vectors(L):
        W = _single_example_first_step(parents, task, eta)
        dz = lf.logits(W, task) - lf.logits(torch.eye(d), task)
        ratio = (dz * dz_par).sum() / (dz_par * dz_par).sum()
        torch.testing.assert_close(dz, ratio * dz_par, atol=1e-12, rtol=1e-10)


def _kappa(task, i):
    p = torch.softmax(task.R @ task.X[i] + task.c, dim=0)
    r = p.clone()
    r[task.Y[i]] -= 1.0
    return (r @ task.R @ task.X[i]).item()


@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("sign", [-1, 1])
def test_first_step_chain_ratio(task, L, sign):
    eta, d = 0.3, task.X.shape[1]
    kappas = [_kappa(task, i) for i in range(len(task.Y))]
    i = max(range(len(kappas)), key=lambda j: sign * kappas[j])  # most positive / most negative kappa
    kappa = kappas[i]
    assert sign * kappa > 0
    W_seq = _single_example_first_step(lf.sequential(L), task, eta, i)
    W_par = _single_example_first_step(lf.parallel(L), task, eta, i)
    z0 = lf.logits(torch.eye(d), task)
    dz_s, dz_p = lf.logits(W_seq, task) - z0, lf.logits(W_par, task) - z0
    ratio = ((dz_s * dz_p).sum() / (dz_p * dz_p).sum()).item()
    assert ratio == pytest.approx(lf.first_step_chain_ratio(L, eta, kappa), rel=1e-10)
    # small-step reading: bigger step on already-favored examples (kappa<0), smaller on disfavored
    assert (ratio > 1) == (kappa < 0)


def test_minibatch_first_step_is_not_a_rescale(task):
    """The only genuinely new first-step structure is the cross-example term for batches (rank G > 1)."""
    eta, d = 0.7, task.X.shape[1]
    grads = lf.param_grads(lf.zeros(2, d), lf.sequential(2), task, idx=slice(0, 16))
    W_s = lf.end_to_end([-eta * g for g in grads], lf.sequential(2))
    W_p = lf.end_to_end([-eta * g for g in grads], lf.parallel(2))
    z0 = lf.logits(torch.eye(d), task)
    dz_s, dz_p = lf.logits(W_s, task) - z0, lf.logits(W_p, task) - z0
    ratio = (dz_s * dz_p).sum() / (dz_p * dz_p).sum()
    assert (dz_s - ratio * dz_p).norm() / dz_s.norm() > 1e-3


# --- v0 fixtures for the matched pair -----------------------------------------


def test_bax_divergence(task):
    eta, d = 0.3, task.X.shape[1]
    gA, gB = lf.param_grads(lf.zeros(2, d), lf.sequential(2), task, idx=slice(0, 8))
    gA_p, gB_p = lf.param_grads(lf.zeros(2, d), lf.parallel(2), task, idx=slice(0, 8))
    torch.testing.assert_close(gA, gA_p)  # identical first gradients
    torch.testing.assert_close(gB, gB_p)
    dA, dB = -eta * gA, -eta * gB
    zC = lf.logits(lf.end_to_end([dA, dB], lf.sequential(2)), task)
    zP = lf.logits(lf.end_to_end([dA, dB], lf.parallel(2)), task)
    torch.testing.assert_close(zC - zP, task.X @ (task.R @ dB @ dA).T)


def test_hessian_difference_is_residual_term():
    task = lf.make_teacher_task(d=2, V=2, n=5, seed=4)
    d = 2
    theta0 = torch.zeros(2 * d * d)

    def unflat(th):
        return [th[: d * d].reshape(d, d), th[d * d :].reshape(d, d)]

    def loss(parents):
        return lambda th: lf.ce_loss(lf.end_to_end(unflat(th), parents), task)

    H_C = torch.autograd.functional.hessian(loss(lf.sequential(2)), theta0)
    H_P = torch.autograd.functional.hessian(loss(lf.parallel(2)), theta0)
    # residual-weighted logit-curvature term of the composed graph, residuals frozen at theta=0
    p = torch.softmax(lf.logits(torch.eye(d), task), dim=1)
    r = p.clone()
    r[torch.arange(len(task.Y)), task.Y] -= 1.0

    def frozen_residual_term(th):
        return (r * lf.logits(lf.end_to_end(unflat(th), lf.sequential(2)), task)).sum() / len(task.Y)

    H_res = torch.autograd.functional.hessian(frozen_residual_term, theta0)
    torch.testing.assert_close(H_C - H_P, H_res)
    assert H_res.abs().max() > 1e-3


# --- Prop. 6 -----------------------------------------------------------------


@pytest.mark.slow
def test_nonseparable_endpoints_coincide():
    task = lf.make_teacher_task(d=4, V=3, n=400, teacher_scale=1.0, seed=2)
    probes = torch.randn(50, 4, generator=torch.Generator().manual_seed(9))
    W_par = lf.end_to_end(lf.train(lf.zeros(2, 4), lf.parallel(2), task, steps=6000, lr=1.0), lf.parallel(2))
    W_seq = lf.end_to_end(lf.train(lf.zeros(2, 4), lf.sequential(2), task, steps=6000, lr=1.0), lf.sequential(2))
    for W in (W_par, W_seq):
        assert lf.grad_W(W, task).norm() < 1e-6  # both converged
    zp = lf.centered(lf.logits(W_par, task, probes))
    zs = lf.centered(lf.logits(W_seq, task, probes))
    assert (zp - zs).norm() / zp.norm() < 1e-4


# --- §8.3 ---------------------------------------------------------------------


def test_weight_decay_breaks_balancedness(task):
    d, eta, steps = task.X.shape[1], 0.005, 800
    no_wd = lf.train(lf.zeros(2, d), lf.sequential(2), task, steps=steps, lr=eta)
    wd = lf.train(lf.zeros(2, d), lf.sequential(2), task, steps=steps, lr=eta, weight_decay=0.2)
    r0 = lf.balancedness_residuals(no_wd)[0].norm()
    r1 = lf.balancedness_residuals(wd)[0].norm()
    assert r1 > 10 * r0, (r0.item(), r1.item())


# --- Corollary 3a ---------------------------------------------------------------


def _reversal_task(seed=0):
    """V = d, R = I; teacher needs one direction flipped (det M* < 0)."""
    g = torch.Generator().manual_seed(seed)
    d = 4
    X = torch.randn(600, d, generator=g)
    M = torch.diag(torch.tensor([-3.0, 2.0, 2.0, 2.0]))
    Y = torch.multinomial(torch.softmax(X @ M.T, 1), 1, generator=g).squeeze(1)
    return lf.Task(X, Y, torch.eye(d), torch.zeros(d))


def test_chain_stays_orientation_preserving():
    task = _reversal_task()
    for parents in (lf.sequential(2), lf.sequential(3), lf.parallel(2)):
        dets = []
        lf.train(lf.zeros(len(parents), 4), parents, task, steps=1500, lr=0.05,
                 callback=lambda t, Ws: dets.append(torch.linalg.det(lf.end_to_end(Ws, parents)).item()))
        if parents == lf.parallel(2):
            assert min(dets) < 0  # the parallel graph does cross det = 0 on this task
        else:
            assert min(dets) > 0  # the chain never does (Corollary 3a)


# --- Prop. 7 reduction -------------------------------------------------------------


def test_mixed_graph_reduces_to_chain_plus_shortcut(task):
    parents = (0, 1, 0, 2)
    d = task.X.shape[1]
    Ws = lf.train(lf.zeros(4, d), parents, task, steps=1500, lr=0.02)
    chain = [Ws[0], Ws[1], Ws[3]]
    P = lf.end_to_end(chain, lf.sequential(3))
    W = lf.end_to_end(Ws, parents)
    torch.testing.assert_close(W, P + Ws[2])  # exact decomposition
    assert max(r.norm().item() for r in lf.balancedness_residuals(chain)) < 5e-3  # chain stays balanced
    # P's velocity is the L=3 Arora flow driven by G(W); W3's velocity is -G(W)
    eta = 1e-7
    grads = lf.param_grads(Ws, parents, task)
    nxt = [w - eta * g for w, g in zip(Ws, grads)]
    P_vel = (lf.end_to_end([nxt[0], nxt[1], nxt[3]], lf.sequential(3)) - P) / eta
    G = lf.grad_W(W, task)
    pred = lf.arora_flow_direction(P, G, 3)
    assert (P_vel - pred).norm() / pred.norm() < 5e-3
    torch.testing.assert_close(grads[2], G)


# --- §8.3 derivative -----------------------------------------------------------------


def test_weight_decay_balancedness_derivative(task):
    d, lam, eta = task.X.shape[1], 0.3, 1e-7
    A, B = lf.small_random(2, d, 0.5, seed=11)
    I = torch.eye(d)

    def D(A, B):
        return (I + B).T @ (I + B) - (I + A) @ (I + A).T

    gA, gB = lf.param_grads([A, B], lf.sequential(2), task)
    A1, B1 = A - eta * (gA + lam * A), B - eta * (gB + lam * B)  # coupled L2, as torch SGD weight_decay
    rate = (D(A1, B1) - D(A, B)) / eta
    F1, F2 = I + A, I + B
    pred = lam * (F2 + F2.T - F1 - F1.T) - 2 * lam * D(A, B)
    assert (rate - pred).norm() / pred.norm() < 1e-5
