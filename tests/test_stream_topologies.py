"""Exact statements behind docs/stream_topologies.md (the numerical explorations themselves live in
experiments/theory/streams/ and are not tests)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "streams"))
from lib import all_programs, loss_fn, nmse, run_student, run_teacher, teacher_prims  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_identical_streams_stay_identical_under_gradient_steps():
    g = torch.Generator().manual_seed(0)
    d, K, P, n, eps = 5, 3, 2, 3, 0.3
    A = teacher_prims(P, d, 0)
    progs = all_programs(P, n)
    B = (0.1 * torch.randn(1, d, d, generator=g)).repeat(K, 1, 1).requires_grad_(True)
    E = torch.zeros(P, K, requires_grad=True)
    opt = torch.optim.SGD([B, E], lr=0.05)
    for _ in range(50):
        x = torch.randn(len(progs), 16, d, generator=g)
        loss = loss_fn(run_student(B, torch.softmax(E, 1)[progs], x, eps), run_teacher(A, progs, x, eps), x)
        opt.zero_grad(); loss.backward(); opt.step()
    assert (B - B[:1]).abs().max().item() < 1e-12
    assert (E - E[:, :1]).abs().max().item() < 1e-12


def test_gauge_change_of_streams_and_unnormalised_gates_leaves_outputs_unchanged():
    g = torch.Generator().manual_seed(1)
    d, K, P, n, eps = 5, 4, 3, 3, 0.3
    B, E, G = torch.randn(K, d, d, generator=g), torch.randn(P, K, generator=g), torch.randn(K, K, generator=g)
    progs = all_programs(P, n)
    x = torch.randn(len(progs), 4, d, generator=g)
    out1 = run_student(B, E[progs], x, eps)
    out2 = run_student(torch.einsum("kj,jab->kab", torch.linalg.inv(G), B), (E @ G)[progs], x, eps)
    assert torch.allclose(out1, out2, atol=1e-9)


def test_two_orders_differ_by_the_commutator_and_a_constant_gate_is_blind_to_order():
    g = torch.Generator().manual_seed(2)
    d, eps = 5, 0.2
    A, Bm = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    I = torch.eye(d)
    assert torch.allclose((I + eps * A) @ (I + eps * Bm) - (I + eps * Bm) @ (I + eps * A), eps ** 2 * (A @ Bm - Bm @ A), atol=1e-12)
    streams = torch.stack([A, Bm])
    x = torch.randn(2, 8, d, generator=g)
    progs = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])                   # the same tokens in two orders
    x[1] = x[0]
    gates = torch.nn.functional.one_hot(progs, 2).double().mean(1, keepdim=True).expand(-1, 4, -1)   # averaged over the program
    out = run_student(streams, gates, x, eps)
    assert torch.allclose(out[0], out[1], atol=1e-12)
    assert not torch.allclose(run_teacher(streams, progs, x, eps)[0], run_teacher(streams, progs, x, eps)[1], atol=1e-6)


def test_ring_coupling_has_a_light_cone():
    import lens4_coupling as L4
    g = torch.Generator().manual_seed(3)
    B = (torch.randn(L4.K, L4.D, L4.D, generator=g) / L4.D ** 0.5)
    x = torch.randn(3, L4.D, generator=g)
    L = L4.lap(L4.graphs()["ring"])
    for n in (1, 2, 3, 4):
        Bp = B.clone().requires_grad_(True)
        X = L4.run(Bp, x, L, 0.1, n)
        for k in range(5):                                               # distance from stream 0 on the ring is k
            gr = torch.autograd.grad(X[:, k].pow(2).sum(), Bp, retain_graph=True)[0][0]
            assert (gr.abs().max().item() == 0.0) == (n < 1 + k), (n, k)


def test_a_state_reading_gate_can_run_a_stored_program_exactly():
    d, P, K, N, eps = 4, 2, 2, 4, 0.4
    A = teacher_prims(P, d, 10)
    T = N * P
    Df = d + T
    S = torch.zeros(T, T)
    for j in range(T - P):
        S[j, j + P] = 1.0
    B = torch.zeros(K, Df, Df)
    V = torch.zeros(K, Df)
    for k in range(K):
        B[k, :d, :d] = A[k]                                              # the stream's own operation on the data
        B[k, d:, d:] = (S - torch.eye(T)) / eps                          # every stream also advances the tape
        V[k, d + k] = 40.0                                               # the gate reads the first slot of the tape
    progs = all_programs(P, N)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(len(progs), 16, d, generator=g)
    tape = torch.nn.functional.one_hot(progs, P).reshape(len(progs), T).double()
    s = torch.cat([x, tape[:, None, :].expand(-1, 16, -1)], 2)
    for _ in range(N):
        f = torch.einsum("kij,mbj->mbki", B, s)
        gate = torch.softmax(torch.einsum("kj,mbj->mbk", V, s), 2)
        s = s + eps * torch.einsum("mbk,mbki->mbi", gate, f)
    assert nmse(s[:, :, :d], run_teacher(A, progs, x, eps), x) < 1e-20
