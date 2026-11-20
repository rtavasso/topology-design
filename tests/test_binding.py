"""What binding is: which parts of a gate over a bank the trained worlds leave undetermined (history Entry 121).

Exact constructions on the worlds teacher, no training. The bank holds the six primitives exactly (block p computes
tanh(A_p h)); a block's bid is lam * (w_p . h) + beta_p; u[k, j] says how much item j excites block k. Three gates:

  pooled        one softmax over blocks of u[k, a] + u[k, b] + bid_k(h)                 (the `streams_self` form)
  per item      each item's load is a softmax over blocks; the items compete by bids    (the `tok_streams` form)
  pooled, norm  one softmax over blocks of log(load_a[k] + load_b[k]) + bid_k(h)        (routing, with each item's
                                                                                         address normalised first)
Predictions (written before the first run): B1-B4 hold as stated; they are consequences of where u's cross terms
enter. In B1 the trained-world error stays below 1e-6 while one held-out world's error passes 0.05.
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "worlds"))
from worlds import DEF, Teacher, sample_worlds  # noqa: E402

C = dict(DEF, task="select")
P, D, N = C["P"], C["d"], C["n"]
LAM, SELF = 50.0, 2000.0


def _setup(split=0):
    T = Teacher(C)
    T.A, T.w = T.A.double(), T.w.double()
    tr, te = sample_worlds(P, C["cover"], split)
    x = torch.randn(1, 512, D, generator=torch.Generator().manual_seed(3)).double()
    return T, tr, te, x


def _run(T, worlds, x, gate, u, beta):
    """Per-world error of the exact bank under a gate, against the teacher. [M]"""
    x = x.expand(len(worlds), -1, -1)
    h = x
    for _ in range(N):
        f = torch.tanh(torch.einsum("pij,mbj->mbpi", T.A, h))                      # every block's output [M, B, P, d]
        bid = LAM * (h @ T.w.T) + beta                                             # [M, B, P]
        ua, ub = u[:, worlds[:, 0]].T[:, None, :], u[:, worlds[:, 1]].T[:, None, :]  # [M, 1, P]
        if gate == "pooled":
            g = torch.softmax(ua + ub + bid, -1)
        else:
            la, lb = torch.softmax(ua, -1), torch.softmax(ub, -1)                  # each item's load, normalised
            if gate == "item":
                share = torch.softmax(torch.stack([(la * bid).sum(-1), (lb * bid).sum(-1)], -1), -1)
                g = share[..., :1] * la + share[..., 1:] * lb
            else:
                g = torch.softmax(torch.log(la + lb + 1e-300) + bid, -1)
        h = h + T.eps * (g[..., None] * f).sum(2)
    tgt = T.run(worlds, x, N)
    return ((h - tgt) ** 2).sum((1, 2)) / ((tgt - x) ** 2).sum((1, 2))


def _ideal():
    return SELF * torch.eye(P), torch.zeros(P)


def test_B0_all_three_gates_reproduce_the_teacher_at_the_ideal_setting():
    T, tr, te, x = _setup()
    u, beta = _ideal()
    for gate in ("pooled", "item", "norm"):
        assert _run(T, torch.cat([tr, te]), x, gate, u, beta).max().item() < 1e-6


def test_B1_pooled_gate_each_held_out_world_has_an_offset_that_no_trained_world_sees():
    T, tr, te, x = _setup()
    u0, beta = _ideal()
    for i, (a, b) in enumerate(te.tolist()):
        u = u0.clone()
        u[a, b] += 300.0; u[b, a] -= 300.0            # cross terms: how much item b excites a's block, and the reverse
        assert _run(T, tr, x, "pooled", u, beta).max().item() < 1e-6                # every trained world: unchanged
        held = _run(T, te, x, "pooled", u, beta)
        assert held[i].item() > 0.05                                               # this held-out world: now wrong
        others = torch.cat([held[:i], held[i + 1:]])
        assert others.max().item() < 1e-6                                          # and only this one


def test_B2_normalising_each_items_address_removes_that_freedom_with_or_without_a_single_softmax():
    T, tr, te, x = _setup()
    u0, beta = _ideal()
    for gate in ("item", "norm"):
        for a, b in te.tolist():
            u = u0.clone()
            u[a, b] += 300.0; u[b, a] -= 300.0
            assert _run(T, torch.cat([tr, te]), x, gate, u, beta).max().item() < 1e-6


def test_B3_what_is_left_is_one_bid_offset_per_primitive_and_connected_trained_worlds_fix_it():
    T, tr, te, x = _setup()
    u, _ = _ideal()
    g = torch.Generator().manual_seed(4)
    for gate in ("item", "norm"):
        for _ in range(5):
            beta = 60.0 * torch.randn(P, generator=g).double()
            assert _run(T, tr, x, gate, u, beta).max().item() > 1e-3               # any uneven offsets show in training
        assert _run(T, torch.cat([tr, te]), x, gate, u, torch.full((P,), 37.0)).max().item() < 1e-6   # a common constant does not


def test_B4_if_the_trained_worlds_fall_into_two_groups_the_offset_between_the_groups_is_free_again():
    T, _, _, x = _setup()
    tr = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])              # two triangles, not connected
    te = torch.tensor([[0, 3], [1, 4], [2, 5]])                                    # worlds across the two groups
    u, _ = _ideal()
    beta = torch.tensor([0.0, 0.0, 0.0, 80.0, 80.0, 80.0])
    for gate in ("item", "norm"):
        assert _run(T, tr, x, gate, u, beta).max().item() < 1e-6
        assert _run(T, te, x, gate, u, beta).min().item() > 0.02
