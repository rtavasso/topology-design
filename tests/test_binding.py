"""What binding is: which parts of a gate over a bank the trained worlds leave undetermined (history Entry 121).

Exact constructions on the worlds teacher, no training. The bank holds the six primitives exactly (block p computes
tanh(A_p h)); a block's bid is lam * (w_p . h) + beta_p; u[k, j] says how much item j excites block k. Three gates:

  pooled        one softmax over blocks of u[k, a] + u[k, b] + bid_k(h)                 (the `streams_self` form)
  per item      each item's load is a softmax over blocks; the items compete by bids    (the `tok_streams` form)
  pooled, norm  one softmax over blocks of log(load_a[k] + load_b[k]) + bid_k(h)        (routing, with each item's
                                                                                         address normalised first)
Predictions (written before the first run): B1-B4 hold as stated; they are consequences of where u's cross terms
enter. In B1 the trained-world error stays below 1e-6 while one held-out world's error passes 0.05.

First run (20 Nov): all five failed, B0 included. With bids of sharpness 50 the exact bank is not exact: near a tie
between the two triggers the softmax blends the two primitives, an error floor of 0.002-0.004 that my absolute
thresholds of 1e-6 ignored. The run therefore said nothing about the structural statements. Second version, below:
sharper bids, and "unchanged" is tested as a difference from the ideal setting's error, not as an absolute size.
Offsets are in units of the bid sharpness (an offset of 6 is six trigger units: it decides every state).
Second run: B0 and B4 failed for the normalised pooled gate only - my guard against log 0 (adding 1e-300) capped
the penalty on an unloaded block at -690, below the bids, so unloaded blocks won. A bug in the construction; removed.
Third run: B1-B4 held. B0's bound of 1e-3 on the ideal setting failed at 0.0022 (one world; the rest 1e-7 to 1e-5):
one state in 512 passes within 1/LAM of a tie. B0 now bounds the worst world at 1e-2, the median at 1e-4, and
requires the three gates to agree with each other to 1e-9 at the ideal setting.
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "worlds"))
from worlds import DEF, Teacher, sample_worlds  # noqa: E402

C = dict(DEF, task="select")
P, D, N = C["P"], C["d"], C["n"]
LAM, SELF = 2e3, 1e6
OFF = 6 * LAM


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
                g = torch.softmax(torch.log(la + lb) + bid, -1)                 # log 0 = -inf: an unloaded block cannot win
        h = h + T.eps * (g[..., None] * f).sum(2)
    tgt = T.run(worlds, x, N)
    return ((h - tgt) ** 2).sum((1, 2)) / ((tgt - x) ** 2).sum((1, 2))


def _ideal():
    return SELF * torch.eye(P), torch.zeros(P)


def test_B0_all_three_gates_reproduce_the_teacher_at_the_ideal_setting():
    T, tr, te, x = _setup()
    u, beta = _ideal()
    allw = torch.cat([tr, te])
    e = {gate: _run(T, allw, x, gate, u, beta) for gate in ("pooled", "item", "norm")}
    assert e["pooled"].max().item() < 1e-2            # floor: a state within 1/LAM of a tie is blended (about 1 in 512)
    assert e["pooled"].median().item() < 1e-4
    assert (e["pooled"] - e["item"]).abs().max().item() < 1e-9 and (e["item"] - e["norm"]).abs().max().item() < 1e-9


def test_B1_pooled_gate_each_held_out_world_has_an_offset_that_no_trained_world_sees():
    T, tr, te, x = _setup()
    u0, beta = _ideal()
    seen0, held0 = _run(T, tr, x, "pooled", u0, beta), _run(T, te, x, "pooled", u0, beta)
    for i, (a, b) in enumerate(te.tolist()):
        u = u0.clone()
        u[a, b] += OFF; u[b, a] -= OFF                # cross terms: how much item b excites a's block, and the reverse
        assert (_run(T, tr, x, "pooled", u, beta) - seen0).abs().max().item() < 1e-9   # every trained world: unchanged
        held = _run(T, te, x, "pooled", u, beta)
        assert held[i].item() > 0.05                                               # this held-out world: now wrong
        d = (held - held0).abs(); d[i] = 0
        assert d.max().item() < 1e-9                                               # and only this one


def test_B2_normalising_each_items_address_removes_that_freedom_with_or_without_a_single_softmax():
    T, tr, te, x = _setup()
    u0, beta = _ideal()
    allw = torch.cat([tr, te])
    for gate in ("item", "norm"):
        base = _run(T, allw, x, gate, u0, beta)
        for a, b in te.tolist():
            u = u0.clone()
            u[a, b] += OFF; u[b, a] -= OFF
            assert (_run(T, allw, x, gate, u, beta) - base).abs().max().item() < 1e-9


def test_B3_what_is_left_is_one_bid_offset_per_primitive_and_connected_trained_worlds_fix_it():
    T, tr, te, x = _setup()
    u, _ = _ideal()
    g = torch.Generator().manual_seed(4)
    for gate in ("item", "norm"):
        for _ in range(5):
            beta = 1.2 * LAM * torch.randn(P, generator=g).double()
            assert _run(T, tr, x, gate, u, beta).max().item() > 1e-2               # any uneven offsets show in training
        allw = torch.cat([tr, te])
        assert (_run(T, allw, x, gate, u, torch.full((P,), 0.7 * LAM)) - _run(T, allw, x, gate, u, torch.zeros(P))).abs().max().item() < 1e-9   # a common constant does not


def test_B4_if_the_trained_worlds_fall_into_two_groups_the_offset_between_the_groups_is_free_again():
    T, _, _, x = _setup()
    tr = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])              # two triangles, not connected
    te = torch.tensor([[0, 3], [1, 4], [2, 5]])                                    # worlds across the two groups
    u, _ = _ideal()
    beta = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) * 1.6 * LAM
    for gate in ("item", "norm"):
        assert (_run(T, tr, x, gate, u, beta) - _run(T, tr, x, gate, u, torch.zeros(P))).abs().max().item() < 1e-9
        assert _run(T, te, x, gate, u, beta).min().item() > 0.02
