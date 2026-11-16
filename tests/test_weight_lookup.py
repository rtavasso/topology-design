"""Exact statements behind docs/weight_lookup.md (history Entry 111): building one block by attention over a bank.

A unit is a pair (read row b_i, write column v_i); a block of units computes sum_i v_i tanh(b_i . h) = V tanh(B h)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "worlds"))
from worlds import DEF, Model, all_worlds  # noqa: E402

torch.set_default_dtype(torch.float64)
N, d = 12, 5


def _bank(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, d, generator=g) / d ** 0.5, torch.randn(d, N, generator=g) / N ** 0.5, g


def test_U1_assembling_the_matrices_from_retrieved_units_is_gating_units():
    # Retrieve units S with weights alpha. The assembled block has B' = rows b_i (i in S) and V' = columns alpha_i v_i.
    # It equals the whole bank with every unit gated by alpha (zero outside S): weight assembly = unit gating.
    B, V, g = _bank()
    h = torch.randn(4, d, generator=g)
    S = torch.tensor([1, 4, 5, 9]); alpha = torch.tensor([0.7, 0.2, 1.0, 0.4])
    assembled = torch.tanh(h @ B[S].T) @ (V[:, S] * alpha).T
    gate = torch.zeros(N); gate[S] = alpha
    assert torch.allclose(assembled, (torch.tanh(h @ B.T) * gate) @ V.T, atol=1e-13)
    # and loading two sets of units is exactly the sum of the two blocks (no interference)
    S2 = torch.tensor([0, 2, 7])
    both = torch.cat([S, S2])
    assert torch.allclose(torch.tanh(h @ B[both].T) @ V[:, both].T,
                          torch.tanh(h @ B[S].T) @ V[:, S].T + torch.tanh(h @ B[S2].T) @ V[:, S2].T, atol=1e-13)


def test_U2_units_respect_the_blocks_permutation_symmetry_and_whole_matrix_mixing_does_not():
    B, V, g = _bank(1)
    h = torch.randn(4, d, generator=g)
    perm = torch.randperm(N, generator=g)
    gate = torch.rand(N, generator=g)
    f = lambda B_, V_, g_: (torch.tanh(h @ B_.T) * g_) @ V_.T
    assert torch.allclose(f(B, V, gate), f(B[perm], V[:, perm], gate[perm]), atol=1e-13)      # relabel the bank: same block
    # two blocks; the second is the SAME function written with its units in another order
    B2, V2, _ = _bank(2)
    same = torch.allclose(torch.tanh(h @ B2.T) @ V2.T, torch.tanh(h @ B2[perm].T) @ V2[:, perm].T, atol=1e-13)
    mix = lambda Ba, Va, Bb, Vb: torch.tanh(h @ (0.5 * Ba + 0.5 * Bb).T) @ (0.5 * Va + 0.5 * Vb).T
    assert same and (mix(B, V, B2, V2) - mix(B, V, B2[perm], V2[:, perm])).abs().max().item() > 1e-2   # averaging matrices: not the same


def test_U3_blending_rows_inside_a_slot_brings_the_interference_back():
    # A slot that takes a soft mixture of two units is a new unit, not the mixture of the two units' outputs.
    B, V, g = _bank(3)
    h = torch.randn(4, d, generator=g)
    a = torch.tensor([0.6, 0.4])
    slot = torch.tanh(h @ (a[0] * B[0] + a[1] * B[1]))[:, None] * (a[0] * V[:, 0] + a[1] * V[:, 1])
    units = a[0] * torch.tanh(h @ B[0])[:, None] * V[:, 0] + a[1] * torch.tanh(h @ B[1])[:, None] * V[:, 1]
    assert (slot - units).abs().max().item() > 1e-2
    one_hot = torch.tanh(h @ (1.0 * B[0] + 0.0 * B[1]))[:, None] * (1.0 * V[:, 0])
    assert torch.allclose(one_hot, torch.tanh(h @ B[0])[:, None] * V[:, 0], atol=1e-14)        # a hard slot is just the unit


def test_U4_the_per_token_streams_arm_of_the_worlds_carrier_is_this_lookup_with_stream_sized_groups():
    # load:    each context item j puts weight a_jk on the units of stream k (address key shared within a stream)
    # compete: an item's bid is the a-weighted bid of its streams; pi = softmax over items
    # apply:   unit weight = sum_j pi_j a_jk; one block of all loaded units
    cfg = dict(DEF, arm="tok_streams")
    S = Model(cfg, 0)
    worlds = torch.tensor(all_worlds(cfg["P"]))[:5]
    x = torch.randn(len(worlds), 3, cfg["d"], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = S.forward(worlds, x, 1)
        p = S.p
        K, w = p["Bs"].shape[0], p["Bs"].shape[1]
        bank_B = p["Bs"].reshape(K * w, -1)                                   # read rows of all units
        bank_V = p["Vs"].permute(1, 0, 2).reshape(-1, K * w)                  # write columns of all units
        a = torch.softmax(p["E"][worlds] @ p["G"].T, -1)                      # [M, items, K]  load
        bid_k = torch.einsum("kq,mbkq->mbk", p["S2"], torch.tanh(torch.einsum("kqj,mbj->mbkq", p["S1"], x) + p["sb"]))
        pi = torch.softmax(torch.einsum("mjk,mbk->mbj", a, bid_k), -1)        # [M, B, items]  compete
        unit_w = torch.einsum("mbj,mjk->mbk", pi, a).repeat_interleave(w, dim=-1)        # [M, B, K*w]
        got = x + S.eps * (torch.tanh(x @ bank_B.T) * unit_w) @ bank_V.T
    assert torch.allclose(got, want, atol=1e-12)
