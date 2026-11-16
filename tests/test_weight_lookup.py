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


# ---- Entry 112: keys computed from the weights, one key head per site -------------------------------------------
# Predictions (written before the first run): U5, U6, U8 are identities and hold exactly. U7: with a linear key on the
# read row, one item's query can pick out a teacher primitive's six units from the other thirty in at most 10% of
# (teacher, primitive) cases, and never all six primitives of a teacher; adding the write column does not change
# "never all six"; a key on the outer product v b^T always can (the features are linearly independent: a table).

def _two_site(Bs1, Vs1, Bs2, Vs2, h):
    h1 = h + torch.tanh(h @ Bs1.T) @ Vs1.T
    return h1 + torch.tanh(h1 @ Bs2.T) @ Vs2.T


def test_U5_sites_can_be_retrieved_from_different_blocks_but_a_read_and_its_write_cannot():
    # K blocks, each with two sites (two hidden layers in sequence on the same state). A site is one tied group.
    K, w = 3, 7
    g = torch.Generator().manual_seed(5)
    B1, V1 = torch.randn(K, w, d, generator=g) / d ** 0.5, torch.randn(K, d, w, generator=g) / w ** 0.5
    B2, V2 = torch.randn(K, w, d, generator=g) / d ** 0.5, torch.randn(K, d, w, generator=g) / w ** 0.5
    h = torch.randn(4, d, generator=g)
    p1 = [torch.randperm(w, generator=g) for _ in range(K)]; p2 = [torch.randperm(w, generator=g) for _ in range(K)]
    outs = []
    for k in range(K):                       # site 1 from block k, site 2 from block l: K^2 assembled blocks
        for l in range(K):
            a = _two_site(B1[k], V1[k], B2[l], V2[l], h)
            b = _two_site(B1[k][p1[k]], V1[k][:, p1[k]], B2[l][p2[l]], V2[l][:, p2[l]], h)   # every source block relabelled, each site by itself
            assert torch.allclose(a, b, atol=1e-13)
            outs.append(a)
    assert min((outs[i] - outs[j]).abs().max().item() for i in range(len(outs)) for j in range(i)) > 1e-3   # K^2 different functions
    # inside one site the read and the write are tied: reads from block 0 with writes from block 1 depends on labelling
    a = torch.tanh(h @ B1[0].T) @ V1[1].T
    b = torch.tanh(h @ B1[0].T) @ V1[1][:, p1[1]].T
    assert (a - b).abs().max().item() > 1e-2


def test_U6_a_key_must_be_unchanged_by_the_units_symmetries_a_linear_key_is_not_a_product_key_is():
    B, V, g = _bank(6)
    h = torch.randn(4, d, generator=g)
    s = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    f = lambda B_, V_: torch.tanh(h @ B_.T) @ V_.T
    assert torch.allclose(f(B, V), f(s[:, None] * B, V * s), atol=1e-13)                  # flip a unit's row and column: same block
    dk = 4
    Pl = torch.randn(dk, 2 * d, generator=g)
    lin = lambda B_, V_: torch.cat([B_, V_.T], 1) @ Pl.T                                 # key = P [b; v]
    assert torch.allclose(lin(s[:, None] * B, V * s), s[:, None] * lin(B, V), atol=1e-13) and (s < 0).any()   # flips with the unit
    U, W = torch.randn(dk, d, generator=g), torch.randn(dk, d, generator=g)
    prod = lambda B_, V_: (B_ @ U.T) * (V_.T @ W.T)                                      # key = (U b) * (W v), a readout of v b^T
    assert torch.allclose(prod(s[:, None] * B, V * s), prod(B, V), atol=1e-13)
    assert (prod(B, V)[0] - prod(B, V)[1]).abs().max().item() > 1e-3                      # and still tells units apart


def test_U7_a_key_from_a_units_own_weights_addresses_by_similarity_and_a_primitives_units_are_not_similar():
    import numpy as np
    from scipy.optimize import linprog
    from worlds import Teacher

    def separable(X, y):                     # is there (s, t) with s.x - t >= 1 on y and <= -1 off y?
        A = np.concatenate([X, -np.ones((len(X), 1))], 1) * np.where(y, -1.0, 1.0)[:, None]
        r = linprog(np.zeros(A.shape[1]), A_ub=A, b_ub=-np.ones(len(X)), bounds=[(None, None)] * A.shape[1], method="highs")
        return r.status == 0

    P, dd = DEF["P"], DEF["d"]
    got = {"row": [], "row+col": [], "outer": []}
    for tseed in range(10, 20):
        A_ = Teacher(dict(DEF, tseed=tseed)).A.numpy()                                   # primitive p = tanh(A_p h): units (row i of A_p, e_i)
        rows = A_.reshape(P * dd, dd)
        cols = np.tile(np.eye(dd), (P, 1))
        outer = np.einsum("ni,nj->nij", cols, rows).reshape(P * dd, -1)
        grp = np.repeat(np.arange(P), dd)
        for name, X in (("row", rows), ("row+col", np.concatenate([rows, cols], 1)), ("outer", outer)):
            got[name].append([separable(X, grp == p) for p in range(P)])
    r = {k: np.array(v) for k, v in got.items()}
    print({k: (round(float(v.mean()), 3), int(v.all(1).sum())) for k, v in r.items()})
    assert r["row"].mean() <= 0.10 and r["row"].all(1).sum() == 0
    assert r["row+col"].all(1).sum() == 0
    assert r["outer"].all()


def test_U8_a_key_made_by_probing_what_the_site_computes_respects_every_symmetry_and_is_shared_by_the_blocks_units():
    B, V, g = _bank(8)
    perm = torch.randperm(N, generator=g)
    s = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    X = torch.randn(3, d, generator=g)                                                   # learned probe states
    key = lambda B_, V_: (torch.tanh(X @ B_.T) @ V_.T).reshape(-1)                       # the block's outputs at the probes
    assert torch.allclose(key(B, V), key((s[:, None] * B)[perm], (V * s)[:, perm]), atol=1e-13)
    B2, V2, _ = _bank(9)
    assert (key(B, V) - key(B2, V2)).abs().max().item() > 1e-2                            # different blocks, different keys
    # the pooled product key of U6 is the same thing to first order: a readout of the block's Jacobian at 0, V B
    U, W = torch.randn(4, d, generator=g), torch.randn(4, d, generator=g)
    pooled = ((B @ U.T) * (V.T @ W.T)).sum(0)
    assert torch.allclose(pooled, torch.einsum("ci,ij,cj->c", W, V @ B, U), atol=1e-12)
