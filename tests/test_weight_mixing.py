"""Exact statements about recombining streams in weight space rather than in output space (history Entry 110).

Streams are one-hidden-layer maps f_k(h) = V_k tanh(B_k h). Output mixing: sum_k g_k f_k(h).
Weight mixing: one block with V = sum_k g_k V_k and B = sum_k g_k B_k, i.e. V tanh(B h)."""
import torch

torch.set_default_dtype(torch.float64)
K, d, w = 3, 5, 7


def _streams(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, w, d, generator=g) / d ** 0.5, torch.randn(K, d, w, generator=g) / w ** 0.5, g


def weight_mix(B, V, gate, h):
    return torch.tanh(h @ torch.einsum("k,kwj->wj", gate, B).T) @ torch.einsum("k,kiw->iw", gate, V).T


def output_mix(B, V, gate, h):
    return torch.einsum("k,kbi->bi", gate, torch.einsum("kbw,kiw->kbi", torch.tanh(torch.einsum("kwj,bj->kbw", B, h)), V))


def test_W1_at_a_vertex_the_two_agree_and_for_linear_streams_they_agree_everywhere():
    B, V, g = _streams()
    h = torch.randn(4, d, generator=g)
    for k in range(K):
        e = torch.zeros(K); e[k] = 1.0
        assert torch.allclose(weight_mix(B, V, e, h), output_mix(B, V, e, h), atol=1e-13)
    gate = torch.tensor([0.5, 0.3, 0.2])
    assert (weight_mix(B, V, gate, h) - output_mix(B, V, gate, h)).abs().max().item() > 1e-2      # inside the simplex: not a blend
    A = torch.randn(K, d, d, generator=g)                                                      # linear streams A_k h
    assert torch.allclose(h @ torch.einsum("k,kij->ij", gate, A).T, torch.einsum("k,kbi->bi", gate, torch.einsum("kij,bj->kbi", A, h)), atol=1e-13)


def test_W2_every_stream_gets_the_same_gradient_direction_scaled_by_its_gate():
    # dL/dW_k = g_k * dL/dW for the mixed block's W: exact, because the mixed weights are linear in each stream's.
    B, V, g = _streams(1)
    B.requires_grad_(True); V.requires_grad_(True)
    h, y = torch.randn(6, d, generator=g), torch.randn(6, d, generator=g)
    gate = torch.tensor([0.6, 0.3, 0.1])
    ((weight_mix(B, V, gate, h) - y) ** 2).sum().backward()
    for G in (B.grad, V.grad):
        for k in range(1, K):
            assert torch.allclose(G[k] / gate[k], G[0] / gate[0], atol=1e-12)
    # in output mixing the directions differ: each stream is differentiated through its own features
    B2, V2 = B.detach().clone().requires_grad_(True), V.detach().clone().requires_grad_(True)
    ((output_mix(B2, V2, gate, h) - y) ** 2).sum().backward()
    cos = torch.nn.functional.cosine_similarity(V2.grad[0].flatten(), V2.grad[1].flatten(), dim=0).abs().item()
    assert cos < 0.9


def test_W3_the_gate_sees_a_stream_only_through_a_first_order_move_of_the_mixed_weights():
    # dL/dg_k = <dL/dB, B_k> + <dL/dV, V_k>, with dL/dB, dL/dV taken at the mixed block.
    B, V, g = _streams(2)
    h, y = torch.randn(6, d, generator=g), torch.randn(6, d, generator=g)
    gate = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
    Bm = torch.einsum("k,kwj->wj", gate, B); Vm = torch.einsum("k,kiw->iw", gate, V)
    Bm.retain_grad(); Vm.retain_grad()
    ((torch.tanh(h @ Bm.T) @ Vm.T - y) ** 2).sum().backward()
    want = torch.stack([(Bm.grad * B[k]).sum() + (Vm.grad * V[k]).sum() for k in range(K)])
    assert torch.allclose(gate.grad, want, atol=1e-12)


def test_W4_adding_weights_adds_functions_exactly_when_streams_use_disjoint_hidden_units():
    B, V, g = _streams(3)
    h = torch.randn(4, d, generator=g)
    Bd, Vd = torch.zeros_like(B), torch.zeros_like(V)
    for k, units in enumerate(([0, 1], [2, 3, 4], [5, 6])):
        Bd[k, units] = B[k, units]; Vd[k][:, units] = V[k][:, units]
    ones = torch.ones(K)
    assert torch.allclose(weight_mix(Bd, Vd, ones, h), output_mix(Bd, Vd, ones, h), atol=1e-13)
    assert (weight_mix(B, V, ones, h) - output_mix(B, V, ones, h)).abs().max().item() > 1e-1     # overlapping units: interference


def test_W5_a_soft_gate_acts_twice_so_its_effect_is_quadratic_for_small_activations():
    B, V, g = _streams(4)
    h = 1e-3 * torch.randn(4, d, generator=g)                       # small activations: tanh is linear
    e = torch.zeros(K); e[0] = 1.0
    full = output_mix(B, V, e, h)
    for gv in (0.5, 0.2):
        assert torch.allclose(weight_mix(B, V, gv * e, h), gv ** 2 * full, rtol=1e-4, atol=1e-12)
        assert torch.allclose(output_mix(B, V, gv * e, h), gv * full, rtol=1e-10, atol=1e-14)


def test_W6_small_deltas_on_a_shared_base_mix_almost_linearly():
    # W = W0 + sum_k g_k D_k with small D_k: the weight-mixed block equals the gate-weighted blend of the K blocks
    # W0 + D_k up to second order in the size of the deltas.
    B, V, g = _streams(5)
    B0, V0 = torch.randn(w, d, generator=g) / d ** 0.5, torch.randn(d, w, generator=g) / w ** 0.5
    h = torch.randn(4, d, generator=g)
    gate = torch.tensor([0.5, 0.3, 0.2])
    gaps = []
    for s in (1e-1, 1e-2):
        Bk, Vk = B0 + s * B, V0 + s * V
        gaps.append((weight_mix(Bk, Vk, gate, h) - output_mix(Bk, Vk, gate, h)).abs().max().item())
    assert gaps[1] < gaps[0] / 50                                    # ten times smaller deltas: about a hundred times closer


# ---- Entry 115: when is a weighted average of stored weights meaningful? ----------------------------------------
# Predictions (before the first run): all three are identities and hold to rounding; the "is not" halves differ by O(1).

def test_W7_averaging_write_matrices_over_a_shared_read_matrix_is_exact_averaging_read_matrices_is_not():
    B, V, g = _streams(7)
    h = torch.randn(4, d, generator=g)
    gate = torch.tensor([0.5, 0.3, 0.2])
    # same read matrix in every stream: the block is linear in its write matrix, so averaging V is averaging outputs
    Bsh = B[:1].expand(K, w, d)
    assert torch.allclose(weight_mix(Bsh, V, gate, h), output_mix(Bsh, V, gate, h), atol=1e-13)
    # same write matrix, different read matrices: the average goes through tanh
    Vsh = V[:1].expand(K, d, w)
    assert (weight_mix(B, Vsh, gate, h) - output_mix(B, Vsh, gate, h)).abs().max().item() > 1e-2


def test_W8_even_without_a_nonlinearity_averaging_two_factors_separately_is_not_averaging_the_maps():
    B, V, g = _streams(8)
    h = torch.randn(4, d, generator=g)
    gate = torch.tensor([0.5, 0.3, 0.2])
    maps = torch.einsum("kiw,kwj->kij", V, B)                                   # stream k is the linear map V_k B_k
    avg_of_maps = torch.einsum("k,kij->ij", gate, maps)
    factors = torch.einsum("k,kiw->iw", gate, V) @ torch.einsum("k,kwj->wj", gate, B)
    assert (factors - avg_of_maps).abs().max().item() > 1e-2                    # cross terms g_k g_l V_k B_l
    cross = torch.einsum("k,l,kiw,lwj->ij", gate, gate, V, B)
    assert torch.allclose(factors, cross, atol=1e-13)
    assert torch.allclose(h @ avg_of_maps.T, torch.einsum("k,kbi->bi", gate, torch.einsum("kij,bj->kbi", maps, h)), atol=1e-13)


def test_W9_stacking_units_is_averaging_write_matrices_written_over_one_common_set_of_read_rows():
    B, V, g = _streams(9)
    h = torch.randn(4, d, generator=g)
    R_all = B.reshape(K * w, d)                                                 # the union of every stream's read rows
    for gate in (torch.tensor([0.5, 0.3, 0.2]), torch.tensor([1.0, 1.0, 0.0]), torch.rand(K, generator=g)):
        W_pad = torch.zeros(K, d, K * w)
        for k in range(K):
            W_pad[k, :, k * w:(k + 1) * w] = V[k]                               # stream k's write matrix, zero elsewhere
        W_bar = torch.einsum("k,kin->in", gate, W_pad)                          # a plain weighted average of matrices
        assert torch.allclose(torch.tanh(h @ R_all.T) @ W_bar.T, output_mix(B, V, gate, h), atol=1e-13)
