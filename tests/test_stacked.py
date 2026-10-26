"""Stacked (run-batched) models must equal independent models, and runs must not interact."""
import torch

from topology_design import stacked as S


def _slice_into(big, small, j):
    sd = {k: v[j:j + 1].clone() for k, v in big.state_dict().items()}
    small.load_state_dict(sd)


def _inputs(R, B=5, K=16, Q=4, p=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda T: torch.randint(0, p, (R, B, T), generator=g)
    return r(K), r(K), r(K), r(Q), r(Q)


def test_stacked_equals_independent_runs():
    torch.manual_seed(0)
    big = S.StackedPFN(R=3, d=32, layers=2, heads=4, mlp=64)
    cx, cy, cf, qx, qy = _inputs(3)
    out = big(cx, cy, cf, qx, qy)
    for j in range(3):
        small = S.StackedPFN(R=1, d=32, layers=2, heads=4, mlp=64)
        _slice_into(big, small, j)
        o1 = small(cx[j:j + 1], cy[j:j + 1], cf[j:j + 1], qx[j:j + 1], qy[j:j + 1])
        assert torch.allclose(out[j:j + 1], o1, atol=1e-5)


def test_runs_do_not_interact_in_gradients():
    big = S.StackedPFN(R=3, d=32, layers=2, heads=4, mlp=64)
    cx, cy, cf, qx, qy = _inputs(3)
    out = big(cx, cy, cf, qx, qy)
    out[0].sum().backward()  # loss of run 0 only
    for name, prm in big.named_parameters():
        assert prm.grad[1:].abs().max() == 0, name


def test_queries_do_not_see_each_other():
    big = S.StackedPFN(R=1, d=32, layers=2, heads=4, mlp=64)
    cx, cy, cf, qx, qy = _inputs(1, Q=4)
    a = big(cx, cy, cf, qx, qy)
    qx2 = qx.clone(); qx2[..., 1:] = (qx2[..., 1:] + 1) % 7  # change other queries only
    b = big(cx, cy, cf, qx2, qy)
    assert torch.allclose(a[..., 0, :], b[..., 0, :], atol=1e-6)
