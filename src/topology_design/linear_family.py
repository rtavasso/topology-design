"""Linear parent-selection topologies, softmax-CE training, and closed-form Phase 0 predictions.

Graph family (docs/phase0_linear_theory.md):
    h_0 = x,  h_i = h_{i-1} + W_i h_{k_i},  0 <= k_i < i,  output h_L = W(theta) x.

Everything is float64 and small; this module is a calibration fixture, not a training framework.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

DTYPE = torch.float64
Parents = tuple[int, ...]


# ---------------------------------------------------------------------------
# Graph family
# ---------------------------------------------------------------------------


def all_parent_vectors(L: int) -> list[Parents]:
    """All L! labeled parent vectors (k_1..k_L) with 0 <= k_i < i (1-indexed i)."""
    return [tuple(p) for p in itertools.product(*[range(i) for i in range(1, L + 1)])]


def parallel(L: int) -> Parents:
    return tuple([0] * L)


def sequential(L: int) -> Parents:
    return tuple(range(L))


def end_to_end(Ws: Sequence[torch.Tensor], parents: Parents) -> torch.Tensor:
    """End-to-end matrix W(theta) such that h_L = W x, built by running the graph on I."""
    if len(Ws) != len(parents):
        raise ValueError("one parent index per learned matrix")
    d = Ws[0].shape[0]
    hs = [torch.eye(d, dtype=Ws[0].dtype)]
    for i, (Wi, k) in enumerate(zip(Ws, parents), start=1):
        if not 0 <= k < i:
            raise ValueError(f"parent k_{i}={k} must satisfy 0 <= k < {i}")
        hs.append(hs[i - 1] + Wi @ hs[k])
    return hs[-1]


# ---------------------------------------------------------------------------
# Softmax cross-entropy with a fixed readout
# ---------------------------------------------------------------------------


@dataclass
class Task:
    X: torch.Tensor  # (n, d)
    Y: torch.Tensor  # (n,) int64
    R: torch.Tensor  # (V, d) fixed readout
    c: torch.Tensor  # (V,) fixed bias


def logits(W: torch.Tensor, task: Task, X: torch.Tensor | None = None) -> torch.Tensor:
    X = task.X if X is None else X
    return X @ (task.R @ W).T + task.c


def ce_loss(W: torch.Tensor, task: Task, idx: torch.Tensor | slice | None = None) -> torch.Tensor:
    X, Y = (task.X, task.Y) if idx is None else (task.X[idx], task.Y[idx])
    return torch.nn.functional.cross_entropy(X @ (task.R @ W).T + task.c, Y)


def grad_W(W: torch.Tensor, task: Task, idx=None) -> torch.Tensor:
    """Analytic dL/dW = R^T E[(p - e_y) x^T] (mean over the selected examples)."""
    X, Y = (task.X, task.Y) if idx is None else (task.X[idx], task.Y[idx])
    p = torch.softmax(X @ (task.R @ W).T + task.c, dim=1)
    r = p.clone()
    r[torch.arange(len(Y)), Y] -= 1.0
    return task.R.T @ (r.T @ X) / len(Y)


def centered(z: torch.Tensor) -> torch.Tensor:
    """Remove the softmax gauge (common logit shift)."""
    return z - z.mean(dim=-1, keepdim=True)


def make_teacher_task(
    d: int = 6,
    V: int = 4,
    n: int = 200,
    teacher_scale: float = 1.0,
    seed: int = 0,
) -> Task:
    """Linear softmax teacher p*(y|x) = softmax(R M* x); labels sampled (non-separable for moderate scale)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(V, d, generator=g, dtype=DTYPE) / d**0.5
    X = torch.randn(n, d, generator=g, dtype=DTYPE)
    M = torch.eye(d, dtype=DTYPE) + teacher_scale * torch.randn(d, d, generator=g, dtype=DTYPE) / d**0.5
    p = torch.softmax(X @ (R @ M).T, dim=1)
    Y = torch.multinomial(p, 1, generator=g).squeeze(1)
    return Task(X=X, Y=Y, R=R, c=torch.zeros(V, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Parameter gradients and training
# ---------------------------------------------------------------------------


def param_grads(Ws: Sequence[torch.Tensor], parents: Parents, task: Task, idx=None) -> list[torch.Tensor]:
    leaves = [W.detach().clone().requires_grad_(True) for W in Ws]
    loss = ce_loss(end_to_end(leaves, parents), task, idx)
    return list(torch.autograd.grad(loss, leaves))


def zeros(L: int, d: int) -> list[torch.Tensor]:
    return [torch.zeros(d, d, dtype=DTYPE) for _ in range(L)]


def small_random(L: int, d: int, scale: float, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [scale * torch.randn(d, d, generator=g, dtype=DTYPE) / d**0.5 for _ in range(L)]


def train(
    Ws: Sequence[torch.Tensor],
    parents: Parents,
    task: Task,
    steps: int,
    lr: float,
    optimizer: str = "sgd",
    weight_decay: float = 0.0,
    callback: Callable[[int, list[torch.Tensor]], None] | None = None,
) -> list[torch.Tensor]:
    """Full-batch training. `weight_decay` is coupled L2 for SGD and decoupled for AdamW."""
    params = [W.detach().clone().requires_grad_(True) for W in Ws]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(optimizer)
    for t in range(steps):
        opt.zero_grad()
        ce_loss(end_to_end(params, parents), task).backward()
        opt.step()
        if callback is not None:
            callback(t, [p.detach() for p in params])
    return [p.detach().clone() for p in params]


# ---------------------------------------------------------------------------
# Closed-form predictions (docs/phase0_linear_theory.md)
# ---------------------------------------------------------------------------


def psd_power(S: torch.Tensor, a: float) -> torch.Tensor:
    """S^a for symmetric PSD S via eigendecomposition (eigenvalues clamped at 0)."""
    evals, evecs = torch.linalg.eigh((S + S.T) / 2)
    return evecs @ torch.diag(evals.clamp_min(0.0) ** a) @ evecs.T


def balancedness_residuals(Ws: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """F_{i+1}^T F_{i+1} - F_i F_i^T for the sequential chain, F_i = I + W_i (Prop. 3)."""
    d = Ws[0].shape[0]
    F = [torch.eye(d, dtype=Ws[0].dtype) + W for W in Ws]
    return [F[i + 1].T @ F[i + 1] - F[i] @ F[i].T for i in range(len(F) - 1)]


def arora_flow_direction(W: torch.Tensor, G: torch.Tensor, L: int) -> torch.Tensor:
    """-sum_j (WW^T)^{(j-1)/L} G (W^TW)^{(L-j)/L}: end-to-end velocity per unit lr (Prop. 3)."""
    WWt, WtW = W @ W.T, W.T @ W
    return -sum(psd_power(WWt, (j - 1) / L) @ G @ psd_power(WtW, (L - j) / L) for j in range(1, L + 1))


def modewise_factor(s: torch.Tensor, L: int) -> torch.Tensor:
    """gamma_ab = (1/L) sum_j s_a^{2(j-1)/L} s_b^{2(L-j)/L}; parallel graph has gamma = 1."""
    sa, sb = s[:, None], s[None, :]
    return sum(sa ** (2 * (j - 1) / L) * sb ** (2 * (L - j) / L) for j in range(1, L + 1)) / L


def ntk_closed_form_chain(W: torch.Tensor, R: torch.Tensor, x: torch.Tensor, xp: torch.Tensor, L: int) -> torch.Tensor:
    """K(x,x') = sum_j R (WW^T)^{(L-j)/L} R^T * x^T (W^TW)^{(j-1)/L} x'   (balanced chain, Prop. 4)."""
    WWt, WtW = W @ W.T, W.T @ W
    return sum(
        (R @ psd_power(WWt, (L - j) / L) @ R.T) * (x @ psd_power(WtW, (j - 1) / L) @ xp) for j in range(1, L + 1)
    )


def ntk_empirical(Ws: Sequence[torch.Tensor], parents: Parents, R: torch.Tensor, x: torch.Tensor, xp: torch.Tensor):
    """Logit-space empirical NTK J(x) J(x')^T (V x V) by autograd."""
    def z(xx):
        return lambda *ws: R @ end_to_end(list(ws), parents) @ xx

    Jx = torch.autograd.functional.jacobian(z(x), tuple(Ws))
    Jxp = torch.autograd.functional.jacobian(z(xp), tuple(Ws))
    V = R.shape[0]
    return sum(a.reshape(V, -1) @ b.reshape(V, -1).T for a, b in zip(Jx, Jxp))


def first_step_chain_ratio(L: int, eta: float, kappa: float) -> float:
    """Sequential/parallel logit-change ratio after one single-example SGD step from zero (Prop. 5)."""
    if abs(eta * kappa) < 1e-12:
        return 1.0
    return (1.0 - (1.0 - eta * kappa) ** L) / (L * eta * kappa)
