"""Shared pieces for the stream-topology explorations (docs/stream_topologies.md). Everything is small and runs on a CPU.

The carrier. A state h in R^d; P instruction tokens; a program is a sequence u = (u_1..u_n) of tokens. The teacher applies
one fixed primitive per pass:      h <- h + eps * A[u_t] h        (linear), with A[i] random d x d.
A student has K streams B[k] and recombines them with a gate vector g_t:   h <- h + eps * sum_k g_t[k] * phi(B[k] h).
What the gate is allowed to see defines the variants (see each script).

Loss: squared error of the change h_n - h_0 relative to the teacher's change (so 1.0 = "learned nothing", 0 = exact).
"""
import itertools
import torch

torch.set_default_dtype(torch.float64)


def teacher_prims(P, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(P, d, d, generator=g) / d ** 0.5


def all_programs(P, n):
    return torch.tensor(list(itertools.product(range(P), repeat=n)))


def split(programs, frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(programs), generator=g)
    k = int(round(frac * len(programs)))
    return programs[perm[:k]], programs[perm[k:]]


def run_teacher(A, progs, x, eps):
    """progs [M, n], x [M, B, d] -> final states [M, B, d]."""
    h = x
    for t in range(progs.shape[1]):
        h = h + eps * torch.einsum("mij,mbj->mbi", A[progs[:, t]], h)
    return h


def run_student(B, gates, x, eps, phi=None):
    """gates [M, n, K] (already normalised as the variant wants); B [K, d, d]."""
    h = x
    for t in range(gates.shape[1]):
        f = torch.einsum("kij,mbj->mbki", B, h)
        if phi is not None:
            f = phi(f)
        h = h + eps * torch.einsum("mk,mbki->mbi", gates[:, t], f)
    return h


def nmse(h, target, x):
    return (((h - target) ** 2).sum() / ((target - x) ** 2).sum()).item()


def loss_fn(h, target, x):
    return ((h - target) ** 2).sum() / ((target - x) ** 2).sum()
