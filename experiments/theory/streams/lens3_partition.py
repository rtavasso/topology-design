"""Lens 3 - a gate that reads the state: a partition of state space, and where the router's signal lives. EXPLORATORY.

Structure. With g = softmax(V h / tau), the loop iterates one glued vector field F(h) = sum_k g_k(h) f_k(h): the gates are
a soft partition of unity over state space and each stream is the local rule on its region. The sequence of regions a
trajectory visits (the top gate at each pass) is its itinerary: the "program" this input runs. The pass Jacobian is
I + sum_k g_k df_k + (1 / tau) F C V with C = diag(g) - g g^T, which vanishes where the gate is decided and is large where
it is not.

Random streams (no training): d = 8, K = 4, tanh streams, step 0.5, 6 passes, 2,000 random inputs.

Predictions written before the run (15 Nov 2026):
  3.1 The typical (median) gain of the loop, sigma_max(d h_n / d h_0), is largest at an intermediate temperature
      (somewhere in 0.25 - 1), not at the sharpest gate.
  3.2 The tail grows as the gate sharpens: the ratio of the 99th percentile to the median of sigma_max at tau = 0.05 is
      more than three times the ratio at tau = 1.
  3.3 The router's training signal concentrates on inputs near a region boundary as the gate sharpens: the tenth of inputs
      with the smallest gate margin carries less than 25 % of the total gate-gradient norm at tau = 1 and more than 50 %
      at tau = 0.1.
  3.4 Itineraries are fragile near boundaries only: under an input perturbation of relative size 1e-2 the share of inputs
      whose itinerary changes is below 10 %, and at least 80 % of the changed ones are in the lowest-margin fifth.
"""
import torch
from torch.func import jacrev, vmap, grad

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
D, K, EPS, N, M = 8, 4, 0.5, 6, 2000


def make(seed):
    g = torch.Generator().manual_seed(seed)
    B = 1.5 * torch.randn(K, D, D, generator=g) / D ** 0.5
    V = torch.randn(K, D, generator=g) / D ** 0.5
    T = torch.randn(D, D, generator=g) / D ** 0.5
    x = torch.randn(M, D, generator=g)
    return B, V, T, x


def loop(h, B, V, tau):
    margins, picks = [], []
    for _ in range(N):
        s = V @ h
        top = torch.topk(s, 2).values
        margins.append(top[0] - top[1]); picks.append(s.argmax())
        gate = torch.softmax(s / tau, 0)
        h = h + EPS * torch.einsum("k,ki->i", gate, torch.tanh(B @ h))
    return h, torch.stack(margins).min(), torch.stack(picks)


if __name__ == "__main__":
    B, V, T, x = make(0)
    print("# Lens 3 — a gate that reads the state (EXPLORATORY)\n")
    print(f"Random tanh streams, d = {D}, K = {K}, step {EPS}, {N} passes, {M} inputs. Predictions in `lens3_partition.py`.\n")
    print("| temperature tau | sigma_max of the loop: median | 99th percentile | ratio | share of the gate-gradient norm from the tenth of inputs nearest a boundary | mean top gate |")
    print("|---|---|---|---|---|---|")
    for tau in (4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05):
        out = lambda h: loop(h, B, V, tau)[0]
        J = vmap(jacrev(out))(x)
        sig = torch.linalg.matrix_norm(J, 2)
        margin = vmap(lambda h: loop(h, B, V, tau)[1])(x)

        def sample_loss(Vp, h):
            return 0.5 * ((loop(h, B, Vp, tau)[0] - T @ h) ** 2).sum()
        gV = vmap(grad(sample_loss), in_dims=(None, 0))(V, x).flatten(1).norm(dim=1)
        order = margin.argsort()
        share = gV[order[: M // 10]].sum() / gV.sum()
        topg = vmap(lambda h: torch.softmax(V @ h / tau, 0).max())(x).mean()
        med, p99 = sig.median().item(), sig.quantile(0.99).item()
        print(f"| {tau} | {med:.2f} | {p99:.2f} | {p99 / med:.2f} | {share.item():.0%} | {topg.item():.2f} |", flush=True)
    print("\n**3.4** itineraries under an input perturbation of relative size 0.01 (tau = 0.25):\n")
    g = torch.Generator().manual_seed(9)
    delta = torch.randn(M, D, generator=g)
    delta = 0.01 * delta / delta.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True)
    it0 = vmap(lambda h: loop(h, B, V, 0.25)[2])(x)
    it1 = vmap(lambda h: loop(h, B, V, 0.25)[2])(x + delta)
    margin = vmap(lambda h: loop(h, B, V, 0.25)[1])(x)
    changed = (it0 != it1).any(1)
    low = torch.zeros(M, dtype=torch.bool); low[margin.argsort()[: M // 5]] = True
    print(f"- inputs whose itinerary changed: {changed.double().mean().item():.1%}; of those, in the lowest-margin fifth: {(changed & low).sum().item() / max(1, changed.sum().item()):.0%}")
    print(f"- distinct itineraries over {M} inputs: {len(set(map(tuple, it0.tolist())))} of {K ** N} possible")
