"""Lens 4 - streams with separate states, coupled through a graph. EXPLORATORY.

Structure. Give each stream its own state X_k and couple them diffusively through a graph with Laplacian L:
    X_k <- X_k + eps * f_k(X_k) + c * sum_j W_kj (X_j - X_k)        i.e.  X <- X + eps F(X) - c L X.
c = 0 is K independent loops (an ensemble). A complete graph with strong coupling is the shared residual stream. The
disagreement between streams lives in the non-constant eigenvectors of L; mode i is damped by (1 - c lambda_i) per pass,
and the explicit update is stable only for c * lambda_max < 2.

Random tanh streams (no training): K = 8 streams, d = 8, step 0.3, 8 passes, all streams start from the same input.

Predictions written before the run (15 Nov 2026):
  4.1 Diversity (root-mean-square distance of the stream states from their mean after 8 passes) is ordered
      none > ring > star > complete at every coupling, and for the complete graph it equals the uncoupled diversity times
      (1 - (1 - cK)^n) / (cKn) to within 30 %.
  4.2 Influence has a light cone: on the ring, the state of stream k after n passes does not depend on the parameters of
      stream j at all (derivative exactly zero) when n < 1 + distance(j, k); on the complete graph every pair is coupled
      from n = 2.
  4.3 Stability threshold: with the complete graph (lambda_max = 8) diversity explodes at c = 0.3 (> 2 / 8) and not at
      c = 0.2; the ring (lambda_max = 4) is stable at both.
  4.4 Coupling damps the loop's sensitivity only in the disagreement directions: the gain for a perturbation applied
      equally to all streams is unchanged by the coupling (within 10 %), while the gain for a perturbation of one stream
      relative to the rest falls with c.
"""
import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
K, D, EPS, N = 8, 8, 0.3, 8


def graphs():
    none = torch.zeros(K, K)
    ring = torch.zeros(K, K)
    for i in range(K):
        ring[i, (i + 1) % K] = ring[(i + 1) % K, i] = 1
    star = torch.zeros(K, K); star[0, 1:] = 1; star[1:, 0] = 1
    comp = torch.ones(K, K) - torch.eye(K)
    return {"none": none, "ring": ring, "star": star, "complete": comp}


def lap(W):
    return torch.diag(W.sum(1)) - W


def run(B, x, L, c, n=N):
    X = x.unsqueeze(1).expand(-1, K, -1)                                  # [M, K, D]
    for _ in range(n):
        X = X + EPS * torch.tanh(torch.einsum("kij,mkj->mki", B, X)) - c * torch.einsum("kj,mjd->mkd", L, X)
    return X


def diversity(X):
    return (X - X.mean(1, keepdim=True)).pow(2).sum((1, 2)).div(K).sqrt().mean().item()


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    B = torch.randn(K, D, D, generator=g) / D ** 0.5
    x = torch.randn(500, D, generator=g)
    G = graphs()
    print("# Lens 4 — separate stream states coupled through a graph (EXPLORATORY)\n")
    print(f"Random tanh streams, K = {K}, d = {D}, step {EPS}, {N} passes. Predictions in `lens4_coupling.py`.\n")
    eig = {k: torch.linalg.eigvalsh(lap(W)) for k, W in G.items()}
    print("Laplacian eigenvalues (second smallest, largest): " + "; ".join(f"{k} {e[1].item():.2f}, {e[-1].item():.2f}" for k, e in eig.items()) + ".\n")
    print("**4.1, 4.3** diversity after 8 passes:\n")
    print("| coupling c | none | ring | star | complete | complete, predicted from the uncoupled value |")
    print("|---|---|---|---|---|---|")
    d0 = diversity(run(B, x, lap(G["none"]), 0.0))
    for c in (0.02, 0.05, 0.1, 0.2, 0.3):
        row = [diversity(run(B, x, lap(W), c)) for W in G.values()]
        pred = d0 * (1 - (1 - c * K) ** N) / (c * K * N)
        print(f"| {c} | " + " | ".join(f"{v:.3g}" for v in row) + f" | {pred:.3g} |", flush=True)
    print("\n**4.2** does stream k's state after n passes depend on stream j's parameters? (ring: j = 0; entries are the largest absolute derivative)\n")
    xs = x[:4]
    for name in ("ring", "complete"):
        L = lap(G[name])
        print(f"- {name}: ", end="")
        cells = []
        for n in (1, 2, 3, 4, 5):
            Bp = B.clone().requires_grad_(True)
            X = run(Bp, xs, L, 0.1, n)
            vals = []
            for k in (0, 1, 2, 3, 4):
                gr = torch.autograd.grad(X[:, k].pow(2).sum(), Bp, retain_graph=True)[0][0]
                vals.append(gr.abs().max().item())
            cells.append(f"n={n}: " + ", ".join("0" if v == 0 else f"{v:.0e}" for v in vals))
        print(" | ".join(cells) + "   (streams k = 0, 1, 2, 3, 4)")
    print("\n**4.4** gain of the loop for a small perturbation of the input states (complete graph):\n")
    print("| coupling c | perturbation applied to all streams equally | perturbation of one stream against the rest |")
    print("|---|---|---|")
    g2 = torch.Generator().manual_seed(1)
    dv = torch.randn(D, generator=g2); dv = 1e-5 * dv / dv.norm()
    for c in (0.0, 0.02, 0.05, 0.1, 0.2):
        L = lap(G["complete"])

        def roll(X0):
            X = X0
            for _ in range(N):
                X = X + EPS * torch.tanh(torch.einsum("kij,mkj->mki", B, X)) - c * torch.einsum("kj,mjd->mkd", L, X)
            return X
        X0 = x[:200].unsqueeze(1).expand(-1, K, -1).clone()
        base = roll(X0)
        common = (roll(X0 + dv) - base).pow(2).sum((1, 2)).sqrt() / (math.sqrt(K) * 1e-5)
        e = torch.zeros(K, 1); e[0] = 1; e = e - 1 / K
        rel = (roll(X0 + e * dv) - base)
        rel = (rel - rel.mean(1, keepdim=True)).pow(2).sum((1, 2)).sqrt() / (e.norm() * 1e-5)
        print(f"| {c} | {common.mean().item():.2f} | {rel.mean().item():.2f} |", flush=True)
