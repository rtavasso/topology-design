"""Lens 1 - symmetry and gauge: streams as a dictionary of operators. EXPLORATORY.

Structure. The gated sum is unchanged by permuting the streams (the symmetric group S_K acts on (B, columns of the gate
table)). With linear streams and gates g_t = E[u_t], the model depends on (E, B) only through the effective operator of
each token, C[i] = sum_k E[i, k] B[k]: a factorisation of the map token -> operator through K "dictionary" operators.

Predictions written before the run (15 Nov 2026):
  1.1 (exact)  Streams that start identical stay identical under training: the symmetric subspace is invariant.
  1.2 (exact)  With unnormalised gates, replacing E by E G and B by G^-1 B (any invertible G) leaves every output unchanged.
  1.3 With softmax gates and K = P + 1 streams, training on 75 % of the programs reaches held-out error < 0.01 and effective
      operators within 10 % of the teacher's, while the gates stay soft (mean top gate < 0.9) and no stream matches a
      teacher primitive (best cosine < 0.95): composition generalises without the streams being the primitives.
  1.4 The number of streams needed is set by the affine dimension of the teacher's primitives, not their count: four
      primitives on a line segment are learned by K = 2 streams (held-out error < 0.01); four generic primitives are not
      learned by K = 2 (> 0.2) or K = 3 (> 0.05) and are by K = 4 (< 0.01).
"""
import torch
from lib import *

D, EPS, N, STEPS, LR, BATCH = 6, 0.3, 4, 4000, 1e-2, 32


def train(A, K, seed, steps=STEPS, sym_init=False, record=False):
    P = A.shape[0]
    progs = all_programs(P, N)
    tr, te = split(progs, 0.75, seed)
    g = torch.Generator().manual_seed(100 + seed)
    B = (0.1 * torch.randn(K, D, D, generator=g) / D ** 0.5)
    if sym_init:
        B = B[:1].repeat(K, 1, 1)
    E = torch.zeros(P, K) if sym_init else 0.1 * torch.randn(P, K, generator=g)
    B.requires_grad_(True); E.requires_grad_(True)
    opt = torch.optim.Adam([B, E], lr=LR)
    for it in range(steps):
        x = torch.randn(len(tr), BATCH, D, generator=g)
        target = run_teacher(A, tr, x, EPS)
        h = run_student(B, torch.softmax(E, 1)[tr], x, EPS)
        loss = loss_fn(h, target, x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gt = torch.softmax(E, 1)
        x = torch.randn(len(te), 256, D, generator=g)
        test = nmse(run_student(B, gt[te], x, EPS), run_teacher(A, te, x, EPS), x)
        x = torch.randn(len(tr), 256, D, generator=g)
        trn = nmse(run_student(B, gt[tr], x, EPS), run_teacher(A, tr, x, EPS), x)
        C = torch.einsum("ik,kab->iab", gt, B)
        op_err = ((C - A).flatten(1).norm(dim=1) / A.flatten(1).norm(dim=1)).mean().item()
        cos = torch.nn.functional.cosine_similarity(B.flatten(1)[:, None, :], A.flatten(1)[None, :, :], dim=2)
        return dict(train=trn, test=test, op_err=op_err, top_gate=gt.max(1).values.mean().item(),
                    best_cos=cos.abs().max().item(), B=B.detach(), E=E.detach())


if __name__ == "__main__":
    print("# Lens 1 — symmetry and gauge (EXPLORATORY)\n")
    print(f"Linear streams, d = {D}, step {EPS}, {N} passes, Adam {LR}, {STEPS} steps, 75 % of programs for training. Predictions in `lens1_symmetry.py`.\n")
    A3 = teacher_prims(3, D, 0)
    r = train(A3, 4, 0, steps=600, sym_init=True)
    print(f"**1.1** identical start, 600 steps: largest difference between any two streams = {(r['B'] - r['B'][:1]).abs().max().item():.1e}; held-out error {r['test']:.3f} (one effective stream).\n")
    g = torch.Generator().manual_seed(5)
    B, E, G = torch.randn(4, D, D, generator=g), torch.randn(3, 4, generator=g), torch.randn(4, 4, generator=g)
    progs = all_programs(3, N); x = torch.randn(len(progs), 8, D, generator=g)
    out1 = run_student(B, E[progs], x, EPS)
    out2 = run_student(torch.einsum("kj,jab->kab", torch.linalg.inv(G), B), (E @ G)[progs], x, EPS)
    print(f"**1.2** gauge change by a random invertible 4 × 4 matrix: largest change in any output = {(out1 - out2).abs().max().item():.1e}.\n")
    print("**1.3** three generic primitives, K = 4 streams, softmax gates, five seeds:\n")
    print("| seed | training error | held-out error | effective-operator error | mean top gate | best stream–primitive cosine |")
    print("|---|---|---|---|---|---|")
    for seed in range(5):
        r = train(A3, 4, seed)
        print(f"| {seed} | {r['train']:.1e} | {r['test']:.1e} | {r['op_err']:.3f} | {r['top_gate']:.2f} | {r['best_cos']:.2f} |", flush=True)
    print("\n**1.4** four primitives; held-out error by number of streams (three seeds each, median [min, max]):\n")
    print("| primitives | K = 2 | K = 3 | K = 4 |")
    print("|---|---|---|---|")
    A0 = teacher_prims(2, D, 1)
    seg = torch.stack([A0[0] + c * (A0[1] - A0[0]) for c in (0.0, 1 / 3, 2 / 3, 1.0)])
    gen = teacher_prims(4, D, 2)
    for name, A in (("on a line segment (affine dimension 1)", seg), ("generic (affine dimension 3)", gen)):
        cells = []
        for K in (2, 3, 4):
            v = sorted(train(A, K, s)["test"] for s in range(3))
            cells.append(f"{v[1]:.1e} [{v[0]:.1e}, {v[2]:.1e}]")
        print(f"| {name} | " + " | ".join(cells) + " |", flush=True)
