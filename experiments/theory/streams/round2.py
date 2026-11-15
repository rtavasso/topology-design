"""Round 2 - revisions of the round-1 predictions that failed, each tested again. EXPLORATORY.

What failed in round 1, why I now think it failed, and the revised prediction (written before this run, 15 Nov 2026):

  R1 (was 1.3: "no stream matches a primitive"). Streams did line up with the teacher's primitives (cosine 0.99-1.00) while
     the gates stayed soft. Revised: the alignment comes from the softmax's competition, not from the function. With
     unnormalised gates (no competition) training still reaches zero error but leaves the streams unaligned: for at least
     one primitive the best stream cosine is below 0.9; with softmax gates every primitive has a stream above 0.98.
  R2 (was 2.3: order-dependent share of the gate gradient linear in eps; measured 2.7-3.3 per doubling). The measurement
     pooled a set of programs closed under reversal, which cancels the first-order term. Revised: computed program by
     program, the share grows linearly: 1.6 to 2.4 per doubling.
  R3 (was 2.4: a state-read gate ends above 0.1 held-out error, and an always-on stream helps threefold; measured 0.036 and
     no help). Revised: 0.036 is the order-blind floor, i.e. the state-read gate learned a bag. (i) A student whose gate
     is constant over the passes has a floor within a factor 2 of the state-read students' held-out error. (ii) The
     ordered solution is representable: a hand-built state-read student (streams advance the tape; the gate reads the first
     slot) has held-out error below 1e-6. (iii) It is locally stable: started from the hand-built solution with 5 % noise,
     training ends below 1e-3. (iv) It is not found: from scratch with three times the steps the held-out error stays
     above half the constant-gate floor.
  R4 (was 3.1 and 4.4: gain peaks at intermediate temperature; coupling leaves the common-mode gain unchanged). Both
     missed one thing: averaging different streams contracts. Revised: (i) the soft-gate loop's median gain at tau = 4 is
     within 10 % of the loop with a uniform gate, and at tau = 0.05 within 25 % of the median gain of single-stream loops;
     (ii) with separate states and a complete graph at c = 1/K (states averaged every pass) the common-mode gain is within
     15 % of the gain of a shared-state loop that adds the mean of the streams.
  R5 (was 3.3: the tenth of inputs nearest a boundary carries over half the gate gradient at tau = 0.1; measured 14 %).
     A tenth was the wrong yardstick: with six passes many inputs pass within a few tau of a boundary. Revised: the
     per-input gate-gradient norm falls with (smallest margin) / tau: inputs with margin above 5 tau have a mean gradient
     norm below a tenth of that of inputs with margin below tau, at tau = 0.05 and 0.02.
  R6 (was 4.1: diversity ordered by the second Laplacian eigenvalue; the star was more diverse than the ring). At eight
     passes every mode counts, not the slowest. Revised: with equal drive in every disagreement mode,
     D = D_0 * sqrt(mean_i phi(c lambda_i)^2), phi(a) = (1 - (1 - a)^n) / (a n), predicts ring, star and complete within
     15 % for c in {0.02, 0.05, 0.1, 0.2}.

Usage: python3 round2.py a|b      (a: R1, R2, R4, R5, R6;  b: R3)
"""
import math, sys
import torch
from torch.func import jacrev, vmap, grad
from lib import *

torch.set_num_threads(1)


def r1():
    import lens1_symmetry as L1
    A3 = teacher_prims(3, L1.D, 0)
    print("**R1** three primitives, K = 4 streams; for each primitive the best |cosine| with any stream (three seeds):\n")
    print("| gates | held-out error (median) | smallest of the three best cosines, by seed |")
    print("|---|---|---|")
    for kind in ("softmax", "unnormalised"):
        errs, mins = [], []
        for seed in range(3):
            P, K = 3, 4
            progs = all_programs(P, L1.N); tr, te = split(progs, 0.75, seed)
            g = torch.Generator().manual_seed(100 + seed)
            B = (0.1 * torch.randn(K, L1.D, L1.D, generator=g) / L1.D ** 0.5).requires_grad_(True)
            E = ((0.1 * torch.randn(P, K, generator=g)) + (0.25 if kind == "unnormalised" else 0.0)).requires_grad_(True)
            gate = (lambda: torch.softmax(E, 1)) if kind == "softmax" else (lambda: E)
            opt = torch.optim.Adam([B, E], lr=L1.LR)
            for it in range(2500):
                x = torch.randn(len(tr), L1.BATCH, L1.D, generator=g)
                loss = loss_fn(run_student(B, gate()[tr], x, L1.EPS), run_teacher(A3, tr, x, L1.EPS), x)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x = torch.randn(len(te), 256, L1.D, generator=g)
                errs.append(nmse(run_student(B, gate()[te], x, L1.EPS), run_teacher(A3, te, x, L1.EPS), x))
                cos = torch.nn.functional.cosine_similarity(B.flatten(1)[:, None, :], A3.flatten(1)[None, :, :], dim=2).abs()
                mins.append(cos.max(0).values.min().item())
        print(f"| {kind} | {sorted(errs)[1]:.1e} | " + ", ".join(f"{m:.2f}" for m in mins) + " |", flush=True)


def r2():
    D, N = 6, 4
    A = teacher_prims(2, D, 3)
    progs = all_programs(2, N)
    progs = progs[torch.tensor([list(p) != list(reversed(p)) for p in progs.tolist()])]
    print("\n**R2** order-dependent share of the gate gradient, computed program by program (mean over programs and five draws):\n")
    print("| step size eps | share | ratio to the previous row |")
    print("|---|---|---|")
    prev = None
    for eps in (0.05, 0.1, 0.2, 0.4):
        vals = []
        for seed in range(5):
            g = torch.Generator().manual_seed(300 + seed)
            B = 0.3 * torch.randn(2, D, D, generator=g) / D ** 0.5
            E = (0.3 * torch.randn(2, 2, generator=g)).requires_grad_(True)
            for m in range(len(progs)):
                pr = progs[m:m + 1]
                x = torch.randn(1, 256, D, generator=g)
                target = run_teacher(A, pr, x, eps)
                gs = [torch.autograd.grad(loss_fn(run_student(B, torch.softmax(E, 1)[q], x, eps), target, x), E)[0] for q in (pr, pr.flip(1))]
                vals.append(((gs[0] - gs[1]).norm() / gs[0].norm()).item())
        v = sum(vals) / len(vals)
        print(f"| {eps} | {v:.3f} | {'–' if prev is None else f'{v / prev:.2f}'} |", flush=True)
        prev = v


def r4_r5():
    import lens3_partition as L3
    B, V, T, x = L3.make(0)
    x = x[:1000]

    def gain(gate_fn):
        def out(h):
            for _ in range(L3.N):
                h = h + L3.EPS * torch.einsum("k,ki->i", gate_fn(h), torch.tanh(B @ h))
            return h
        return torch.linalg.matrix_norm(vmap(jacrev(out))(x), 2)
    uni = gain(lambda h: torch.full((L3.K,), 1 / L3.K)).median().item()
    single = torch.cat([gain(lambda h, k=k: torch.nn.functional.one_hot(torch.tensor(k), L3.K).double()) for k in range(L3.K)]).median().item()
    soft = {tau: gain(lambda h, tau=tau: torch.softmax(V @ h / tau, 0)).median().item() for tau in (4.0, 0.05)}
    print("\n**R4 (i)** median gain of the six-pass loop: uniform gate "
          f"{uni:.2f}; soft gate at tau = 4: {soft[4.0]:.2f}; soft gate at tau = 0.05: {soft[0.05]:.2f}; single streams alone: {single:.2f}.")
    import lens4_coupling as L4
    g = torch.Generator().manual_seed(0)
    B4 = torch.randn(L4.K, L4.D, L4.D, generator=g) / L4.D ** 0.5
    x4 = torch.randn(500, L4.D, generator=g)[:200]
    g2 = torch.Generator().manual_seed(1)
    dv = torch.randn(L4.D, generator=g2); dv = 1e-5 * dv / dv.norm()
    Lc = L4.lap(L4.graphs()["complete"])

    def roll(X0, c):
        X = X0
        for _ in range(L4.N):
            X = X + L4.EPS * torch.tanh(torch.einsum("kij,mkj->mki", B4, X)) - c * torch.einsum("kj,mjd->mkd", Lc, X)
        return X
    X0 = x4.unsqueeze(1).expand(-1, L4.K, -1).clone()
    cm = ((roll(X0 + dv, 1 / L4.K) - roll(X0, 1 / L4.K)).mean(1).norm(dim=1) / 1e-5).mean().item()

    def shared(h):
        for _ in range(L4.N):
            h = h + L4.EPS * torch.tanh(torch.einsum("kij,mj->mki", B4, h)).mean(1)
        return h
    sh = ((shared(x4 + dv) - shared(x4)).norm(dim=1) / 1e-5).mean().item()
    print(f"\n**R4 (ii)** gain for a common perturbation: separate states, complete graph, c = 1/K: {cm:.2f}; shared state adding the mean of the streams: {sh:.2f}.")
    print("\n**R5** mean per-input gate-gradient norm by (smallest margin) / tau:\n")
    print("| tau | margin < tau (share of inputs) | tau to 5 tau | > 5 tau (share of inputs) | ratio of the last to the first |")
    print("|---|---|---|---|---|")
    for tau in (0.05, 0.02):
        def sample_loss(Vp, h):
            return 0.5 * ((L3.loop(h, B, Vp, tau)[0] - T @ h) ** 2).sum()
        gV = vmap(grad(sample_loss), in_dims=(None, 0))(V, x).flatten(1).norm(dim=1)
        m = vmap(lambda h: L3.loop(h, B, V, tau)[1])(x) / tau
        lo, mid, hi = m < 1, (m >= 1) & (m <= 5), m > 5
        a, b, c = gV[lo].mean().item(), gV[mid].mean().item(), gV[hi].mean().item()
        print(f"| {tau} | {a:.3g} ({lo.double().mean().item():.0%}) | {b:.3g} | {c:.3g} ({hi.double().mean().item():.0%}) | {c / a:.3f} |", flush=True)


def r6():
    import lens4_coupling as L4
    g = torch.Generator().manual_seed(0)
    B = torch.randn(L4.K, L4.D, L4.D, generator=g) / L4.D ** 0.5
    x = torch.randn(500, L4.D, generator=g)
    G = L4.graphs()
    d0 = L4.diversity(L4.run(B, x, L4.lap(G["none"]), 0.0))
    phi = lambda a: 1.0 if a == 0 else (1 - (1 - a) ** L4.N) / (a * L4.N)
    print("\n**R6** diversity after eight passes: measured / predicted from the whole Laplacian spectrum:\n")
    print("| coupling c | ring | star | complete |")
    print("|---|---|---|---|")
    for c in (0.02, 0.05, 0.1, 0.2):
        cells = []
        for name in ("ring", "star", "complete"):
            lam = torch.linalg.eigvalsh(L4.lap(G[name]))[1:]
            pred = d0 * math.sqrt(sum(phi(c * l.item()) ** 2 for l in lam) / len(lam))
            meas = L4.diversity(L4.run(B, x, L4.lap(G[name]), c))
            cells.append(f"{meas:.2f} / {pred:.2f} ({meas / pred - 1:+.0%})")
        print(f"| {c} | " + " | ".join(cells) + " |", flush=True)


def r3():
    d, P, K, N, eps = 4, 2, 2, 4, 0.4
    A = teacher_prims(P, d, 10)
    T = N * P; Df = d + T
    progs = all_programs(P, N)
    tape = lambda pr: torch.nn.functional.one_hot(pr, P).reshape(len(pr), T).to(torch.get_default_dtype())

    def forward(B, V, pr, x):
        s = torch.cat([x, tape(pr)[:, None, :].expand(-1, x.shape[1], -1)], 2)
        for t in range(N):
            f = torch.einsum("kij,mbj->mbki", B, s)
            gate = torch.softmax(torch.einsum("kj,mbj->mbk", V, s), 2)
            s = s + eps * torch.einsum("mbk,mbki->mbi", gate, f)
        return s[:, :, :d]

    def evaluate(B, V, pr, g):
        x = torch.randn(len(pr), 512, d, generator=g)
        return nmse(forward(B, V, pr, x), run_teacher(A, pr, x, eps), x)

    def built():
        S = torch.zeros(T, T)
        for j in range(T - P):
            S[j, j + P] = 1.0                                             # slot j takes the content of the next slot
        B = torch.zeros(K, Df, Df)
        for k in range(K):
            B[k, :d, :d] = A[k]
            B[k, d:, d:] = (S - torch.eye(T)) / eps
        V = torch.zeros(K, Df)
        for k in range(K):
            V[k, d + k] = 30.0
        return B, V

    def train(B, V, tr, g, steps, lr):
        B = B.clone().requires_grad_(True); V = V.clone().requires_grad_(True)
        opt = torch.optim.Adam([B, V], lr=lr)
        for it in range(steps):
            x = torch.randn(len(tr), 32, d, generator=g)
            loss = loss_fn(forward(B, V, tr, x), run_teacher(A, tr, x, eps), x)
            opt.zero_grad(); loss.backward(); opt.step()
        return B.detach(), V.detach()
    print("**R3** program stored in the state; two primitives, four passes, step 0.4; three seeds, median [min, max] held-out error:\n")
    rows = {"const": [], "built": [], "near": [], "scratch": []}
    for seed in range(3):
        tr, te = split(progs, 0.75, seed)
        g = torch.Generator().manual_seed(500 + seed)
        # (i) gate constant over the passes: one free gate vector per program, streams on the data only
        Bc = (0.1 * torch.randn(K, d, d, generator=g) / d ** 0.5).requires_grad_(True)
        Ec = (0.1 * torch.randn(len(progs), K, generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([Bc, Ec], lr=1e-2)
        idx = lambda pr: torch.tensor([int("".join(map(str, p)), P) for p in pr.tolist()])
        for it in range(2500):
            x = torch.randn(len(progs), 32, d, generator=g)
            gates = torch.softmax(Ec, 1)[:, None, :].expand(-1, N, -1)
            loss = loss_fn(run_student(Bc, gates, x, eps), run_teacher(A, progs, x, eps), x)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x = torch.randn(len(te), 512, d, generator=g)
            rows["const"].append(nmse(run_student(Bc, torch.softmax(Ec, 1)[idx(te)][:, None, :].expand(-1, N, -1), x, eps), run_teacher(A, te, x, eps), x))
        B0, V0 = built()
        rows["built"].append(evaluate(B0, V0, te, g))
        Bn = B0 + 0.05 * B0.pow(2).mean().sqrt() * torch.randn(B0.shape, generator=g)
        Vn = V0 + 0.05 * V0.pow(2).mean().sqrt() * torch.randn(V0.shape, generator=g)
        Bt, Vt = train(Bn, Vn, tr, g, 1500, 1e-3)
        rows["near"].append(evaluate(Bt, Vt, te, g))
        Bs = 0.1 * torch.randn(K, Df, Df, generator=g) / Df ** 0.5
        Vs = 0.1 * torch.randn(K, Df, generator=g)
        Bt, Vt = train(Bs, Vs, tr, g, 9000, 3e-3)
        rows["scratch"].append(evaluate(Bt, Vt, te, g))
        print(f"<!-- seed {seed} done -->", flush=True)
    names = {"const": "(i) gate constant over the passes (order-blind reference; one free gate per program, fitted on all programs)",
             "built": "(ii) hand-built state-read solution", "near": "(iii) trained from the hand-built solution with 5 % noise",
             "scratch": "(iv) state-read gate trained from scratch, 9,000 steps"}
    print("\n| student | held-out error |")
    print("|---|---|")
    for k, v in rows.items():
        v = sorted(v)
        print(f"| {names[k]} | {v[1]:.1e} [{v[0]:.1e}, {v[2]:.1e}] |")


if __name__ == "__main__":
    if sys.argv[1] == "a":
        print("# Round 2 — revised predictions, tested (EXPLORATORY)\n\nPredictions in `round2.py`.\n")
        r1(); r2(); r4_r5(); r6()
    else:
        r3()
