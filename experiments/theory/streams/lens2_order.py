"""Lens 2 - order: streams as control vector fields; what carries the information about order. EXPLORATORY.

Structure. h <- h + eps * sum_k g_t[k] f_k(h) is one Euler step of a control-affine system dh/dt = sum_k g_k(t) f_k(h):
the streams are vector fields, the gate is the control, a program is a control signal. Two orders of the same steps differ
by the commutator: (I + eps A)(I + eps B) - (I + eps B)(I + eps A) = eps^2 [A, B]. A gate that does not change over the
passes realises only (I + eps * sum_k g_k B_k)^n, which is blind to order.

Predictions written before the run (15 Nov 2026):
  2.2 A "bag" student (its gate is the average of the program's token gates, the same at every pass) has an error floor
      that grows as eps^2: each doubling of eps multiplies it by 3 to 5 for eps <= 0.2. The ordered student is below 1e-4
      at every eps. So with small steps a loop has almost nothing to gain from knowing the order.
  2.3 The part of the gate's gradient that depends on order (gradient with the true program minus gradient with the
      program reversed, relative to the gradient's size) grows linearly with eps: 1.6 to 2.4 per doubling.
  2.4 Program stored in the state (a tape of one-hot tokens appended to the data), gate reading only the state:
      (b) a gate given the pass index and the tape generalises to held-out programs (error < 0.01);
      (c1) a gate that reads the state alone does not within the budget (held-out error > 0.1), because every stream must
           also learn to advance the tape;
      (c2) adding one always-on, ungated stream (shared bookkeeping) lowers (c1)'s held-out error at least threefold, and
           stays above (b).
"""
import sys
import torch
from lib import *

torch.set_num_threads(1)
D, N, LR, BATCH = 6, 4, 1e-2, 32


def fit(A, eps, mode, seed, steps=1500):
    P, K = A.shape[0], A.shape[0]
    progs = all_programs(P, N)
    g = torch.Generator().manual_seed(200 + seed)
    B = (0.1 * torch.randn(K, D, D, generator=g) / D ** 0.5).requires_grad_(True)
    E = (0.1 * torch.randn(P, K, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([B, E], lr=LR)

    def gates():
        gt = torch.softmax(E, 1)[progs]                                   # [M, n, K]
        return gt.mean(1, keepdim=True).expand(-1, N, -1) if mode == "bag" else gt
    for it in range(steps):
        x = torch.randn(len(progs), BATCH, D, generator=g)
        loss = loss_fn(run_student(B, gates(), x, eps), run_teacher(A, progs, x, eps), x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.randn(len(progs), 512, D, generator=g)
        return nmse(run_student(B, gates(), x, eps), run_teacher(A, progs, x, eps), x)


def order_fraction(A, eps, seed):
    P = K = A.shape[0]
    progs = all_programs(P, N)
    keep = torch.tensor([list(p) != list(reversed(p)) for p in progs.tolist()])
    progs = progs[keep]
    g = torch.Generator().manual_seed(300 + seed)
    B = 0.3 * torch.randn(K, D, D, generator=g) / D ** 0.5
    E = (0.3 * torch.randn(P, K, generator=g)).requires_grad_(True)
    x = torch.randn(len(progs), 256, D, generator=g)
    target = run_teacher(A, progs, x, eps)
    grads = []
    for pr in (progs, progs.flip(1)):
        loss = loss_fn(run_student(B, torch.softmax(E, 1)[pr], x, eps), target, x)
        grads.append(torch.autograd.grad(loss, E)[0])
    return ((grads[0] - grads[1]).norm() / grads[0].norm()).item()


def tape_task(variant, seed, steps=3000, eps=0.4, d=4, P=2, K=2):
    A = teacher_prims(P, d, 10)
    progs = all_programs(P, N)
    tr, te = split(progs, 0.75, seed)
    T = N * P
    Dfull = d + T
    g = torch.Generator().manual_seed(400 + seed)
    tape = lambda pr: torch.nn.functional.one_hot(pr, P).reshape(len(pr), T).to(torch.get_default_dtype())
    if variant == "clock":
        B = (0.1 * torch.randn(K, d, d, generator=g) / d ** 0.5).requires_grad_(True)
        W = (0.1 * torch.randn(N, K, T, generator=g)).requires_grad_(True)
        params = [B, W]
    else:
        nstream = K + (1 if variant == "state+shared" else 0)
        B = (0.1 * torch.randn(nstream, Dfull, Dfull, generator=g) / Dfull ** 0.5).requires_grad_(True)
        V = (0.1 * torch.randn(K, Dfull, generator=g)).requires_grad_(True)
        params = [B, V]
    opt = torch.optim.Adam(params, lr=3e-3 if variant != "clock" else LR)

    def forward(pr, x):
        if variant == "clock":
            gates = torch.softmax(torch.einsum("tkj,mj->mtk", W, tape(pr)), 2)
            return run_student(B, gates, x, eps)
        s = torch.cat([x, tape(pr)[:, None, :].expand(-1, x.shape[1], -1)], 2)
        for t in range(N):
            f = torch.einsum("kij,mbj->mbki", B, s)
            gate = torch.softmax(torch.einsum("kj,mbj->mbk", V, s), 2)
            upd = torch.einsum("mbk,mbki->mbi", gate, f[:, :, -K:] if variant == "state+shared" else f)
            if variant == "state+shared":
                upd = upd + f[:, :, 0]
            s = s + eps * upd
        return s[:, :, :d]
    for it in range(steps):
        x = torch.randn(len(tr), BATCH, d, generator=g)
        loss = loss_fn(forward(tr, x), run_teacher(A, tr, x, eps), x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = []
        for pr in (tr, te):
            x = torch.randn(len(pr), 512, d, generator=g)
            out.append(nmse(forward(pr, x), run_teacher(A, pr, x, eps), x))
        return out


if __name__ == "__main__":
    part = sys.argv[1]
    A = teacher_prims(2, D, 3)
    comm = (A[0] @ A[1] - A[1] @ A[0]).norm().item() / (A[0].norm() * A[1].norm()).item()
    if part == "a":
        print("# Lens 2 — order (EXPLORATORY)\n")
        print(f"Two teacher primitives, d = {D}, {N} passes; relative commutator ‖[A, B]‖ / (‖A‖‖B‖) = {comm:.2f}. Predictions in `lens2_order.py`.\n")
        print("**2.2** error floor of a student blind to order, and of the ordered student (median of three seeds):\n")
        print("| step size eps | bag student | ratio to the previous row | ordered student |")
        print("|---|---|---|---|")
        prev = None
        for eps in (0.05, 0.1, 0.2, 0.4):
            b = sorted(fit(A, eps, "bag", s) for s in range(3))[1]
            o = sorted(fit(A, eps, "instr", s) for s in range(3))[1]
            print(f"| {eps} | {b:.2e} | {'–' if prev is None else f'{b / prev:.1f}'} | {o:.1e} |", flush=True)
            prev = b
        print("\n**2.3** share of the gate's gradient that depends on the order of the program (mean of five draws):\n")
        print("| step size eps | order-dependent share | ratio to the previous row |")
        print("|---|---|---|")
        prev = None
        for eps in (0.05, 0.1, 0.2, 0.4):
            v = sum(order_fraction(A, eps, s) for s in range(5)) / 5
            print(f"| {eps} | {v:.3f} | {'–' if prev is None else f'{v / prev:.2f}'} |", flush=True)
            prev = v
    else:
        print("\n**2.4** program stored in the state (two primitives, four passes, step 0.4, 12 programs for training, 4 held out; three seeds):\n")
        print("| what the gate sees | training error, median [min, max] | held-out error, median [min, max] |")
        print("|---|---|---|")
        for variant, name in (("clock", "(b) pass index and the tape"), ("state", "(c1) the state only"), ("state+shared", "(c2) the state only, plus one always-on stream")):
            res = [tape_task(variant, s) for s in range(3)]
            tr = sorted(r[0] for r in res); te = sorted(r[1] for r in res)
            print(f"| {name} | {tr[1]:.1e} [{tr[0]:.1e}, {tr[2]:.1e}] | {te[1]:.1e} [{te[0]:.1e}, {te[2]:.1e}] |", flush=True)
