"""What a state-reading gate is given, on the stored-program task. EXPLORATORY.

Question (the researcher, 15 Nov 2026): a looped block usually re-injects the input every pass; if the gate is also fed
the input, or the previous pass's state, does it then learn a gate that changes over the passes?

Task as in lens 2: two primitives, four passes, step 0.4; 12 programs for training, 4 held out. Streams act on the data
only (so they cannot keep a counter). The gate is a small MLP (one hidden layer of 32) over: the current data state and the
stored program, plus what the variant adds.

Predictions written before the run:
  G0  state + program only: at the order-blind floor (held-out error above 0.017, half of 0.034).
  G1  + the input x: helps but does not solve - between 1e-3 and 0.017. The displacement from the input says roughly how
      much has been applied (a clock of sorts), but it must be decoded from two noisy vectors.
  G2  + the previous pass's state: no better than G0 (above 0.017). The last write says what was just applied, not where
      in the program the loop is.
  G3  + both: like G1.
  G4  + the pass index: solves it (below 1e-3), as the clock gate did in lens 2.
"""
import sys
import torch
from lib import *

torch.set_num_threads(1)
d, P, K, N, eps, H = 4, 2, 2, 4, 0.4, 32


def run(variant, seed, steps=3000):
    A = teacher_prims(P, d, 10)
    progs = all_programs(P, N)
    tr, te = split(progs, 0.75, seed)
    T = N * P
    g = torch.Generator().manual_seed(600 + seed)
    extra = {"G0": 0, "G1": d, "G2": d, "G3": 2 * d, "G4": N}[variant]
    F = d + T + extra
    B = (0.1 * torch.randn(K, d, d, generator=g) / d ** 0.5).requires_grad_(True)
    W1 = (torch.randn(H, F, generator=g) / F ** 0.5).requires_grad_(True)
    b1 = torch.zeros(H, requires_grad=True)
    W2 = (0.1 * torch.randn(K, H, generator=g) / H ** 0.5).requires_grad_(True)
    opt = torch.optim.Adam([B, W1, b1, W2], lr=3e-3)
    tape = lambda pr: torch.nn.functional.one_hot(pr, P).reshape(len(pr), T).to(torch.get_default_dtype())

    def forward(pr, x):
        tp = tape(pr)[:, None, :].expand(-1, x.shape[1], -1)
        h, prev = x, x
        for t in range(N):
            feats = [h, tp]
            if variant in ("G1", "G3"):
                feats.append(x)
            if variant in ("G2", "G3"):
                feats.append(prev)
            if variant == "G4":
                feats.append(torch.nn.functional.one_hot(torch.tensor(t), N).to(h.dtype).expand(h.shape[0], h.shape[1], -1))
            z = torch.tanh(torch.cat(feats, 2) @ W1.T + b1)
            gate = torch.softmax(z @ W2.T, 2)
            f = torch.einsum("kij,mbj->mbki", B, h)
            prev, h = h, h + eps * torch.einsum("mbk,mbki->mbi", gate, f)
        return h
    for it in range(steps):
        x = torch.randn(len(tr), 32, d, generator=g)
        loss = loss_fn(forward(tr, x), run_teacher(A, tr, x, eps), x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = []
        for pr in (tr, te):
            x = torch.randn(len(pr), 512, d, generator=g)
            out.append(nmse(forward(pr, x), run_teacher(A, pr, x, eps), x))
        return out


if __name__ == "__main__":
    names = {"G0": "state and stored program", "G1": "+ the input", "G2": "+ the previous pass's state", "G3": "+ both", "G4": "+ the pass index"}
    for v in sys.argv[1:]:
        res = [run(v, s) for s in range(3)]
        tr = sorted(r[0] for r in res); te = sorted(r[1] for r in res)
        print(f"| {v} {names[v]} | {tr[1]:.1e} [{tr[0]:.1e}, {tr[2]:.1e}] | {te[1]:.1e} [{te[0]:.1e}, {te[2]:.1e}] |", flush=True)
