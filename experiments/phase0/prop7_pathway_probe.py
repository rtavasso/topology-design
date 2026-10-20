"""EXPLORATORY probe of Prop. 7 (docs/phase0_linear_theory.md): mixed graph k=(0,1,0,2), W = (I+W4)(I+W2)(I+W1) + W3.

Teacher modes: 3 to amplify (s* = 4, 2.5, 1.5), 3 to shrink (0.6, 0.3, 0.1). Measures the shortcut W3's share of the
learned change W - I, per teacher mode and per subspace. First-order baseline share is 1/4 (all four matrices receive
the same gradient at zero). Run:  PYTHONPATH=src python experiments/phase0/prop7_pathway_probe.py

Result on 2026-10-19 (4 data seeds): amplify-subspace share 0.24, erase-subspace share 0.40-0.42; per-mode
[0.16-0.17, 0.25-0.26, 0.39-0.40, 0.47-0.49, -0.17..-0.29, 0.44]. Subspace-level supports the conjecture;
per-mode is non-monotone and one erased mode has the shortcut OPPOSING the change. Unexplained -> derive before relying on it.
"""
import torch; torch.set_default_dtype(torch.float64)
from topology_design import linear_family as lf
d=6; g=torch.Generator().manual_seed(0)
U,_ = torch.linalg.qr(torch.randn(d,d,generator=g)); Vq,_ = torch.linalg.qr(torch.randn(d,d,generator=g))
s_star = torch.tensor([4.0,2.5,1.5,0.6,0.3,0.1])
M = U@torch.diag(s_star)@Vq.T
def sub_share(sc, tot, idx):
    Pu, Pv = U[:,idx]@U[:,idx].T, Vq[:,idx]@Vq[:,idx].T
    a = (Pu@sc@Pv); b=(Pu@tot@Pv)
    return ((a*b).sum()/(b*b).sum()).item()
PARENTS = (0, 1, 0, 2)
SHORTCUT = 2  # index of W3
for seed in range(4):
    gg = torch.Generator().manual_seed(seed + 10)
    X = torch.randn(8000, d, generator=gg)
    Y = torch.multinomial(torch.softmax(X @ M.T, 1), 1, generator=gg).squeeze(1)
    task = lf.Task(X, Y, torch.eye(d), torch.zeros(d))
    Ws = lf.train(lf.zeros(len(PARENTS), d), PARENTS, task, steps=4000, lr=0.5)
    tot = lf.end_to_end(Ws, PARENTS) - torch.eye(d)
    sc = Ws[SHORTCUT]
    per = [round(sub_share(sc, tot, [i]), 2) for i in range(d)]
    print(seed, "amplify-subspace share %.2f  erase-subspace share %.2f  per-mode"
          % (sub_share(sc, tot, [0, 1, 2]), sub_share(sc, tot, [3, 4, 5])), per)
