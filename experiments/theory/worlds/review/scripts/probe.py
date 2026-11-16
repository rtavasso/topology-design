"""(The __main__ guard was added after the run so probe2.py can import the functions; nothing else changed.)
Review probe 2 (small training runs, one process, time-guarded). Untuned: lr 0.01 unless stated, 4000 steps, teacher 10.
Adds one arm that is NOT in the design: `tok`, the monolith applied to each token separately ([h; E[p]] for each of the
world's two primitives, the same two-hidden-layer net of 40), outputs pooled by a fixed rule (sum: added; select: a
scalar bid per token, softmax over the two tokens). No streams, no learned per-primitive modules; every input it
receives in a held-out world was received in training.
Diagnostic on held-out worlds (one pass from fresh states): least squares of the model's update on the teacher's
primitives: 'chosen/other' = coefficients on the primitive the teacher applies / the world's other primitive
(select; ideal 1, 0; blend 0.5, 0.5), 'in/out' = mean coefficient on the world's two primitives / mean |coef| on the
four others (ideal 1 or ~0.5.., 0), R2 of the 6-primitive fit."""
import sys, time, math, json
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/worlds")
import torch
from worlds import DEF, EPS, Teacher, Model, sample_worlds, references, err, run

T0 = time.time(); BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 230.0
RES = []

class Tok:
    def __init__(self, c, seed):
        self.c = c
        g = torch.Generator().manual_seed(1000 + seed)
        rn = lambda *s: torch.randn(*s, generator=g)
        d, mv, P, W = c["d"], c["mv"], c["P"], c["W"]
        p = {"E": rn(P, mv) / mv ** 0.5, "B1": rn(W, d + mv) / (d + mv) ** 0.5, "b1": torch.zeros(W),
             "B2": rn(W, W) / W ** 0.5, "b2": torch.zeros(W), "V": 0.1 * rn(d + 1, W) / W ** 0.5}
        self.p = {k: v.requires_grad_(True) for k, v in p.items()}
        self.eps = EPS[c["task"]]
    def params(self): return list(self.p.values())
    def forward(self, worlds, x, n, teacher=None):
        p, c = self.p, self.c
        e = p["E"][worlds][:, None, :, :].expand(-1, x.shape[1], -1, -1)             # [M, B, 2, mv]
        h = x
        for _ in range(n):
            hh = h[:, :, None, :].expand(-1, -1, 2, -1)
            z = torch.tanh(torch.cat([hh, e], -1) @ p["B1"].T + p["b1"])
            y = torch.tanh(z @ p["B2"].T + p["b2"]) @ p["V"].T                          # [M, B, 2, d+1]
            u, s = y[..., :-1], y[..., -1] * 10.0
            out = u.sum(2) if c["task"] == "sum" else (torch.softmax(s, -1)[..., None] * u).sum(2)
            h = h + self.eps * out
        return h

def train(over, seed):
    c = dict(DEF); c.update(over)
    if c["arm"] != "tok":
        out, S, T, tr, te = run(over, seed, ret_model=True)
        return c, out, S, T, tr, te
    T, S = Teacher(c), Tok(c, seed)
    tr, te = sample_worlds(c["P"], c["cover"], seed)
    opt = torch.optim.AdamW(S.params(), lr=c["lr"], weight_decay=c["wd"])
    g = torch.Generator().manual_seed(5000 + seed)
    for it in range(c["steps"]):
        for gr in opt.param_groups: gr["lr"] = c["lr"] * 0.5 * (1 + math.cos(math.pi * it / c["steps"]))
        x = torch.randn(len(tr), c["B"], c["d"], generator=g)
        loss = err(S.forward(tr, x, c["n"], T), T.run(tr, x, c["n"]), x)
        opt.zero_grad(); loss.backward(); opt.step()
    out = dict(nparams=sum(v.numel() for v in S.params()))
    with torch.no_grad():
        for nm, ws in (("seen", tr), ("held", te)):
            x = torch.randn(len(ws), 256, c["d"], generator=g)
            out[nm] = err(S.forward(ws, x, c["n"], T), T.run(ws, x, c["n"]), x).item() if len(ws) else float("nan")
        if len(te): out["ref"] = references(c, te)
    return c, out, S, T, tr, te

def diag(c, S, T, ws, seed=7):
    with torch.no_grad():
        x = torch.randn(len(ws), 512, c["d"], generator=torch.Generator().manual_seed(seed))
        u = (S.forward(ws, x, 1, T) - x) / EPS[c["task"]]                                # [M, B, d]
        F = torch.tanh(torch.einsum("pij,mbj->mbpi", T.A, x))                             # [M, B, P, d]
        tu = (T.step(ws, x) - x) / EPS[c["task"]]
        one = [(((u[m] - tu[m]) ** 2).sum() / (tu[m] ** 2).sum()).item() for m in range(len(ws))]
        cin, cout, r2, ch, ot = [], [], [], [], []
        for m in range(len(ws)):
            A = F[m].permute(0, 2, 1).reshape(-1, c["P"]); b = u[m].reshape(-1, 1)
            co = torch.linalg.lstsq(A, b).solution[:, 0]
            r2.append(1 - ((A @ co - b[:, 0]) ** 2).sum().item() / (b ** 2).sum().item())
            w = ws[m].tolist(); others = [p for p in range(c["P"]) if p not in w]
            cin.append(co[w].mean().item()); cout.append(co[others].abs().mean().item())
            if c["task"] == "select":
                cc = T.choice(ws[m:m + 1], x[m:m + 1])[0]                                 # [B]
                fa, fb = F[m, :, w[0]], F[m, :, w[1]]
                fc = torch.where(cc[:, None] == 0, fa, fb); fo = torch.where(cc[:, None] == 0, fb, fa)
                A2 = torch.stack([fc.reshape(-1), fo.reshape(-1)], 1)
                c2 = torch.linalg.lstsq(A2, b).solution[:, 0]
                ch.append(c2[0].item()); ot.append(c2[1].item())
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        return dict(onestep=mean(one), onestep_minmax=(min(one), max(one)), cin=mean(cin), cout=mean(cout), r2=mean(r2), chosen=mean(ch), other=mean(ot))

PLAN = [
    dict(arm="mono", task="select", cover=9, seed=0), dict(arm="mono", task="sum", cover=9, seed=0),
    dict(arm="tok", task="select", cover=9, seed=0), dict(arm="tok", task="sum", cover=9, seed=0),
    dict(arm="streams_self", task="select", cover=9, seed=0), dict(arm="streams_joint", task="select", cover=9, seed=0),
    dict(arm="streams_self", task="sum", cover=9, seed=0), dict(arm="unit", task="sum", cover=9, seed=0),
    dict(arm="mono", task="select", cover=15, seed=0),
    dict(arm="mono", task="select", cover=9, seed=1), dict(arm="mono", task="sum", cover=9, seed=1),
    dict(arm="mono", task="select", cover=9, seed=0, lr=0.03), dict(arm="streams_joint", task="sum", cover=9, seed=0),
    dict(arm="unit", task="select", cover=9, seed=0),
]
if len(sys.argv) > 2:
    PLAN = json.loads(sys.argv[2])
if __name__ == "__main__":
    print("arm task cover seed lr | seen held (blend one) | held-out one-pass err [min,max over worlds] | chosen/other | in/out | R2 | seen-world chosen/other | s", flush=True)
    for job in PLAN:
        if time.time() - T0 > BUDGET:
            print("budget reached; skipped", job, flush=True); continue
        t = time.time(); job = dict(job); seed = job.pop("seed")
        c, out, S, T, tr, te = train(job, seed)
        dh = diag(c, S, T, te) if len(te) else None; ds = diag(c, S, T, tr)
        ref = out.get("ref", {})
        line = f"{c['arm']} {c['task']} {c['cover']} {seed} {c['lr']} | {out['seen']:.4f} {out['held']:.4f} ({ref.get('blend', float('nan')):.2f} {ref.get('one', float('nan')):.2f}) | "
        if dh:
            line += f"{dh['onestep']:.3f} [{dh['onestep_minmax'][0]:.3f},{dh['onestep_minmax'][1]:.3f}] | {dh['chosen']:.2f}/{dh['other']:.2f} | {dh['cin']:.2f}/{dh['cout']:.2f} | {dh['r2']:.2f} | "
        else:
            line += "- | - | - | - | "
        line += f"{ds['chosen']:.2f}/{ds['other']:.2f} (one-pass seen {ds['onestep']:.4f}) | {time.time() - t:.0f}"
        print(line, flush=True)
        RES.append(dict(job=job, seed=seed, out=out, held_diag=dh, seen_diag=ds))
        json.dump(RES, open("probe_results.json", "a" if False else "w"), default=str)
    print("total", round(time.time() - T0, 1))
