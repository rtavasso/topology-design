"""Review of round M results: small extra runs. One process at a time; torch threads 1 (set by loopmoe).
modes: probe | branchgate <lr> <seeds..> | m8lr <lr> <seeds..> | slack <seeds..> | row <key> <lr> <seeds..>"""
import json, sys, time, statistics as st
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/streams")
import torch
import loopmoe
from loopmoe import run, rms, Student
import arch_m
H = "/home/claude/topology-design/experiments/theory/streams/"
D = {}
for f in ("arch_m_results_a.json", "arch_m_results_b.json"):
    D.update(json.load(open(H + f)))
ROWS = arch_m.ROWS


class GateStudent(Student):
    """Per-unit gate whose logits also read the data through an MLP (the candidate's branch gate, one gate per unit):
    z = sigmoid(C r + b + W2 tanh(W1 [r; h] + b1)) * tanh(B h);  h <- h + eps V z.   Active when cfg has gateH."""
    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        if cfg.get("gateH"):
            c = cfg; Hh, W = c["gateH"], c["W"]
            rn = lambda *s: torch.randn(*s, generator=self.g)
            self.p["W1"] = (rn(Hh, c["mv"] + c["d"]) / (c["mv"] + c["d"]) ** 0.5).requires_grad_(True)
            self.p["b1"] = torch.zeros(Hh).requires_grad_(True)
            self.p["W2"] = (0.1 * rn(W, Hh) / Hh ** 0.5).requires_grad_(True)

    def forward(self, progs, x, train=False, hard=True, record=False):
        c, p = self.c, self.p
        if not c.get("gateH"):
            return super().forward(progs, x, train=train, hard=hard, record=record)
        M, Bx, n = progs.shape[0], x.shape[1], progs.shape[1]
        vals, keys = p["E"][progs], p["keys"][:n]
        h = x; reg = p["c0"].expand(M, Bx, -1)
        aux = dict(logits=[], gates=[], attn=[], states=[], surr=0.0, ran=0.0)
        for t in range(n):
            q = p["beta"] * (rms(reg) if c["ln"] else reg)
            a = torch.softmax(q @ keys.T, -1)
            r = torch.einsum("mbi,miv->mbv", a, vals)
            if record:
                aux["attn"].append(a.detach())
            s = r @ p["mC"].T + p["mb"] + torch.tanh(torch.cat([r, h], -1) @ p["W1"].T + p["b1"]) @ p["W2"].T
            z = torch.sigmoid(s) * torch.tanh(h @ p["mB"].T)
            h = h + c["eps"] * z @ p["mV"].T
            reg = reg + reg @ p["M"].T
        return h, aux


loopmoe.Student = GateStudent


def attn_on_t(S, te, L, d=6):
    x = torch.randn(len(te[L]), 8, d, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        _, ax = S.forward(te[L], x, record=True)
    return [round(ax["attn"][t][..., t].mean().item(), 2) for t in range(L)]


mode = sys.argv[1]
T0 = time.time()
if mode == "probe":
    # (e) the length-8 jump of the per-unit gate, seed 9, at the selected rate and at 0.003; (c) how sparse its gates are
    for lr in (0.01, 0.003):
        t0 = time.time()
        out, S, T, tr, te = run(dict(ROWS["M4"][2], lr=lr), 9, ret_model=True)
        print(f"M4 seed 9 lr {lr}: te4 {out['by_len'][4]:.2e} te6 {out['by_len'][6]:.2e} te8 {out['by_len'][8]:.2e} ratio8/4 {out['by_len'][8]/out['by_len'][4]:.1f} "
              f"beta {S.p['beta'].item():.2f} attn on t, length 8: {attn_on_t(S, te, 8)}  ({time.time()-t0:.0f}s)", flush=True)
        if lr == 0.01:
            print("   stored:", {k: f"{v:.2e}" for k, v in D["M4"][9].items() if k in ("te4", "te8")})
            with torch.no_grad():
                p = S.p
                G = torch.sigmoid(p["E"] @ p["mC"].T + p["mb"])          # [tokens, units]
                print("   gates per token: >0.5:", (G > 0.5).sum(1).tolist(), " <0.05:", (G < 0.05).sum(1).tolist(),
                      " in between:", ((G >= 0.05) & (G <= 0.5)).sum(1).tolist(), " of", G.shape[1])
                print("   units open (>0.5) for 0/1/2/3 tokens:", [int(((G > 0.5).sum(0) == k).sum()) for k in range(4)])
                # run with only the k most open units per token computed (others set to zero), exact instruction read
                def ev(k):
                    pr = te[4]; x = torch.randn(len(pr), 128, 6, generator=torch.Generator().manual_seed(3)); h = x
                    for t in range(4):
                        g = G[pr[:, t]]                                   # [M, units]
                        if k:
                            keep = torch.zeros_like(g).scatter(-1, g.topk(k, -1).indices, 1.0); g = g * keep
                        h = h + 0.4 * (g[:, None, :] * torch.tanh(h @ p["mB"].T)) @ p["mV"].T
                    tg = T(pr, x)
                    return (((h - tg) ** 2).sum() / ((tg - x) ** 2).sum()).item()
                print("   length-4 error, exact address, top-k units per token computed:", {k or 29: f"{ev(k):.1e}" for k in (0, 18, 12, 9, 6)})
elif mode == "branchgate":
    lr = float(sys.argv[2])
    cfg = dict(ROWS["M13"][2], block="mono_mul", W=27, gateH=12, lr=lr)
    for s in map(int, sys.argv[3:]):
        t0 = time.time()
        r = run(cfg, s)
        c, m13 = D["M12"][s], D["M13"][s]
        print(f"branch per-unit MLP gate (27 units, gate hidden 12) lr {lr} seed {s}: params {r['nparams']} train4 {r['train_by_len'][4]:.2e} te4 {r['by_len'][4]:.2e} "
              f"te8 {r['by_len'][8]:.2e} x2 {r['scaled'][2.0][4]:.2e} c1 {r['curve'][1000]:.1e} c2 {r['curve'][2000]:.1e} | M12 te4 {c['te4']:.2e} x2 {c['x2']:.2e} | M13 te4 {m13['te4']:.2e} "
              f"| ratio to M12: te4 {r['by_len'][4]/c['te4']:.2f} te8 {r['by_len'][8]/c['te8']:.2f} x2 {r['scaled'][2.0][4]/c['x2']:.2f} ({time.time()-t0:.0f}s)", flush=True)
elif mode == "m8lr":
    lr = float(sys.argv[2])
    for s in map(int, sys.argv[3:]):
        t0 = time.time()
        r = run(dict(ROWS["M8"][2], lr=lr), s)
        print(f"M8 lr {lr} seed {s}: train4 {r['train_by_len'][4]:.2e} te4 {r['by_len'][4]:.2e} te8 {r['by_len'][8]:.2e} x2 {r['scaled'][2.0][4]:.2e} c1 {r['curve'][1000]:.1e} c2 {r['curve'][2000]:.1e} switch {r['switch']} "
              f"| stored M8@0.01 te4 {D['M8'][s]['te4']:.2e} | M10 te4 {D['M10'][s]['te4']:.2e} ratio M10/this {D['M10'][s]['te4']/r['by_len'][4]:.2f} ({time.time()-t0:.0f}s)", flush=True)
elif mode == "slack":
    # is the 18-unit per-unit gate's loss a compute effect or a no-slack effect? candidate with 3 streams of 6 (also 18 units, no slack)
    for s in map(int, sys.argv[2:]):
        t0 = time.time()
        r = run(dict(ROWS["M1"][2], K=3, w=6, lr=0.01), s)
        print(f"candidate K=3 w=6 (18 units in all, 6 per pass) seed {s}: params {r['nparams']} te4 {r['by_len'][4]:.2e} | M4a te4 {D['M4a'][s]['te4']:.2e} | M1 te4 {D['M1'][s]['te4']:.2e} ({time.time()-t0:.0f}s)", flush=True)
elif mode == "row":                          # row <key> <lr> <seeds..>: any round-M row at another rate
    key, lr = sys.argv[2], float(sys.argv[3])
    for s in map(int, sys.argv[4:]):
        t0 = time.time()
        r = run(dict(ROWS[key][2], lr=lr), s)
        print(f"{key} lr {lr} seed {s}: train4 {r['train_by_len'][4]:.2e} te4 {r['by_len'][4]:.2e} te8 {r['by_len'][8]:.2e} x2 {r['scaled'][2.0][4]:.2e} c1 {r['curve'][1000]:.1e} c2 {r['curve'][2000]:.1e} "
              f"| stored {key} te4 {D[key][s]['te4']:.2e} ({time.time()-t0:.0f}s)", flush=True)
print(f"total {time.time()-T0:.0f}s")
