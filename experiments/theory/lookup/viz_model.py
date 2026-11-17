"""ILLUSTRATION ONLY - not a stage of DESIGN.md, and no reading is taken from it (one seed, one split; lesson L30).

Trains one lookup block per kind of item on the worlds carrier (`select`, coverage 9, split 0, teacher 10) and writes
the weights to viz_model.json, so that the visualiser of docs/weight_lookup.md can show a trained block's actual
matrices instead of made-up numbers. History Entry 114.

The block is the one of docs/weight_lookup.md, sections 4 and 8, with one site:
  values   K stored blocks; block k has a read matrix B_k [w, d] and a write matrix V_k [d, w]; f_k(h) = V_k tanh(B_k h)
  keys     name:  P f_k(probe)                 - the block's output at one learned probe state, projected
           demo:  f_k(x)                       - the block's output at the item's own state
  queries  name:  the item's token embedding   - load = softmax_k(query . key)
           demo:  the output y the item shows  - load = softmax_k(-|c f_k(x) - y|^2 / tau)
  state    bid of block k = beta_k . tanh(B_k h), its own units; an item's bid is its load-weighted bid;
           share = softmax over the two items
  build    weight of block k = sum over items of share x load; update = sum_k weight_k f_k(h)  (= stacked units, U1)
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "worlds"))
from worlds import DEF, EPS, Teacher, err, sample_worlds  # noqa: E402

torch.set_num_threads(1)
K, W = 8, 16


class Lookup:
    def __init__(self, kind, seed, d, P, mv):
        g = torch.Generator().manual_seed(1000 + seed)
        rn = lambda *s: torch.randn(*s, generator=g)
        p = dict(Bs=rn(K, W, d) / d ** 0.5, Vs=0.1 * rn(K, d, W) / W ** 0.5, beta=0.1 * rn(K, W) / W ** 0.5)
        if kind == "name":
            p.update(E=rn(P, mv) / mv ** 0.5, probe=rn(d), Pm=rn(mv, d) / d ** 0.5)
        else:
            p.update(logtau=torch.zeros(()), c=torch.ones(()))
        self.kind, self.p, self.eps = kind, {k: v.requires_grad_(True) for k, v in p.items()}, EPS["select"]

    def params(self):
        return list(self.p.values())

    def block(self, x):
        """x [..., d] -> unit activations [..., K, W] and block outputs [..., K, d]"""
        z = torch.tanh(torch.einsum("kwj,...j->...kw", self.p["Bs"], x))
        return z, torch.einsum("kiw,...kw->...ki", self.p["Vs"], z)

    def load(self, worlds, demos):
        """[M, B or 1, 2, K]: how much each of the world's two items loads each block"""
        p = self.p
        if self.kind == "name":
            keys = self.block(p["probe"])[1] @ p["Pm"].T                                   # [K, mv]
            return torch.softmax(p["E"][worlds] @ keys.T, -1)[:, None]                     # [M, 1, 2, K]
        x, y = demos
        f = self.block(x)[1]                                                               # [M, B, 2, K, d]
        return torch.softmax(-((p["c"] * f - y[..., None, :]) ** 2).sum(-1) / p["logtau"].exp(), -1)

    def forward(self, worlds, x, n, demos=None):
        h, a = x, self.load(worlds, demos)
        for _ in range(n):
            z, f = self.block(h)                                                           # [M, B, K, W], [M, B, K, d]
            bid = (self.p["beta"] * z).sum(-1)                                             # [M, B, K]
            share = torch.softmax((a * bid[:, :, None, :]).sum(-1), -1)                    # [M, B, 2]
            g = (share[..., None] * a).sum(2)                                              # [M, B, K]
            h = h + self.eps * (g[..., None] * f).sum(2)
        return h


def demos_for(T, worlds, B, gen):
    x = torch.randn(len(worlds), B, 2, T.A.shape[-1], generator=gen)
    y = torch.tanh(torch.einsum("mtij,mbtj->mbti", T.A[worlds], x))
    return x, y


def train(kind, steps, lr=0.01, seed=0):
    c = dict(DEF, task="select")
    T, S = Teacher(c), Lookup(kind, seed, c["d"], c["P"], c["mv"])
    tr, te = sample_worlds(c["P"], c["cover"], c["split"])
    opt = torch.optim.AdamW(S.params(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(5000 + seed)
    for it in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * it / steps))
        x = torch.randn(len(tr), c["B"], c["d"], generator=g)
        dm = demos_for(T, tr, c["B"], g) if kind == "demo" else None
        loss = err(S.forward(tr, x, c["n"], dm), T.run(tr, x, c["n"]), x)
        opt.zero_grad(); loss.backward(); opt.step()

    def ev(ws, tag):
        gg = torch.Generator().manual_seed(777 + tag)
        x = torch.randn(len(ws), 256, c["d"], generator=gg)
        dm = demos_for(T, ws, 256, gg) if kind == "demo" else None
        with torch.no_grad():
            return err(S.forward(ws, x, c["n"], dm), T.run(ws, x, c["n"]), x).item()

    return S, T, tr, te, ev(tr, 0), ev(te, 1), c


def match(S, T):
    """Which teacher primitive does each stored block compute (best scalar allowed)?"""
    x = torch.randn(4000, T.A.shape[-1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        f = S.block(x)[1]                                                                  # [N, K, d]
        t = torch.tanh(torch.einsum("pij,nj->npi", T.A, x))                                # [N, P, d]
        out = []
        for k in range(K):
            best = None
            for p in range(T.A.shape[0]):
                s = (f[:, k] * t[:, p]).sum() / (f[:, k] ** 2).sum().clamp_min(1e-12)
                rel = (((s * f[:, k] - t[:, p]) ** 2).sum() / (t[:, p] ** 2).sum()).item()
                if best is None or rel < best[1]:
                    best = (p, rel, s.item())
            out.append(dict(prim=best[0], rel=round(best[1], 4), scale=round(best[2], 3)))
    return out


if __name__ == "__main__":
    r4 = lambda t: json.loads(json.dumps(t.detach().tolist()), parse_float=lambda s: round(float(s), 4))
    out = {}
    for kind, steps in (("name", 4000), ("demo", 8000)):
        S, T, tr, te, seen, held, c = train(kind, steps)
        print(kind, "seen %.4f held %.4f" % (seen, held), flush=True)
        m = dict(seen=seen, held=held, steps=steps, match=match(S, T), **{k: r4(v) for k, v in S.p.items()})
        if kind == "demo":
            m["tau"] = round(S.p["logtau"].exp().item(), 4)
        # a fixed case for checking the page's own arithmetic against this code
        gg = torch.Generator().manual_seed(42)
        ws = torch.tensor([[1, 3]]); x = torch.randn(1, 1, c["d"], generator=gg)
        dm = demos_for(T, ws, 1, gg) if kind == "demo" else None
        with torch.no_grad():
            m["check"] = dict(world=[1, 3], h0=r4(x[0, 0]), h4=r4(S.forward(ws, x, c["n"], dm)[0, 0]),
                              teacher_h4=r4(T.run(ws, x, c["n"])[0, 0]),
                              demos=None if dm is None else dict(x=r4(dm[0][0, 0]), y=r4(dm[1][0, 0])))
        out[kind] = m
        print("  blocks:", [(b["prim"] + 1, b["rel"]) for b in m["match"]], flush=True)
    out["teacher"] = dict(A=r4(T.A), w=r4(T.w), eps=EPS["select"], n=c["n"], train=tr.tolist(), held=te.tolist())
    json.dump(out, open(os.path.join(HERE, "viz_model.json"), "w"), separators=(",", ":"))
