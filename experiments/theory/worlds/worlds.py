"""The "worlds" carrier (experiments/theory/worlds/DESIGN.md). EXPLORATORY.

A world is a set of two primitives out of P, stated in the input. Some worlds are never trained on (every primitive
is). One looped block is applied n times; on each pass it is given the state h and a read r of the world (the sum of
the two primitives' learned embeddings) - nothing else, no clock, no shared controller.

Tasks (teacher primitives f_p(h) = tanh(A_p h), triggers w_p):
  select   each pass applies ONE of the world's two primitives: the one whose trigger <w_p, h> is larger. Which one
           is not named anywhere; it depends on the state and changes from pass to pass.
  sum      each pass applies BOTH primitives of the world, summed.
Arms: read of the world (summed, or one branch per token) x block (one monolithic block, or streams):
  mono | streams_joint | streams_self | unit | oracle      summed read r = E[a] + E[b]
  tok_mono | tok_streams                                    one branch per token, pooled by a softmax over bids
See DESIGN.md.
Error: squared error of the change h_n - h_0 relative to the teacher's change.
"""
import itertools
import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

DEF = dict(task="select", d=6, P=6, n=4, mv=8, arm="mono", score="softmax",
           W=40, K=8, w=16, H=36, q=13, Wu=60, Hg=16,
           steps=4000, lr=0.01, B=24, wd=0.0, cover=9, tseed=10, split=0, same=False, ckpts=())
EPS = {"select": 0.4, "sum": 0.2}


def all_worlds(P):
    return list(itertools.combinations(range(P), 2))


def graph_ok(P, edges):
    """2-connected (connected, and still connected with any one primitive removed) and not bipartite."""
    def connected(nodes, es):
        nodes = set(nodes)
        adj = {i: set() for i in nodes}
        for a, b in es:
            if a in nodes and b in nodes:
                adj[a].add(b); adj[b].add(a)
        start = next(iter(nodes)); colour, stack, odd = {start: 0}, [start], False
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in colour:
                    colour[v] = 1 - colour[u]; stack.append(v)
                elif colour[v] == colour[u]:
                    odd = True
        return len(colour) == len(nodes), odd
    conn, odd = connected(range(P), edges)
    return conn and odd and all(connected([i for i in range(P) if i != v], edges)[0] for v in range(P))


def world_order(P, split, first=7):
    """A random order of all worlds whose first `first` worlds already meet graph_ok; training sets of every
    coverage are prefixes of it, so the dial is nested."""
    allw = all_worlds(P)
    g = torch.Generator().manual_seed(9000 + split)
    while True:
        order = [allw[i] for i in torch.randperm(len(allw), generator=g).tolist()]
        if graph_ok(P, order[:first]):
            return order


def sample_worlds(P, cover, split):
    """(training worlds, held-out worlds) as tensors [*, 2]; depends on the split only, not on the model seed."""
    order = world_order(P, split)
    tr, te = sorted(order[:cover]), sorted(order[cover:])
    return torch.tensor(tr), (torch.tensor(te) if te else torch.zeros(0, 2, dtype=torch.long))


class Teacher:
    def __init__(self, cfg):
        g = torch.Generator().manual_seed(cfg["tseed"])
        P, d = cfg["P"], cfg["d"]
        self.A = torch.randn(P, d, d, generator=g) / d ** 0.5
        w = torch.randn(P, d, generator=g)
        self.w = w / w.norm(dim=1, keepdim=True)
        if cfg["same"]:                          # control: all primitives identical, so the world is irrelevant
            self.A = self.A[:1].expand(P, -1, -1).clone(); self.w = self.w[:1].expand(P, -1).clone()
        self.task, self.eps = cfg["task"], EPS[cfg["task"]]
        self.trained = None                      # [P, P] 0/1 matrix of trained worlds, for the transitivity policy

    def determined(self, worlds, h):
        """select: is the choice in world {a, b} at state h forced by trained worlds through a chain
        a > c > ... > b of trained comparisons?  [M, B] bool"""
        s = h @ self.w.T                                                 # [M, B, P]
        D = ((s[..., :, None] > s[..., None, :]) & (self.trained[None, None] > 0)).double()
        Rm = D.clone()
        for _ in range(s.shape[-1] - 2):
            Rm = ((Rm + Rm @ D) > 0).double()
        a = worlds[:, 0][:, None].expand(-1, h.shape[1]); b = worlds[:, 1][:, None].expand(-1, h.shape[1])
        m = torch.arange(h.shape[0])[:, None].expand(-1, h.shape[1]); bb = torch.arange(h.shape[1])[None, :].expand(h.shape[0], -1)
        return (Rm[m, bb, a, b] + Rm[m, bb, b, a]) > 0

    def choice(self, worlds, h):
        """Index (0 or 1) into the world of the primitive the select task applies at state h. [M, B]"""
        s = torch.einsum("mbj,mkj->mbk", h, self.w[worlds])              # [M, B, 2]
        return (s[..., 1] > s[..., 0]).long()

    def step(self, worlds, h, policy="true"):
        f = torch.tanh(torch.einsum("mkij,mbj->mbki", self.A[worlds], h))  # [M, B, 2, d]
        scale = 1 if self.task == "select" else 2
        if policy == "blend":
            upd = f.mean(2) * scale
        elif policy == "allmean":                # ignores the world: the mean of all primitives
            upd = torch.tanh(torch.einsum("pij,mbj->mbpi", self.A, h)).mean(2) * scale
        elif policy == "first":
            upd = f[:, :, 0]
        elif policy == "trans":                  # select: the true choice where trained worlds force it, else the blend
            c = self.choice(worlds, h)
            true = torch.where(c[..., None] == 0, f[:, :, 0], f[:, :, 1])
            upd = torch.where(self.determined(worlds, h)[..., None], true, f.mean(2))
        elif self.task == "sum":
            upd = f.sum(2)
        else:
            c = self.choice(worlds, h)
            upd = torch.where(c[..., None] == 0, f[:, :, 0], f[:, :, 1])
        return h + self.eps * upd

    def run(self, worlds, x, n, policy="true"):
        h = x
        for _ in range(n):
            h = self.step(worlds, h, policy)
        return h


class Model:
    def __init__(self, cfg, seed):
        self.c = c = cfg
        g = torch.Generator().manual_seed(1000 + seed)
        rn = lambda *s: torch.randn(*s, generator=g)
        d, mv, P = c["d"], c["mv"], c["P"]
        p = {"E": rn(P, mv) / mv ** 0.5}
        arm = c["arm"]
        if arm == "mono":
            W = c["W"]
            p.update(B1=rn(W, d + mv) / (d + mv) ** 0.5, b1=torch.zeros(W), B2=rn(W, W) / W ** 0.5, b2=torch.zeros(W),
                     V=0.1 * rn(d, W) / W ** 0.5)
        elif arm == "tok_mono":                   # the same block once per token, on [h; E[p]]; it also emits a bid
            W = c["W"]
            p.update(B1=rn(W, d + mv) / (d + mv) ** 0.5, b1=torch.zeros(W), B2=rn(W, W) / W ** 0.5, b2=torch.zeros(W),
                     V=0.1 * rn(d, W) / W ** 0.5, vb=0.1 * rn(W) / W ** 0.5)
        elif arm == "unit":
            W, Hg = c["Wu"], c["Hg"]
            p.update(uB=rn(W, d) / d ** 0.5, uV=0.1 * rn(d, W) / W ** 0.5, uC=rn(W, mv) / mv ** 0.5, ub=torch.zeros(W),
                     gW1=rn(Hg, mv + d) / (mv + d) ** 0.5, gb1=torch.zeros(Hg), gW2=0.1 * rn(W, Hg) / Hg ** 0.5)
        else:
            K = P if arm == "oracle" else c["K"]
            w = c["w"]
            p.update(Bs=rn(K, w, d) / d ** 0.5, Vs=0.1 * rn(K, d, w) / w ** 0.5)
            if arm == "streams_joint":
                H = c["H"]
                p.update(G=0.1 * rn(K, mv), W1=rn(H, mv + d) / (mv + d) ** 0.5, b1=torch.zeros(H), W2=0.1 * rn(K, H) / H ** 0.5)
            elif arm in ("streams_self", "tok_streams"):   # each stream scores itself from the state
                q = c["q"]
                p.update(G=0.1 * rn(K, mv), S1=rn(K, q, d) / d ** 0.5, sb=torch.zeros(K, q), S2=0.1 * rn(K, q) / q ** 0.5)
        self.p = {k: v.requires_grad_(True) for k, v in p.items()}
        self.K = None if arm in ("mono", "unit", "tok_mono") else (P if arm == "oracle" else c["K"])
        self.eps = EPS[c["task"]]

    def params(self):
        return list(self.p.values())

    def gate(self, worlds, r, h, teacher):
        c, p = self.c, self.p
        if c["arm"] == "oracle":                 # the gate is set from the truth: stream k IS primitive k
            g = torch.zeros(h.shape[0], h.shape[1], self.K)
            if c["task"] == "sum":
                idx = worlds[:, None, :].expand(-1, h.shape[1], -1)
            else:
                ch = teacher.choice(worlds, h.detach())
                idx = torch.gather(worlds[:, None, :].expand(-1, h.shape[1], -1), 2, ch[..., None])
            return g.scatter(2, idx, 1.0), None
        if c["arm"] == "streams_joint":
            s = r @ p["G"].T + torch.tanh(torch.cat([r, h], -1) @ p["W1"].T + p["b1"]) @ p["W2"].T
        else:
            sc = torch.einsum("kq,mbkq->mbk", p["S2"], torch.tanh(torch.einsum("kqj,mbj->mbkq", p["S1"], h) + p["sb"]))
            s = r @ p["G"].T + sc
        return (torch.softmax(s, -1) if c["score"] == "softmax" else torch.sigmoid(s)), s

    def forward(self, worlds, x, n, teacher=None, record=False):
        c, p = self.c, self.p
        r = p["E"][worlds].sum(1)[:, None, :].expand(-1, x.shape[1], -1)        # [M, B, mv]
        h, gates = x, []
        self.pools, self.assign = [], None       # filled when record=True (per-token arms)
        for _ in range(n):
            if c["arm"] == "mono":
                z = torch.tanh(torch.cat([h, r], -1) @ p["B1"].T + p["b1"])
                out = torch.tanh(z @ p["B2"].T + p["b2"]) @ p["V"].T
            elif c["arm"] == "tok_mono":
                e = p["E"][worlds][:, None, :, :].expand(-1, x.shape[1], -1, -1)             # [M, B, 2, mv]
                z = torch.tanh(torch.cat([h[:, :, None, :].expand(-1, -1, 2, -1), e], -1) @ p["B1"].T + p["b1"])
                z = torch.tanh(z @ p["B2"].T + p["b2"])
                pool = torch.softmax(z @ p["vb"], -1)                                        # [M, B, 2]
                out = (pool[..., None] * (z @ p["V"].T)).sum(2)
                if record:
                    gates.append(pool.detach()); self.pools.append(pool.detach())
            elif c["arm"] == "tok_streams":
                a = torch.softmax(p["E"][worlds] @ p["G"].T, -1)                             # [M, 2, K] token -> streams
                f = torch.einsum("kiw,mbkw->mbki", p["Vs"], torch.tanh(torch.einsum("kwj,mbj->mbkw", p["Bs"], h)))
                sc = torch.einsum("kq,mbkq->mbk", p["S2"], torch.tanh(torch.einsum("kqj,mbj->mbkq", p["S1"], h) + p["sb"]))
                pool = torch.softmax(torch.einsum("mtk,mbk->mbt", a, sc), -1)                # bids of the two tokens
                g = torch.einsum("mbt,mtk->mbk", pool, a)                                    # effective gate on streams
                out = (g[..., None] * f).sum(2)
                if record:
                    gates.append(g.detach()); self.pools.append(pool.detach()); self.assign = a.detach()
            elif c["arm"] == "unit":
                lg = r @ p["uC"].T + p["ub"] + torch.tanh(torch.cat([r, h], -1) @ p["gW1"].T + p["gb1"]) @ p["gW2"].T
                out = (torch.sigmoid(lg) * torch.tanh(h @ p["uB"].T)) @ p["uV"].T
            else:
                g, _ = self.gate(worlds, r, h, teacher)
                f = torch.einsum("kiw,mbkw->mbki", p["Vs"], torch.tanh(torch.einsum("kwj,mbj->mbkw", p["Bs"], h)))
                out = (g[..., None] * f).sum(2)
                if record:
                    gates.append(g.detach())
            h = h + self.eps * out
        return (h, gates) if record else h


def err(h, tgt, x):
    return ((h - tgt) ** 2).sum() / ((tgt - x) ** 2).sum()


def references(cfg, tr, te, seed=0):
    """Errors on held-out worlds of policies that do not compose. no change = 1 by definition.
    allmean: ignores the world. blend: knows the two primitives, cannot choose (select). one: applies one of the two.
    trans (select): chooses correctly wherever trained worlds force the choice by transitivity, blends elsewhere."""
    c = dict(DEF); c.update(cfg)
    T = Teacher(c)
    T.trained = torch.zeros(c["P"], c["P"])
    for a, b in tr.tolist():
        T.trained[a, b] = T.trained[b, a] = 1.0
    x = torch.randn(len(te), 2048, c["d"], generator=torch.Generator().manual_seed(seed))
    tgt = T.run(te, x, c["n"])
    e = lambda pol, ws=te: err(T.run(ws, x, c["n"], pol), tgt, x).item()
    out = {"allmean": e("allmean"), "one": 0.5 * (e("first") + e("first", te.flip(1)))}
    if c["task"] == "select":
        out["blend"] = e("blend"); out["trans"] = e("trans")
        out["forced"] = T.determined(te, x).double().mean().item()        # share of first-pass decisions that are forced
    return out


def run(over, seed, ret_model=False):
    c = dict(DEF); c.update(over)
    T, S = Teacher(c), Model(c, seed)
    tr, te = sample_worlds(c["P"], c["cover"], c["split"])
    opt = torch.optim.AdamW(S.params(), lr=c["lr"], weight_decay=c["wd"])
    g = torch.Generator().manual_seed(5000 + seed)

    def ev(ws, nb=256, tag=0):
        if len(ws) == 0:
            return float("nan"), []
        x = torch.randn(len(ws), nb, c["d"], generator=torch.Generator().manual_seed(777 + c["split"] + tag))
        with torch.no_grad():
            h, tgt = S.forward(ws, x, c["n"], T), T.run(ws, x, c["n"])
            per = (((h - tgt) ** 2).sum((1, 2)) / ((tgt - x) ** 2).sum((1, 2))).tolist()
            v = err(h, tgt, x).item()
        return (v if math.isfinite(v) else float("inf")), per

    curve = {}
    for it in range(c["steps"]):
        for gr in opt.param_groups:
            gr["lr"] = c["lr"] * 0.5 * (1 + math.cos(math.pi * it / c["steps"]))
        x = torch.randn(len(tr), c["B"], c["d"], generator=g)
        loss = err(S.forward(tr, x, c["n"], T), T.run(tr, x, c["n"]), x)
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) in c["ckpts"]:
            curve[it + 1] = (ev(tr, 64, 1)[0], ev(te, 64, 1)[0])
    out = dict(nparams=sum(v.numel() for v in S.params()), ntrain=len(tr), nheld=len(te), curve=curve)
    out["seen"], out["seen_per"] = ev(tr)
    out["held"], out["held_per"] = ev(te)
    if len(te):
        out["ref"] = references(c, tr, te)
        # one pass on held-out worlds: regress the block's update on the six true primitive outputs at the same state
        with torch.no_grad():
            x = torch.randn(len(te), 256, c["d"], generator=torch.Generator().manual_seed(4242))
            upd = (S.forward(te, x, 1, T) - x) / S.eps                                      # [M, B, d]
            F = torch.tanh(torch.einsum("pij,mbj->mbpi", T.A, x))                            # [M, B, P, d]
            r2, mass = [], []
            for m in range(len(te)):
                X = F[m].permute(0, 2, 1).reshape(-1, c["P"]); y = upd[m].reshape(-1, 1)
                coef = torch.linalg.lstsq(X, y).solution
                r2.append(1 - ((X @ coef - y) ** 2).sum().item() / (y ** 2).sum().item())
                mass.append((coef[te[m]].abs().sum() / coef.abs().sum()).item())
            out["r2"], out["mass_on_pair"] = sum(r2) / len(r2), sum(mass) / len(mass)
            # one pass: is the block's update closer to the true update or to the blend of the world's two primitives?
            true, blend = (T.step(te, x) - x) / T.eps, (T.step(te, x, "blend") - x) / T.eps
            den = (true ** 2).sum().item()
            out["one_true"], out["one_blend"] = ((upd - true) ** 2).sum().item() / den, ((upd - blend) ** 2).sum().item() / den
    return (out, S, T, tr, te) if ret_model else out
