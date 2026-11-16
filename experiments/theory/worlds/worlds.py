"""The "worlds" carrier (experiments/theory/worlds/DESIGN.md). EXPLORATORY.

A world is a set of two primitives out of P, stated in the input. Some worlds are never trained on (every primitive
is). One looped block is applied n times; on each pass it is given the state h and a read r of the world (the sum of
the two primitives' learned embeddings) - nothing else, no clock, no shared controller.

Tasks (teacher primitives f_p(h) = tanh(A_p h), triggers w_p):
  select   each pass applies ONE of the world's two primitives: the one whose trigger <w_p, h> is larger. Which one
           is not named anywhere; it depends on the state and changes from pass to pass.
  sum      each pass applies BOTH primitives of the world, summed.
Arms (what the block is): mono | streams_joint | streams_self | unit | oracle - see DESIGN.md.
Error: squared error of the change h_n - h_0 relative to the teacher's change.
"""
import itertools
import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

DEF = dict(task="select", d=6, P=6, n=4, mv=8, arm="mono", score="softmax",
           W=40, K=8, w=16, H=36, q=13, Wu=60, Hg=16,
           steps=4000, lr=0.01, B=24, wd=0.0, cover=9, tseed=10)
EPS = {"select": 0.4, "sum": 0.2}


def all_worlds(P):
    return list(itertools.combinations(range(P), 2))


def graph_ok(P, edges):
    """Connected, every primitive in at least two worlds, and not bipartite (contains an odd cycle)."""
    adj = {i: set() for i in range(P)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    if min(len(v) for v in adj.values()) < 2:
        return False
    colour, stack, odd = {0: 0}, [0], False
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in colour:
                colour[v] = 1 - colour[u]; stack.append(v)
            elif colour[v] == colour[u]:
                odd = True
    return len(colour) == P and odd


def sample_worlds(P, cover, seed):
    """(training worlds, held-out worlds) as tensors [*, 2]."""
    allw = all_worlds(P)
    if cover >= len(allw):
        return torch.tensor(allw), torch.zeros(0, 2, dtype=torch.long)
    g = torch.Generator().manual_seed(9000 + 131 * seed + cover)
    while True:
        perm = torch.randperm(len(allw), generator=g).tolist()
        tr = sorted(allw[i] for i in perm[:cover])
        if graph_ok(P, tr):
            te = sorted(allw[i] for i in perm[cover:])
            return torch.tensor(tr), torch.tensor(te)


class Teacher:
    def __init__(self, cfg):
        g = torch.Generator().manual_seed(cfg["tseed"])
        P, d = cfg["P"], cfg["d"]
        self.A = torch.randn(P, d, d, generator=g) / d ** 0.5
        w = torch.randn(P, d, generator=g)
        self.w = w / w.norm(dim=1, keepdim=True)
        self.task, self.eps = cfg["task"], EPS[cfg["task"]]

    def choice(self, worlds, h):
        """Index (0 or 1) into the world of the primitive the select task applies at state h. [M, B]"""
        s = torch.einsum("mbj,mkj->mbk", h, self.w[worlds])              # [M, B, 2]
        return (s[..., 1] > s[..., 0]).long()

    def step(self, worlds, h, policy="true"):
        f = torch.tanh(torch.einsum("mkij,mbj->mbki", self.A[worlds], h))  # [M, B, 2, d]
        if policy == "blend":
            upd = f.mean(2) * (1 if self.task == "select" else 2)
        elif policy == "first":
            upd = f[:, :, 0]
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
            elif arm == "streams_self":       # each stream scores itself from the state; the world only adds a bias
                q = c["q"]
                p.update(G=0.1 * rn(K, mv), S1=rn(K, q, d) / d ** 0.5, sb=torch.zeros(K, q), S2=0.1 * rn(K, q) / q ** 0.5)
        self.p = {k: v.requires_grad_(True) for k, v in p.items()}
        self.K = None if arm in ("mono", "unit") else (P if arm == "oracle" else c["K"])
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
        for _ in range(n):
            if c["arm"] == "mono":
                z = torch.tanh(torch.cat([h, r], -1) @ p["B1"].T + p["b1"])
                out = torch.tanh(z @ p["B2"].T + p["b2"]) @ p["V"].T
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


def references(cfg, worlds, seed=0):
    """Errors, on the given worlds, of policies that know the world's two primitives but do not compose them."""
    c = dict(DEF); c.update(cfg)
    T = Teacher(c)
    x = torch.randn(len(worlds), 2048, c["d"], generator=torch.Generator().manual_seed(seed))
    tgt = T.run(worlds, x, c["n"])
    out = {"blend": err(T.run(worlds, x, c["n"], "blend"), tgt, x).item() if c["task"] == "select" else float("nan")}
    f = err(T.run(worlds, x, c["n"], "first"), tgt, x).item()
    s = err(T.run(worlds.flip(1), x, c["n"], "first"), tgt, x).item()
    out["one"] = 0.5 * (f + s)
    return out


def run(over, seed, ret_model=False):
    c = dict(DEF); c.update(over)
    T, S = Teacher(c), Model(c, seed)
    tr, te = sample_worlds(c["P"], c["cover"], seed)
    opt = torch.optim.AdamW(S.params(), lr=c["lr"], weight_decay=c["wd"])
    g = torch.Generator().manual_seed(5000 + seed)
    curve = {}
    for it in range(c["steps"]):
        for gr in opt.param_groups:
            gr["lr"] = c["lr"] * 0.5 * (1 + math.cos(math.pi * it / c["steps"]))
        x = torch.randn(len(tr), c["B"], c["d"], generator=g)
        loss = err(S.forward(tr, x, c["n"], T), T.run(tr, x, c["n"]), x)
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward(); opt.step()
    out = dict(nparams=sum(v.numel() for v in S.params()), ntrain=len(tr), nheld=len(te))
    with torch.no_grad():
        def ev(ws, scale=1.0):
            if len(ws) == 0:
                return float("nan")
            x = scale * torch.randn(len(ws), 256, c["d"], generator=g)
            v = err(S.forward(ws, x, c["n"], T), T.run(ws, x, c["n"]), x).item()
            return v if math.isfinite(v) else float("inf")
        out["seen"], out["held"] = ev(tr), ev(te)
        if len(te):
            out["ref"] = references(c, te)
    return (out, S, T, tr, te) if ret_model else out
