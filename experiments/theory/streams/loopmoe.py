"""Testbed for the architecture search on streams gated inside one looped block (docs/moe_in_loops.md). EXPLORATORY.

One looped block. K streams read the same state h and write back a gated sum:

    h_{t+1} = h_t + eps * sum_k g_{t,k} f_k(h_t),          f_k(h) = V_k tanh(B_k h).

The gate is computed BEFORE the streams (so a stream whose gate is zero need not be run) from two things:
  the original sequence   tokens u_1..u_n, embedded as values E[u_i] and addressed through keys (one per position);
  the state               a query built from the state attends over the keys; the value it reads back, and optionally
                          the (normalised) data state itself, give the K selection logits.
What the query is built from is the `progress` option:
  none     a constant query: the gate sees the whole program but not where the loop is (order-blind reference)
  clock    one learned query per pass (the analogue of "one router per loop")
  data     a linear read of the normalised data state
  reg0     a control register c, part of the state, advanced every pass by an always-on linear map: c <- c + M c;
           M starts near zero (the register does not move at initialisation)
  regrot   the same, M starts as a random rotation minus identity (the register moves from the first step), learned
  regfix   the same, M frozen at its initial rotation (a clock carried in the state)

Tasks. Teacher primitives are tanh(A_p h).
  program  the operation of pass t is the t-th token of the program
  branch   one extra token "IF": apply primitive 0 if <w, h> > 0, else primitive 1 (needs a gate that reads the data)

Error: squared error of the change h_n - h_0 relative to the teacher's change (1 = learned nothing, 0 = exact),
on programs never trained on.
"""
import math
import torch
from lib import all_programs, split

torch.set_num_threads(1)

DEF = dict(task="program", d=6, P=3, n=4, eps=0.4, K=4, w=12, mc=8, mv=8, H=24,
           progress="clock", gate_h=False, ln=True, ln_h=None, keys="learned",
           score="softmax", topk=0, renorm=False,
           noise=0.0, xdrop=0.0, ent=0.0, zloss=0.0, fill="none", anneal=0.0,
           sched=None, explore="none", topk_eval=None, auto=0.0, tseed=10, probe=False,
           steps=3000, lr=3e-3, B=16, ntrain=40, clip=0.0, lens=None, test_lens=None)
LMAX = 16


def rms(h):
    return h / (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


def sinus_keys(mc, L=LMAX):
    i = torch.arange(L)[:, None]
    j = torch.arange(1, mc // 2 + 1)[None, :]
    ang = 2 * math.pi * i * j / L
    return torch.cat([ang.cos(), ang.sin()], 1)                      # [L, mc]


def rot_minus_id(angles):
    """Block rotation acting on vectors laid out as [cos-parts, sin-parts]; returns R - I."""
    m = len(angles)
    R = torch.zeros(2 * m, 2 * m)
    for j, a in enumerate(angles):
        R[j, j] = math.cos(a); R[j, m + j] = -math.sin(a)
        R[m + j, j] = math.sin(a); R[m + j, m + j] = math.cos(a)
    return R - torch.eye(2 * m)


class Teacher:
    def __init__(self, cfg, seed=None):
        g = torch.Generator().manual_seed(cfg["tseed"] if seed is None else seed)
        self.A = torch.randn(cfg["P"], cfg["d"], cfg["d"], generator=g) / cfg["d"] ** 0.5
        w = torch.randn(cfg["d"], generator=g)
        self.w = w / w.norm()
        self.P, self.eps, self.branch = cfg["P"], cfg["eps"], cfg["task"] == "branch"

    def __call__(self, progs, x):
        h = x
        for t in range(progs.shape[1]):
            u = progs[:, t][:, None].expand(-1, h.shape[1])
            if self.branch:
                u = torch.where(u == self.P, torch.where(h @ self.w > 0, 0, 1), u)
            h = h + self.eps * torch.tanh(torch.einsum("mbij,mbj->mbi", self.A[u], h))
        return h


class Student:
    def __init__(self, cfg, seed):
        self.c = c = cfg
        g = torch.Generator().manual_seed(1000 + seed)
        rn = lambda *s: torch.randn(*s, generator=g)
        d, K, w, mc, mv, H = c["d"], c["K"], c["w"], c["mc"], c["mv"], c["H"]
        ntok = c["P"] + (1 if c["task"] == "branch" else 0)
        p = {}
        p["B"] = rn(K, w, d) / d ** 0.5
        p["V"] = 0.1 * rn(K, d, w) / w ** 0.5
        p["E"] = rn(ntok, mv) / mv ** 0.5
        p["G"] = 0.1 * rn(K, mv)
        frozen = set()
        if c["keys"] == "learned":
            p["keys"] = rn(LMAX, mc) / mc ** 0.5
        else:
            p["keys"] = sinus_keys(mc); frozen.add("keys")
        pr = c["progress"]
        if pr == "none":
            p["q0"] = rn(mc) / mc ** 0.5
        elif pr == "clock":
            p["Q"] = 0.1 * rn(LMAX, mc)
        elif pr == "data":
            p["Wq"] = rn(mc, d) / d ** 0.5
        else:
            p["c0"] = rn(mc) / mc ** 0.5
            p["beta"] = torch.tensor(1.0)
            if pr == "reg0":
                p["M"] = 0.01 * rn(mc, mc)
            else:
                ang = (0.25 + 0.5 * torch.rand(mc // 2, generator=g)) * math.pi
                p["M"] = rot_minus_id(ang.tolist())
                if pr == "regfix":
                    frozen.add("M")
                if pr == "regsin":                                    # the hand-aligned shift: one position per pass
                    p["M"] = rot_minus_id([2 * math.pi * j / LMAX for j in range(1, mc // 2 + 1)])
                    p["c0"] = sinus_keys(mc)[0].clone(); p["beta"] = torch.tensor(3.0)
        if c["gate_h"]:
            p["W1"] = rn(H, mv + d) / (mv + d) ** 0.5
            p["b1"] = torch.zeros(H)
            p["W2"] = 0.1 * rn(K, H) / H ** 0.5
        if c["fill"] == "lin":
            p["L"] = torch.zeros(K, d, d)
        self.p = {k: (v if k in frozen else v.requires_grad_(True)) for k, v in p.items()}
        self.ema = torch.zeros(K, d)
        self.g = g
        self.k_now, self.explore_now = c["topk"], c["explore"] != "none"

    def params(self):
        return [v for v in self.p.values() if v.requires_grad]

    def gate(self, s, train, hard):
        """s [M,B,K] logits -> (gates g [M,B,K] with zeros for streams that are not run, mask, soft scores)."""
        c = self.c
        sel = s
        if train and c["noise"] > 0:
            sel = sel + c["noise"] * torch.randn(s.shape, generator=self.g)
        drop = None
        if train and c["xdrop"] > 0:
            drop = torch.rand(s.shape, generator=self.g) < c["xdrop"]
            sel = sel.masked_fill(drop, -1e9)
        if c["score"] == "softmax":
            soft = torch.softmax(s if drop is None else s.masked_fill(drop, -1e9), -1)
        else:
            soft = torch.sigmoid(s)
            if drop is not None:
                soft = soft * (~drop)
        k = self.k_now if train else (c["topk_eval"] if c["topk_eval"] is not None else c["topk"])
        if k and hard and train and self.explore_now:
            # run the top stream plus one explorer: sampled from the gate's own distribution, or uniformly
            top = sel.argmax(-1, keepdim=True)
            pe = torch.softmax(s.detach(), -1) if c["explore"] == "sample" else torch.ones_like(s)
            pe = pe.scatter(-1, top, 0.0) + 1e-12
            ex = torch.multinomial(pe.reshape(-1, pe.shape[-1]), 1, generator=self.g).reshape(top.shape)
            mask = torch.zeros_like(s).scatter(-1, top, 1.0).scatter(-1, ex, 1.0)
        elif k and hard:
            idx = sel.topk(k, -1).indices
            mask = torch.zeros_like(s).scatter(-1, idx, 1.0)
        else:
            mask = torch.ones_like(s)
        gg = soft * mask
        if c["renorm"] and c["score"] == "softmax":
            gg = gg / gg.sum(-1, keepdim=True)
        return gg, mask, soft

    def forward(self, progs, x, train=False, hard=True, record=False):
        c, p = self.c, self.p
        M, Bx, n = progs.shape[0], x.shape[1], progs.shape[1]
        vals, keys = p["E"][progs], p["keys"][:n]
        h = x
        pr = c["progress"]
        if pr.startswith("reg"):
            reg = p["c0"].expand(M, Bx, -1)
        aux = dict(logits=[], gates=[], attn=[], states=[], surr=0.0, ran=0.0)
        for t in range(n):
            h_in = h
            hn = rms(h) if (c["ln"] if c["ln_h"] is None else c["ln_h"]) else h
            if pr == "none":
                q = p["q0"].expand(M, Bx, -1)
            elif pr == "clock":
                q = p["Q"][t].expand(M, Bx, -1)
            elif pr == "data":
                q = hn @ p["Wq"].T
            else:
                q = p["beta"] * (rms(reg) if c["ln"] else reg)
            a = torch.softmax(q @ keys.T, -1)                          # [M,B,n]
            r = torch.einsum("mbi,miv->mbv", a, vals)
            s = r @ p["G"].T
            if c["gate_h"]:
                s = s + torch.tanh(torch.cat([r, hn], -1) @ p["W1"].T + p["b1"]) @ p["W2"].T
            g, mask, soft = self.gate(s, train, hard)
            f = torch.einsum("kiw,mbkw->mbki", p["V"], torch.tanh(torch.einsum("kwj,mbj->mbkw", p["B"], h)))
            out = (g[..., None] * f).sum(2)
            if c["fill"] != "none" and c["topk"] and hard:   # stand-ins are not combined with schedules
                un = soft * (1 - mask)
                if c["fill"] == "ema":
                    out = out + un @ self.ema
                    if train:
                        with torch.no_grad():
                            cnt = mask.sum((0, 1)).clamp(min=1)
                            mean = (mask[..., None] * f).sum((0, 1)) / cnt[:, None]
                            upd = (mask.sum((0, 1)) > 0)[:, None]
                            self.ema = torch.where(upd, 0.9 * self.ema + 0.1 * mean, self.ema)
                else:
                    lin = torch.einsum("kij,mbj->mbki", p["L"], h.detach())
                    out = out + (un[..., None] * lin.detach()).sum(2)
                    aux["surr"] = aux["surr"] + (mask[..., None] * (lin - f.detach()) ** 2).sum() / mask.sum().clamp(min=1)
            h = h + c["eps"] * out
            if pr.startswith("reg"):
                reg = reg + reg @ p["M"].T
            aux["logits"].append(s); aux["ran"] += mask.sum(-1).mean().item() / n
            if record:
                aux["gates"].append(g.detach()); aux["attn"].append(a.detach()); aux["states"].append(h_in)
        return h, aux


def make_programs(cfg, seed):
    ntok = cfg["P"] + (1 if cfg["task"] == "branch" else 0)
    g = torch.Generator().manual_seed(77 + seed)
    if cfg["lens"]:                                                  # mixed lengths: random programs per length
        tr = {L: torch.randint(ntok, (cfg["ntrain"], L), generator=g) for L in cfg["lens"]}
        te = {L: torch.randint(ntok, (100, L), generator=g) for L in cfg["test_lens"]}
        return tr, te
    allp = all_programs(ntok, cfg["n"])
    tr, te = split(allp, cfg["ntrain"] / len(allp), seed)
    return {cfg["n"]: tr}, {cfg["n"]: te}


def err(h, tgt, x):
    return ((h - tgt) ** 2).sum() / ((tgt - x) ** 2).sum()


def gains(net, progs, x):
    """Spectral norm of d h_n / d h_0, one per (program, input)."""
    out = []
    for m in range(len(progs)):
        for b in range(x.shape[1]):
            f = lambda v: net(progs[m:m + 1], v.reshape(1, 1, -1)).reshape(-1)
            out.append(torch.linalg.matrix_norm(torch.autograd.functional.jacobian(f, x[m, b]), 2).item())
    return sorted(out)


def probe(S, pr, cfg):
    """With every stream run: is the address learned, and does each token have its own top stream?"""
    with torch.no_grad():
        x = torch.randn(len(pr), 16, cfg["d"], generator=torch.Generator().manual_seed(999))
        _, aux = S.forward(pr, x, hard=False, record=True)
        top = torch.stack(aux["gates"], 1).mean(2).argmax(-1)                       # [M, n]
        att = torch.stack(aux["attn"], 1).mean(2).argmax(-1)                        # [M, n]
        addr = (att == torch.arange(pr.shape[1])[None, :]).double().mean().item()
        ntok = int(pr.max()) + 1
        tab = torch.stack([torch.bincount(top[pr == u], minlength=cfg["K"]).double() for u in range(ntok)])
        purity = (tab.max(1).values.sum() / tab.sum()).item()
        modal = tab.argmax(1)
        conf = torch.softmax(torch.stack(aux["logits"]), -1).max(-1).values.mean().item()
        return dict(addr=addr, purity=purity, one2one=bool(len(set(modal.tolist())) == ntok and purity > 0.9), conf=conf)


def run(over, seed, ret_model=False):
    cfg = dict(DEF); cfg.update(over)
    T = Teacher(cfg)
    S = Student(cfg, seed)
    tr, te = make_programs(cfg, seed)
    opt = torch.optim.Adam(S.params(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(5000 + seed)
    lens = list(tr.keys())
    diverged = False
    switch = None
    at_switch = None
    ran_sum = 0.0
    for it in range(cfg["steps"]):
        L = lens[it % len(lens)]
        pr = tr[L]
        x = torch.randn(len(pr), cfg["B"], cfg["d"], generator=g)
        hard = it >= cfg["anneal"] * cfg["steps"]
        if cfg["auto"]:                      # dense until the gate is confident (mean top probability > auto), then sparse
            hard = switch is not None
        if cfg["probe"] and hard and at_switch is None:
            at_switch = probe(S, pr, cfg)
        if cfg["sched"]:                     # list of (fraction of training, k, explorer on?), k = 0 meaning dense
            for frac, k, ex in cfg["sched"]:
                if it >= frac * cfg["steps"]:
                    S.k_now, S.explore_now = k, ex
        h, aux = S.forward(pr, x, train=True, hard=hard)
        loss = err(h, T(pr, x), x)
        ran_sum += aux["ran"]
        if cfg["ent"] or cfg["zloss"]:
            s = torch.stack(aux["logits"])
            if cfg["ent"]:
                pbar = torch.softmax(s, -1).mean((0, 1, 2))
                loss = loss + cfg["ent"] * (pbar * (pbar + 1e-12).log()).sum()
            if cfg["zloss"]:
                loss = loss + cfg["zloss"] * torch.logsumexp(s, -1).pow(2).mean()
        if cfg["fill"] == "lin":
            loss = loss + aux["surr"]
        if cfg["auto"] and switch is None:
            with torch.no_grad():
                conf = torch.softmax(torch.stack(aux["logits"]), -1).max(-1).values.mean().item()
            if conf > cfg["auto"] or it >= cfg["steps"] // 2:
                switch = it + 1
        if not torch.isfinite(loss):
            diverged = True; break
        opt.zero_grad(); loss.backward()
        if cfg["clip"]:
            torch.nn.utils.clip_grad_norm_(S.params(), cfg["clip"])
        opt.step()
    out = dict(diverged=diverged, switch=switch, train_ran=ran_sum / cfg["steps"])
    if cfg["probe"]:
        out["at_switch"] = at_switch
        out["at_end"] = probe(S, tr[lens[0]], cfg)
    with torch.no_grad():
        def ev(prs, hard=True):
            num = den = 0.0
            for L, pr in prs.items():
                x = torch.randn(len(pr), 128, cfg["d"], generator=g)
                h, aux = S.forward(pr, x, hard=hard)
                tgt = T(pr, x)
                num += ((h - tgt) ** 2).sum().item(); den += ((tgt - x) ** 2).sum().item()
            return num / den if math.isfinite(num) else float("inf")
        out["train"], out["test"] = ev(tr), ev(te)
        out["test_dense"] = ev(te, hard=False)
        if cfg["test_lens"]:
            out["by_len"] = {L: ev({L: pr}) for L, pr in te.items()}
        # which stream runs for which token (program task): purity of the token -> stream map, and pass dependence
        L0 = lens[0]; pr = te[L0] if L0 in te else tr[L0]
        x = torch.randn(len(pr), 32, cfg["d"], generator=g)
        _, aux = S.forward(pr, x, record=True)
        G = torch.stack(aux["gates"], 1).mean(2)                      # [M, n, K]
        top = G.argmax(-1)                                            # [M, n]
        ntok = int(pr.max()) + 1
        tab = torch.zeros(ntok, cfg["K"])
        for u in range(ntok):
            tab[u] = torch.bincount(top[pr == u], minlength=cfg["K"]).double()
        out["purity"] = (tab.max(1).values.sum() / tab.sum()).item()
        out["used"] = int((tab.sum(0) > 0.02 * tab.sum()).sum())
        out["ran"] = aux["ran"]
    if cfg.get("gain"):
        pr = te[lens[0]][:24]
        x = torch.randn(len(pr), 4, cfg["d"], generator=g)
        gs = gains(lambda q, v: S.forward(q, v)[0], pr, x)
        gt = gains(T, pr, x)
        out["gain_med"], out["gain_max"] = gs[len(gs) // 2], gs[-1]
        out["tgain_med"], out["tgain_max"] = gt[len(gt) // 2], gt[-1]
    return (out, S, T, tr, te) if ret_model else out


def table(rows, seeds, extra=("purity", "used", "ran")):
    """rows: list of (label, overrides). Prints a markdown table of medians [min, max] over seeds."""
    med = lambda v: sorted(v)[len(v) // 2]
    print("| variant | training error | held-out error | solved (<1e-2) | purity | streams used | streams run per pass |")
    print("|---|---|---|---|---|---|---|", flush=True)
    res = {}
    for label, over in rows:
        rs = [run(over, s) for s in seeds]
        res[label] = rs
        tr = [r["train"] for r in rs]; te = [r["test"] for r in rs]
        print(f"| {label} | {med(tr):.1e} | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/{len(te)} | "
              f"{med([r['purity'] for r in rs]):.2f} | {med([r['used'] for r in rs])} | {med([r['ran'] for r in rs]):.2f} |", flush=True)
    return res
