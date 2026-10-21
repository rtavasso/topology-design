"""Fleet trainer: many independent runs trained as one batched model on GPU (CUDA), Apple GPU (MPS) or CPU.

Design (experiments/fleet/DESIGN.md):
- Every run in a batch is independent: its own parameters, AdamW state, data split, labels and hyperparameters.
- Runs in one batch may differ in graph (parent vector), learning rate, weight decay, per-block lr/wd multipliers,
  block output multipliers (alphas), beta, out_scale, init scale, seeds and label overrides (rule + exceptions).
  They must share shapes: p, d, hidden, number of blocks L, training-set size, horizon and eval schedule.
- Initial weights are copied from `grokking.build_model` for each run, so a run is bit-identical at step 0 to the
  single-run trainer, and the AdamW update has torch.optim.AdamW semantics (tested against it).
- Measurement is fully batched on the device: one full-grid forward per eval plus one per ablation, for all runs at
  once; only small per-run scalars are copied to the host.
- No per-step host synchronisation: divergence and early stopping are tracked in device tensors and read at evals.

Tasks:
- 'modadd':      y = (a + b) mod p (as in Studies 1-4; identical splits via grokking.split).
- 'modadd_exc':  as modadd, but a fixed random fraction `exc_frac` of the TRAINING pairs get arbitrary labels
                 (exceptions, never equal to the rule label). Exceptions can only be memorized.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from . import grokking as gk


@dataclass
class FleetConfig:
    arm: str = "pk0000"
    task: str = "modadd"
    p: int = 59
    train_frac: float = 0.4
    exc_frac: float = 0.0  # fraction of training pairs relabelled as exceptions ('modadd_exc')
    d: int = 128
    hidden: int = 256
    beta: float = 1.0
    out_scale: float = 1.0
    init_scale: float = 1.0
    alphas: tuple | None = None  # per-block output multipliers; None -> the arm's (unit) multipliers
    lr: float = 2e-3
    weight_decay: float = 1.0
    block_lr_mult: tuple | None = None  # per-block lr multipliers (len L); None -> ones
    block_wd_mult: tuple | None = None  # per-block weight-decay multipliers (len L); None -> ones
    steps: int = 40000
    eval_every: int = 100
    dense_early: bool = True
    stop_after_grok: int = 0  # 0 = fixed horizon
    grok_thresh: float = 0.95
    branches: tuple | None = None  # e.g. ((1, 2, 3), (4, 5, 6)): block sets ablated jointly at each eval (1-based)
    model_seed: int = 0
    data_seed: int = 0
    tag: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def L(self) -> int:
        return len(gk.arm_spec(self.arm)[0])

    @property
    def run_id(self) -> str:
        parts = [self.task, self.arm, f"p{self.p}", f"tf{self.train_frac:g}"]
        if self.exc_frac:
            parts.append(f"exc{self.exc_frac:g}")
        parts += [f"lr{self.lr:g}", f"wd{self.weight_decay:g}"]
        if self.block_lr_mult:
            parts.append("blr" + "-".join(f"{x:g}" for x in self.block_lr_mult))
        if self.block_wd_mult:
            parts.append("bwd" + "-".join(f"{x:g}" for x in self.block_wd_mult))
        if self.alphas:
            parts.append("al" + "-".join(f"{x:g}" for x in self.alphas))
        if self.out_scale != 1.0:
            parts.append(f"os{self.out_scale:g}")
        if self.beta != 1.0:
            parts.append(f"b{self.beta:g}")
        if self.init_scale != 1.0:
            parts.append(f"is{self.init_scale:g}")
        parts += [f"m{self.model_seed}", f"d{self.data_seed}"]
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


STRUCT_FIELDS = ("task", "p", "train_frac", "d", "hidden", "steps", "eval_every", "dense_early", "stop_after_grok",
                 "grok_thresh", "branches")


def struct_key(cfg: FleetConfig) -> tuple:
    return (cfg.L,) + tuple(getattr(cfg, f) for f in STRUCT_FIELDS)


def is_eval_step(step: int, cfg: FleetConfig) -> bool:
    if cfg.dense_early and step < 1000:
        return step % (10 if step < 200 else 25) == 0
    return step % cfg.eval_every == 0


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def build_task(cfg: FleetConfig) -> dict:
    """Grid tokens, labels and index sets for one run (all as CPU tensors over the p*p grid)."""
    if cfg.task not in ("modadd", "modadd_exc"):
        raise ValueError(cfg.task)
    a, b, y = gk.full_grid(cfg.p)
    tr, te = gk.split(cfg.p, cfg.train_frac, cfg.data_seed)
    y = y.clone()
    exc = torch.zeros(cfg.p * cfg.p, dtype=torch.bool)
    if cfg.task == "modadd_exc" and cfg.exc_frac > 0:
        g = torch.Generator().manual_seed(30_000 + cfg.data_seed)
        n_exc = int(round(cfg.exc_frac * len(tr)))
        pick = tr[torch.randperm(len(tr), generator=g)[:n_exc]]
        shift = torch.randint(1, cfg.p, (n_exc,), generator=g)  # never 0 -> never the rule label
        y[pick] = (y[pick] + shift) % cfg.p
        exc[pick] = True
    train = torch.zeros(cfg.p * cfg.p, dtype=torch.bool)
    train[tr] = True
    return {"a": a, "b": b, "y": y, "tr": tr, "te": te, "train": train, "test": ~train, "exc": exc,
            "rule_train": train & ~exc, "key": (a + b) % cfg.p}


# ---------------------------------------------------------------------------
# Batched model + optimizer
# ---------------------------------------------------------------------------


class Fleet:
    """R independent RoutedMLPs (same shapes) stacked along a leading run dimension."""

    def __init__(self, cfgs: list[FleetConfig], device: str = "cpu"):
        keys = {struct_key(c) for c in cfgs}
        if len(keys) != 1:
            raise ValueError("all runs in a batch must share shapes, task family, horizon and eval schedule")
        self.cfgs, self.device, self.R = cfgs, torch.device(device), len(cfgs)
        c0 = cfgs[0]
        self.L, self.p = c0.L, c0.p
        dev = self.device
        singles = []
        for c in cfgs:
            rc = gk.RunConfig(arm=c.arm, p=c.p, d=c.d, hidden=c.hidden, beta=c.beta, out_scale=c.out_scale,
                              model_seed=c.model_seed, init_scale=c.init_scale)
            singles.append(gk.build_model(rc))
        self.n_params = gk.n_params(singles[0])
        dt = singles[0].unembed.weight.dtype
        self.dtype = dt
        self.parents = torch.tensor([gk.arm_spec(c.arm)[0] for c in cfgs], dtype=torch.long, device=dev)  # (R, L)
        self.shared_parents = [int(self.parents[0, i]) if bool((self.parents[:, i] == self.parents[0, i]).all())
                               else None for i in range(self.L)]
        self.alphas = torch.tensor([list(c.alphas) if c.alphas else list(gk.arm_spec(c.arm)[1]) for c in cfgs],
                                   dtype=dt, device=dev)  # (R, L)
        self.beta = torch.tensor([c.beta for c in cfgs], dtype=dt, device=dev)
        self.out_scale = torch.tensor([c.out_scale for c in cfgs], dtype=dt, device=dev)
        st = lambda get: torch.stack([get(m).detach().clone() for m in singles]).to(dev).requires_grad_(True)
        P = {"embed_a": st(lambda m: m.embed_a.weight), "embed_b": st(lambda m: m.embed_b.weight),
             "unembed": st(lambda m: m.unembed.weight)}
        for i in range(self.L):
            P[f"w1_{i}"] = st(lambda m, i=i: m.w1[i].weight)  # (R, hidden, d)
            P[f"w2_{i}"] = st(lambda m, i=i: m.w2[i].weight)  # (R, d, hidden)
        self.params = P
        # per-tensor, per-run lr and wd multipliers
        ones = [1.0] * self.L
        blr = torch.tensor([list(c.block_lr_mult or ones) for c in cfgs], dtype=dt, device=dev)
        bwd = torch.tensor([list(c.block_wd_mult or ones) for c in cfgs], dtype=dt, device=dev)
        lr = torch.tensor([c.lr for c in cfgs], dtype=dt, device=dev)
        wd = torch.tensor([c.weight_decay for c in cfgs], dtype=dt, device=dev)
        self.lr_t, self.wd_t = {}, {}
        for k in P:
            if k.startswith("w"):
                i = int(k.split("_")[1])
                self.lr_t[k], self.wd_t[k] = lr * blr[:, i], wd * bwd[:, i]
            else:
                self.lr_t[k], self.wd_t[k] = lr.clone(), wd.clone()

    def _select_parent(self, hs: list[torch.Tensor], i: int) -> torch.Tensor:
        k = self.shared_parents[i]
        if k is not None:
            return hs[k]
        stacked = torch.stack(hs)  # (i+1, R, n, d)
        return stacked[self.parents[:, i], torch.arange(self.R, device=self.device)]

    def block(self, i: int, x: torch.Tensor) -> torch.Tensor:
        P = self.params
        z = torch.relu(torch.bmm(x, P[f"w1_{i}"].transpose(1, 2)))
        return torch.bmm(z, P[f"w2_{i}"].transpose(1, 2))

    def forward(self, A: torch.Tensor, B: torch.Tensor, alphas: torch.Tensor | None = None, parts: bool = False):
        """A, B: (R, n) or (n,) token indices. Returns logits (R, n, p), and if parts=True the per-component
        direct contributions as a list of L+1 tensors (R, n, p) [embedding path, block 1..L] (they sum to logits)."""
        P = self.params
        al = self.alphas if alphas is None else alphas
        if A.dim() == 1:
            h0 = P["embed_a"][:, A] + P["embed_b"][:, B]
        else:
            r = torch.arange(self.R, device=self.device)[:, None]
            h0 = P["embed_a"][r, A] + P["embed_b"][r, B]
        hs, writes = [h0], []
        bview = self.beta.view(-1, 1, 1)
        for i in range(self.L):
            w = al[:, i, None, None] * self.block(i, self._select_parent(hs, i))
            writes.append(w)
            hs.append(bview * hs[-1] + w)
        U = P["unembed"].transpose(1, 2)
        osv = self.out_scale.view(-1, 1, 1)
        logits = osv * torch.bmm(hs[-1], U)
        if not parts:
            return logits
        Lb = self.L
        comps = [osv * bview ** Lb * torch.bmm(h0, U)]
        comps += [osv * bview ** (Lb - 1 - i) * torch.bmm(w, U) for i, w in enumerate(writes)]
        return logits, comps

    def single(self, j: int) -> gk.RoutedMLP:
        """Run j's current parameters as a CPU RoutedMLP (for cross-checks)."""
        c = self.cfgs[j]
        m = gk.build_model(gk.RunConfig(arm=c.arm, p=c.p, d=c.d, hidden=c.hidden, beta=c.beta,
                                        out_scale=c.out_scale, model_seed=c.model_seed))
        P = self.params
        with torch.no_grad():
            m.alphas.copy_(self.alphas[j].cpu())
            m.embed_a.weight.copy_(P["embed_a"][j].cpu())
            m.embed_b.weight.copy_(P["embed_b"][j].cpu())
            m.unembed.weight.copy_(P["unembed"][j].cpu())
            for i in range(self.L):
                m.w1[i].weight.copy_(P[f"w1_{i}"][j].cpu())
                m.w2[i].weight.copy_(P[f"w2_{i}"][j].cpu())
        return m


class FleetAdamW:
    """torch.optim.AdamW (betas 0.9/0.98, eps 1e-8) with per-run, per-tensor lr and weight decay.
    `active` (R,) bool freezes finished or diverged runs (no step, no decay)."""

    def __init__(self, fleet: Fleet, betas=(0.9, 0.98), eps: float = 1e-8):
        self.f, self.b1, self.b2, self.eps = fleet, betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in fleet.params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in fleet.params.items()}
        self.t = 0

    @torch.no_grad()
    def step(self, active: torch.Tensor):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2_sqrt = (1 - self.b2 ** self.t) ** 0.5
        act = active.to(self.f.dtype)
        for k, p in self.f.params.items():
            if p.grad is None:
                continue
            shape = (-1,) + (1,) * (p.dim() - 1)
            lr = (self.f.lr_t[k] * act).view(shape)
            wd = self.f.wd_t[k].view(shape)
            p.mul_(1 - lr * wd)
            self.m[k].lerp_(p.grad, 1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            denom = (self.v[k].sqrt() / bc2_sqrt).add_(self.eps)
            p.sub_((lr / bc1) * self.m[k] / denom)

    def zero_grad(self):
        for p in self.f.params.values():
            p.grad = None


# ---------------------------------------------------------------------------
# Batched measurement (on device)
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure(fleet: Fleet, grid: dict, chunk: int = 1024) -> dict:
    """Batched measurements for all runs, accumulated over chunks of the input grid (bounded memory).

    grid: tokens a, b and rule key (N,); per-run labels y (R, N) and masks (R, N) 'train', 'test', 'exc', 'rule_train'.
    Per run: loss/accuracy on each set; per-component (embedding path, blocks) leak-free supports
    gen = mean test-label centered logit, mem = mean train-label centered logit - gen, exception support;
    per-component logit RMS; test-only rule fraction; single-block and branch ablations (accuracy, gen, mem)."""
    R, L, p = fleet.R, fleet.L, fleet.p
    dev = fleet.device
    sets = ["train", "test", "rule_train", "exc"]
    P = fleet.params
    U = P["unembed"].transpose(1, 2)
    osv, bview = fleet.out_scale.view(-1, 1, 1), fleet.beta.view(-1, 1, 1)
    N = grid["a"].shape[0]
    dt = fleet.dtype
    z = lambda *shape: torch.zeros(*shape, device=dev, dtype=dt)
    cnt = {s: grid[s].to(dt).sum(1).clamp_min(1.0) for s in sets}  # (R,)
    acc = {f"{s}_{m}": z(R) for s in sets for m in ("loss", "acc")}
    lab_sum = {s: z(R, L + 1) for s in sets}
    energy = z(R, L + 1)
    rule_sum = z(R, p, p)
    rule_cnt = z(R, p)
    zc_energy = z(R)
    br = fleet.cfgs[0].branches
    abl_sets = [[i] for i in range(L)] + ([[j - 1 for j in s] for s in br] + [[j - 1 for s in br for j in s]] if br else [])
    abl_acc = {k: z(R, len(abl_sets)) for k in ("train_acc", "test_acc", "exc_acc")}
    abl_lab = {k: z(R, len(abl_sets)) for k in ("train", "test")}

    def h0_of(a, b):
        return P["embed_a"][:, a] + P["embed_b"][:, b]

    for lo in range(0, N, chunk):
        sl = slice(lo, min(N, lo + chunk))
        a, b, key = grid["a"][sl], grid["b"][sl], grid["key"][sl]
        y = grid["y"][:, sl]
        yi = y.unsqueeze(-1)
        m = {s: grid[s][:, sl].to(dt) for s in sets}
        lab_c = lambda Z: Z.gather(-1, yi).squeeze(-1) - Z.mean(-1)  # centered true-label logit (R, n)
        # forward with per-component accounting
        h0 = h0_of(a, b)
        hs = [h0]
        comp = osv * bview ** L * torch.bmm(h0, U)
        comp_labs = [lab_c(comp)]
        energy[:, 0] += (comp - comp.mean(-1, keepdim=True)).pow(2).sum((1, 2))
        for i in range(L):
            w = fleet.alphas[:, i, None, None] * fleet.block(i, fleet._select_parent(hs, i))
            hs.append(bview * hs[-1] + w)
            comp = osv * bview ** (L - 1 - i) * torch.bmm(w, U)
            comp_labs.append(lab_c(comp))
            energy[:, i + 1] += (comp - comp.mean(-1, keepdim=True)).pow(2).sum((1, 2))
        logits = osv * torch.bmm(hs[-1], U)
        del hs, comp
        labs = torch.stack(comp_labs, 1)  # (R, L+1, n)
        ce = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        cor = (logits.argmax(-1) == y).to(dt)
        for s in sets:
            acc[f"{s}_loss"] += (ce * m[s]).sum(1)
            acc[f"{s}_acc"] += (cor * m[s]).sum(1)
            lab_sum[s] += (labs * m[s].unsqueeze(1)).sum(-1)
        Zc = logits - logits.mean(-1, keepdim=True)
        zc_energy += Zc.pow(2).sum((1, 2))
        rule_sum.index_add_(1, key, Zc * m["test"].unsqueeze(-1))
        rule_cnt.index_add_(1, key, m["test"])
        del logits, Zc, labs
        for q, blk in enumerate(abl_sets):
            al = fleet.alphas.clone()
            al[:, blk] = 0.0
            hs2 = [h0]
            for i in range(L):
                w = al[:, i, None, None] * fleet.block(i, fleet._select_parent(hs2, i))
                hs2.append(bview * hs2[-1] + w)
            lg = osv * torch.bmm(hs2[-1], U)
            c2 = (lg.argmax(-1) == y).to(dt)
            l2 = lab_c(lg)
            abl_acc["train_acc"][:, q] += (c2 * m["train"]).sum(1)
            abl_acc["test_acc"][:, q] += (c2 * m["test"]).sum(1)
            abl_acc["exc_acc"][:, q] += (c2 * m["exc"]).sum(1)
            abl_lab["train"][:, q] += (l2 * m["train"]).sum(1)
            abl_lab["test"][:, q] += (l2 * m["test"]).sum(1)
            del hs2, lg
    out = {}
    for s in sets:
        out[f"{s}_loss"] = acc[f"{s}_loss"] / cnt[s]
        out[f"{s}_acc"] = acc[f"{s}_acc"] / cnt[s]
    sup = {s: lab_sum[s] / cnt[s].unsqueeze(1) for s in sets}
    out["gen"] = sup["test"]
    out["mem"] = sup["train"] - sup["test"]
    out["exc_support"] = sup["exc"]
    out["rule_train_support"] = sup["rule_train"]
    out["part_rms"] = (energy / (N * p)).sqrt()
    # test-only rule projection: each grid point gets its key's test-mean; energy = sum_key N_key * |mu_key|^2
    n_key = torch.bincount(grid["key"], minlength=p).to(rule_sum.dtype)  # grid points per key
    mu = rule_sum / rule_cnt.clamp_min(1.0).unsqueeze(-1)
    out["rule_frac_test"] = (n_key.view(1, -1) * mu.pow(2).sum(-1)).sum(1) / zc_energy.clamp_min(1e-12)
    nb = L
    gen_abl = abl_lab["test"] / cnt["test"].unsqueeze(1)
    mem_abl = abl_lab["train"] / cnt["train"].unsqueeze(1) - gen_abl
    for k, v in (("train_acc", abl_acc["train_acc"] / cnt["train"].unsqueeze(1)),
                 ("test_acc", abl_acc["test_acc"] / cnt["test"].unsqueeze(1)),
                 ("exc_acc", abl_acc["exc_acc"] / cnt["exc"].unsqueeze(1)), ("gen", gen_abl), ("mem", mem_abl)):
        out[f"abl_block_{k}"] = v[:, :nb]
        if br:
            out[f"abl_branch_{k}"] = v[:, nb:]  # each branch, then all branches together
    return out


def grad_norms(fleet: Fleet) -> torch.Tensor:
    """(R, L+2): per-block gradient norms, then embeddings, then readout."""
    P = fleet.params
    sq = lambda k: P[k].grad.pow(2).flatten(1).sum(1) if P[k].grad is not None else torch.zeros(fleet.R, device=fleet.device)
    cols = [(sq(f"w1_{i}") + sq(f"w2_{i}")).sqrt() for i in range(fleet.L)]
    cols += [(sq("embed_a") + sq("embed_b")).sqrt(), sq("unembed").sqrt()]
    return torch.stack(cols, 1)


def weight_norms(fleet: Fleet) -> torch.Tensor:
    P = fleet.params
    sq = lambda k: P[k].detach().pow(2).flatten(1).sum(1)
    cols = [(sq(f"w1_{i}") + sq(f"w2_{i}")).sqrt() for i in range(fleet.L)]
    cols += [(sq("embed_a") + sq("embed_b")).sqrt(), sq("unembed").sqrt()]
    return torch.stack(cols, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _record(meas: dict, gn: torch.Tensor, wn: torch.Tensor, j: int, step: int, L: int) -> dict:
    g = lambda k: meas[k][j].tolist() if meas[k].dim() > 1 else float(meas[k][j])
    rec = {"step": step}
    for k, v in meas.items():
        rec[k] = g(k)
    rec["grad_norms"] = gn[j].tolist()
    rec["weight_norms"] = wn[j].tolist()
    return rec


def summarize(traj: list[dict], cfg: FleetConfig, diverged_at: int | None) -> dict:
    first = lambda key, th: next((r["step"] for r in traj if r[key] >= th), None)
    t_fit, t_grok = first("train_acc", 0.99), first("test_acc", cfg.grok_thresh)
    post = [r["train_acc"] for r in traj if t_fit is not None and r["step"] >= t_fit]
    n_coll = sum(1 for i in range(1, len(post)) if post[i] < 0.9 <= post[i - 1])
    last = traj[-1] if traj else {}
    return {"t_fit": t_fit, "t_grok": t_grok,
            "grok_delay": (t_grok - t_fit) if (t_fit is not None and t_grok is not None) else None,
            "grokked": t_grok is not None, "final_train_acc": last.get("train_acc"),
            "final_test_acc": last.get("test_acc"), "final_exc_acc": last.get("exc_acc"),
            "post_fit_min_train_acc": min(post) if post else 0.0, "n_collapses": n_coll,
            "stable": diverged_at is None and t_fit is not None and bool(post) and min(post) >= 0.9,
            "diverged_at": diverged_at, "steps_run": last.get("step")}


def train_fleet(cfgs: list[FleetConfig], device: str = "cpu", measure_chunk: int = 1024, verbose: bool = False) -> list[dict]:
    t0 = time.time()
    fleet = Fleet(cfgs, device)
    dev, R, c0 = fleet.device, fleet.R, cfgs[0]
    tasks = [build_task(c) for c in cfgs]
    grid = {"a": tasks[0]["a"].to(dev), "b": tasks[0]["b"].to(dev), "key": tasks[0]["key"].to(dev),
            "y": torch.stack([t["y"] for t in tasks]).to(dev)}
    for s in ("train", "test", "exc", "rule_train"):
        grid[s] = torch.stack([t[s] for t in tasks]).to(dev)
    tr = torch.stack([t["tr"] for t in tasks]).to(dev)  # (R, n): same n for all runs (same train_frac)
    A, B = grid["a"][tr], grid["b"][tr]
    Y = grid["y"].gather(1, tr)
    opt = FleetAdamW(fleet)
    active = torch.ones(R, dtype=torch.bool, device=dev)
    div_step = torch.full((R,), -1, dtype=torch.long, device=dev)
    grok_step = [None] * R
    trajs: list[list[dict]] = [[] for _ in range(R)]
    n = A.shape[1]
    for step in range(c0.steps + 1):
        opt.zero_grad()
        logits = fleet.forward(A, B)
        per_run = F.cross_entropy(logits.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1)
        finite = torch.isfinite(per_run)
        div_step = torch.where((~finite) & (div_step < 0), torch.full_like(div_step, step), div_step)
        active = active & finite
        torch.where(active, per_run, torch.zeros_like(per_run)).sum().backward()
        if is_eval_step(step, c0):
            meas = measure(fleet, grid, measure_chunk)
            gn, wn = grad_norms(fleet), weight_norms(fleet)
            meas = {k: v.detach().cpu() for k, v in meas.items()}
            gn, wn, act, divs = gn.cpu(), wn.cpu(), active.cpu(), div_step.cpu()
            for j in range(R):
                if not bool(act[j]) and (divs[j] >= 0 or (grok_step[j] is not None and c0.stop_after_grok)):
                    continue
                rec = _record(meas, gn, wn, j, step, fleet.L)
                trajs[j].append(rec)
                if grok_step[j] is None and rec["test_acc"] >= c0.grok_thresh:
                    grok_step[j] = step
                if c0.stop_after_grok and grok_step[j] is not None and step - grok_step[j] >= c0.stop_after_grok:
                    active[j] = False
            if verbose:
                print(f"step {step}: active {int(active.sum())}/{R}", flush=True)
            if not bool(active.any()):
                break
        opt.step(active)
    secs = time.time() - t0
    divs = div_step.cpu().tolist()
    out = []
    for j, c in enumerate(cfgs):
        d = divs[j] if divs[j] >= 0 else None
        out.append({"config": asdict(c), "run_id": c.run_id, "n_params": fleet.n_params, "trajectory": trajs[j],
                    "summary": summarize(trajs[j], c, d),
                    "fleet": {"device": str(dev), "R": R, "seconds_batch": secs}})
    return out
