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
    # ---- Phase 5 (experiments/fastslow/DESIGN.md); all off by default ----
    branch_lr_mult: tuple | None = None  # per-branch lr multipliers (requires branches; multiplies block_lr_mult)
    branch_wd_mult: tuple | None = None  # per-branch weight-decay multipliers
    branch_alpha: tuple | None = None  # per-branch output multipliers (scale matching; multiplies alphas)
    unmask_branch: int | None = None  # 0-based branch whose params learn from CE(z0 + z_branch) only (5B)
    route: str | None = None  # direction-only routing of the preconditioned update: time|agree|rand|topk|rot (5C)
    route_branch: int = 0  # receiving branch (0-based)
    route_alpha: float = 0.998  # EMA rate of the preconditioned update (time routing, rotation null)
    route_tau: float = 0.75  # sign-agreement threshold for the agreement mask
    route_rot_angle: float | None = None  # rot null: fixed rotation angle in degrees (the time arm's median angle, as
    #   DESIGN 5C specifies); None -> each step's own angle between u and EMA(u)
    grokfast: float = 0.0  # lambda of uniform Grokfast-EMA (g + lambda*EMA(g)) on all params: 5C positive control
    grokfast_alpha: float = 0.98
    measure_path_eff: bool = False  # path efficiency ||sum u|| / sum ||u|| per branch (cumulative), at evals
    groups: int = 8  # training-set groups for agreement masks (resampled every step)
    consolidate: float = 0.0  # kappa of function-space consolidation (5D)
    consolidate_to: int = 0  # receiving branch
    consolidate_from: int = 1  # source branch
    consolidate_beta: float = 0.99  # EMA rate of the source branch's logits
    consolidate_null: bool = False  # targets from a fixed permutation of training examples
    measure_signal: bool = False  # 5A signal / masking measurements at evals
    arrive_start: int = 0  # 5E: training pairs present at step 0 (0 = all at once)
    arrive_n: int = 0  # pairs added per increment
    arrive_every: int = 0  # steps between increments
    probe_steps: tuple | None = None  # 5E: steps at which first-order probes run
    probe_pairs: int = 50
    probe_matched: int = 0  # 5E protocol-matched probes: AdamW steps on copies with the pair added (0 = off; design 50)
    # ---- "diagnose the walk" (experiments/walk/PREREGISTRATION.md) ----
    measure_horizon: bool = False  # E2: closed-form readout refit + drift extrapolation at evals
    horizon_extrap: tuple = (1000, 4000, 16000)  # E2: extrapolation horizons (steps) along the parameter drift
    readout_swap_at: int | None = None  # E2 intervention: at this step replace the readout by the refit readout
    drift_jump_at: int | None = None  # E2 intervention: at this step jump theta += H * drift (needs measure_horizon)
    drift_jump_h: int = 1000
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
        # every field enters a short hash, so configs differing in anything (horizon, width, branches, ...) never collide
        import hashlib
        import json as _json
        h = hashlib.sha1(_json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:8]
        parts.append(h)
        return "_".join(parts)


STRUCT_FIELDS = ("task", "p", "train_frac", "d", "hidden", "steps", "eval_every", "dense_early", "stop_after_grok",
                 "grok_thresh", "branches", "measure_signal", "arrive_start", "arrive_n", "arrive_every", "probe_steps",
                 "probe_pairs", "probe_matched", "groups", "measure_horizon", "horizon_extrap")
GROUP_ROUTES = ("agree", "rand", "topk")  # routing modes that need per-step group gradients


def struct_key(cfg: FleetConfig) -> tuple:
    # batches with group-gradient routing pay ~1 extra pass per step, so they are kept separate
    return (cfg.L, cfg.route in GROUP_ROUTES) + tuple(getattr(cfg, f) for f in STRUCT_FIELDS)


def branch_blocks(cfg: FleetConfig) -> list[list[int]]:
    """0-based block lists of the configured branches."""
    return [[j - 1 for j in b] for b in (cfg.branches or ())]


def validate_config(cfg: FleetConfig) -> None:
    nb = len(cfg.branches or ())
    for name in ("branch_lr_mult", "branch_wd_mult", "branch_alpha"):
        v = getattr(cfg, name)
        if v is not None and len(v) != nb:
            raise ValueError(f"{name} needs one value per branch ({nb})")
    for name in ("unmask_branch",):
        v = getattr(cfg, name)
        if v is not None and not 0 <= v < nb:
            raise ValueError(f"{name}={v} but there are {nb} branches")
    if cfg.route is not None:
        if cfg.route not in ("time",) + GROUP_ROUTES + ("rot",):
            raise ValueError(cfg.route)
        if not 0 <= cfg.route_branch < nb:
            raise ValueError("route_branch out of range")
    if cfg.consolidate and not (0 <= cfg.consolidate_to < nb and 0 <= cfg.consolidate_from < nb):
        raise ValueError("consolidate_to/from out of range")
    if cfg.route in GROUP_ROUTES and (cfg.unmask_branch is not None or cfg.consolidate):
        raise ValueError("unmask_branch / consolidate cannot be combined with group-gradient routing")
    if cfg.route_rot_angle is not None and cfg.route != "rot":
        raise ValueError("route_rot_angle applies to route='rot' only")
    if cfg.arrive_start and (cfg.arrive_n <= 0 or cfg.arrive_every <= 0):
        raise ValueError("arrivals need arrive_n and arrive_every")


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
        self._refresh_shared_parents()
        for c in cfgs:
            validate_config(c)

        def per_block(c, block_vals, branch_vals):
            v = list(block_vals) if block_vals else [1.0] * self.L
            for bi, blocks in enumerate(branch_blocks(c)):
                if branch_vals is not None:
                    for i in blocks:
                        v[i] *= branch_vals[bi]
            return v

        self.alphas = torch.tensor([per_block(c, c.alphas or gk.arm_spec(c.arm)[1], c.branch_alpha) for c in cfgs],
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
        blr = torch.tensor([per_block(c, c.block_lr_mult, c.branch_lr_mult) for c in cfgs], dtype=dt, device=dev)
        bwd = torch.tensor([per_block(c, c.block_wd_mult, c.branch_wd_mult) for c in cfgs], dtype=dt, device=dev)
        lr = torch.tensor([c.lr for c in cfgs], dtype=dt, device=dev)
        wd = torch.tensor([c.weight_decay for c in cfgs], dtype=dt, device=dev)
        self.lr_t, self.wd_t = {}, {}
        for k in P:
            if k.startswith("w"):
                i = int(k.split("_")[1])
                self.lr_t[k], self.wd_t[k] = lr * blr[:, i], wd * bwd[:, i]
            else:
                self.lr_t[k], self.wd_t[k] = lr.clone(), wd.clone()

    def _refresh_shared_parents(self):
        self.shared_parents = [int(self.parents[0, i]) if bool((self.parents[:, i] == self.parents[0, i]).all())
                               else None for i in range(self.L)]

    def subset(self, idx: torch.Tensor):
        """Keep only runs `idx` (a LongTensor on the device), in place. Used to compact a batch when runs finish."""
        self.cfgs = [self.cfgs[int(j)] for j in idx.tolist()]
        self.R = len(self.cfgs)
        for k in list(self.params):
            self.params[k] = self.params[k].detach()[idx].clone().requires_grad_(True)
            self.lr_t[k], self.wd_t[k] = self.lr_t[k][idx], self.wd_t[k][idx]
        self.parents, self.alphas = self.parents[idx], self.alphas[idx]
        self.beta, self.out_scale = self.beta[idx], self.out_scale[idx]
        self._refresh_shared_parents()

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

    def forward(self, A: torch.Tensor, B: torch.Tensor, alphas: torch.Tensor | None = None, parts: bool = False,
                stream: bool = False):
        """A, B: (R, n) or (n,) token indices. Returns logits (R, n, p), and if parts=True the per-component
        direct contributions as a list of L+1 tensors (R, n, p) [embedding path, block 1..L] (they sum to logits).
        stream=True returns (logits, final residual stream (R, n, d)) instead."""
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
        if stream:
            return logits, hs[-1]
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


ROUTE_CODES = {None: 0, "time": 1, "agree": 2, "rand": 3, "topk": 4, "rot": 5}


def _bshape(x: torch.Tensor) -> tuple:
    return (-1,) + (1,) * (x.dim() - 1)


class FleetAdamW:
    """torch.optim.AdamW (betas 0.9/0.98, eps 1e-8) with per-run, per-tensor lr and weight decay.

    `active` (R,) bool freezes finished or diverged runs exactly (parameters and moments unchanged).

    Phase 5C routing (per run, on its receiving branch only): the preconditioned update u = m_hat/(sqrt(v_hat)+eps),
    computed from moments of the RAW gradient, is replaced by a G-part rescaled to the norm of u over the branch
    (direction-only, one-sided, hard). Modes: time = EMA_alpha(u) (bias-corrected, including the current u); agree =
    u * agreement mask from group-gradient signs; rand = u * random mask of the same density; topk = the same number of
    largest-|u| coordinates; rot = u rotated toward a random orthogonal direction by the angle between u and EMA(u), or
    by the fixed `route_rot_angle`. If the G-part is empty (norm < 1e-12) the unrouted update is used. Decay is
    unchanged. Coordinates that are not routed are updated with exactly the plain AdamW arithmetic, so an unrouted run
    is bit-identical whatever else shares its batch. The angle between routed and unrouted update (degrees) is
    recorded per run and step (manipulation check).

    Also: uniform Grokfast-EMA (5C positive control), EMAs of the raw gradient (5A time splits) and of u, and
    cumulative path efficiency per branch. Random draws use the optimizer's own generator."""

    def __init__(self, fleet: Fleet, betas=(0.9, 0.98), eps: float = 1e-8):
        self.f, self.b1, self.b2, self.eps = fleet, betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in fleet.params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in fleet.params.items()}
        self.t = 0
        cf = fleet.cfgs
        dev, dt = fleet.device, fleet.dtype
        import hashlib
        seed = int(hashlib.sha1("|".join(c.run_id for c in cf).encode()).hexdigest()[:8], 16)
        self.gen = make_generator(dev, seed)
        self.mode = torch.tensor([ROUTE_CODES[c.route] for c in cf], device=dev)
        self.recv = torch.zeros(fleet.R, fleet.L, dtype=torch.bool, device=dev)
        for r, c in enumerate(cf):
            if c.route is not None:
                self.recv[r, branch_blocks(c)[c.route_branch]] = True
        self.route_alpha = torch.tensor([c.route_alpha for c in cf], dtype=dt, device=dev)
        self.tau = torch.tensor([c.route_tau for c in cf], dtype=dt, device=dev)
        self.rot_angle = torch.tensor([math.radians(c.route_rot_angle) if c.route_rot_angle is not None else math.nan
                                       for c in cf], dtype=dt, device=dev)
        self.routing = bool((self.mode > 0).any())
        self.track_u = (self.routing and bool(((self.mode == 1) | (self.mode == 5)).any())) or any(c.measure_signal for c in cf)
        self.track_g = any(c.measure_signal for c in cf)
        self.ema_u = {k: torch.zeros_like(v) for k, v in fleet.params.items()} if self.track_u else None
        self.ema_g = ({a: {k: torch.zeros_like(v) for k, v in fleet.params.items()} for a in (0.98, 0.998)}
                      if self.track_g else None)
        self.gf_lam = torch.tensor([c.grokfast for c in cf], dtype=dt, device=dev)
        self.gf_alpha = torch.tensor([c.grokfast_alpha for c in cf], dtype=dt, device=dev)
        self.grokfast = bool((self.gf_lam > 0).any())
        self.gf_h = {k: torch.zeros_like(v) for k, v in fleet.params.items()} if self.grokfast else None
        self.path_eff = any(c.measure_path_eff for c in cf)
        self.units = branch_blocks(cf[0]) or [[i] for i in range(fleet.L)]
        if self.path_eff:
            self.pe_sum = {k: torch.zeros_like(v) for k, v in fleet.params.items() if k.startswith("w")}
            self.pe_norm = torch.zeros(fleet.R, len(self.units), dtype=dt, device=dev)
        self.angles: list[torch.Tensor] = []
        self.sign_sum = None  # set by the trainer each step for group routes: {param: (R, ...) sum of group signs}
        self.n_groups = cf[0].groups

    def subset(self, idx: torch.Tensor):
        for k in list(self.m):
            self.m[k], self.v[k] = self.m[k][idx], self.v[k][idx]
            if self.ema_u is not None:
                self.ema_u[k] = self.ema_u[k][idx]
            if self.ema_g is not None:
                for a in self.ema_g:
                    self.ema_g[a][k] = self.ema_g[a][k][idx]
            if self.gf_h is not None:
                self.gf_h[k] = self.gf_h[k][idx]
        if self.path_eff:
            self.pe_sum = {k: v[idx] for k, v in self.pe_sum.items()}
            self.pe_norm = self.pe_norm[idx]
        self.mode, self.recv = self.mode[idx], self.recv[idx]
        self.route_alpha, self.tau, self.rot_angle = self.route_alpha[idx], self.tau[idx], self.rot_angle[idx]
        self.gf_lam, self.gf_alpha = self.gf_lam[idx], self.gf_alpha[idx]
        self.angles = [a[idx] for a in self.angles]

    def _moments(self, k: str, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Next moments, with exactly the arithmetic of torch AdamW (lerp; mul + addcmul)."""
        return torch.lerp(self.m[k], g, 1 - self.b1), torch.addcmul(self.v[k] * self.b2, g, g, value=1 - self.b2)

    def preconditioned(self, grads: dict | None = None) -> dict:
        """The update u this optimizer would apply at the next step given gradients (default p.grad), without changing
        any state. Used by 5A measurement (shadow update)."""
        t = self.t + 1
        bc1, bc2_sqrt = 1 - self.b1 ** t, (1 - self.b2 ** t) ** 0.5
        out = {}
        for k, p in self.f.params.items():
            g = p.grad if grads is None else grads[k]
            m, v = self._moments(k, g)
            out[k] = (m / bc1) / ((v.sqrt() / bc2_sqrt) + self.eps)
        return out

    def _grokfast(self, active):
        for k, p in self.f.params.items():
            if p.grad is None:
                continue
            sh = _bshape(p)
            a, lam = self.gf_alpha.view(sh), self.gf_lam.view(sh)
            on = (active & (self.gf_lam > 0)).view(sh)
            self.gf_h[k] = torch.where(on, self.gf_h[k] * a + (1 - a) * p.grad, self.gf_h[k])
            p.grad = torch.where(on, p.grad + lam * self.gf_h[k], p.grad)

    @torch.no_grad()
    def step(self, active: torch.Tensor):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2_sqrt = (1 - self.b2 ** self.t) ** 0.5
        if self.track_g:  # EMAs of the raw (CE + any auxiliary) gradient, before Grokfast
            for a, store in self.ema_g.items():
                for k, p in self.f.params.items():
                    if p.grad is not None:
                        act = active.view(_bshape(p))
                        store[k] = torch.where(act, store[k] * a + (1 - a) * torch.nan_to_num(p.grad), store[k])
        if self.grokfast:
            self._grokfast(active)
        if not self.routing and not self.track_u and not self.path_eff:
            return self._plain_step(active, bc1, bc2_sqrt)
        # general path: moments and u for all tensors, EMAs, routing; non-routed coordinates use the plain arithmetic
        U, M, V, D = {}, {}, {}, {}
        for k, p in self.f.params.items():
            if p.grad is None:
                continue
            act = active.view(_bshape(p))
            g = torch.where(act, p.grad, torch.zeros_like(p.grad))
            M[k], V[k] = self._moments(k, g)
            D[k] = (V[k].sqrt() / bc2_sqrt) + self.eps
            U[k] = (M[k] / bc1) / D[k]
        if self.track_u:
            for k in U:
                sh = _bshape(U[k])
                a = self.route_alpha.view(sh)
                self.ema_u[k] = torch.where(active.view(sh), self.ema_u[k] * a + (1 - a) * U[k], self.ema_u[k])
        used = {}
        if self.routing:
            Ur, used = self._route(U, active)
        for k, p in self.f.params.items():
            if k not in U:
                continue
            sh = _bshape(p)
            act = active.view(sh)
            lr, wd = self.f.lr_t[k].view(sh), self.f.wd_t[k].view(sh)
            term = (lr / bc1) * M[k] / D[k]
            if k in used:
                term = torch.where(used[k].view(sh), lr * Ur[k], term)
                U[k] = torch.where(used[k].view(sh), Ur[k], U[k])
            p_new = p * (1 - lr * wd) - term
            p.copy_(torch.where(act, p_new, p))
            self.m[k] = torch.where(act, M[k], self.m[k])
            self.v[k] = torch.where(act, V[k], self.v[k])
        if self.path_eff:
            for q, blocks in enumerate(self.units):
                names = [f"w1_{i}" for i in blocks] + [f"w2_{i}" for i in blocks]
                sq = sum(U[k].flatten(1).pow(2).sum(1) for k in names)
                self.pe_norm[:, q] += torch.where(active, sq.sqrt(), torch.zeros_like(sq))
                for k in names:
                    self.pe_sum[k] = torch.where(active.view(_bshape(U[k])), self.pe_sum[k] + U[k], self.pe_sum[k])

    def path_efficiency(self) -> torch.Tensor:
        """(R, units): ||sum_t u_t|| / sum_t ||u_t|| per branch (or block), cumulative since step 1 (u as applied)."""
        out = []
        for q, blocks in enumerate(self.units):
            names = [f"w1_{i}" for i in blocks] + [f"w2_{i}" for i in blocks]
            out.append(sum(self.pe_sum[k].flatten(1).pow(2).sum(1) for k in names).sqrt() / self.pe_norm[:, q].clamp_min(1e-30))
        return torch.stack(out, 1)

    def _plain_step(self, active, bc1, bc2_sqrt):
        all_active = bool(active.all()) if active.numel() < 4096 else False
        for k, p in self.f.params.items():
            if p.grad is None:
                continue
            shape = _bshape(p)
            lr = self.f.lr_t[k].view(shape)
            wd = self.f.wd_t[k].view(shape)
            g = p.grad
            if all_active:
                p.mul_(1 - lr * wd)
                self.m[k].lerp_(g, 1 - self.b1)
                self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                denom = (self.v[k].sqrt() / bc2_sqrt).add_(self.eps)
                p.sub_((lr / bc1) * self.m[k] / denom)
            else:
                act = active.view(shape)
                g = torch.where(act, g, torch.zeros_like(g))
                m_new, v_new = self._moments(k, g)
                denom = (v_new.sqrt() / bc2_sqrt).add_(self.eps)
                p_new = p * (1 - lr * wd) - (lr / bc1) * m_new / denom
                p.copy_(torch.where(act, p_new, p))
                self.m[k] = torch.where(act, m_new, self.m[k])
                self.v[k] = torch.where(act, v_new, self.v[k])

    def _route(self, U: dict, active: torch.Tensor) -> tuple[dict, dict]:
        """Returns (routed updates for the receiving tensors, per-tensor (R,) mask of runs whose update was routed)."""
        f, R, L = self.f, self.f.R, self.f.L
        dt, dev = f.dtype, f.device
        names = [(i, f"w1_{i}") for i in range(L)] + [(i, f"w2_{i}") for i in range(L)]
        names = [(i, k) for i, k in names if k in U]
        rmask = {k: (self.recv[:, i] & (self.mode > 0)) for i, k in names}  # (R,) receives on this tensor
        flat = lambda x: x.flatten(1)
        dot = lambda a, b: sum((flat(a[k]) * flat(b[k])).sum(1) * rmask[k] for _, k in names)
        u_sq = dot(U, U)
        G = {k: torch.zeros_like(U[k]) for _, k in names}
        md = self.mode
        if bool((md == 1).any()) or bool((md == 5).any()):
            corr = (1 - self.route_alpha ** self.t)
            E = {k: self.ema_u[k] / corr.view(_bshape(U[k])) for _, k in names}
        if bool((md == 1).any()):
            for _, k in names:
                G[k] = torch.where((md == 1).view(_bshape(U[k])), E[k], G[k])
        if bool(((md == 2) | (md == 3) | (md == 4)).any()):
            if self.sign_sum is None:
                raise RuntimeError("group-gradient routing needs sign_sum from the trainer")
            amask = {k: (self.sign_sum[k].abs() / self.n_groups) >= self.tau.view(_bshape(U[k])) for _, k in names}
            count = sum(flat(amask[k]).sum(1).to(dt) * rmask[k] for _, k in names)
            total = sum(torch.full((R,), float(flat(U[k]).shape[1]), dtype=dt, device=dev) * rmask[k] for _, k in names)
            density = count / total.clamp_min(1.0)
            if bool((md == 2).any()):
                for _, k in names:
                    G[k] = torch.where((md == 2).view(_bshape(U[k])), U[k] * amask[k], G[k])
            if bool((md == 3).any()):
                for _, k in names:
                    sh = _bshape(U[k])
                    rnd = rand_like_gen(U[k], self.gen) < density.view(sh)
                    G[k] = torch.where((md == 3).view(sh), U[k] * rnd, G[k])
            if bool((md == 4).any()):
                allabs = torch.cat([flat(U[k]).abs() * rmask[k].view(-1, 1) - (~rmask[k]).view(-1, 1).to(dt)
                                    for _, k in names], 1)  # non-receiving coordinates -> -1
                srt = allabs.sort(1, descending=True).values
                kk = count.long().clamp(min=1, max=srt.shape[1]) - 1
                thr = srt.gather(1, kk.view(-1, 1)).squeeze(1)
                thr = torch.where(count > 0, thr, torch.full_like(thr, float("inf")))
                for _, k in names:
                    sh = _bshape(U[k])
                    G[k] = torch.where((md == 4).view(sh), U[k] * (U[k].abs() >= thr.view(sh)), G[k])
        if bool((md == 5).any()):
            ue = dot(U, E)
            e_sq = dot(E, E)
            cos_t = (ue / (u_sq.sqrt() * e_sq.sqrt()).clamp_min(1e-30)).clamp(-1.0, 1.0)
            theta = torch.where(torch.isnan(self.rot_angle), torch.arccos(cos_t), self.rot_angle)
            Rn = {k: rand_like_gen(U[k], self.gen, normal=True) for _, k in names}
            ru = dot(Rn, U)
            Rp = {k: Rn[k] - (ru / u_sq.clamp_min(1e-30)).view(_bshape(U[k])) * U[k] for _, k in names}
            rp_norm = dot(Rp, Rp).sqrt().clamp_min(1e-30)
            un = u_sq.sqrt().clamp_min(1e-30)
            for _, k in names:
                sh = _bshape(U[k])
                rot = torch.cos(theta).view(sh) * U[k] / un.view(sh) + torch.sin(theta).view(sh) * Rp[k] / rp_norm.view(sh)
                G[k] = torch.where((md == 5).view(sh), rot, G[k])
        g_sq = dot(G, G)
        ok = (g_sq > 1e-24) & (self.mode > 0)
        scale = torch.where(ok, (u_sq / g_sq.clamp_min(1e-30)).sqrt(), torch.zeros_like(u_sq))
        out, used = {}, {}
        for _, k in names:
            sh = _bshape(U[k])
            used[k] = rmask[k] & ok
            out[k] = torch.where(used[k].view(sh), G[k] * scale.view(sh), U[k])
        # manipulation check: angle between routed and unrouted update on the receiving branch (NaN if not routed)
        full = {k: out.get(k, U[k]) for _, k in names}
        cosang = dot(full, U) / (dot(full, full).sqrt() * u_sq.sqrt()).clamp_min(1e-30)
        ang = torch.rad2deg(torch.arccos(cosang.clamp(-1.0, 1.0)))
        self.angles.append(torch.where((self.mode > 0) & active, ang, torch.full_like(ang, float("nan"))))
        return out, used

    def take_angle(self) -> dict:
        """Median and mean routing angle (degrees) over the steps since the last call, per run (NaN if not routed)."""
        if not self.angles:
            nan = torch.full((self.f.R,), float("nan"), dtype=self.f.dtype, device=self.f.device)
            return {"route_angle": nan, "route_angle_mean": nan.clone()}
        a = torch.stack(self.angles, 1)
        self.angles = []
        return {"route_angle": a.nanmedian(1).values, "route_angle_mean": a.nanmean(1)}

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
    per-component logit RMS; test-only rule fraction; single-block and branch ablations (accuracy, train/test loss,
    gen, mem). If grid has a 'new' mask (5E: the latest increment), also mem_new = per-component mean new-pair
    centered logit - gen, and new_acc / new_loss."""
    R, L, p = fleet.R, fleet.L, fleet.p
    dev = fleet.device
    sets = ["train", "test", "rule_train", "exc"] + (["new"] if "new" in grid else [])
    P = fleet.params
    U = P["unembed"].transpose(1, 2)
    osv, bview = fleet.out_scale.view(-1, 1, 1), fleet.beta.view(-1, 1, 1)
    N = grid["a"].shape[0]
    dt = fleet.dtype
    z = lambda *shape: torch.zeros(*shape, device=dev, dtype=dt)
    cnt = {s: grid[s].to(dt).sum(1) for s in sets}  # (R,); an empty set (no exceptions) gives NaN -> None in records
    acc = {f"{s}_{m}": z(R) for s in sets for m in ("loss", "acc")}
    lab_sum = {s: z(R, L + 1) for s in sets}
    energy = z(R, L + 1)
    rule_sum = z(R, p, p)
    rule_cnt = z(R, p)
    zc_energy = z(R)
    br = fleet.cfgs[0].branches
    br_blocks = [[j - 1 for j in s_] for s_ in br] if br else []
    br_energy = z(R, len(br_blocks))
    abl_sets = [[i] for i in range(L)] + ([[j - 1 for j in s] for s in br] + [[j - 1 for s in br for j in s]] if br else [])
    abl_acc = {k: z(R, len(abl_sets)) for k in ("train_acc", "test_acc", "exc_acc", "train_loss", "test_loss")}
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
        br_sum = [None] * len(br_blocks)
        energy[:, 0] += (comp - comp.mean(-1, keepdim=True)).pow(2).sum((1, 2))
        for i in range(L):
            w = fleet.alphas[:, i, None, None] * fleet.block(i, fleet._select_parent(hs, i))
            hs.append(bview * hs[-1] + w)
            comp = osv * bview ** (L - 1 - i) * torch.bmm(w, U)
            comp_labs.append(lab_c(comp))
            energy[:, i + 1] += (comp - comp.mean(-1, keepdim=True)).pow(2).sum((1, 2))
            for q, blocks in enumerate(br_blocks):
                if i in blocks:
                    br_sum[q] = comp if br_sum[q] is None else br_sum[q] + comp
        for q in range(len(br_blocks)):
            br_energy[:, q] += (br_sum[q] - br_sum[q].mean(-1, keepdim=True)).pow(2).sum((1, 2))
        del br_sum
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
            ce2 = F.cross_entropy(lg.transpose(1, 2), y, reduction="none")
            l2 = lab_c(lg)
            abl_acc["train_loss"][:, q] += (ce2 * m["train"]).sum(1)
            abl_acc["test_loss"][:, q] += (ce2 * m["test"]).sum(1)
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
    if "new" in sup:
        out["mem_new"] = sup["new"] - sup["test"]
    out["part_rms"] = (energy / (N * p)).sqrt()
    if br_blocks:
        out["branch_rms"] = (br_energy / (N * p)).sqrt()  # RMS of each branch's summed centered logits (scale matching)
    # test-only rule projection: each grid point gets its key's test-mean; energy = sum_key N_key * |mu_key|^2
    n_key = grid["n_key"].to(rule_sum.dtype)  # grid points per key (precomputed on the host)
    mu = rule_sum / rule_cnt.clamp_min(1.0).unsqueeze(-1)
    out["rule_frac_test"] = (n_key.view(1, -1) * mu.pow(2).sum(-1)).sum(1) / zc_energy.clamp_min(1e-12)
    nb = L
    gen_abl = abl_lab["test"] / cnt["test"].unsqueeze(1)
    mem_abl = abl_lab["train"] / cnt["train"].unsqueeze(1) - gen_abl
    for k, v in (("train_acc", abl_acc["train_acc"] / cnt["train"].unsqueeze(1)),
                 ("test_acc", abl_acc["test_acc"] / cnt["test"].unsqueeze(1)),
                 ("exc_acc", abl_acc["exc_acc"] / cnt["exc"].unsqueeze(1)), ("gen", gen_abl), ("mem", mem_abl),
                 ("train_loss", abl_acc["train_loss"] / cnt["train"].unsqueeze(1)),
                 ("test_loss", abl_acc["test_loss"] / cnt["test"].unsqueeze(1))):
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


def _clean(x):
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return None if (isinstance(x, float) and math.isnan(x)) else x


def _records(meas: dict, gn: torch.Tensor, wn: torch.Tensor, step: int) -> list[dict]:
    """One bulk host transfer per quantity; NaN (e.g. exception metrics without exceptions) becomes None."""
    cols = {k: _clean(v.tolist()) for k, v in meas.items()}
    cols["grad_norms"], cols["weight_norms"] = gn.tolist(), wn.tolist()
    R = len(cols["grad_norms"])
    return [{"step": step, **{k: v[j] for k, v in cols.items()}} for j in range(R)]


def summarize(traj: list[dict], cfg: FleetConfig, diverged_at: int | None) -> dict:
    first = lambda key, th: next((r["step"] for r in traj if r[key] >= th), None)
    t_fit, t_grok = first("train_acc", 0.99), first("test_acc", cfg.grok_thresh)
    post = [r["train_acc"] for r in traj if t_fit is not None and r["step"] >= t_fit]
    n_coll = sum(1 for i in range(1, len(post)) if post[i] < 0.9 <= post[i - 1])
    last = traj[-1] if traj else {}
    post_min = 0.0 if (diverged_at is not None or t_fit is None or not post) else min(post)  # as gk.stability_summary
    return {"t_fit": t_fit, "t_grok": t_grok,
            "grok_delay": (t_grok - t_fit) if (t_fit is not None and t_grok is not None) else None,
            "grokked": t_grok is not None, "final_train_acc": last.get("train_acc"),
            "final_test_acc": last.get("test_acc"), "final_exc_acc": last.get("exc_acc"),
            "final_rule_frac_test": last.get("rule_frac_test"),
            "post_fit_min_train_acc": post_min, "n_collapses": n_coll,
            "stable": diverged_at is None and t_fit is not None and post_min >= 0.9,
            "diverged_at": diverged_at, "steps_run": last.get("step")}


def _ce(logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Per-example CE (R, n), in the flat layout of the baseline trainer (bit-identical to it)."""
    R, n, p = logits.shape
    return F.cross_entropy(logits.reshape(R * n, p), Y.reshape(-1), reduction="none").view(R, n)


def _run_loss(ce: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Per-run mean CE over the present examples. w=None: all present (exactly the baseline's .mean(1))."""
    if w is None:
        return ce.mean(1)
    return (ce * w).sum(1) / w.sum(1).clamp_min(1e-12)


def make_generator(device, seed: int) -> torch.Generator:
    """A generator on `device` if the backend supports it, else on the CPU (draws are then moved)."""
    try:
        g = torch.Generator(device=device)
    except RuntimeError:
        g = torch.Generator()
    g.manual_seed(seed)
    return g


def rand_like_gen(x: torch.Tensor, gen: torch.Generator, normal: bool = False) -> torch.Tensor:
    f = torch.randn if normal else torch.rand
    dt = x.dtype if x.is_floating_point() else torch.float32
    return f(x.shape, generator=gen, device=gen.device, dtype=dt).to(x.device)


def seen_groups(w: torch.Tensor, m: int, gen: torch.Generator) -> list[torch.Tensor]:
    """m disjoint groups of positions (R, ~n/m each), drawn at random from the present examples (w > 0): present
    positions are shuffled to the front and dealt round-robin, so every group gets an equal share (+-1) of them."""
    R, n = w.shape
    key = rand_like_gen(w, gen) + (w <= 0).to(w.dtype if w.is_floating_point() else torch.float32) * 2.0
    order = torch.argsort(key, 1)
    return [order[:, j::m] for j in range(m)]


def derangement(n: int, gen: torch.Generator) -> torch.Tensor:
    """A uniformly random permutation with no fixed points (rejection sampling; ~e tries)."""
    while True:
        perm = torch.randperm(n, generator=gen)
        if not bool((perm == torch.arange(n)).any()):
            return perm


def _arrivals(cfgs: list[FleetConfig], n: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per run: arrival step of each training position (R, n) and its increment index (0 = initial set)."""
    c0 = cfgs[0]
    if not c0.arrive_start:
        z = torch.zeros(len(cfgs), n, dtype=torch.long, device=device)
        return z, z.clone(), 0
    K = math.ceil((n - c0.arrive_start) / c0.arrive_n)
    arr, inc = [], []
    for c in cfgs:
        g = torch.Generator().manual_seed(40_000 + c.data_seed)
        order = torch.randperm(n, generator=g)
        rank = torch.empty(n, dtype=torch.long)
        rank[order] = torch.arange(n)
        k = torch.where(rank < c0.arrive_start, torch.zeros_like(rank), (rank - c0.arrive_start) // c0.arrive_n + 1)
        arr.append(k * c0.arrive_every)
        inc.append(k)
    return torch.stack(arr).to(device), torch.stack(inc).to(device), K


def _loss_and_grads(fleet: Fleet, opt: FleetAdamW, A, B, Y, w, active, st: dict, update_state: bool = True,
                    wc: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + backward for one step. Fills p.grad. Returns per-run loss (R,), per-example CE (R, n) and correctness
    (R, n) (both detached). Handles group gradients (agreement routing), the unmasked-branch arm and consolidation.

    w: (R, n) presence weights, or None when every position is present (then the loss is exactly the baseline's).
    wc: weights for the consolidation term (default w). update_state=False leaves the consolidation EMA unchanged
    (used when a step's gradient is recomputed after compaction)."""
    R, n = A.shape
    c0 = fleet.cfgs[0]
    P = fleet.params
    keys = list(P)
    if c0.route in GROUP_ROUTES:  # whole batch is group-routed (struct key); no unmasked arm / consolidation
        wf = torch.ones(R, n, device=A.device, dtype=fleet.dtype) if w is None else w
        wsum = wf.sum(1).clamp_min(1e-12)
        ce_all = torch.zeros(R, n, device=A.device, dtype=fleet.dtype)
        cor_all = torch.zeros(R, n, device=A.device, dtype=torch.bool)
        total = torch.zeros(R, device=A.device, dtype=fleet.dtype)
        grads = {k: torch.zeros_like(P[k]) for k in keys}
        signs = {k: torch.zeros_like(P[k]) for k in keys if k.startswith("w")}
        for idx in seen_groups(wf, c0.groups, opt.gen):
            Ak, Bk, Yk, wk = A.gather(1, idx), B.gather(1, idx), Y.gather(1, idx), wf.gather(1, idx)
            lg = fleet.forward(Ak, Bk)
            ce = _ce(lg, Yk)
            omega = wk.sum(1) / wsum
            loss_k = (ce * wk).sum(1) / wk.sum(1).clamp_min(1e-12)
            total = total + omega * loss_k.detach()
            ok = active & torch.isfinite(loss_k)
            gs = torch.autograd.grad(torch.where(ok, omega * loss_k, torch.zeros_like(loss_k)).sum(), [P[k] for k in keys])
            for k, gk_ in zip(keys, gs):
                grads[k] += gk_
                if k in signs:
                    signs[k] += torch.sign(gk_)
            ce_all.scatter_(1, idx, ce.detach())
            cor_all.scatter_(1, idx, lg.argmax(-1) == Yk)
        for k in keys:
            P[k].grad = grads[k]
        opt.sign_sum = signs
        return total, ce_all, cor_all
    need_parts = any(c.unmask_branch is not None or c.consolidate for c in fleet.cfgs)
    if need_parts:
        logits, comps = fleet.forward(A, B, parts=True)
    else:
        logits = fleet.forward(A, B)
    ce = _ce(logits, Y)
    per_run = _run_loss(ce, w)
    ok = active & torch.isfinite(per_run)
    torch.where(ok, per_run, torch.zeros_like(per_run)).sum().backward(retain_graph=need_parts)
    if need_parts:
        brs = [branch_blocks(c) for c in fleet.cfgs]
        # unmasked arm: the branch's own params learn from CE(z0 + z_branch) only
        for b in sorted({c.unmask_branch for c in fleet.cfgs if c.unmask_branch is not None}):
            flag = torch.tensor([c.unmask_branch == b for c in fleet.cfgs], device=A.device)
            blocks = brs[[i for i, c in enumerate(fleet.cfgs) if c.unmask_branch == b][0]][b]
            part = comps[0] + sum(comps[i + 1] for i in blocks)
            lp = _run_loss(_ce(part, Y), w)
            names = [f"w1_{i}" for i in blocks] + [f"w2_{i}" for i in blocks]
            gp = torch.autograd.grad(torch.where(flag & ok, lp, torch.zeros_like(lp)).sum(), [P[k] for k in names],
                                     retain_graph=True)
            for k, gk_ in zip(names, gp):
                P[k].grad = torch.where(flag.view((-1,) + (1,) * (gk_.dim() - 1)), gk_, P[k].grad)
        # consolidation: kappa * ||z_to - sg(EMA of z_from)||^2 (centered direct logits, summed over classes),
        # averaged over present training examples
        kappa = st.get("kappa")
        if kappa is not None and bool((kappa > 0).any()):
            wcon = wc if wc is not None else (torch.ones(R, n, device=A.device, dtype=fleet.dtype) if w is None else w)
            wcsum = wcon.sum(1).clamp_min(1e-12)
            pairs = sorted({(c.consolidate_to, c.consolidate_from) for c in fleet.cfgs if c.consolidate})
            for to_b, from_b in pairs:
                sel = torch.tensor([bool(c.consolidate) and (c.consolidate_to, c.consolidate_from) == (to_b, from_b)
                                    for c in fleet.cfgs], device=A.device)
                r0 = int(sel.nonzero()[0])
                to_blocks, from_blocks = brs[r0][to_b], brs[r0][from_b]
                zt = sum(comps[i + 1] for i in to_blocks)
                zt = zt - zt.mean(-1, keepdim=True)
                zs = sum(comps[i + 1] for i in from_blocks).detach()
                zs = zs - zs.mean(-1, keepdim=True)
                if st["cons_ema"] is None:
                    st["cons_ema"] = torch.zeros_like(zs)
                first = (sel & ~st["cons_init"]).view(-1, 1, 1)  # EMA starts at this run's own source logits
                ema = torch.where(first, zs, st["cons_ema"])
                tgt = ema.gather(1, st["cons_perm"].unsqueeze(-1).expand(-1, -1, zs.shape[-1]))
                lc = (((zt - tgt) ** 2).sum(-1) * wcon).sum(1) / wcsum
                names = [f"w1_{i}" for i in to_blocks] + [f"w2_{i}" for i in to_blocks]
                gc = torch.autograd.grad(torch.where(sel & ok, kappa * lc, torch.zeros_like(lc)).sum(),
                                         [P[k] for k in names], retain_graph=True)
                for k, gk_ in zip(names, gc):
                    P[k].grad = P[k].grad + gk_
                if update_state:
                    beta = st["cons_beta"].view(-1, 1, 1)
                    upd = (sel & active).view(-1, 1, 1)
                    st["cons_ema"] = torch.where(upd, beta * ema + (1 - beta) * zs, ema)
                    st["cons_init"] = st["cons_init"] | sel
        del comps
    return per_run.detach(), ce.detach(), (logits.argmax(-1) == Y).detach()


def train_fleet(cfgs: list[FleetConfig], device: str = "cpu", measure_chunk: int = 1024, verbose: bool = False,
                compact_below: float = 0.5) -> list[dict]:
    """Train all runs together. Finished (stop_after_grok) and diverged runs are frozen; when the active fraction
    falls below `compact_below`, the batch is compacted to the active runs so their compute is not wasted."""
    from . import fleet_horizon as fh
    from . import fleet_signal as fs
    t0 = time.time()
    fleet = Fleet(cfgs, device)
    dev, c0 = fleet.device, cfgs[0]
    R0 = fleet.R
    n_params = fleet.n_params
    tasks = [build_task(c) for c in cfgs]
    grid = {"a": tasks[0]["a"].to(dev), "b": tasks[0]["b"].to(dev), "key": tasks[0]["key"].to(dev),
            "n_key": torch.bincount(tasks[0]["key"], minlength=c0.p).to(dev),
            "y": torch.stack([t["y"] for t in tasks]).to(dev)}
    for s in ("train", "test", "exc", "rule_train"):
        grid[s] = torch.stack([t[s] for t in tasks]).to(dev)
    tr = torch.stack([t["tr"] for t in tasks]).to(dev)  # (R, n): same n for all runs (same train_frac)
    n = tr.shape[1]
    opt = FleetAdamW(fleet)
    arrive, inc, K = _arrivals(cfgs, n, dev)
    st = {"tr": tr, "arrive": arrive, "inc": inc,
          "lat": torch.full((R0, K + 1), -1, dtype=torch.long, device=dev),
          "active": torch.ones(R0, dtype=torch.bool, device=dev),
          "div_step": torch.full((R0,), -1, dtype=torch.long, device=dev)}
    if any(c.consolidate for c in cfgs):
        st["kappa"] = torch.tensor([c.consolidate for c in cfgs], dtype=fleet.dtype, device=dev)
        st["cons_beta"] = torch.tensor([c.consolidate_beta for c in cfgs], dtype=fleet.dtype, device=dev)
        perms = []
        for c in cfgs:
            if c.consolidate_null:  # each example's target is a *different* example's
                perms.append(derangement(n, torch.Generator().manual_seed(60_000 + c.data_seed)))
            else:
                perms.append(torch.arange(n))
        st["cons_perm"] = torch.stack(perms).to(dev)
        st["cons_ema"] = None
        st["cons_init"] = torch.zeros(R0, dtype=torch.bool, device=dev)
    if c0.probe_steps:
        pp = []
        for c, t in zip(cfgs, tasks):
            gp = torch.Generator().manual_seed(50_000 + c.data_seed)
            pp.append(t["te"][torch.randperm(len(t["te"]), generator=gp)[:c0.probe_pairs]])
        st["probe_pairs"] = torch.stack(pp).to(dev)
    if K > 0:  # prequential (predict-then-learn) code length: each pair's CE at the step it arrives, before training
        st["preq_nats"] = torch.zeros(R0, K + 1, dtype=fleet.dtype, device=dev)
        st["preq_cor"] = torch.zeros(R0, K + 1, dtype=fleet.dtype, device=dev)
        st["preq_n"] = torch.zeros(R0, K + 1, dtype=fleet.dtype, device=dev)
    hz = c0.measure_horizon
    swap = any(c.readout_swap_at is not None for c in cfgs)
    if hz or swap:
        st["hz_val"] = fh.holdout_mask(cfgs, grid["a"].shape[0], dev)
    if swap:
        st["swap_at"] = torch.tensor([c.readout_swap_at if c.readout_swap_at is not None else -1 for c in cfgs],
                                     dtype=torch.long, device=dev)
        st["swap_acc"] = torch.full((R0,), float("nan"), dtype=fleet.dtype, device=dev)
        st["swap_scale"] = torch.full((R0,), float("nan"), dtype=fleet.dtype, device=dev)
    vel, t_vel = ({k: torch.zeros_like(v) for k, v in fleet.params.items()}, 0) if hz else (None, 0)
    jump = any(c.drift_jump_at is not None for c in cfgs)
    if jump:
        if not hz:
            raise ValueError("drift_jump_at needs measure_horizon (the drift estimate)")
        st["jump_at"] = torch.tensor([c.drift_jump_at if c.drift_jump_at is not None else -1 for c in cfgs],
                                     dtype=torch.long, device=dev)
        st["jump_h"] = torch.tensor([float(c.drift_jump_h) for c in cfgs], dtype=fleet.dtype, device=dev)
        st["jump_acc_before"] = torch.full((R0,), float("nan"), dtype=fleet.dtype, device=dev)
        st["jump_acc_after"] = torch.full((R0,), float("nan"), dtype=fleet.dtype, device=dev)
    orig = list(range(R0))
    grok_step: list[int | None] = [None] * R0
    diverged: list[int | None] = [None] * R0
    trajs: list[list[dict]] = [[] for _ in range(R0)]
    probes: list[list[dict]] = [[] for _ in range(R0)]
    saved_lat: dict[int, list] = {}
    saved_extra: dict[int, dict] = {}  # per-run end-of-run state of runs compacted away
    END_KEYS = ("preq_nats", "preq_cor", "preq_n", "swap_acc", "swap_scale", "jump_acc_before", "jump_acc_after")
    inc_steps = torch.arange(K + 1, device=dev) * max(c0.arrive_every, 0)

    def batch_tensors():
        tr_ = st["tr"]
        return grid["a"][tr_], grid["b"][tr_], grid["y"].gather(1, tr_)

    A, B, Y = batch_tensors()
    exc_tr = None
    for step in range(c0.steps + 1):
        R = fleet.R
        w = (st["arrive"] <= step).to(fleet.dtype) if K > 0 else None
        if swap and bool((st["swap_at"] == step).any()):  # E2 intervention, before this step's update
            if K > 0:
                seen = torch.zeros_like(grid["train"])
                seen.scatter_(1, st["tr"], st["arrive"] <= step)
                grid["train"], grid["rule_train"] = seen, seen & ~grid["exc"]
            runs = (st["swap_at"] == step) & st["active"]
            sw = fh.swap_readout(fleet, opt, grid, st["hz_val"], runs, A, B, Y, w)
            st["swap_acc"] = torch.where(runs, sw["refit_test_acc"], st["swap_acc"])
            st["swap_scale"] = torch.where(runs, sw["scale"], st["swap_scale"])
        if jump and bool((st["jump_at"] == step).any()) and t_vel > 0:  # E2 intervention: skip ahead along the drift
            runs = (st["jump_at"] == step) & st["active"]
            before = fh.test_acc(fleet, grid)
            corr = 1.0 - fh.DRIFT_BETA ** t_vel
            with torch.no_grad():
                for k, v in fleet.params.items():
                    sh = (-1,) + (1,) * (v.dim() - 1)
                    v.add_(torch.where(runs.view(sh), (st["jump_h"].view(sh) / corr) * vel[k], torch.zeros_like(v)))
            after = fh.test_acc(fleet, grid)
            st["jump_acc_before"] = torch.where(runs, before, st["jump_acc_before"])
            st["jump_acc_after"] = torch.where(runs, after, st["jump_acc_after"])
        opt.zero_grad()
        per_run, ce, cor = _loss_and_grads(fleet, opt, A, B, Y, w, st["active"], st)
        finite = torch.isfinite(per_run)
        st["div_step"] = torch.where((~finite) & (st["div_step"] < 0), torch.full_like(st["div_step"], step), st["div_step"])
        st["active"] = st["active"] & finite
        if K > 0:
            newly = (st["arrive"] == step).to(fleet.dtype) * st["active"].to(fleet.dtype).view(-1, 1)
            st["preq_nats"] = st["preq_nats"].scatter_add(1, st["inc"], torch.nan_to_num(ce.detach(), nan=0.0) * newly)
            st["preq_cor"] = st["preq_cor"].scatter_add(1, st["inc"], cor.to(fleet.dtype) * newly)
            st["preq_n"] = st["preq_n"].scatter_add(1, st["inc"], newly)
        if K > 0:  # memorization latency per increment, from this step's training logits
            arrived = st["arrive"] <= step
            cnt = torch.zeros(R, K + 1, device=dev, dtype=fleet.dtype).scatter_add_(1, st["inc"], arrived.to(fleet.dtype))
            hit = torch.zeros_like(cnt).scatter_add_(1, st["inc"], (cor & arrived).to(fleet.dtype))
            acc_inc = hit / cnt.clamp_min(1.0)
            newly = (st["lat"] < 0) & (cnt > 0) & (acc_inc >= 0.9) & (inc_steps.view(1, -1) <= step)
            st["lat"] = torch.where(newly, step - inc_steps.view(1, -1), st["lat"])
        if is_eval_step(step, c0):
            if K > 0:  # training set = pairs that have arrived
                seen = torch.zeros_like(grid["train"])
                seen.scatter_(1, st["tr"], st["arrive"] <= step)
                grid["train"], grid["rule_train"] = seen, seen & ~grid["exc"]
                latest = st["inc"].masked_fill(st["arrive"] > step, -1).max(1).values  # (R,)
                is_new = (st["inc"] == latest.view(-1, 1)) & (st["arrive"] <= step)
                is_old = (st["inc"] < latest.view(-1, 1)) & (st["arrive"] <= step)
                newset = torch.zeros_like(grid["train"])
                newset.scatter_(1, st["tr"], is_new)
                grid["new"] = newset  # latest increment: per-component mem attribution (mem_new) in measure()
            meas = measure(fleet, grid, measure_chunk)
            extra = {}
            if c0.measure_signal:
                wb = w if w is not None else torch.ones(A.shape, dtype=fleet.dtype, device=dev)
                sig = fs.signal_measure(fleet, opt, grid, {"A": A, "B": B, "Y": Y, "w": wb}, st["active"], st)
                extra.update({f"sig_{k}": v for k, v in sig.items()})
            if opt.routing:
                extra.update(opt.take_angle())
            if opt.path_eff:
                extra["path_eff"] = opt.path_efficiency()
            if hz:
                rf = fh.refit_readout(fleet, grid, st["hz_val"])
                extra.update({k: rf[k] for k in ("refit_test_acc", "refit_train_acc", "refit_lam", "ridge_test_acc",
                                                 "ridge_train_acc")})
                extra.update(fh.extrapolated_acc(fleet, grid, vel, t_vel, c0.horizon_extrap))
            if K > 0:
                extra["old_loss"] = (ce * is_old).sum(1) / is_old.sum(1).clamp_min(1)
                extra["n_seen"] = (st["arrive"] <= step).sum(1).to(fleet.dtype)
            meas.update(extra)
            recs = _records({k: v.detach().cpu() for k, v in meas.items()}, grad_norms(fleet).cpu(),
                            weight_norms(fleet).cpu(), step)
            act, divs = st["active"].cpu().tolist(), st["div_step"].cpu().tolist()
            for j in range(R):
                o = orig[j]
                if divs[j] >= 0 and diverged[o] is None:
                    diverged[o] = divs[j]
                if not act[j]:
                    continue
                trajs[o].append(recs[j])
                if grok_step[o] is None and recs[j]["test_acc"] >= c0.grok_thresh:
                    grok_step[o] = step
                if c0.stop_after_grok and grok_step[o] is not None and step - grok_step[o] >= c0.stop_after_grok:
                    act[j] = False
            st["active"] = torch.tensor(act, dtype=torch.bool, device=dev)
            if verbose:
                print(f"step {step}: active {sum(act)}/{R} (of {R0})", flush=True)
            if not any(act):
                break
            if sum(act) < compact_below * R:
                keep = torch.tensor([j for j in range(R) if act[j]], dtype=torch.long, device=dev)
                lat_now = st["lat"].cpu().tolist()
                ends = {k: st[k].cpu().tolist() for k in END_KEYS if k in st}
                for j in range(R):
                    if not act[j]:
                        saved_lat[orig[j]] = lat_now[j]
                        saved_extra[orig[j]] = {k: v[j] for k, v in ends.items()}
                if vel is not None:
                    vel = {k: v[keep] for k, v in vel.items()}
                fleet.subset(keep)
                opt.subset(keep)
                for k in ("y", "train", "test", "exc", "rule_train", "new"):
                    if k in grid:
                        grid[k] = grid[k][keep]
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == R:
                        st[k] = v[keep]
                orig = [orig[j] for j in keep.tolist()]
                A, B, Y = batch_tensors()
                if w is not None:
                    w = w[keep]
                opt.zero_grad()
                # this step's gradient for the compacted batch (the consolidation EMA was already advanced)
                _loss_and_grads(fleet, opt, A, B, Y, w, st["active"], st, update_state=False)
        if c0.probe_steps and step in c0.probe_steps:
            pr = fs.probe(fleet, grid, st["probe_pairs"])
            if c0.probe_matched:
                pr.update(fs.matched_probe(fleet, opt, grid, st, A, B, Y, w, step, c0.probe_matched))
            prl = {k: v.cpu().tolist() for k, v in pr.items()}
            act = st["active"].cpu().tolist()
            for j in range(fleet.R):
                if act[j]:
                    probes[orig[j]].append({"step": step, **{k: v[j] for k, v in prl.items()}})
        if hz:
            prev = {k: v.detach().clone() for k, v in fleet.params.items()}
        opt.step(st["active"])
        if hz:  # drift: bias-corrected EMA of the per-step parameter change (E2 extrapolation)
            with torch.no_grad():
                for k, v in fleet.params.items():
                    vel[k] = fh.DRIFT_BETA * vel[k] + (1 - fh.DRIFT_BETA) * (v.detach() - prev[k])
            t_vel += 1
    secs = time.time() - t0
    divs = st["div_step"].cpu().tolist()
    lat = st["lat"].cpu().tolist()
    lat_by_orig = {**saved_lat, **{orig[j]: lat[j] for j in range(fleet.R)}}
    ends = {k: st[k].cpu().tolist() for k in END_KEYS if k in st}
    extra_by_orig = {**saved_extra, **{orig[j]: {k: v[j] for k, v in ends.items()} for j in range(fleet.R)}}
    for j in range(fleet.R):  # runs still in the batch that diverged after the last eval
        if divs[j] >= 0 and diverged[orig[j]] is None:
            diverged[orig[j]] = divs[j]
    out = []
    for j, c in enumerate(cfgs):
        summ = summarize(trajs[j], c, diverged[j])
        if K > 0:
            summ["latency"] = lat_by_orig.get(j)  # per increment (index 0 = initial set); -1 = never reached 90%
            e = extra_by_orig.get(j, {})
            if "preq_n" in e:  # per increment: summed CE (nats) at arrival, before training on it; n; accuracy
                summ["prequential"] = {"nats": e["preq_nats"], "n": e["preq_n"],
                                       "acc": [h / max(n_, 1.0) for h, n_ in zip(e["preq_cor"], e["preq_n"])]}
        if c.readout_swap_at is not None:
            e = extra_by_orig.get(j, {})
            summ["swap"] = {"step": c.readout_swap_at, "refit_test_acc": _clean([e.get("swap_acc")])[0],
                            "scale": _clean([e.get("swap_scale")])[0]}
        if c.drift_jump_at is not None:
            e = extra_by_orig.get(j, {})
            summ["jump"] = {"step": c.drift_jump_at, "h": c.drift_jump_h,
                            "test_acc_before": _clean([e.get("jump_acc_before")])[0],
                            "test_acc_after": _clean([e.get("jump_acc_after")])[0]}
        res = {"config": asdict(c), "run_id": c.run_id, "n_params": n_params, "trajectory": trajs[j], "summary": summ,
               "fleet": {"device": str(dev), "R": R0, "seconds_batch": secs}}
        if c0.probe_steps:
            res["probes"] = probes[j]
        out.append(res)
    return out
