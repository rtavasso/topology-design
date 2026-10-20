"""Batched trainer: many grokking runs sharing one routing structure, trained as one stacked model.

Runs in a batch must share (parents, p, d, hidden, beta, out_scale, train_frac, steps, eval_every, init_scale).
They may differ in model_seed, data_seed, lr, weight_decay, and branch multipliers alpha (so all sink_s arms,
which share parents (0,1,0), batch together).

Equivalence to `grokking.train_run`:
- initial weights are copied from `grokking.build_model(cfg)` for each run, so they are identical;
- each run uses its own train split, its own mean cross-entropy, and its own AdamW state; runs share no
  parameters, so the summed loss gives exactly each run's own gradient;
- AdamW is re-implemented with the same update order as torch.optim.AdamW (decoupled decay, then moments,
  bias correction, eps outside the sqrt);
- measurements are taken by loading each run's slice into a RoutedMLP and calling `grokking.measure`.

Floating-point summation order differs from the unbatched path (bmm vs mm), so trajectories agree to float
tolerance early on and may drift apart later in chaotic phases; tests check both early agreement and isolation
between runs.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from . import grokking as gk

STRUCT_FIELDS = ("p", "d", "hidden", "beta", "out_scale", "train_frac", "steps", "eval_every", "init_scale",
                 "grok_thresh", "stop_after_grok")


def struct_key(cfg: gk.RunConfig) -> tuple:
    parents, _ = gk.arm_spec(cfg.arm)
    return (parents,) + tuple(getattr(cfg, f) for f in STRUCT_FIELDS)


class BatchedRoutedMLP:
    """Stacked parameters for R runs. Parameter names mirror RoutedMLP."""

    def __init__(self, cfgs: list[gk.RunConfig], device: str = "cpu"):
        keys = {struct_key(c) for c in cfgs}
        if len(keys) != 1:
            raise ValueError("all runs in a batch must share the routing structure and shape fields")
        self.cfgs, self.device = cfgs, torch.device(device)
        self.R = len(cfgs)
        c0 = cfgs[0]
        self.parents, _ = gk.arm_spec(c0.arm)
        self.L, self.p, self.beta, self.out_scale = len(self.parents), c0.p, c0.beta, c0.out_scale
        singles = [gk.build_model(c) for c in cfgs]  # exact per-run initialization
        self.alphas = torch.stack([m.alphas for m in singles]).to(self.device)  # (R, L)
        st = lambda get: torch.stack([get(m).detach().clone() for m in singles]).to(self.device).requires_grad_(True)
        self.params = {
            "embed_a": st(lambda m: m.embed_a.weight),
            "embed_b": st(lambda m: m.embed_b.weight),
            "unembed": st(lambda m: m.unembed.weight),
        }
        for i in range(self.L):
            self.params[f"w1_{i}"] = st(lambda m, i=i: m.w1[i].weight)  # (R, hidden, d)
            self.params[f"w2_{i}"] = st(lambda m, i=i: m.w2[i].weight)  # (R, d, hidden)
        self._template = singles[0]  # reused for per-run measurement

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: (R, n) token indices -> logits (R, n, p)."""
        P = self.params
        r = torch.arange(self.R, device=self.device)[:, None]
        hs = [P["embed_a"][r, a] + P["embed_b"][r, b]]
        for i, k in enumerate(self.parents):
            z = torch.relu(torch.bmm(hs[k], P[f"w1_{i}"].transpose(1, 2)))
            w = self.alphas[:, i, None, None] * torch.bmm(z, P[f"w2_{i}"].transpose(1, 2))
            hs.append(self.beta * hs[-1] + w)
        return self.out_scale * torch.bmm(hs[-1], P["unembed"].transpose(1, 2))

    def single(self, j: int) -> gk.RoutedMLP:
        """Load run j's current parameters into a CPU RoutedMLP (for measurement)."""
        m = self._template
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

    def group_grad_norms(self, j: int) -> dict[str, float]:
        P = self.params
        g = lambda name: float(P[name].grad[j].pow(2).sum())
        out = {"embed": (g("embed_a") + g("embed_b")) ** 0.5, "unembed": g("unembed") ** 0.5}
        for i in range(self.L):
            out[f"block{i + 1}"] = (g(f"w1_{i}") + g(f"w2_{i}")) ** 0.5
        return out


class BatchedAdamW:
    """torch.optim.AdamW semantics with per-run lr and weight decay (vectors over the leading run dim)."""

    def __init__(self, params: dict[str, torch.Tensor], lr: torch.Tensor, wd: torch.Tensor,
                 betas=(0.9, 0.98), eps: float = 1e-8):
        self.params, self.lr, self.wd = params, lr, wd
        self.b1, self.b2, self.eps = betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2_sqrt = (1 - self.b2 ** self.t) ** 0.5
        for k, p in self.params.items():
            shape = (-1,) + (1,) * (p.dim() - 1)
            lr, wd = self.lr.view(shape), self.wd.view(shape)
            p.mul_(1 - lr * wd)
            self.m[k].lerp_(p.grad, 1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            denom = (self.v[k].sqrt() / bc2_sqrt).add_(self.eps)
            p.sub_((lr / bc1) * self.m[k] / denom)

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None


def train_batch(cfgs: list[gk.RunConfig], device: str = "cpu", verbose: bool = False) -> list[dict]:
    """Train all runs in `cfgs` together. Returns one result dict per run, same format as gk.train_run."""
    model = BatchedRoutedMLP(cfgs, device)
    dev = model.device
    c0 = cfgs[0]
    a_full, b_full, y_full = gk.full_grid(c0.p)
    splits = [gk.split(c.p, c.train_frac, c.data_seed) for c in cfgs]
    tr_idx = torch.stack([tr for tr, _ in splits])  # (R, n_train), same n for all
    A, B, Y = a_full[tr_idx].to(dev), b_full[tr_idx].to(dev), y_full[tr_idx].to(dev)
    lr = torch.tensor([c.lr for c in cfgs], dtype=torch.float32, device=dev)
    wd = torch.tensor([c.weight_decay for c in cfgs], dtype=torch.float32, device=dev)
    opt = BatchedAdamW(model.params, lr, wd)
    R, n = A.shape
    trajs: list[list[dict]] = [[] for _ in range(R)]
    grok_step: list[int | None] = [None] * R
    done = [False] * R
    t0 = time.time()
    for step in range(c0.steps + 1):
        opt.zero_grad()
        logits = model.forward(A, B)  # (R, n, p)
        per_run = F.cross_entropy(logits.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1)
        per_run.sum().backward()
        if step % c0.eval_every == 0:
            with torch.no_grad():
                pr = torch.softmax(logits.detach(), -1)
                pr.scatter_add_(-1, Y[..., None], -torch.ones_like(Y[..., None], dtype=pr.dtype))
                res = pr.norm(dim=-1).mean(1).cpu()
            for j in range(R):
                if done[j]:
                    continue
                rec = {"step": step, **gk.measure(model.single(j), splits[j][0], splits[j][1]),
                       "residual_norm": float(res[j]), "grad_norms": model.group_grad_norms(j)}
                trajs[j].append(rec)
                if grok_step[j] is None and rec["test_acc"] >= c0.grok_thresh:
                    grok_step[j] = step
                if c0.stop_after_grok and grok_step[j] is not None and step - grok_step[j] >= c0.stop_after_grok:
                    done[j] = True
            if verbose:
                print(f"step {step}: {sum(done)}/{R} done; test acc {[round(t[-1]['test_acc'], 2) for t in trajs]}",
                      flush=True)
            if all(done):
                break
        opt.step()
    secs = time.time() - t0
    out = []
    for j, c in enumerate(cfgs):
        out.append({
            "config": asdict(c), "run_id": c.run_id, "n_params": gk.n_params(model._template),
            "seconds": secs, "seconds_note": f"wall time of a batch of {R} runs on {device}",
            "trajectory": trajs[j], "summary": gk.summarize(trajs[j], c), "batched": True,
        })
    return out
