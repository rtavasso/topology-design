"""Run-stacked rung-0 models: R independent runs of one arm in one model with a leading run dimension, so one
forward/backward trains all of them (every linear layer is a batched matmul over runs). Runs never share parameters
or data; the per-run losses are summed, so gradients are exactly those of separate training. Mirrors model.py.

Also StackedAdamW (a per-run learning-rate AdamW over leading-R parameters) and per-run gradient clipping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PtrConfig
from .model import FrozenSymbols, ModelConfig, PtrModel


class SLinear(nn.Module):
    def __init__(self, R, din, dout, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(R, din, dout))
        self.bias = nn.Parameter(torch.zeros(R, 1, dout)) if bias else None
        bound = 1.0 / math.sqrt(din)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        R = x.shape[0]
        y = torch.bmm(x.reshape(R, -1, x.shape[-1]), self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1)


class SLayerNorm(nn.Module):
    def __init__(self, R, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(R, 1, d))
        self.bias = nn.Parameter(torch.zeros(R, 1, d))

    def forward(self, x):
        R = x.shape[0]
        y = F.layer_norm(x, x.shape[-1:])
        return (y.reshape(R, -1, x.shape[-1]) * self.weight + self.bias).reshape(x.shape)


class SBlock(nn.Module):
    """Per-run pre-norm causal attention + MLP + linear map on the stream (identity at init)."""

    def __init__(self, R, d, heads, mlp):
        super().__init__()
        self.n1, self.n2 = SLayerNorm(R, d), SLayerNorm(R, d)
        self.qkv, self.o = SLinear(R, d, 3 * d), SLinear(R, d, d)
        self.f1, self.f2 = SLinear(R, d, mlp * d), SLinear(R, mlp * d, d)
        self.lin = SLinear(R, d, d)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(d).expand(R, d, d))
            self.lin.bias.zero_()
        self.h = heads

    def forward(self, x, mask):
        R, B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, -1)
        sh = lambda t: t.reshape(R * B, N, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(R, B, N, d))
        x = x + self.f2(F.gelu(self.f1(self.n2(x))))
        return self.lin(x)


class StackedPtrModel(nn.Module):
    def __init__(self, R: int, mc: ModelConfig, dc: PtrConfig):
        super().__init__()
        self.R, self.mc, self.dc = R, mc, dc
        d = mc.d
        self.sym = FrozenSymbols(dc.n_sym, d, mc.emb_seed)
        self.bos = nn.Parameter(torch.randn(R, d) * 0.02)
        self.marker = nn.Parameter(torch.randn(R, d) * 0.02)
        self.ans_tok = nn.Parameter(torch.randn(R, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(R, 1 + dc.P + 2 * dc.Qn + 64, d) * 0.02)
        if mc.kind == "static":
            self.blocks = nn.ModuleList(SBlock(R, d, mc.heads, mc.mlp) for _ in range(mc.layers))
        else:
            self.block = SBlock(R, d, mc.heads, mc.mlp)
            if mc.halting == "learned":
                self.halt_norm = SLayerNorm(R, d)
                if mc.halt_input == "index":
                    self.halt_bias = nn.Parameter(torch.zeros(R, 64) + mc.halt_bias_init)
                elif mc.halt_input == "delta":
                    self.halt = SLinear(R, 2, 1)
                    with torch.no_grad():
                        self.halt.weight.normal_(std=0.02)
                        self.halt.bias.fill_(mc.halt_bias_init)
                else:
                    self.halt = SLinear(R, d + (16 if mc.halt_input == "state+index" else 0), 1)
                    with torch.no_grad():
                        self.halt.weight.normal_(std=0.02)
                        self.halt.bias.fill_(mc.halt_bias_init)
                    if mc.halt_input == "state+index":
                        self.step_emb = nn.Parameter(torch.randn(R, 64, 16) * 0.1)
            if mc.halting == "value":
                self.val_norm = SLayerNorm(R, d)
                self.val = SLinear(R, d, 1)
        self.out_norm = SLayerNorm(R, d)

    def embed(self, b: dict):
        """b: tensors (R, B, ·) on the model's device."""
        R, B, P = self.R, b["pair_key"].shape[1], self.dc.P
        Qn = b["q_start"].shape[2]
        pairs = self.sym.key_slot[b["pair_key"]] + self.sym.val_slot[b["pair_val"]]           # (R, B, P, d)
        queries = self.marker[:, None, None, :] + self.sym.reg_slot[b["q_start"]]
        answers = self.ans_tok[:, None, None, :] + self.sym.reg_slot[b["q_ans"]]
        qa = torch.stack([queries, answers], 3).reshape(R, B, 2 * Qn, -1)
        e0 = torch.cat([self.bos[:, None, None, :].expand(R, B, 1, -1), pairs, qa], 2)
        N = e0.shape[2]
        e0 = e0 + self.pos[:, None, :N]
        dev = e0.device
        qpos = 1 + P + 2 * torch.arange(Qn, device=dev)
        targets = b["q_ans"] - self.dc.n_keys
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=dev))
        if self.mc.pair_pair_mask:
            pp = torch.zeros(N, N, dtype=torch.bool, device=dev)
            pp[1:P + 1, 1:P + 1] = True
            pp[torch.arange(1, P + 1), torch.arange(1, P + 1)] = False
            mask = mask & ~pp
        return e0, qpos, targets, mask

    def readout(self, h):
        z = self.out_norm(h)
        return z @ self.sym.reg_slot[self.dc.n_keys:self.dc.n_keys + self.dc.n_ans].T

    def forward(self, b: dict, *, T=None, steps=None, mode="train"):
        mc = self.mc
        e0, qpos, targets, mask = self.embed(b)
        R, B, N, d = e0.shape
        dev = e0.device
        out = {"targets": targets, "qpos": qpos}
        if mc.kind == "static":
            h = e0
            for blk in self.blocks:
                h = blk(h, mask)
            out["logits"] = self.readout(h[:, :, qpos])
            out["steps"] = torch.full((R, B, N), float(mc.layers), device=dev)
            return out
        T = mc.T if T is None else T
        h = e0
        active = torch.ones(R, B, N, dtype=torch.bool, device=dev)
        is_query = torch.zeros(N, dtype=torch.bool, device=dev)
        is_query[qpos] = True
        used = torch.zeros(R, B, N, device=dev)
        per_step_logits, lams, vals, hs, p_halt = [], [], [], [], []
        remaining = torch.ones(R, B, N, device=dev)
        delta = mc.halting == "learned" and mc.halt_input == "delta"
        if delta:
            prev_h, prev_lp = e0, F.log_softmax(self.readout(e0).float(), -1)
        for t in range(1, T + 1):
            x = h + e0 if mc.injection else h
            h_new = self.block(x, mask)
            if mode == "ponder_train" and mc.halting == "learned":
                upd = active | is_query
                h = torch.where(upd.unsqueeze(-1), h_new, h)
                used += upd.float()
            else:
                h = torch.where(active.unsqueeze(-1), h_new, h)
                used += active.float()
            hs.append(h[:, :, qpos])          # per-step states at the query positions only (traces, probe, LR)
            if mc.ctx_single_step:
                active = active & is_query
            if mc.halting == "learned":
                if mc.halt_input == "index":
                    lam = torch.sigmoid(self.halt_bias[:, min(t - 1, 63)].float()).view(R, 1, 1).expand(R, B, N)
                elif delta:
                    lp = F.log_softmax(self.readout(h).float(), -1)
                    feats = PtrModel.delta_features(h, prev_h, lp, prev_lp).to(self.halt.weight.dtype)
                    lam = torch.sigmoid(self.halt(feats).float()).squeeze(-1)
                    prev_h, prev_lp = h, lp
                else:
                    z = self.halt_norm(h)
                    if mc.halt_input == "state+index":
                        z = torch.cat([z, self.step_emb[:, min(t - 1, 63)].view(R, 1, 1, -1).expand(R, B, N, -1)], -1)
                    lam = torch.sigmoid(self.halt(z).float()).squeeze(-1)     # fp32: p_halt must not underflow
                if t == T:
                    lam = torch.ones_like(lam)
                p_halt.append(remaining * lam)
                remaining = remaining * (1 - lam)
                lams.append(lam)
                per_step_logits.append(self.readout(h[:, :, qpos]))
                cum = torch.stack(p_halt, 0).sum(0)
                active = active & (cum.detach() < 0.5)
            elif mc.halting == "value":
                v = self.val(self.val_norm(h)).float().squeeze(-1)
                vals.append(v)
                if mode != "train":
                    active = active & (v >= 0.5)
            elif mc.halting == "oracle":
                assert steps is not None
                active = active & (used < steps)
        out.update(h=h, hs=hs, steps=used, logits=self.readout(h[:, :, qpos]))
        if mc.halting == "learned":
            out["p_halt"] = torch.stack(p_halt, 0)                     # (T, R, B, N)
            out["lams"] = torch.stack(lams, 0)
            out["per_step_logits"] = torch.stack(per_step_logits, 0)   # (T, R, B, Q, n_ans)
            with torch.no_grad():
                out["halted_logits"], out["halt_step"] = PtrModel.halted_readout(out["p_halt"].float(),
                                                                                 out["per_step_logits"], qpos)
        if mc.halting == "value":
            out["vals"] = torch.stack(vals, 0)
        return out

    def loss(self, out, remaining_hops=None):
        """Per-run losses (R,) and their sum; each run's gradient equals separate training."""
        mc = self.mc
        tgt = out["targets"]                                            # (R, B, Q)
        R = tgt.shape[0]
        ce_of = lambda lg: F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                           reduction="none").view(tgt.shape)
        extra = {}
        if mc.kind == "looped" and mc.halting == "learned" and "per_step_logits" in out:
            T = out["per_step_logits"].shape[0]
            ce = torch.stack([ce_of(out["per_step_logits"][t]) for t in range(T)], 0)      # (T, R, B, Q)
            pq = out["p_halt"][:, :, :, out["qpos"]]
            expected = (pq * ce).sum(0).mean((1, 2))                                        # (R,)
            p = out["p_halt"].float().clamp_min(1e-9)
            if mc.ctx_single_step:
                p = p[:, :, :, out["qpos"]]
            geo = torch.tensor([mc.lam_p * (1 - mc.lam_p) ** t for t in range(T)], device=tgt.device)
            geo = geo / geo.sum()
            kl = (p * (p.log() - geo.log().view(T, 1, 1, 1))).sum(0).mean((1, 2))          # (R,)
            per_run = expected + mc.beta * kl
            extra = {"ce": expected.detach(), "kl": kl.detach()}
            return per_run.sum(), per_run.detach(), extra
        ce = ce_of(out["logits"]).mean((1, 2))                                              # (R,)
        extra = {"ce": ce.detach()}
        per_run = ce
        if mc.kind == "looped" and mc.halting == "value":
            vl = F.smooth_l1_loss(out["vals"].float(), remaining_hops, reduction="none").mean((0, 2, 3))
            per_run = per_run + vl
            extra["value"] = vl.detach()
        return per_run.sum(), per_run.detach(), extra


def oracle_steps_t(b: dict, dc: PtrConfig) -> torch.Tensor:
    R, B, Qn = b["q_k"].shape
    s = torch.ones(R, B, 1 + dc.P + 2 * Qn, device=b["q_k"].device)
    s[:, :, 1 + dc.P::2] = b["q_k"].float()
    return s


def remaining_after_t(b: dict, dc: PtrConfig, T: int) -> torch.Tensor:
    R, B, Qn = b["q_k"].shape
    r = torch.zeros(T, R, B, 1 + dc.P + 2 * Qn, device=b["q_k"].device)
    k = b["q_k"].float()
    for t in range(1, T + 1):
        r[t - 1, :, :, 1 + dc.P::2] = (k - t).clamp_min(0)
    return r


class StackedAdamW:
    """AdamW over parameters with a leading run dimension, with a per-run learning rate (R,)."""

    def __init__(self, params, lr: torch.Tensor, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr.clone()
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self, scale: float = 1.0, active: torch.Tensor | None = None):
        """active: (R,) bool; runs marked False (gradient overflow) skip this step entirely, moments included."""
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if active is not None:
                a = active.view(-1, *([1] * (p.dim() - 1))).to(g.dtype)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * a
                m.mul_(torch.where(a > 0, torch.full_like(a, self.b1), torch.ones_like(a))).add_(g * (1 - self.b1))
                v.mul_(torch.where(a > 0, torch.full_like(a, self.b2), torch.ones_like(a))).addcmul_(g, g, value=1 - self.b2)
            else:
                a = None
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            lr = (self.lr * scale).view(-1, *([1] * (p.dim() - 1)))
            if a is not None:
                lr = lr * a
            upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            p.mul_(1 - lr * self.wd).sub_(lr * upd)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


@torch.no_grad()
def clip_grad_norm_per_run(params, max_norm: float) -> torch.Tensor:
    """Clip each run's gradient (over all its parameters) to max_norm; returns the per-run norms (R,)."""
    params = [p for p in params if p.grad is not None]
    R = params[0].shape[0]
    sq = torch.zeros(R, device=params[0].device)
    for p in params:
        sq += p.grad.reshape(R, -1).pow(2).sum(1)
    norms = sq.sqrt()
    coef = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    coef = torch.where(torch.isfinite(norms), coef, torch.zeros_like(coef))
    for p in params:
        p.grad.mul_(coef.view(-1, *([1] * (p.dim() - 1))))
    return norms
