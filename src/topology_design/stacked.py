"""Stacked (run-batched) transformer pieces for the worlds round (experiments/worlds/DESIGN.md §8).

R independent models of identical shape are stored with a leading run dimension, so one forward/backward pass
trains all of them: every linear layer is a batched matmul over runs. Runs never interact (no shared parameters,
per-run losses are summed), so the result is identical to training them separately, up to floating-point order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SLinear(nn.Module):
    """R independent linear maps: x (R, ..., din) -> (R, ..., dout)."""

    def __init__(self, R: int, din: int, dout: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(R, din, dout))
        self.bias = nn.Parameter(torch.zeros(R, 1, dout)) if bias else None
        bound = 1.0 / math.sqrt(din)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = x.shape[0]
        y = torch.bmm(x.reshape(R, -1, x.shape[-1]), self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1)


class SLayerNorm(nn.Module):
    def __init__(self, R: int, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(R, 1, d))
        self.bias = nn.Parameter(torch.zeros(R, 1, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = x.shape[0]
        y = F.layer_norm(x, x.shape[-1:])
        return (y.reshape(R, -1, x.shape[-1]) * self.weight + self.bias).reshape(x.shape)


class SAttention(nn.Module):
    """Multi-head attention per run. q_in (R, B, Tq, d), kv_in (R, B, Tk, d), mask (R, B, Tq, Tk) bool (True = keep)."""

    def __init__(self, R: int, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (SLinear(R, d, d) for _ in range(4))

    def forward(self, q_in, kv_in, mask=None):
        R, B, Tq, d = q_in.shape
        Tk = kv_in.shape[2]
        sh = lambda t, T: t.reshape(R * B, T, self.h, self.dh).transpose(1, 2)  # (R*B, h, T, dh)
        q, k, v = sh(self.q(q_in), Tq), sh(self.k(kv_in), Tk), sh(self.v(kv_in), Tk)
        m = None if mask is None else mask.reshape(R * B, 1, Tq, Tk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(out.transpose(1, 2).reshape(R, B, Tq, d))


class SBlock(nn.Module):
    """Pre-norm transformer block per run; optional cross-attention."""

    def __init__(self, R: int, d: int, heads: int, mlp: int, cross: bool = False):
        super().__init__()
        self.n1, self.att = SLayerNorm(R, d), SAttention(R, d, heads)
        self.cross = cross
        if cross:
            self.nc, self.catt = SLayerNorm(R, d), SAttention(R, d, heads)
        self.n2, self.f1, self.f2 = SLayerNorm(R, d), SLinear(R, d, mlp), SLinear(R, mlp, d)

    def forward(self, x, mask=None, mem=None, mem_mask=None):
        h = self.n1(x)
        x = x + self.att(h, h, mask)
        if self.cross:
            x = x + self.catt(self.nc(x), mem, mem_mask)
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class StackedPFN(nn.Module):
    """Monolithic PFN for the worlds round, R runs stacked: context tokens (x, y, f) and query tokens (x, y, ?);
    queries attend to the context (and themselves) through an encoder; output: logits over Z_p per query."""

    def __init__(self, R: int, p: int = 7, d: int = 128, layers: int = 4, heads: int = 4, mlp: int = 512):
        super().__init__()
        self.R, self.p = R, p
        self.ex = nn.Parameter(torch.randn(R, p, d) * 0.02)
        self.ey = nn.Parameter(torch.randn(R, p, d) * 0.02)
        self.ef = nn.Parameter(torch.randn(R, p + 1, d) * 0.02)  # index p = unknown (query)
        self.blocks = nn.ModuleList(SBlock(R, d, heads, mlp) for _ in range(layers))
        self.norm, self.head = SLayerNorm(R, d), SLinear(R, d, p)

    def embed(self, x, y, f):
        """x, y, f: (R, B, T) integer tensors -> (R, B, T, d)."""
        g = lambda E, i: torch.gather(E.unsqueeze(1).expand(-1, i.shape[1], -1, -1), 2,
                                      i.unsqueeze(-1).expand(-1, -1, -1, E.shape[-1]))
        return g(self.ex, x) + g(self.ey, y) + g(self.ef, f)

    def forward(self, cx, cy, cf, qx, qy):
        """Context (R, B, K) and queries (R, B, Q). Queries see the context and themselves only (no query-query)."""
        R, B, K = cx.shape
        Q = qx.shape[2]
        unk = torch.full_like(qx, self.p)
        h = torch.cat([self.embed(cx, cy, cf), self.embed(qx, qy, unk)], 2)
        T = K + Q
        mask = torch.zeros(T, T, dtype=torch.bool, device=h.device)
        mask[:, :K] = True
        mask[torch.arange(T), torch.arange(T)] = True
        mask = mask.expand(R, B, T, T)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.norm(h[:, :, K:]))
