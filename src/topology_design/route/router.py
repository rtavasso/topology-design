"""The Markov router of rung 1 (experiments/route/DESIGN.md §4), run-stacked (leading run dimension R).

pi_theta(a | s, c): the state token (frozen register embedding of symbol s + a learned marker) cross-attends over the
context's pair tokens (frozen key/value/prev slot embeddings + learned positions, each passed through a small learned
per-token MLP), then LayerNorm -> MLP -> 3 logits [HOP, PREV, HALT]. The router sees no step index and no history.
Because pi depends only on (s, c), it is evaluated once per episode for all n_sym symbols: logits (R, B, S, 3).

Variants: halt_input="index" gives a step-indexed, state-blind policy (a learned fixed program): logits (T, R, 1, 3).
plant(router) sets the weights to the exact rule (DESIGN.md §6), one unit from the init.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ptr.model import FrozenSymbols
from ..ptr.stacked import SLinear, SLayerNorm
from .data import RouteConfig


@dataclass
class RouterConfig:
    kind: str = "state"           # "state" (Markov in the symbolic state) | "index" (step-indexed, state-blind)
    d: int = 344                  # 5 * 66 + 8 -> rounded up to a multiple of 8: five disjoint symbol subspaces
    heads: int = 2
    mlp: int = 2
    T: int = 12                   # training horizon (forced halt)
    lam: float = 0.02             # step cost per circuit application
    eps: float = 0.02             # readout error of a halted answer
    emb_seed: int = 1234
    logit_scale: float = 1.0      # init scale of the head weights (N(0, 0.02^2) x scale)

    def dict(self):
        return asdict(self)


class StackedRouter(nn.Module):
    def __init__(self, R: int, rc: RouterConfig, dc: RouteConfig):
        super().__init__()
        self.R, self.rc, self.dc = R, rc, dc
        d = rc.d
        self.sym = FrozenSymbols(dc.n_sym, d, rc.emb_seed, slots=5)
        assert self.sym.exact, "d must hold five disjoint symbol subspaces"
        if rc.kind == "index":
            self.table = nn.Parameter(torch.zeros(R, 64, 3))
            return
        self.marker = nn.Parameter(torch.randn(R, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(R, dc.P + 1, d) * 0.02)
        self.ctx_norm = SLayerNorm(R, d)
        self.ctx_f1, self.ctx_f2 = SLinear(R, d, rc.mlp * d), SLinear(R, rc.mlp * d, d)
        self.q, self.k, self.v, self.o = SLinear(R, d, d), SLinear(R, d, d), SLinear(R, d, d), SLinear(R, d, d)
        self.norm = SLayerNorm(R, d)
        self.f1, self.f2 = SLinear(R, d, rc.mlp * d), SLinear(R, rc.mlp * d, d)
        self.head_norm = SLayerNorm(R, d)
        self.head = SLinear(R, d, 3)
        with torch.no_grad():
            self.head.weight.normal_(std=0.02 * rc.logit_scale)
            self.head.bias.zero_()

    def context(self, b: dict):
        """(R, B, P + 1, d): BOS sink then the pair tokens, after the per-token MLP (no attention between pairs:
        the context is read-only, as in rung 0)."""
        R, B, P = self.R, b["pair_key"].shape[1], self.dc.P
        x = self.sym.key_slot[b["pair_key"]] + self.sym.val_slot[b["pair_val"]] + self.sym.prev_slot[b["pair_prev"]]
        bos = torch.zeros(R, B, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([bos, x], 2) + self.pos[:, None, :P + 1]
        x = x + self.ctx_f2(F.gelu(self.ctx_f1(self.ctx_norm(x))))
        return x

    def logits(self, b: dict, T: int | None = None):
        """State-conditioned: (R, B, S, 3) for every symbol. Index-conditioned: (T, R, 1, 3) (broadcast over B)."""
        R = self.R
        if self.rc.kind == "index":
            T = self.rc.T if T is None else T
            idx = torch.clamp(torch.arange(T, device=self.table.device), max=self.rc.T - 1)   # repeat the last trained logit
            return self.table[:, idx].permute(1, 0, 2).unsqueeze(2)                           # (T, R, 1, 3)
        ctx = self.context(b)                                                                  # (R, B, N, d)
        B, N, d = ctx.shape[1], ctx.shape[2], ctx.shape[3]
        S = self.dc.n_sym
        st = self.sym.reg_slot.unsqueeze(0).expand(R, S, d) + self.marker[:, None, :]           # (R, S, d)
        h = self.rc.heads; hd = d // h
        q = self.q(st).view(R, 1, S, h, hd).expand(R, B, S, h, hd).reshape(R * B, S, h, hd).transpose(1, 2)
        k = self.k(ctx).view(R * B, N, h, hd).transpose(1, 2)
        v = self.v(ctx).view(R * B, N, h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)                                            # (R*B, h, S, hd)
        a = a.transpose(1, 2).reshape(R, B, S, d)
        x = st.unsqueeze(1) + self.o(a)
        x = x + self.f2(F.gelu(self.f1(self.norm(x))))
        return self.head(self.head_norm(x)).float()                                            # (R, B, S, 3)

    def features(self, b: dict):
        """The head's input (R, B, S, d) after the LayerNorm, for the remaining-distance probe."""
        ctx = self.context(b)
        R, B, N, d = ctx.shape
        S = self.dc.n_sym
        st = self.sym.reg_slot.unsqueeze(0).expand(R, S, d) + self.marker[:, None, :]
        h = self.rc.heads; hd = d // h
        q = self.q(st).view(R, 1, S, h, hd).expand(R, B, S, h, hd).reshape(R * B, S, h, hd).transpose(1, 2)
        k = self.k(ctx).view(R * B, N, h, hd).transpose(1, 2)
        v = self.v(ctx).view(R * B, N, h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(R, B, S, d)
        x = st.unsqueeze(1) + self.o(a)
        x = x + self.f2(F.gelu(self.f1(self.norm(x))))
        return self.head_norm(x)


@torch.no_grad()
def plant(m: StackedRouter, beta: float = 30.0, kappa: float = 15.0, gain: float = 3.0, hop_bias: float = 4.0) -> StackedRouter:
    """The exact rule (DESIGN.md §6). Head 0 matches the state's register against the pair keys (score beta on the
    match; a BOS sink at kappa for symbols with no pair) and reads the pair's value slot back into the stream. The
    linear head then tests the read value: PREV fires on the ARROW direction, HALT on the answer directions of the
    register itself (a state holding an answer has no matching pair, so the sink wins and the read value is zero),
    and HOP wins by a constant bias when neither fires (a key or an answer in the value slot: hop to it). The per-token MLPs
    are zeroed. Head weights have norm `gain` and the HOP bias `hop_bias`: a few units from the init (L24)."""
    R, d, dc = m.R, m.rc.d, m.dc
    hd = d // m.rc.heads
    n = dc.n_sym
    assert n <= hd
    for lin in (m.ctx_f1, m.ctx_f2, m.f1, m.f2, m.o, m.q, m.k, m.v):
        lin.weight.zero_(); lin.bias.zero_()
    m.pos.zero_(); m.marker.zero_()
    scale = beta * math.sqrt(hd)                                 # undo SDPA's 1/sqrt(hd)
    Q = torch.zeros(d, d); K = torch.zeros(d, d); V = torch.zeros(d, d); O = torch.zeros(d, d)
    Q[:, :n] = m.sym.reg_slot.T * scale                           # state register -> one-hot coordinates of head 0
    K[:, :n] = m.sym.key_slot.T                                   # pair key -> the same coordinates
    V[:, :n] = m.sym.val_slot.T                                   # pair value -> coordinates, read back by o
    O[:n, :] = m.sym.val_slot                                     # coordinates -> the value-slot subspace
    for lin, W in ((m.q, Q), (m.k, K), (m.v, V), (m.o, O)):
        lin.weight.copy_(W.unsqueeze(0).expand(R, d, d))
    # the BOS sink: K^T pos[0] = (kappa / beta) * ones(n) gives every state a score kappa with BOS
    m.pos[:, 0] = (m.sym.key_slot.T @ torch.full((n,), kappa / beta)).unsqueeze(0).expand(R, d)
    ans_reg = m.sym.reg_slot[dc.n_keys:dc.n_keys + dc.n_ans].sum(0); ans_reg = ans_reg / ans_reg.norm()
    arrow = m.sym.val_slot[dc.ARROW]
    W = torch.stack([torch.zeros(d), arrow * gain, ans_reg * gain], 1)                 # (d, 3): HOP, PREV, HALT
    m.head.weight.copy_(W.unsqueeze(0).expand(R, d, 3))
    m.head.bias.zero_(); m.head.bias[:, :, 0] = hop_bias
    return m
