"""The Markov router of rung 1 (experiments/route/DESIGN.md §4, revision 2), run-stacked (leading run dimension R).

pi_theta(a | state, c): the state token (the frozen register embedding of the symbol plus, when state_input = "full",
frozen directions carrying the pending count and the set-here bit; the symbol alone when "symbol") cross-attends over
the context's pair tokens (frozen key/value/prev slot embeddings, a frozen flag direction, learned positions, a
learned per-token MLP; no attention between pairs), then LayerNorm -> MLP -> LayerNorm -> 4 logits [HOP, PREV, SET,
HALT]. The router sees no step index and no history. Since pi depends only on (state, c) it is evaluated once per
episode for all NS product states: logits (R, B, NS, 4).

kind = "index": a learned table pi(a | t), state-blind (a learned fixed program): logits (T, R, 1, 4).
plant(router) sets the exact rule for the configured deferral (DESIGN.md §6).
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
from .exec import PER_SYM, NH, HOP, PREV, SET, HALT


@dataclass
class RouterConfig:
    kind: str = "state"           # "state" (Markov in the product state) | "index" (step-indexed, state-blind)
    state_input: str = "full"     # "full" (symbol, pending, set-here) | "symbol" (the symbol only: the F4 null)
    d: int = 344                  # 5 * 65 + 19: five disjoint symbol subspaces + free directions (flag, count, set)
    heads: int = 2
    mlp: int = 2
    T: int = 14                   # training horizon (forced halt): k_max + the SET steps + slack
    lam: float = 0.02             # step cost per action other than HALT
    eps: float = 0.02             # readout error of a halted answer
    emb_seed: int = 1234

    def dict(self):
        return asdict(self)


class StackedRouter(nn.Module):
    FLAG, CNT, SETH = 3, 4, 5     # rows of the frozen free directions used for the flag, the count and set-here

    def __init__(self, R: int, rc: RouterConfig, dc: RouteConfig):
        super().__init__()
        self.R, self.rc, self.dc = R, rc, dc
        d = rc.d
        self.sym = FrozenSymbols(dc.n_sym, d, rc.emb_seed, slots=5)
        assert self.sym.exact and self.sym.free.shape[0] >= 6, "d must hold five symbol subspaces and six free directions"
        self.NS = dc.n_sym * PER_SYM
        if rc.kind == "index":
            self.table = nn.Parameter(torch.zeros(R, 64, 4))
            return
        self.marker = nn.Parameter(torch.randn(R, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(R, dc.P + 1, d) * 0.02)
        self.ctx_norm = SLayerNorm(R, d)
        self.ctx_f1, self.ctx_f2 = SLinear(R, d, rc.mlp * d), SLinear(R, rc.mlp * d, d)
        self.q, self.k, self.v, self.o = SLinear(R, d, d), SLinear(R, d, d), SLinear(R, d, d), SLinear(R, d, d)
        self.norm = SLayerNorm(R, d)
        self.f1, self.f2 = SLinear(R, d, rc.mlp * d), SLinear(R, rc.mlp * d, d)
        self.head_norm = SLayerNorm(R, d)
        self.head = SLinear(R, d, 4)
        with torch.no_grad():
            self.head.weight.normal_(std=0.02)
            self.head.bias.zero_()

    def state_tokens(self):
        """(NS, d): frozen embeddings of every product state (symbol register + count and set-here markers)."""
        ids = torch.arange(self.NS, device=self.sym.reg_slot.device)
        s, c, h = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
        x = self.sym.reg_slot[s]
        if self.rc.state_input == "full":
            x = x + c.float().unsqueeze(-1) * self.sym.free[self.CNT] + h.float().unsqueeze(-1) * self.sym.free[self.SETH]
        return x

    def context(self, b: dict):
        """(R, B, P + 1, d): BOS sink then the pair tokens after the per-token MLP (no attention between pairs)."""
        R, B, P = self.R, b["pair_key"].shape[1], self.dc.P
        x = (self.sym.key_slot[b["pair_key"]] + self.sym.val_slot[b["pair_val"]] + self.sym.prev_slot[b["pair_prev"]]
             + b["pair_flag"].to(self.sym.free.dtype).unsqueeze(-1) * self.sym.free[self.FLAG])
        bos = torch.zeros(R, B, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([bos, x], 2) + self.pos[:, None, :P + 1]
        return x + self.ctx_f2(F.gelu(self.ctx_f1(self.ctx_norm(x))))

    def _trunk(self, b: dict):
        ctx = self.context(b)
        R, B, N, d = ctx.shape
        st = self.state_tokens().unsqueeze(0).expand(R, self.NS, d) + self.marker[:, None, :]
        h = self.rc.heads; hd = d // h
        q = self.q(st).view(R, 1, self.NS, h, hd).expand(R, B, self.NS, h, hd).reshape(R * B, self.NS, h, hd).transpose(1, 2)
        k = self.k(ctx).view(R * B, N, h, hd).transpose(1, 2)
        v = self.v(ctx).view(R * B, N, h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(R, B, self.NS, d)
        x = st.unsqueeze(1) + self.o(a)
        x = x + self.f2(F.gelu(self.f1(self.norm(x))))
        return self.head_norm(x)                                                               # (R, B, NS, d)

    def logits(self, b: dict, T: int | None = None):
        """State-conditioned: (R, B, NS, 4) for every product state. Index-conditioned: (T, R, 1, 4)."""
        if self.rc.kind == "index":
            T = self.rc.T if T is None else T
            idx = torch.clamp(torch.arange(T, device=self.table.device), max=self.rc.T - 1)   # repeat the last trained logit
            return self.table[:, idx].permute(1, 0, 2).unsqueeze(2)
        return self.head(self._trunk(b)).float()

    def features(self, b: dict):
        """The head's input (R, B, NS, d) after the LayerNorm."""
        return self._trunk(b)


@torch.no_grad()
def plant(m: StackedRouter, beta: float = 30.0, kappa: float = 15.0, gain: float = 3.0, hop_bias: float = 4.0) -> StackedRouter:
    """The exact rule (DESIGN.md §6), linear in the LayerNormed head input. Head 0 matches the state's register
    against the pair keys (score beta on the match, a BOS sink at kappa for symbols without a pair) and reads the
    matched pair's flag coordinate back into the stream along the flag direction. The head then tests
      HALT: the register's answer (or dead) directions;
      SET:  flag - set_here            (positive only at a flagged pair not yet set here);
      PREV: pending - set_here (deferral 1) or pending (deferral 0), minus half the flag (so SET wins where both fire);
      HOP:  a constant bias, when nothing else fires.
    All frozen directions are orthogonal to the ones direction, so the LayerNorm only rescales them (by sqrt(d)/|x|).
    Head weights have norm ~ gain .. 2 gain and the HOP bias hop_bias, a few units from the init (L24); the committed
    route is exact at the defaults, the marginal is sharp at gain 6 / bias 8."""
    R, d, dc = m.R, m.rc.d, m.dc
    hd = d // m.rc.heads
    n = dc.n_sym
    assert n <= hd and m.rc.kind == "state" and m.rc.state_input == "full"
    for lin in (m.ctx_f1, m.ctx_f2, m.f1, m.f2, m.o, m.q, m.k, m.v):
        lin.weight.zero_(); lin.bias.zero_()
    m.pos.zero_(); m.marker.zero_()
    scale = beta * math.sqrt(hd)                                 # undo SDPA's 1/sqrt(hd)
    Q = torch.zeros(d, d); K = torch.zeros(d, d); V = torch.zeros(d, d); O = torch.zeros(d, d)
    Q[:, :n] = m.sym.reg_slot.T * scale                           # state register -> one-hot coordinates of head 0
    K[:, :n] = m.sym.key_slot.T                                   # pair key -> the same coordinates
    flag, cnt, seth = m.sym.free[m.FLAG], m.sym.free[m.CNT], m.sym.free[m.SETH]
    V[:, 0] = flag                                                # coordinate 0 of head 0 = the pair's flag bit
    O[0, :] = flag                                                # back into the stream along the flag direction
    for lin, W in ((m.q, Q), (m.k, K), (m.v, V), (m.o, O)):
        lin.weight.copy_(W.unsqueeze(0).expand(R, d, d))
    # the BOS sink: K^T pos[0] = (kappa / beta) * ones(n) gives every state a score kappa with BOS
    m.pos[:, 0] = (m.sym.key_slot.T @ torch.full((n,), kappa / beta)).unsqueeze(0).expand(R, d)
    halt_dir = m.sym.reg_slot[dc.n_keys:].sum(0); halt_dir = halt_dir / halt_dir.norm()
    prev_dir = cnt - (seth if dc.deferral == 1 else 0.0 * seth)
    W = torch.stack([torch.zeros(d), prev_dir * gain - flag * gain * 0.5, (flag - seth) * gain * 2, halt_dir * gain * 2], 1)
    m.head.weight.copy_(W.unsqueeze(0).expand(R, d, 4))
    m.head.bias.zero_(); m.head.bias[:, :, HOP] = hop_bias
    return m
