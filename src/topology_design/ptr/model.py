"""Static and looped transformers with per-token halting for pointer chasing (experiments/ptr/DESIGN.md §3).

Tokens: P pair tokens (embedding = key-slot(key) + value-slot(value)), then for each query a query token
(marker + start-slot(start)) followed by the answer token. The loss is the cross-entropy over answer symbols at the
query positions, read out from the final state through the frozen answer embeddings.

Symbol embeddings are frozen and orthogonal: the key slot, the value slot, the register (readout) basis and a scratch
basis live in four disjoint subspaces of the stream, so content lives in the context, not in the weights, and the
planted hop block (planted.py) is exact. A BOS token (an attention sink), the query marker, the answer-token marker
and the positions are learned.

Halting modes for the looped model:
  fixed    every token runs T loops;
  oracle   a token runs its ground-truth number of steps (queries: k hops; pairs: 1) and is then frozen;
  learned  PonderNet: a halting head on the state (optionally plus a step-index embedding); trained with the
           expected loss over halting times plus a KL to a geometric prior; at test, halt at cumulative p >= 0.5
           and freeze;
  value    a head regresses the remaining hops (privileged supervision); at test, halt when the estimate < 0.5.
Frozen tokens keep their state; later tokens attend to the frozen state. Compute is counted as loops per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Episodes, PtrConfig


@dataclass
class ModelConfig:
    kind: str = "looped"          # "static" | "looped"
    d: int = 264                  # 4 * n_sym + 8 with n_sym = 64: four disjoint symbol subspaces (planted block exact)
    heads: int = 2
    mlp: int = 4                  # MLP width multiplier
    layers: int = 4               # static only
    T: int = 10                   # looped: loops in training (and the default at test); >= kmax + 2
    halting: str = "fixed"        # looped: fixed | oracle | learned | value
    halt_input: str = "state"     # learned halting head input: state | state+index | index (index-only = constant-depth null)
    halt_bias_init: float = -2.0  # initial halting logit (lambda ~ 0.12), so the halting mass starts spread over ~8 steps
    injection: bool = True        # add the token embedding at every loop
    pair_pair_mask: bool = True   # context tokens are read-only: pairs attend to BOS and themselves only (no doubling)
    ctx_single_step: bool = True  # looped: non-query tokens run exactly one step (coherent train/test semantics)
    lam_p: float = 0.2            # PonderNet geometric prior
    beta: float = 0.01            # PonderNet KL weight
    emb_seed: int = 1234

    def dict(self):
        return asdict(self)


class FrozenSymbols(nn.Module):
    """Orthonormal symbol bases in disjoint subspaces: key slot, value slot, register (readout) and scratch.
    With d >= 4 * n_sym all four are disjoint (looped arms); with 3 * n_sym <= d < 4 * n_sym the scratch basis is
    omitted; below that the bases are random unit vectors (static arms at small d), which trained models handle."""

    def __init__(self, n_sym: int, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # every basis vector is orthogonal to the all-ones direction, so LayerNorm's mean subtraction is a no-op on
        # any combination of symbols (needed for the planted block to be exact)
        ones = torch.ones(d, 1) / math.sqrt(d)
        M = torch.randn(d, d, generator=g)
        M = M - ones @ (ones.T @ M)
        Q, _ = torch.linalg.qr(torch.cat([ones, M[:, :-1]], 1))      # column 0 is the ones direction; drop it
        Q = Q[:, 1:]                                                    # (d, d-1), all orthogonal to ones
        if d - 1 >= 3 * n_sym:
            cols = [Q[:, i * n_sym:(i + 1) * n_sym].T.contiguous() for i in range(min(4, (d - 1) // n_sym))]
            used = n_sym * len(cols)
            while len(cols) < 4:
                cols.append(cols[-1])
        else:
            cols = [F.normalize(Q[:, torch.randint(0, d - 1, (n_sym,), generator=g)].T, dim=-1) for _ in range(4)]
            used = 0
        self.exact = d - 1 >= 4 * n_sym
        for name, c in zip(("key_slot", "val_slot", "reg_slot", "scratch"), cols):
            self.register_buffer(name, c)
        self.register_buffer("free", Q[:, used:].T.contiguous())      # leftover directions (markers, BOS)


class Block(nn.Module):
    """Pre-norm causal attention + MLP, then a linear map on the stream (identity at init) so a block can overwrite
    a subspace; needed for looped execution (register replacement)."""

    def __init__(self, d, heads, mlp):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)
        self.lin = nn.Linear(d, d)
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.h = heads

    def forward(self, x, mask):
        """x (B, N, d); mask (N, N) or (B, N, N) bool, True = may attend."""
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, -1)
        sh = lambda t: t.view(B, N, self.h, d // self.h).transpose(1, 2)
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=m.unsqueeze(1))
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        x = x + self.f2(F.gelu(self.f1(self.n2(x))))
        return self.lin(x)


class PtrModel(nn.Module):
    def __init__(self, mc: ModelConfig, dc: PtrConfig):
        super().__init__()
        self.mc, self.dc = mc, dc
        d = mc.d
        self.sym = FrozenSymbols(dc.n_sym, d, mc.emb_seed)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)          # attention sink at position 0
        self.marker = nn.Parameter(torch.randn(d) * 0.02)       # query marker
        self.ans_tok = nn.Parameter(torch.randn(d) * 0.02)      # added to answer tokens (they are inputs only)
        self.pos = nn.Parameter(torch.randn(1 + dc.P + 2 * dc.Qn + 64, d) * 0.02)
        if mc.kind == "static":
            self.blocks = nn.ModuleList(Block(d, mc.heads, mc.mlp) for _ in range(mc.layers))
        else:
            self.block = Block(d, mc.heads, mc.mlp)
            if mc.halting == "learned":
                self.halt_norm = nn.LayerNorm(d)
                if mc.halt_input == "index":
                    self.halt_bias = nn.Parameter(torch.zeros(64) + mc.halt_bias_init)   # one logit per step; the last repeats beyond
                else:
                    self.halt = nn.Linear(d + (16 if mc.halt_input == "state+index" else 0), 1)
                    nn.init.normal_(self.halt.weight, std=0.02)
                    nn.init.constant_(self.halt.bias, mc.halt_bias_init)
                    if mc.halt_input == "state+index":
                        self.step_emb = nn.Parameter(torch.randn(64, 16) * 0.1)
            if mc.halting == "value":
                self.val_norm = nn.LayerNorm(d)
                self.val = nn.Linear(d, 1)
        self.out_norm = nn.LayerNorm(d)

    # ---------- tokens ----------
    def embed(self, ep: Episodes, dev):
        """Returns e0 (B, N, d), the query positions (Qn,), targets (B, Qn) as answer indices, and the masks."""
        B, P, Qn = ep.B, self.dc.P, ep.q_start.shape[1]
        pk = torch.as_tensor(ep.pair_key, device=dev)
        pv = torch.as_tensor(ep.pair_val, device=dev)
        qs = torch.as_tensor(ep.q_start, device=dev)
        qa = torch.as_tensor(ep.q_ans, device=dev)
        pairs = self.sym.key_slot[pk] + self.sym.val_slot[pv]                       # (B, P, d)
        queries = self.marker + self.sym.reg_slot[qs]                                # register starts at the start key
        answers = self.ans_tok + self.sym.reg_slot[qa]
        qa_tokens = torch.stack([queries, answers], 2).reshape(B, 2 * Qn, -1)       # q1 a1 q2 a2 ...
        e0 = torch.cat([self.bos.expand(B, 1, -1), pairs, qa_tokens], 1)
        N = e0.shape[1]
        e0 = e0 + self.pos[:N]
        qpos = 1 + P + 2 * torch.arange(Qn, device=dev)
        targets = qa - self.dc.n_keys
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=dev))
        if self.mc.pair_pair_mask:                      # pair tokens see only BOS and themselves
            pp = torch.zeros(N, N, dtype=torch.bool, device=dev)
            pp[1:P + 1, 1:P + 1] = True
            pp[torch.arange(1, P + 1), torch.arange(1, P + 1)] = False
            mask = mask & ~pp
        return e0, qpos, targets, mask

    def readout(self, h):
        """Logits over answer symbols from the register basis (frozen)."""
        z = self.out_norm(h)
        return z @ self.sym.reg_slot[self.dc.n_keys:self.dc.n_keys + self.dc.n_ans].T

    # ---------- forward ----------
    def forward(self, ep: Episodes, dev, *, T: int | None = None, steps: torch.Tensor | None = None,
                mode: str = "train"):
        """Returns dict with logits (B, Qn, n_ans) [or per step], per-token steps used, halting weights, etc.
        steps: (B, N) ground-truth steps for oracle halting (built by caller via oracle_steps)."""
        mc = self.mc
        e0, qpos, targets, mask = self.embed(ep, dev)
        B, N, d = e0.shape
        out = {"targets": targets, "qpos": qpos}
        if mc.kind == "static":
            h = e0
            for blk in self.blocks:
                h = blk(h, mask)
            out["logits"] = self.readout(h[:, qpos])
            out["steps"] = torch.full((B, N), float(mc.layers), device=dev)
            return out
        T = mc.T if T is None else T
        h = e0
        active = torch.ones(B, N, dtype=torch.bool, device=dev)
        is_query = torch.zeros(N, dtype=torch.bool, device=dev)
        is_query[qpos] = True
        used = torch.zeros(B, N, device=dev)
        per_step_logits, lams, vals, hs = [], [], [], []
        remaining = torch.ones(B, N, device=dev)      # PonderNet: probability of not having halted yet
        p_halt = []
        for t in range(1, T + 1):
            x = h + e0 if mc.injection else h
            h_new = self.block(x, mask)
            if mode == "ponder_train" and mc.halting == "learned":
                # queries run all T steps (their per-step readouts carry the expected loss); context tokens follow
                # the same rule as at test (single step, or their own halting decision), so train and test agree
                upd = active | is_query
                h = torch.where(upd.unsqueeze(-1), h_new, h)
                used += upd.float()
            else:
                h = torch.where(active.unsqueeze(-1), h_new, h)
                used += active.float()
            hs.append(h)
            if mc.ctx_single_step:
                active = active & is_query
            if mc.halting == "learned":
                if mc.halt_input == "index":
                    lam = torch.sigmoid(self.halt_bias[min(t - 1, 63)]).expand(B, N)
                else:
                    z = self.halt_norm(h)
                    if mc.halt_input == "state+index":
                        z = torch.cat([z, self.step_emb[min(t - 1, 63)].expand(B, N, -1)], -1)
                    lam = torch.sigmoid(self.halt(z)).squeeze(-1)   # (B, N)
                if t == T:
                    lam = torch.ones_like(lam)
                p_halt.append(remaining * lam)
                remaining = remaining * (1 - lam)
                lams.append(lam)
                per_step_logits.append(self.readout(h[:, qpos]))
                cum = torch.stack(p_halt, 0).sum(0)
                active = active & (cum.detach() < 0.5)
            elif mc.halting == "value":
                v = self.val(self.val_norm(h)).squeeze(-1)           # remaining hops estimate
                vals.append(v)
                if mode != "train":
                    active = active & (v >= 0.5)
            elif mc.halting == "oracle":
                assert steps is not None
                active = active & (used < steps)
            # fixed: always active
        out["h"] = h
        out["hs"] = hs
        out["steps"] = used
        if mc.halting == "learned":
            out["p_halt"] = torch.stack(p_halt, 0)                    # (T, B, N)
            out["lams"] = torch.stack(lams, 0)
            out["per_step_logits"] = torch.stack(per_step_logits, 0)  # (T, B, Qn, n_ans)
        if mc.halting == "value":
            out["vals"] = torch.stack(vals, 0)
        out["logits"] = self.readout(h[:, qpos])
        return out

    # ---------- losses ----------
    def loss(self, out, ep: Episodes, dev, remaining_hops: torch.Tensor | None = None):
        """Training loss. remaining_hops (T, B, N) for the value arm: ground-truth hops left after each step."""
        mc = self.mc
        tgt = out["targets"]
        if mc.kind == "looped" and mc.halting == "learned" and "per_step_logits" in out:
            T = out["per_step_logits"].shape[0]
            ce = torch.stack([F.cross_entropy(out["per_step_logits"][t].reshape(-1, out["per_step_logits"].shape[-1]),
                                              tgt.reshape(-1), reduction="none").view(tgt.shape) for t in range(T)], 0)
            pq = out["p_halt"][:, :, out["qpos"]]                     # (T, B, Qn)
            expected = (pq * ce).sum(0).mean()
            # KL(p || geometric(lam_p)) over the tokens whose depth is learned (queries; all tokens if not single-step)
            p = out["p_halt"].clamp_min(1e-9)                          # (T, B, N)
            if mc.ctx_single_step:
                p = p[:, :, out["qpos"]]
            geo = torch.tensor([mc.lam_p * (1 - mc.lam_p) ** t for t in range(T)], device=dev)
            geo = geo / geo.sum()
            kl = (p * (p.log() - geo.log().view(T, 1, 1))).sum(0).mean()
            return expected + mc.beta * kl, {"ce": expected.item(), "kl": kl.item()}
        ce = F.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]), tgt.reshape(-1))
        extra = {"ce": ce.item()}
        if mc.kind == "looped" and mc.halting == "value":
            assert remaining_hops is not None
            vl = F.smooth_l1_loss(out["vals"], remaining_hops)
            extra["value"] = vl.item()
            return ce + vl, extra
        return ce, extra


# ---------- ground-truth helpers ----------
def oracle_steps(ep: Episodes, cfg: PtrConfig, dev) -> torch.Tensor:
    """(B, N) steps each token runs under oracle halting: BOS, pairs and answer tokens 1; queries k."""
    B, Qn = ep.B, ep.q_start.shape[1]
    s = torch.ones(B, 1 + cfg.P + 2 * Qn, device=dev)
    s[:, 1 + cfg.P::2] = torch.as_tensor(ep.q_k, device=dev, dtype=torch.float)
    return s


def remaining_after(ep: Episodes, cfg: PtrConfig, T: int, dev) -> torch.Tensor:
    """(T, B, N) ground-truth hops remaining after each loop, for the value arm: queries max(k - t, 0); others 0."""
    B, Qn = ep.B, ep.q_start.shape[1]
    r = torch.zeros(T, B, 1 + cfg.P + 2 * Qn, device=dev)
    k = torch.as_tensor(ep.q_k, device=dev, dtype=torch.float)
    for t in range(1, T + 1):
        r[t - 1, :, 1 + cfg.P::2] = (k - t).clamp_min(0)
    return r
