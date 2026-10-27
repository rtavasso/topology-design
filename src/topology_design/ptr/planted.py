"""The planted hop block (experiments/ptr/DESIGN.md §5, gate G1): weights set by hand so that one application of the
looped block performs exactly one pointer hop on query tokens and leaves every other token unchanged.

Stream subspaces (frozen, orthonormal, all orthogonal to the ones direction): key slot K, value slot V, register R,
scratch S, and a few free directions for the BOS sink, the query marker and the answer-token marker.

Attention (head 0): a query token's register, mapped into the key basis, matches the unique pair token with that key
(score beta); every token also scores kappa on BOS (value 0), which is where tokens with no match go. The matched
pair's value slot is copied into the query's scratch subspace. Head 1 is zero.
MLP: a gated linear map, on query tokens only (gated by the marker coordinate): clear the key part of the register,
write scratch into the register, clear scratch. The answer part of the register is kept, so a query that has reached
its answer stays there on further loops.
The stream linear map is the identity; LayerNorms are at their default (gamma 1, beta 0).
"""

from __future__ import annotations

import math

import torch

from .model import PtrModel


@torch.no_grad()
def plant(m: PtrModel, beta: float = 30.0, kappa: float = 15.0, gate: float = 20.0) -> PtrModel:
    assert m.mc.kind == "looped" and m.sym.exact, "planting needs the looped model with disjoint subspaces"
    sym, blk, d = m.sym, m.block, m.mc.d
    n_sym, n_keys = m.dc.n_sym, m.dc.n_keys
    h = m.mc.heads
    hd = d // h
    assert hd >= n_sym + 1
    K, V, R, S, Fr = sym.key_slot, sym.val_slot, sym.reg_slot, sym.scratch, sym.free
    bos, marker, ans_tok = Fr[0], Fr[1], Fr[2]
    m.bos.copy_(bos); m.marker.copy_(marker); m.ans_tok.copy_(ans_tok); m.pos.zero_()
    for p in list(blk.parameters()):
        p.zero_()
    for ln in (blk.n1, blk.n2, m.out_norm):
        ln.weight.fill_(1.0); ln.bias.zero_()
    sd = math.sqrt(d)
    # LayerNorm scales a token by sqrt(d)/||x||; norms are sqrt(2) for pairs (key+value) and for queries at attention
    # time (marker+register), 1 for BOS. The scales below undo that where magnitudes matter.
    Wq, Wk, Wv = (torch.zeros(d, d) for _ in range(3))
    bq = torch.zeros(3 * d)
    # head 0, coordinates 0..n_sym-1: key-basis coordinates
    # q: register content expressed in the key basis: coordinate i = (R_i . x); so row i of Wq = R_i (scaled)
    Wq[:n_sym] = (beta / sd) * R
    # k: key-slot content: coordinate i = (K_i . x) (scaled so that match gives beta * sqrt(2)*sqrt(2)/... = beta)
    Wk[:n_sym] = (1.0 / sd) * K
    # scores = (sum_i q_i k_i) / sqrt(hd) with q_i = beta/sd * (sqrt(d)/|x_q|) [reg_i], k_i = 1/sd * (sqrt(d)/|x_k|) [key_i]
    # = beta * [match] / (|x_q| |x_k| sqrt(hd)) = beta / (2 sqrt(hd)) for norms sqrt(2) each; so scale by 2 sqrt(hd)
    Wq[:n_sym] *= 2.0 * math.sqrt(hd)
    # BOS sink: coordinate n_sym: k = (bos . x~) = sqrt(d) on BOS, q = constant bias; score = kappa after the 1/sqrt(hd)
    Wk[n_sym] = bos
    bq[n_sym] = kappa * math.sqrt(hd) / sd
    # v: value-slot coordinates, rescaled so the copied scratch vector is a unit vector: (V_i . x~) = sqrt(d)/sqrt(2) [val_i]
    Wv[:n_sym] = (math.sqrt(2) / sd) * V
    blk.qkv.weight.copy_(torch.cat([Wq, Wk, Wv], 0))
    blk.qkv.bias.copy_(bq)
    # output projection: head-0 coordinate i -> scratch basis vector S_i
    Wo = torch.zeros(d, d)
    Wo[:, :n_sym] = S.T
    blk.o.weight.copy_(Wo)
    # MLP: gated linear map on query tokens. Input x~ = x * sqrt(d)/|x|, with |x| = sqrt(3) for a chasing query
    # (marker + register + scratch). Gate coordinate g = marker . x~ = sqrt(d)/sqrt(3) for queries, 0 otherwise.
    g_on = sd / math.sqrt(3)
    theta = g_on / 2
    c = gate / theta
    in_dirs = torch.cat([R[:n_keys], S], 0)                            # (n_keys + n_sym, d): register keys, scratch
    out_dirs = torch.cat([-R[:n_keys], R - S], 0)                       # clear key part; scratch -> register, clear scratch
    n_pairs = in_dirs.shape[0]
    assert 2 * n_pairs <= blk.f1.out_features
    W1 = torch.zeros(blk.f1.out_features, d)
    b1 = torch.zeros(blk.f1.out_features)
    W2 = torch.zeros(d, blk.f1.out_features)
    comp = math.sqrt(3) / sd                                            # undo LayerNorm's scale for a chasing query
    for i in range(n_pairs):
        W1[2 * i] = c * marker + in_dirs[i]
        W1[2 * i + 1] = c * marker - in_dirs[i]
        b1[2 * i] = b1[2 * i + 1] = -c * theta
        W2[:, 2 * i] = 0.5 * comp * out_dirs[i]
        W2[:, 2 * i + 1] = -0.5 * comp * out_dirs[i]
    blk.f1.weight.copy_(W1); blk.f1.bias.copy_(b1)
    blk.f2.weight.copy_(W2); blk.f2.bias.zero_()
    blk.lin.weight.copy_(torch.eye(d)); blk.lin.bias.zero_()
    return m
