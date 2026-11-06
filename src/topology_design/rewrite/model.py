"""The rung-4 router and its identical-architecture control (DESIGN §1 "The router"; review 1 B4).

Input: [axiom presence × 4] [goal s] [SEP] [current term t] (and, for RT-examples, two example proofs after the term).
Term tokens are the prefix-order tokens ("+" for an internal node, an atom for a leaf).

Structural router: per-token tree features (depth, left/right path, subtree leaves) added to the token embedding, a
relative attention bias indexed by the pair's (segment pair, path relation) — equal path, ancestor by d, descendant by d,
disjoint — and a pointer head: eight rule logits from every term token's final representation, HALT from SEP.
Absolute control: learned absolute positions, no tree features, no relative bias, and a categorical head over
(rule, flat prefix position 0..12) + HALT read from the SEP token. Both produce logits over the same 105 action slots
(index = prefix position × 8 + rule; HALT last); slots the pointer head cannot address (positions past the term's node
count) are −inf for it and ordinary classes for the control. The two are parameter-matched within 1 % by the FFN width.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rules import AXIOMS, HALT, RULES
from .term import nodes, parse

MAX_NODES = 13                       # a term of 7 leaves has 13 nodes
N_RULES = len(RULES)
N_ACTIONS = MAX_NODES * N_RULES + 1  # 105
HALT_INDEX = N_ACTIONS - 1
MAX_DEPTH = 8
MAX_LEN = 128
RULE_INDEX = {r: i for i, r in enumerate(RULES)}

# vocabulary
_TOK = ["PAD", "SEP", "?", "EX", "TO", "BY", "+", "a", "b", "c", "d", "0"]
for ax in AXIOMS:
    _TOK += [f"{ax}+", f"{ax}-"]
_TOK += list(RULES)
VOCAB = {t: i for i, t in enumerate(_TOK)}
SEG = {"pres": 0, "goal": 1, "sep": 2, "term": 3, "ex": 4}
N_SEG = 5
N_PATHREL = 2 + 2 * MAX_DEPTH + 1     # n/a, equal, ancestor by 1..8, descendant by 1..8, disjoint
N_REL = N_SEG * N_SEG * N_PATHREL


def action_slot(a, path_index: dict) -> int:
    if a == HALT:
        return HALT_INDEX
    rule, path = a
    return path_index[path] * N_RULES + RULE_INDEX[rule]


def slot_action(slot: int, paths: list):
    """slot -> (rule, path) or HALT; paths = the term's node paths in prefix order. A slot naming a position past the
    term's node count is returned as (rule, None): an invalid action under the untyped semantics."""
    if slot == HALT_INDEX:
        return HALT
    pos, r = divmod(slot, N_RULES)
    return (RULES[r], paths[pos] if pos < len(paths) else None)


# ---------------- encoding ----------------

def _tree_tokens(t, seg: int):
    """Per node in prefix order: (token id, seg, depth, path, subtree leaves)."""
    out = []
    for path, sub, depth in nodes(t):
        tok = VOCAB["+"] if isinstance(sub, tuple) else VOCAB[sub]
        leaves = _leaves(sub)
        out.append((tok, seg, depth, path, leaves))
    return out


def _leaves(u) -> int:
    return 1 if not isinstance(u, tuple) else _leaves(u[0]) + _leaves(u[1])


def encode(world: str | None, s_str: str, t_str: str, examples: list | None = None) -> dict:
    """One input. world = canonical name ("ACI") or None for unstated axioms (RT-examples). examples = [(t, s, [rules])].
    Returns token ids, segments, depths, paths (strings), subtree sizes, the term's node paths in prefix order."""
    toks = []
    if world is None:
        for ax in AXIOMS:
            toks.append((VOCAB["?"], SEG["pres"], 0, None, 0))
    else:
        for ax in AXIOMS:
            toks.append((VOCAB[f"{ax}{'+' if ax in world else '-'}"], SEG["pres"], 0, None, 0))
    toks += _tree_tokens(parse(s_str), SEG["goal"])
    toks.append((VOCAB["SEP"], SEG["sep"], 0, None, 0))
    term_start = len(toks)
    term_nodes = _tree_tokens(parse(t_str), SEG["term"])
    toks += term_nodes
    term_paths = [p for _, _, _, p, _ in term_nodes]
    for (et, es, rules) in examples or []:
        toks.append((VOCAB["EX"], SEG["ex"], 0, None, 0))
        toks += _tree_tokens(parse(et), SEG["ex"])
        toks.append((VOCAB["TO"], SEG["ex"], 0, None, 0))
        toks += _tree_tokens(parse(es), SEG["ex"])
        toks.append((VOCAB["BY"], SEG["ex"], 0, None, 0))
        for r in rules:
            toks.append((VOCAB[r], SEG["ex"], 0, None, 0))
    assert len(toks) <= MAX_LEN
    return {"tok": [x[0] for x in toks], "seg": [x[1] for x in toks], "depth": [x[2] for x in toks],
            "path": [x[3] for x in toks], "size": [x[4] for x in toks], "term_start": term_start, "term_paths": term_paths}


def _pathrel(pi, pj) -> int:
    if pi is None or pj is None:
        return 0
    if pi == pj:
        return 1
    if pj.startswith(pi):
        return 2 + min(len(pj) - len(pi), MAX_DEPTH) - 1
    if pi.startswith(pj):
        return 2 + MAX_DEPTH + min(len(pi) - len(pj), MAX_DEPTH) - 1
    return N_PATHREL - 1


def _relation_matrix(segs, paths):
    """Vectorised relation ids for one sequence: (segment pair) × (path relation), path relation 0 for non-tree tokens."""
    n = len(segs)
    has = np.array([p is not None for p in paths])
    dep = np.array([len(p) if p is not None else 0 for p in paths], dtype=np.int64)
    bits = np.array([int("1" + p.replace("L", "0").replace("R", "1"), 2) if p is not None else 0 for p in paths], dtype=np.int64)
    di, dj = dep[:, None], dep[None, :]
    bi, bj = bits[:, None], bits[None, :]
    both = has[:, None] & has[None, :]
    equal = both & (bi == bj) & (di == dj)
    # i ancestor of j: dep_i < dep_j and bits_j >> (dep_j - dep_i) == bits_i
    sh = np.clip(dj - di, 0, 62)
    anc = both & (di < dj) & ((bj >> sh) == bi)
    sh2 = np.clip(di - dj, 0, 62)
    desc = both & (dj < di) & ((bi >> sh2) == bj)
    pr = np.where(both, N_PATHREL - 1, 0)
    pr = np.where(equal, 1, pr)
    pr = np.where(anc, 2 + np.minimum(dj - di, MAX_DEPTH) - 1, pr)
    pr = np.where(desc, 2 + MAX_DEPTH + np.minimum(di - dj, MAX_DEPTH) - 1, pr)
    sg = np.array(segs, dtype=np.int64)
    return (sg[:, None] * N_SEG + sg[None, :]) * N_PATHREL + pr


def collate(encs: list, device="cpu") -> dict:
    """Pad a list of encodings. Returns tensors: tok, seg, depth, size, pathcode (B, L, MAX_DEPTH), rel (B, L, L), mask (B, L),
    term_pos (B, MAX_NODES) = sequence index of each term node or -1, n_nodes (B)."""
    B = len(encs)
    L = max(len(e["tok"]) for e in encs)
    tok = np.zeros((B, L), dtype=np.int64)
    seg = np.zeros((B, L), dtype=np.int64)
    depth = np.zeros((B, L), dtype=np.int64)
    size = np.zeros((B, L), dtype=np.int64)
    pathcode = np.zeros((B, L, MAX_DEPTH), dtype=np.int64)
    rel = np.zeros((B, L, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    term_pos = np.full((B, MAX_NODES), -1, dtype=np.int64)
    n_nodes = np.zeros(B, dtype=np.int64)
    for b, e in enumerate(encs):
        n = len(e["tok"])
        tok[b, :n] = e["tok"]; seg[b, :n] = e["seg"]
        depth[b, :n] = np.minimum(e["depth"], MAX_DEPTH)
        size[b, :n] = np.minimum(e["size"], MAX_DEPTH)
        mask[b, :n] = True
        if "pathcode" not in e:
            pc = np.zeros((n, MAX_DEPTH), dtype=np.int64)
            for i, p in enumerate(e["path"]):
                if p:
                    for lvl, ch in enumerate(p[:MAX_DEPTH]):
                        pc[i, lvl] = 1 if ch == "L" else 2
            e["pathcode"] = pc
            e["rel"] = _relation_matrix(e["seg"], e["path"])
        pathcode[b, :n] = e["pathcode"]
        rel[b, :n, :n] = e["rel"]
        k = len(e["term_paths"])
        n_nodes[b] = k
        term_pos[b, :k] = np.arange(e["term_start"], e["term_start"] + k)
    out = {"tok": tok, "seg": seg, "depth": depth, "size": size, "pathcode": pathcode, "rel": rel, "mask": mask,
           "term_pos": term_pos, "n_nodes": n_nodes}
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


# ---------------- the encoder ----------------

@dataclass
class RouterConfig:
    structural: bool = True       # tree features + relative bias + pointer head; False = absolute positions + categorical head
    d: int = 128
    layers: int = 3
    heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0

    def dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class Router(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.tok = nn.Embedding(len(VOCAB), d)
        self.seg = nn.Embedding(N_SEG, d)
        if cfg.structural:
            self.depth = nn.Embedding(MAX_DEPTH + 1, d)
            self.size = nn.Embedding(MAX_DEPTH + 1, d)
            self.path = nn.Embedding(MAX_DEPTH * 3, d)        # (level, {none, L, R})
            self.relbias = nn.Embedding(N_REL, cfg.heads)
            self.rule_head = nn.Linear(d, N_RULES)
            self.halt_head = nn.Linear(d, 1)
        else:
            self.pos = nn.Embedding(MAX_LEN, d)
            self.cat_head = nn.Linear(d, N_ACTIONS)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.tok.weight, std=0.02)
        if cfg.structural:
            nn.init.zeros_(self.relbias.weight)

    def encode(self, b: dict):
        x = self.tok(b["tok"]) + self.seg(b["seg"])
        B, L = b["tok"].shape
        if self.cfg.structural:
            x = x + self.depth(b["depth"]) + self.size(b["size"])
            lvl = torch.arange(MAX_DEPTH, device=x.device)
            x = x + self.path(lvl * 3 + b["pathcode"]).sum(2)
            bias = self.relbias(b["rel"]).permute(0, 3, 1, 2)                     # (B, H, L, L)
        else:
            x = x + self.pos(torch.arange(L, device=x.device))[None]
            bias = torch.zeros(B, self.cfg.heads, L, L, device=x.device)
        pad = (~b["mask"])[:, None, None, :].float() * -1e9
        bias = bias + pad
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln(x)

    def forward(self, b: dict) -> torch.Tensor:
        """Action logits (B, N_ACTIONS): slot = prefix position × 8 + rule, HALT last. Unaddressable slots are −inf for
        the pointer head."""
        h = self.encode(b)
        B = h.shape[0]
        if self.cfg.structural:
            idx = b["term_pos"].clamp(min=0)                                    # (B, MAX_NODES)
            ht = torch.gather(h, 1, idx[:, :, None].expand(-1, -1, h.shape[-1]))
            rl = self.rule_head(ht)                                             # (B, MAX_NODES, 8)
            valid = (b["term_pos"] >= 0)[:, :, None].expand_as(rl)
            rl = rl.masked_fill(~valid, float("-inf")).reshape(B, MAX_NODES * N_RULES)
            # SEP is the token right before the term
            sep_idx = (b["term_pos"][:, 0] - 1).clamp(min=0)
            hs = h[torch.arange(B, device=h.device), sep_idx]
            halt = self.halt_head(hs)
            return torch.cat([rl, halt], 1)
        sep_idx = (b["term_pos"][:, 0] - 1).clamp(min=0)
        hs = h[torch.arange(B, device=h.device), sep_idx]
        return self.cat_head(hs)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_configs(d=128, layers=3, heads=4, d_ff=512, tol=0.01):
    """The structural and absolute configurations parameter-matched within tol by the structural model's FFN width."""
    abs_cfg = RouterConfig(structural=False, d=d, layers=layers, heads=heads, d_ff=d_ff)
    n_abs = n_params(Router(abs_cfg))
    best = None
    for ff in range(d_ff - 64, d_ff + 129, 2):
        cfg = RouterConfig(structural=True, d=d, layers=layers, heads=heads, d_ff=ff)
        n = n_params(Router(cfg))
        gap = abs(n - n_abs) / n_abs
        if best is None or gap < best[0]:
            best = (gap, cfg, n)
    gap, cfg, n = best
    assert gap <= tol, (gap, n, n_abs)
    return cfg, abs_cfg, n, n_abs
