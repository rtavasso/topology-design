"""Episodes for rung 1 (experiments/route/DESIGN.md §2): chains over keys whose steps are one of two generic circuits,
HOP (follow the pair's value) or PREV (go to the key of the pair immediately before, written as the value ←).

Symbol ids: keys 0..n_keys-1, answers n_keys..n_keys+n_ans-1, ARROW = n_keys+n_ans (the ← marker, a value only),
DEAD = ARROW + 1 (⊥, the executor's dead state; also the "no previous pair" marker in the prev slot).

A chain is s_0 -> s_1 -> ... -> s_k with s_k an answer and a word w in {H, P}^k ending in H:
  H: pair(s_{i-1}).value = s_i;      P: pair(s_{i-1}).value = ARROW and pair(s_i) sits right before pair(s_{i-1}).
Every pair token carries (key, value, prev) with prev = the key of the pair before it in context order (DEAD for the
first). Everything is numpy; tensors are built by the model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

H, P = 0, 1          # action codes in words (HALT is 2 in the router's action space)


@dataclass
class RouteConfig:
    n_keys: int = 48
    n_ans: int = 16
    P: int = 48                 # pairs per episode; Qn * kmax <= P
    Qn: int = 6
    alpha: float = 1.5          # Zipf exponent of queried chain lengths
    kmax: int = 8
    p_P: float = 0.25           # probability that a non-final step is P (training family: no two consecutive P's)
    max_prun: int = 1           # longest P-run allowed in training words
    tail_factor: float = 1.0    # pooling manipulation as in rung 0 (unused here; kept for config compatibility)
    k_deep: int = 6
    seed: int = 0

    def dict(self):
        return asdict(self)

    @property
    def ARROW(self):
        return self.n_keys + self.n_ans

    @property
    def DEAD(self):
        return self.n_keys + self.n_ans + 1

    @property
    def n_sym(self):
        return self.n_keys + self.n_ans + 2


def zipf_pmf(alpha, kmax, tail_factor=1.0, k_deep=6):
    k = np.arange(1, kmax + 1, dtype=float)
    p = k ** (-alpha)
    p[k >= k_deep] *= tail_factor
    return p / p.sum()


@dataclass
class Episodes:
    pair_key: np.ndarray    # (B, P)
    pair_val: np.ndarray    # (B, P): key, answer or ARROW
    pair_prev: np.ndarray   # (B, P): key of the previous pair in context order, DEAD for the first
    q_start: np.ndarray     # (B, Qn)
    q_ans: np.ndarray       # (B, Qn)
    q_k: np.ndarray         # (B, Qn)
    q_word: np.ndarray      # (B, Qn, kmax_eval): H/P codes, -1 beyond k
    meta: dict = field(default_factory=dict)

    @property
    def B(self):
        return self.pair_key.shape[0]


# ---------- words ----------
def sample_word(rng, k, p_P, max_prun):
    """A word of length k ending in H, with P-runs of at most max_prun; P chosen with probability p_P where allowed."""
    w = np.zeros(k, dtype=np.int64)
    run = 0
    for i in range(k - 1):
        if run < max_prun and rng.random() < p_P:
            w[i] = P; run += 1
        else:
            w[i] = H; run = 0
    w[k - 1] = H
    return w


def word_class(w):
    """Descriptors used by the evaluation strata: longest P-run, number of P's, deepest P (1-based step), length."""
    w = np.asarray(w)
    k = len(w)
    runs, best, cur = [], 0, 0
    for a in w:
        cur = cur + 1 if a == P else 0
        best = max(best, cur)
    nP = int((w == P).sum())
    deepest = int(np.where(w == P)[0].max()) + 1 if nP else 0
    return {"k": k, "max_prun": best, "nP": nP, "deepest_P": deepest}


def forced_word(rng, k, kind, p_P=0.25):
    """Words for the evaluation strata. kind: 'train' (training statistics), 'pp' (contains a PP run; PPP allowed),
    'alt' (alternation H P H P ... ending in H), 'deepP' (a P at step >= 6, training statistics otherwise),
    'none' (all H)."""
    if kind == "train":
        return sample_word(rng, k, p_P, 1)
    if kind == "none":
        return np.zeros(k, dtype=np.int64)
    if kind == "alt":
        w = np.array([P if (k - 1 - i) % 2 == 1 else H for i in range(k)], dtype=np.int64)   # ends in H
        return w
    if kind == "pp":
        assert k >= 3
        w = sample_word(rng, k, p_P, 3)
        # ensure at least one PP: plant a run of length 2 or 3 at a random admissible place
        run = 2 if (k < 5 or rng.random() < 0.6) else 3
        start = int(rng.integers(0, k - run))        # steps start .. start+run-1 are P; the last step stays H
        w[start:start + run] = P
        w[k - 1] = H
        return w
    if kind == "deepP":
        assert k >= 7
        w = sample_word(rng, k, p_P, 1)
        pos = int(rng.integers(5, k - 1))             # 0-based step index >= 5 means step >= 6
        w[pos] = P
        if pos > 0:
            w[pos - 1] = H
        if pos + 1 < k - 1:
            w[pos + 1] = H
        return w
    raise ValueError(kind)


# ---------- episodes ----------
def _chain_lengths(rng, Qn, cfg, forced_k=None):
    pmf = zipf_pmf(cfg.alpha, cfg.kmax, cfg.tail_factor, cfg.k_deep)
    while True:
        ks = rng.choice(np.arange(1, cfg.kmax + 1), size=Qn, p=pmf)
        if forced_k is not None:
            ks[0] = forced_k
        if ks.sum() <= cfg.P:
            return ks


def generate(cfg: RouteConfig, B: int, rng: np.random.Generator, forced_k=None, forced_kind=None, Qn=None,
             kmax_eval=None) -> Episodes:
    """B episodes. forced_k / forced_kind: query 0 gets that length and word class (evaluation strata)."""
    Qn = cfg.Qn if Qn is None else Qn
    kmax_eval = max(cfg.kmax, forced_k or 0) if kmax_eval is None else kmax_eval
    Pn = cfg.P
    pair_key = np.zeros((B, Pn), np.int64); pair_val = np.zeros((B, Pn), np.int64); pair_prev = np.zeros((B, Pn), np.int64)
    q_start = np.zeros((B, Qn), np.int64); q_ans = np.zeros((B, Qn), np.int64); q_k = np.zeros((B, Qn), np.int64)
    q_word = np.full((B, Qn, kmax_eval), -1, np.int64)
    for b in range(B):
        keys = rng.permutation(cfg.n_keys)
        ks = _chain_lengths(rng, Qn, cfg, forced_k)
        chains = []          # list of (key list, word, answer)
        used = 0
        for j, k in enumerate(ks):
            w = forced_word(rng, k, forced_kind, cfg.p_P) if (j == 0 and forced_kind) else sample_word(rng, k, cfg.p_P, cfg.max_prun)
            chains.append((keys[used:used + k], w, cfg.n_keys + int(rng.integers(cfg.n_ans))))
            used += k
        # distractor chains over the remaining keys, from the training family
        pmf = zipf_pmf(cfg.alpha, cfg.kmax, cfg.tail_factor, cfg.k_deep)
        while used < cfg.n_keys:
            k = min(int(rng.choice(np.arange(1, cfg.kmax + 1), p=pmf)), cfg.n_keys - used)
            chains.append((keys[used:used + k], sample_word(rng, k, cfg.p_P, cfg.max_prun), cfg.n_keys + int(rng.integers(cfg.n_ans))))
            used += k
        # pairs and the adjacency blocks: a P-step from s_{i-1} to s_i puts pair(s_i) right before pair(s_{i-1})
        blocks = []
        val = {}
        for ck, w, ans in chains:
            k = len(ck)
            for i in range(k):
                nxt = ck[i + 1] if i + 1 < k else ans
                val[ck[i]] = nxt if w[i] == H else cfg.ARROW
            # split the chain into blocks at H-steps; within a block (a P-run) the order is reversed
            block = [ck[0]]
            for i in range(k - 1):
                if w[i] == P:
                    block.append(ck[i + 1])
                else:
                    blocks.append(block[::-1]); block = [ck[i + 1]]
            blocks.append(block[::-1])
        order = rng.permutation(len(blocks))
        seq = [kk for o in order for kk in blocks[o]]
        assert len(seq) == Pn
        pair_key[b] = seq
        pair_val[b] = [val[kk] for kk in seq]
        pair_prev[b] = [cfg.DEAD] + seq[:-1]
        for j, (ck, w, ans) in enumerate(chains[:Qn]):
            q_start[b, j], q_ans[b, j], q_k[b, j] = ck[0], ans, len(ck)
            q_word[b, j, :len(w)] = w
    return Episodes(pair_key, pair_val, pair_prev, q_start, q_ans, q_k, q_word, meta={"forced_kind": forced_kind})


# ---------- exact reference executor (numpy, one state) ----------
def step(ep: Episodes, b: int, s: int, a: int, cfg: RouteConfig) -> int:
    """exec_a(s) for a in {H, P} on episode b."""
    if s >= cfg.n_keys:
        return s if s < cfg.n_keys + cfg.n_ans else cfg.DEAD         # answers absorb; ARROW/DEAD are dead
    pos = int(np.where(ep.pair_key[b] == s)[0][0])
    if a == H:
        v = int(ep.pair_val[b, pos])
        return v if v != cfg.ARROW else cfg.DEAD
    return int(ep.pair_prev[b, pos])


def follow(ep: Episodes, b: int, j: int, cfg: RouteConfig):
    """States along query j's oracle route: s_0 .. s_k."""
    s = int(ep.q_start[b, j]); states = [s]
    for a in ep.q_word[b, j, :ep.q_k[b, j]]:
        s = step(ep, b, s, int(a), cfg); states.append(s)
    return states


def check_episode(ep: Episodes, cfg: RouteConfig):
    """Every key once; every query's oracle route reaches its answer; the word is readable from the data."""
    for b in range(ep.B):
        assert sorted(ep.pair_key[b].tolist()) == list(range(cfg.n_keys))
        for j in range(ep.q_start.shape[1]):
            st = follow(ep, b, j, cfg)
            assert st[-1] == ep.q_ans[b, j], (b, j, st)
            assert all(cfg.n_keys > x for x in st[:-1])
            # the word is written in the data: value == ARROW exactly at the P-steps
            for i, a in enumerate(ep.q_word[b, j, :ep.q_k[b, j]]):
                pos = int(np.where(ep.pair_key[b] == st[i])[0][0])
                assert (ep.pair_val[b, pos] == cfg.ARROW) == (a == P)
