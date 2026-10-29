"""Episodes for rung 1 (experiments/route/DESIGN.md §2, revision 2): chains over keys whose steps are one of two
generic circuits, HOP (follow the pair's value) or PREV (go to the key of the pair immediately before). A P-step is
announced by a FLAG on a pair `deferral` states earlier: deferral 0 flags the pair the P-step leaves from (a local
trigger); deferral 1 flags the pair one state before that ("the step out of the next state is PREV"), so the route
must carry a pending bit across a hop (a non-local trigger; the thesis setting).

Symbol ids: keys 0..n_keys-1, answers n_keys..n_keys+n_ans-1, DEAD = n_keys + n_ans (⊥; also the "no previous pair"
marker in the prev slot).

A chain is s_0 -> s_1 -> ... -> s_k with s_k an answer and a word w in {H, P}^k, w_1 = H and w_k = H:
  H at step i: pair(s_{i-1}).value = s_i;
  P at step i: pair(s_i) sits right before pair(s_{i-1}) in context order; pair(s_{i-1}).value is a DECOY (a random
               key of another chain, so HOP from s_{i-1} wanders rather than dies); the flag sits on
               pair(s_{i-1-deferral}).
Every pair token carries (key, value, prev, flag). Everything is numpy; tensors are built by the model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

H, P = 0, 1          # step codes in words


@dataclass
class RouteConfig:
    n_keys: int = 48
    n_ans: int = 16
    P: int = 48                 # pairs per episode; Qn * kmax <= P
    Qn: int = 6
    alpha: float = 1.5          # Zipf exponent of queried chain lengths
    kmax: int = 8
    p_P: float = 0.3            # probability that an admissible step (2 .. k-1) is P
    max_prun: int = 1           # longest P-run allowed in training words
    deferral: int = 1           # where the flag sits: 0 = on the P-step's own pair, 1 = one state earlier
    tail_factor: float = 1.0
    k_deep: int = 6
    seed: int = 0
    ans_pool: int = 0           # rung 2 dose: if > 0, each episode's chains end in only this many distinct answers
                                # (drawn per episode from all n_ans); the readout and likelihood keep all n_ans

    def dict(self):
        return asdict(self)

    @property
    def DEAD(self):
        return self.n_keys + self.n_ans

    @property
    def n_sym(self):
        return self.n_keys + self.n_ans + 1


def zipf_pmf(alpha, kmax, tail_factor=1.0, k_deep=6):
    k = np.arange(1, kmax + 1, dtype=float)
    p = k ** (-alpha)
    p[k >= k_deep] *= tail_factor
    return p / p.sum()


@dataclass
class Episodes:
    pair_key: np.ndarray    # (B, P)
    pair_val: np.ndarray    # (B, P): a key or an answer (a decoy key at P-steps)
    pair_prev: np.ndarray   # (B, P): key of the previous pair in context order, DEAD for the first
    pair_flag: np.ndarray   # (B, P): 1 where a P-step is announced
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
    """A word of length k with w_1 = H and w_k = H, P-runs of at most max_prun, P with probability p_P where allowed."""
    w = np.zeros(k, dtype=np.int64)
    run = 0
    for i in range(1, k - 1):
        if run < max_prun and rng.random() < p_P:
            w[i] = P; run += 1
        else:
            w[i] = H; run = 0
    return w


def word_class(w):
    w = np.asarray(w)
    k = len(w)
    best, cur = 0, 0
    for a in w:
        cur = cur + 1 if a == P else 0
        best = max(best, cur)
    nP = int((w == P).sum())
    deepest = int(np.where(w == P)[0].max()) + 1 if nP else 0
    return {"k": k, "max_prun": best, "nP": nP, "deepest_P": deepest}


def forced_word(rng, k, kind, p_P=0.3):
    """Evaluation strata. 'train': training statistics; 'none': all H; 'pp': contains a PP run (PPP allowed);
    'alt': H P H P ... with w_1 = w_k = H; 'deepP': a P at step >= 6."""
    if kind == "train":
        return sample_word(rng, k, p_P, 1)
    if kind == "none":
        return np.zeros(k, dtype=np.int64)
    if kind == "alt":
        assert k >= 3
        w = np.zeros(k, dtype=np.int64)
        for i in range(1, k - 1):
            w[i] = P if (i % 2 == 1) else H
        return w
    if kind == "pp":
        assert k >= 4
        w = sample_word(rng, k, p_P, 1)
        run = 2 if (k < 6 or rng.random() < 0.6) else 3
        start = int(rng.integers(1, k - run))        # steps start .. start+run-1 (0-based) are P; w_0 and w_{k-1} stay H
        w[start:start + run] = P
        w[start - 1] = H                             # the run is exactly `run` long: H on both sides
        if start + run < k:
            w[start + run] = H
        w[0] = H; w[k - 1] = H
        return w
    if kind == "deepP":
        assert k >= 7
        w = sample_word(rng, k, p_P, 1)
        pos = int(rng.integers(5, k - 1))
        w[pos] = P
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
    pair_key = np.zeros((B, Pn), np.int64); pair_val = np.zeros((B, Pn), np.int64)
    pair_prev = np.zeros((B, Pn), np.int64); pair_flag = np.zeros((B, Pn), np.int64)
    q_start = np.zeros((B, Qn), np.int64); q_ans = np.zeros((B, Qn), np.int64); q_k = np.zeros((B, Qn), np.int64)
    q_word = np.full((B, Qn, kmax_eval), -1, np.int64)
    pmf = zipf_pmf(cfg.alpha, cfg.kmax, cfg.tail_factor, cfg.k_deep)
    for b in range(B):
        keys = rng.permutation(cfg.n_keys)
        ks = _chain_lengths(rng, Qn, cfg, forced_k)
        pool = rng.choice(cfg.n_ans, cfg.ans_pool, replace=False) if cfg.ans_pool else None
        draw_ans = (lambda: cfg.n_keys + int(rng.integers(cfg.n_ans))) if pool is None else \
                   (lambda: cfg.n_keys + int(pool[rng.integers(cfg.ans_pool)]))
        chains = []
        used = 0
        for j, k in enumerate(ks):
            w = forced_word(rng, k, forced_kind, cfg.p_P) if (j == 0 and forced_kind) else sample_word(rng, k, cfg.p_P, cfg.max_prun)
            chains.append((keys[used:used + k], w, draw_ans()))
            used += k
        while used < cfg.n_keys:
            k = min(int(rng.choice(np.arange(1, cfg.kmax + 1), p=pmf)), cfg.n_keys - used)
            chains.append((keys[used:used + k], sample_word(rng, k, cfg.p_P, cfg.max_prun), draw_ans()))
            used += k
        val, flag = {}, {}
        for ck, w, ans in chains:
            k = len(ck)
            for i in range(k):                                # step i+1 leaves s_i
                nxt = ck[i + 1] if i + 1 < k else ans
                if w[i] == H:
                    val[ck[i]] = nxt
                else:
                    other = [kk for kk in keys if kk not in ck]   # a decoy: HOP wanders into another chain
                    val[ck[i]] = int(rng.choice(other)) if other else int(ck[0])
                    flag[ck[max(i - cfg.deferral, 0)]] = 1
        blocks = []
        for ck, w, ans in chains:
            k = len(ck)
            # a P at step i (leaving s_{i-1}) puts pair(s_i) right before pair(s_{i-1}): a P-run is one reversed block
            blocks_c, block = [], [ck[0]]
            for i in range(1, k):                             # state s_i, reached by step i (w[i-1])
                if w[i - 1] == P:
                    block.append(ck[i])
                else:
                    blocks_c.append(block[::-1]); block = [ck[i]]
            blocks_c.append(block[::-1])
            blocks.extend(blocks_c)
        order = rng.permutation(len(blocks))
        seq = [kk for o in order for kk in blocks[o]]
        assert len(seq) == Pn
        pair_key[b] = seq
        pair_val[b] = [val[kk] for kk in seq]
        pair_prev[b] = [cfg.DEAD] + seq[:-1]
        pair_flag[b] = [flag.get(kk, 0) for kk in seq]
        for j, (ck, w, ans) in enumerate(chains[:Qn]):
            q_start[b, j], q_ans[b, j], q_k[b, j] = ck[0], ans, len(ck)
            q_word[b, j, :len(w)] = w
    return Episodes(pair_key, pair_val, pair_prev, pair_flag, q_start, q_ans, q_k, q_word, meta={"forced_kind": forced_kind})


# ---------- the exact reference executor (numpy, one query) ----------
def oracle_actions(ep: Episodes, b: int, j: int, cfg: RouteConfig):
    """The oracle route of query j as (states, actions) over the product state (symbol, pending, set_here), following
    the written rule: HALT on an answer; SET at a flagged pair not yet set here; PREV when a pending flag applies to
    this exit (deferral 1: pending minus the one set here; deferral 0: pending); else HOP."""
    from .exec import HOP, PREV, SET, HALT   # action codes
    pos = {int(k): i for i, k in enumerate(ep.pair_key[b])}
    s, c, h = int(ep.q_start[b, j]), 0, 0
    states, acts = [(s, c, h)], []
    for _ in range(200):
        if s >= cfg.n_keys:
            acts.append(HALT); break
        p = pos[s]
        fl = int(ep.pair_flag[b, p])
        if fl and h == 0:
            a = SET; c, h = min(c + 1, 2), 1
        elif (c - (h if cfg.deferral == 1 else 0)) >= 1:
            a = PREV; s, c, h = int(ep.pair_prev[b, p]), c - 1, 0
        else:
            a = HOP; s, c, h = int(ep.pair_val[b, p]), c, 0
        acts.append(a); states.append((s, c, h))
    return states, acts


def check_episode(ep: Episodes, cfg: RouteConfig):
    """Every key once; every query's oracle route reaches its answer with the word's P-steps as PREV actions."""
    from .exec import PREV, HALT
    for b in range(ep.B):
        assert sorted(ep.pair_key[b].tolist()) == list(range(cfg.n_keys))
        for j in range(ep.q_start.shape[1]):
            states, acts = oracle_actions(ep, b, j, cfg)
            assert acts[-1] == HALT and states[-1][0] == ep.q_ans[b, j], (b, j, states, acts)
            w = ep.q_word[b, j, :ep.q_k[b, j]]
            assert sum(a == PREV for a in acts) == int((w == P).sum())
            moves = [a for a in acts if a in (0, 1)]
            assert [int(m == PREV) for m in moves] == w.tolist()
