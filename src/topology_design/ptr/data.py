"""Episode generator and exact symbolic executor for pointer chasing (experiments/ptr/DESIGN.md §2).

An episode is a forest of chains over key symbols; each chain ends at an answer symbol. The context is the set of
pairs (key -> next), one token per pair, in shuffled order. Then come the queries: a start key (a chain head) whose
answer is the terminal reached by following the pairs. The chain length k is hidden until the chain is followed.

Symbol ids: keys are 0 .. n_keys-1, answers are n_keys .. n_keys+n_ans-1. Everything is numpy; tensors are built by
the model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class PtrConfig:
    n_keys: int = 48            # key symbols (every episode uses all of them, permuted); n_keys_train <= n_keys
    n_keys_train: int = 48
    n_ans: int = 16             # answer symbols
    n_ans_train: int = 16
    P: int = 48                 # pairs per episode (also the number of distinct keys used); Qn*kmax <= P so no length is ever rejected
    Qn: int = 6                 # queries per episode
    alpha: float = 1.5          # Zipf exponent of the queried chain lengths
    kmax: int = 8               # truncation of the length distribution for training
    tail_factor: float = 1.0    # P-pool manipulation: multiply the pmf of k >= k_deep by this factor (then renormalize)
    k_deep: int = 6
    seed: int = 0

    def dict(self):
        return asdict(self)

    @property
    def n_sym(self):
        return self.n_keys + self.n_ans


def zipf_pmf(alpha: float, kmax: int, tail_factor: float = 1.0, k_deep: int = 6) -> np.ndarray:
    k = np.arange(1, kmax + 1, dtype=float)
    p = k ** (-alpha)
    p[k >= k_deep] *= tail_factor
    return p / p.sum()


@dataclass
class Episodes:
    """A batch of episodes (B of them). pair_val holds key ids or answer ids (unified id space)."""
    pair_key: np.ndarray   # (B, P) int
    pair_val: np.ndarray   # (B, P) int
    q_start: np.ndarray    # (B, Qn) int, a key id
    q_ans: np.ndarray      # (B, Qn) int, an answer id (unified space)
    q_k: np.ndarray        # (B, Qn) int, hops from start to answer
    meta: dict = field(default_factory=dict)

    @property
    def B(self):
        return self.pair_key.shape[0]


def _chain_lengths(rng, Qn, alpha, kmax, P, forced=None, tail_factor=1.0, k_deep=6):
    """Queried chain lengths: Zipf(alpha) truncated at kmax (tail scaled by tail_factor), resampled until they fit in
    P pairs. forced: optional length for query 0 (evaluation strata)."""
    pmf = zipf_pmf(alpha, kmax, tail_factor, k_deep)
    while True:
        ks = rng.choice(np.arange(1, kmax + 1), size=Qn, p=pmf)
        if forced is not None:
            ks[0] = forced
        if ks.sum() <= P:
            return ks


def generate(cfg: PtrConfig, B: int, rng: np.random.Generator, *, key_pool: int | None = None,
             ans_pool: int | None = None, forced_k: int | None = None, Qn: int | None = None) -> Episodes:
    """B episodes. key_pool / ans_pool: how many symbols to draw from (defaults: the training pools).
    forced_k: query 0 of every episode has exactly this length (evaluation by stratum). Qn overrides cfg.Qn."""
    key_pool = cfg.n_keys_train if key_pool is None else key_pool
    ans_pool = cfg.n_ans_train if ans_pool is None else ans_pool
    Qn = cfg.Qn if Qn is None else Qn
    P = cfg.P
    assert key_pool >= P, "a forest needs P distinct keys"
    pair_key = np.zeros((B, P), np.int64)
    pair_val = np.zeros((B, P), np.int64)
    q_start = np.zeros((B, Qn), np.int64)
    q_ans = np.zeros((B, Qn), np.int64)
    q_k = np.zeros((B, Qn), np.int64)
    kmax_here = cfg.kmax if forced_k is None else max(cfg.kmax, forced_k)
    for b in range(B):
        ks = _chain_lengths(rng, Qn, cfg.alpha, cfg.kmax, P, forced=forced_k, tail_factor=cfg.tail_factor,
                            k_deep=cfg.k_deep)
        keys = rng.choice(key_pool, size=P, replace=False)
        # distractor chains fill the remaining pairs
        lens = list(ks)
        rem = P - int(ks.sum())
        pmf = zipf_pmf(cfg.alpha, cfg.kmax)
        while rem > 0:
            L = int(rng.choice(np.arange(1, cfg.kmax + 1), p=pmf))
            L = min(L, rem)
            lens.append(L)
            rem -= L
        pos = 0
        pk, pv = [], []
        for ci, L in enumerate(lens):
            chain = keys[pos:pos + L]
            pos += L
            ans = cfg.n_keys + int(rng.integers(ans_pool))
            for j in range(L):
                pk.append(chain[j])
                pv.append(chain[j + 1] if j + 1 < L else ans)
            if ci < Qn:
                q_start[b, ci] = chain[0]
                q_ans[b, ci] = ans
                q_k[b, ci] = L
        perm = rng.permutation(P)
        pair_key[b] = np.asarray(pk)[perm]
        pair_val[b] = np.asarray(pv)[perm]
        qperm = rng.permutation(Qn)
        q_start[b], q_ans[b], q_k[b] = q_start[b, qperm], q_ans[b, qperm], q_k[b, qperm]
    return Episodes(pair_key, pair_val, q_start, q_ans, q_k,
                    meta={"key_pool": key_pool, "ans_pool": ans_pool, "forced_k": forced_k, "kmax": kmax_here})


def follow(ep: Episodes, b: int, start: int, steps: int) -> int:
    """Exact executor: the symbol in the register after `steps` hops from `start` (stays at an answer once reached)."""
    nxt = dict(zip(ep.pair_key[b].tolist(), ep.pair_val[b].tolist()))
    r = int(start)
    for _ in range(steps):
        if r not in nxt:
            break
        r = nxt[r]
    return r


def remaining_hops(ep: Episodes, b: int, sym: int, n_keys: int) -> int:
    """Exact branch value: hops from `sym` to its answer (0 if sym is an answer)."""
    nxt = dict(zip(ep.pair_key[b].tolist(), ep.pair_val[b].tolist()))
    r, h = int(sym), 0
    while r < n_keys:
        r = nxt[r]
        h += 1
    return h


def check_forest(ep: Episodes, cfg: PtrConfig) -> None:
    """Every pair key unique; every chain ends at an answer; query k and answers agree with the executor."""
    for b in range(ep.B):
        assert len(set(ep.pair_key[b].tolist())) == cfg.P
        for s, a, k in zip(ep.q_start[b], ep.q_ans[b], ep.q_k[b]):
            assert remaining_hops(ep, b, int(s), cfg.n_keys) == int(k)
            assert follow(ep, b, int(s), int(k)) == int(a) and int(a) >= cfg.n_keys
            assert follow(ep, b, int(s), int(k) - 1) < cfg.n_keys
        for key in ep.pair_key[b]:
            assert remaining_hops(ep, b, int(key), cfg.n_keys) <= cfg.P
