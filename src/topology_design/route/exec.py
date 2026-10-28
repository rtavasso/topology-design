"""The exact symbolic executor and the exact route marginal (experiments/route/DESIGN.md §§3–4).

maps(ep): per-episode transition tables hop[b, s], prev[b, s] over all n_sym symbols (answers absorb; ARROW/DEAD are
dead; PREV from the first pair is dead). marginal(pi, ...) runs the forward recursion over symbols for a Markov
router pi (B, S, 3) [or a step-indexed pi (T, B, 1, 3)], returning q(y) for every query, the expected number of
steps, and the halting-mass profile. committed(...) runs the argmax route and returns its states, actions and steps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import RouteConfig, Episodes

HOP, PREV, HALT = 0, 1, 2


def tables(ep: Episodes, cfg: RouteConfig, dev) -> dict:
    """Integer tensors of the episode batch on dev, plus the transition tables (B, S) for HOP and PREV."""
    pk = torch.as_tensor(ep.pair_key, device=dev); pv = torch.as_tensor(ep.pair_val, device=dev)
    pp = torch.as_tensor(ep.pair_prev, device=dev)
    return {"pair_key": pk, "pair_val": pv, "pair_prev": pp, "q_start": torch.as_tensor(ep.q_start, device=dev),
            "q_ans": torch.as_tensor(ep.q_ans, device=dev), "q_k": torch.as_tensor(ep.q_k, device=dev),
            "q_word": torch.as_tensor(ep.q_word, device=dev), **maps(pk, pv, pp, cfg)}


def maps(pair_key, pair_val, pair_prev, cfg: RouteConfig) -> dict:
    """hop[b, s] and prev[b, s] for every symbol s (keys from the pairs; answers absorb; ARROW/DEAD -> DEAD)."""
    B, S = pair_key.shape[0], cfg.n_sym
    dev = pair_key.device
    sym = torch.arange(S, device=dev)
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.n_keys + cfg.n_ans)
    base = torch.where(is_ans, sym, torch.full_like(sym, cfg.DEAD)).unsqueeze(0).expand(B, S).clone()
    hop, prev = base.clone(), base.clone()
    val = torch.where(pair_val == cfg.ARROW, torch.full_like(pair_val, cfg.DEAD), pair_val)
    hop.scatter_(1, pair_key, val)
    prev.scatter_(1, pair_key, pair_prev)
    return {"hop": hop, "prev": prev}


def readout(cfg: RouteConfig, eps: float, dev) -> torch.Tensor:
    """(S, n_ans): p(y | register = s). Answers: 1 - eps on themselves; everything else uniform."""
    S, A = cfg.n_sym, cfg.n_ans
    R = torch.full((S, A), 1.0 / A, device=dev)
    for a in range(A):
        R[cfg.n_keys + a] = eps / (A - 1)
        R[cfg.n_keys + a, a] = 1 - eps
    return R


def marginal(pi: torch.Tensor, hop: torch.Tensor, prev: torch.Tensor, q_start: torch.Tensor, T: int, R: torch.Tensor):
    """Exact mixture over all routes of length <= T under a Markov router.

    pi: (B, S, 3) state-conditioned probabilities [HOP, PREV, HALT], or (T, B, 1, 3) step-conditioned (state-blind).
    Returns q (B, Q, n_ans), expected steps (B, Q), halt-mass profile (T + 1, B, Q) and the alive mass at T."""
    B, Q = q_start.shape
    S = hop.shape[1]
    dev = hop.device
    b = F.one_hot(q_start, S).to(R.dtype)                                   # (B, Q, S)
    q = torch.zeros(B, Q, R.shape[1], device=dev, dtype=R.dtype)
    esteps = torch.zeros(B, Q, device=dev, dtype=R.dtype)
    profile = []
    hop_e = hop.unsqueeze(1).expand(B, Q, S); prev_e = prev.unsqueeze(1).expand(B, Q, S)
    for t in range(T):
        p = pi[t] if pi.dim() == 4 else pi                                   # (B, S|1, 3)
        p = p.unsqueeze(1)                                                   # (B, 1, S|1, 3)
        h = b * p[..., HALT]                                                 # halt now, after t steps
        q = q + h @ R
        esteps = esteps + t * h.sum(-1)
        profile.append(h.sum(-1))
        nb = torch.zeros_like(b)
        nb.scatter_add_(2, hop_e, b * p[..., HOP])
        nb.scatter_add_(2, prev_e, b * p[..., PREV])
        b = nb
    # forced halt at T
    q = q + b @ R
    esteps = esteps + T * b.sum(-1)
    profile.append(b.sum(-1))
    return q, esteps, torch.stack(profile, 0), b.sum(-1)


@torch.no_grad()
def committed(pi: torch.Tensor, hop, prev, q_start, T: int, R: torch.Tensor):
    """The argmax route: at each state take the most probable action; stop at HALT or after T steps.
    Returns final states (B, Q), steps taken (B, Q), the action sequence (B, Q, T) with -1 after halting, and the
    state sequence (B, Q, T + 1)."""
    B, Q = q_start.shape
    s = q_start.clone()
    done = torch.zeros(B, Q, dtype=torch.bool, device=s.device)
    steps = torch.zeros(B, Q, dtype=torch.long, device=s.device)
    acts = torch.full((B, Q, T), -1, dtype=torch.long, device=s.device)
    states = [s.clone()]
    for t in range(T):
        p = pi[t] if pi.dim() == 4 else pi                                   # (B, S|1, 3)
        if p.shape[1] == 1:
            logits = p[:, 0, :].unsqueeze(1).expand(B, Q, 3)
        else:
            logits = p[torch.arange(B, device=s.device).unsqueeze(1), s]    # (B, Q, 3)
        a = logits.argmax(-1)
        a = torch.where(done, torch.full_like(a, HALT), a)
        acts[:, :, t] = torch.where(done, torch.full_like(a, -1), a)
        nxt = torch.where(a == HOP, hop.gather(1, s), torch.where(a == PREV, prev.gather(1, s), s))
        move = ~done & (a != HALT)
        s = torch.where(move, nxt, s)
        steps = steps + move.long()
        done = done | (a == HALT)
        states.append(s.clone())
    return s, steps, acts, torch.stack(states, 2)


def oracle_route_ok(acts: torch.Tensor, steps: torch.Tensor, q_word: torch.Tensor, q_k: torch.Tensor):
    """Route accuracy: the committed actions equal the oracle word and the route halts right after it."""
    B, Q, T = acts.shape
    k = q_k
    ok = steps == k
    idx = torch.arange(T, device=acts.device).view(1, 1, T)
    w = torch.where(idx < k.unsqueeze(-1), q_word[:, :, :T] if q_word.shape[-1] >= T else
                    F.pad(q_word, (0, T - q_word.shape[-1]), value=-1), torch.full_like(acts, -1))
    same = (acts == w) | (idx >= k.unsqueeze(-1))
    return ok & same.all(-1)


def remaining_distance(states: torch.Tensor, hop, prev, q_word, q_k, cfg: RouteConfig):
    """Exact remaining distance (hops to the answer along the oracle rule) for every state in a state sequence
    (B, Q, L): the number of steps the oracle rule needs from that symbol, or -1 if the symbol is not on any chain
    (dead) — computed by following the written rule (ARROW -> PREV, key -> HOP) up to n_sym steps."""
    B, Q, L = states.shape
    S = hop.shape[1]
    dev = states.device
    # distance table per episode: iterate the rule from every symbol
    dist = torch.full((B, S), -1, dtype=torch.long, device=dev)
    sym = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.n_keys + cfg.n_ans)
    dist[is_ans] = 0
    # the rule's next state: hop if the pair's value is not ARROW (hop != DEAD), else prev
    nxt = torch.where(hop != cfg.DEAD, hop, prev)
    cur = sym.clone()
    for d in range(1, S + 1):
        cur = nxt.gather(1, cur)
        reached = (dist == -1) & (dist.gather(1, cur) == 0) & ~is_ans
        # a symbol whose d-step successor is an answer has distance d (first time it is set)
        dist = torch.where(reached, torch.full_like(dist, d), dist)
    return dist.gather(1, states.reshape(B, Q * L)).view(B, Q, L)
