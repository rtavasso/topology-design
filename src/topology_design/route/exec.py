"""The exact symbolic executor and the exact route marginal (experiments/route/DESIGN.md §§3–4, revision 2).

State: (symbol s, pending c ∈ {0, 1, 2}, set_here h ∈ {0, 1}), id = s * 6 + c * 2 + h; NS = n_sym * 6 states.
Actions: HOP (s <- value of s's pair; c kept; h <- 0), PREV (needs c >= 1: s <- key of the pair before; c - 1; h <- 0;
else DEAD), SET (c <- min(c + 1, 2), h <- 1; a setup step: the symbol does not move), HALT. Answers absorb under HOP
and PREV; DEAD absorbs everything.

maps(...) builds the three transition tables (B, NS) per episode; marginal(...) runs the forward recursion over the
product state for a Markov router pi (B, NS, 4) [or a step-indexed pi (T, B, 1, 4)]; committed(...) runs the argmax
route; oracle(...) gives the oracle action sequence from the written rule.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import RouteConfig, Episodes

HOP, PREV, SET, HALT = 0, 1, 2, 3
NC, NH = 3, 2           # pending counts, set-here values
PER_SYM = NC * NH       # product states per symbol


def sid(s, c, h):
    return s * PER_SYM + c * NH + h


def tables(ep: Episodes, cfg: RouteConfig, dev) -> dict:
    t = {k: torch.as_tensor(getattr(ep, k), device=dev) for k in
         ("pair_key", "pair_val", "pair_prev", "pair_flag", "q_start", "q_ans", "q_k", "q_word")}
    return {**t, **maps(t["pair_key"], t["pair_val"], t["pair_prev"], t["pair_flag"], cfg)}


def maps(pair_key, pair_val, pair_prev, pair_flag, cfg: RouteConfig) -> dict:
    """Transition tables over product states: hop, prev, set (B, NS) -> next state id; flag (B, S) per symbol;
    plus the per-symbol next tables sym_hop / sym_prev (B, S)."""
    B, S = pair_key.shape[0], cfg.n_sym
    dev = pair_key.device
    sym = torch.arange(S, device=dev)
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.n_keys + cfg.n_ans)
    base = torch.where(is_ans, sym, torch.full_like(sym, cfg.DEAD)).unsqueeze(0).expand(B, S).clone()
    sym_hop, sym_prev = base.clone(), base.clone()
    sym_hop.scatter_(1, pair_key, pair_val)
    sym_prev.scatter_(1, pair_key, pair_prev)
    flag = torch.zeros(B, S, dtype=torch.long, device=dev)
    flag.scatter_(1, pair_key, pair_flag)
    NS = S * PER_SYM
    ids = torch.arange(NS, device=dev)
    s_flat, c_flat, h_flat = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
    dead_state = sid(cfg.DEAD, 0, 0)
    is_ans_s = is_ans[s_flat].unsqueeze(0).expand(B, NS)
    self_state = ids.unsqueeze(0).expand(B, NS)
    zeros = torch.zeros(B, NS, dtype=torch.long, device=dev)
    c_b = c_flat.unsqueeze(0).expand(B, NS)
    # HOP: the symbol moves, pending kept, set_here cleared; dead stays dead; answers absorb
    nh = sym_hop[:, s_flat]
    hop = torch.where(nh == cfg.DEAD, torch.full_like(nh, dead_state), sid(nh, c_b, zeros))
    hop = torch.where(is_ans_s, self_state, hop)
    # PREV: needs pending >= 1; moves to the previous pair's key; pending - 1; set_here cleared
    npv = sym_prev[:, s_flat]
    prev = sid(npv, (c_b - 1).clamp_min(0), zeros)
    prev = torch.where((npv == cfg.DEAD) | (c_b < 1), torch.full_like(npv, dead_state), prev)
    prev = torch.where(is_ans_s, self_state, prev)
    # SET: the symbol stays, pending + 1 (capped), set_here = 1; a no-op on answers and dead
    sett = sid(s_flat, (c_flat + 1).clamp_max(NC - 1), torch.ones_like(h_flat)).unsqueeze(0).expand(B, NS)
    sett = torch.where(is_ans_s | (s_flat.unsqueeze(0) == cfg.DEAD), self_state, sett)
    return {"hop": hop, "prev": prev, "set": sett, "flag": flag, "sym_hop": sym_hop, "sym_prev": sym_prev}


def readout(cfg: RouteConfig, eps: float, dev) -> torch.Tensor:
    """(NS, n_ans): p(y | state). Answer symbols: 1 - eps on themselves; everything else uniform."""
    S, A = cfg.n_sym, cfg.n_ans
    R = torch.full((S, A), 1.0 / A, device=dev, dtype=torch.float32)
    for a in range(A):
        R[cfg.n_keys + a] = eps / (A - 1)
        R[cfg.n_keys + a, a] = 1 - eps
    return R.repeat_interleave(PER_SYM, 0)


def oracle_policy(mp: dict, cfg: RouteConfig) -> torch.Tensor:
    """The written rule as a deterministic Markov policy over product states: (B, NS, 4). HALT on an answer or
    dead; SET at a flagged pair not yet set here; PREV when a pending flag applies to this exit (deferral 1: pending
    minus the one set here; deferral 0: pending); else HOP."""
    B, S = mp["flag"].shape
    NS = S * PER_SYM
    dev = mp["flag"].device
    ids = torch.arange(NS, device=dev)
    s_flat, c_flat, h_flat = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
    is_ans = (s_flat >= cfg.n_keys) & (s_flat < cfg.n_keys + cfg.n_ans)
    halt = (is_ans | (s_flat == cfg.DEAD)).unsqueeze(0).expand(B, NS)
    fl = mp["flag"][:, s_flat] == 1
    pending_exit = (c_flat - (h_flat if cfg.deferral == 1 else 0)).unsqueeze(0).expand(B, NS)
    do_set = ~halt & fl & (h_flat.unsqueeze(0) == 0)
    do_prev = ~halt & ~do_set & (pending_exit >= 1)
    do_hop = ~halt & ~do_set & ~do_prev
    return torch.stack([do_hop, do_prev, do_set, halt], -1).float()


def marginal(pi: torch.Tensor, mp: dict, q_start: torch.Tensor, T: int, R: torch.Tensor):
    """Exact mixture over all routes of length <= T under a Markov router.
    pi: (B, NS, 4) or step-indexed (T, B, 1, 4). Returns q (B, Q, n_ans), expected steps (B, Q), the halting-mass
    profile (T + 1, B, Q) and the alive mass at T."""
    B, Q = q_start.shape
    NS = mp["hop"].shape[1]
    dev = mp["hop"].device
    pi = pi.to(R.dtype)
    b = F.one_hot(sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start)), NS).to(R.dtype)   # (B, Q, NS)
    q = torch.zeros(B, Q, R.shape[1], device=dev, dtype=R.dtype)
    esteps = torch.zeros(B, Q, device=dev, dtype=R.dtype)
    profile = []
    tabs = [mp["hop"].unsqueeze(1).expand(B, Q, NS), mp["prev"].unsqueeze(1).expand(B, Q, NS),
            mp["set"].unsqueeze(1).expand(B, Q, NS)]
    for t in range(T):
        p = (pi[t] if pi.dim() == 4 else pi).unsqueeze(1)                   # (B, 1, NS|1, 4)
        h = b * p[..., HALT]
        q = q + h @ R
        esteps = esteps + t * h.sum(-1)
        profile.append(h.sum(-1))
        nb = torch.zeros_like(b)
        for a, tab in zip((HOP, PREV, SET), tabs):
            nb.scatter_add_(2, tab, b * p[..., a])
        b = nb
    q = q + b @ R
    esteps = esteps + T * b.sum(-1)
    profile.append(b.sum(-1))
    return q, esteps, torch.stack(profile, 0), b.sum(-1)


@torch.no_grad()
def committed(pi: torch.Tensor, mp: dict, q_start: torch.Tensor, T: int):
    """The argmax route. Returns final symbols (B, Q), steps taken (circuit applications incl. SET) (B, Q), actions
    (B, Q, T) with -1 after halting, and the product-state sequence (B, Q, T + 1)."""
    B, Q = q_start.shape
    dev = q_start.device
    st = sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start))
    done = torch.zeros(B, Q, dtype=torch.bool, device=dev)
    steps = torch.zeros(B, Q, dtype=torch.long, device=dev)
    acts = torch.full((B, Q, T), -1, dtype=torch.long, device=dev)
    states = [st.clone()]
    ar = torch.arange(B, device=dev).unsqueeze(1)
    for t in range(T):
        p = pi[t] if pi.dim() == 4 else pi
        logits = p[:, 0, :].unsqueeze(1).expand(B, Q, 4) if p.shape[1] == 1 else p[ar, st]
        a = logits.argmax(-1)
        a = torch.where(done, torch.full_like(a, HALT), a)
        acts[:, :, t] = torch.where(done, torch.full_like(a, -1), a)
        nxt = torch.where(a == HOP, mp["hop"].gather(1, st), torch.where(a == PREV, mp["prev"].gather(1, st),
                          torch.where(a == SET, mp["set"].gather(1, st), st)))
        move = ~done & (a != HALT)
        st = torch.where(move, nxt, st)
        steps = steps + move.long()
        done = done | (a == HALT)
        states.append(st.clone())
    return st // PER_SYM, steps, acts, torch.stack(states, 2)


@torch.no_grad()
def oracle(mp: dict, q_start: torch.Tensor, T: int, cfg: RouteConfig):
    """The oracle route's actions (B, Q, T), its length (B, Q) and its states, under the written rule."""
    s, steps, acts, states = committed(oracle_policy(mp, cfg), mp, q_start, T)
    return acts, steps, states


def route_ok(acts: torch.Tensor, steps: torch.Tensor, oracle_acts: torch.Tensor, oracle_steps: torch.Tensor):
    """Route accuracy: the committed actions equal the oracle's and halt at the same step."""
    return (steps == oracle_steps) & ((acts == oracle_acts) | (oracle_acts == -1)).all(-1)


def remaining_distance(states: torch.Tensor, mp: dict, cfg: RouteConfig):
    """Exact remaining circuit applications to the answer under the written rule, from every product state in a
    state sequence (B, Q, L): -1 if the rule never reaches an answer from there."""
    B, Q, L = states.shape
    NS = mp["hop"].shape[1]
    dev = states.device
    a = oracle_policy(mp, cfg).argmax(-1)                                        # (B, NS)
    ids = torch.arange(NS, device=dev).unsqueeze(0).expand(B, NS)
    nxt = torch.where(a == HOP, mp["hop"], torch.where(a == PREV, mp["prev"], torch.where(a == SET, mp["set"], ids)))
    s_flat = torch.arange(NS, device=dev) // PER_SYM
    is_ans = ((s_flat >= cfg.n_keys) & (s_flat < cfg.n_keys + cfg.n_ans)).unsqueeze(0).expand(B, NS)
    dist = torch.where(is_ans, torch.zeros_like(ids), torch.full_like(ids, -1))
    cur = ids.clone()
    for d in range(1, 3 * NS):
        cur = nxt.gather(1, cur)
        reached = (dist == -1) & is_ans.gather(1, cur)
        dist = torch.where(reached, torch.full_like(dist, d), dist)
        if not (dist == -1)[(~is_ans) & (mp["sym_hop"][:, s_flat] != cfg.DEAD)].any() and d > 60:
            break
    return dist.gather(1, states.reshape(B, Q * L)).view(B, Q, L)
