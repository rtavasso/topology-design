"""Route-finding without the exact marginal (rung 2, experiments/routefind/DESIGN.md).

Rung 1 trained the Markov router by the exact forward recursion over all 390 product states. Here the same router, on
the same executor and data, is trained by estimators that only ever see the routes they sample or search, each scored
exactly by the executor and the readout (the label is used only through p(y | route end), as in the marginal):

  iw  — the K-sample importance-weighted bound  log (1/K) sum_k p(y | r_k)  (r_k ~ pi), with VIMCO leave-one-out
        control variates (Mnih & Rezende 2016): a sampled estimate of the exact objective; as K -> inf it is it.
  rf  — REINFORCE on the expected log-likelihood  E_pi[log p(y | r)]  with RLOO baselines: the "RL" objective,
        a Jensen lower bound of the marginal that rewards each route on its own (mode-seeking).
  ei  — expert iteration (hard EM): a beam search under the current router (mixed with uniform, eps) over product
        states, the most probable route that reaches the label kept, the router trained by teacher forcing on it.

All three carry the step cost: iw and rf by a leave-one-out score-function term on the step count, ei in the route selection. The router is evaluated densely at every state for convenience (pi is Markov), but
gradients reach only the states the sampled / searched routes visit.
"""

from __future__ import annotations

import math

import torch

from .data import RouteConfig
from .exec import committed, sid, PER_SYM, HOP, PREV, SET, HALT


# ---------- shared pieces ----------
@torch.no_grad()
def sample_routes(pi: torch.Tensor, mp: dict, q_start: torch.Tensor, T: int, K: int, generator=None):
    """K routes per query sampled from pi (B, NS, 4). Returns final symbols (B, Q, K), steps (B, Q, K), actions
    (B, Q, K, T) with -1 after halting, product states (B, Q, K, T + 1)."""
    B, Q = q_start.shape
    qs = q_start.repeat_interleave(K, dim=1)
    s, steps, acts, states = committed(pi.detach(), mp, qs, T, sample=True, generator=generator)
    return s.view(B, Q, K), steps.view(B, Q, K), acts.view(B, Q, K, T), states.view(B, Q, K, T + 1)


def route_logprob(pi: torch.Tensor, acts: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """sum_t log pi(a_t | x_t) along given routes, differentiable in pi (B, NS, 4). acts (B, ..., T) with -1 after
    the route ends; states (B, ..., T + 1). A route forced to halt at T has no HALT term (as in the marginal)."""
    assert pi.dim() == 3, "state routers only"
    B = pi.shape[0]
    T = acts.shape[-1]
    logpi = torch.log(pi.float() + 1e-12)
    st = states[..., :T]
    bidx = torch.arange(B, device=pi.device).view(B, *([1] * (st.dim() - 1)))
    lp = logpi[bidx.expand_as(st), st]                                               # (B, ..., T, 4)
    lp = lp.gather(-1, acts.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.where(acts >= 0, lp, torch.zeros_like(lp)).sum(-1)


def readout_logp(sym: torch.Tensor, q_ans: torch.Tensor, cfg: RouteConfig, eps: float) -> torch.Tensor:
    """log p(y | the route ends at symbol sym), the rung-1 readout: 1 - eps on the right answer, eps / 15 on a wrong
    answer, 1/16 on anything else."""
    A = cfg.n_ans
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.DEAD)
    right = sym == q_ans
    f = lambda v: torch.full(sym.shape, v, device=sym.device, dtype=torch.float32)     # float32 under any default dtype
    return torch.where(right, f(math.log(1 - eps)), torch.where(is_ans, f(math.log(eps / (A - 1))), f(math.log(1.0 / A))))


def _loo_mean(x: torch.Tensor) -> torch.Tensor:
    """Leave-one-out mean over the last axis."""
    K = x.shape[-1]
    return (x.sum(-1, keepdim=True) - x) / (K - 1)


# ---------- iw: the K-sample bound with VIMCO control variates ----------
def iw_loss(lp: torch.Tensor, steps: torch.Tensor, logp_route: torch.Tensor, lam: float):
    """lp = log p(y | r_k) (B, Q, K), steps (B, Q, K), logp_route = log pi(r_k) (B, Q, K) with grad.
    Returns (surrogate per query, whose gradient is the VIMCO estimator of -grad[L_K] + lam grad E[steps];
    objective per query, -L_K + lam * mean steps, detached)."""
    K = lp.shape[-1]
    assert K >= 2
    lp = lp.float(); steps = steps.float()
    L = torch.logsumexp(lp, -1) - math.log(K)                                         # (B, Q)
    rep = lp.unsqueeze(-2).expand(*lp.shape, K).clone()                               # (B, Q, K[k], K[j])
    idx = torch.arange(K, device=lp.device)
    rep[..., idx, idx] = _loo_mean(lp)                                               # sample k replaced by the others' geometric mean
    L_minus = torch.logsumexp(rep, -1) - math.log(K)                                 # (B, Q, K)
    adv = (L.unsqueeze(-1) - L_minus).detach()
    s_adv = (steps - _loo_mean(steps)).detach()
    surrogate = -(adv * logp_route).sum(-1) + lam * (s_adv * logp_route).mean(-1)
    objective = (-L + lam * steps.mean(-1)).detach()
    return surrogate, objective


# ---------- rf: REINFORCE on the expected log-likelihood (RLOO) ----------
def rf_loss(lp: torch.Tensor, steps: torch.Tensor, logp_route: torch.Tensor, lam: float):
    """Reward R_k = log p(y | r_k) - lam * steps_k; surrogate whose gradient is the RLOO estimator of -grad E[R]."""
    K = lp.shape[-1]
    assert K >= 2
    R = lp.float() - lam * steps.float()
    adv = (R - _loo_mean(R)).detach()
    return -(adv * logp_route).mean(-1), (-R.mean(-1)).detach()


# ---------- ei: beam search over product states, then imitation ----------
@torch.no_grad()
def beam_search(pi: torch.Tensor, mp: dict, q_start: torch.Tensor, q_ans: torch.Tensor, T: int, W: int,
                cfg: RouteConfig, eps_mix: float = 0.1, lam: float = 0.02):
    """A width-W beam over product states under pi~ = (1 - eps) pi + eps / 4: at each step every kept state is
    expanded by HOP / PREV / SET, routes that reach the same state are merged (the most probable prefix kept; pi is
    Markov, so the merge loses nothing for the continuation), and the W most probable states are kept. Every kept
    state is also completed by HALT; the forced halt at T completes the rest. Among the completions that reach the
    label the one kept maximizes log pi~(r) - lam * moves (the MAP route under the current router and the step cost:
    hard EM over the beam). Not "the shortest correct route": with 16 answers a wide search reaches the label by
    spurious routes through decoys for almost every query, and the shortest of them is usually not the oracle's.
    Moves out of answers or dead states, into dead states, and no-op moves are pruned.
    Returns found (B, Q) [a completion reaches the label], actions (B, Q, T) of the kept route with HALT included and
    -1 after it, its moves (B, Q)."""
    B, Q = q_start.shape
    dev = pi.device
    logpi = torch.log((1 - eps_mix) * pi.float() + eps_mix / 4)                       # (B, NS, 4)
    st = sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start)).unsqueeze(-1).expand(B, Q, W).clone()
    lp = torch.full((B, Q, W), float("-inf"), device=dev); lp[..., 0] = 0.0
    hist = torch.full((B, Q, W, T), -1, dtype=torch.long, device=dev)
    best_key = torch.full((B, Q), float("-inf"), device=dev)
    best_acts = torch.full((B, Q, T), -1, dtype=torch.long, device=dev)
    best_steps = torch.zeros(B, Q, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev).view(B, 1, 1)
    sym_of = lambda x: x // PER_SYM
    is_ans = lambda x: (sym_of(x) >= cfg.n_keys) & (sym_of(x) < cfg.DEAD)
    is_dead = lambda x: sym_of(x) == cfg.DEAD
    tabs = (mp["hop"], mp["prev"], mp["set"])                                          # action order HOP, PREV, SET
    parent_of = torch.arange(W, device=dev).repeat_interleave(3)
    act_of = torch.arange(3, device=dev).repeat(W)
    tri = torch.tril(torch.ones(3 * W, 3 * W, dtype=torch.bool, device=dev), -1)      # [i, j] = j < i

    def complete(key, acts_new, t):
        nonlocal best_key, best_acts, best_steps
        bk, bi = key.max(-1)
        upd = bk > best_key
        chosen = acts_new.gather(2, bi.view(B, Q, 1, 1).expand(B, Q, 1, T)).squeeze(2)
        best_key = torch.where(upd, bk, best_key)
        best_acts = torch.where(upd.unsqueeze(-1), chosen, best_acts)
        best_steps = torch.where(upd, torch.full_like(best_steps, t), best_steps)

    def keys(lp_c, x, t):
        right = (sym_of(x) == q_ans.unsqueeze(-1)).float()
        k = right * 1e6 + (lp_c - lam * t).clamp(min=-999.0)
        return torch.where(torch.isfinite(lp_c), k, torch.full_like(k, float("-inf")))

    for t in range(T):
        l = logpi[ar, st]                                                              # (B, Q, W, 4)
        acts_halt = hist.clone(); acts_halt[..., t] = HALT
        complete(keys(lp + l[..., HALT], st, t), acts_halt, t)
        nxt = torch.stack([tab.gather(1, st.reshape(B, -1)).view(B, Q, W) for tab in tabs], -1)   # (B, Q, W, 3)
        nlp = lp.unsqueeze(-1) + l[..., :3]
        bad = (is_ans(st) | is_dead(st)).unsqueeze(-1) | is_dead(nxt) | (nxt == st.unsqueeze(-1))
        nlp = torch.where(bad, torch.full_like(nlp, float("-inf")), nlp)
        cs, clp = nxt.reshape(B, Q, 3 * W), nlp.reshape(B, Q, 3 * W)
        order = clp.argsort(-1, descending=True)
        s_sorted, lp_sorted = cs.gather(-1, order), clp.gather(-1, order)
        dup = ((s_sorted.unsqueeze(-1) == s_sorted.unsqueeze(-2)) & tri).any(-1)
        lp_sorted = torch.where(dup, torch.full_like(lp_sorted, float("-inf")), lp_sorted)
        top_lp, top_i = lp_sorted.topk(W, -1)
        sel = order.gather(-1, top_i)
        st, lp = cs.gather(-1, sel), top_lp
        parent, a = parent_of[sel], act_of[sel]
        hist = hist.gather(2, parent.unsqueeze(-1).expand(B, Q, W, T)).clone()
        hist[..., t] = a
    complete(keys(lp, st, T), hist, T)                                                 # the forced halt at T
    found = best_key >= 1e6 - 1e5
    return found, best_acts, best_steps


@torch.no_grad()
def replay(mp: dict, q_start: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """The product states (B, Q, T + 1) visited by given action sequences (-1 / HALT end a route)."""
    B, Q, T = acts.shape
    st = sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start))
    out = [st]
    for t in range(T):
        a = acts[:, :, t]
        nxt = torch.where(a == HOP, mp["hop"].gather(1, st), torch.where(a == PREV, mp["prev"].gather(1, st),
                          torch.where(a == SET, mp["set"].gather(1, st), st)))
        st = nxt
        out.append(st)
    return torch.stack(out, 2)
