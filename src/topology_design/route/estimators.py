"""Route-finding without the exact marginal (rung 2, experiments/routefind/DESIGN.md).

Rung 1 trained the Markov router by the exact forward recursion over all 390 product states. Here the same router, on
the same executor and data, is trained by estimators that only ever see the routes they sample or search, each scored
exactly by the executor and the readout (the label is used only through p(y | route end), as in the marginal):

  iw  — the K-sample importance-weighted bound  log (1/K) sum_k p(y | r_k)  (r_k ~ pi), with VIMCO leave-one-out
        control variates (Mnih & Rezende 2016): a sampled estimate of the exact objective; as K -> inf it is it.
  ell — the expected log-likelihood E_pi[log p(y | r)] (the K = 1 importance-weighted bound, an ELBO with the prior
        as posterior), estimated with K samples and RLOO baselines; "rf" is an alias. Its per-route reward prefers
        halting (log 1/16) to guessing (log eps/15), unlike the marginal (review 1, B3).
  rl  — REINFORCE with reward p(y | r) - lam * steps and RLOO: its expectation is q(y) - lam E[steps], so it
        maximizes sum q rather than sum log q (the "RL" side of Guu et al. 2017's RL-vs-MML comparison).
  si  — sample-then-imitate (STaR / ReST-EM style): K samples, the most probable sample that reaches the label kept,
        the router trained by teacher forcing on it (hard EM over samples instead of a beam).
  ei  — expert iteration (hard EM): a beam search under the current router (mixed with uniform, eps) over product
        states, the most probable route that reaches the label kept, the router trained by teacher forcing on it.

Revision 3-4 (the router's training signal; DESIGN.md §3) adds, with the answer-only likelihood L_y (answer_likelihood):
  pf    — supervision by the exact fixed-prior posterior p0(r | y) (posterior_counts, pf_loss: the PFN signal; it
          enumerates the state space, so it is a reference, not a candidate).
  pfis  — the same forward KL from K on-policy samples by self-normalized importance weights (pfis_loss).
  tb    — trajectory balance toward the same posterior (tb_loss; on-policy, i.e. its reverse KL).
  ac    — actor-critic with the router's learned contextual value V(x) (ac_loss; TD(gae)).
  exal  — (in the trainer) the exact current-policy marginal with the answer-only likelihood.

Every estimator carries the step cost. The router is evaluated densely at every state for convenience (pi is Markov),
but the sampled and searched estimators' gradients reach only the states their routes visit.
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
def rl_loss(lp: torch.Tensor, steps: torch.Tensor, logp_route: torch.Tensor, lam: float):
    """Reward R_k = p(y | r_k) - lam * steps_k (expectation q(y) - lam E[steps]); RLOO surrogate of -grad E[R]."""
    K = lp.shape[-1]
    assert K >= 2
    R = lp.float().exp() - lam * steps.float()
    adv = (R - _loo_mean(R)).detach()
    return -(adv * logp_route).mean(-1), (-R.mean(-1)).detach()


@torch.no_grad()
def best_correct_sample(sym, q_ans, logp_route, steps, lam):
    """For sample-then-imitate: per query the index of the sample that reaches the label with the highest
    log pi - lam * steps, and whether any does. sym, logp_route, steps (B, Q, K)."""
    score = torch.where(sym == q_ans.unsqueeze(-1), logp_route.double() - lam * steps.double(),
                        torch.full(sym.shape, float("-inf"), dtype=torch.float64, device=sym.device))
    best = score.max(-1)
    return torch.isfinite(best.values), best.indices


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
    label the one kept maximizes log pi~(r) - lam * moves: the MAP route under the current router and the step cost
    over the PRUNED route space (no no-op moves, halting at the first arrival at an answer — code review 1, finding
    7: in the unpruned space a route that loops at the answer and is forced to halt at T can score higher under a
    random policy); hard EM over the beam. Not "the shortest correct route": with 16 answers a wide search reaches the label by
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
    best_key = torch.full((B, Q), float("-inf"), device=dev, dtype=torch.float64)
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
        """log pi~ - lam * moves for completions that reach the label, -inf otherwise (float64: no quantization)."""
        right = sym_of(x) == q_ans.unsqueeze(-1)
        k = lp_c.double() - lam * t
        return torch.where(right & torch.isfinite(lp_c), k, torch.full_like(k, float("-inf")))

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
    found = torch.isfinite(best_key)
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


# ---------- rung 2 revision 3: amortized-posterior and value signals ----------
def answer_likelihood(sym: torch.Tensor, q_ans: torch.Tensor, cfg: RouteConfig, eps: float) -> torch.Tensor:
    """p(y | a route that halts at symbol sym) for the POSTERIOR targets: 1 - eps on the label, eps / (A - 1) on another
    answer, and 0 on a key or dead. A route explains the observed answer only if it outputs an answer; without this the
    flat-prior posterior is dominated by "halt at once and guess" (1/4 x 1/16 per query beats every real computation)."""
    A = cfg.n_ans
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.DEAD)
    return torch.where(sym == q_ans, torch.full(sym.shape, 1 - eps, dtype=torch.float64, device=sym.device),
                       torch.where(is_ans, torch.full(sym.shape, eps / (A - 1), dtype=torch.float64, device=sym.device),
                                   torch.zeros(sym.shape, dtype=torch.float64, device=sym.device)))


def flat_prior(mp: dict, cfg: RouteConfig) -> torch.Tensor:
    """The fixed route prior pi0 (B, NS, 4): uniform over the four actions at a key; HALT at an answer or dead (the
    executor's convention that computation stops at an answer)."""
    B, S = mp["flag"].shape
    NS = S * PER_SYM
    s = torch.arange(NS, device=mp["flag"].device) // PER_SYM
    term = (s >= cfg.n_keys)                                             # answers and dead
    pi0 = torch.full((NS, 4), 0.25, dtype=torch.float64, device=s.device)
    pi0[term] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=s.device)
    return pi0.unsqueeze(0).expand(B, NS, 4)


@torch.no_grad()
def posterior_counts(pi0: torch.Tensor, mp: dict, q_start: torch.Tensor, q_ans: torch.Tensor, T: int, cfg: RouteConfig,
                     lam: float, eps: float):
    """The exact posterior over routes given the answer, under a fixed prior pi0 (B, NS, 4), the step cost (factor
    exp(-lam) per move) and the answer likelihood, by forward-backward over (state, step):
        p0(r | y) ∝ prod_t pi0(a_t | x_t) exp(-lam [a_t is a move]) * L_y(end of r),   forced halt at T.
    Returns counts (B, Q, NS, 4): the posterior expected number of times action a is taken at state x (summed over
    steps), and Z (B, Q) = p0(y) (0 where no route of length <= T reaches an answer)."""
    B, Q = q_start.shape
    NS = pi0.shape[1]
    dev = q_start.device
    sym = torch.arange(NS, device=dev) // PER_SYM
    Ly = answer_likelihood(sym.view(1, 1, NS).expand(B, Q, NS), q_ans.unsqueeze(-1), cfg, eps)          # (B, Q, NS)
    w = pi0.double().clone()
    w[..., :3] = w[..., :3] * math.exp(-lam)
    w = w.unsqueeze(1).expand(B, Q, NS, 4)
    tabs = [mp[k].unsqueeze(1).expand(B, Q, NS) for k in ("hop", "prev", "set")]
    alpha = [F_onehot(sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start)), NS)]
    for t in range(T):
        a = alpha[-1]
        nb = torch.zeros_like(a)
        for i, tab in enumerate(tabs):
            nb.scatter_add_(2, tab, a * w[..., i])
        alpha.append(nb)
    beta = [None] * (T + 1)
    beta[T] = Ly
    for t in range(T - 1, -1, -1):
        nxt = beta[t + 1]
        beta[t] = w[..., HALT] * Ly + sum(w[..., i] * nxt.gather(2, tab) for i, tab in enumerate(tabs))
    s0 = sid(q_start, torch.zeros_like(q_start), torch.zeros_like(q_start))
    Z = beta[0].gather(2, s0.unsqueeze(-1)).squeeze(-1)
    counts = torch.zeros(B, Q, NS, 4, dtype=torch.float64, device=dev)
    for t in range(T):
        a, nxt = alpha[t], beta[t + 1]
        for i, tab in enumerate(tabs):
            counts[..., i] += a * w[..., i] * nxt.gather(2, tab)
        counts[..., HALT] += a * w[..., HALT] * Ly
    counts = counts / Z.clamp_min(1e-300).view(B, Q, 1, 1)
    counts = torch.where((Z > 0).view(B, Q, 1, 1), counts, torch.zeros_like(counts))
    return counts, Z


def F_onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx, n).double()


def pf_loss(pi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Amortized-posterior supervision (the PFN signal): -sum_{x,a} counts(x, a) log pi(a | x) per query, i.e. the
    expected log-likelihood of the router's routes under the FIXED-prior posterior. Its gradient is
    E_{p0(r | y)}[grad log pi(r)], where the exact marginal's (EX) is E_{p_pi(r | y)}[grad log pi(r)]: the same
    M-step with the posterior taken under the fixed prior instead of the router's own policy."""
    logpi = torch.log(pi.float() + 1e-12)                                               # (B, NS, 4)
    return -(counts.float() * logpi.unsqueeze(1)).sum((-1, -2))                         # (B, Q)


@torch.no_grad()
def canonical_routes(acts: torch.Tensor, states: torch.Tensor, cfg: RouteConfig):
    """Cut sampled routes at their first arrival at an answer or dead state (code review 3, bug 1). Answers and dead
    absorb, so the end symbol is unchanged, but a sampled route may keep "moving" there (no-op moves) before halting,
    where the flat prior pi0 has zero mass. The canonical route ENDS on arrival with no action there (as the forced halt
    at T: pi0 halts at an answer or dead with probability 1, so the cut route carries no HALT term). Including a
    log pi(HALT | arrival) term instead lets trajectory balance shrink it freely, which makes routes into dead look
    under-weighted and drives the router there (revision-4 CPU run: TB16 collapsed into PREV-to-dead). Halting at the
    arrival is taught by a separate local term (arrival_halt_loss). A route that halts on a key keeps its HALT.
    Returns (actions, moves, arrival state (or -1 when the route ends on a key or at T on a key))."""
    T = acts.shape[-1]
    term = (states // PER_SYM) >= cfg.n_keys                                           # (..., T + 1): answer or dead
    arrived = term.any(-1)            # a route that halts on a key stays there, so it never arrives
    t_arr = torch.where(arrived, term.float().argmax(-1), torch.full(term.shape[:-1], T + 1, device=acts.device))
    tt = torch.arange(T, device=acts.device).view(*([1] * (acts.dim() - 1)), T)
    a = torch.where(tt < t_arr.unsqueeze(-1), acts, torch.full_like(acts, -1))
    moves = ((a >= 0) & (a != HALT)).sum(-1)
    arr_state = torch.where(arrived, states.gather(-1, t_arr.clamp_max(T).unsqueeze(-1)).squeeze(-1), torch.full_like(t_arr, -1))
    return a, moves, arr_state


def arrival_halt_loss(pi: torch.Tensor, arr_state: torch.Tensor) -> torch.Tensor:
    """-log pi(HALT | the answer or dead state a sampled route arrived at), averaged over the routes that arrived (per
    query): the prior knowledge that computation stops at an answer, which pi0 encodes (HALT with probability 1), taught
    locally because the canonical route carries no action there."""
    B = pi.shape[0]
    ok = arr_state >= 0
    bidx = torch.arange(B, device=pi.device).view(B, *([1] * (arr_state.dim() - 1))).expand_as(arr_state)
    lp = torch.log(pi[bidx, arr_state.clamp_min(0), HALT].float() + 1e-12)
    return -(lp * ok.float()).sum(-1) / ok.float().sum(-1).clamp_min(1)


def tb_loss(lpy_route: torch.Tensor, steps: torch.Tensor, logp_route: torch.Tensor, logp0_route: torch.Tensor,
            lam: float, floor: float = 1e-6):
    """GFlowNet trajectory balance with log Z estimated per query as the mean over its K samples (the VarGrad / log-Z-free
    form): loss = Var_k[ log pi(r_k) - log R(r_k) ], with the reward R(r) = pi0(r) exp(-lam steps) max(L_y(r), floor),
    whose normalized form is the fixed-prior posterior (the PF target). On-policy samples, on CANONICAL routes (cut at
    the first arrival at an answer or dead state: canonical_routes). On-policy VarGrad TB is the gradient of the reverse
    KL(pi || p0(.|y)) (an entropy-regularized REINFORCE on log R; design review 3, B5), not PF's forward KL.
    lpy_route = log L_y at the route's end, logp0_route = log pi0(r). Returns (loss per query, mean log-ratio per query
    detached)."""
    logR = logp0_route - lam * steps.float() + torch.clamp(lpy_route, min=math.log(floor))
    d = logp_route - logR.detach()
    return ((d - d.mean(-1, keepdim=True)) ** 2).mean(-1), d.mean(-1).detach()


def pfis_loss(lpy_route: torch.Tensor, steps: torch.Tensor, logp_route: torch.Tensor, logp0_route: torch.Tensor,
              lam: float):
    """The fixed-prior posterior's forward KL from samples (design review 3, B5): self-normalized importance sampling
    with the router as proposal, weights w_k ∝ pi0(r_k) exp(-lam steps_k) L_y(r_k) / pi(r_k) on canonical routes;
    loss = -sum_k w~_k log pi(r_k), whose gradient estimates PF's E_{p0(r|y)}[grad log pi(r)]. It differs from IW16's
    VIMCO-free form only by the factor pi0 / pi in the weights (IW16's posterior is the router's own). Queries whose
    samples all miss an answer give no gradient. Returns (loss per query, effective sample size / K per query)."""
    K = logp_route.shape[-1]
    logw = (logp0_route.float() - lam * steps.float() + lpy_route.float() - logp_route.detach().float())
    ok = torch.isfinite(logw).any(-1, keepdim=True)
    logw = torch.where(ok, logw, torch.zeros_like(logw))
    wn = torch.softmax(logw, -1) * ok.float()
    ess = (1.0 / (wn ** 2).sum(-1).clamp_min(1e-30)) * ok.squeeze(-1).float() / K
    return -(wn.detach() * logp_route).sum(-1), ess.detach()


def ac_loss(pi: torch.Tensor, V: torch.Tensor, acts: torch.Tensor, states: torch.Tensor, reward: torch.Tensor,
            lam: float, gae: float = 0.0, actor_scale: float = 1.0):
    """Actor-critic with a learned contextual value V(x) (B, NS) — the router's own estimate of a state's value given
    the context, without the label. Reward: -lam per move, `reward` (B, Q, K) at the end (p(y | r), as RL16's), so
    RL16 vs AC16 isolates bootstrapping. TD(lambda=gae) advantages along K sampled routes; gae = 0 is TD(0) (pure
    bootstrapping), gae = 1 the Monte-Carlo return. Critic loss: squared error to the lambda-return (detached).
    actor_scale = 0 during the critic's warm-up (policy evaluation before improvement: with V = 0, TD(0) charges every
    move -lam and pays only the final action, so the router collapses into halting at once; revision-4 CPU run).
    Returns (actor loss + critic loss, per query; mean return per query detached)."""
    B = pi.shape[0]
    T = acts.shape[-1]
    logpi = torch.log(pi.float() + 1e-12)
    st = states                                                                          # (B, Q, K, T + 1)
    bidx = torch.arange(B, device=pi.device).view(B, 1, 1, 1)
    v = V.float()[bidx.expand_as(st), st]                                                # (B, Q, K, T + 1)
    valid = acts >= 0                                                                    # (B, Q, K, T)
    is_move = valid & (acts != HALT)
    n_steps = valid.sum(-1)                                                              # actions incl. HALT
    last = (n_steps - 1).clamp_min(0)
    # the value after the final action: 0 (terminal) for HALT and for the forced halt at T
    r = torch.where(is_move, torch.full(acts.shape, -lam, device=pi.device), torch.zeros(acts.shape, device=pi.device))
    endmask = torch.zeros_like(r).scatter_(-1, last.unsqueeze(-1), 1.0) * (n_steps > 0).unsqueeze(-1).float()
    r = r + endmask * reward.float().unsqueeze(-1)
    v_next = torch.cat([v[..., 1:T + 1], torch.zeros_like(v[..., :1])], -1)[..., :T]
    v_next = torch.where(endmask.bool(), torch.zeros_like(v_next), v_next)
    delta = (r + v_next - v[..., :T]).detach() * valid.float()
    adv = torch.zeros_like(delta)
    run = torch.zeros_like(delta[..., 0])
    for t in range(T - 1, -1, -1):
        run = delta[..., t] + gae * run * valid[..., t].float()
        adv[..., t] = run
    target = (adv + v[..., :T]).detach()
    lp = logpi[bidx.expand_as(st[..., :T]), st[..., :T]].gather(-1, acts.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    actor = -(adv * lp * valid.float()).sum(-1) * actor_scale
    critic = 0.5 * (((v[..., :T] - target) ** 2) * valid.float()).sum(-1)
    ret = (r * valid.float()).sum(-1)
    with torch.no_grad():   # critic diagnostics, as per-episode sums (the trainer aggregates them per run)
        vm = valid.float()
        G = torch.flip(torch.cumsum(torch.flip(r * vm, [-1]), -1), [-1])               # MC return-to-go
        res = G - v[..., :T]
        sm = lambda x: (x * vm).sum((1, 2, 3))
        diag = {"n": sm(torch.ones_like(G)), "sG": sm(G), "sG2": sm(G ** 2), "sR": sm(res), "sR2": sm(res ** 2),
                "sTD": sm(delta.abs())}
    ac_loss.diag = diag
    return (actor + critic).mean(-1), (-ret.mean(-1)).detach()
