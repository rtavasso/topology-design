"""Training of the routed arms of rung 1 (experiments/route/DESIGN.md §§4, 7, revision 2): a stack of runs of one arm
trained together, each on its own episode pool, by the exact route marginal (or by teacher forcing on the oracle
route).

A run config is a flat dict: data fields (RouteConfig), router fields (RouterConfig) and training fields
(TRAIN_DEFAULTS). run_stack(cfgs, dev) returns one record per run: the loss curve, the committed route accuracy by k
and the teacher-forced accuracy by action on a HELD-OUT criterion set (every 100 steps; D97), evaluations by stratum
(route / answer / marginal accuracy, steps, halting by remaining distance, the first deviation from the oracle route),
and the router's head parameters.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from ..ptr.stacked import StackedAdamW, clip_grad_norm_per_run
from .data import RouteConfig, generate
from .exec import (marginal, committed, oracle, route_ok, readout, maps, remaining_distance, counterfactual_probes,
                   HOP, PREV, SET, HALT, PER_SYM)
from .pool import EpisodePool, stack_batches, strata_episodes, to_dict, STRATA
from .router import StackedRouter, RouterConfig
from . import estimators as E

DATA_KEYS = {f.name for f in fields(RouteConfig)}
ROUTER_KEYS = {f.name for f in fields(RouterConfig)}
TRAIN_DEFAULTS = dict(steps=6000, batch=64, lr=1e-3, warmup=500, seed=0, eval_every=2000, T_eval=64, eval_n=250,
                      pool_n=60000, wd=0.01, grad_clip=1.0, teacher=False, crit_n=256, crit_every=100, eval_chunk=50, samples=4,
                      estimator="exact", K=16, W=16, ei_eps=0.1, warm_steps=0, per_run_init=False, gae=0.0)
# rung 2 (experiments/routefind/DESIGN.md): route-finding estimators; "rf" is an alias of "ell"
ESTIMATORS = ("exact", "teacher", "iw", "ell", "rf", "rl", "si", "ei", "pf", "tb", "ac")
SAMPLED = ("iw", "ell", "rf", "rl", "si", "tb", "ac")
# revision 3: pf = exact fixed-prior posterior supervision (the PFN signal); tb = GFlowNet trajectory balance toward
# the same posterior from samples; ac = actor-critic with the router's learned contextual value V(x)
PER_RUN_KEYS = ("seed", "lr")
ACTION_NAMES = {HOP: "HOP", PREV: "PREV", SET: "SET", HALT: "HALT"}


def split_cfg(cfg):
    dc = RouteConfig(**{k: v for k, v in cfg.items() if k in DATA_KEYS})
    rc = RouterConfig(**{k: v for k, v in cfg.items() if k in ROUTER_KEYS})
    tc = {**TRAIN_DEFAULTS, **{k: v for k, v in cfg.items() if k in TRAIN_DEFAULTS}}
    return dc, rc, tc


def flat(b: dict):
    """(R, B, ...) -> (R*B, ...)."""
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}


def policy(m: StackedRouter, b: dict, T: int, chunk: int | None = None):
    """pi for the exec functions: (R*B, NS, 4) for the state router; (T, R*B, 1, 4) for the index router.
    chunk: evaluate the state router in slices of the batch (memory)."""
    R, B = b["pair_key"].shape[:2]
    if m.rc.kind == "index":
        lg = m.logits(b, T=T)
        return torch.softmax(lg, -1).unsqueeze(2).expand(T, R, B, 1, 4).reshape(T, R * B, 1, 4)
    if chunk is None or chunk >= B:
        return torch.softmax(m.logits(b), -1).reshape(R * B, -1, 4)
    parts = [torch.softmax(m.logits({k: v[:, i:i + chunk] for k, v in b.items()}), -1) for i in range(0, B, chunk)]
    return torch.cat(parts, 1).reshape(R * B, -1, 4)


def gather_pi(pi, states, t):
    """pi(. | state) along a route: (B', Q, 4) at step t for states (B', Q)."""
    p = pi[t] if pi.dim() == 4 else pi
    if p.shape[1] == 1:
        return p[:, 0, :].unsqueeze(1).expand(states.shape[0], states.shape[1], 4)
    ar = torch.arange(states.shape[0], device=states.device).unsqueeze(1)
    return p[ar, states]


def teacher_loss(pi, oracle_acts, oracle_states):
    """-sum_t log pi(a_t* | s_t*) along the oracle route (HALT included); per query."""
    T = oracle_acts.shape[-1]
    ll = torch.zeros(oracle_acts.shape[:2], device=oracle_acts.device, dtype=torch.float32)
    for t in range(T):
        a = oracle_acts[:, :, t]
        valid = a >= 0
        if not valid.any():
            break
        lp = torch.log(gather_pi(pi, oracle_states[:, :, t], t).float() + 1e-9)
        ll = ll + torch.where(valid, lp.gather(-1, a.clamp_min(0).unsqueeze(-1)).squeeze(-1), torch.zeros_like(ll))
    return -ll


@torch.no_grad()
def action_accuracy(pi, oracle_acts, oracle_states):
    """Teacher-forced accuracy by action: among oracle steps with action a, the fraction where argmax pi at the
    oracle state is a. Returns hits (4,), counts (4,) per run-slice given as flat tensors."""
    T = oracle_acts.shape[-1]
    hits = torch.zeros(4, device=oracle_acts.device, dtype=torch.float32); cnt = torch.zeros_like(hits)
    for t in range(T):
        a = oracle_acts[:, :, t]
        valid = a >= 0
        if not valid.any():
            break
        pred = gather_pi(pi, oracle_states[:, :, t], t).argmax(-1)
        av = a[valid]
        cnt.scatter_add_(0, av, torch.ones_like(av, dtype=torch.float32))
        hits.scatter_add_(0, av, (pred[valid] == av).to(torch.float32))
    return hits, cnt


@torch.no_grad()
def criterion(m: StackedRouter, dc, rc, tc, crit: list[dict], dev):
    """Held-out criterion set (one per run, training distribution): committed route accuracy by k at the training
    horizon and the teacher-forced accuracy by action. Returns per run a dict."""
    R = m.R
    b = stack_batches([crit[r] for r in range(R)])
    fb = flat(b)
    mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
    pi = policy(m, b, rc.T, chunk=tc["eval_chunk"])
    s_fin, steps, acts, states_c = committed(pi, mp, fb["q_start"], rc.T)
    oa, osteps, ostates = oracle(mp, fb["q_start"], rc.T, dc)
    ok = route_ok(acts, steps, oa, osteps).float()
    ok_ans = (s_fin == fb["q_ans"]).float()
    min_ok, on_flag = program_stats(acts, states_c, steps, s_fin, fb["q_ans"], osteps, mp, oa, ostates)
    min_ok = min_ok.float()
    q, es, _, _ = marginal(pi, mp, fb["q_start"], rc.T, readout(dc, rc.eps, dev))     # the exact objective (every estimator)
    nll = -torch.log(q.gather(-1, (fb["q_ans"] - dc.n_keys).unsqueeze(-1)).squeeze(-1) + 1e-9) + rc.lam * es
    depth = deepest_P(fb["q_word"], fb["q_k"])
    B = b["pair_key"].shape[1]
    out = []
    for r in range(R):
        sl = slice(r * B, (r + 1) * B)
        kk = fb["q_k"][sl].reshape(-1); okr = ok[sl].reshape(-1); oka = ok_ans[sl].reshape(-1); dp = depth[sl].reshape(-1)
        okm = min_ok[sl].reshape(-1); ofl = on_flag[sl].reshape(-1)
        acc, ans_acc, by_depth, min_acc = {}, {}, {}, {}
        for k in range(1, dc.kmax + 1):
            mk = kk == k
            if mk.any():
                acc[k] = float(okr[mk].mean()); ans_acc[k] = float(oka[mk].mean()); min_acc[k] = float(okm[mk].mean())
        for d_ in range(0, dc.kmax + 1):                     # 0: no P; d: the deepest P is at step d
            md = dp == d_
            if md.any():
                by_depth[d_] = float(okr[md].mean())
        pir = pi[:, sl] if pi.dim() == 4 else pi[sl]
        h, c = action_accuracy(pir, oa[sl], ostates[sl])
        ha, ca, hb, cb = counterfactual_probes(pir, oa[sl], ostates[sl])
        out.append({"acc_by_k": acc, "ans_acc_by_k": ans_acc, "acc_by_Pdepth": by_depth, "acc": float(okr.mean()),
                    "min_acc_by_k": min_acc, "min_acc": float(okm.mean()),
                    "set_late": float(torch.nanmean(ofl)) if torch.isfinite(ofl).any() else None,
                    "crit_obj": float(nll[sl].mean()),
                    "act_acc": {ACTION_NAMES[a]: float(h[a] / c[a]) for a in range(4) if c[a] > 0},
                    "cf_count2_prev": float(ha / ca) if ca else None, "cf_pending_set": float(hb / cb) if cb else None})
    return out


@torch.no_grad()
def program_stats(acts, states, steps, s_fin, q_ans, osteps, mp, oracle_acts=None, oracle_states=None):
    """Program-agnostic success and program identity (rung 2: each trigger placement admits two equally short correct
    programs, because PREV needs only a pending count). min_ok: the committed route reaches the label in the oracle's
    number of moves (any shortest correct route). set_late: per query, the fraction of the committed route's SETs taken
    AT a P-leaving state (a symbol the oracle PREVs out of): program L ("late": SET where the P-step leaves, PREV at
    once) = 1; program E ("early": SET one state before, carry the count across a hop) = 0. The oracle is E under
    deferral 1 and L under deferral 0. NaN where the route takes no SET."""
    min_ok = (s_fin == q_ans) & (steps == osteps)
    B, Q, T = acts.shape
    if oracle_acts is None:
        return min_ok, torch.full(min_ok.shape, float("nan"), device=acts.device)
    S = mp["flag"].shape[1]
    pleave = torch.zeros(B, S, dtype=torch.bool, device=acts.device)
    To = oracle_acts.shape[-1]
    osym = oracle_states[:, :, :To] // PER_SYM
    m = oracle_acts == PREV
    bidx = torch.arange(B, device=acts.device).view(B, 1, 1).expand_as(osym)
    pleave[bidx[m], osym[m]] = True
    sym = states[:, :, :T] // PER_SYM
    late = pleave.gather(1, sym.reshape(B, -1)).view(B, Q, T)
    is_set = acts == SET
    n_set = is_set.sum(-1).float()
    frac = (is_set & late).sum(-1).float() / n_set.clamp_min(1)
    return min_ok, torch.where(n_set > 0, frac, torch.full_like(frac, float("nan")))


def deepest_P(q_word, q_k):
    """(B, Q): the 1-based step of the deepest P in each query's word (0 when the word is all H)."""
    W = q_word.shape[-1]
    pos = torch.arange(1, W + 1, device=q_word.device)
    return ((q_word == 1) * pos).amax(-1)


@torch.no_grad()
def evaluate(m: StackedRouter, dc, rc, tc, strata: list[dict], dev):
    """strata: per run, name -> Episodes. Returns per run a dict name -> metrics (the forced query of each episode)."""
    R = m.R
    res = [{} for _ in range(R)]
    Rd = readout(dc, rc.eps, dev)
    T = tc["T_eval"]
    for name, (k, kind) in STRATA:
        b = stack_batches([to_dict(strata[r][name], dev) for r in range(R)])
        fb = flat(b)
        mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
        pi = policy(m, b, T, chunk=tc["eval_chunk"])
        q, es, prof, alive = marginal(pi, mp, fb["q_start"], T, Rd)
        s, steps, acts, states = committed(pi, mp, fb["q_start"], T)
        oa, osteps, ostates = oracle(mp, fb["q_start"], T, dc)
        ok_route = route_ok(acts, steps, oa, osteps)
        ok_min, on_flag = program_stats(acts, states, steps, s, fb["q_ans"], osteps, mp, oa, ostates)
        # the stochastic-system reading: K routes sampled from pi per query (review 2, finding 3)
        gen = torch.Generator(device=dev).manual_seed(12345)
        samp_ok, samp_route, samp_steps = [], [], []
        for _ in range(tc["samples"]):
            s_s, st_s, a_s, _ = committed(pi, mp, fb["q_start"], T, sample=True, generator=gen)
            samp_ok.append((s_s == fb["q_ans"]).float()); samp_route.append(route_ok(a_s, st_s, oa, osteps).float())
            samp_steps.append(st_s.float())
        samp_ok, samp_route, samp_steps = (torch.stack(x, 0).mean(0) for x in (samp_ok, samp_route, samp_steps))
        ans = fb["q_ans"] - dc.n_keys
        ok_ans = (s == fb["q_ans"])
        ok_marg = (q.argmax(-1) == ans)
        is_ans = (s >= dc.n_keys) & (s < dc.DEAD)
        dist = remaining_distance(states, mp, dc)                                               # (B', Q, T+1)
        # p(HALT) and entropy at the states visited by the committed route, by remaining distance
        if pi.dim() == 3:
            idx = torch.arange(pi.shape[0], device=dev).unsqueeze(1).unsqueeze(2)
            p_halt = pi[:, :, HALT][idx, states]
            ent = -(pi * torch.log(pi + 1e-9)).sum(-1)[idx, states]
        else:
            L = states.shape[2]
            ph = F.pad(pi[:, :, 0, HALT].T, (0, L), value=1.0)[:, :L]
            en = F.pad(-(pi * torch.log(pi + 1e-9)).sum(-1)[:, :, 0].T, (0, L), value=0.0)[:, :L]
            p_halt = ph.unsqueeze(1).expand(-1, states.shape[1], -1)
            ent = en.unsqueeze(1).expand(-1, states.shape[1], -1)
        # the first deviation from the oracle route: (oracle action, taken action) at the first differing step
        diff = (acts != oa)
        first = torch.where(diff.any(-1), diff.float().argmax(-1), torch.full_like(steps, -1))
        B = b["pair_key"].shape[1]
        for r in range(R):
            sl = slice(r * B, (r + 1) * B)
            sel = torch.zeros_like(fb["q_k"][sl], dtype=torch.bool); sel[:, 0] = True      # the forced query
            pir = pi[:, sl] if pi.dim() == 4 else pi[sl]
            h, c = action_accuracy(pir, oa[sl][sel].unsqueeze(1), ostates[sl][sel].unsqueeze(1))
            ha, ca, hb, cb = counterfactual_probes(pir, oa[sl][sel].unsqueeze(1), ostates[sl][sel].unsqueeze(1))
            rec = {"n": int(sel.sum()), "route_acc": float(ok_route[sl][sel].float().mean()),
                   "min_acc": float(ok_min[sl][sel].float().mean()),
                   "samp_acc": float(samp_ok[sl][sel].mean()), "samp_route_acc": float(samp_route[sl][sel].mean()),
                   "samp_steps": float(samp_steps[sl][sel].mean()),
                   "cf_count2_prev": float(ha / ca) if ca else None, "cf_pending_set": float(hb / cb) if cb else None,
                   "acc": float(ok_ans[sl][sel].float().mean()), "marg_acc": float(ok_marg[sl][sel].float().mean()),
                   "q_steps": float(steps[sl][sel].float().mean()), "oracle_steps": float(osteps[sl][sel].float().mean()),
                   "e_steps": float(es[sl][sel].mean()),
                   "p_correct_marg": float(q[sl][sel].gather(-1, ans[sl][sel].unsqueeze(-1)).mean()),
                   "alive_at_T": float(alive[sl][sel].mean()), "end_ans": float(is_ans[sl][sel].float().mean()),
                   "end_dead": float((s[sl][sel] == dc.DEAD).float().mean()),
                   "act_acc": {ACTION_NAMES[a]: float(h[a] / c[a]) for a in range(4) if c[a] > 0}}
            hasP = (fb["q_word"][sl][sel] == 1).any(-1)
            for lab, mask in (("P", hasP), ("noP", ~hasP)):
                if mask.any():
                    rec[f"n_{lab}"] = int(mask.sum()); rec[f"route_acc_{lab}"] = float(ok_route[sl][sel][mask].float().mean())
                    rec[f"min_acc_{lab}"] = float(ok_min[sl][sel][mask].float().mean())
                    of = on_flag[sl][sel][mask]
                    rec[f"set_late_{lab}"] = float(torch.nanmean(of)) if torch.isfinite(of).any() else None
                    rec[f"acc_{lab}"] = float(ok_ans[sl][sel][mask].float().mean())
                    rec[f"marg_acc_{lab}"] = float(ok_marg[sl][sel][mask].float().mean())
                    rec[f"p_correct_marg_{lab}"] = float(q[sl][sel][mask].gather(-1, ans[sl][sel][mask].unsqueeze(-1)).mean())
                    rec[f"samp_acc_{lab}"] = float(samp_ok[sl][sel][mask].mean()); rec[f"samp_steps_{lab}"] = float(samp_steps[sl][sel][mask].mean())
                    rec[f"e_steps_{lab}"] = float(es[sl][sel][mask].mean()); rec[f"oracle_steps_{lab}"] = float(osteps[sl][sel][mask].float().mean())
            d = dist[sl][sel]; ph = p_halt[sl][sel]; en = ent[sl][sel]
            visited = torch.arange(d.shape[-1], device=dev).unsqueeze(0) <= steps[sl][sel].unsqueeze(-1)
            for lab, mask in (("d0", d == 0), ("d1", d == 1), ("d2p", d >= 2), ("dead", d < 0)):
                mm = mask & visited
                if mm.any():
                    rec[f"p_halt_{lab}"] = float(ph[mm].mean()); rec[f"entropy_{lab}"] = float(en[mm].mean())
            f = first[sl][sel]; fails = f >= 0
            if fails.any():
                dev_o = oa[sl][sel][fails, f[fails]]; dev_t = acts[sl][sel][fails, f[fails]]
                conf = {}
                for o, t_ in zip(dev_o.tolist(), dev_t.tolist()):
                    key = f"{ACTION_NAMES.get(o, 'none')}>{ACTION_NAMES.get(t_, 'none')}"
                    conf[key] = conf.get(key, 0) + 1
                rec["first_dev"] = conf
            res[r][name] = rec
    return res


def _slice_mp(mp: dict, sl: slice) -> dict:
    return {k: v[sl] for k, v in mp.items()}


def sample_per_run(pi, mp, q_start, T, K, gens, R):
    """K routes per query, each run's slice drawn from its own generator (a run's samples do not depend on its stack)."""
    Bt = q_start.shape[0]; B = Bt // R
    outs = [E.sample_routes(pi[r * B:(r + 1) * B], _slice_mp(mp, slice(r * B, (r + 1) * B)), q_start[r * B:(r + 1) * B], T, K,
                            generator=gens[r]) for r in range(R)]
    return tuple(torch.cat([o[i] for o in outs], 0) for i in range(4))


def per_run_init_(m: StackedRouter, seeds, rc, dc):
    """Re-initialize run r's parameter slice from its own seed (code review 1, finding 6): a run's init then does not
    depend on which runs share its stack. Buffers (the frozen symbol bases) are seeded by emb_seed and are shared."""
    for r, sd in enumerate(seeds):
        torch.manual_seed(1000003 * sd + 7)
        one = StackedRouter(1, rc, dc)
        with torch.no_grad():
            for (n, p), (n1, p1) in zip(m.named_parameters(), one.named_parameters()):
                assert n == n1 and p.shape[0] == m.R and p1.shape[0] == 1, n
                p[r].copy_(p1[0].to(p.device))


def estimator_loss(est, pi, mp, fb, R, tc, rc, dc, gens, Rd, V=None):
    """(surrogate per run [backward through it], reported objective per run) for one training batch."""
    if est == "pf":
        counts, Z = E.posterior_counts(E.flat_prior(mp, dc), mp, fb["q_start"], fb["q_ans"], rc.T, dc, rc.lam, rc.eps)
        per_q = E.pf_loss(pi, counts)
        n_ok = (Z > 0).view(R, -1).sum(1).clamp_min(1).float()
        sur = per_q.view(R, -1).sum(1) / n_ok
        return sur, sur.detach()
    if est == "teacher":
        oa, osteps, ostates = oracle(mp, fb["q_start"], rc.T, dc)
        per_run = teacher_loss(pi, oa, ostates).view(R, -1).mean(1)
        return per_run, per_run
    if est == "exact":
        q, es, _, _ = marginal(pi, mp, fb["q_start"], rc.T, Rd)
        ans = fb["q_ans"] - dc.n_keys
        per_run = (-torch.log(q.gather(-1, ans.unsqueeze(-1)).squeeze(-1) + 1e-9) + rc.lam * es).view(R, -1).mean(1)
        return per_run, per_run
    if est in SAMPLED:
        sym, st_k, a_k, x_k = sample_per_run(pi, mp, fb["q_start"], rc.T, tc["K"], gens, R)
        lpr = E.route_logprob(pi, a_k, x_k)
        if est == "si":
            ok, idx = E.best_correct_sample(sym, fb["q_ans"], lpr.detach(), st_k, rc.lam)
            a_t = a_k.gather(2, idx.view(*idx.shape, 1, 1).expand(*idx.shape, 1, a_k.shape[-1])).squeeze(2)
            a_t = torch.where(ok.unsqueeze(-1), a_t, torch.full_like(a_t, -1))
            per_q = teacher_loss(pi, a_t, E.replay(mp, fb["q_start"], a_t))
            sur = per_q.view(R, -1).sum(1) / ok.view(R, -1).sum(1).clamp_min(1).float()
            return sur, sur.detach()
        if est == "tb":
            lLy = torch.log(E.answer_likelihood(sym, fb["q_ans"].unsqueeze(-1), dc, rc.eps)).float()
            lp0 = E.route_logprob(E.flat_prior(mp, dc), a_k, x_k)
            sur, obj = E.tb_loss(lLy, st_k, lpr, lp0, rc.lam)
            return sur.view(R, -1).mean(1), obj.view(R, -1).mean(1)
        if est == "ac":
            lpy = E.readout_logp(sym, fb["q_ans"].unsqueeze(-1), dc, rc.eps)
            sur, obj = E.ac_loss(pi, V, a_k, x_k, lpy.exp(), rc.lam, gae=tc["gae"])
            return sur.view(R, -1).mean(1), obj.view(R, -1).mean(1)
        lpy = E.readout_logp(sym, fb["q_ans"].unsqueeze(-1), dc, rc.eps)
        fn = {"iw": E.iw_loss, "ell": E.rf_loss, "rf": E.rf_loss, "rl": E.rl_loss}[est]
        sur, obj = fn(lpy, st_k, lpr, rc.lam)
        return sur.view(R, -1).mean(1), obj.view(R, -1).mean(1)
    # ei
    found, a_b, _ = E.beam_search(pi.detach(), mp, fb["q_start"], fb["q_ans"], rc.T, tc["W"], dc, eps_mix=tc["ei_eps"], lam=rc.lam)
    a_t = torch.where(found.unsqueeze(-1), a_b, torch.full_like(a_b, -1))
    per_q = teacher_loss(pi, a_t, E.replay(mp, fb["q_start"], a_t))
    sur = per_q.view(R, -1).sum(1) / found.view(R, -1).sum(1).clamp_min(1).float()
    return sur, sur.detach()


@torch.no_grad()
def search_stats(m, dc, rc, tc, crit, est, R, dev):
    """What the estimator sees, on the held-out criterion set (fixed sampling seed, so the training stream is not
    perturbed): per run and k, over all words and over words WITH a P, the fraction of queries on which the samples /
    the beam reach the label ("hit"), contain the oracle route ("oracle_any", sampled arms), and on which the route
    that imitation would use is the oracle's ("oracle_kept": EI's beam route, SI's best correct sample)."""
    if est not in SAMPLED + ("ei",):
        return None
    b = stack_batches([crit[r] for r in range(R)]); fb = flat(b)
    mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
    pi = policy(m, b, rc.T, chunk=tc["eval_chunk"])
    oa, osteps, _ = oracle(mp, fb["q_start"], rc.T, dc)
    if est == "ei":
        found, a_b, st_b = E.beam_search(pi, mp, fb["q_start"], fb["q_ans"], rc.T, tc["W"], dc, eps_mix=tc["ei_eps"], lam=rc.lam)
        hit, orc_any, orc_kept = found, None, found & route_ok(a_b, st_b, oa, osteps)
    else:
        gens = [torch.Generator(device=dev).manual_seed(4242 + r) for r in range(R)]
        sym, st_k, a_k, x_k = sample_per_run(pi, mp, fb["q_start"], rc.T, tc["K"], gens, R)
        hit = (sym == fb["q_ans"].unsqueeze(-1)).any(-1)
        oks = torch.stack([route_ok(a_k[:, :, j], st_k[:, :, j], oa, osteps) for j in range(tc["K"])], -1)
        orc_any = oks.any(-1)
        ok, idx = E.best_correct_sample(sym, fb["q_ans"], E.route_logprob(pi, a_k, x_k), st_k, rc.lam)
        orc_kept = ok & oks.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    hasP = (fb["q_word"] == 1).any(-1)
    B = b["pair_key"].shape[1]
    out = []
    for r in range(R):
        sl = slice(r * B, (r + 1) * B)
        kk = fb["q_k"][sl]; hp = hasP[sl]
        rec = {}
        for name, x in (("hit", hit), ("oracle_any", orc_any), ("oracle_kept", orc_kept)):
            if x is None:
                continue
            xr = x[sl].float()
            rec[name] = float(xr.mean())
            rec[name + "_P"] = float(xr[hp].mean()) if hp.any() else None
            rec[name + "_by_k"] = {k: float(xr[kk == k].mean()) for k in range(1, dc.kmax + 1) if (kk == k).any()}
            rec[name + "_P_by_k"] = {k: float(xr[(kk == k) & hp].mean()) for k in range(1, dc.kmax + 1) if ((kk == k) & hp).any()}
        out.append(rec)
    return out


def run_stack(cfgs: list[dict], dev: str = "cuda", log=print) -> list[dict]:
    R = len(cfgs)
    dc, rc, tc = split_cfg({k: v for k, v in cfgs[0].items() if k not in PER_RUN_KEYS})
    seeds = [int(c.get("seed", 0)) for c in cfgs]
    lrs = torch.tensor([float(c.get("lr", tc["lr"])) for c in cfgs], device=dev)
    torch.manual_seed(1000003 * seeds[0] + sum(seeds))
    m = StackedRouter(R, rc, dc)
    if tc["per_run_init"]:
        per_run_init_(m, seeds, rc, dc)
    m = m.to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad) // R
    opt = StackedAdamW(m.parameters(), lrs, betas=(0.9, 0.95), weight_decay=tc["wd"])
    sched = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc["steps"])))
    t0 = time.time()
    pools = [EpisodePool(dc, tc["pool_n"], seed=s * 7919 + 11, dev=dev) for s in seeds]
    strata = [strata_episodes(dc, tc["eval_n"], seed=s * 7919 + 12345) for s in seeds]
    crit = [to_dict(generate(dc, tc["crit_n"], np.random.default_rng(s * 7919 + 999)), dev) for s in seeds]
    log(f"pools, strata and criterion sets generated in {time.time() - t0:.0f}s")
    Rd = readout(dc, rc.eps, dev)
    est = "teacher" if tc["teacher"] else tc["estimator"]
    assert est in ESTIMATORS, est
    rpq = tc["K"] if est in SAMPLED else (tc["W"] if est == "ei" else 0)
    recs = [{"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"), "arm": c.get("arm"),
             "data": dc.dict(), "router": rc.dict(),
             "train": {**tc, "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R, "stack_seeds": seeds, "rollouts_per_query": rpq},
             "n_params": n_params, "curve": [], "evals": [], "device": dev} for r, c in enumerate(cfgs)]
    st0 = search_stats(m, dc, rc, tc, crit, est, R, dev)                     # what the estimator sees at init (step 0)
    if st0 is not None:
        for r in range(R):
            recs[r]["search_init"] = st0[r]
    gens = [torch.Generator(device=dev).manual_seed(1000003 * sd + 17) for sd in seeds]
    run_loss = torch.zeros(R, device=dev, dtype=torch.float32)
    clip_hits = torch.zeros(R, device=dev, dtype=torch.float32)
    hits = torch.zeros(R, dc.kmax + 1, device=dev, dtype=torch.float32); cnt = torch.zeros_like(hits)
    for s in range(tc["steps"]):
        b = stack_batches([p.sample(tc["batch"]) for p in pools])
        fb = flat(b)
        mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
        est_now = "exact" if s < tc["warm_steps"] else est                   # warm start inside the exact marginal's basin
        V = None
        if est_now == "ac":
            lg, vv = m.logits_and_values(b)
            pi = torch.softmax(lg, -1).reshape(R * fb["pair_key"].shape[0] // R, -1, 4); V = vv.reshape(pi.shape[0], -1)
        else:
            pi = policy(m, b, rc.T)
        surrogate, per_run = estimator_loss(est_now, pi, mp, fb, R, tc, rc, dc, gens, Rd, V=V)
        opt.zero_grad()
        surrogate.sum().backward()
        gn = clip_grad_norm_per_run(m.parameters(), tc["grad_clip"])
        clip_hits += (gn > tc["grad_clip"]).to(clip_hits.dtype)
        opt.step(scale=sched(s))
        run_loss += per_run.detach().to(run_loss.dtype)
        with torch.no_grad():   # committed-route accuracy on the training batch, by k (memorization reference)
            _, steps, acts, _ = committed(pi.detach(), mp, fb["q_start"], rc.T)
            oa, osteps, _ = oracle(mp, fb["q_start"], rc.T, dc)
            ok = route_ok(acts, steps, oa, osteps).to(torch.float32).view(R, -1)
            kk = fb["q_k"].view(R, -1)
            hits.scatter_add_(1, kk, ok); cnt.scatter_add_(1, kk, torch.ones_like(ok))
        if (s + 1) % tc["crit_every"] == 0:
            m.eval(); cr = criterion(m, dc, rc, tc, crit, dev); ss = search_stats(m, dc, rc, tc, crit, est, R, dev); m.train()
            rl = (run_loss / tc["crit_every"]).tolist(); h, c = hits.cpu().numpy(), cnt.cpu().numpy(); gl = gn.tolist()
            cf = (clip_hits / tc["crit_every"]).tolist()
            for r in range(R):
                recs[r]["curve"].append({"step": s + 1, "loss": rl[r], "grad_norm": float(gl[r]), "clip_frac": cf[r],
                                         "acc_by_k": cr[r]["acc_by_k"], "ans_acc_by_k": cr[r]["ans_acc_by_k"],
                                         "acc_by_Pdepth": cr[r]["acc_by_Pdepth"], "acc": cr[r]["acc"], "act_acc": cr[r]["act_acc"],
                                         "cf_count2_prev": cr[r]["cf_count2_prev"], "cf_pending_set": cr[r]["cf_pending_set"],
                                         "crit_obj": cr[r]["crit_obj"], "estimator_now": est_now,
                                         "min_acc_by_k": cr[r]["min_acc_by_k"], "min_acc": cr[r]["min_acc"], "set_late": cr[r]["set_late"],
                                         "train_acc_by_k": {int(k): float(h[r, k] / c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0}})
                if ss is not None:
                    recs[r]["curve"][-1]["search"] = ss[r]
            run_loss.zero_(); hits.zero_(); cnt.zero_(); clip_hits.zero_()
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == tc["steps"]:
            m.eval()
            ev = evaluate(m, dc, rc, tc, strata, dev)
            m.train()
            for r in range(R):
                recs[r]["evals"].append({"step": s + 1, "eval": ev[r], "time": time.time() - t0})
            e0 = ev[0]; c0 = recs[0]["curve"][-1] if recs[0]["curve"] else {}
            log(f"step {s+1} [{R} runs] loss(run0) {c0.get('loss', float('nan')):.3f} crit {c0.get('acc', float('nan')):.2f} "
                f"act {c0.get('act_acc', {})} route/ans acc by stratum (run0): "
                + " ".join(f"{n}:{v['route_acc']:.2f}/{v['acc']:.2f}" for n, v in e0.items() if isinstance(v, dict))
                + f"  [{time.time()-t0:.0f}s]")
    for r in range(R):
        recs[r]["time"] = time.time() - t0
        if rc.kind == "state":
            recs[r]["head"] = {"w_norm": m.head.weight[r].norm(dim=0).tolist(), "b": m.head.bias[r, 0].tolist()}
        else:
            recs[r]["head"] = {"table": torch.softmax(m.table[r, :rc.T], -1).tolist()}
    return recs
