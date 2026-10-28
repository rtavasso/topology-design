"""Training of the routed arms of rung 1 (experiments/route/DESIGN.md §§4, 7): a stack of runs of one arm trained
together, each on its own episode pool, by the exact route marginal (or by teacher forcing on the oracle route).

A run config is a flat dict: data fields (RouteConfig), router fields (RouterConfig) and training fields
(TRAIN_DEFAULTS). run_stack(cfgs, dev) returns one record per run: loss and training-batch route accuracy by k
(100-step windows), evaluations by stratum (route / answer / marginal accuracy, steps, halting-by-distance, the
remaining-distance probe), and the router's head parameters.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from ..ptr.stacked import StackedAdamW, clip_grad_norm_per_run
from .data import RouteConfig
from .exec import marginal, committed, oracle_route_ok, readout, tables, maps, remaining_distance, HOP, PREV, HALT
from .pool import EpisodePool, stack_batches, strata_episodes, to_dict, STRATA
from .router import StackedRouter, RouterConfig

DATA_KEYS = {f.name for f in fields(RouteConfig)}
ROUTER_KEYS = {f.name for f in fields(RouterConfig)}
TRAIN_DEFAULTS = dict(steps=6000, batch=64, lr=1e-3, warmup=500, seed=0, eval_every=2000, T_eval=40, eval_n=250,
                      pool_n=60000, wd=0.01, grad_clip=1.0, teacher=False, probe_n=2000)
PER_RUN_KEYS = ("seed", "lr")


def split_cfg(cfg):
    dc = RouteConfig(**{k: v for k, v in cfg.items() if k in DATA_KEYS})
    rc = RouterConfig(**{k: v for k, v in cfg.items() if k in ROUTER_KEYS})
    tc = {**TRAIN_DEFAULTS, **{k: v for k, v in cfg.items() if k in TRAIN_DEFAULTS}}
    return dc, rc, tc


def flat(b: dict):
    """(R, B, ...) -> (R*B, ...)."""
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}


def policy(m: StackedRouter, b: dict, T: int):
    """pi for the exec functions: (R*B, S, 3) for the state router; (T, R*B, 1, 3) for the index router."""
    R, B = b["pair_key"].shape[:2]
    lg = m.logits(b, T=T)
    if m.rc.kind == "index":
        return torch.softmax(lg, -1).unsqueeze(2).expand(T, R, B, 1, 3).reshape(T, R * B, 1, 3)
    return torch.softmax(lg, -1).reshape(R * B, -1, 3)


def teacher_loss(pi, q_start, q_word, q_k, hop, prev, T):
    """-sum_t log pi(a_t* | s_t*) along the oracle route, HALT at the end; per query."""
    Bq, Q = q_start.shape
    s = q_start.clone()
    ll = torch.zeros(Bq, Q, device=s.device)
    ar = torch.arange(Bq, device=s.device).unsqueeze(1)
    for t in range(T):
        a = torch.where(t < q_k, q_word[:, :, t] if t < q_word.shape[-1] else torch.full_like(q_k, -1), torch.full_like(q_k, HALT))
        valid = t <= q_k
        p = pi[t] if pi.dim() == 4 else pi
        lp = torch.log((p[:, 0, :].unsqueeze(1).expand(Bq, Q, 3) if p.shape[1] == 1 else p[ar, s]) + 1e-9)
        ll = ll + torch.where(valid, lp.gather(-1, a.clamp_min(0).unsqueeze(-1)).squeeze(-1), torch.zeros_like(ll))
        nxt = torch.where(a == HOP, hop.gather(1, s), torch.where(a == PREV, prev.gather(1, s), s))
        s = torch.where(valid & (a != HALT), nxt, s)
    return -ll


@torch.no_grad()
def evaluate(m: StackedRouter, dc, rc, tc, strata: list[dict], dev):
    """strata: per run, name -> Episodes. Returns per run a dict name -> metrics, plus the residual study and probe."""
    R = m.R
    res = [{} for _ in range(R)]
    Rd = readout(dc, rc.eps, dev)
    pf, pl = [[] for _ in range(R)], [[] for _ in range(R)]
    for name, (k, kind) in STRATA:
        b = stack_batches([to_dict(strata[r][name], dev) for r in range(R)])
        fb = flat(b)
        mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], dc)
        pi = policy(m, b, tc["T_eval"])
        q, es, prof, alive = marginal(pi, mp["hop"], mp["prev"], fb["q_start"], tc["T_eval"], Rd)
        s, steps, acts, states = committed(pi, mp["hop"], mp["prev"], fb["q_start"], tc["T_eval"], Rd)
        ok_route = oracle_route_ok(acts, steps, fb["q_word"], fb["q_k"])
        ans = fb["q_ans"] - dc.n_keys
        ok_ans = (s == fb["q_ans"])
        ok_marg = (q.argmax(-1) == ans)
        dist = remaining_distance(states, mp["hop"], mp["prev"], fb["q_word"], fb["q_k"], dc)          # (B', Q, T+1)
        # the halting probability at the states visited by the committed route, by remaining distance
        if pi.dim() == 3:
            p_halt = pi[:, :, HALT][torch.arange(pi.shape[0], device=dev).unsqueeze(1).unsqueeze(2), states]   # (B', Q, T+1)
            ent = -(pi * torch.log(pi + 1e-9)).sum(-1)[torch.arange(pi.shape[0], device=dev).unsqueeze(1).unsqueeze(2), states]
        else:                                                    # step-indexed policy: the same at every state
            L = states.shape[2]
            ph = pi[:, :, 0, HALT].T                                                      # (B', T)
            ph = F.pad(ph, (0, L - ph.shape[1]), value=1.0)[:, :L]
            en = -(pi * torch.log(pi + 1e-9)).sum(-1)[:, :, 0].T
            en = F.pad(en, (0, L - en.shape[1]), value=0.0)[:, :L]
            p_halt = ph.unsqueeze(1).expand(-1, states.shape[1], -1)
            ent = en.unsqueeze(1).expand(-1, states.shape[1], -1)
        B = b["pair_key"].shape[1]
        probe_here = rc.kind == "state" and name in ("len16", "len32", "in8", "alt32")
        feats_all = m.features(b) if probe_here else None                                   # (R, B, S, d)
        for r in range(R):
            sl = slice(r * B, (r + 1) * B)
            sel = torch.zeros_like(fb["q_k"][sl], dtype=torch.bool); sel[:, 0] = True      # the forced query
            rec = {"n": int(sel.sum()), "route_acc": float(ok_route[sl][sel].float().mean()),
                   "acc": float(ok_ans[sl][sel].float().mean()), "marg_acc": float(ok_marg[sl][sel].float().mean()),
                   "q_steps": float(steps[sl][sel].float().mean()), "e_steps": float(es[sl][sel].mean()),
                   "p_correct_marg": float(q[sl][sel].gather(-1, ans[sl][sel].unsqueeze(-1)).mean()),
                   "alive_at_T": float(alive[sl][sel].mean())}
            # residual study: p(HALT) and entropy by remaining distance along the committed route (distance 0, 1, >= 2)
            d = dist[sl][sel]; ph = p_halt[sl][sel]; en = ent[sl][sel]
            visited = torch.arange(d.shape[-1], device=dev).unsqueeze(0) <= steps[sl][sel].unsqueeze(-1)
            for lab, mask in (("d0", d == 0), ("d1", d == 1), ("d2p", d >= 2), ("dead", d < 0)):
                mm = mask & visited
                if mm.any():
                    rec[f"p_halt_{lab}"] = float(ph[mm].mean()); rec[f"entropy_{lab}"] = float(en[mm].mean())
            res[r][name] = rec
            if probe_here:                                        # probe features at the committed route's states
                st_or = states[sl][sel]; dd = dist[sl][sel]
                feats = feats_all[r]                                                               # (B, S, d)
                idx = torch.arange(st_or.shape[0], device=dev).unsqueeze(1)
                fsel = feats[idx, st_or[:, :min(k + 1, st_or.shape[1])]]                          # (n, k+1, d)
                dsel = dd[:, :min(k + 1, dd.shape[1])]
                keep = dsel >= 0
                pf[r].append(fsel[keep][: tc["probe_n"]]); pl[r].append(dsel[keep][: tc["probe_n"]].float())
    for r in range(R):
        if pf[r]:
            X = torch.cat(pf[r]); y = torch.cat(pl[r])
            res[r]["probe_r2"] = ridge_r2(X, y)
    return res


@torch.no_grad()
def ridge_r2(X, y, lam=1.0):
    n = X.shape[0]
    perm = torch.randperm(n, device=X.device)
    tr, te = perm[: n // 2], perm[n // 2:]
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    A = Xb[tr].T @ Xb[tr] + lam * torch.eye(Xb.shape[1], device=X.device)
    w = torch.linalg.solve(A, Xb[tr].T @ y[tr])
    pred = Xb[te] @ w
    ss = ((y[te] - y[te].mean()) ** 2).sum()
    return float(1 - ((y[te] - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def run_stack(cfgs: list[dict], dev: str = "cuda", log=print) -> list[dict]:
    R = len(cfgs)
    dc, rc, tc = split_cfg({k: v for k, v in cfgs[0].items() if k not in PER_RUN_KEYS})
    seeds = [int(c.get("seed", 0)) for c in cfgs]
    lrs = torch.tensor([float(c.get("lr", tc["lr"])) for c in cfgs], device=dev)
    torch.manual_seed(1000003 * seeds[0] + sum(seeds))
    m = StackedRouter(R, rc, dc).to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad) // R
    opt = StackedAdamW(m.parameters(), lrs, betas=(0.9, 0.95), weight_decay=tc["wd"])
    sched = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc["steps"])))
    t0 = time.time()
    pools = [EpisodePool(dc, tc["pool_n"], seed=s * 7919 + 11, dev=dev) for s in seeds]
    strata = [strata_episodes(dc, tc["eval_n"], seed=s * 7919 + 12345) for s in seeds]
    log(f"pools and strata generated in {time.time() - t0:.0f}s")
    Rd = readout(dc, rc.eps, dev)
    recs = [{"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"), "arm": c.get("arm"),
             "data": dc.dict(), "router": rc.dict(), "train": {**tc, "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R},
             "n_params": n_params, "curve": [], "evals": [], "device": dev} for r, c in enumerate(cfgs)]
    run_loss = torch.zeros(R, device=dev)
    hits = torch.zeros(R, dc.kmax + 1, device=dev); cnt = torch.zeros(R, dc.kmax + 1, device=dev)
    for s in range(tc["steps"]):
        b = stack_batches([p.sample(tc["batch"]) for p in pools])
        fb = flat(b)
        mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], dc)
        pi = policy(m, b, rc.T)
        if tc["teacher"]:
            per_q = teacher_loss(pi, fb["q_start"], fb["q_word"], fb["q_k"], mp["hop"], mp["prev"], rc.T)
        else:
            q, es, _, _ = marginal(pi, mp["hop"], mp["prev"], fb["q_start"], rc.T, Rd)
            ans = fb["q_ans"] - dc.n_keys
            per_q = -torch.log(q.gather(-1, ans.unsqueeze(-1)).squeeze(-1) + 1e-9) + rc.lam * es
        per_run = per_q.view(R, -1).mean(1)
        opt.zero_grad()
        per_run.sum().backward()
        gn = clip_grad_norm_per_run(m.parameters(), tc["grad_clip"])
        opt.step(scale=sched(s))
        run_loss += per_run.detach()
        with torch.no_grad():   # committed-route accuracy on the training batch, by k
            _, steps, acts, _ = committed(pi.detach(), mp["hop"], mp["prev"], fb["q_start"], rc.T, Rd)
            ok = oracle_route_ok(acts, steps, fb["q_word"], fb["q_k"]).float().view(R, -1)
            kk = fb["q_k"].view(R, -1)
            hits.scatter_add_(1, kk, ok); cnt.scatter_add_(1, kk, torch.ones_like(ok))
        if (s + 1) % 100 == 0:
            rl = (run_loss / 100).tolist(); h, c = hits.cpu().numpy(), cnt.cpu().numpy(); gl = gn.tolist()
            for r in range(R):
                recs[r]["curve"].append({"step": s + 1, "loss": rl[r], "grad_norm": float(gl[r]),
                                         "acc_by_k": {int(k): float(h[r, k] / c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0},
                                         "n_by_k": {int(k): int(c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0}})
            run_loss.zero_(); hits.zero_(); cnt.zero_()
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == tc["steps"]:
            m.eval()
            ev = evaluate(m, dc, rc, tc, strata, dev)
            m.train()
            for r in range(R):
                recs[r]["evals"].append({"step": s + 1, "eval": ev[r], "time": time.time() - t0})
            e0 = ev[0]
            log(f"step {s+1} [{R} runs] loss(run0) {(recs[0]['curve'][-1]['loss'] if recs[0]['curve'] else float('nan')):.3f} route/ans acc by stratum (run0): "
                + " ".join(f"{n}:{v['route_acc']:.2f}/{v['acc']:.2f}" for n, v in e0.items() if isinstance(v, dict))
                + (f" probe {e0.get('probe_r2', float('nan')):.2f}" if 'probe_r2' in e0 else "") + f"  [{time.time()-t0:.0f}s]")
    for r in range(R):
        recs[r]["time"] = time.time() - t0
        if rc.kind == "state":
            recs[r]["head"] = {"w_norm": m.head.weight[r].norm(dim=0).tolist(), "b": m.head.bias[r, 0].tolist()}
        else:
            recs[r]["head"] = {"table": torch.softmax(m.table[r, :rc.T], -1).tolist()}
    return recs
