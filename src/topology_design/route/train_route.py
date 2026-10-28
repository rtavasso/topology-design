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
from .exec import marginal, committed, oracle, route_ok, readout, maps, remaining_distance, HOP, PREV, SET, HALT
from .pool import EpisodePool, stack_batches, strata_episodes, to_dict, STRATA
from .router import StackedRouter, RouterConfig

DATA_KEYS = {f.name for f in fields(RouteConfig)}
ROUTER_KEYS = {f.name for f in fields(RouterConfig)}
TRAIN_DEFAULTS = dict(steps=6000, batch=64, lr=1e-3, warmup=500, seed=0, eval_every=2000, T_eval=64, eval_n=250,
                      pool_n=60000, wd=0.01, grad_clip=1.0, teacher=False, crit_n=256, crit_every=100, eval_chunk=50)
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
    s_fin, steps, acts, _ = committed(pi, mp, fb["q_start"], rc.T)
    oa, osteps, ostates = oracle(mp, fb["q_start"], rc.T, dc)
    ok = route_ok(acts, steps, oa, osteps).float()
    ok_ans = (s_fin == fb["q_ans"]).float()
    depth = deepest_P(fb["q_word"], fb["q_k"])
    B = b["pair_key"].shape[1]
    out = []
    for r in range(R):
        sl = slice(r * B, (r + 1) * B)
        kk = fb["q_k"][sl].reshape(-1); okr = ok[sl].reshape(-1); oka = ok_ans[sl].reshape(-1); dp = depth[sl].reshape(-1)
        acc, ans_acc, by_depth = {}, {}, {}
        for k in range(1, dc.kmax + 1):
            mk = kk == k
            if mk.any():
                acc[k] = float(okr[mk].mean()); ans_acc[k] = float(oka[mk].mean())
        for d_ in range(0, dc.kmax + 1):                     # 0: no P; d: the deepest P is at step d
            md = dp == d_
            if md.any():
                by_depth[d_] = float(okr[md].mean())
        pir = pi[:, sl] if pi.dim() == 4 else pi[sl]
        h, c = action_accuracy(pir, oa[sl], ostates[sl])
        out.append({"acc_by_k": acc, "ans_acc_by_k": ans_acc, "acc_by_Pdepth": by_depth, "acc": float(okr.mean()),
                    "act_acc": {ACTION_NAMES[a]: float(h[a] / c[a]) for a in range(4) if c[a] > 0}})
    return out


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
            rec = {"n": int(sel.sum()), "route_acc": float(ok_route[sl][sel].float().mean()),
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
                    rec[f"acc_{lab}"] = float(ok_ans[sl][sel][mask].float().mean())
                    rec[f"marg_acc_{lab}"] = float(ok_marg[sl][sel][mask].float().mean())
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
    crit = [to_dict(generate(dc, tc["crit_n"], np.random.default_rng(s * 7919 + 999)), dev) for s in seeds]
    log(f"pools, strata and criterion sets generated in {time.time() - t0:.0f}s")
    Rd = readout(dc, rc.eps, dev)
    recs = [{"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"), "arm": c.get("arm"),
             "data": dc.dict(), "router": rc.dict(), "train": {**tc, "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R},
             "n_params": n_params, "curve": [], "evals": [], "device": dev} for r, c in enumerate(cfgs)]
    run_loss = torch.zeros(R, device=dev, dtype=torch.float32)
    hits = torch.zeros(R, dc.kmax + 1, device=dev, dtype=torch.float32); cnt = torch.zeros_like(hits)
    for s in range(tc["steps"]):
        b = stack_batches([p.sample(tc["batch"]) for p in pools])
        fb = flat(b)
        mp = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
        pi = policy(m, b, rc.T)
        if tc["teacher"]:
            oa, osteps, ostates = oracle(mp, fb["q_start"], rc.T, dc)
            per_q = teacher_loss(pi, oa, ostates)
        else:
            q, es, _, _ = marginal(pi, mp, fb["q_start"], rc.T, Rd)
            ans = fb["q_ans"] - dc.n_keys
            per_q = -torch.log(q.gather(-1, ans.unsqueeze(-1)).squeeze(-1) + 1e-9) + rc.lam * es
        per_run = per_q.view(R, -1).mean(1)
        opt.zero_grad()
        per_run.sum().backward()
        gn = clip_grad_norm_per_run(m.parameters(), tc["grad_clip"])
        opt.step(scale=sched(s))
        run_loss += per_run.detach().to(run_loss.dtype)
        with torch.no_grad():   # committed-route accuracy on the training batch, by k (memorization reference)
            _, steps, acts, _ = committed(pi.detach(), mp, fb["q_start"], rc.T)
            oa, osteps, _ = oracle(mp, fb["q_start"], rc.T, dc)
            ok = route_ok(acts, steps, oa, osteps).to(torch.float32).view(R, -1)
            kk = fb["q_k"].view(R, -1)
            hits.scatter_add_(1, kk, ok); cnt.scatter_add_(1, kk, torch.ones_like(ok))
        if (s + 1) % tc["crit_every"] == 0:
            m.eval(); cr = criterion(m, dc, rc, tc, crit, dev); m.train()
            rl = (run_loss / tc["crit_every"]).tolist(); h, c = hits.cpu().numpy(), cnt.cpu().numpy(); gl = gn.tolist()
            for r in range(R):
                recs[r]["curve"].append({"step": s + 1, "loss": rl[r], "grad_norm": float(gl[r]),
                                         "acc_by_k": cr[r]["acc_by_k"], "ans_acc_by_k": cr[r]["ans_acc_by_k"],
                                         "acc_by_Pdepth": cr[r]["acc_by_Pdepth"], "acc": cr[r]["acc"], "act_acc": cr[r]["act_acc"],
                                         "train_acc_by_k": {int(k): float(h[r, k] / c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0}})
            run_loss.zero_(); hits.zero_(); cnt.zero_()
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
