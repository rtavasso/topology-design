"""The learned neural baselines of rung 1 on the route data (experiments/route/DESIGN.md §7, revision 2): rung 0's
static stacks and looped block (`ptr.stacked.StackedPtrModel`, five symbol slots plus the flag direction) trained on
the same episode pools and evaluated on the same strata as the routed arms. Fixed-loop runs also get the stability
exit on their own readouts (LR), an evaluation at their training depth, and their accuracy at the ORACLE depth
(loop k, k + 1, k + #P; and at any loop) from the per-loop readouts, so that a failure is attributable to the block
or to the exit rule (review 1, F). Records are in the routed arms' format (evals by stratum) plus a held-out
criterion curve every 100 steps (acc_by_k, acc_by_Pdepth; D97) and rung 0's training-curve fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields, replace

import numpy as np
import torch
import torch.nn.functional as F

from ..ptr.model import ModelConfig, kl_exit
from ..ptr.stacked import StackedPtrModel, StackedAdamW, clip_grad_norm_per_run
from ..ptr.train import KL_EXIT_THRESHOLDS
from .data import RouteConfig, generate
from .pool import EpisodePool, stack_batches, strata_episodes, to_dict, STRATA

DATA_KEYS = {f.name for f in fields(RouteConfig)}
MODEL_KEYS = {f.name for f in fields(ModelConfig)}
TRAIN_DEFAULTS = dict(steps=6000, batch=64, lr=1e-3, warmup=500, seed=0, eval_every=2000, T_eval=40, eval_n=250,
                      pool_n=60000, wd=0.01, grad_clip=1.0, crit_n=256, crit_every=100)
PER_RUN_KEYS = ("seed", "lr")


def split_cfg(cfg):
    dc = RouteConfig(**{k: v for k, v in cfg.items() if k in DATA_KEYS})
    mc = ModelConfig(**{"slots": 5, "d": 344, **{k: v for k, v in cfg.items() if k in MODEL_KEYS}})
    tc = {**TRAIN_DEFAULTS, **{k: v for k, v in cfg.items() if k in TRAIN_DEFAULTS}}
    return dc, mc, tc


def deepest_P(q_word):
    """(..., Q): the 1-based step of the deepest P in each word (0 for all H)."""
    pos = torch.arange(1, q_word.shape[-1] + 1, device=q_word.device)
    return ((q_word == 1) * pos).amax(-1)


@torch.no_grad()
def per_loop_ok(m, out):
    """(T, R, B, Q) correctness of the readout after every loop (fixed-loop arms)."""
    return torch.stack([m.readout(hq).argmax(-1) == out["targets"] for hq in out["hs"]], 0)


@torch.no_grad()
def evaluate(m: StackedPtrModel, dc, mc, tc, strata, dev, T=None, own_block=False):
    m.eval()
    R = m.R
    T = tc["T_eval"] if T is None else T
    res = [{} for _ in range(R)]
    for name, (k, kind) in STRATA:
        b = stack_batches([to_dict(strata[r][name], dev) for r in range(R)])
        qw = b.pop("q_word")
        out = m(b, T=None if mc.kind == "static" else T, mode="eval")
        lg, tg = out["logits"], out["targets"]
        ok = lg.argmax(-1) == tg
        qsteps = out["steps"][:, :, out["qpos"]]
        lr_stats = own = loop_ok = None
        if mc.kind == "looped" and mc.halting == "fixed":
            loop_ok = per_loop_ok(m, out)                                                   # (T, R, B, Q)
            psl = torch.stack([m.readout(hq) for hq in out["hs"]], 0)
            lr_stats = kl_exit(psl, tg, KL_EXIT_THRESHOLDS)
        elif mc.kind == "looped" and own_block:
            m.mc = replace(mc, halting="fixed")
            try:
                out_f = m(b, T=T, mode="eval")
            finally:
                m.mc = mc
            own = kl_exit(torch.stack([m.readout(hq) for hq in out_f["hs"]], 0), tg, KL_EXIT_THRESHOLDS)
            loop_ok = per_loop_ok(m, out_f)
        hasP = (qw[:, :, 0] == 1).any(-1)                                                   # (R, B) forced query
        nP = (qw[:, :, 0] == 1).sum(-1)
        kk = b["q_k"][:, :, 0]
        for r in range(R):
            rec = {"n": int(ok.shape[1]), "acc": float(ok[r, :, 0].float().mean()), "q_steps": float(qsteps[r, :, 0].mean()),
                   "max_steps": float(out["steps"][r].max())}
            for lab, mask in (("P", hasP[r]), ("noP", ~hasP[r])):
                if mask.any():
                    rec[f"n_{lab}"] = int(mask.sum()); rec[f"acc_{lab}"] = float(ok[r, :, 0][mask].float().mean())
            if lr_stats is not None:
                rec["kl_exit"] = {str(thr): {"acc": float(okk[r, :, 0].float().mean()), "q_steps": float(stp[r, :, 0].mean())}
                                  for thr, (okk, stp) in lr_stats.items()}
            if own is not None:
                rec["kl_exit_own"] = {str(thr): {"acc": float(okk[r, :, 0].float().mean()), "q_steps": float(stp[r, :, 0].mean())}
                                      for thr, (okk, stp) in own.items()}
            if loop_ok is not None:                                                        # accuracy at the oracle depth
                lo = loop_ok[:, r, :, 0]                                                    # (T, B)
                Tl = lo.shape[0]
                ar = torch.arange(lo.shape[1], device=dev)
                at = lambda t: float(lo[t.clamp(0, Tl - 1), ar][t < Tl].float().mean()) if bool((t < Tl).any()) else float("nan")
                rec["acc_at_k"] = at(kk[r] - 1); rec["acc_at_k1"] = at(kk[r]); rec["acc_at_kP"] = at(kk[r] + nP[r] - 1)
                rec["acc_any"] = float(lo.any(0).float().mean())
                for lab, mask in (("P", hasP[r]), ("noP", ~hasP[r])):          # the oracle-depth reading on P-words (review 2)
                    if mask.any():
                        lom = lo[:, mask]; arm = torch.arange(int(mask.sum()), device=dev)
                        atm = lambda t: float(lom[t.clamp(0, Tl - 1), arm][t < Tl].float().mean()) if bool((t < Tl).any()) else float("nan")
                        rec[f"acc_at_k_{lab}"] = atm(kk[r][mask] - 1); rec[f"acc_at_kP_{lab}"] = atm(kk[r][mask] + nP[r][mask] - 1)
                        rec[f"acc_any_{lab}"] = float(lom.any(0).float().mean())
                rec["acc_by_loop"] = lo.float().mean(1).tolist()
            res[r][name] = rec
    m.train()
    return res


@torch.no_grad()
def criterion(m: StackedPtrModel, dc, mc, tc, crit, dev):
    """Held-out criterion set per run (training distribution): answer accuracy by k and by deepest-P depth."""
    m.eval()
    R = m.R
    b = stack_batches([crit[r] for r in range(R)])
    qw = b.pop("q_word")
    out = m(b, mode="eval")
    ok = (out["logits"].argmax(-1) == out["targets"]).float()                                # (R, B, Q)
    kk = b["q_k"]; dp = deepest_P(qw)
    res = []
    for r in range(R):
        o, k_, d_ = ok[r].reshape(-1), kk[r].reshape(-1), dp[r].reshape(-1)
        res.append({"acc": float(o.mean()),
                    "acc_by_k": {k: float(o[k_ == k].mean()) for k in range(1, dc.kmax + 1) if (k_ == k).any()},
                    "acc_by_Pdepth": {d: float(o[d_ == d].mean()) for d in range(0, dc.kmax + 1) if (d_ == d).any()}})
    m.train()
    return res


def run_stack(cfgs: list[dict], dev: str = "cuda", log=print, amp: bool | None = None) -> list[dict]:
    R = len(cfgs)
    dc, mc, tc = split_cfg({k: v for k, v in cfgs[0].items() if k not in PER_RUN_KEYS})
    seeds = [int(c.get("seed", 0)) for c in cfgs]
    lrs = torch.tensor([float(c.get("lr", tc["lr"])) for c in cfgs], device=dev)
    torch.manual_seed(1000003 * seeds[0] + sum(seeds))
    m = StackedPtrModel(R, mc, dc).to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad) // R
    opt = StackedAdamW(m.parameters(), lrs, betas=(0.9, 0.95), weight_decay=tc["wd"])
    sched = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc["steps"])))
    t0 = time.time()
    pools = [EpisodePool(dc, tc["pool_n"], seed=s * 7919 + 11, dev=dev) for s in seeds]
    strata = [strata_episodes(dc, tc["eval_n"], seed=s * 7919 + 12345) for s in seeds]
    crit = [to_dict(generate(dc, tc["crit_n"], np.random.default_rng(s * 7919 + 999)), dev) for s in seeds]
    log(f"pools, strata and criterion sets generated in {time.time() - t0:.0f}s")
    use_amp = dev.startswith("cuda") if amp is None else amp
    loss_scale, clean = (2.0 ** 14 if use_amp else 1.0), 0
    recs = [{"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"), "arm": c.get("arm"),
             "data": dc.dict(), "model": mc.dict(), "train": {**tc, "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R, "amp": use_amp},
             "n_params": n_params, "curve": [], "evals": [], "device": dev} for r, c in enumerate(cfgs)]
    run_loss = torch.zeros(R, device=dev, dtype=torch.float32)
    hits = torch.zeros(R, dc.kmax + 1, device=dev, dtype=torch.float32); cnt = torch.zeros_like(hits)
    for s in range(tc["steps"]):
        b = stack_batches([p.sample(tc["batch"]) for p in pools])
        b.pop("q_word")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = m(b, mode="train")
            loss, per_run, extra = m.loss(out)
        opt.zero_grad()
        (loss * loss_scale).backward()
        if loss_scale != 1.0:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / loss_scale)
        gn = clip_grad_norm_per_run(m.parameters(), tc["grad_clip"])
        finite = torch.isfinite(gn) & torch.isfinite(per_run)
        opt.step(scale=sched(s), active=finite if use_amp else None)
        if use_amp:
            if bool((~finite).any()):
                loss_scale, clean = max(loss_scale / 2, 1.0), 0
            else:
                clean += 1
                if clean >= 1000 and loss_scale < 2.0 ** 16:
                    loss_scale, clean = loss_scale * 2, 0
        run_loss += torch.nan_to_num(per_run, nan=0.0, posinf=0.0, neginf=0.0).to(run_loss.dtype)
        with torch.no_grad():
            ok = (out["logits"].argmax(-1) == out["targets"]).to(torch.float32).reshape(R, -1)
            kk = b["q_k"].reshape(R, -1)
            hits.scatter_add_(1, kk, ok); cnt.scatter_add_(1, kk, torch.ones_like(ok))
        if (s + 1) % tc["crit_every"] == 0:
            cr = criterion(m, dc, mc, tc, crit, dev)
            rl = (run_loss / tc["crit_every"]).tolist(); h, c = hits.cpu().numpy(), cnt.cpu().numpy(); gl = gn.tolist()
            for r in range(R):
                recs[r]["curve"].append({"step": s + 1, "loss": rl[r], "grad_norm": float(gl[r]), "loss_scale": loss_scale,
                                         "acc": cr[r]["acc"], "acc_by_k": cr[r]["acc_by_k"], "acc_by_Pdepth": cr[r]["acc_by_Pdepth"],
                                         "train_acc_by_k": {int(k): float(h[r, k] / c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0}})
            run_loss.zero_(); hits.zero_(); cnt.zero_()
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == tc["steps"]:
            final = s + 1 == tc["steps"]
            ev = evaluate(m, dc, mc, tc, strata, dev, own_block=final)
            for r in range(R):
                recs[r]["evals"].append({"step": s + 1, "eval": ev[r], "time": time.time() - t0})
            if final and mc.kind == "looped" and mc.halting == "fixed":
                ev_t = evaluate(m, dc, mc, tc, strata, dev, T=mc.T)
                for r in range(R):
                    recs[r]["eval_T_train"] = ev_t[r]
            e0 = ev[0]
            log(f"step {s+1} [{R} runs] loss(run0) {(recs[0]['curve'][-1]['loss'] if recs[0]['curve'] else float('nan')):.3f} acc by stratum (run0): "
                + " ".join(f"{n}:{v['acc']:.2f}" for n, v in e0.items()) + f"  [{time.time()-t0:.0f}s]")
    for r in range(R):
        recs[r]["time"] = time.time() - t0
    return recs
