"""The learned neural baselines of rung 1 on the route data (experiments/route/DESIGN.md §7): rung 0's static stacks
and looped block (`ptr.stacked.StackedPtrModel`, five symbol slots) trained on the same episode pools and evaluated
on the same strata as the routed arms. Fixed-loop runs also get the stability exit on their own readouts (LR) and
an evaluation at their training depth. Records are in the routed arms' format (evals by stratum) plus rung 0's
training-curve fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields, replace

import torch
import torch.nn.functional as F

from ..ptr.model import ModelConfig, kl_exit
from ..ptr.stacked import StackedPtrModel, StackedAdamW, clip_grad_norm_per_run
from ..ptr.train import KL_EXIT_THRESHOLDS
from .data import RouteConfig
from .pool import EpisodePool, stack_batches, strata_episodes, to_dict, STRATA

DATA_KEYS = {f.name for f in fields(RouteConfig)}
MODEL_KEYS = {f.name for f in fields(ModelConfig)}
TRAIN_DEFAULTS = dict(steps=6000, batch=64, lr=1e-3, warmup=500, seed=0, eval_every=2000, T_eval=40, eval_n=250,
                      pool_n=60000, wd=0.01, grad_clip=1.0)
PER_RUN_KEYS = ("seed", "lr")


def split_cfg(cfg):
    dc = RouteConfig(**{k: v for k, v in cfg.items() if k in DATA_KEYS})
    mc = ModelConfig(**{"slots": 5, "d": 344, **{k: v for k, v in cfg.items() if k in MODEL_KEYS}})
    tc = {**TRAIN_DEFAULTS, **{k: v for k, v in cfg.items() if k in TRAIN_DEFAULTS}}
    return dc, mc, tc


@torch.no_grad()
def evaluate(m: StackedPtrModel, dc, mc, tc, strata, dev, T=None, own_block=False):
    m.eval()
    R = m.R
    T = tc["T_eval"] if T is None else T
    res = [{} for _ in range(R)]
    for name, (k, kind) in STRATA:
        b = stack_batches([to_dict(strata[r][name], dev) for r in range(R)])
        b = {kk: v for kk, v in b.items() if kk != "q_word"}
        out = m(b, T=None if mc.kind == "static" else T, mode="eval")
        lg, tg = out["logits"], out["targets"]
        ok = lg.argmax(-1) == tg
        qsteps = out["steps"][:, :, out["qpos"]]
        lr_stats = own = None
        if mc.kind == "looped" and mc.halting == "fixed":
            psl = torch.stack([m.readout(hq) for hq in out["hs"]], 0)
            lr_stats = kl_exit(psl, tg, KL_EXIT_THRESHOLDS)
        elif mc.kind == "looped" and own_block:
            m.mc = replace(mc, halting="fixed")
            try:
                out_f = m(b, T=T, mode="eval")
            finally:
                m.mc = mc
            own = kl_exit(torch.stack([m.readout(hq) for hq in out_f["hs"]], 0), tg, KL_EXIT_THRESHOLDS)
        for r in range(R):
            rec = {"n": int(ok.shape[1]), "acc": float(ok[r, :, 0].float().mean()), "q_steps": float(qsteps[r, :, 0].mean()),
                   "max_steps": float(out["steps"][r].max())}
            if lr_stats is not None:
                rec["kl_exit"] = {str(thr): {"acc": float(okk[r, :, 0].float().mean()), "q_steps": float(stp[r, :, 0].mean())}
                                  for thr, (okk, stp) in lr_stats.items()}
            if own is not None:
                rec["kl_exit_own"] = {str(thr): {"acc": float(okk[r, :, 0].float().mean()), "q_steps": float(stp[r, :, 0].mean())}
                                      for thr, (okk, stp) in own.items()}
            res[r][name] = rec
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
    log(f"pools and strata generated in {time.time() - t0:.0f}s")
    use_amp = dev.startswith("cuda") if amp is None else amp
    loss_scale, clean = (2.0 ** 14 if use_amp else 1.0), 0
    recs = [{"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"), "arm": c.get("arm"),
             "data": dc.dict(), "model": mc.dict(), "train": {**tc, "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R, "amp": use_amp},
             "n_params": n_params, "curve": [], "evals": [], "device": dev} for r, c in enumerate(cfgs)]
    run_loss = torch.zeros(R, device=dev)
    hits = torch.zeros(R, dc.kmax + 1, device=dev); cnt = torch.zeros(R, dc.kmax + 1, device=dev)
    # a per-(circuit, depth) curve: accuracy on queries whose word has a P at step d (d = 1..kmax) and on all-H words
    hitsP = torch.zeros(R, dc.kmax + 1, device=dev); cntP = torch.zeros(R, dc.kmax + 1, device=dev)
    for s in range(tc["steps"]):
        b = stack_batches([p.sample(tc["batch"]) for p in pools])
        qw = b.pop("q_word")
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
        run_loss += torch.nan_to_num(per_run, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            ok = (out["logits"].argmax(-1) == out["targets"]).float().reshape(R, -1)
            kk = b["q_k"].reshape(R, -1)
            hits.scatter_add_(1, kk, ok); cnt.scatter_add_(1, kk, torch.ones_like(ok))
            hasP = (qw == 1)                                                            # (R, B, Q, kw)
            for d in range(1, min(dc.kmax, qw.shape[-1]) + 1):
                sel = hasP[..., d - 1].reshape(R, -1)
                hitsP[:, d] += (ok * sel).sum(1); cntP[:, d] += sel.sum(1)
            allH = ~hasP.any(-1).reshape(R, -1)
            hitsP[:, 0] += (ok * allH).sum(1); cntP[:, 0] += allH.sum(1)
        if (s + 1) % 100 == 0:
            rl = (run_loss / 100).tolist(); h, c = hits.cpu().numpy(), cnt.cpu().numpy(); gl = gn.tolist()
            hp, cp = hitsP.cpu().numpy(), cntP.cpu().numpy()
            for r in range(R):
                recs[r]["curve"].append({"step": s + 1, "loss": rl[r], "grad_norm": float(gl[r]), "loss_scale": loss_scale,
                                         "acc_by_k": {int(k): float(h[r, k] / c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0},
                                         "n_by_k": {int(k): int(c[r, k]) for k in range(c.shape[1]) if c[r, k] > 0},
                                         "acc_by_Pdepth": {int(d): float(hp[r, d] / cp[r, d]) for d in range(cp.shape[1]) if cp[r, d] > 0},
                                         "n_by_Pdepth": {int(d): int(cp[r, d]) for d in range(cp.shape[1]) if cp[r, d] > 0}})
            run_loss.zero_(); hits.zero_(); cnt.zero_(); hitsP.zero_(); cntP.zero_()
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
