"""Train a stack of rung-0 runs of one arm together (experiments/ptr/DESIGN.md §6; stacked.py).

`run_stack(cfgs, dev)` takes run configs that differ only in seed, lr and tail_factor, trains them as one stacked
model, and returns one record per run in the same format as train.run(), so analyze.py reads both.

Data is paired by seed: run r draws its episodes from a generator seeded by its own seed, independent of which other
runs share the stack, so seed s sees the same episode stream in every arm.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import PtrConfig
from .gen_torch import TorchGen
from .model import ModelConfig, kl_exit
from .stacked import StackedPtrModel, StackedAdamW, clip_grad_norm_per_run, oracle_steps_t, remaining_after_t
from .train import split_cfg, train_mode, step_probe, KL_EXIT_THRESHOLDS

PER_RUN_KEYS = ("seed", "lr", "tail_factor")


def est_cost(cfg: dict) -> float:
    """Rough GPU-seconds of one run on a T4 (pilots 1-2): looped arms 0.065 s per step at T = 10, scaling with T;
    static arms 0.0065 s per layer-step. Used only to balance stacks across shards."""
    steps = int(cfg.get("steps", 20000))
    if cfg.get("kind") == "static":
        return steps * 0.0065 * int(cfg.get("layers", 4))
    return steps * 0.065 * int(cfg.get("T", 10)) / 10


def assign_shards(chunks: list[list[dict]], k: int):
    """Longest-first greedy assignment of stacks to k shards by estimated cost. Deterministic given the chunks, so
    every shard process computes the same assignment. Returns (per-shard chunk lists, per-shard loads)."""
    cost = [sum(est_cost(c) for c in ch) for ch in chunks]
    order = sorted(range(len(chunks)), key=lambda i: (-cost[i], chunks[i][0].get("name", "")))
    out, loads = [[] for _ in range(k)], [0.0] * k
    for i in order:
        j = min(range(k), key=lambda j: (loads[j], j))
        out[j].append(chunks[i])
        loads[j] += cost[i]
    return out, loads


def _group_ok(cfgs):
    base = {k: v for k, v in cfgs[0].items() if k not in PER_RUN_KEYS and k not in ("name", "arm")}
    for c in cfgs[1:]:
        other = {k: v for k, v in c.items() if k not in PER_RUN_KEYS and k not in ("name", "arm")}
        assert other == base, f"runs in a stack must agree on everything but {PER_RUN_KEYS}: {c.get('name')}"


def sample_stack(gens, B, tail, **kw):
    parts = [g.sample(1, B, tail_factor=[tail[r]], **kw) for r, g in enumerate(gens)]
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


@torch.no_grad()
def evaluate_stack(m, dc, mc, tc, gens_eval, tail, trace=False, T=None):
    m.eval()
    R = m.R
    T = tc["T_eval"] if T is None else T
    res = [{"in": {}} for _ in range(R)]
    traces = [{} for _ in range(R)]
    pf, ps = [[] for _ in range(R)], [[] for _ in range(R)]
    for k in tc["eval_ks"]:
        b = sample_stack(gens_eval, tc["eval_n"], tail, forced_k=k, Qn=tc["eval_Qn"])
        st = oracle_steps_t(b, dc) if mc.halting == "oracle" else None
        out = m(b, T=None if mc.kind == "static" else T, steps=st, mode="eval")
        lg, tg = out["logits"], out["targets"]                                            # (R, B, Q, A), (R, B, Q)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none").view(tg.shape)
        ok = (lg.argmax(-1) == tg)
        sel = b["q_k"] == k
        steps = out["steps"]
        qsteps = steps[:, :, out["qpos"]]
        psteps = steps[:, :, 1:dc.P + 1]
        pv = b["pair_val"]
        present = torch.zeros(R, pv.shape[1], dc.n_sym, device=pv.device)
        present.scatter_(2, pv, 1.0)
        n_term = present[:, :, dc.n_keys:].sum(2)                                                  # (R, B) distinct terminals
        lr_stats = None
        if mc.kind == "looped" and mc.halting == "fixed":      # arm LR: the stability exit on this fixed loop's readouts
            psl = torch.stack([m.readout(hq) for hq in out["hs"]], 0)                             # (T, R, B, Q, A)
            lr_stats = kl_exit(psl, tg, KL_EXIT_THRESHOLDS)
        for r in range(R):
            s = sel[r]
            rec = {"n": int(s.sum()), "acc": float(ok[r][s].float().mean()), "ce": float(ce[r][s].mean()),
                   "q_steps": float(qsteps[r][s].mean()), "pair_steps": float(psteps[r].mean()),
                   "max_steps": float(steps[r].max()), "baseline_acc": float((1.0 / n_term[r]).mean()),
                   "baseline_ce": float(n_term[r].log().mean())}
            if mc.kind == "looped" and mc.halting == "learned":
                lam = out["lams"][:, r][:, :, out["qpos"]]
                rec["lam_by_step"] = [float(lam[t][s].mean()) for t in range(lam.shape[0])]
            if lr_stats is not None:
                rec["kl_exit"] = {str(thr): {"acc": float(okk[r][s].float().mean()), "q_steps": float(stp[r][s].mean())}
                                  for thr, (okk, stp) in lr_stats.items()}
            res[r]["in"][k] = rec
            if trace and mc.kind == "looped":
                traces[r][k] = _traces(m, out, b, r, s, tc["trace_n"], dc)
                if k >= 8:   # steps t < k are chasing steps: features for the step-index probe
                    idx = s.nonzero()[: tc["probe_n"] // 4]
                    for t in range(min(k, T, 32)):
                        hq = out["hs"][t][r][idx[:, 0], idx[:, 1]]
                        pf[r].append(F.layer_norm(hq, (mc.d,)))
                        ps[r].append(torch.full((idx.shape[0],), t, device=hq.device))
    for r in range(R):
        if pf[r]:
            res[r]["step_probe_acc"] = step_probe(torch.cat(pf[r]), torch.cat(ps[r]))
    m.train()
    return res, traces


@torch.no_grad()
def _traces(m, out, b, r, sel, n, dc):
    hs = out["hs"]                      # per-step states at the query positions: (R, B, Qn, d) per step
    T = len(hs)
    idx = sel.nonzero()[:n]
    bb, jj = idx[:, 0], idx[:, 1]
    k = b["q_k"][r][bb, jj]
    tg = out["targets"][r][bb, jj]
    dh, kl, ent, corr, rem = [], [], [], [], []
    prev = None
    for t in range(T):
        h = hs[t][r][bb, jj]
        lg = (m.out_norm(h.unsqueeze(0).unsqueeze(0).expand(m.R, 1, -1, -1))[r, 0]
              @ m.sym.reg_slot[dc.n_keys:dc.n_keys + dc.n_ans].T)
        lp = F.log_softmax(lg, -1)
        dh.append((h - hs[t - 1][r][bb, jj]).norm(dim=-1) if t > 0 else h.norm(dim=-1))
        kl.append((lp.exp() * (lp - prev)).sum(-1) if prev is not None else torch.zeros_like(k, dtype=torch.float))
        ent.append(-(lp.exp() * lp).sum(-1))
        corr.append((lg.argmax(-1) == tg).float())
        rem.append((k - (t + 1)).clamp_min(0).float())
        prev = lp
    pack = lambda L: torch.stack(L, 0).cpu().numpy().round(4).tolist()
    return {"dh": pack(dh), "kl": pack(kl), "entropy": pack(ent), "correct": pack(corr), "remaining": pack(rem),
            "k": k.cpu().tolist()}


def run_stack(cfgs: list[dict], dev: str = "cuda", log=print, amp: bool | None = None) -> list[dict]:
    _group_ok(cfgs)
    R = len(cfgs)
    dc, mc, tc = split_cfg({k: v for k, v in cfgs[0].items() if k not in PER_RUN_KEYS})
    seeds = [int(c.get("seed", 0)) for c in cfgs]
    lrs = torch.tensor([float(c.get("lr", tc["lr"])) for c in cfgs], device=dev)
    tail = [float(c.get("tail_factor", 1.0)) for c in cfgs]
    torch.manual_seed(1000003 * seeds[0] + sum(seeds))
    m = StackedPtrModel(R, mc, dc).to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad) // R
    opt = StackedAdamW(m.parameters(), lrs, betas=(0.9, 0.95), weight_decay=tc["wd"])
    sched = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc["steps"])))
    gens = [TorchGen(dc, dev, seed=s * 7919 + 11) for s in seeds]
    gens_eval = [TorchGen(dc, dev, seed=s * 7919 + 12345) for s in seeds]
    use_amp = (dev.startswith("cuda")) if amp is None else amp
    # manual dynamic loss scaling (PyTorch's GradScaler needs a torch optimizer); runs whose gradients overflow skip
    # that step on their own, so runs stay independent
    loss_scale, clean_steps = (2.0 ** 14 if use_amp else 1.0), 0
    recs = []
    for r, c in enumerate(cfgs):
        recs.append({"cfg": {k: v for k, v in c.items() if k not in ("name", "arm")}, "name": c.get("name"),
                     "arm": c.get("arm"), "data": {**dc.dict(), "tail_factor": tail[r]}, "model": mc.dict(),
                     "train": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in tc.items()},
                               "lr": float(lrs[r]), "seed": seeds[r], "stack_size": R, "amp": use_amp},
                     "n_params": n_params, "curve": [], "evals": [], "device": dev})
    t0 = time.time()
    run_loss = torch.zeros(R, device=dev)
    acc_hits = torch.zeros(R, dc.kmax + 1, device=dev)
    acc_cnt = torch.zeros(R, dc.kmax + 1, device=dev)
    halt_hits = torch.zeros(R, dc.kmax + 1, device=dev)     # learned halting: accuracy of the halted readout
    mode = train_mode(mc)
    for s in range(tc["steps"]):
        b = sample_stack(gens, tc["batch"], tail)
        st = oracle_steps_t(b, dc) if mc.halting == "oracle" else None
        rem = remaining_after_t(b, dc, mc.T) if (mc.kind == "looped" and mc.halting == "value") else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = m(b, steps=st, mode=mode)
            loss, per_run, extra = m.loss(out, remaining_hops=rem)
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
                loss_scale, clean_steps = max(loss_scale / 2, 1.0), 0
            else:
                clean_steps += 1
                if clean_steps >= 1000 and loss_scale < 2.0 ** 16:
                    loss_scale, clean_steps = loss_scale * 2, 0
        run_loss += torch.nan_to_num(per_run, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():   # accuracy by k accumulated on the device; read back once per 100 steps (no syncs)
            ok = (out["logits"].argmax(-1) == out["targets"]).float().reshape(R, -1)              # (R, B*Q)
            kk = b["q_k"].reshape(R, -1)
            acc_hits.scatter_add_(1, kk, ok)
            acc_cnt.scatter_add_(1, kk, torch.ones_like(ok))
            if "halted_logits" in out:
                okh = (out["halted_logits"].argmax(-1) == out["targets"]).float().reshape(R, -1)
                halt_hits.scatter_add_(1, kk, okh)
        if (s + 1) % 100 == 0:
            rl = (run_loss / 100).tolist()
            hits, cnt = acc_hits.cpu().numpy(), acc_cnt.cpu().numpy()
            hhits = halt_hits.cpu().numpy() if "halted_logits" in out else None
            gnl = gn.tolist()
            exl = {k: v.tolist() for k, v in extra.items()}
            for r in range(R):
                entry = {"step": s + 1, "loss": rl[r], "grad_norm": float(gnl[r]), "loss_scale": loss_scale,
                         "acc_by_k": {int(k): float(hits[r, k] / cnt[r, k]) for k in range(cnt.shape[1]) if cnt[r, k] > 0},
                         "n_by_k": {int(k): int(cnt[r, k]) for k in range(cnt.shape[1]) if cnt[r, k] > 0},
                         **{k: float(v[r]) for k, v in exl.items()}}
                if hhits is not None:   # the policy's own prediction; acc_by_k is the T-loop readout for these arms
                    entry["acc_halt_by_k"] = {int(k): float(hhits[r, k] / cnt[r, k]) for k in range(cnt.shape[1])
                                              if cnt[r, k] > 0}
                recs[r]["curve"].append(entry)
            run_loss.zero_()
            acc_hits.zero_(); acc_cnt.zero_(); halt_hits.zero_()
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == tc["steps"]:
            final = s + 1 == tc["steps"]
            ev, traces = evaluate_stack(m, dc, mc, tc, gens_eval, tail, trace=final)
            for r in range(R):
                recs[r]["evals"].append({"step": s + 1, "eval": ev[r], "time": time.time() - t0})
                if final:
                    recs[r]["traces"] = traces[r]
            if final and mc.kind == "looped" and mc.halting == "fixed":
                ev_t, _ = evaluate_stack(m, dc, mc, tc, gens_eval, tail, T=mc.T)
                for r in range(R):
                    recs[r]["eval_T_train"] = ev_t[r]
            e0 = ev[0]["in"]
            log(f"step {s+1} [{R} runs] loss(run0) {recs[0]['curve'][-1]['loss'] if recs[0]['curve'] else float('nan'):.3f} "
                + "acc/steps by k (run0): " + " ".join(f"{k}:{v['acc']:.2f}/{v['q_steps']:.1f}" for k, v in e0.items())
                + f"  [{time.time()-t0:.0f}s]")
    for r in range(R):
        recs[r]["time"] = time.time() - t0
    return recs
