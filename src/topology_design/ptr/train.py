"""Training and evaluation of one rung-0 run (experiments/ptr/DESIGN.md §§3, 6).

A run config is a flat dict: data fields (PtrConfig), model fields (ModelConfig) and training fields
(steps, batch, lr, warmup, seed, eval_every, T_eval, eval_ks, eval_n). `run()` returns a JSON-serializable record:
loss curves, accuracy by k over training, the final evaluation by k on three symbol sets, per-step residual signals
for the fixed arm (for the residual-triggered halting analysis), and timings.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from .data import PtrConfig, generate
from .model import ModelConfig, PtrModel, oracle_steps, remaining_after

DATA_KEYS = {f.name for f in fields(PtrConfig)}
MODEL_KEYS = {f.name for f in fields(ModelConfig)}
TRAIN_DEFAULTS = dict(steps=20000, batch=64, lr=3e-4, warmup=500, seed=0, eval_every=2500, T_eval=40,
                      eval_ks=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32), eval_n=250, eval_Qn=3,
                      trace_n=64, probe_n=400, wd=0.01, grad_clip=1.0)


def split_cfg(cfg: dict):
    dc = PtrConfig(**{k: v for k, v in cfg.items() if k in DATA_KEYS})
    mc = ModelConfig(**{k: v for k, v in cfg.items() if k in MODEL_KEYS})
    tc = {**TRAIN_DEFAULTS, **{k: v for k, v in cfg.items() if k in TRAIN_DEFAULTS}}
    return dc, mc, tc


def train_mode(mc: ModelConfig):
    if mc.kind == "static":
        return "train"
    return {"learned": "ponder_train"}.get(mc.halting, "train")


def step_loss(m: PtrModel, ep, dev, mc: ModelConfig, dc: PtrConfig):
    st = oracle_steps(ep, dc, dev) if mc.halting == "oracle" else None
    out = m(ep, dev, steps=st, mode=train_mode(mc))
    rem = remaining_after(ep, dc, mc.T, dev) if (mc.kind == "looped" and mc.halting == "value") else None
    loss, extra = m.loss(out, ep, dev, remaining_hops=rem)
    with torch.no_grad():
        acc_by_k = {}
        pred = out["logits"].argmax(-1).cpu().numpy()
        ok = pred == ep.q_ans - dc.n_keys
        for k in np.unique(ep.q_k):
            acc_by_k[int(k)] = float(ok[ep.q_k == k].mean())
    return loss, extra, acc_by_k


@torch.no_grad()
def evaluate(m: PtrModel, dc: PtrConfig, mc: ModelConfig, tc: dict, dev, rng, *, T=None, trace=False):
    """Stratified by k on three symbol sets. Returns {set: {k: metrics}} plus optional per-step traces."""
    m.eval()
    T = tc["T_eval"] if T is None else T
    res = {"in": {}}
    traces = {}
    probe_feats, probe_steps = [], []
    for k in tc["eval_ks"]:
        ep = generate(dc, tc["eval_n"], rng, forced_k=k, Qn=tc["eval_Qn"])
        st = oracle_steps(ep, dc, dev) if mc.halting == "oracle" else None
        out = m(ep, dev, T=None if mc.kind == "static" else T, steps=st, mode="eval")
        logits = out["logits"]
        tg = out["targets"]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tg.reshape(-1), reduction="none").view(tg.shape)
        ok = (logits.argmax(-1) == tg)
        sel = torch.as_tensor(ep.q_k == k, device=dev)                      # the forced query of each episode
        steps = out["steps"]
        qsteps = steps[:, out["qpos"]]
        psteps = steps[:, 1:dc.P + 1]
        # no-hop baseline: uniform over the answer symbols that appear as terminals in the context
        n_term = np.array([len(set(v[v >= dc.n_keys].tolist())) for v in ep.pair_val])
        rec = {"n": int(sel.sum()), "acc": float(ok[sel].float().mean()), "ce": float(ce[sel].mean()),
               "q_steps": float(qsteps[sel].mean()), "pair_steps": float(psteps.mean()),
               "max_steps": float(steps.max()), "baseline_acc": float((1.0 / n_term).mean()),
               "baseline_ce": float(np.log(n_term).mean())}
        if mc.kind == "looped" and mc.halting == "learned":
            lam = out["lams"][:, :, out["qpos"]]                               # (T, B, Qn)
            rec["lam_by_step"] = [float(lam[t][sel].mean()) for t in range(lam.shape[0])]
        res["in"][k] = rec
        if trace and mc.kind == "looped":
            traces[k] = residual_traces(m, out, ep, dc, dev, sel, tc["trace_n"])
            if k >= T:   # every step is a chasing step: features for the step-index probe
                idx = sel.nonzero()[:tc["probe_n"] // 4]
                for t in range(min(T, 32)):
                    probe_feats.append(m.halt_norm(out["hs"][t][idx[:, 0], out["qpos"][idx[:, 1]]])
                                       if mc.halting == "learned" and mc.halt_input != "index"
                                       else F.layer_norm(out["hs"][t][idx[:, 0], out["qpos"][idx[:, 1]]], (mc.d,)))
                    probe_steps.append(torch.full((idx.shape[0],), t, device=dev))
    if probe_feats:
        res["step_probe_acc"] = step_probe(torch.cat(probe_feats), torch.cat(probe_steps))
    m.train()
    return res, traces


@torch.no_grad()
def step_probe(X, y):
    """Linear probe for the step index from the normalized state (one-vs-rest least squares, half/half split)."""
    n = X.shape[0]
    perm = torch.randperm(n, device=X.device)
    tr, te = perm[: n // 2], perm[n // 2:]
    Y = F.one_hot(y, int(y.max()) + 1).float()
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    W = torch.linalg.lstsq(Xb[tr], Y[tr]).solution
    pred = (Xb[te] @ W).argmax(-1)
    return float((pred == y[te]).float().mean())


@torch.no_grad()
def residual_traces(m, out, ep, dc, dev, sel, n):
    """Per-step signals on query tokens of the fixed arm: ||dh||, readout step-KL, readout entropy, correctness,
    and the exact remaining hops; for the first n selected queries."""
    hs = out["hs"]
    qpos = out["qpos"]
    T = len(hs)
    idx = sel.nonzero()[:n]                                                        # (n, 2) = (b, j)
    b, j = idx[:, 0], idx[:, 1]
    k = torch.as_tensor(ep.q_k, device=dev)[b, j]
    tg = out["targets"][b, j]
    dh, kl, ent, corr, rem = [], [], [], [], []
    prev = None
    for t in range(T):
        h = hs[t][b, qpos[j]]
        lg = m.readout(h.unsqueeze(1)).squeeze(1)
        lp = F.log_softmax(lg, -1)
        dh.append((h - hs[t - 1][b, qpos[j]]).norm(dim=-1) if t > 0 else h.norm(dim=-1))
        kl.append((lp.exp() * (lp - prev)).sum(-1) if prev is not None else torch.zeros_like(k, dtype=torch.float))
        ent.append(-(lp.exp() * lp).sum(-1))
        corr.append((lg.argmax(-1) == tg).float())
        rem.append((k - (t + 1)).clamp_min(0).float())
        prev = lp
    pack = lambda L: torch.stack(L, 0).cpu().numpy().round(4).tolist()
    return {"dh": pack(dh), "kl": pack(kl), "entropy": pack(ent), "correct": pack(corr), "remaining": pack(rem),
            "k": k.cpu().tolist()}


def run(cfg: dict, dev: str = "cpu", log=print) -> dict:
    dc, mc, tc = split_cfg(cfg)
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"] + 1000 * int(dc.alpha * 10) + 7)
    m = PtrModel(mc, dc).to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=tc["wd"], betas=(0.9, 0.95))
    sched = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc["steps"])))
    rec = {"cfg": cfg, "data": dc.dict(), "model": mc.dict(), "train": {k: (list(v) if isinstance(v, tuple) else v)
                                                                       for k, v in tc.items()},
           "n_params": n_params, "curve": [], "evals": [], "device": dev}
    t0 = time.time()
    run_loss, run_acc = 0.0, {}
    for s in range(tc["steps"]):
        for g in opt.param_groups:
            g["lr"] = tc["lr"] * sched(s)
        ep = generate(dc, tc["batch"], rng)
        loss, extra, acc_by_k = step_loss(m, ep, dev, mc, dc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), tc["grad_clip"])
        opt.step()
        run_loss += loss.item()
        for k, v in acc_by_k.items():
            run_acc.setdefault(k, []).append(v)
        if (s + 1) % 100 == 0:
            rec["curve"].append({"step": s + 1, "loss": run_loss / 100, "grad_norm": float(gn),
                                 "acc_by_k": {k: float(np.mean(v)) for k, v in sorted(run_acc.items())},
                                 **{k: v for k, v in extra.items()}})
            run_loss, run_acc = 0.0, {}
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == tc["steps"]:
            final = s + 1 == tc["steps"]
            ev, traces = evaluate(m, dc, mc, tc, dev, rng, trace=final)
            ev_in = {k: (v["acc"], v["q_steps"]) for k, v in ev["in"].items() if isinstance(v, dict)}
            rec["evals"].append({"step": s + 1, "eval": ev, "time": time.time() - t0})
            if final:
                rec["traces"] = traces
                if mc.kind == "looped" and mc.halting == "fixed":   # the fixed arm at its training depth too
                    ev9, _ = evaluate(m, dc, mc, tc, dev, rng, T=mc.T)
                    rec["eval_T_train"] = ev9
            last_loss = rec["curve"][-1]["loss"] if rec["curve"] else float("nan")
            log(f"step {s+1} loss {last_loss:.3f} acc/steps by k (in): "
                + " ".join(f"{k}:{a:.2f}/{q:.1f}" for k, (a, q) in ev_in.items() if a is not None)
                + f"  [{time.time()-t0:.0f}s]")
    rec["time"] = time.time() - t0
    return rec
