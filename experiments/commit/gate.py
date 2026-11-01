"""Rung 3 derive step between stages (experiments/commit/DESIGN.md §3 gate, §5.3; review 1 B5):

  1. the commitment gate on every phase-1 record (clean evaluation when present): shortest-correct committed accuracy
     >= 0.95 at in4 and in8 on P-words, program label E (set_late <= 0.25) for E starts and L (>= 0.75) for L starts,
     sampled shortest-correct >= 0.95, not program R;
  2. for every passing checkpoint, what phase 2 will have to work with, under each shift's data:
       - the logit margin at the class the shift relabels (shift A: HOP - PREV at own-flag = 1, c = 1, set-here = 1, on
         the states the committed route visits under delta 0);
       - coverage of the correct new route by the eps-mixed beam (eps 0.1 and 0.3, W = 16) and by K = 16 samples, on
         P-words (shift A: the delta-0 oracle route; shift B: any shortest correct completion on PP words);
       - the router's own exact posterior mass on the correct route (what EX's target gives it);
  3. a JSON report that make_grids.py reads to build stage 2, and a markdown table.

    PYTHONPATH=src python experiments/commit/gate.py --results results/kaggle_route-cm_stage1 [--n 96] [--out experiments/commit/results_data/gate.json]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from topology_design.route.data import RouteConfig, generate  # noqa: E402
from topology_design.route.exec import tables, oracle, marginal, readout, route_ok, committed, PER_SYM, NH, HOP, PREV, SET  # noqa: E402
from topology_design.route.router import StackedRouter, RouterConfig  # noqa: E402
from topology_design.route.train_route import load_router_state_, policy, flat, isolated_P  # noqa: E402
from topology_design.route.pool import to_dict  # noqa: E402
from topology_design.route import estimators as E  # noqa: E402

T, LAM, EPS, K, W = 14, 0.02, 0.02, 16, 16


def load_router(path, dev="cpu"):
    st = torch.load(path, map_location="cpu")
    rc, dc = RouterConfig(**st["router"]), RouteConfig(**st["data"])
    m = StackedRouter(1, rc, dc)
    load_router_state_(m, 0, st["state"])
    return m.to(dev).eval(), rc, dc, st


def gate(rec, start):
    """(passes, reasons, label) from a phase-1 record; the clean evaluation is used when present."""
    ev = rec["evals"][-1]
    e = ev.get("clean") or ev["eval"]
    i4, i8 = e["in4"], e["in8"]
    reasons = []
    if (i4.get("min_acc_P") or 0) < 0.95 or (i8.get("min_acc_P") or 0) < 0.95:
        reasons.append(f"not learned ({i4.get('min_acc_P')}/{i8.get('min_acc_P')})")
    late = i8.get("set_late_P"); oth = i8.get("set_other_P") or 0
    label = "–" if late is None else ("other" if oth > 0.25 else ("L" if late >= 0.75 else ("E" if late <= 0.25 else "mix")))
    want = {"E": "E", "L": "L"}[start]
    if label != want:
        reasons.append(f"program {label} ({late})")
    if (i8.get("samp_min_acc_P") or 0) < 0.95:
        reasons.append(f"not deterministic ({i8.get('samp_min_acc_P')})")
    plus_one = all(abs((e[st]["q_steps"] - e[st]["oracle_steps"]) - 1.0) <= 0.15 and e[st]["acc"] >= 0.95 for st in ("in4", "in8", "none8"))
    if plus_one:
        reasons.append("program R")
    return not reasons, reasons, label


@torch.no_grad()
def what_it_sees(m, rc, dc_target, n, seed, dev, shift):
    """Coverage, margins and the router's own posterior mass on the correct route, on n episodes of the target data."""
    ep = generate(dc_target, n, np.random.default_rng(seed))
    b = {k: v.unsqueeze(0) for k, v in to_dict(ep, dev).items()}
    fb = flat(b)
    mp = tables(ep, dc_target, dev)
    pi = policy(m, b, T)
    oa, osteps, ostates = oracle(mp, fb["q_start"], T, dc_target)
    qw = fb["q_word"]
    hasP = (qw == 1).any(-1)
    if shift == "A":
        words = hasP                                            # every P-word relabels the decision
    else:
        words = hasP & ~isolated_P(qw)                          # PP words
    out = {"n_words": int(words.sum())}
    # coverage by the beam at eps 0.1 / 0.3: a correct completion at the oracle's length (shift B: any shortest correct)
    for eps_mix in (0.1, 0.3):
        found, a_b, st_b = E.beam_search(pi, mp, fb["q_start"], fb["q_ans"], T, W, dc_target, eps_mix=eps_mix, lam=LAM)
        kept_short = found & (st_b == osteps)                      # the kept route is shortest-correct (any program)
        if shift == "A":
            ok = found & route_ok(a_b, st_b, oa, osteps)           # the kept route IS the oracle's (the expressible L route)
        else:
            ok = kept_short
        out[f"beam_eps{eps_mix}"] = float(ok[words].float().mean()) if words.any() else None
        out[f"beam_eps{eps_mix}_kept_shortest"] = float(kept_short[words].float().mean()) if words.any() else None
        # the oracle route among ALL completions (not only the kept one), and the kept route's program
        comp = E.beam_completions(pi, mp, fb["q_start"], T, W, dc_target, eps_mix=eps_mix, lam=LAM)
        xs = E.replay(mp, fb["q_start"], comp["acts"])
        oks = torch.stack([route_ok(comp["acts"][:, :, j], comp["moves"][:, :, j], oa, osteps) & comp["valid"][:, :, j] for j in range(comp["acts"].shape[2])], -1)
        out[f"beam_eps{eps_mix}_oracle_among_completions"] = float(oks.any(-1)[words].float().mean()) if words.any() else None
        xb = E.replay(mp, fb["q_start"], a_b)
        from topology_design.route.train_route import program_stats
        _, late, oth = program_stats(a_b, xb, st_b, xb[:, :, -1] // PER_SYM, fb["q_ans"], osteps, mp, oa, ostates)
        lw = late[words & kept_short]
        out[f"beam_eps{eps_mix}_kept_set_late"] = float(torch.nanmean(lw)) if torch.isfinite(lw).any() else None
    g = torch.Generator(device=dev).manual_seed(4242)
    sym, st_k, a_k, x_k = E.sample_routes(pi, mp, fb["q_start"], T, K, generator=g)
    if shift == "A":
        oks = torch.stack([route_ok(a_k[:, :, j], st_k[:, :, j], oa, osteps) for j in range(K)], -1)
    else:
        oks = (sym == fb["q_ans"].unsqueeze(-1)) & (st_k == osteps.unsqueeze(-1))
    out["samples_K16"] = float(oks.any(-1)[words].float().mean()) if words.any() else None
    # the router's own exact posterior mass on the oracle route (EX's target): pi(r*) p(y | r*) / q(y)
    Rd = readout(dc_target, EPS, dev)
    q, es, _, _ = marginal(pi, mp, fb["q_start"], T, Rd)
    lpr = E.route_logprob(pi, oa, ostates)
    ans = fb["q_ans"] - dc_target.n_keys
    qy = q.gather(-1, ans.unsqueeze(-1)).squeeze(-1)
    post = (lpr.exp() * (1 - EPS) / qy.clamp_min(1e-12)).clamp(max=1.0)
    out["own_posterior_on_oracle"] = float(post[words].mean()) if words.any() else None
    # the committed route under the target data, and the margin at the relabeled class
    s_fin, steps, acts, states = committed(pi, mp, fb["q_start"], T)
    out["committed_min_acc_words"] = float(((s_fin == fb["q_ans"]) & (steps == osteps))[words].float().mean()) if words.any() else None
    if shift == "A":
        lg = m.logits(b)[0]                                     # (B, NS, 4)
        NS = lg.shape[1]
        ids = torch.arange(NS, device=dev); s_, c_, h_ = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
        own = mp["flag"][:, s_] == 1
        cls_prev = own & (c_ == 1).unsqueeze(0) & (h_ == 1).unsqueeze(0)     # (B, NS): where L must PREV
        cls_set = own & (c_ == 0).unsqueeze(0) & (h_ == 0).unsqueeze(0)      # where L must SET
        # the states the ORACLE route visits on P-words (the committed route of an L router never reaches (s, 1, 1))
        vis = torch.zeros_like(cls_prev)
        To = oa.shape[-1]
        stv = ostates[:, :, :To]
        bidx = torch.arange(stv.shape[0], device=dev).view(-1, 1, 1).expand_as(stv)
        m_ = (oa >= 0) & words.unsqueeze(-1)
        vis[bidx[m_], stv[m_]] = True
        pr = torch.softmax(lg, -1)
        for key, cls, a2 in (("hop_minus_prev", cls_prev, PREV), ("hop_minus_set", cls_set, SET)):
            sel = cls & vis
            marg = (lg[..., HOP] - lg[..., a2])[sel]
            out[f"margin_{key}"] = {"mean": float(marg.mean()), "min": float(marg.min()), "max": float(marg.max()), "n": int(sel.sum())} if sel.any() else None
            out[f"p_{'prev' if a2 == PREV else 'set'}_at_class"] = float(pr[..., a2][sel].mean()) if sel.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="compute what_it_sees for checkpoints that fail the gate too (CPU sanity)")
    args = ap.parse_args()
    recs = {}
    for p in sorted(glob.glob(os.path.join(args.results, "results", "*.json.gz"))):
        r = json.load(gzip.open(p)); recs[r["name"]] = r
    ck = os.path.join(args.results, "ckpt")
    report = {"gate": {}, "sees": {}}
    for name, r in sorted(recs.items()):
        arm = r.get("arm") or ""
        if not r.get("ckpt"):
            continue
        start = "E" if arm.startswith("EX") else ("L" if arm.startswith("EI16") else None)
        if start is None:
            continue
        ok, reasons, label = gate(r, start)
        report["gate"][name] = {"arm": arm, "seed": r["train"]["seed"], "start": start, "passes": ok, "reasons": reasons, "label": label, "ckpt": r["ckpt"]}
        print(f"{name}: {'PASS' if ok else 'FAIL'} {label} {reasons}", flush=True)
        if not ok and not args.force:
            continue
        m, rc, dc, st = load_router(os.path.join(ck, r["ckpt"]), args.device)
        sees = {}
        for shift, dct in (("A", RouteConfig(**{**dc.dict(), "deferral": 0})), ("B", RouteConfig(**{**dc.dict(), "pp_frac": 1.0}))):
            sees[shift] = what_it_sees(m, rc, dct, args.n, 999, args.device, shift)
            print(f"   shift {shift}: {sees[shift]}", flush=True)
        report["sees"][name] = sees
    out = args.out or os.path.join(args.results, "gate.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
