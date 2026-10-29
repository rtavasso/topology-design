"""EXPLORATORY diagnostic (Entry 63): why the deferral-0 router does not find SET->PREV. Usage:
    PYTHONPATH=src python experiments/route/diag_deferral0.py <deferral 0|1> <init|planted> <lam> <steps> <out.json>
"""
"""Diagnostic: why does the deferral-0 router not find SET->PREV? Train a 1-run router on CPU and log, every 50
steps, the argmax / probabilities at the four decision types on a fixed probe batch: (flagged,0,0) [oracle SET],
(flagged,1,1) [d0: PREV] / landing (s,1,0) [d1: PREV], stray (unflagged,1,1) [HOP], stray (unflagged,1,0) [HOP].
Optional: start from the planted solution at a small gain (reachability / stability test)."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(3)
from topology_design.route.data import RouteConfig, generate
from topology_design.route.router import StackedRouter, RouterConfig, plant
from topology_design.route.exec import maps, marginal, readout, oracle, sid, PER_SYM, NH, HOP, PREV, SET, HALT
from topology_design.route.pool import EpisodePool, stack_batches, to_dict
from topology_design.ptr.stacked import StackedAdamW, clip_grad_norm_per_run
import math
deferral = int(sys.argv[1]); planted = sys.argv[2] == "planted"; lam = float(sys.argv[3]); steps = int(sys.argv[4]); out = sys.argv[5]
dc = RouteConfig(deferral=deferral); rc = RouterConfig(lam=lam)
torch.manual_seed(0)
m = StackedRouter(1, rc, dc)
if planted:
    plant(m, gain=1.0, hop_bias=1.5)
opt = StackedAdamW(m.parameters(), torch.tensor([1e-3]), betas=(0.9, 0.95), weight_decay=0.01)
warm, total = 100, steps
sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
pool = EpisodePool(dc, 6000, seed=11)
Rd = readout(dc, rc.eps, "cpu")
# probe batch: 64 episodes, oracle routes -> states of each type
pb = to_dict(generate(dc, 64, np.random.default_rng(5)), "cpu")
pbb = {k: v.unsqueeze(0) for k, v in pb.items()}
mp = maps(pb["pair_key"], pb["pair_val"], pb["pair_prev"], pb["pair_flag"], dc)
oa, osteps, ost = oracle(mp, pb["q_start"], 14, dc)
B = 64; ar = torch.arange(B).unsqueeze(1).unsqueeze(2)
flag_of = lambda st: mp["flag"].gather(1, (st // PER_SYM).reshape(B, -1)).reshape(st.shape)
types = {}
st_all, a_all = ost[:, :, :-1], oa
sym = st_all // PER_SYM; c = (st_all % PER_SYM) // NH; h = st_all % NH; fl = flag_of(st_all)
types["flagged00->SET"] = (a_all == SET) & (c == 0)
if deferral == 0:
    types["flagged11->PREV"] = (a_all == PREV)
    types["stray_unflagged11->HOP"] = None  # counterfactual: (s,1,1) at unflagged key states where oracle HOPs at (s,0,0)
    types["stray_unflagged10->HOP"] = None
else:
    types["landing10->PREV"] = (a_all == PREV)
    types["flagged11->HOP"] = (a_all == HOP) & (h == 1)
    types["stray_unflagged10->HOP"] = None
unflagged_hop = (a_all == HOP) & (c == 0) & (h == 0) & (fl == 0)
log = []
t0 = time.time()
for s in range(steps + 1):
    if s % 50 == 0:
        with torch.no_grad():
            pi = torch.softmax(m.logits(pbb), -1)[0]                      # (B, NS, 4)
            rec = {"step": s}
            for name, mask in types.items():
                if mask is None:
                    cc, hh = (1, 1) if name.endswith("11->HOP") else (1, 0)
                    states = sid(sym, cc, hh); mask2 = unflagged_hop; want = HOP
                else:
                    states = st_all; mask2 = mask; want = {"SET": SET, "PREV": PREV, "HOP": HOP}[name.split("->")[1]]
                p = pi[torch.arange(B).view(B, 1, 1).expand_as(states), states]      # (B, Q, T, 4)
                pm = p[mask2]
                rec[name] = {"argmax_ok": float((pm.argmax(-1) == want).float().mean()), "p": [round(float(x), 3) for x in pm.mean(0)]}
            log.append(rec)
            print(f"[{time.time()-t0:.0f}s] step {s} " + " | ".join(f"{k}: ok {v['argmax_ok']:.2f} p{v['p']}" for k, v in rec.items() if k != "step"), flush=True)
    if s == steps:
        break
    b = stack_batches([pool.sample(16)])
    fb = {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}
    mpb = maps(fb["pair_key"], fb["pair_val"], fb["pair_prev"], fb["pair_flag"], dc)
    pi = torch.softmax(m.logits(b), -1).reshape(16, -1, 4)
    q, es, _, _ = marginal(pi, mpb, fb["q_start"], rc.T, Rd)
    ans = fb["q_ans"] - dc.n_keys
    loss = (-torch.log(q.gather(-1, ans.unsqueeze(-1)).squeeze(-1) + 1e-9) + lam * es).mean()
    opt.zero_grad(); loss.backward(); clip_grad_norm_per_run(m.parameters(), 1.0); opt.step(scale=sched(s))
json.dump(log, open(out, "w"))
