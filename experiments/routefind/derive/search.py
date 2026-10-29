"""Derived before measuring (experiments/routefind/DESIGN.md §5): what each route-finding estimator *sees* at the two
policies that matter, without training anything.

  uniform  — the router at init (every action 1/4).
  stray    — the policy both rung-1 routers reached in their first 50 steps (Entry 63, F29; diag_deferral0.py):
             HALT at answers; at a key with set-here = 1: HOP 0.99; with a carried count (c >= 1, h = 0): PREV 0.99;
             at a flagged key with (0, 0): SET 0.22, HOP 0.77; elsewhere HOP 0.99 (the remaining mass spread).
             Under deferral 1 this is the oracle rule with a weak SET; under deferral 0 it is the stalled basin.

For each: the fraction of training queries (by chain length k) on which K sampled routes contain a route reaching the
label ("hit") and contain the oracle route ("oracle"), and on which a width-W beam (eps-mixed) returns a correct MAP
route that is the oracle's. Also the exact marginal's posterior share of the oracle route among label-reaching mass.

    PYTHONPATH=src python experiments/routefind/derive/search.py
"""
import sys
import numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(2)
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, oracle, route_ok, sid, PER_SYM, NH, HOP, PREV, SET, HALT
from topology_design.route.estimators import sample_routes, beam_search

def stray_policy(mp, cfg, p_set=0.22, sharp=0.99):
    B, S = mp["flag"].shape
    NS = S * PER_SYM
    ids = torch.arange(NS)
    s, c, h = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
    is_ans = (s >= cfg.n_keys) & (s < cfg.DEAD); dead = s == cfg.DEAD
    fl = mp["flag"][:, s] == 1                                                     # (B, NS)
    pi = torch.full((B, NS, 4), (1 - sharp) / 3)
    def put(mask, a, p):
        rest = (1 - p) / 3
        row = torch.full((4,), rest); row[a] = p
        pi[mask] = row
    key = ~(is_ans | dead)
    put(((is_ans | dead).unsqueeze(0).expand(B, NS)), HALT, sharp)
    put((key & (h == 1)).unsqueeze(0).expand(B, NS), HOP, sharp)
    put((key & (h == 0) & (c >= 1)).unsqueeze(0).expand(B, NS), PREV, sharp)
    m = (key & (h == 0) & (c == 0)).unsqueeze(0) & fl
    row = torch.tensor([1 - p_set - 0.01, 0.005, p_set, 0.005]); pi[m] = row
    put((key & (h == 0) & (c == 0)).unsqueeze(0) & ~fl, HOP, sharp)
    return pi

def table(name, vals):
    print(f"  {name:34s} " + " ".join(f"k={k}:{v:.2f}" for k, v in vals.items()))

for deferral in (1, 0):
    cfg = RouteConfig(deferral=deferral)
    ep = generate(cfg, 400, np.random.default_rng(0)); tb = tables(ep, cfg, "cpu")
    oa, osteps, _ = oracle(tb, tb["q_start"], 14, cfg)
    kk = tb["q_k"]; hasP = (tb["q_word"] == 1).any(-1)
    print(f"\n=== deferral {deferral}: {kk.numel()} training queries; restricted to words WITH a P where marked")
    for pname in ("uniform", "stray"):
        pi = torch.full((400, cfg.n_sym * PER_SYM, 4), 0.25) if pname == "uniform" else stray_policy(tb, cfg)
        print(f" policy {pname}")
        g = torch.Generator().manual_seed(0)
        for K in (4, 16):
            sym, st, a, x = sample_routes(pi, tb, tb["q_start"], 14, K, generator=g)
            hit = (sym == tb["q_ans"].unsqueeze(-1)).any(-1)
            orc = torch.stack([route_ok(a[:, :, j], st[:, :, j], oa, osteps) for j in range(K)], -1).any(-1)
            table(f"K={K} hit (all words)", {k: float(hit[kk == k].float().mean()) for k in range(1, 9)})
            table(f"K={K} oracle among samples (P-words)", {k: float(orc[(kk == k) & hasP].float().mean()) for k in range(3, 9)})
        for W in (4, 16):
            f, acts, steps = beam_search(pi, tb, tb["q_start"], tb["q_ans"], 14, W, cfg, eps_mix=0.1)
            ok = f & route_ok(acts, steps, oa, osteps)
            table(f"W={W} found (all words)", {k: float(f[kk == k].float().mean()) for k in range(1, 9)})
            table(f"W={W} MAP correct = oracle (P-words)", {k: float(ok[(kk == k) & hasP].float().mean()) for k in range(3, 9)})
            table(f"W={W} MAP correct = oracle (all-H)", {k: float(ok[(kk == k) & ~hasP].float().mean()) for k in range(1, 9)})
