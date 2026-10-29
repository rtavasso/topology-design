"""Derived before measuring (experiments/routefind/DESIGN.md §5): what each route-finding estimator *sees* at the two
policies that matter, without training anything.

  init     — the actual router at init, per seed (per-run init as in the grids: seed -> 1000003 * seed + 7), with
             ties broken by the router's own logits (code review 1, finding 1 / design review 1, B1: an exactly
             uniform policy makes every prefix tie and the beam's answer an artefact of sort order).
  stray    — the policy both rung-1 routers reached in their first 50 steps (Entry 63, F29; diag_deferral0.py):
             HALT at answers; at a key with set-here = 1: HOP 0.99; with a carried count (c >= 1, h = 0): PREV 0.99;
             at a flagged key with (0, 0): SET 0.22, HOP 0.77; elsewhere HOP 0.99 (the remaining mass spread).
             Under deferral 1 this is the oracle rule with a weak SET; under deferral 0 it is the stalled basin.

For each: the fraction of training queries (by chain length k) on which K sampled routes contain a route reaching the
label ("hit") and contain the oracle route ("oracle"), and on which a width-W beam (eps-mixed) returns a correct MAP
route that is the oracle's. Also the exact marginal's posterior share of the oracle route among label-reaching mass.

    PYTHONPATH=src python experiments/routefind/derive/search.py [ans_pool]
"""
import sys
import numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(2)
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, oracle, route_ok, sid, PER_SYM, NH, HOP, PREV, SET, HALT
from topology_design.route.estimators import sample_routes, beam_search, best_correct_sample, route_logprob
from topology_design.route.router import StackedRouter, RouterConfig
from topology_design.route.pool import to_dict
from topology_design.route.train_route import per_run_init_, policy, flat

SEEDS = range(5)

def init_policies(ep, cfg):
    """(seed -> pi (B, NS, 4)) for the router at init."""
    rc = RouterConfig()
    m = StackedRouter(len(SEEDS), rc, cfg)
    per_run_init_(m, list(SEEDS), rc, cfg)
    d = to_dict(ep, "cpu")
    b = {k: v.unsqueeze(0).expand(len(SEEDS), *v.shape) for k, v in d.items()}
    with torch.no_grad():
        pis = policy(m, b, rc.T, chunk=50).view(len(SEEDS), ep.B, -1, 4)
    return {sd: pis[i] for i, sd in enumerate(SEEDS)}

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

def measure(pi, tb, cfg, oa, osteps, kk, hasP, g):
    out = {}
    for K in (4, 16):
        sym, st, a, x = sample_routes(pi, tb, tb["q_start"], 14, K, generator=g)
        oks = torch.stack([route_ok(a[:, :, j], st[:, :, j], oa, osteps) for j in range(K)], -1)
        ok, idx = best_correct_sample(sym, tb["q_ans"], route_logprob(pi, a, x), st, cfg_lam)
        kept = ok & oks.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        out[f"K{K} hit"] = (sym == tb["q_ans"].unsqueeze(-1)).any(-1)
        out[f"K{K} oracle among samples"] = oks.any(-1)
        out[f"K{K} SI kept = oracle"] = kept
    for W in (4, 16):
        f, acts, steps = beam_search(pi, tb, tb["q_start"], tb["q_ans"], 14, W, cfg, eps_mix=0.1)
        out[f"W{W} found"] = f
        out[f"W{W} MAP = oracle"] = f & route_ok(acts, steps, oa, osteps)
    return out

cfg_lam = 0.02
ZW = None
ANS_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 4 = the spurious-density dose (revision 4)
for deferral in (1, 0):
    cfg = RouteConfig(deferral=deferral, ans_pool=ANS_POOL)
    ep = generate(cfg, 200, np.random.default_rng(0)); tb = tables(ep, cfg, "cpu")
    oa, osteps, _ = oracle(tb, tb["q_start"], 14, cfg)
    kk = tb["q_k"]; hasP = (tb["q_word"] == 1).any(-1)
    print(f"\n=== deferral {deferral}: {kk.numel()} training queries; P-word rows restricted to words WITH a P (k = 3 / 4 / 5 / 8)")
    pols = {f"init s{sd}": p for sd, p in init_policies(ep, cfg).items()}
    pols["stray"] = stray_policy(tb, cfg)
    rows = {}
    for pname, pi in pols.items():
        g = torch.Generator().manual_seed(0)
        res = measure(pi, tb, cfg, oa, osteps, kk, hasP, g)
        for name, x in res.items():
            pk = [float(x[(kk == k) & hasP].float().mean()) for k in (3, 4, 5, 8)]
            allk = [float(x[kk == k].float().mean()) for k in (1, 2, 3, 8)]
            rows.setdefault(name, []).append((pname, pk, allk))
    for name, lst in rows.items():
        print(f" {name}")
        for pname, pk, allk in lst:
            print(f"   {pname:8s} P-words {' '.join(f'{v:.2f}' for v in pk)}   | all words k=1/2/3/8 {' '.join(f'{v:.2f}' for v in allk)}")
