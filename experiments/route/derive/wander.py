"""The wandering mass (DESIGN.md §5): under a uniform router, how much of the route mass that halts on the target
answer comes from the oracle route, by chain length. Also the fraction of wrong routes that reach *an* answer.

    PYTHONPATH=src python experiments/route/derive/wander.py
"""
import sys
import numpy as np, torch
sys.path.insert(0, "src")
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, marginal, readout, HOP, PREV, HALT

cfg = RouteConfig(); rng = np.random.default_rng(0); dev = "cpu"
R = readout(cfg, 0.0, dev)                                     # exact readout for this count
for k in (1, 2, 3, 4, 6, 8, 12, 16):
    ep = generate(cfg, 200, rng, forced_k=k, forced_kind="train", Qn=3)
    tb = tables(ep, cfg, dev); B = 200; S = cfg.n_sym
    pi = torch.full((B, S, 3), 1 / 3)
    q, es, prof, alive = marginal(pi, tb["hop"], tb["prev"], tb["q_start"][:, :1], max(k + 2, 12), R)
    p_target = q[:, 0].gather(-1, (tb["q_ans"][:, :1] - cfg.n_keys)).squeeze(-1)
    oracle_mass = (1 / 3) ** (k + 1)                              # k correct actions then HALT
    p_any = q[:, 0].sum(-1)                                       # mass that halts on some answer (exact readout)
    print(f"k={k:2d}: P(target | uniform router) = {p_target.mean():.2e}; oracle route's share of it "
          f"{oracle_mass / p_target.mean():.2f}; mass halting on any answer {p_any.mean():.3f}")
