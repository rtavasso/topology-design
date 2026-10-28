"""The wandering mass (DESIGN.md §5, revision 2): under a uniform router over the four actions, how much of the route
mass that halts on the target answer comes from the oracle route, by chain length; and how much mass halts on *an*
answer at all (the rest sits on keys or DEAD when the horizon forces the halt).

    PYTHONPATH=src python experiments/route/derive/wander.py
"""
import sys
import numpy as np, torch
sys.path.insert(0, "src")
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, marginal, readout, oracle, PER_SYM

cfg = RouteConfig(); rng = np.random.default_rng(0); dev = "cpu"
R = readout(cfg, 0.0, dev)                                     # exact readout on answers ...
is_ans = ((torch.arange(cfg.n_sym) >= cfg.n_keys) & (torch.arange(cfg.n_sym) < cfg.DEAD)).repeat_interleave(PER_SYM)
R0 = R * is_ans.unsqueeze(-1)                                  # ... and zero rows elsewhere: q.sum = mass halted on an answer
for k in (1, 2, 3, 4, 6, 8, 12, 16):
    ep = generate(cfg, 200, rng, forced_k=k, forced_kind="train", Qn=3)
    tb = tables(ep, cfg, dev); B = 200
    T = 2 * k + 16
    pi = torch.full((B, cfg.n_sym * PER_SYM, 4), 1 / 4)
    q, es, prof, alive = marginal(pi, tb, tb["q_start"][:, :1], T, R)
    q0, _, _, _ = marginal(pi, tb, tb["q_start"][:, :1], T, R0)
    _, osteps, _ = oracle(tb, tb["q_start"][:, :1], T, cfg)
    p_target = q[:, 0].gather(-1, (tb["q_ans"][:, :1] - cfg.n_keys)).squeeze(-1)
    # the oracle route's mass: (1/4)^osteps to reach the absorbing answer; it then halts on it with mass -> 1 by T
    oracle_mass = (0.25 ** osteps[:, 0].float()) * (1 - 0.75 ** (T - osteps[:, 0].float()))
    print(f"k={k:2d} (oracle steps {osteps[:, 0].float().mean():5.2f}): P(target | uniform router) = {p_target.mean():.2e}; "
          f"oracle route's share of it {float((oracle_mass / p_target).mean()):.2f}; mass halting on any answer {q0[:, 0].sum(-1).mean():.3f}")
