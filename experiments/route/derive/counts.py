"""Counts per (circuit, depth) in the training stream (DESIGN.md §5): P-steps and H-steps at each depth per 100
training steps of 64 episodes x 6 queries, from the word family and the Zipf lengths.

    PYTHONPATH=src python experiments/route/derive/counts.py
"""
import sys
import numpy as np
sys.path.insert(0, "src")
from topology_design.route.data import RouteConfig, generate, P

cfg = RouteConfig(); rng = np.random.default_rng(0)
ep = generate(cfg, 6400, rng)                                   # 100 training steps of 64 episodes
ws = [ep.q_word[b, j, :ep.q_k[b, j]] for b in range(6400) for j in range(cfg.Qn)]
print("per 100 training steps (38,400 queries): steps at depth d that are P / H, and queries with k >= d")
for d in range(1, cfg.kmax + 1):
    at = [w[d - 1] for w in ws if len(w) >= d]
    print(f"  depth {d}: P {sum(a == P for a in at):6d}   H {sum(a != P for a in at):6d}   queries reaching depth {len(at):6d}")
print(f"queries with any P: {np.mean([(w == P).any() for w in ws]):.3f}; P-steps overall: {np.mean([a for w in ws for a in w]):.3f} of steps")
