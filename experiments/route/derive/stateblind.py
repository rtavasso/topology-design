"""The state-blind references (DESIGN.md §5, revision 2; review 1 B): a step-indexed policy pi(a | t) fitted by the
exact marginal on the training stream, read two ways. Committed: its argmax program (one word for every query), which
succeeds on the queries whose oracle route it is (the fixed-program coverage). Marginal: the Bayes mixture of all
routes under pi(a | t), which can be right on short P-words because the wandering routes spread over wrong answers
while the oracle route's mass lands on the target. Both are the references for P7.

    PYTHONPATH=src python experiments/route/derive/stateblind.py
"""
import sys
import numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(4)
from topology_design.route.train_route import run_stack

cfg = dict(kind="index", lam=0.02, lr=1e-2, steps=800, batch=64, pool_n=8000, eval_n=200, eval_every=800, crit_n=256,
           crit_every=100, T_eval=64, warmup=50, T=14)
recs = run_stack([{**cfg, "seed": 0}, {**cfg, "seed": 1}], dev="cpu", log=lambda s: None)
for r in recs:
    ev = r["evals"][-1]["eval"]
    print(f"seed {r['train']['seed']}: held-out committed ANSWER accuracy by k {({k: round(v, 2) for k, v in r['curve'][-1]['ans_acc_by_k'].items()})}"
          f" (route accuracy is 0 by construction: a state-blind policy cannot halt on the answer)")
    from topology_design.route.data import zipf_pmf
    pmf = zipf_pmf(1.5, 8)
    print(f"  Zipf-weighted in-range: committed answer acc {sum(pmf[k-1] * ev[f'in{k}']['acc'] for k in range(1, 9)):.3f}, "
          f"marginal acc {sum(pmf[k-1] * ev[f'in{k}']['marg_acc'] for k in range(1, 9)):.3f}")
    print("  committed answer acc by stratum: " + " ".join(f"{n}:{ev[n]['acc']:.2f}" for n in ev))
    print("  marginal acc by stratum:         " + " ".join(f"{n}:{ev[n]['marg_acc']:.2f}" for n in ev))
    print("  table p(HALT) by t: " + " ".join(f"{p[3]:.2f}" for p in r["head"]["table"]))
    print("  table argmax by t:  " + " ".join("HPSX"[int(np.argmax(p))] for p in r["head"]["table"]))
