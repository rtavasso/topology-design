"""Power of a proposed replacement for beta: counterfactual single-rejection probes at a FIXED state.
For a decision (state, budget, P, A) on the q-greedy trajectory and every rule r not yet tested that has a valid action
there and that some surviving world lacks:  dq_r = q(. | A + r) - RM_r[q(. | A)], where RM_r deletes all actions of rule r
and renormalises.  dq_r is the part of the exact update on 'r rejected' that is NOT a rule-identity lookup and that cannot
be inferred from the state (the state is the same on both sides).  A model's dp_r is formed from its own two outputs."""
import random, sys
import numpy as np
from core import *
from reftrace import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(0)
rows = []
for it in range(n):
    ins = Inst(rng)
    recs, _ = run(ins, "q", order="power", extra=False)
    for d in recs:
        v, b, P, A, q = d["node"], d["b"], d["P"], d["A"], d["q"]
        e0, e1 = ins.estart[v], ins.estart[v + 1]
        rules = ins.er[e0:e1]
        wm = ins.wmask(v, b, P, A)
        pw = ins.post(wm)
        for r in sorted(set(int(x) for x in rules)):
            if (P | A) >> r & 1:
                continue
            wr = wm & ~HAS[:, r]
            if not wr.any():
                continue
            q1 = ins.mix(v, wr)
            rm = masked(q, [j for j in range(e1 - e0) if rules[j] == r])
            if q1 is None or rm is None:
                continue
            rows.append((it, d["calls"] == 0, float(pw[wr].sum()), tv(q1, rm), float(((q1 - rm) ** 2).sum()),
                         bool(q[[j for j in range(e1 - e0) if rules[j] == r]].sum() > 0)))
R = np.array([(a, b, c, d, e, f) for a, b, c, d, e, f in rows], dtype=float)
for name, sel in (("all decisions on the q-greedy trajectory", np.ones(len(R), bool)), ("first decision only", R[:, 1] == 1),
                  ("first decision, rule has posterior mass among the optimal actions", (R[:, 1] == 1) & (R[:, 5] == 1))):
    S = R[sel]
    print(f"{name}: {len(S)} probes ({len(S)/n:.1f} per instance); mean P(rule absent | evidence) {S[:,2].mean():.2f}; "
          f"TV(exact, rule mask of the previous exact) mean {S[:,3].mean():.3f}, > 0.05 on {np.mean(S[:,3]>0.05):.2f}, > 0.10 on {np.mean(S[:,3]>0.10):.2f}; "
          f"energy per instance {S[:,4].sum()/n:.3f}; weighted by P(absent): mean TV {(S[:,3]*S[:,2]).sum()/S[:,2].sum():.3f}")
