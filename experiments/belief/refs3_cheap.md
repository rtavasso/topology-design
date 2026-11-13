<!-- draw 7 done -->
<!-- draw 8 done -->
<!-- draw 9 done -->
# Reference numbers for design revision 3 — two cheap mechanisms that hold no belief (model-free)

H8, 80 held-out-world instances per draw, three draws. Definitions in `refs3.py` (`cheap_part`).

Rule table: fitted on 1058 probes of the exact full-prior posterior from 80 other instances; squared error per probe 0.0285 for plain deletion, 0.0235 with the fitted table.

Why RM_r is applied to both of the model's outputs: with dp = p(E + r−) − RM_r[p(E)], a model that ignores the verdict (p = q(E) both times) reads 0.09 on the single direction dq; with RM_r on both it reads 0 exactly.

Single-direction readings (levels: (p − q_T) on (q − q_T); changes: on (dq, g) out of support and (dq_T, dq) in support) — the statistics as they stood before the planner was named.

| draw | training worlds | optimistic planner: gamma | optimistic planner: beta_cf out of support | optimistic planner: in-support probes (dq_T, dq) | rule table: beta_cf out of support | rule table: in-support probes (dq_T, dq) |
|---|---|---|---|---|---|---|
| 7 | 4 | 1.26 | 1.31 | -0.23, 1.41 | 0.14 | 0.03, 0.11 |
| 7 | 8 | 1.26 | 1.34 | -0.09, 1.31 | 0.14 | 0.02, 0.11 |
| 7 | 16 | 1.12 | 1.36 | -0.07, 1.31 | 0.13 | -0.01, 0.14 |
| 7 | 32 | 0.97 | 1.40 | -0.04, 1.29 | 0.12 | -0.03, 0.15 |
| 7 | 64 | 1.79 | 2.00 | -0.14, 1.42 | 0.03 | -0.02, 0.16 |
| 8 | 4 | 0.93 | 1.25 | -0.00, 1.02 | 0.14 | -0.09, 0.07 |
| 8 | 8 | 1.15 | 1.13 | -0.05, 1.24 | 0.16 | -0.03, 0.09 |
| 8 | 16 | 1.28 | 1.07 | -0.07, 1.28 | 0.15 | -0.01, 0.09 |
| 8 | 32 | 1.07 | 0.77 | -0.00, 1.20 | 0.41 | -0.02, 0.12 |
| 8 | 64 | 1.48 | 0.00 | -0.20, 1.39 | 0.41 | -0.01, 0.11 |
| 9 | 4 | 0.09 | 1.14 | 0.18, 1.28 | 0.18 | -0.07, 0.11 |
| 9 | 8 | 0.54 | 1.13 | 0.10, 1.30 | 0.18 | -0.03, 0.14 |
| 9 | 16 | 0.77 | 1.00 | -0.01, 1.35 | 0.22 | -0.02, 0.14 |
| 9 | 32 | 0.29 | 1.24 | 0.08, 1.19 | 0.37 | -0.01, 0.15 |
| 9 | 64 | 0.25 | 1.47 | 0.07, 1.20 | 0.40 | -0.01, 0.16 |
