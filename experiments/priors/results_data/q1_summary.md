# Q1 — is the data gap the prior or the search?

176 ungrokking runs, 288 scratch runs.

## Ungrokking: seeds keeping the rule (test ≥ 0.95 throughout the last 2,000 steps) / seeds shrunk

| graph | width | 44 | 87 | 174 | 261 | 348 | 435 | 522 | 696 | 870 | 1044 | 1218 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| asym | base | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 4/4 |
| sym | base | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 4/4 | 3/4 | 4/4 |
| sym | narrow | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| sym | wide | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/4 | 4/4 | 4/4 | 3/4 |

Columns are subset sizes in pairs. 'of n' = runs in the cell, when some never reached sustained grokking before the deadline (excluded). A run that diverged after the shrink counts as not keeping the rule.

## Scratch: seeds with sustained grokking (≥ 0.95 held 2,000 steps) within the budget, per learning rate

| graph | width | lr | 348 | 522 | 696 | 870 | 1044 | 1218 | 1392 |
|---|---|---|---|---|---|---|---|---|---|
| asym | base | 0.001 |  |  |  | 0/3 | 0/3 | 2/3 | 2/3 |
| asym | base | 0.002 |  |  |  | 0/3 | 1/3 | 0/3 | 2/3 |
| asym | base | 0.004 |  |  |  | 0/3 | 0/3 | 0/3 | 0/3 |
| sym | base | 0.002 | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| sym | base | 0.004 | 0/3 | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 |
| sym | base | 0.008 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| sym | base | 0.016 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| sym | narrow | 0.002 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| sym | narrow | 0.004 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| sym | narrow | 0.008 | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| sym | narrow | 0.016 | 0/3 | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 |
| sym | wide | 0.002 | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| sym | wide | 0.004 | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| sym | wide | 0.008 | 0/3 | 0/3 | 0/3 | 1/3 | 2/3 | 3/3 | 3/3 |
| sym | wide | 0.016 | 0/3 | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 |

## Decomposition (matched criteria)

| graph | width | n_crit | n_floor (best lr) | S = n_floor / n_crit | search share | flags |
|---|---|---|---|---|---|---|
| asym | base | 1218 | 1218 at lr 0.001 | 1.00 | 0.00 | budget_sensitive |
| sym | base | 870 | 870 at lr 0.004 | 1.00 | 0.00 |  |
| sym | narrow | 522 | 870 at lr 0.016 | 1.67 | 0.09 | best_lr_at_edge, budget_sensitive |
| sym | wide | 696 | 870 at lr 0.016 | 1.25 | 0.04 | best_lr_at_edge |

Search share = log S / log(n_floor / 3) is reported as a measurement, not a verdict.

## Verdicts (preregistered)
- **P1 (ungrokking exists, sym base):** fewer than half the seeds keep the rule at 174 pairs, at least three quarters at 1,218: 0.00 and 1.00 → **supported**
- **P2 (search accounts for at most ~3×, sym base):** S ≤ 3 supported, S ≥ 4 falsified: S = 1.00 → SUPPORTED
- *Replication check (Varma et al.: S ≈ 2):* S ≥ 1.5 → no
- **P3 (wiring acts on search, base):** n_crit(asym)/n_crit(sym) < n_floor(asym)/n_floor(sym) → INCONCLUSIVE (tie)
- **P4 (width; prediction C):** ordering of n_crit = non-monotone or indeterminate (ties/censoring) → NOT SUPPORTED
- **P4b (generic width does not close the gap):** every width's n_floor > 348 pairs → **supported**
- *Sensitivity:* S under the mismatched criteria (median final ≥ 0.5 / first crossing) is in the json.
- *Exploratory (json):* dropped-pair accuracy, weight norms, rule fraction at the shrink, semi-grokked runs, collapses after the shrink, half-budget floors.
