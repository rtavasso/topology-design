# Stage 0 — the two monoliths, no streams (EXPLORATORY)

Design and predictions R0a–R0e: `DESIGN.md` (revision 2, committed before these runs at d2ace8f). Six splits (three teachers), 4,000 steps unless stated. Errors are median [min, max] over runs. References are exact, from the teacher: `allmean` ignores the world, `blend` knows the two primitives and cannot choose, `trans` chooses wherever trained comparisons force the choice. Regimes as in DESIGN.md §2. The last two columns: after one pass on held-out worlds, the squared distance of the block's update to the true update and to the blend, relative to the true update's size.

## Rates (coverage 9, three other splits)

| run | seen | held out | references (median) | regime (runs) | one pass on held-out worlds: distance to the true update | distance to the blend |
|---|---|---|---|---|---|---|
| select mono width 40 rate 0.003 | 0.13 [0.091, 0.13] | 1.5 [1.1, 1.7] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 1.48 | 1.21 |
| select mono width 40 rate 0.01 | 0.093 [0.069, 0.094] | 1.3 [1.1, 1.4] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 1.23 | 0.89 |
| select mono width 40 rate 0.03 | 0.075 [0.054, 0.08] | 1.2 [1.1, 1.4] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 1.12 | 0.79 |
| select mono width 40 rate 0.1 | 0.38 [0.23, 1.1] | 1.5 [1.2, 12] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 712.66 | 717.87 |
| select mono width 120 rate 0.003 | 0.05 [0.036, 0.052] | 0.97 [0.9, 1] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 0.98 | 0.78 |
| select mono width 120 rate 0.01 | 0.031 [0.023, 0.031] | 0.87 [0.76, 0.87] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 2, worse-than-blend 1 | 0.86 | 0.64 |
| select mono width 120 rate 0.03 | 0.14 [0.11, 0.63] | 1.1 [0.87, 1.7] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 21.87 | 21.54 |
| select mono width 120 rate 0.1 | 0.44 [0.36, 0.94] | 1 [0.82, 42] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 1634.63 | 1639.44 |
| select tok_mono width 40 rate 0.003 | 0.04 [0.03, 0.042] | 0.071 [0.063, 0.074] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | full 3 | 0.10 | 0.38 |
| select tok_mono width 40 rate 0.01 | 0.024 [0.019, 0.028] | 0.051 [0.049, 0.051] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | full 3 | 0.08 | 0.40 |
| select tok_mono width 40 rate 0.03 | 0.025 [0.018, 0.028] | 0.058 [0.053, 0.077] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | beyond 1, full 2 | 0.09 | 0.40 |
| select tok_mono width 40 rate 0.1 | 0.98 [0.42, 1] | 1 [0.91, 1] | allmean 0.78; one 0.96; blend 0.46; trans 0.21 | garbage 3 | 1827.09 | 1825.40 |
| sum mono width 40 rate 0.003 | 0.015 [0.013, 0.018] | 0.94 [0.86, 1.1] | allmean 0.57; one 0.48 | garbage 3 | 0.98 | 0.98 |
| sum mono width 40 rate 0.01 | 0.0066 [0.006, 0.0071] | 1.1 [0.89, 1.2] | allmean 0.57; one 0.48 | garbage 3 | 1.10 | 1.10 |
| sum mono width 40 rate 0.03 | 0.0031 [0.0029, 0.0037] | 1.1 [0.99, 1.1] | allmean 0.57; one 0.48 | garbage 3 | 1.08 | 1.08 |
| sum mono width 40 rate 0.1 | 0.0045 [0.0041, 0.0046] | 1.1 [1, 1.1] | allmean 0.57; one 0.48 | garbage 3 | 1.04 | 1.04 |
| sum tok_mono width 40 rate 0.003 | 0.0089 [0.0085, 0.01] | 0.023 [0.013, 0.024] | allmean 0.57; one 0.48 | full 3 | 0.02 | 0.02 |
| sum tok_mono width 40 rate 0.01 | 0.0022 [0.002, 0.0022] | 0.005 [0.0029, 0.005] | allmean 0.57; one 0.48 | full 3 | 0.00 | 0.00 |
| sum tok_mono width 40 rate 0.03 | 0.001 [0.00097, 0.0013] | 0.002 [0.0015, 0.0032] | allmean 0.57; one 0.48 | full 3 | 0.00 | 0.00 |
| sum tok_mono width 40 rate 0.1 | 0.001 [0.00082, 0.0011] | 0.0022 [0.0019, 0.0026] | allmean 0.57; one 0.48 | full 3 | 0.00 | 0.00 |
| sum tok_mono width 40 rate 0.3 | 73 [16, 4.6e+02] | 3.3e+02 [46, 4.4e+02] | allmean 0.57; one 0.48 | garbage 3 | 24067.61 | 24067.61 |

Chosen by seen error: select — mono 0.03 (width 40), 0.01 (width 120), tok_mono 0.01; sum — mono 0.03, tok_mono 0.1 (0.3 diverges).

## The dial

| run | seen | held out | references (median) | regime (runs) | one pass on held-out worlds: distance to the true update | distance to the blend |
|---|---|---|---|---|---|---|
| select mono rate 0.03, coverage 7 | 0.054 [0.05, 0.058] | 1.3 [1.1, 1.6] | allmean 0.80; one 0.99; blend 0.48; trans 0.28 | garbage 6 | 1.31 | 0.95 |
| select mono rate 0.03, coverage 9 | 0.066 [0.06, 0.07] | 1.3 [1, 1.7] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | garbage 6 | 1.39 | 1.02 |
| select mono rate 0.03, coverage 11 | 0.082 [0.073, 0.088] | 1.2 [1, 1.7] | allmean 0.80; one 1.00; blend 0.49; trans 0.15 | garbage 6 | 1.28 | 0.92 |
| select mono rate 0.03, coverage 13 | 0.084 [0.076, 0.088] | 1.1 [0.85, 1.7] | allmean 0.80; one 0.98; blend 0.49; trans 0.11 | garbage 6 | 1.18 | 0.86 |
| select mono rate 0.03, coverage 15 | 0.093 [0.075, 0.1] | - | - | - | - | - |
| select tok_mono rate 0.01, coverage 7 | 0.024 [0.02, 0.027] | 0.076 [0.052, 0.1] | allmean 0.80; one 0.99; blend 0.48; trans 0.28 | beyond 3, full 3 | 0.10 | 0.41 |
| select tok_mono rate 0.01, coverage 9 | 0.024 [0.02, 0.026] | 0.058 [0.031, 0.078] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | beyond 1, full 5 | 0.08 | 0.41 |
| select tok_mono rate 0.01, coverage 11 | 0.025 [0.022, 0.027] | 0.043 [0.021, 0.053] | allmean 0.80; one 1.00; blend 0.49; trans 0.15 | full 6 | 0.07 | 0.44 |
| select tok_mono rate 0.01, coverage 13 | 0.024 [0.021, 0.025] | 0.03 [0.02, 0.06] | allmean 0.80; one 0.98; blend 0.49; trans 0.11 | full 6 | 0.06 | 0.44 |
| select tok_mono rate 0.01, coverage 15 | 0.023 [0.022, 0.025] | - | - | - | - | - |
| sum mono rate 0.03, coverage 7 | 0.0028 [0.0025, 0.0032] | 1.3 [0.79, 1.4] | allmean 0.61; one 0.50 | garbage 6 | 1.25 | 1.25 |
| sum mono rate 0.03, coverage 9 | 0.0038 [0.0032, 0.0044] | 1 [0.88, 1.5] | allmean 0.61; one 0.49 | garbage 6 | 1.03 | 1.03 |
| sum mono rate 0.03, coverage 15 | 0.0075 [0.007, 0.011] | - | - | - | - | - |
| sum tok_mono rate 0.1, coverage 7 | 0.001 [0.0009, 0.0014] | 0.0025 [0.0019, 0.0041] | allmean 0.61; one 0.50 | full 5, recombines 1 | 0.00 | 0.00 |
| sum tok_mono rate 0.1, coverage 9 | 0.0009 [0.00078, 0.0011] | 0.0016 [0.0013, 0.0021] | allmean 0.61; one 0.49 | full 6 | 0.00 | 0.00 |
| sum tok_mono rate 0.1, coverage 15 | 0.00075 [0.0005, 0.00084] | - | - | - | - | - |

## Fit and length of training (select, coverage 9)

| run | seen | held out | references (median) | regime (runs) | one pass on held-out worlds: distance to the true update | distance to the blend |
|---|---|---|---|---|---|---|
| select mono rate 0.03, coverage 9, 8,000 steps | 0.055 [0.05, 0.057] | 1.5 [1.1, 1.7] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | garbage 6 | 1.48 | 1.08 |
| select mono rate 0.01, coverage 9, width 120 | 0.029 [0.028, 0.032] | 1 [0.9, 1.1] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | garbage 6 | 1.07 | 0.78 |
| select mono rate 0.01, coverage 9, width 120, 8,000 steps | 0.022 [0.02, 0.024] | 1 [0.94, 1.1] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | garbage 6 | 1.02 | 0.73 |
| select tok_mono rate 0.01, coverage 9, 8,000 steps | 0.018 [0.016, 0.02] | 0.053 [0.025, 0.074] | allmean 0.80; one 0.95; blend 0.46; trans 0.20 | beyond 4, full 2 | 0.07 | 0.42 |

## Control: all primitives identical, so the world is irrelevant (select, coverage 9, three splits)

| run | seen | held out | references (median) | regime (runs) | one pass on held-out worlds: distance to the true update | distance to the blend |
|---|---|---|---|---|---|---|
| select mono rate 0.03, irrelevant world, coverage 9 | 5.8e-06 [4.9e-06, 6.1e-06] | 5.3e-06 [5.3e-06, 8.5e-06] | - | held out within 1.5x of seen (or below 1e-3): 3/3 | - | - |
| select tok_mono rate 0.01, irrelevant world, coverage 9 | 1.3e-05 [1.1e-05, 1.3e-05] | 1.7e-05 [1.7e-05, 2.3e-05] | - | held out within 1.5x of seen (or below 1e-3): 3/3 | - | - |

## Reading

- **A monolith that reads the world as one summed vector is in the garbage regime on held-out worlds at every coverage** (held-out error 1.1–1.3 on `select`, above not moving at all; 6/6 splits each), at width 40 and at width 120, where its error on seen worlds matches the per-token monolith's (0.022 against 0.018–0.024). It is not under-fitting, and it is not the unfamiliar input alone: when the world is irrelevant its held-out error equals its seen error (5e-6). It is an unconstrained function of a vector it has not seen.
- **A monolith that runs once per token and pools the branches by their bids recombines** — the same weights, the same parameter count. On `select` its held-out error is 0.076, 0.058, 0.043, 0.030 at coverage 7, 9, 11, 13 against 0.024 on seen worlds; that is below half of what the trained comparisons force (`trans` 0.28 → 0.11) in every run, and within 3× of its seen error in 3, 5, 6, 6 of 6 splits. On `sum` it is within 3× in 5/6 and 6/6.
- So on this carrier whether "a monolithic block fails on unseen combinations" is decided by how it reads the set, before any question of streams. What is left for streams to fix under the per-token read is a factor of about 3 at coverage 7 and 2.4 at coverage 9.
- Twice as many steps changes seen error by 1.2–1.3×: 4,000 steps are kept.

Predictions: R0a held (garbage 6/6 at both widths). R0b held. R0c partly (recombines 6/6, below `trans` 6/6; but *full* in 5/6 where I predicted at most 2). R0d partly (`sum`: full 5/6 at coverage 7, 6/6 at 9). R0e held.

Stop rule: `mono` is not full, so stage 2 runs. `tok_mono` leaves room only at coverages 7 and 9, so stage 2 uses those (DESIGN.md §6 had named 9 and 11 with 7 secondary; changed here, before stage 2, on stage 0's evidence).
