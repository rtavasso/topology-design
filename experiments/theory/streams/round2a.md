# Round 2 — revised predictions, tested (EXPLORATORY)

Predictions in `round2.py`.

**R1** three primitives, K = 4 streams; for each primitive the best |cosine| with any stream (three seeds):

| gates | held-out error (median) | smallest of the three best cosines, by seed |
|---|---|---|
| softmax | 8.3e-32 | 0.99, 0.99, 0.99 |
| unnormalised | 4.7e-32 | 0.77, 0.90, 0.85 |

**R2** order-dependent share of the gate gradient, computed program by program (mean over programs and five draws):

| step size eps | share | ratio to the previous row |
|---|---|---|
| 0.05 | 0.020 | – |
| 0.1 | 0.040 | 2.00 |
| 0.2 | 0.095 | 2.40 |
| 0.4 | 0.281 | 2.94 |

**R4 (i)** median gain of the six-pass loop: uniform gate 2.97; soft gate at tau = 4: 3.04; soft gate at tau = 0.05: 7.75; single streams alone: 5.17.

**R4 (ii)** gain for a common perturbation: separate states, complete graph, c = 1/K: 1.28; shared state adding the mean of the streams: 1.24.

**R5** mean per-input gate-gradient norm by (smallest margin) / tau:

| tau | margin < tau (share of inputs) | tau to 5 tau | > 5 tau (share of inputs) | ratio of the last to the first |
|---|---|---|---|---|
| 0.05 | 155 (40%) | 57.8 | 0.744 (22%) | 0.005 |
| 0.02 | 417 (18%) | 121 | 1.02 (43%) | 0.002 |

**R6** diversity after eight passes: measured / predicted from the whole Laplacian spectrum:

| coupling c | ring | star | complete |
|---|---|---|---|
| 0.02 | 3.34 / 3.33 (+1%) | 3.45 / 3.44 (+0%) | 2.33 / 2.27 (+2%) |
| 0.05 | 2.78 / 2.76 (+1%) | 3.06 / 3.04 (+0%) | 1.23 / 1.19 (+4%) |
| 0.1 | 2.20 / 2.18 (+1%) | 2.58 / 2.56 (+1%) | 0.63 / 0.60 (+4%) |
| 0.2 | 1.61 / 1.59 (+1%) | 1.90 / 1.86 (+2%) | 0.31 / 0.30 (+4%) |
