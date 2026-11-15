# Lens 1 — symmetry and gauge (EXPLORATORY)

Linear streams, d = 6, step 0.3, 4 passes, Adam 0.01, 4000 steps, 75 % of programs for training. Predictions in `lens1_symmetry.py`.

**1.1** identical start, 600 steps: largest difference between any two streams = 0.0e+00; held-out error 0.444 (one effective stream).

**1.2** gauge change by a random invertible 4 × 4 matrix: largest change in any output = 1.2e-14.

**1.3** three generic primitives, K = 4 streams, softmax gates, five seeds:

| seed | training error | held-out error | effective-operator error | mean top gate | best stream–primitive cosine |
|---|---|---|---|---|---|
| 0 | 4.6e-32 | 4.9e-32 | 0.000 | 0.85 | 1.00 |
| 1 | 4.9e-32 | 4.8e-32 | 0.000 | 0.87 | 1.00 |
| 2 | 5.2e-32 | 5.4e-32 | 0.000 | 0.80 | 1.00 |
| 3 | 5.7e-32 | 5.8e-32 | 0.000 | 0.80 | 1.00 |
| 4 | 7.1e-32 | 6.7e-32 | 0.000 | 0.84 | 1.00 |

**1.4** four primitives; held-out error by number of streams (three seeds each, median [min, max]):

| primitives | K = 2 | K = 3 | K = 4 |
|---|---|---|---|
| on a line segment (affine dimension 1) | 1.1e-31 [9.1e-32, 2.9e-31] | 1.1e-08 [5.9e-09, 3.5e-08] | 2.1e-21 [2.2e-28, 2.7e-07] |
| generic (affine dimension 3) | 2.4e-01 [2.3e-01, 2.4e-01] | 1.2e-01 [1.0e-01, 1.2e-01] | 9.6e-32 [5.3e-32, 2.4e-31] |
