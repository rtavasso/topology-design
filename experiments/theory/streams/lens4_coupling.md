# Lens 4 — separate stream states coupled through a graph (EXPLORATORY)

Random tanh streams, K = 8, d = 8, step 0.3, 8 passes. Predictions in `lens4_coupling.py`.

Laplacian eigenvalues (second smallest, largest): none 0.00, 0.00; ring 0.59, 4.00; star 1.00, 8.00; complete 8.00, 8.00.

**4.1, 4.3** diversity after 8 passes:

| coupling c | none | ring | star | complete | complete, predicted from the uncoupled value |
|---|---|---|---|---|---|
| 0.02 | 3.86 | 3.34 | 3.45 | 2.33 | 2.27 |
| 0.05 | 3.86 | 2.78 | 3.06 | 1.23 | 1.19 |
| 0.1 | 3.86 | 2.2 | 2.58 | 0.626 | 0.604 |
| 0.2 | 3.86 | 1.61 | 1.9 | 0.309 | 0.297 |
| 0.3 | 3.86 | 1.29 | 2.27 | 4.17 | -2.77 |

**4.2** does stream k's state after n passes depend on stream j's parameters? (ring: j = 0; entries are the largest absolute derivative)

- ring: n=1: 4e+00, 0, 0, 0, 0 | n=2: 8e+00, 4e-01, 0, 0, 0 | n=3: 1e+01, 1e+00, 3e-02, 0, 0 | n=4: 2e+01, 2e+00, 1e-01, 4e-03, 0 | n=5: 2e+01, 3e+00, 3e-01, 2e-02, 9e-04   (streams k = 0, 1, 2, 3, 4)
- complete: n=1: 4e+00, 0, 0, 0, 0 | n=2: 6e+00, 4e-01, 4e-01, 4e-01, 4e-01 | n=3: 7e+00, 8e-01, 9e-01, 8e-01, 9e-01 | n=4: 7e+00, 1e+00, 1e+00, 1e+00, 1e+00 | n=5: 9e+00, 2e+00, 2e+00, 2e+00, 2e+00   (streams k = 0, 1, 2, 3, 4)

**4.4** gain of the loop for a small perturbation of the input states (complete graph):

| coupling c | perturbation applied to all streams equally | perturbation of one stream against the rest |
|---|---|---|
| 0.0 | 2.16 | 2.57 |
| 0.02 | 1.69 | 0.83 |
| 0.05 | 1.42 | 0.14 |
| 0.1 | 1.31 | 0.03 |
| 0.2 | 1.27 | 0.09 |
