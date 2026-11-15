# Round B3 — how short the dense start can be (EXPLORATORY)

Softmax scores, register `reg0`; eight streams with five primitives, or four with three. Five seeds, median [min, max]. Predictions in `arch_b3.py`.

| variant | training error | held-out error | solved (<1e-2) | purity | dense steps of 3,000 |
|---|---|---|---|---|---|
| F1 8 streams: dense 10 % then top-1 | 2.6e-05 | 2.8e-05 [1.9e-05, 3.2e-05] | 5/5 | 1.00 | 300 |
| F2 8 streams: dense 3 % then top-1 | 1.2e-01 | 1.7e-01 [8.8e-02, 3.4e-01] | 0/5 | 0.60 | 90 |
| F3 4 streams: dense 3 % then top-1 | 1.5e-01 | 1.9e-01 [3.2e-05, 2.4e-01] | 1/5 | 0.80 | 90 |
| F4 8 streams: dense until confident, then top-1 | 2.2e-05 | 2.2e-05 [2.1e-05, 3.0e-05] | 5/5 | 1.00 | 813 [531, 1271] |
| F5 4 streams: dense until confident, then top-1 | 2.7e-05 | 3.0e-05 [2.3e-05, 3.8e-05] | 5/5 | 1.00 | 443 [306, 916] |
