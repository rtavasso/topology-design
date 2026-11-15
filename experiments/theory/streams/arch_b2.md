# Round B2 — sparse training that is never dense; more streams (EXPLORATORY)

Softmax scores, register `reg0`. E1–E5: four streams, three primitives. E6–E10: eight streams, five primitives, 80 training programs of 625. Five seeds, median [min, max]. Predictions in `arch_b2.py`.

| variant | training error | held-out error | solved (<1e-2) | purity | streams used | streams run per pass |
|---|---|---|---|---|---|---|
| E1 top-2, run at top-1 | 3.5e-02 | 3.7e-02 [2.3e-04, 1.3e-01] | 1/5 | 1.00 | 3 | 1.00 |
| E2 top-2 then top-1 | 5.6e-05 | 5.6e-05 [2.1e-05, 9.3e-03] | 5/5 | 1.00 | 3 | 1.00 |
| E3 top-1 + sampled explorer, then top-1 | 6.1e-03 | 1.1e-02 [1.2e-05, 2.4e-02] | 2/5 | 0.62 | 3 | 1.00 |
| E4 top-1 + uniform explorer, then top-1 | 1.9e-05 | 1.9e-05 [1.8e-05, 5.3e-02] | 3/5 | 1.00 | 3 | 1.00 |
| E5 dense 10 % then top-1 | 2.5e-05 | 2.9e-05 [2.4e-05, 3.8e-05] | 5/5 | 1.00 | 3 | 1.00 |
| E6 8 streams: dense | 2.6e-05 | 2.8e-05 [2.5e-05, 3.4e-05] | 5/5 | 1.00 | 5 | 8.00 |
| E7 8 streams: top-2 | 1.2e-02 | 1.6e-02 [9.2e-05, 3.3e-02] | 2/5 | 0.60 | 5 | 2.00 |
| E8 8 streams: top-2 then top-1 | 2.9e-02 | 2.8e-02 [4.5e-05, 8.0e-02] | 1/5 | 0.60 | 5 | 1.00 |
| E9 8 streams: dense then top-1 | 2.1e-05 | 2.3e-05 [1.6e-05, 2.8e-05] | 5/5 | 1.00 | 5 | 1.00 |
| E10 8 streams: top-1 + sampled explorer, then top-1 | 1.3e-02 | 1.6e-02 [1.8e-05, 5.7e-02] | 1/5 | 0.60 | 5 | 1.00 |
