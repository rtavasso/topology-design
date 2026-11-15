# Round B — gating before the streams (EXPLORATORY)

Program task as in round A, register `reg0`, four streams; evaluated with only the top-k streams computed. Five seeds, median [min, max]. Predictions in `arch_b.py`.

| variant | training error | held-out error | solved (<1e-2) | purity | streams used | streams run per pass |
|---|---|---|---|---|---|---|
| B1 top-1 softmax renormalised | 2.9e-01 | 3.6e-01 [2.9e-01, 4.0e-01] | 0/5 | 0.78 | 2 | 1.00 |
| B2 top-1 softmax | 1.8e-01 | 2.3e-01 [1.9e-01, 3.9e-01] | 0/5 | 0.67 | 2 | 1.00 |
| B3 top-1 sigmoid | 1.4e-01 | 1.8e-01 [2.4e-02, 3.9e-01] | 0/5 | 0.63 | 3 | 1.00 |
| B4 top-2 sigmoid | 7.0e-03 | 1.0e-02 [3.9e-05, 2.4e-02] | 2/5 | 0.66 | 3 | 2.00 |
| B5 top-2 softmax | 4.0e-05 | 4.1e-05 [2.1e-05, 5.7e-03] | 5/5 | 1.00 | 3 | 2.00 |
| B6 top-2 sigmoid + dropout + entropy | 1.3e-02 | 1.6e-02 [9.1e-03, 2.4e-02] | 1/5 | 0.68 | 4 | 2.00 |
| B7 top-1 sigmoid + dropout 0.2 | 3.5e-02 | 7.4e-02 [2.2e-02, 1.0e-01] | 0/5 | 0.58 | 4 | 1.00 |
| B8 top-1 softmax + noise | 3.1e-01 | 3.1e-01 [2.7e-01, 4.3e-01] | 0/5 | 0.79 | 2 | 1.00 |
| B9 dense then top-1 | 2.8e-05 | 2.8e-05 [1.7e-05, 5.3e-05] | 5/5 | 1.00 | 3 | 1.00 |
| B10 top-1 + running-mean stand-ins | 1.8e-01 | 2.6e-01 [2.0e-01, 3.9e-01] | 0/5 | 0.68 | 2 | 1.00 |
| B11 top-1 + linear stand-ins | 1.5e-01 | 2.2e-01 [5.6e-05, 3.9e-01] | 1/5 | 1.00 | 2 | 1.00 |
| B12 dense sigmoid | 7.6e-05 | 8.5e-05 [5.5e-05, 1.0e-04] | 5/5 | 1.00 | 3 | 4.00 |
