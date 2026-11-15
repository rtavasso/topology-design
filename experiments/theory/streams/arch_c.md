# Round C — a gate that must read the data state (EXPLORATORY)

Branch task: three primitives plus a token IF (primitive 0 if ⟨w, h⟩ > 0, else primitive 1); 64 of 256 programs for training; four streams; softmax scores; register `reg0` unless stated. Gain = largest singular value of ∂h₄/∂h₀ over 96 held-out cases (median over seeds of the per-seed median and maximum; in brackets the largest maximum). Five seeds, median [min, max]. Predictions in `arch_c.py`.

| variant | training error | held-out error | solved (<1e-2) | gain: median | gain: max | teacher gain: median, max | streams run per pass |
|---|---|---|---|---|---|---|---|
| C0 gate blind to the data | 9.5e-02 | 1.1e-01 [1.0e-01, 1.1e-01] | 0/5 | 1.97 | 4.08 [5.47] | 2.03, 4.88 | 4.00 |
| C1 gate reads normalised data | 2.2e-03 | 2.1e-03 [1.5e-03, 1.3e-02] | 3/5 | 2.06 | 11.21 [14.30] | 2.03, 4.88 | 4.00 |
| C2 gate reads raw data | 1.4e-03 | 1.5e-03 [1.3e-03, 1.8e-03] | 5/5 | 2.06 | 11.29 [12.61] | 2.03, 4.88 | 4.00 |
| C3 C1 + logit penalty | 1.7e-03 | 1.6e-03 [1.6e-03, 1.2e-02] | 4/5 | 2.05 | 10.81 [11.28] | 2.03, 4.88 | 4.00 |
| C4 clock + data | 1.5e-03 | 1.5e-03 [1.2e-03, 8.4e-03] | 5/5 | 2.11 | 11.06 [13.43] | 2.03, 4.88 | 4.00 |
| C5 data query, no register | 2.7e-02 | 3.1e-02 [3.1e-02, 3.3e-02] | 0/5 | 2.10 | 5.30 [8.70] | 2.03, 4.88 | 4.00 |
| C6 C1 dense then top-1 | 1.4e-03 | 1.6e-03 [9.2e-04, 1.3e-02] | 3/5 | 2.06 | 7.66 [8.94] | 2.03, 4.88 | 1.00 |
| C7 C1 top-2 | 1.2e-02 | 1.3e-02 [1.3e-03, 2.6e-02] | 1/5 | 2.07 | 6.71 [10.41] | 2.03, 4.88 | 2.00 |
| C8 program task, gate reads data | 2.7e-05 | 2.7e-05 [2.3e-05, 4.2e-05] | 5/5 | 1.98 | 4.30 [4.52] | 1.98, 4.26 | 4.00 |
| C9 C1 at lr 1e-2 | 4.0e-03 | 5.6e-03 [1.7e-03, 1.3e-02] | 3/5 | 2.06 | 19.86 [23.48] | 2.03, 4.88 | 4.00 |
| C10 C2 at lr 1e-2 | 2.1e-03 | 1.7e-03 [1.2e-03, 3.3e-03] | 5/5 | 2.06 | 16.02 [16.71] | 2.03, 4.88 | 4.00 |
