# Round A — what tells a tied gate which pass it is in (EXPLORATORY)

Program task: three primitives, four passes, step 0.4; 40 programs for training, 41 held out; four streams; dense softmax gates. Five seeds, median [min, max]. Predictions in `arch_a.py`; reading in `docs/moe_in_loops.md`.

| variant | training error | held-out error | solved (<1e-2) | purity | streams used | streams run per pass |
|---|---|---|---|---|---|---|
| A0 none | 2.0e-02 | 3.0e-02 [2.6e-02, 3.1e-02] | 0/5 | 0.57 | 4 | 4.00 |
| A1 clock | 2.4e-05 | 2.6e-05 [2.3e-05, 5.0e-05] | 5/5 | 1.00 | 3 | 4.00 |
| A2 data | 2.0e-02 | 3.1e-02 [2.6e-02, 3.1e-02] | 0/5 | 0.55 | 4 | 4.00 |
| A3 reg0 | 2.7e-05 | 2.7e-05 [2.0e-05, 4.6e-05] | 5/5 | 1.00 | 3 | 4.00 |
| A4 regrot | 3.8e-05 | 3.7e-05 [2.4e-05, 3.7e-03] | 5/5 | 1.00 | 3 | 4.00 |
| A5 regfix | 4.3e-05 | 4.9e-05 [3.5e-05, 6.0e-03] | 5/5 | 1.00 | 3 | 4.00 |
| A6 regrot + data | 2.6e-03 | 4.4e-03 [3.1e-05, 7.0e-03] | 5/5 | 0.68 | 3 | 4.00 |
| A7 clock + data | 2.7e-05 | 3.1e-05 [2.4e-05, 3.6e-05] | 5/5 | 1.00 | 3 | 4.00 |
