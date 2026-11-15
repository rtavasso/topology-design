# Round H — checks called for by the independent review (EXPLORATORY)

Candidate network (register started as the exact shift, sinusoidal keys, softmax, raw data read) unless stated; H8 uses the register that starts still. Ten seeds, median [min, max]; H9 five teachers, one seed each. Predictions in `arch_h.py`, written after reading `review/review_moe_in_loops.md`.

| variant | held-out error | solved (<1e-2) | streams used | streams per pass in training |
|---|---|---|---|---|
| H1 program, 4 streams, gate reads data, top-1 from the first step | 8.8e-05 [1.8e-05, 1.3e-01] | 8/10 | 3 | 1.00 |
| H2 program, 4 streams, token-only gate, top-1 from the first step | 2.1e-01 [2.0e-05, 3.9e-01] | 2/10 | 2 | 1.00 |
| H5 branch, top-1 from the first step | 1.1e-03 [7.3e-04, 1.7e-03] | 10/10 | 3 | 1.00 |

| H8: dense steps | address learned at switch | one-to-one at switch | mean top probability at switch | ends solved | solved agrees with (address and one-to-one) |
|---|---|---|---|---|---|
| 90 | 0/10 | 0/10 | 0.42 [0.35, 0.50] | 1/10 | 9/10 |
| 150 | 2/10 | 0/10 | 0.52 [0.48, 0.56] | 8/10 | 2/10 |
| 200 | 9/10 | 3/10 | 0.64 [0.55, 0.68] | 9/10 | 4/10 |
| 300 | 10/10 | 7/10 | 0.77 [0.68, 0.80] | 10/10 | 7/10 |

H8 agreement over all runs: 22/40

| H9: other teachers | held-out error | solved (<1e-2) | streams used | streams per pass in training |
|---|---|---|---|---|
| H9 program, teachers 11-15 | 1.9e-05 [1.2e-05, 3.0e-05] | 5/5 | 3 | 1.21 |
| H9 branch, teachers 11-15 | 1.2e-03 [7.5e-04, 1.2e-03] | 5/5 | 3 | 1.14 |

| variant | held-out error | solved (<1e-2) | streams used | streams per pass in training |
|---|---|---|---|---|
| H3 program, 8 streams, gate reads data, top-1 from the first step | 5.6e-02 [1.6e-05, 1.0e-01] | 3/10 | 4 | 1.00 |
| H6 program, 8 streams, gate reads data, dense until confident then top-1 | 1.8e-05 [1.2e-05, 2.6e-05] | 10/10 | 5 | 1.58 |
| H7a program, 8 streams, token-only, dense 10 % then top-1, sigmoid | 8.5e-05 [5.4e-05, 1.0e-01] | 9/10 | 5 | 1.70 |
| H7b program, 8 streams, token-only, dense 10 % then top-1, softmax | 1.8e-05 [1.3e-05, 8.2e-02] | 8/10 | 5 | 1.70 |
