# Round E — the candidate at ten seeds (EXPLORATORY)

Candidate: register started as the exact one-position shift, sinusoidal keys; softmax over all streams, not renormalised; every stream runs until the gate's mean top probability passes 0.8, then only the top stream; on the branch task the gate also reads the data state. Reference: the same network with every stream always run. Ten seeds, median [min, max]. For the length rows the first error column is length 4. Predictions in `arch_e.py`.

| variant | held-out error | solved (<1e-2) | length 6 | length 8 | solved at 6, at 8 | streams per pass: training, afterwards | dense steps of 3,000 |
|---|---|---|---|---|---|---|---|
| G1 program, 4 streams: candidate | 3.0e-05 [1.1e-05, 3.8e-05] | 10/10 | - | - | - | 1.20, 1.00 | 198 [154, 866] |
| G2 program, 4 streams: every stream run | 3.2e-05 [2.1e-05, 4.8e-05] | 10/10 | - | - | - | 4.00, 4.00 | - |
| G5 branch: candidate (gate reads raw data) | 1.1e-03 [6.8e-04, 1.7e-03] | 10/10 | - | - | - | 1.12, 1.00 | 123 [90, 223] |
| G6 branch: candidate, gate reads normalised data | 9.5e-04 [6.7e-04, 1.5e-03] | 10/10 | - | - | - | 1.12, 1.00 | 119 [87, 217] |
| G7 branch: every stream run | 1.4e-03 [1.2e-03, 1.7e-03] | 10/10 | - | - | - | 4.00, 4.00 | - |
| G3 program, 8 streams: candidate | 1.8e-05 [1.4e-05, 3.3e-05] | 10/10 | - | - | - | 2.32, 1.00 | 566 [425, 784] |
| G4 program, 8 streams: every stream run | 2.6e-05 [2.3e-05, 3.8e-05] | 10/10 | - | - | - | 8.00, 8.00 | - |
| G8 program, lengths 3-5 -> 6, 8: candidate | 2.7e-05 [1.5e-05, 4.9e-05] | 10/10 | 2.7e-05 [1.6e-05, 5.3e-05] | 3.1e-05 [1.8e-05, 6.1e-05] | 10/10, 10/10 | 1.21, 1.00 | 212 [134, 565] |
| G9 branch, lengths 3-5 -> 6, 8: candidate | 9.4e-04 [7.1e-04, 1.4e-03] | 10/10 | 9.5e-04 [6.9e-04, 1.2e-03] | 9.6e-04 [6.6e-04, 1.1e-03] | 10/10, 10/10 | 1.12, 1.00 | 125 [92, 138] |
