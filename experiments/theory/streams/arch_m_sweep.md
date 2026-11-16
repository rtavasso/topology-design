# Round M — learning-rate selection (EXPLORATORY)

Three seeds that are not among the ten (100–102). Each family runs the main comparison at the rate with the lowest median training error at length 4. A fourth rate (0.03) was added for the arms whose best rate sat at the edge of the grid; it was worse or equal in every one.

| family (arm) | rate | training error, length 4 (median of 3 seeds) | held-out, length 4 |
|---|---|---|---|
| cand (M1) | 0.001 | 4.3e-04 | 4.8e-04 |
| cand (M1) | 0.003 | 3.5e-05 | 3.5e-05 |
| cand (M1) | 0.01 | 6.9e-06 | 6.5e-06 |
| add (M3) | 0.001 | 1.5e-02 | 1.8e-02 |
| add (M3) | 0.003 | 6.9e-03 | 1.0e-02 |
| add (M3) | 0.01 | 4.5e-03 | 5.3e-03 |
| two (M16) | 0.001 | 1.2e-02 | 1.4e-02 |
| two (M16) | 0.003 | 5.0e-03 | 5.8e-03 |
| two (M16) | 0.01 | 4.2e-03 | 4.5e-03 |
| mul (M4) | 0.001 | 3.0e-03 | 3.0e-03 |
| mul (M4) | 0.003 | 1.4e-04 | 1.7e-04 |
| mul (M4) | 0.01 | 2.5e-05 | 2.2e-05 |
| cand_br (M12) | 0.001 | 2.0e-03 | 1.9e-03 |
| cand_br (M12) | 0.003 | 9.8e-04 | 8.2e-04 |
| cand_br (M12) | 0.01 | 1.6e-03 | 1.0e-03 |
| two_br (M13) | 0.001 | 2.7e-02 | 2.8e-02 |
| two_br (M13) | 0.003 | 1.9e-02 | 1.9e-02 |
| two_br (M13) | 0.01 | 9.9e-03 | 1.0e-02 |
| add_br (M15) | 0.001 | 3.1e-02 | 3.2e-02 |
| add_br (M15) | 0.003 | 2.2e-02 | 2.3e-02 |
| add_br (M15) | 0.01 | 2.1e-02 | 2.2e-02 |
| cand (M1) | 0.03 | 1.4e-04 | 1.4e-04 |
| add (M3) | 0.03 | 6.8e-03 | 9.6e-03 |
| two (M16) | 0.03 | 6.0e-03 | 5.6e-03 |
| mul (M4) | 0.03 | 2.7e-04 | 2.6e-04 |
| two_br (M13) | 0.03 | 1.0e-02 | 1.0e-02 |
| add_br (M15) | 0.03 | 2.5e-02 | 2.6e-02 |

Chosen: candidate 0.01 (program), 0.003 (branch); every monolith and the per-unit gate 0.01.
