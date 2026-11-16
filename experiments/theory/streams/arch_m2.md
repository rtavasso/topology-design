# Round M2 — follow-up to round M after the review of its results (EXPLORATORY)

Design and predictions S1–S8 and the extension X1–X6: `arch_m2.py`. Review answered: `review/review_round_m_results.md`. Cosine decay of the learning rate to zero for every arm; each row at its own rate. Ten paired seeds, one teacher; medians [min, max]; ratio columns are medians of per-seed ratios.

## As designed: rates 0.003 and 0.01 (every row chose 0.01)

Rate selection (seeds 100–102):

| row | rate | training error, length 4 (median of 3 seeds) | held-out, length 4 |
|---|---|---|---|
| N1 | 0.003 | 2.6e-04 | 2.5e-04 |
| N1 | 0.01 | 1.5e-05 | 1.4e-05 |
| N4 | 0.003 | 1.3e-03 | 1.5e-03 |
| N4 | 0.01 | 5.9e-05 | 7.3e-05 |
| N3 | 0.003 | 1.2e-02 | 1.5e-02 |
| N3 | 0.01 | 5.2e-03 | 6.4e-03 |
| N1s | 0.003 | 3.8e-03 | 4.5e-03 |
| N1s | 0.01 | 4.2e-05 | 4.8e-05 |
| N4a | 0.003 | 4.5e-03 | 5.2e-03 |
| N4a | 0.01 | 9.4e-04 | 1.3e-03 |
| N12 | 0.003 | 1.2e-03 | 1.3e-03 |
| N12 | 0.01 | 4.8e-04 | 5.3e-04 |
| N8 | 0.003 | 1.3e-04 | 1.3e-04 |
| N8 | 0.01 | 1.1e-05 | 1.2e-05 |
| N10 | 0.003 | 4.2e-03 | 4.3e-03 |
| N10 | 0.01 | 3.6e-04 | 3.9e-04 |
| N14 | 0.003 | 3.4e-03 | 3.7e-03 |
| N14 | 0.01 | 1.2e-03 | 1.2e-03 |
| N13 | 0.003 | 2.8e-02 | 2.9e-02 |
| N13 | 0.01 | 1.1e-02 | 1.2e-02 |

| block | rate | parameters | multiply-adds per pass | training, length 4 | held out, length 4 | held out / training | length 8 / length 4 | inputs x2 / x1 | at step 1,000; 2,000 | address kept |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 streams, candidate (4 streams of 12) | 0.01 | 705 | 176 | 1.8e-05 [7.2e-06, 2.4e-05] | 1.8e-05 [7.4e-06, 2.6e-05] | 1.04 | 1.15 | 10.9 | 1.0e-04; 2.1e-05 | 1.00 |
| N4 per-unit gate, same parameters (29 units) | 0.01 | 706 | 580 | 5.8e-05 [3.8e-05, 8.9e-05] | 6.2e-05 [3.7e-05, 9.6e-05] | 1.06 | 1.14 | 6.4 | 2.9e-04; 7.8e-05 | 1.00 |
| N3 one block, additive, same parameters (29 units) | 0.01 | 706 | 580 | 5.9e-03 [1.4e-03, 8.0e-03] | 6.7e-03 [1.7e-03, 8.5e-03] | 1.11 | 0.99 | 7.0 | 1.1e-02; 7.3e-03 | 1.00 |
| N1s streams, candidate, 3 streams of 6 (18 units) | 0.01 | 337 | 96 | 4.5e-05 [1.6e-05, 2.4e-02] | 4.3e-05 [1.6e-05, 1.8e-02] | 1.02 | 1.23 | 4.9 | 1.0e-03; 6.6e-05 | 1.00 |
| N4a per-unit gate, 18 units | 0.01 | 475 | 360 | 9.9e-04 [2.4e-04, 2.0e-03] | 1.2e-03 [2.3e-04, 1.8e-03] | 0.96 | 1.16 | 3.9 | 2.5e-03; 1.3e-03 | 1.00 |
| N12 branch: streams, candidate | 0.01 | 1169 | 608 | 5.8e-04 [3.9e-04, 9.1e-04] | 6.3e-04 [4.9e-04, 8.4e-04] | 1.07 | 0.90 | 20.1 | 1.6e-03; 8.2e-04 | 1.00 |
| N8 8 streams: candidate | 0.01 | 1329 | 208 | 1.3e-05 [7.2e-06, 2.4e-05] | 1.4e-05 [7.7e-06, 2.2e-05] | 1.01 | 1.20 | 11.0 | 6.7e-05; 1.8e-05 | 1.00 |
| N10 8 streams: per-unit gate, same parameters (58 units) | 0.01 | 1331 | 1160 | 3.0e-04 [1.4e-04, 4.4e-04] | 3.1e-04 [1.5e-04, 4.7e-04] | 1.02 | 1.09 | 6.8 | 1.7e-03; 4.3e-04 | 1.00 |
| N14 branch: per-unit gate that reads the data, same parameters (27 units) | 0.01 | 1176 | 1032 | 1.4e-03 [1.0e-03, 1.7e-03] | 1.3e-03 [1.1e-03, 1.9e-03] | 1.03 | 1.00 | 11.8 | 2.8e-03; 1.8e-03 | 1.00 |
| N13 branch: one block, two layers, same parameters (23 + 23) | 0.01 | 1140 | 989 | 1.0e-02 [7.0e-03, 1.7e-02] | 1.0e-02 [8.1e-03, 1.7e-02] | 1.06 | 1.00 | 4.3 | 1.8e-02; 1.1e-02 | 1.00 |

| block | error / its candidate's, length 4 (median of per-seed ratios) [min, max] | seeds where the candidate is better | same at step 1,000 | same at length 8 | same with inputs x2 | seeds with length 8 / length 4 above 3 |
|---|---|---|---|---|---|---|
| N1 streams, candidate (4 streams of 12) | (candidate) | | | | | 1/10 |
| N4 per-unit gate, same parameters (29 units) | 3.98 [2.21, 10.0] | 10/10 | 3.3, candidate ahead 10/10 | 3.04 | 2.08 | 1/10 |
| N3 one block, additive, same parameters (29 units) | 397.93 [72.50, 795.8] | 10/10 | 107.3, candidate ahead 10/10 | 342.88 | 270.96 | 0/10 |
| N1s streams, candidate, 3 streams of 6 (18 units) | (candidate) | | | | | 1/10 |
| N4a per-unit gate, 18 units | 16.10 [0.04, 75.5] | 9/10 | 2.3, candidate ahead 9/10 | 16.63 | 10.30 | 1/10 |
| N8 8 streams: candidate | (candidate) | | | | | 0/10 |
| N10 8 streams: per-unit gate, same parameters (58 units) | 19.51 [7.22, 58.1] | 10/10 | 24.5, candidate ahead 9/10 | 18.92 | 12.49 | 1/10 |
| N12 branch: streams, candidate | (candidate) | | | | | 0/10 |
| N14 branch: per-unit gate that reads the data, same parameters (27 units) | 2.10 [1.29, 3.2] | 10/10 | 1.6, candidate ahead 10/10 | 2.32 | 1.42 | 0/10 |
| N13 branch: one block, two layers, same parameters (23 + 23) | 18.43 [9.92, 31.9] | 10/10 | 13.2, candidate ahead 10/10 | 19.33 | 3.48 | 2/10 |

Check at a higher rate (0.03, seeds 100–102), which prompted the extension:

| row | rate | training error, length 4 | held-out, length 4 |
|---|---|---|---|
| N4 | 0.03 | 6.5e-06 | 7.5e-06 |
| N4a | 0.03 | 4.5e-06 | 4.9e-06 |
| N1 | 0.03 | 1.6e-06 | 1.9e-06 |
| N14 | 0.03 | 6.9e-04 | 8.6e-04 |
| N10 | 0.03 | 1.6e-05 | 1.7e-05 |
| N8 | 0.03 | 3.2e-06 | 3.0e-06 |

## Extension: each row at its best of 0.01, 0.03, 0.1 (decaying)

Rate selection (seeds 100–102; 0.01 from the table above; 0.3 was tried for the rows that chose 0.1 and was far worse in all six):

| row | rate | training error, length 4 (median of 3 seeds) | held-out, length 4 |
|---|---|---|---|
| N1 | 0.03 | 1.6e-06 | 1.9e-06 |
| N1 | 0.1 | 5.3e-06 | 4.7e-06 |
| N4 | 0.03 | 6.5e-06 | 7.5e-06 |
| N4 | 0.1 | 1.5e-06 | 1.8e-06 |
| N3 | 0.03 | 1.6e-03 | 2.0e-03 |
| N3 | 0.1 | 2.3e-04 | 3.0e-04 |
| N1s | 0.03 | 6.8e-05 | 8.9e-05 |
| N1s | 0.1 | 2.1e-07 | 2.0e-07 |
| N4a | 0.03 | 4.5e-06 | 4.9e-06 |
| N4a | 0.1 | 1.8e-06 | 1.6e-06 |
| N12 | 0.03 | 2.6e-04 | 2.7e-04 |
| N12 | 0.1 | 2.6e-04 | 2.2e-04 |
| N8 | 0.03 | 3.2e-06 | 3.0e-06 |
| N8 | 0.1 | 4.7e-06 | 4.6e-06 |
| N10 | 0.03 | 1.6e-05 | 1.7e-05 |
| N10 | 0.1 | 3.2e-06 | 3.1e-06 |
| N14 | 0.03 | 6.9e-04 | 8.6e-04 |
| N14 | 0.1 | 7.2e-04 | 6.4e-04 |
| N13 | 0.03 | 5.0e-03 | 4.7e-03 |
| N13 | 0.1 | 4.7e-03 | 4.6e-03 |
| N4 | 0.3 | 9.7e-04 | 1.4e-03 |
| N3 | 0.3 | 2.4e-01 | 2.3e-01 |
| N1s | 0.3 | 3.6e-02 | 3.3e-02 |
| N4a | 0.3 | 2.4e-03 | 4.0e-03 |
| N10 | 0.3 | 1.0e+00 | 1.0e+00 |
| N13 | 0.3 | 3.7e-01 | 3.6e-01 |

| block | rate | parameters | multiply-adds per pass | training, length 4 | held out, length 4 | held out / training | length 8 / length 4 | inputs x2 / x1 | at step 1,000; 2,000 | address kept |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 streams, candidate (4 streams of 12) | 0.03 | 705 | 176 | 3.4e-06 [1.6e-06, 6.8e-06] | 3.4e-06 [1.6e-06, 6.4e-06] | 1.02 | 22721.26 | 11.4 | 4.2e-05; 4.2e-06 | 1.00 |
| N4 per-unit gate, same parameters (29 units) | 0.1 | 706 | 580 | 2.1e-06 [9.2e-07, 6.3e-06] | 2.3e-06 [1.0e-06, 5.9e-06] | 1.06 | 59245.95 | 4.3 | 4.6e-04; 3.0e-05 | 1.00 |
| N3 one block, additive, same parameters (29 units) | 0.1 | 706 | 580 | 3.0e-03 [6.7e-04, 1.1e-01] | 3.0e-03 [8.0e-04, 1.4e-01] | 1.05 | 62.71 | 11.4 | 1.7e-02; 5.0e-03 | 1.00 |
| N1s streams, candidate, 3 streams of 6 (18 units) | 0.1 | 337 | 96 | 1.4e-06 [7.2e-08, 2.4e-02] | 1.4e-06 [6.2e-08, 2.2e-02] | 0.91 | 86871.52 | 4.1 | 7.9e-04; 6.1e-05 | 1.00 |
| N4a per-unit gate, 18 units | 0.1 | 475 | 360 | 4.4e-04 [7.7e-07, 2.4e-02] | 4.4e-04 [7.9e-07, 2.4e-02] | 0.99 | 951.64 | 2.9 | 1.2e-03; 8.3e-04 | 1.00 |
| N12 branch: streams, candidate | 0.03 | 1169 | 608 | 3.1e-04 [1.8e-04, 6.1e-04] | 3.2e-04 [2.8e-04, 6.1e-04] | 1.16 | 3.73 | 23.8 | 1.8e-03; 7.8e-04 | 1.00 |
| N8 8 streams: candidate | 0.03 | 1329 | 208 | 6.4e-06 [2.3e-06, 1.7e-05] | 7.1e-06 [2.1e-06, 1.8e-05] | 1.00 | 5669.20 | 12.0 | 1.6e-04; 1.0e-05 | 1.00 |
| N10 8 streams: per-unit gate, same parameters (58 units) | 0.1 | 1331 | 1160 | 4.6e-06 [2.1e-06, 6.9e-06] | 4.9e-06 [2.2e-06, 6.7e-06] | 1.02 | 47132.28 | 5.7 | 3.6e-04; 9.3e-06 | 1.00 |
| N14 branch: per-unit gate that reads the data, same parameters (27 units) | 0.03 | 1176 | 1032 | 7.0e-04 [5.4e-04, 9.3e-04] | 7.4e-04 [6.4e-04, 8.3e-04] | 1.03 | 79.72 | 13.0 | 2.4e-03; 1.3e-03 | 1.00 |
| N13 branch: one block, two layers, same parameters (23 + 23) | 0.1 | 1140 | 989 | 5.3e-03 [3.4e-03, 2.1e-02] | 5.3e-03 [3.5e-03, 1.9e-02] | 1.04 | 23.36 | 5.3 | 2.7e-02; 9.1e-03 | 1.00 |

| block | error / its candidate's, length 4 (median of per-seed ratios) [min, max] | seeds where the candidate is better | same at step 1,000 | same at length 8 | same with inputs x2 | seeds with length 8 / length 4 above 3 |
|---|---|---|---|---|---|---|
| N1 streams, candidate (4 streams of 12) | (candidate) | | | | | 9/10 |
| N4 per-unit gate, same parameters (29 units) | 0.63 [0.23, 2.6] | 3/10 | 7.9, candidate ahead 10/10 | 2.76 | 0.23 | 10/10 |
| N3 one block, additive, same parameters (29 units) | 726.66 [193.95, 23895.8] | 10/10 | 649.8, candidate ahead 10/10 | 3.04 | 946.43 | 8/10 |
| N1s streams, candidate, 3 streams of 6 (18 units) | (candidate) | | | | | 10/10 |
| N4a per-unit gate, 18 units | 12.66 [0.01, 6746.4] | 6/10 | 1.1, candidate ahead 6/10 | 0.94 | 8.87 | 10/10 |
| N8 8 streams: candidate | (candidate) | | | | | 8/10 |
| N10 8 streams: per-unit gate, same parameters (58 units) | 0.62 [0.23, 3.0] | 3/10 | 0.9, candidate ahead 5/10 | 346.39 | 0.30 | 10/10 |
| N12 branch: streams, candidate | (candidate) | | | | | 5/10 |
| N14 branch: per-unit gate that reads the data, same parameters (27 units) | 2.12 [1.35, 2.9] | 10/10 | 1.3, candidate ahead 6/10 | 3.87 | 1.36 | 7/10 |
| N13 branch: one block, two layers, same parameters (23 + 23) | 14.60 [5.65, 58.4] | 10/10 | 13.6, candidate ahead 10/10 | 87.02 | 3.63 | 10/10 |

At these rates the trained register loses the address beyond the trained lengths in most seeds of every block (last column), which is what the very large length-8 ratios are.

Probe (seed 0, rerun): attention weight on position t at passes 6, 7, 8 of length-8 programs. As designed (rate 0.01): candidate 1.0, 0.99, 0.98; per-unit gate 1.0, 1.0, 1.0. Extension: candidate (0.03) 0.69, 0.01, 0.02; per-unit gate (0.1) 0.74, 0.02, 0.0.
