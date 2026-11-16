# Round M — streams against one monolithic looped block (EXPLORATORY)

Design, matching rule and both sets of predictions: `arch_m.py` (first design committed at a737c25, revision 2 at b56f337, both before any run). Design review: `review/review_round_m_design.md`. Learning rates: `arch_m_sweep.md`. Per-seed numbers: `arch_m_results_a.json`, `arch_m_results_b.json`.

Trained on programs of lengths 3–5, 3,000 steps, ten paired seeds, one teacher. Errors are median [min, max]. The three ratio columns are medians of per-seed ratios within the same network. "Address kept" is the share of passes at length 4 in which the register's attention peaks on position t.

| block | rate | parameters | multiply-adds per pass | training, length 4 | held out, length 4 | held out / training | length 8 / length 4 | inputs x2 / x1 | at step 1,000; 2,000 | address kept |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 streams, candidate | 0.01 | 705 | 176 | 6.0e-05 [2.9e-06, 1.7e-04] | 6.5e-05 [3.1e-06, 1.9e-04] | 1.02 | 1.64 | 2.1 | 8.3e-05; 6.8e-05 | 1.00 |
| M3 one block, additive, same parameters (29 units) | 0.01 | 706 | 580 | 5.5e-03 [2.0e-03, 9.2e-03] | 5.8e-03 [2.0e-03, 9.5e-03] | 1.07 | 1.05 | 7.5 | 1.0e-02; 7.6e-03 | 1.00 |
| M3a one block, additive, 18 units | 0.01 | 475 | 360 | 1.7e-02 [1.4e-02, 2.9e-02] | 1.8e-02 [1.4e-02, 3.0e-02] | 1.06 | 1.04 | 4.1 | 2.4e-02; 2.0e-02 | 1.00 |
| M3b one block, additive, twice the parameters (58 units) | 0.01 | 1315 | 1160 | 1.9e-03 [7.9e-04, 4.8e-03] | 1.8e-03 [9.3e-04, 4.6e-03] | 1.07 | 1.15 | 13.8 | 4.6e-03; 3.1e-03 | 1.00 |
| M16 one block, two layers, same parameters (16 + 16) | 0.01 | 705 | 576 | 4.8e-03 [2.5e-03, 9.0e-03] | 4.7e-03 [2.5e-03, 8.6e-03] | 0.99 | 1.34 | 5.0 | 1.1e-02; 6.4e-03 | 1.00 |
| M4 per-unit gate, same parameters (29 units) | 0.01 | 706 | 580 | 2.8e-05 [1.4e-05, 1.7e-04] | 2.8e-05 [1.4e-05, 1.7e-04] | 1.03 | 1.44 | 3.5 | 2.3e-04; 6.7e-05 | 1.00 |
| M4a per-unit gate, 18 units | 0.01 | 475 | 360 | 6.9e-04 [5.6e-06, 1.6e-03] | 8.8e-04 [5.6e-06, 1.5e-03] | 1.00 | 1.26 | 3.7 | 2.1e-03; 1.0e-03 | 1.00 |
| M7 one block, no attention, 48 units | 0.01 | 2233 | 2112 | 2.4e-02 [1.9e-02, 2.8e-02] | 3.1e-02 [2.7e-02, 3.6e-02] | 1.20 | 12.91 | 2.6 | 3.7e-02; 3.2e-02 | 1.00 |
| M12 branch: streams, candidate | 0.003 | 1169 | 608 | 8.4e-04 [6.7e-04, 1.2e-03] | 1.1e-03 [6.9e-04, 1.3e-03] | 1.22 | 0.90 | 13.0 | 2.6e-03; 1.2e-03 | 1.00 |
| M8 8 streams: candidate | 0.01 | 1329 | 208 | 6.8e-05 [2.5e-05, 1.3e-04] | 6.8e-05 [2.4e-05, 1.2e-04] | 0.98 | 1.43 | 1.8 | 5.5e-05; 1.8e-04 | 1.00 |
| M9 8 streams: one block, additive, same parameters (58 units) | 0.01 | 1331 | 1160 | 1.1e-02 [6.0e-03, 1.6e-02] | 1.1e-02 [6.5e-03, 1.6e-02] | 1.00 | 1.01 | 5.4 | 1.6e-02; 1.2e-02 | 1.00 |
| M17 8 streams: one block, two layers, same parameters (26 + 26) | 0.01 | 1361 | 1196 | 6.0e-03 [5.0e-03, 9.3e-03] | 6.1e-03 [5.1e-03, 1.1e-02] | 1.01 | 1.13 | 4.8 | 1.3e-02; 8.4e-03 | 1.00 |
| M10 8 streams: per-unit gate, same parameters (58 units) | 0.01 | 1331 | 1160 | 6.9e-05 [2.3e-05, 1.3e-04] | 7.0e-05 [2.4e-05, 1.2e-04] | 1.03 | 1.32 | 4.3 | 1.4e-03; 2.4e-04 | 1.00 |
| M13 branch: one block, two layers, same parameters (23 + 23) | 0.01 | 1140 | 989 | 8.1e-03 [6.0e-03, 1.6e-02] | 8.0e-03 [6.5e-03, 1.6e-02] | 1.01 | 1.15 | 4.6 | 1.8e-02; 1.2e-02 | 1.00 |
| M13b branch: one block, two layers, twice the parameters (36 + 36) | 0.01 | 2193 | 2016 | 6.7e-03 [4.3e-03, 8.1e-03] | 6.3e-03 [4.7e-03, 7.6e-03] | 1.01 | 1.07 | 3.9 | 1.1e-02; 7.9e-03 | 1.00 |
| M15 branch: one block, one layer, same parameters (50 units) | 0.01 | 1155 | 1000 | 2.0e-02 [1.6e-02, 5.6e-02] | 2.1e-02 [1.9e-02, 5.9e-02] | 1.05 | 2.16 | 3.6 | 2.7e-02; 2.4e-02 | 1.00 |

## Each block against its candidate, seed by seed

| block | error / candidate's, length 4 (median of per-seed ratios) [min, max] | seeds where the candidate is better | same at length 8 | same with inputs x2 |
|---|---|---|---|---|
| M3 one block, additive, same parameters (29 units) | 93.3 [10.9, 958.4] | 10/10 | 65.7 | 330.0 |
| M3a one block, additive, 18 units | 288.4 [85.5, 6353.4] | 10/10 | 207.9 | 667.3 |
| M3b one block, additive, twice the parameters (58 units) | 36.2 [7.8, 765.7] | 10/10 | 19.6 | 223.1 |
| M16 one block, two layers, same parameters (16 + 16) | 79.4 [29.4, 1305.1] | 10/10 | 74.9 | 215.4 |
| M4 per-unit gate, same parameters (29 units) | 0.8 [0.1, 4.6] | 3/10 | 0.7 | 1.0 |
| M4a per-unit gate, 18 units | 7.4 [0.1, 290.1] | 8/10 | 5.2 | 19.4 |
| M7 one block, no attention, 48 units | 527.0 [171.1, 9875.0] | 10/10 | 2596.5 | 694.4 |
| M9 8 streams: one block, additive, same parameters (58 units) | 174.1 [81.5, 474.7] | 10/10 | 136.3 | 485.6 |
| M17 8 streams: one block, two layers, same parameters (26 + 26) | 104.7 [47.8, 357.3] | 10/10 | 93.8 | 237.7 |
| M10 8 streams: per-unit gate, same parameters (58 units) | 1.1 [0.5, 2.4] | 5/10 | 1.0 | 2.5 |
| M13 branch: one block, two layers, same parameters (23 + 23) | 7.8 [5.8, 23.1] | 10/10 | 12.3 | 3.0 |
| M13b branch: one block, two layers, twice the parameters (36 + 36) | 5.8 [4.4, 9.2] | 10/10 | 7.0 | 2.0 |
| M15 branch: one block, one layer, same parameters (50 units) | 20.9 [14.7, 54.2] | 10/10 | 43.3 | 6.1 |

## Seeds solved

| block | solved (<1e-2) of 10: length 4, held out | length 8 | length 4, inputs x2 |
|---|---|---|---|
| M1 streams, candidate | 10 | 10 | 10 |
| M3 one block, additive, same parameters (29 units) | 10 | 8 | 0 |
| M3a one block, additive, 18 units | 0 | 0 | 0 |
| M3b one block, additive, twice the parameters (58 units) | 10 | 10 | 0 |
| M16 one block, two layers, same parameters (16 + 16) | 10 | 6 | 0 |
| M4 per-unit gate, same parameters (29 units) | 10 | 9 | 10 |
| M4a per-unit gate, 18 units | 10 | 9 | 10 |
| M7 one block, no attention, 48 units | 0 | 0 | 0 |
| M8 8 streams: candidate | 10 | 10 | 10 |
| M9 8 streams: one block, additive, same parameters (58 units) | 2 | 2 | 0 |
| M17 8 streams: one block, two layers, same parameters (26 + 26) | 9 | 7 | 0 |
| M10 8 streams: per-unit gate, same parameters (58 units) | 10 | 10 | 10 |
| M12 branch: streams, candidate | 10 | 10 | 0 |
| M13 branch: one block, two layers, same parameters (23 + 23) | 7 | 4 | 0 |
| M13b branch: one block, two layers, twice the parameters (36 + 36) | 10 | 9 | 0 |
| M15 branch: one block, one layer, same parameters (50 units) | 0 | 0 | 0 |

## Twice as long (three seeds, 6,000 steps; medians)

| block | held-out error, length 4, at step 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 |
|---|---|---|---|---|---|---|
| M1 streams, candidate | 8.8e-05 | 2.1e-04 | 1.2e-04 | 4.7e-06 | 6.3e-05 | 4.6e-05 |
| M3 one block, additive, same parameters (29 units) | 6.2e-03 | 3.2e-03 | 3.8e-03 | 2.3e-03 | 2.0e-03 | 1.9e-03 |
| M16 one block, two layers, same parameters (16 + 16) | 1.1e-02 | 5.8e-03 | 4.4e-03 | 2.8e-03 | 1.8e-03 | 2.6e-03 |
| M4 per-unit gate, same parameters (29 units) | 2.3e-04 | 3.3e-05 | 5.5e-05 | 1.8e-05 | 1.6e-04 | 2.3e-05 |

## Five other teachers (one seed each; held-out error at length 4)

| teacher | M1 streams, candidate | M3 one block, additive | M4 per-unit gate |
|---|---|---|---|
| 11 | 1.0e-04 | 9.3e-03 | 9.6e-05 |
| 12 | 1.5e-05 | 4.2e-03 | 8.2e-06 |
| 13 | 2.7e-05 | 7.7e-03 | 4.2e-05 |
| 14 | 2.8e-05 | 6.8e-03 | 1.1e-05 |
| 15 | 1.4e-04 | 8.0e-03 | 3.9e-05 |

## Where the error jumps at eight passes

In one seed each of M1 and M4 and three seeds of M16 the error at length 8 is more than ten times the error at length 4 (the largest per-seed ratio is above ten in ten of the sixteen rows). Probing the worst seed of each (rerun, same seed): the register's attention weight on position t at passes 6, 7, 8 is 0.99, 0.86, 0.78 (M1, seed 6), 0.9, 0.4, 0.04 (M4, seed 9) and 0.1, 0.0, 0.0 (M16, seed 1; already 0.21 at pass 6 of length 6). The trained register has drifted from the exact shift; the loss is in the shared controller, in every kind of block.
