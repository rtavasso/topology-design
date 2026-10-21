# Stage-0 short-path share of total-effect gen at end of training (EXPLORATORY)

Computed on the Mac from the full Stage-0 run files (21 Oct 2026). Grokked runs only; each run averages its last 5 evals.
Value: (sum over depth-1 blocks of gen_tshap) / (sum over all blocks). Cell: median over seeds (n).

| graph | lr 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 |
|---|---|---|---|---|---|
| pk0100 | 0.65 (3) | 0.64 (3) | 0.50 (3) | 0.50 (3) | 0.76 (3) |
| pk0110 | 0.58 (2) | 0.55 (3) | 0.50 (3) | 0.50 (3) | 0.64 (3) |
| pk0111 | 0.52 (3) | 0.51 (3) | 0.50 (3) | 0.50 (3) | 0.51 (3) |
| pk0120 | 0.43 (1) | 0.49 (3) | 0.52 (3) | 0.68 (3) | 0.87 (3) |
| pk0121 | 0.41 (1) | 0.37 (3) | 0.38 (3) | 0.41 (3) | 0.39 (3) |
| pk0122 | 0.35 (3) | 0.34 (3) | 0.35 (3) | 0.35 (2) | 0.29 (3) |
| pk0123 | — | 0.28 (3) | 0.28 (2) | 0.13 (1) | 0.18 (3) |

Utility of the H2 block sets by lr (per seed, mean of last 5 evals):

| graph (set) | 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 |
|---|---|---|---|---|---|
| pk0120 {2,3} | 2.97*, 0.87, 0.90 | 0.73, 0.78, 0.92 | 0.73, 0.75, 0.56 | 0.42, 0.50, 0.48 | 0.16, 0.26, 0.20 |
| pk0121 {2,3} | 0.75, 0.57, 0.69 | 0.86, 0.83, 0.78 | 0.82, 0.63, 0.71 | 0.58, 0.51, 0.45 | 0.53, 0.29, 0.28 |
| pk0100 {2} | 0.73, 0.60, 0.70 | 0.84, 0.72, 0.60 | 1.00, 0.99, 0.99 | 0.99, 0.94, 0.99 | 0.61, 0.48, 0.10 |
| pk0110 {2} | 0.44, 0.41, 0.34 | 0.45, 0.56, 0.56 | 0.22, 0.50, 0.51 | 0.50, 0.54, 0.52 | 0.34, 0.32, 0.46 |

\* this run never grokked (gen_total near 0, so the ratio is unstable).
