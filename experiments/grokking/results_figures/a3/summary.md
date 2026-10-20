# Grokking × topology — results summary

80 runs. Endpoints as preregistered; delay = t_grok − t_fit (censored runs counted at budget).

| arm | wd | lr | n | median t_fit | median delay | # censored | localization index [95% CI] | peak memorization (excl. train acc) | final test acc | test acc, block 3 ablated | final rule share in block 3 | median max branch dominance (⚠ >0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| par | 1 | 0.01 | 10 | 250 | 500 | 0 | -0.024 [-0.049, +0.003] | 0.99 | 1.000 | 0.904 | 0.31 | 0.42 |
| seq | 1 | 0.003 | 10 | 250 | 6500 | 2 | +0.033 [+0.025, +0.041] | 0.92 | 0.891 | 0.072 | 1.00 | 1.00 ⚠ |
| sink_0 | 1 | 0.01 | 10 | 250 | 1250 | 0 | +0.000 [+0.000, +0.000] | 0.95 | 0.996 | 0.996 | 0.00 | 1.00 ⚠ |
| sink_0.25 | 1 | 0.01 | 10 | 250 | 1000 | 0 | -0.004 [-0.011, -0.000] | 0.94 | 0.915 | 0.848 | 0.07 | 1.00 ⚠ |
| sink_0.5 | 1 | 0.01 | 10 | 250 | 1000 | 0 | -0.020 [-0.046, -0.000] | 0.93 | 0.998 | 0.619 | 0.28 | 1.00 ⚠ |
| sink_1 | 1 | 0.01 | 10 | 250 | 875 | 0 | -0.000 [-0.000, -0.000] | 0.97 | 0.976 | 0.415 | 0.49 | 1.00 ⚠ |
| sink_2 | 1 | 0.01 | 10 | 250 | 750 | 0 | +0.004 [-0.036, +0.058] | 0.94 | 0.999 | 0.591 | 0.26 | 1.00 ⚠ |
| sink_4 | 1 | 0.01 | 10 | 250 | 500 | 0 | -0.059 [-0.092, -0.026] | 0.79 | 0.999 | 0.669 | 0.18 | 1.00 ⚠ |

## Paired contrasts on grokking delay

P(row arm groks faster than column arm), paired over shared seeds [95% bootstrap CI].

- wd=1, lr=0.01: P(par faster than sink_0) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.01: P(par faster than sink_0.25) = 0.95 [0.85, 1.00] (n=10)
- wd=1, lr=0.01: P(par faster than sink_0.5) = 0.95 [0.85, 1.00] (n=10)
- wd=1, lr=0.01: P(par faster than sink_1) = 0.95 [0.85, 1.00] (n=10)
- wd=1, lr=0.01: P(par faster than sink_2) = 0.90 [0.75, 1.00] (n=10)
- wd=1, lr=0.01: P(par faster than sink_4) = 0.55 [0.30, 0.80] (n=10)
- wd=1, lr=0.01: P(sink_0 faster than sink_0.25) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.01: P(sink_0 faster than sink_0.5) = 0.10 [0.00, 0.30] (n=10)
- wd=1, lr=0.01: P(sink_0 faster than sink_1) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.01: P(sink_0 faster than sink_2) = 0.10 [0.00, 0.25] (n=10)
- wd=1, lr=0.01: P(sink_0 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.01: P(sink_0.25 faster than sink_0.5) = 0.55 [0.25, 0.85] (n=10)
- wd=1, lr=0.01: P(sink_0.25 faster than sink_1) = 0.25 [0.05, 0.45] (n=10)
- wd=1, lr=0.01: P(sink_0.25 faster than sink_2) = 0.25 [0.05, 0.45] (n=10)
- wd=1, lr=0.01: P(sink_0.25 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.01: P(sink_0.5 faster than sink_1) = 0.40 [0.20, 0.65] (n=10)
- wd=1, lr=0.01: P(sink_0.5 faster than sink_2) = 0.35 [0.10, 0.60] (n=10)
- wd=1, lr=0.01: P(sink_0.5 faster than sink_4) = 0.10 [0.00, 0.25] (n=10)
- wd=1, lr=0.01: P(sink_1 faster than sink_2) = 0.40 [0.15, 0.65] (n=10)
- wd=1, lr=0.01: P(sink_1 faster than sink_4) = 0.10 [0.00, 0.25] (n=10)
- wd=1, lr=0.01: P(sink_2 faster than sink_4) = 0.20 [0.05, 0.35] (n=10)
