# Grokking × topology — results summary

190 runs. Endpoints as preregistered; delay = t_grok − t_fit (censored runs counted at budget).

| arm | wd | lr | n | median t_fit | median delay | # censored | localization index [95% CI] | peak memorization (excl. train acc) | final test acc | test acc, block 3 ablated | final rule share in block 3 | median max branch dominance (⚠ >0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| par | 0 | 0.003 | 10 | 250 | 19751 | 10 | -0.008 [-0.035, +0.021] | 0.98 | 0.000 | 0.001 | 0.33 | 0.37 |
| seq | 0 | 0.003 | 10 | 250 | 19751 | 10 | +0.027 [+0.016, +0.041] | 0.92 | 0.003 | 0.004 | 0.99 | 0.99 ⚠ |
| sink_1 | 0 | 0.003 | 10 | 250 | 19751 | 10 | -0.084 [-0.100, -0.070] | 0.93 | 0.001 | 0.002 | 0.08 | 0.98 ⚠ |
| par | 1 | 0.001 | 10 | 250 | 6625 | 0 | -0.022 [-0.048, +0.007] | 1.00 | 0.999 | 0.868 | 0.30 | 0.43 |
| par | 1 | 0.003 | 10 | 250 | 1875 | 0 | -0.023 [-0.051, +0.007] | 1.00 | 1.000 | 0.948 | 0.32 | 0.44 |
| seq | 1 | 0.001 | 10 | 250 | 19751 | 10 | +0.010 [+0.007, +0.013] | 0.92 | 0.213 | 0.076 | 0.99 | 1.00 ⚠ |
| seq | 1 | 0.003 | 10 | 250 | 6500 | 2 | +0.033 [+0.025, +0.041] | 0.92 | 0.891 | 0.072 | 1.00 | 1.00 ⚠ |
| sink_0 | 1 | 0.001 | 10 | 250 | 19751 | 9 | +0.000 [+0.000, +0.000] | 0.99 | 0.357 | 0.357 | 0.00 | 1.00 ⚠ |
| sink_0 | 1 | 0.003 | 10 | 250 | 6500 | 0 | +0.000 [+0.000, +0.000] | 0.99 | 0.978 | 0.978 | 0.00 | 1.00 ⚠ |
| sink_0.25 | 1 | 0.001 | 10 | 250 | 19751 | 9 | -0.000 [-0.000, -0.000] | 0.99 | 0.297 | 0.277 | 0.01 | 1.00 ⚠ |
| sink_0.25 | 1 | 0.003 | 10 | 250 | 5875 | 0 | -0.000 [-0.000, -0.000] | 0.99 | 0.978 | 0.972 | 0.00 | 1.00 ⚠ |
| sink_0.5 | 1 | 0.001 | 10 | 250 | 19751 | 8 | -0.001 [-0.004, -0.000] | 0.99 | 0.416 | 0.351 | 0.03 | 1.00 ⚠ |
| sink_0.5 | 1 | 0.003 | 10 | 250 | 5875 | 0 | -0.009 [-0.026, -0.000] | 0.99 | 0.985 | 0.869 | 0.02 | 1.00 ⚠ |
| sink_1 | 1 | 0.001 | 10 | 250 | 19751 | 6 | -0.002 [-0.006, -0.000] | 1.00 | 0.548 | 0.388 | 0.13 | 1.00 ⚠ |
| sink_1 | 1 | 0.003 | 10 | 250 | 4375 | 0 | -0.006 [-0.013, -0.000] | 0.99 | 0.985 | 0.617 | 0.11 | 1.00 ⚠ |
| sink_2 | 1 | 0.001 | 10 | 250 | 17750 | 1 | -0.001 [-0.002, -0.001] | 1.00 | 0.890 | 0.431 | 0.28 | 1.00 ⚠ |
| sink_2 | 1 | 0.003 | 10 | 250 | 3500 | 0 | -0.007 [-0.021, -0.000] | 1.00 | 0.990 | 0.686 | 0.10 | 1.00 ⚠ |
| sink_4 | 1 | 0.001 | 10 | 250 | 12625 | 0 | -0.014 [-0.025, -0.005] | 0.99 | 0.998 | 0.210 | 0.52 | 1.00 ⚠ |
| sink_4 | 1 | 0.003 | 10 | 250 | 3375 | 0 | -0.028 [-0.068, -0.001] | 1.00 | 0.999 | 0.346 | 0.57 | 1.00 ⚠ |

## Paired contrasts on grokking delay

P(row arm groks faster than column arm), paired over shared seeds [95% bootstrap CI].

- wd=0, lr=0.003: P(par faster than seq) = 0.50 [0.50, 0.50] (n=10)
- wd=0, lr=0.003: P(par faster than sink_1) = 0.50 [0.50, 0.50] (n=10)
- wd=0, lr=0.003: P(seq faster than sink_1) = 0.50 [0.50, 0.50] (n=10)
- wd=1, lr=0.001: P(par faster than seq) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_0) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_0.25) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_0.5) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_1) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_2) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(par faster than sink_4) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_0) = 0.45 [0.35, 0.50] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_0.25) = 0.45 [0.35, 0.50] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_0.5) = 0.40 [0.25, 0.50] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_1) = 0.30 [0.15, 0.45] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_2) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.001: P(seq faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.001: P(sink_0 faster than sink_0.25) = 0.55 [0.50, 0.65] (n=10)
- wd=1, lr=0.001: P(sink_0 faster than sink_0.5) = 0.40 [0.25, 0.50] (n=10)
- wd=1, lr=0.001: P(sink_0 faster than sink_1) = 0.30 [0.15, 0.45] (n=10)
- wd=1, lr=0.001: P(sink_0 faster than sink_2) = 0.15 [0.00, 0.35] (n=10)
- wd=1, lr=0.001: P(sink_0 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.001: P(sink_0.25 faster than sink_0.5) = 0.40 [0.25, 0.50] (n=10)
- wd=1, lr=0.001: P(sink_0.25 faster than sink_1) = 0.30 [0.15, 0.45] (n=10)
- wd=1, lr=0.001: P(sink_0.25 faster than sink_2) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.001: P(sink_0.25 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.001: P(sink_0.5 faster than sink_1) = 0.40 [0.20, 0.60] (n=10)
- wd=1, lr=0.001: P(sink_0.5 faster than sink_2) = 0.15 [0.00, 0.35] (n=10)
- wd=1, lr=0.001: P(sink_0.5 faster than sink_4) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.001: P(sink_1 faster than sink_2) = 0.15 [0.00, 0.35] (n=10)
- wd=1, lr=0.001: P(sink_1 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.001: P(sink_2 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(par faster than seq) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_0) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_0.25) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_0.5) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_1) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_2) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(par faster than sink_4) = 1.00 [1.00, 1.00] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_0) = 0.30 [0.00, 0.60] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_0.25) = 0.40 [0.10, 0.70] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_0.5) = 0.45 [0.15, 0.75] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_1) = 0.40 [0.10, 0.70] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_2) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(seq faster than sink_4) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.003: P(sink_0 faster than sink_0.25) = 0.35 [0.10, 0.65] (n=10)
- wd=1, lr=0.003: P(sink_0 faster than sink_0.5) = 0.40 [0.15, 0.70] (n=10)
- wd=1, lr=0.003: P(sink_0 faster than sink_1) = 0.10 [0.00, 0.30] (n=10)
- wd=1, lr=0.003: P(sink_0 faster than sink_2) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(sink_0 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(sink_0.25 faster than sink_0.5) = 0.45 [0.20, 0.70] (n=10)
- wd=1, lr=0.003: P(sink_0.25 faster than sink_1) = 0.10 [0.00, 0.25] (n=10)
- wd=1, lr=0.003: P(sink_0.25 faster than sink_2) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(sink_0.25 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(sink_0.5 faster than sink_1) = 0.30 [0.00, 0.60] (n=10)
- wd=1, lr=0.003: P(sink_0.5 faster than sink_2) = 0.05 [0.00, 0.15] (n=10)
- wd=1, lr=0.003: P(sink_0.5 faster than sink_4) = 0.00 [0.00, 0.00] (n=10)
- wd=1, lr=0.003: P(sink_1 faster than sink_2) = 0.10 [0.00, 0.30] (n=10)
- wd=1, lr=0.003: P(sink_1 faster than sink_4) = 0.10 [0.00, 0.30] (n=10)
- wd=1, lr=0.003: P(sink_2 faster than sink_4) = 0.40 [0.15, 0.65] (n=10)
