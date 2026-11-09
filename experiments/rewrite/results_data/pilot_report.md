# Rung 4 pilot: the readings PILOT.md names (24 runs, seed [100], [3000] steps)

Read here: lr per arm, steps, throughput, harness sanity — on each arm's own training-world rows of eval5. Not read: held-out worlds, the binding stratum, sizes 6–7, k = 7–8, any arm against another (L30).

Curve subset check (first 100 per world of the per-instance records = the intermediate points' set): **holds** on 24 runs.

## 1. lr per arm: training-world eval5 masked success at the last step (full 600 per world); ties by the lower loss

| arm | lr | n | curve on the first 100 per world (steps 0, 500, 1000, 1500, 2000, 2500, 3000) | final masked | final unmasked | RT-all on the eight | loss (last interval) |  |
|---|---|---|---|---|---|---|---|---|
| RT-canon | 0.0003 | 4800 | 0.00 0.47 0.77 0.80 0.87 0.88 0.90 | 0.902 | 0.895 |  | 0.197 |  |
| RT-canon | 0.001 | 4800 | 0.00 0.76 0.84 0.88 0.91 0.93 0.93 | 0.938 | 0.933 |  | 0.148 | **chosen** |
| RT-canon | 0.003 | 4800 | 0.00 0.74 0.81 0.84 0.88 0.87 0.89 | 0.888 | 0.877 |  | 0.218 |  |
| RT-post | 0.0003 | 4800 | 0.00 0.46 0.77 0.83 0.87 0.88 0.91 | 0.919 | 0.910 |  | 0.512 | **chosen** |
| RT-post | 0.001 | 4800 | 0.00 0.68 0.78 0.87 0.85 0.90 0.92 | 0.916 | 0.913 |  | 0.509 |  |
| RT-post | 0.003 | 4800 | 0.00 0.58 0.70 0.75 0.80 0.81 0.83 | 0.823 | 0.815 |  | 0.668 |  |
| RT-all | 0.0003 | 9000 | 0.00 0.38 0.67 0.75 0.79 0.81 0.85 | 0.844 | 0.833 | 0.886 | 0.346 |  |
| RT-all | 0.001 | 9000 | 0.00 0.65 0.79 0.82 0.85 0.89 0.91 | 0.905 | 0.898 | 0.927 | 0.251 | **chosen** |
| RT-all | 0.003 | 9000 | 0.00 0.60 0.75 0.79 0.77 0.82 0.83 | 0.841 | 0.831 | 0.876 | 0.363 |  |
| RT-absolute | 0.0003 | 4800 | 0.01 0.05 0.39 0.48 0.62 0.69 0.74 | 0.734 | 0.698 |  | 0.451 |  |
| RT-absolute | 0.001 | 4800 | 0.01 0.07 0.40 0.55 0.67 0.77 0.76 | 0.756 | 0.723 |  | 0.376 | **chosen** |
| RT-absolute | 0.003 | 4800 | 0.01 0.13 0.25 0.34 0.44 0.54 0.56 | 0.562 | 0.527 |  | 0.703 |  |
| EI16 | 0.0003 | 4800 | 0.00 0.49 0.74 0.84 0.87 0.87 0.89 | 0.888 | 0.877 |  | 0.115 |  |
| EI16 | 0.001 | 4800 | 0.00 0.57 0.74 0.83 0.89 0.89 0.91 | 0.899 | 0.891 |  | 0.109 | **chosen** |
| EI16 | 0.003 | 4800 | 0.00 0.40 0.47 0.53 0.58 0.64 0.68 | 0.673 | 0.639 |  | 0.412 |  |
| PF | 0.0003 | 4800 | 0.00 0.35 0.67 0.77 0.83 0.86 0.88 | 0.873 | 0.873 |  | 1.134 |  |
| PF | 0.001 | 4800 | 0.00 0.46 0.76 0.84 0.87 0.91 0.92 | 0.921 | 0.920 |  | 1.108 | **chosen** |
| PF | 0.003 | 4800 | 0.00 0.43 0.61 0.72 0.79 0.80 0.84 | 0.839 | 0.832 |  | 1.182 |  |
| RT-examples | 0.0003 | 4800 | 0.00 0.39 0.68 0.70 0.79 0.83 0.87 | 0.867 | 0.855 |  | 0.287 |  |
| RT-examples | 0.001 | 4800 | 0.00 0.68 0.78 0.80 0.83 0.84 0.87 | 0.871 | 0.860 |  | 0.262 | **chosen** |
| RT-examples | 0.003 | 4800 | 0.00 0.57 0.69 0.71 0.75 0.76 0.84 | 0.846 | 0.833 |  | 0.361 |  |
| RT-shuffled | 0.0003 | 4800 | 0.00 0.54 0.76 0.82 0.85 0.86 0.90 | 0.910 | 0.728 |  | 0.377 |  |
| RT-shuffled | 0.001 | 4800 | 0.00 0.76 0.84 0.90 0.90 0.92 0.93 | 0.934 | 0.751 |  | 0.318 | **chosen** |
| RT-shuffled | 0.003 | 4800 | 0.00 0.75 0.81 0.86 0.88 0.91 0.91 | 0.914 | 0.722 |  | 0.374 |  |

Chosen: RT-canon 0.001, RT-post 0.0003, RT-all 0.001, RT-absolute 0.001, EI16 0.001, PF 0.001, RT-examples 0.001, RT-shuffled 0.001.

## 2. Steps: is any arm's training-world curve still rising at the last step (at its chosen lr)?

| arm | lr | n | step 2000 | step 2500 | step 3000 | gain, last interval | gain, last two | binomial s.e. | loss change, last interval | reading |
|---|---|---|---|---|---|---|---|---|---|---|
| RT-canon | 0.001 | 800 | 0.914 | 0.926 | 0.929 | +0.002 | +0.015 | 0.009 | -12.2 % | rising |
| RT-post | 0.0003 | 800 | 0.873 | 0.880 | 0.914 | +0.034 | +0.041 | 0.010 | -6.5 % | rising |
| RT-all | 0.001 | 1500 | 0.849 | 0.887 | 0.907 | +0.021 | +0.059 | 0.007 | -10.0 % | rising |
| RT-absolute | 0.001 | 800 | 0.669 | 0.765 | 0.759 | -0.006 | +0.090 | 0.015 | -19.5 % | rising |
| EI16 | 0.001 | 800 | 0.887 | 0.887 | 0.905 | +0.018 | +0.018 | 0.010 | -18.1 % | rising |
| PF | 0.001 | 800 | 0.866 | 0.909 | 0.919 | +0.010 | +0.052 | 0.010 | -1.6 % | rising |
| RT-examples | 0.001 | 800 | 0.835 | 0.840 | 0.870 | +0.030 | +0.035 | 0.012 | -11.5 % | rising |
| RT-shuffled | 0.001 | 800 | 0.902 | 0.916 | 0.934 | +0.017 | +0.031 | 0.009 | -9.2 % | rising |

"Rising" = the gain over the last two intervals exceeds two binomial standard errors, or the training loss still fell by more than 5 % over the last interval. Rising: 8 of 8 arms → **12,000 steps** for the main grid (PILOT.md item 2).

## 3. Throughput, and the main grid at 12,000 steps with curves every 250

| arm | per training step | curve evaluation | full evaluation | pilot run (median of 3) | projected main run | seeds | arm total |
|---|---|---|---|---|---|---|---|
| RT-canon | 63 ms | 9.0 s | 54 s | 329 s | 21 min | 10 | 3.6 h |
| RT-post | 66 ms | 10.0 s | 60 s | 353 s | 23 min | 10 | 3.8 h |
| RT-all | 65 ms | 9.5 s | 57 s | 349 s | 22 min | 3 | 1.1 h |
| RT-absolute | 53 ms | 9.2 s | 55 s | 309 s | 20 min | 5 | 1.6 h |
| EI16 | 678 ms | 13.0 s | 78 s | 2140 s | 148 min | 5 | 12.3 h |
| PF | 87 ms | 10.8 s | 65 s | 421 s | 28 min | 3 | 1.4 h |
| RT-examples | 158 ms | 23.1 s | 139 s | 773 s | 53 min | 3 | 2.6 h |
| RT-shuffled | 65 ms | 7.8 s | 47 s | 324 s | 21 min | 2 | 0.7 h |

Sum over the 41 runs: **27.1 process-hours** → 13.6 h wall on two shards (one per T4), 6.8 h on four (two per T4) if the four processes do not slow one another. Relative cost per run (RT-canon = 1): RT-canon 1.0, RT-post 1.1, RT-all 1.0, RT-absolute 0.9, EI16 6.9, PF 1.3, RT-examples 2.5, RT-shuffled 1.0.

EI16's beam: share of instances with a route found, per interval (rows imitated per step in brackets; RT trains on 256):

| lr | to step 500 | to step 1000 | to step 1500 | to step 2000 | to step 2500 | to step 3000 |
|---|---|---|---|---|---|---|
| 0.0003 | 0.75 (86) | 0.95 (123) | 0.98 (130) | 0.99 (131) | 0.99 (134) | 0.99 (132) |
| 0.001 | 0.80 (97) | 0.95 (127) | 0.98 (130) | 0.99 (133) | 0.99 (133) | 0.99 (133) |
| 0.003 | 0.78 (93) | 0.85 (105) | 0.89 (113) | 0.91 (117) | 0.94 (126) | 0.95 (126) |

## 4. Harness sanity

| arm | cells with unmasked > masked | NaN losses | masked categories | unmasked categories | per-instance records | curve points | ladder stored | masked − unmasked on its training worlds (per lr) |
|---|---|---|---|---|---|---|---|---|
| RT-canon | 0 | 0 | horizon, ok, wrong halt | horizon, invalid: no match, invalid: out of world, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.007 +0.005 +0.011 |
| RT-post | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.008 +0.003 +0.008 |
| RT-all | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.011 +0.007 +0.011 |
| RT-absolute | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.036 +0.034 +0.036 |
| EI16 | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.011 +0.008 +0.034 |
| PF | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.000 +0.001 +0.007 |
| RT-examples | 0 | 0 | horizon, no halt at goal, ok, wrong halt | horizon, invalid: no match, invalid: out of world, no halt at goal, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.012 +0.011 +0.013 |
| RT-shuffled | 0 | 0 | horizon, ok, wrong halt | horizon, invalid: no match, invalid: out of world, ok, wrong halt | eval5 9000; eval7 9000; evalk78 600 | [7] | yes | +0.182 +0.183 +0.192 |
