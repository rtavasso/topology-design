# Stage 2 and stage 3 — read × block (EXPLORATORY)

Design, pre-fixed readings and predictions R2a–R2g: `DESIGN.md` (revision 2) and `stage2.py`, committed before these runs. Twelve splits (three teachers), 4,000 steps; errors are median [min, max]; references and regimes as in `stage0.md`. Per-run numbers: `stage2_<task>_<coverage>_<arm>.json`.

## Tuning (coverage 9, splits 100–102; chosen by seen error; held-out error shown for every configuration)

| arm, configuration | seen | held out | regime (runs) |
|---|---|---|---|
| select streams_self {'lr': 0.003, 'score': 'softmax'} | 0.02 [0.013, 0.023] | 0.4 [0.38, 0.51] | partial 2, worse-than-blend 1 |
| select streams_self {'lr': 0.01, 'score': 'softmax'} | 0.0092 [0.0063, 0.012] | 0.41 [0.34, 0.51] | partial 2, worse-than-blend 1 |
| select streams_self {'lr': 0.03, 'score': 'softmax'} | 0.0048 [0.0042, 0.01] | 0.46 [0.35, 0.55] | partial 2, worse-than-blend 1 |
| select streams_self {'lr': 0.1, 'score': 'softmax'} | 11 [0.69, 27] | 39 [0.83, 45] | garbage 3 |
| select streams_self {'lr': 0.003, 'score': 'sigmoid'} | 0.15 [0.13, 0.16] | 0.67 [0.57, 0.86] | garbage 1, worse-than-blend 2 |
| select streams_self {'lr': 0.01, 'score': 'sigmoid'} | 0.11 [0.096, 0.12] | 0.78 [0.7, 0.88] | garbage 1, worse-than-blend 2 |
| select streams_self {'lr': 0.03, 'score': 'sigmoid'} | 0.072 [0.056, 0.077] | 0.85 [0.69, 0.86] | garbage 2, worse-than-blend 1 |
| select streams_self {'lr': 0.1, 'score': 'sigmoid'} | 0.2 [0.17, 1] | 0.86 [0.64, 1] | garbage 2, worse-than-blend 1 |
| select streams_joint {'lr': 0.003, 'score': 'softmax'} | 0.016 [0.012, 0.018] | 0.44 [0.42, 0.66] | partial 2, worse-than-blend 1 |
| select streams_joint {'lr': 0.01, 'score': 'softmax'} | 0.0092 [0.0066, 0.013] | 0.58 [0.4, 0.65] | partial 1, worse-than-blend 2 |
| select streams_joint {'lr': 0.03, 'score': 'softmax'} | 0.0076 [0.0075, 0.0093] | 0.59 [0.37, 0.62] | partial 1, worse-than-blend 2 |
| select streams_joint {'lr': 0.1, 'score': 'softmax'} | 7.7 [0.64, 17] | 5.9 [0.9, 31] | garbage 3 |
| select streams_joint {'lr': 0.003, 'score': 'sigmoid'} | 0.035 [0.03, 0.035] | 0.64 [0.53, 0.81] | garbage 1, worse-than-blend 2 |
| select streams_joint {'lr': 0.01, 'score': 'sigmoid'} | 0.014 [0.013, 0.018] | 0.64 [0.56, 0.67] | worse-than-blend 3 |
| select streams_joint {'lr': 0.03, 'score': 'sigmoid'} | 0.0076 [0.0069, 0.011] | 0.74 [0.65, 0.79] | garbage 2, worse-than-blend 1 |
| select streams_joint {'lr': 0.1, 'score': 'sigmoid'} | 1 [1, 1] | 1 [1, 1] | garbage 3 |
| select tok_streams {'lr': 0.003} | 0.013 [0.0078, 0.013] | 0.031 [0.027, 0.053] | beyond 1, full 2 |
| select tok_streams {'lr': 0.01} | 0.0076 [0.0047, 0.0084] | 0.034 [0.026, 0.077] | beyond 3 |
| select tok_streams {'lr': 0.03} | 0.0077 [0.0061, 0.0084] | 0.055 [0.046, 0.092] | beyond 3 |
| select tok_streams {'lr': 0.1} | 0.012 [0.0061, 0.013] | 0.08 [0.071, 0.11] | beyond 2, recombines 1 |
| select unit {'lr': 0.003} | 0.063 [0.05, 0.064] | 0.55 [0.49, 0.61] | worse-than-blend 3 |
| select unit {'lr': 0.01} | 0.031 [0.027, 0.035] | 0.64 [0.58, 0.78] | worse-than-blend 3 |
| select unit {'lr': 0.03} | 0.012 [0.012, 0.019] | 0.58 [0.56, 0.73] | worse-than-blend 3 |
| select unit {'lr': 0.1} | 0.041 [0.037, 0.079] | 0.68 [0.65, 0.84] | garbage 1, worse-than-blend 2 |
| select oracle {'lr': 0.003} | 0.00069 [0.00042, 0.0007] | 0.00054 [0.00027, 0.00061] | full 3 |
| select oracle {'lr': 0.01} | 0.00016 [0.00015, 0.00045] | 8.4e-05 [6.1e-06, 0.00014] | full 3 |
| select oracle {'lr': 0.03} | 7.6e-05 [2.5e-05, 9.9e-05] | 3.5e-06 [3.2e-06, 4.7e-05] | full 3 |
| select oracle {'lr': 0.1} | 5.7e-05 [2.1e-06, 9.2e-05] | 2.6e-06 [2.1e-06, 0.00016] | full 3 |
| select oracle {'lr': 0.3} | 5.3e+02 [9.8e-05, 7.6e+02] | 7.5e+02 [0.00019, 8.9e+02] | full 1, garbage 2 |
| sum streams_self {'lr': 0.003, 'score': 'softmax'} | 0.00025 [0.00022, 0.00029] | 0.26 [0.22, 0.38] | partial 2, recombines 1 |
| sum streams_self {'lr': 0.01, 'score': 'softmax'} | 2.1e-05 [1.3e-05, 2.9e-05] | 0.24 [0.24, 0.25] | partial 2, recombines 1 |
| sum streams_self {'lr': 0.03, 'score': 'softmax'} | 8e-06 [6.2e-06, 8e-06] | 0.3 [0.22, 0.49] | partial 2, recombines 1 |
| sum streams_self {'lr': 0.1, 'score': 'softmax'} | 7.7e-06 [7.3e-06, 9.6e-06] | 0.17 [0.016, 0.56] | garbage 1, recombines 2 |
| sum streams_self {'lr': 0.003, 'score': 'sigmoid'} | 0.00077 [0.00031, 0.0009] | 0.025 [0.0049, 0.38] | partial 1, recombines 2 |
| sum streams_self {'lr': 0.01, 'score': 'sigmoid'} | 4.1e-05 [3.3e-05, 6.2e-05] | 0.032 [0.0056, 0.42] | partial 1, recombines 2 |
| sum streams_self {'lr': 0.03, 'score': 'sigmoid'} | 6e-06 [4.6e-06, 1.4e-05] | 0.31 [0.15, 0.53] | garbage 1, partial 1, recombines 1 |
| sum streams_self {'lr': 0.1, 'score': 'sigmoid'} | 6.4e-06 [6.1e-06, 7.4e-06] | 0.36 [0.13, 0.5] | partial 2, recombines 1 |
| sum tok_streams {'lr': 0.003} | 0.00016 [0.00015, 0.0002] | 0.00038 [0.00016, 0.00045] | full 3 |
| sum tok_streams {'lr': 0.01} | 9.3e-06 [8.7e-06, 1.4e-05] | 1.9e-05 [1e-05, 3.5e-05] | full 3 |
| sum tok_streams {'lr': 0.03} | 2.8e-06 [1.9e-06, 3.1e-06] | 3.7e-06 [3.7e-06, 7.2e-06] | full 3 |
| sum tok_streams {'lr': 0.1} | 1.4e-06 [1.3e-06, 1.7e-06] | 3e-06 [2.4e-06, 3.2e-06] | full 3 |
| sum tok_streams {'lr': 0.3} | 1.2e-05 [1e-05, 1.2e-05] | 3e-05 [1.8e-05, 4.3e-05] | full 2, recombines 1 |
| sum unit {'lr': 0.003} | 0.0016 [0.0011, 0.0017] | 0.15 [0.035, 0.34] | partial 1, recombines 2 |
| sum unit {'lr': 0.01} | 4e-05 [3.4e-05, 6.5e-05] | 0.14 [0.015, 0.32] | partial 1, recombines 2 |
| sum unit {'lr': 0.03} | 4.3e-06 [3.6e-06, 4.8e-06] | 0.14 [0.038, 0.34] | partial 1, recombines 2 |
| sum unit {'lr': 0.1} | 9.4e-07 [6.3e-07, 1e-06] | 0.3 [0.25, 0.51] | partial 3 |
| sum unit {'lr': 0.3} | 0.39 [0.27, 0.83] | 0.85 [0.8, 0.92] | garbage 3 |
| sum oracle {'lr': 0.003} | 8.2e-05 [5.7e-05, 9.8e-05] | 0.00012 [9.6e-05, 0.00014] | full 3 |
| sum oracle {'lr': 0.01} | 4.1e-06 [4.1e-06, 4.3e-06] | 6.6e-06 [6.1e-06, 7e-06] | full 3 |
| sum oracle {'lr': 0.03} | 4.8e-07 [4.7e-07, 5.6e-07] | 7.3e-07 [6.3e-07, 9.6e-07] | full 3 |
| sum oracle {'lr': 0.1} | 4.3e-07 [3e-07, 6.2e-07] | 4.9e-07 [4.7e-07, 7.7e-07] | full 3 |

Chosen: select — streams_self 0.03 softmax, streams_joint 0.03 softmax, tok_streams 0.01, unit 0.03, oracle 0.1; sum — streams_self 0.03 sigmoid, tok_streams 0.1, unit 0.1, oracle 0.1. `mono` and `tok_mono` from stage 0. Where the best rate was 0.1, 0.3 was tried and was worse.

## Main runs

| task, coverage, arm, configuration | parameters | seen | held out | regime (runs of 12) | one pass on held-out worlds: distance to the true update | distance to the blend |
|---|---|---|---|---|---|---|
| select, coverage 9, mono {'lr': 0.03} | 2528 | 0.066 [0.055, 0.075] | 1.3 [1, 1.7] | garbage 12 | 1.35 | 1.00 |
| select, coverage 9, tok_mono {'lr': 0.01} | 2568 | 0.025 [0.02, 0.027] | 0.059 [0.031, 0.1] | beyond 3, full 9 | 0.08 | 0.40 |
| select, coverage 9, streams_self {'lr': 0.03, 'score': 'softmax'} | 2480 | 0.0071 [0.0066, 0.0091] | 0.42 [0.19, 0.63] | partial 7, recombines 1, worse-than-blend 4 | 0.46 | 0.56 |
| select, coverage 9, tok_streams {'lr': 0.01} | 2480 | 0.0081 [0.0055, 0.011] | 0.041 [0.015, 0.092] | beyond 10, full 2 | 0.05 | 0.42 |
| select, coverage 9, null_tok_mono {'lr': 0.01} | 2568 | 0.025 [0.022, 0.028] | 0.056 [0.031, 0.095] | beyond 2, full 10 | 0.08 | 0.41 |
| select, coverage 9, null_mono {'lr': 0.03} | 2528 | 0.068 [0.061, 0.077] | 1.3 [1.1, 1.9] | garbage 12 | 1.38 | 1.04 |
| select, coverage 9, tok_streams_lr0.03 {'lr': 0.03} | 2480 | 0.0071 [0.0054, 0.0096] | 0.061 [0.016, 0.12] | beyond 11, full 1 | 0.08 | 0.43 |
| select, coverage 9, tok_mono_lr0.03 {'lr': 0.03} | 2568 | 0.023 [0.02, 0.027] | 0.072 [0.036, 0.15] | beyond 8, full 4 | 0.09 | 0.41 |
| select, coverage 9, streams_joint {'lr': 0.03, 'score': 'softmax'} | 2476 | 0.0088 [0.0057, 0.014] | 0.39 [0.23, 0.68] | partial 8, recombines 1, worse-than-blend 3 | 0.44 | 0.53 |
| select, coverage 9, unit {'lr': 0.03} | 2508 | 0.015 [0.011, 0.019] | 0.63 [0.47, 0.79] | garbage 1, partial 1, worse-than-blend 10 | 0.68 | 0.41 |
| select, coverage 9, oracle {'lr': 0.1} | 1200 | 2e-05 [1.2e-06, 0.00035] | 6e-06 [1.2e-06, 0.00025] | beyond 4, full 8 | 0.00 | 0.48 |
| select, coverage 9, mono_wd0.001 {'lr': 0.03, 'wd': 0.001} | 2528 | 0.066 [0.055, 0.075] | 1.3 [1, 1.7] | garbage 12 | 1.33 | 0.98 |
| select, coverage 9, mono_wd0.01 {'lr': 0.03, 'wd': 0.01} | 2528 | 0.067 [0.058, 0.076] | 1.2 [1, 1.5] | garbage 12 | 1.21 | 0.86 |
| select, coverage 9, mono_wd0.1 {'lr': 0.03, 'wd': 0.1} | 2528 | 0.088 [0.073, 0.096] | 0.92 [0.72, 1.2] | garbage 9, worse-than-blend 3 | 0.91 | 0.59 |
| select, coverage 9, tok_streams_lr0.1 {'lr': 0.1} | 2480 | 0.013 [0.0073, 66] | 0.12 [0.042, 84] | beyond 6, garbage 3, recombines 3 | 0.16 | 0.44 |
| select, coverage 9, mono_wide {'lr': 0.01, 'W': 120, 'steps': 8000} | 17088 | 0.023 [0.02, 0.024] | 1 [0.8, 1.2] | garbage 12 | 1.03 | 0.72 |
| select, coverage 7, mono {'lr': 0.03} | 2528 | 0.054 [0.049, 0.064] | 1.3 [1.1, 1.7] | garbage 12 | 1.31 | 0.91 |
| select, coverage 7, tok_mono {'lr': 0.01} | 2568 | 0.025 [0.02, 0.028] | 0.081 [0.041, 0.11] | beyond 7, full 5 | 0.10 | 0.41 |
| select, coverage 7, streams_self {'lr': 0.03, 'score': 'softmax'} | 2480 | 0.0084 [0.0045, 0.017] | 0.7 [0.62, 0.92] | garbage 4, worse-than-blend 8 | 0.78 | 0.71 |
| select, coverage 7, tok_streams {'lr': 0.01} | 2480 | 0.009 [0.0058, 0.013] | 0.054 [0.018, 0.12] | beyond 11, full 1 | 0.06 | 0.43 |
| select, coverage 7, null_tok_mono {'lr': 0.01} | 2568 | 0.026 [0.022, 0.03] | 0.086 [0.044, 0.13] | beyond 8, full 4 | 0.11 | 0.40 |
| select, coverage 7, tok_streams_lr0.03 {'lr': 0.03} | 2480 | 0.0089 [0.0057, 0.012] | 0.073 [0.03, 0.15] | beyond 11, full 1 | 0.09 | 0.43 |
| select, coverage 7, tok_mono_lr0.03 {'lr': 0.03} | 2568 | 0.025 [0.018, 0.031] | 0.11 [0.077, 0.17] | beyond 8, full 2, recombines 2 | 0.14 | 0.39 |
| select, coverage 7, streams_joint {'lr': 0.03, 'score': 'softmax'} | 2476 | 0.009 [0.0059, 0.016] | 0.74 [0.59, 0.95] | garbage 4, worse-than-blend 8 | 0.82 | 0.68 |
| select, coverage 7, unit {'lr': 0.03} | 2508 | 0.012 [0.0076, 0.014] | 0.75 [0.6, 0.86] | garbage 4, worse-than-blend 8 | 0.78 | 0.50 |
| select, coverage 7, oracle {'lr': 0.1} | 1200 | 2.2e-05 [6.2e-07, 0.00022] | 3.2e-05 [6.3e-07, 0.00012] | beyond 3, full 9 | 0.00 | 0.48 |
| select, coverage 7, null_mono {'lr': 0.03} | 2528 | 0.055 [0.046, 0.065] | 1.3 [1.2, 1.7] | garbage 12 | 1.32 | 0.95 |
| sum, coverage 9, mono {'lr': 0.03} | 2528 | 0.0039 [0.0032, 0.0057] | 1.1 [0.88, 1.5] | garbage 12 | 1.06 | 1.06 |
| sum, coverage 9, tok_mono {'lr': 0.1} | 2568 | 0.0009 [0.00074, 0.0011] | 0.0015 [0.0012, 0.0021] | full 12 | 0.00 | 0.00 |
| sum, coverage 9, streams_self {'lr': 0.03, 'score': 'sigmoid'} | 2480 | 6.3e-06 [4.3e-06, 1.5e-05] | 0.15 [0.088, 0.31] | partial 4, recombines 8 | 0.14 | 0.14 |
| sum, coverage 9, tok_streams {'lr': 0.1} | 2480 | 1.5e-06 [9.1e-07, 2.1e-06] | 2.7e-06 [1.6e-06, 3.6e-06] | full 12 | 0.00 | 0.00 |
| sum, coverage 9, unit {'lr': 0.1} | 2508 | 8.1e-07 [7.1e-07, 1.1e-06] | 0.27 [0.1, 0.58] | garbage 1, partial 7, recombines 4 | 0.26 | 0.26 |
| sum, coverage 9, oracle {'lr': 0.1} | 1200 | 7.8e-07 [2.5e-07, 2e-06] | 9.4e-07 [3.6e-07, 2.3e-06] | full 12 | 0.00 | 0.00 |

Arms ending in `_lr…` are the same arm at another rate whose seen error is within 2× of its best; `_wd…` is `mono` with weight decay; `null_…` is the same arm with another initialisation; `mono_wide` has width 120 and 8,000 steps.

## Contrasts: select 9

| contrast (A / B) | held-out error ratio: median of 12 paired [5 %, 95 % bootstrap] | splits where A is lower | same ratio on seen worlds |
|---|---|---|---|
| C1 streams under the per-token read: tok_streams / tok_mono | 0.65 [0.51, 0.86] | 9/12 | 0.33 |
| C2 streams under the summed read: streams_self / mono | 0.32 [0.30, 0.37] | 12/12 | 0.11 |
| C2' joint-gate streams under the summed read: streams_joint / mono | 0.32 [0.27, 0.35] | 12/12 | 0.14 |
| C3 the read, one block: tok_mono / mono | 0.05 [0.04, 0.06] | 12/12 | 0.37 |
| C4 the read, streams: tok_streams / streams_self | 0.10 [0.06, 0.14] | 12/12 | 1.10 |
| gating without grouping against streams: unit / streams_self | 1.37 [1.19, 1.56] | 0/12 | 1.87 |
| null: tok_mono, another initialisation: null_tok_mono / tok_mono | 0.94 [0.86, 0.98] | 9/12 | 1.02 |
| null: mono, another initialisation: null_mono / mono | 1.03 [0.91, 1.10] | 6/12 | 1.03 |

## Contrasts: select 7

| contrast (A / B) | held-out error ratio: median of 12 paired [5 %, 95 % bootstrap] | splits where A is lower | same ratio on seen worlds |
|---|---|---|---|
| C1 streams under the per-token read: tok_streams / tok_mono | 0.85 [0.59, 0.95] | 9/12 | 0.34 |
| C2 streams under the summed read: streams_self / mono | 0.55 [0.51, 0.62] | 12/12 | 0.15 |
| C2' joint-gate streams under the summed read: streams_joint / mono | 0.58 [0.52, 0.65] | 12/12 | 0.15 |
| C3 the read, one block: tok_mono / mono | 0.07 [0.05, 0.08] | 12/12 | 0.46 |
| C4 the read, streams: tok_streams / streams_self | 0.08 [0.06, 0.11] | 12/12 | 1.02 |
| gating without grouping against streams: unit / streams_self | 1.00 [0.90, 1.06] | 6/12 | 1.29 |
| null: tok_mono, another initialisation: null_tok_mono / tok_mono | 1.05 [0.88, 1.13] | 4/12 | 1.02 |
| null: mono, another initialisation: null_mono / mono | 1.05 [0.93, 1.19] | 5/12 | 0.98 |

## Contrasts: sum 9

| contrast (A / B) | held-out error ratio: median of 12 paired [5 %, 95 % bootstrap] | splits where A is lower | same ratio on seen worlds |
|---|---|---|---|
| C1 streams under the per-token read: tok_streams / tok_mono | 0.00 [0.00, 0.00] | 12/12 | 0.00 |
| C2 streams under the summed read: streams_self / mono | 0.14 [0.10, 0.23] | 12/12 | 0.00 |
| C3 the read, one block: tok_mono / mono | 0.00 [0.00, 0.00] | 12/12 | 0.21 |
| C4 the read, streams: tok_streams / streams_self | 0.00 [0.00, 0.00] | 12/12 | 0.21 |
| gating without grouping against streams: unit / streams_self | 1.62 [1.05, 2.09] | 3/12 | 0.12 |

## Stage 3 — mechanism (select, coverage 9, six splits, retrained)

| arm | decision accuracy, one pass: seen worlds | held out, forced by trained comparisons | held out, not forced | share of held-out decisions that are forced |
|---|---|---|---|---|
| tok_mono | 0.982 [0.982, 0.985] | 0.999 [0.996, 1.000] | 0.927 [0.901, 0.967] | 0.45 |
| tok_streams | 0.990 [0.988, 0.991] | 0.999 [0.999, 1.000] | 0.943 [0.870, 0.976] | 0.45 |

tok_streams: each token has its own stream in 6/6 splits; removing a token's stream raises the error on seen worlds containing that token 9719 times more than on the others (median over 36 removals; minimum 101.4).
