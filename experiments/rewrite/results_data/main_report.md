runs: {'RT-canon': 10, 'RT-post': 10, 'RT-all': 3, 'RT-absolute': 5, 'EI16': 5, 'PF': 3, 'RT-examples': 3, 'RT-shuffled': 2}

The sections P2–P8 below are §6's readings as preregistered: printed, **not scored**. The scored verdicts, under the pass rules as amended before the main grid (DESIGN §17), are the last section.

## Runs classified from the curve (masked success, eval5 curve subsets; training worlds | held-out worlds)

| run | class (train) | max | final | class (held-out) | final | loss | time |
|---|---|---|---|---|---|---|---|
| RT-canon_s0 | learned | 0.96 | 0.96 | learned | 0.69 | 0.077 | 30 min |
| RT-canon_s1 | learned | 0.97 | 0.96 | learned | 0.70 | 0.071 | 21 min |
| RT-canon_s2 | learned | 0.98 | 0.98 | learned | 0.66 | 0.055 | 22 min |
| RT-canon_s3 | learned | 0.98 | 0.96 | learned (partial) | 0.70 | 0.056 | 20 min |
| RT-canon_s4 | learned | 0.97 | 0.97 | learned | 0.70 | 0.072 | 26 min |
| RT-canon_s5 | learned | 0.98 | 0.98 | learned (partial) | 0.66 | 0.057 | 20 min |
| RT-canon_s6 | learned | 0.98 | 0.98 | learned | 0.79 | 0.061 | 22 min |
| RT-canon_s7 | learned | 0.98 | 0.97 | learned | 0.68 | 0.061 | 19 min |
| RT-canon_s8 | learned | 0.95 | 0.95 | learned | 0.66 | 0.093 | 21 min |
| RT-canon_s9 | learned | 0.98 | 0.98 | learned | 0.63 | 0.060 | 22 min |
| RT-post_s0 | learned | 0.99 | 0.99 | learned | 0.80 | 0.359 | 37 min |
| RT-post_s1 | learned | 0.99 | 0.99 | learned | 0.77 | 0.352 | 39 min |
| RT-post_s2 | learned | 0.98 | 0.98 | learned | 0.73 | 0.360 | 36 min |
| RT-post_s3 | learned | 0.99 | 0.99 | learned | 0.73 | 0.359 | 39 min |
| RT-post_s4 | learned | 0.99 | 0.99 | learned | 0.80 | 0.369 | 37 min |
| RT-post_s5 | learned | 0.99 | 0.98 | learned | 0.67 | 0.353 | 40 min |
| RT-post_s6 | learned | 0.98 | 0.98 | learned | 0.79 | 0.368 | 36 min |
| RT-post_s7 | learned | 0.99 | 0.99 | learned | 0.77 | 0.355 | 39 min |
| RT-post_s8 | learned | 0.99 | 0.99 | learned | 0.80 | 0.359 | 37 min |
| RT-post_s9 | learned | 0.99 | 0.99 | learned | 0.80 | 0.351 | 36 min |
| RT-all_s0 | learned | 0.97 | 0.96 | learned | 0.92 | 0.125 | 27 min |
| RT-all_s1 | learned | 0.97 | 0.97 | learned | 0.93 | 0.125 | 29 min |
| RT-all_s2 | learned | 0.98 | 0.98 | learned | 0.96 | 0.082 | 33 min |
| RT-absolute_s0 | learned | 0.74 | 0.73 | learned | 0.52 | 0.336 | 31 min |
| RT-absolute_s1 | learned | 0.98 | 0.97 | learned (partial) | 0.65 | 0.064 | 33 min |
| RT-absolute_s2 | learned | 0.98 | 0.98 | learned | 0.73 | 0.068 | 31 min |
| RT-absolute_s3 | learned | 0.97 | 0.97 | flat | 0.34 | 0.079 | 33 min |
| RT-absolute_s4 | learned | 0.97 | 0.97 | learned | 0.73 | 0.078 | 32 min |
| EI16_s0 | learned | 0.96 | 0.96 | learned | 0.71 | 0.042 | 276 min |
| EI16_s1 | learned | 0.95 | 0.94 | learned (partial) | 0.63 | 0.058 | 276 min |
| EI16_s2 | learned | 0.96 | 0.96 | learned (partial) | 0.55 | 0.047 | 273 min |
| EI16_s3 | learned | 0.97 | 0.96 | learned | 0.69 | 0.038 | 272 min |
| EI16_s4 | learned | 0.97 | 0.96 | learned | 0.78 | 0.049 | 281 min |
| PF_s0 | learned | 1.00 | 1.00 | flat | 0.24 | 1.049 | 50 min |
| PF_s1 | learned | 1.00 | 1.00 | flat | 0.38 | 1.043 | 50 min |
| PF_s2 | learned | 0.99 | 0.99 | learned (partial) | 0.49 | 1.046 | 50 min |
| RT-examples_s0 | learned | 0.95 | 0.95 | learned | 0.59 | 0.109 | 87 min |
| RT-examples_s1 | learned | 0.96 | 0.96 | learned | 0.62 | 0.077 | 87 min |
| RT-examples_s2 | learned | 0.97 | 0.97 | learned | 0.63 | 0.069 | 84 min |
| RT-shuffled_s0 | learned | 0.96 | 0.96 | learned | 0.78 | 0.207 | 27 min |
| RT-shuffled_s1 | learned | 0.96 | 0.96 | learned | 0.80 | 0.199 | 29 min |

## Success on eval5 by arm × world class × binding: masked | unmasked | first decision correct; greedy rung on the same instances

| arm | class | binding | n/seed | masked | unmasked | first ok | greedy d=1 |
|---|---|---|---|---|---|---|---|
| RT-canon | train | 1 | 1408 | 0.94 [0.90, 0.96] n=10 | 0.93 [0.89, 0.95] n=10 | 0.94 [0.93, 0.96] n=10 | 0.40 |
| RT-canon | train | 0 | 3392 | 0.98 [0.97, 0.99] n=10 | 0.98 [0.97, 0.99] n=10 | 0.98 [0.97, 0.98] n=10 | 0.81 |
| RT-canon | pair | 1 | 608 | 0.77 [0.62, 0.87] n=10 | 0.63 [0.58, 0.79] n=10 | 0.91 [0.75, 0.96] n=10 | 0.41 |
| RT-canon | pair | 0 | 592 | 0.83 [0.77, 0.92] n=10 | 0.77 [0.69, 0.91] n=10 | 0.84 [0.79, 0.87] n=10 | 0.77 |
| RT-canon | triple | 1 | 1065 | 0.56 [0.48, 0.73] n=10 | 0.48 [0.40, 0.61] n=10 | 0.79 [0.71, 0.85] n=10 | 0.28 |
| RT-canon | triple | 0 | 1335 | 0.71 [0.65, 0.80] n=10 | 0.66 [0.63, 0.77] n=10 | 0.78 [0.75, 0.80] n=10 | 0.51 |
| RT-canon | full | 1 | 0 | – | – | – | – |
| RT-canon | full | 0 | 600 | 0.60 [0.52, 0.67] n=10 | 0.56 [0.49, 0.63] n=10 | 0.73 [0.68, 0.74] n=10 | 0.38 |
| RT-post | train | 1 | 1408 | 0.97 [0.95, 0.98] n=10 | 0.97 [0.95, 0.97] n=10 | 0.97 [0.96, 0.98] n=10 | 0.40 |
| RT-post | train | 0 | 3392 | 0.99 [0.99, 1.00] n=10 | 0.99 [0.99, 0.99] n=10 | 0.99 [0.99, 0.99] n=10 | 0.81 |
| RT-post | pair | 1 | 608 | 0.85 [0.69, 0.90] n=10 | 0.82 [0.65, 0.86] n=10 | 0.94 [0.87, 0.97] n=10 | 0.41 |
| RT-post | pair | 0 | 592 | 0.91 [0.85, 0.96] n=10 | 0.87 [0.82, 0.95] n=10 | 0.89 [0.88, 0.91] n=10 | 0.77 |
| RT-post | triple | 1 | 1065 | 0.66 [0.52, 0.70] n=10 | 0.63 [0.50, 0.65] n=10 | 0.84 [0.80, 0.87] n=10 | 0.28 |
| RT-post | triple | 0 | 1335 | 0.82 [0.73, 0.85] n=10 | 0.78 [0.71, 0.82] n=10 | 0.84 [0.80, 0.86] n=10 | 0.51 |
| RT-post | full | 1 | 0 | – | – | – | – |
| RT-post | full | 0 | 600 | 0.69 [0.60, 0.75] n=10 | 0.65 [0.57, 0.70] n=10 | 0.81 [0.76, 0.83] n=10 | 0.38 |
| RT-all | train | 1 | 1408 | 0.93 [0.91, 0.94] n=3 | 0.93 [0.91, 0.94] n=3 | 0.95 [0.94, 0.95] n=3 | 0.40 |
| RT-all | train | 0 | 3392 | 0.98 [0.98, 0.99] n=3 | 0.98 [0.98, 0.99] n=3 | 0.97 [0.97, 0.98] n=3 | 0.81 |
| RT-all | pair | 1 | 608 | 0.97 [0.97, 0.97] n=3 | 0.97 [0.97, 0.97] n=3 | 0.97 [0.97, 0.98] n=3 | 0.41 |
| RT-all | pair | 0 | 592 | 0.99 [0.99, 0.99] n=3 | 0.99 [0.99, 0.99] n=3 | 0.97 [0.96, 0.98] n=3 | 0.77 |
| RT-all | triple | 1 | 1065 | 0.90 [0.89, 0.92] n=3 | 0.89 [0.89, 0.91] n=3 | 0.91 [0.90, 0.91] n=3 | 0.28 |
| RT-all | triple | 0 | 1335 | 0.93 [0.91, 0.96] n=3 | 0.93 [0.90, 0.96] n=3 | 0.89 [0.89, 0.91] n=3 | 0.51 |
| RT-all | full | 1 | 0 | – | – | – | – |
| RT-all | full | 0 | 600 | 0.91 [0.90, 0.95] n=3 | 0.90 [0.90, 0.95] n=3 | 0.87 [0.86, 0.89] n=3 | 0.38 |
| RT-absolute | train | 1 | 1408 | 0.95 [0.59, 0.95] n=5 | 0.93 [0.57, 0.93] n=5 | 0.94 [0.89, 0.96] n=5 | 0.40 |
| RT-absolute | train | 0 | 3392 | 0.98 [0.78, 0.99] n=5 | 0.97 [0.76, 0.98] n=5 | 0.98 [0.87, 0.98] n=5 | 0.81 |
| RT-absolute | pair | 1 | 608 | 0.70 [0.18, 0.75] n=5 | 0.47 [0.07, 0.62] n=5 | 0.83 [0.59, 0.93] n=5 | 0.41 |
| RT-absolute | pair | 0 | 592 | 0.81 [0.32, 0.84] n=5 | 0.67 [0.16, 0.74] n=5 | 0.80 [0.60, 0.87] n=5 | 0.77 |
| RT-absolute | triple | 1 | 1065 | 0.60 [0.36, 0.66] n=5 | 0.43 [0.29, 0.60] n=5 | 0.77 [0.68, 0.82] n=5 | 0.28 |
| RT-absolute | triple | 0 | 1335 | 0.66 [0.39, 0.76] n=5 | 0.57 [0.33, 0.69] n=5 | 0.71 [0.60, 0.79] n=5 | 0.51 |
| RT-absolute | full | 1 | 0 | – | – | – | – |
| RT-absolute | full | 0 | 600 | 0.52 [0.35, 0.68] n=5 | 0.48 [0.32, 0.63] n=5 | 0.67 [0.54, 0.74] n=5 | 0.38 |
| EI16 | train | 1 | 1408 | 0.92 [0.90, 0.92] n=5 | 0.91 [0.88, 0.92] n=5 | 0.88 [0.87, 0.89] n=5 | 0.40 |
| EI16 | train | 0 | 3392 | 0.98 [0.96, 0.98] n=5 | 0.97 [0.96, 0.98] n=5 | 0.91 [0.91, 0.92] n=5 | 0.81 |
| EI16 | pair | 1 | 608 | 0.81 [0.58, 0.92] n=5 | 0.68 [0.37, 0.78] n=5 | 0.93 [0.77, 0.96] n=5 | 0.41 |
| EI16 | pair | 0 | 592 | 0.83 [0.76, 0.92] n=5 | 0.77 [0.49, 0.82] n=5 | 0.82 [0.78, 0.84] n=5 | 0.77 |
| EI16 | triple | 1 | 1065 | 0.57 [0.33, 0.68] n=5 | 0.44 [0.29, 0.53] n=5 | 0.73 [0.69, 0.78] n=5 | 0.28 |
| EI16 | triple | 0 | 1335 | 0.70 [0.64, 0.78] n=5 | 0.59 [0.55, 0.73] n=5 | 0.72 [0.68, 0.74] n=5 | 0.51 |
| EI16 | full | 1 | 0 | – | – | – | – |
| EI16 | full | 0 | 600 | 0.52 [0.50, 0.69] n=5 | 0.50 [0.47, 0.66] n=5 | 0.65 [0.56, 0.68] n=5 | 0.38 |
| PF | train | 1 | 1408 | 0.99 [0.98, 1.00] n=3 | 0.99 [0.97, 1.00] n=3 | 0.96 [0.95, 0.97] n=3 | 0.40 |
| PF | train | 0 | 3392 | 1.00 [0.99, 1.00] n=3 | 1.00 [0.99, 1.00] n=3 | 0.98 [0.98, 0.99] n=3 | 0.81 |
| PF | pair | 1 | 608 | 0.31 [0.06, 0.59] n=3 | 0.31 [0.06, 0.59] n=3 | 0.84 [0.61, 0.91] n=3 | 0.41 |
| PF | pair | 0 | 592 | 0.57 [0.17, 0.73] n=3 | 0.55 [0.16, 0.72] n=3 | 0.82 [0.72, 0.84] n=3 | 0.77 |
| PF | triple | 1 | 1065 | 0.23 [0.16, 0.28] n=3 | 0.23 [0.16, 0.27] n=3 | 0.74 [0.71, 0.78] n=3 | 0.28 |
| PF | triple | 0 | 1335 | 0.50 [0.32, 0.51] n=3 | 0.50 [0.32, 0.50] n=3 | 0.75 [0.68, 0.75] n=3 | 0.51 |
| PF | full | 1 | 0 | – | – | – | – |
| PF | full | 0 | 600 | 0.37 [0.30, 0.51] n=3 | 0.37 [0.30, 0.51] n=3 | 0.64 [0.63, 0.71] n=3 | 0.38 |
| RT-examples | train | 1 | 1408 | 0.91 [0.90, 0.94] n=3 | 0.90 [0.89, 0.92] n=3 | 0.93 [0.92, 0.95] n=3 | 0.40 |
| RT-examples | train | 0 | 3392 | 0.98 [0.97, 0.98] n=3 | 0.98 [0.97, 0.98] n=3 | 0.97 [0.96, 0.98] n=3 | 0.81 |
| RT-examples | pair | 1 | 608 | 0.64 [0.59, 0.70] n=3 | 0.52 [0.50, 0.53] n=3 | 0.89 [0.88, 0.93] n=3 | 0.41 |
| RT-examples | pair | 0 | 592 | 0.70 [0.67, 0.74] n=3 | 0.62 [0.58, 0.63] n=3 | 0.76 [0.75, 0.79] n=3 | 0.77 |
| RT-examples | triple | 1 | 1065 | 0.52 [0.47, 0.54] n=3 | 0.39 [0.31, 0.41] n=3 | 0.75 [0.74, 0.78] n=3 | 0.28 |
| RT-examples | triple | 0 | 1335 | 0.66 [0.65, 0.70] n=3 | 0.60 [0.58, 0.62] n=3 | 0.74 [0.72, 0.77] n=3 | 0.51 |
| RT-examples | full | 1 | 0 | – | – | – | – |
| RT-examples | full | 0 | 600 | 0.57 [0.57, 0.61] n=3 | 0.55 [0.53, 0.56] n=3 | 0.69 [0.69, 0.71] n=3 | 0.38 |
| RT-shuffled | train | 1 | 1408 | 0.91 [0.90, 0.91] n=2 | 0.58 [0.58, 0.58] n=2 | 0.93 [0.92, 0.93] n=2 | 0.40 |
| RT-shuffled | train | 0 | 3392 | 0.98 [0.97, 0.98] n=2 | 0.87 [0.87, 0.87] n=2 | 0.97 [0.97, 0.97] n=2 | 0.81 |
| RT-shuffled | pair | 1 | 608 | 0.85 [0.85, 0.85] n=2 | 0.54 [0.52, 0.55] n=2 | 0.92 [0.91, 0.92] n=2 | 0.41 |
| RT-shuffled | pair | 0 | 592 | 0.86 [0.86, 0.86] n=2 | 0.66 [0.63, 0.70] n=2 | 0.79 [0.79, 0.80] n=2 | 0.77 |
| RT-shuffled | triple | 1 | 1065 | 0.70 [0.68, 0.72] n=2 | 0.42 [0.42, 0.42] n=2 | 0.80 [0.80, 0.80] n=2 | 0.28 |
| RT-shuffled | triple | 0 | 1335 | 0.81 [0.81, 0.81] n=2 | 0.70 [0.69, 0.70] n=2 | 0.77 [0.76, 0.78] n=2 | 0.51 |
| RT-shuffled | full | 1 | 0 | – | – | – | – |
| RT-shuffled | full | 0 | 600 | 0.77 [0.76, 0.78] n=2 | 0.73 [0.72, 0.74] n=2 | 0.76 [0.76, 0.76] n=2 | 0.38 |

## P2 (Q1): held-out pairs' binding stratum, eval5 — RT-canon against the greedy rung on the same instances; RT-all's ceiling. Pass rule on the masked reading; unmasked alongside.
| arm | worlds | n/seed | masked (median [min,max]) | unmasked | greedy d=1 | > greedy | ≥ greedy+0.2 and ≥ 0.7 (same seed) | ≥ 0.9 |
|---|---|---|---|---|---|---|---|---|
| RT-canon | pair | 608 | 0.77 [0.62, 0.87] n=10 | 0.63 [0.58, 0.79] n=10 | 0.41 | 10/10 | 7/10 | 0/10 |
| RT-canon | triple+full | 1065 | 0.56 [0.48, 0.73] n=10 | 0.48 [0.40, 0.61] n=10 | 0.28 | 10/10 | 1/10 | 0/10 |
| RT-post | pair | 608 | 0.85 [0.69, 0.90] n=10 | 0.82 [0.65, 0.86] n=10 | 0.41 | 10/10 | 9/10 | 1/10 |
| RT-post | triple+full | 1065 | 0.66 [0.52, 0.70] n=10 | 0.63 [0.50, 0.65] n=10 | 0.28 | 10/10 | 0/10 | 0/10 |
| RT-all | pair | 608 | 0.97 [0.97, 0.97] n=3 | 0.97 [0.97, 0.97] n=3 | 0.41 | 3/3 | 3/3 | 3/3 |
| RT-all | triple+full | 1065 | 0.90 [0.89, 0.92] n=3 | 0.89 [0.89, 0.91] n=3 | 0.28 | 3/3 | 3/3 | 1/3 |
| EI16 | pair | 608 | 0.81 [0.58, 0.92] n=5 | 0.68 [0.37, 0.78] n=5 | 0.41 | 5/5 | 4/5 | 1/5 |
| EI16 | triple+full | 1065 | 0.57 [0.33, 0.68] n=5 | 0.44 [0.29, 0.53] n=5 | 0.28 | 5/5 | 0/5 | 0/5 |
| RT-examples | pair | 608 | 0.64 [0.59, 0.70] n=3 | 0.52 [0.50, 0.53] n=3 | 0.41 | 3/3 | 1/3 | 0/3 |
| RT-examples | triple+full | 1065 | 0.52 [0.47, 0.54] n=3 | 0.39 [0.31, 0.41] n=3 | 0.28 | 3/3 | 0/3 | 0/3 |
| RT-shuffled | pair | 608 | 0.85 [0.85, 0.85] n=2 | 0.54 [0.52, 0.55] n=2 | 0.41 | 2/2 | 2/2 | 0/2 |
| RT-shuffled | triple+full | 1065 | 0.70 [0.68, 0.72] n=2 | 0.42 [0.42, 0.42] n=2 | 0.28 | 2/2 | 1/2 | 0/2 |
| PF | pair | 608 | 0.31 [0.06, 0.59] n=3 | 0.31 [0.06, 0.59] n=3 | 0.41 | 1/3 | 0/3 | 0/3 |
| PF | triple+full | 1065 | 0.23 [0.16, 0.28] n=3 | 0.23 [0.16, 0.27] n=3 | 0.28 | 1/3 | 0/3 | 0/3 |
| RT-absolute | pair | 608 | 0.70 [0.18, 0.75] n=5 | 0.47 [0.07, 0.62] n=5 | 0.41 | 4/5 | 2/5 | 0/5 |
| RT-absolute | triple+full | 1065 | 0.60 [0.36, 0.66] n=5 | 0.43 [0.29, 0.60] n=5 | 0.28 | 5/5 | 0/5 | 0/5 |
(§6's rule as preregistered, printed and not scored — the scored rule is in the Verdicts section, DESIGN §17: P2 passes when RT-canon on the pairs meets both conditions in ≥ 6/10 seeds, exceeds greedy on triples+full in ≥ 6/10, and RT-all is ≥ 0.9 on the same strata.)

## P3 (Q2): sizes 6–7 (eval7, k ≤ 4) against size ≤ 5 on the same (world, k, binding) cells reweighted to eval7's composition, masked
| arm | worlds | size ≤ 5, k ≤ 4 | sizes 6–7 | matched ratio | eval7 n matched | ratio ≥ 0.9 | split by the categorical head's labelled slots (masked) |
|---|---|---|---|---|---|---|---|
| RT-canon | train | 0.99 [0.98, 0.99] n=10 | 0.51 [0.33, 0.56] n=10 | 0.52 [0.33, 0.58] n=10 | 4799/4800 | 0/10 | a shortest proof on labelled slots 0.56 [0.41, 0.61] n=10 (n 2793); none 0.43 [0.21, 0.51] n=10 (n 2007) |
| RT-canon | held-out | 0.79 [0.73, 0.87] n=10 | 0.22 [0.09, 0.32] n=10 | 0.30 [0.11, 0.40] n=10 | 4200/4200 | 0/10 |  |
| RT-absolute | train | 0.99 [0.78, 0.99] n=5 | 0.01 [0.01, 0.02] n=5 | 0.01 [0.01, 0.03] n=5 | 4799/4800 | 0/5 | a shortest proof on labelled slots 0.02 [0.02, 0.04] n=5 (n 2793); none 0.00 [0.00, 0.00] n=5 (n 2007) |
| RT-absolute | held-out | 0.73 [0.38, 0.81] n=5 | 0.00 [0.00, 0.01] n=5 | 0.01 [0.00, 0.01] n=5 | 4200/4200 | 0/5 |  |
| RT-post | train | 0.99 [0.99, 1.00] n=10 | 0.34 [0.31, 0.43] n=10 | 0.34 [0.31, 0.43] n=10 | 4799/4800 | 0/10 | a shortest proof on labelled slots 0.41 [0.37, 0.48] n=10 (n 2793); none 0.26 [0.21, 0.35] n=10 (n 2007) |
| RT-post | held-out | 0.86 [0.75, 0.89] n=10 | 0.15 [0.13, 0.21] n=10 | 0.18 [0.15, 0.25] n=10 | 4200/4200 | 0/10 |  |
| RT-all | train | 0.99 [0.98, 0.99] n=3 | 0.46 [0.40, 0.59] n=3 | 0.47 [0.40, 0.61] n=3 | 4799/4800 | 0/3 | a shortest proof on labelled slots 0.54 [0.50, 0.64] n=3 (n 2793); none 0.33 [0.26, 0.51] n=3 (n 2007) |
| RT-all | held-out | 0.97 [0.97, 0.98] n=3 | 0.24 [0.21, 0.25] n=3 | 0.25 [0.22, 0.26] n=3 | 4200/4200 | 0/3 |  |
(§6's reading, printed and not scored: the bare ratio ≥ 0.9 passes a router that lost a sixth of its success, DESIGN §17.)
P3 passes when the structural RT-canon's matched ratio on the training worlds is ≥ 0.9 in ≥ 7/10 seeds, and RT-absolute's failures are the unaddressable instances (success on addressable ≈ its size-5 success, ≈ 0 on unaddressable).

## P4 (Q3): training worlds at k ≥ 4 (eval5), on cells whose rungs are separated (d2 − d1 ≥ 0.1 and d3 − d2 ≥ 0.1); pass reading on the §16 cells (all strata, (world, binding, GP1, k), n ≥ 10), masked; the §6 cells (GP1 by (world, k)) beside it

The §16 cells recomputed from the run records equal the derive step's (6 cells).

Random-tie-break rungs on the §16 cells (derive step): d=1 0.38, d=2 0.47, d=3 0.72; the lower separation fails under it on 5 of 6 cells.
| arm | cells | n/seed (separated cells) | masked | instance bootstrap 95 % (seeds' envelope) | unmasked | d=1 | d=2 | d=3 | strictly between d=2 and d=3 | ≥ d=3 | ≤ d=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RT-canon | §16 cells | 170 | 0.94 [0.88, 0.96] n=10 | [0.84, 0.99] | 0.92 [0.86, 0.96] n=10 | 0.29 | 0.45 | 0.72 | 0/10 | 10/10 | 0/10 |
| RT-canon | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/10 | 0/10 |
| RT-canon | upper separation only (sensitivity) | 1245 | 0.91 [0.85, 0.94] n=10 | [0.83, 0.96] | 0.90 [0.84, 0.93] n=10 | 0.22 | 0.27 | 0.56 | 0/10 | 10/10 | 0/10 |
| RT-post | §16 cells | 170 | 0.98 [0.96, 0.99] n=10 | [0.93, 1.00] | 0.97 [0.96, 0.99] n=10 | 0.29 | 0.45 | 0.72 | 0/10 | 10/10 | 0/10 |
| RT-post | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/10 | 0/10 |
| RT-post | upper separation only (sensitivity) | 1245 | 0.96 [0.93, 0.97] n=10 | [0.92, 0.98] | 0.95 [0.93, 0.96] n=10 | 0.22 | 0.27 | 0.56 | 0/10 | 10/10 | 0/10 |
| RT-all | §16 cells | 170 | 0.95 [0.93, 0.96] n=3 | [0.89, 0.99] | 0.94 [0.92, 0.96] n=3 | 0.29 | 0.45 | 0.72 | 0/3 | 3/3 | 0/3 |
| RT-all | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/3 | 0/3 |
| RT-all | upper separation only (sensitivity) | 1245 | 0.90 [0.88, 0.93] n=3 | [0.86, 0.94] | 0.90 [0.88, 0.92] n=3 | 0.22 | 0.27 | 0.56 | 0/3 | 3/3 | 0/3 |
| RT-absolute | §16 cells | 170 | 0.94 [0.68, 0.96] n=5 | [0.60, 0.98] | 0.88 [0.67, 0.93] n=5 | 0.29 | 0.45 | 0.72 | 1/5 | 4/5 | 0/5 |
| RT-absolute | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/5 | 0/5 |
| RT-absolute | upper separation only (sensitivity) | 1245 | 0.93 [0.50, 0.94] n=5 | [0.47, 0.96] | 0.91 [0.49, 0.92] n=5 | 0.22 | 0.27 | 0.56 | 1/5 | 4/5 | 0/5 |
| EI16 | §16 cells | 170 | 0.94 [0.85, 0.95] n=5 | [0.79, 0.98] | 0.90 [0.85, 0.95] n=5 | 0.29 | 0.45 | 0.72 | 0/5 | 5/5 | 0/5 |
| EI16 | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/5 | 0/5 |
| EI16 | upper separation only (sensitivity) | 1245 | 0.88 [0.84, 0.89] n=5 | [0.81, 0.90] | 0.88 [0.81, 0.88] n=5 | 0.22 | 0.27 | 0.56 | 0/5 | 5/5 | 0/5 |
| PF | §16 cells | 170 | 1.00 [0.99, 1.00] n=3 | [0.98, 1.00] | 1.00 [0.99, 1.00] n=3 | 0.29 | 0.45 | 0.72 | 0/3 | 3/3 | 0/3 |
| PF | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/3 | 0/3 |
| PF | upper separation only (sensitivity) | 1245 | 0.99 [0.96, 0.99] n=3 | [0.95, 1.00] | 0.99 [0.96, 0.99] n=3 | 0.22 | 0.27 | 0.56 | 0/3 | 3/3 | 0/3 |
| RT-examples | §16 cells | 170 | 0.92 [0.91, 0.92] n=3 | [0.86, 0.96] | 0.89 [0.88, 0.91] n=3 | 0.29 | 0.45 | 0.72 | 0/3 | 3/3 | 0/3 |
| RT-examples | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/3 | 0/3 |
| RT-examples | upper separation only (sensitivity) | 1245 | 0.88 [0.85, 0.92] n=3 | [0.83, 0.93] | 0.87 [0.85, 0.90] n=3 | 0.22 | 0.27 | 0.56 | 0/3 | 3/3 | 0/3 |
| RT-shuffled | §16 cells | 170 | 0.91 [0.90, 0.92] n=2 | [0.85, 0.96] | 0.55 [0.55, 0.56] n=2 | 0.29 | 0.45 | 0.72 | 0/2 | 2/2 | 0/2 |
| RT-shuffled | §6 cells (GP1) | 0 | – | — | – | – | – | – | rungs not separated | 0/2 | 0/2 |
| RT-shuffled | upper separation only (sensitivity) | 1245 | 0.86 [0.86, 0.87] n=2 | [0.84, 0.89] | 0.62 [0.62, 0.63] n=2 | 0.22 | 0.27 | 0.56 | 0/2 | 2/2 | 0/2 |
P4 passes when RT-canon lies strictly between the d = 2 and d = 3 rungs in ≥ 7/10 seeds on the §16 cells; ≥ d = 3 in ≥ 7/10 falsifies toward 'three steps absorbed', ≤ d = 1 in ≥ 7/10 toward 'none'; if no count reaches 7/10 (the router within ≈ 0.03 of a rung), P4 is **undecided** and reported so. The seed count covers training variance only (the instances are fixed); the instance bootstrap column is the envelope over seeds of each seed's 95 % interval. The §6 row is the preregistered GP1-only reading and the upper-only row a sensitivity set; both reported, not scored.

RT-canon by k × binding × GP1 (training worlds pooled, eval5): reach = the largest d whose rung RT-canon matches

| k | binding | GP1 | n/seed | masked | unmasked | d=1 | d=2 | d=3 | reach over seeds | separated |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 655 | 1.00 [0.99, 1.00] n=10 | 1.00 [0.99, 1.00] n=10 | 0.98 | 1.00 | 1.00 | median 1.0 (min 1, max 2) | no |
| 2 | 0 | 1 | 10 | 1.00 [1.00, 1.00] n=10 | 1.00 [0.90, 1.00] n=10 | 0.00 | 1.00 | 1.00 | median 2.0 (min 2, max 2) | no |
| 2 | 1 | 0 | 80 | 0.99 [0.97, 1.00] n=10 | 0.99 [0.96, 1.00] n=10 | 1.00 | 1.00 | 1.00 | median 0.5 (min 0, max 1) | no |
| 2 | 1 | 1 | 4 | 1.00 [1.00, 1.00] n=10 | 1.00 [1.00, 1.00] n=10 | 0.00 | 1.00 | 1.00 | median 2.0 (min 2, max 2) | no |
| 3 | 0 | 0 | 670 | 0.99 [0.98, 1.00] n=10 | 0.99 [0.98, 1.00] n=10 | 0.82 | 0.96 | 1.00 | median 2.0 (min 2, max 2) | no |
| 3 | 0 | 1 | 156 | 0.97 [0.92, 0.99] n=10 | 0.96 [0.90, 0.98] n=10 | 0.11 | 0.45 | 1.00 | median 2.0 (min 2, max 2) | yes |
| 3 | 1 | 0 | 320 | 0.99 [0.97, 1.00] n=10 | 0.99 [0.95, 1.00] n=10 | 0.81 | 0.93 | 1.00 | median 2.0 (min 2, max 3) | no |
| 3 | 1 | 1 | 35 | 0.97 [0.94, 1.00] n=10 | 0.96 [0.94, 1.00] n=10 | 0.03 | 0.20 | 1.00 | median 2.0 (min 2, max 3) | yes |
| 4 | 0 | 0 | 297 | 0.98 [0.96, 0.99] n=10 | 0.97 [0.96, 0.99] n=10 | 0.77 | 0.82 | 0.99 | median 2.0 (min 2, max 3) | no |
| 4 | 0 | 1 | 154 | 0.94 [0.86, 0.98] n=10 | 0.92 [0.85, 0.97] n=10 | 0.06 | 0.14 | 0.39 | median 3.0 (min 3, max 3) | no |
| 4 | 1 | 0 | 344 | 0.97 [0.92, 0.99] n=10 | 0.96 [0.91, 0.98] n=10 | 0.57 | 0.62 | 0.88 | median 3.0 (min 3, max 3) | no |
| 4 | 1 | 1 | 101 | 0.94 [0.87, 0.98] n=10 | 0.93 [0.87, 0.98] n=10 | 0.04 | 0.11 | 0.28 | median 3.0 (min 3, max 3) | no |
| 5 | 0 | 0 | 33 | 0.88 [0.76, 0.94] n=10 | 0.88 [0.67, 0.94] n=10 | 0.15 | 0.24 | 0.61 | median 3.0 (min 3, max 3) | no |
| 5 | 0 | 1 | 76 | 0.86 [0.76, 0.95] n=10 | 0.84 [0.75, 0.95] n=10 | 0.00 | 0.03 | 0.32 | median 3.0 (min 3, max 3) | no |
| 5 | 1 | 0 | 209 | 0.92 [0.87, 0.96] n=10 | 0.92 [0.86, 0.96] n=10 | 0.12 | 0.19 | 0.47 | median 3.0 (min 3, max 3) | no |
| 5 | 1 | 1 | 84 | 0.88 [0.79, 0.96] n=10 | 0.88 [0.79, 0.93] n=10 | 0.01 | 0.04 | 0.18 | median 3.0 (min 3, max 3) | no |
| 6 | 0 | 0 | 11 | 0.91 [0.64, 1.00] n=10 | 0.91 [0.64, 1.00] n=10 | 0.09 | 0.18 | 0.73 | median 3.0 (min 2, max 3) | no |
| 6 | 0 | 1 | 37 | 0.73 [0.59, 0.86] n=10 | 0.73 [0.54, 0.86] n=10 | 0.00 | 0.00 | 0.19 | median 3.0 (min 3, max 3) | no |
| 6 | 1 | 0 | 156 | 0.86 [0.79, 0.90] n=10 | 0.84 [0.77, 0.88] n=10 | 0.02 | 0.04 | 0.37 | median 3.0 (min 3, max 3) | no |
| 6 | 1 | 1 | 75 | 0.77 [0.60, 0.88] n=10 | 0.76 [0.59, 0.87] n=10 | 0.00 | 0.00 | 0.35 | median 3.0 (min 3, max 3) | no |

## P5 (Q5): held-out binding stratum — RT-canon median 0.63, EI16 0.68 [0.42, 0.77] n=5; EI16 bootstrap 95 % CIs per seed: [('0.69', '0.73'), ('0.56', '0.61'), ('0.40', '0.44'), ('0.66', '0.70'), ('0.75', '0.79')]; EI16 [routes found, instances tried, rows imitated]: [[378296, 384000, 1581552], [374706, 384000, 1571447], [377990, 384000, 1574547], [377808, 384000, 1581202], [377839, 384000, 1574626]]

## P6 (Q6): multi-class training instances — the router's mass on shortest-path first actions, its total-variation distance to the RT-post target (0 = the target mixture), and the share of first decisions equal to the lexicographic (canonical) action

| run | n | mass on shortest-path actions | TV to RT-post target | canonical first action |
|---|---|---|---|---|
| RT-canon_s0 | 2213 | 0.96 | 0.60 | 0.93 |
| RT-canon_s1 | 2213 | 0.95 | 0.60 | 0.93 |
| RT-canon_s2 | 2213 | 0.96 | 0.60 | 0.93 |
| RT-canon_s3 | 2213 | 0.96 | 0.60 | 0.92 |
| RT-canon_s4 | 2213 | 0.96 | 0.59 | 0.92 |
| RT-canon_s5 | 2213 | 0.96 | 0.60 | 0.93 |
| RT-canon_s6 | 2213 | 0.96 | 0.60 | 0.92 |
| RT-canon_s7 | 2213 | 0.96 | 0.59 | 0.92 |
| RT-canon_s8 | 2213 | 0.94 | 0.59 | 0.90 |
| RT-canon_s9 | 2213 | 0.96 | 0.60 | 0.92 |
| RT-post_s0 | 2213 | 0.96 | 0.12 | 0.40 |
| RT-post_s1 | 2213 | 0.96 | 0.12 | 0.41 |
| RT-post_s2 | 2213 | 0.96 | 0.13 | 0.35 |
| RT-post_s3 | 2213 | 0.96 | 0.12 | 0.36 |
| RT-post_s4 | 2213 | 0.96 | 0.13 | 0.32 |
| RT-post_s5 | 2213 | 0.96 | 0.12 | 0.35 |
| RT-post_s6 | 2213 | 0.96 | 0.14 | 0.43 |
| RT-post_s7 | 2213 | 0.97 | 0.12 | 0.35 |
| RT-post_s8 | 2213 | 0.97 | 0.12 | 0.33 |
| RT-post_s9 | 2213 | 0.97 | 0.12 | 0.33 |
| PF_s0 | 2213 | 0.82 | 0.11 | 0.24 |
| PF_s1 | 2213 | 0.83 | 0.10 | 0.30 |
| PF_s2 | 2213 | 0.83 | 0.11 | 0.33 |
| RT-all_s0 | 2213 | 0.95 | 0.59 | 0.91 |
| RT-all_s1 | 2213 | 0.96 | 0.59 | 0.90 |
| RT-all_s2 | 2213 | 0.97 | 0.60 | 0.93 |

## P7 (Q7): RT-examples on the binding stratum against its derived ceiling (weighted by binding count) and stated-axiom RT-canon

| run | class | n | RT-examples masked | unmasked | derived ceiling | RT-canon |
|---|---|---|---|---|---|---|
| RT-examples_s0 | train | 1408 | 0.90 | 0.89 | 0.95 | 0.94 [0.90, 0.96] n=10 |
| RT-examples_s0 | pair | 608 | 0.59 | 0.53 | 0.91 | 0.77 [0.62, 0.87] n=10 |
| RT-examples_s0 | triple+full | 1065 | 0.47 | 0.39 | 0.84 | 0.56 [0.48, 0.73] n=10 |
| RT-examples_s1 | train | 1408 | 0.91 | 0.90 | 0.95 | 0.94 [0.90, 0.96] n=10 |
| RT-examples_s1 | pair | 608 | 0.70 | 0.50 | 0.91 | 0.77 [0.62, 0.87] n=10 |
| RT-examples_s1 | triple+full | 1065 | 0.52 | 0.31 | 0.84 | 0.56 [0.48, 0.73] n=10 |
| RT-examples_s2 | train | 1408 | 0.94 | 0.92 | 0.95 | 0.94 [0.90, 0.96] n=10 |
| RT-examples_s2 | pair | 608 | 0.64 | 0.52 | 0.91 | 0.77 [0.62, 0.87] n=10 |
| RT-examples_s2 | triple+full | 1065 | 0.54 | 0.41 | 0.84 | 0.56 [0.48, 0.73] n=10 |

## P8 (null): RT-shuffled per world against the cue-blind ceiling and against ceiling × RT-canon's non-binding success; binding stratum against RT-canon

| world | seed | n | shuffled (all) | cue-blind ceiling | ceiling × RT-canon non-binding | binding n | shuffled (binding) | RT-canon (binding) |
|---|---|---|---|---|---|---|---|---|
| A | 0 | 600 | 0.99 | 0.89 | 0.89 | 8 | 1.00 | 1.00 [0.88, 1.00] n=10 |
| A | 1 | 600 | 1.00 | 0.89 | 0.89 | 8 | 1.00 | 1.00 [0.88, 1.00] n=10 |
| C | 0 | 600 | 1.00 | 0.89 | 0.89 | 57 | 1.00 | 1.00 [0.95, 1.00] n=10 |
| C | 1 | 600 | 0.99 | 0.89 | 0.89 | 57 | 0.98 | 1.00 [0.95, 1.00] n=10 |
| I | 0 | 600 | 0.99 | 0.68 | 0.68 | 125 | 0.99 | 0.99 [0.98, 1.00] n=10 |
| I | 1 | 600 | 1.00 | 0.68 | 0.68 | 125 | 1.00 | 0.99 [0.98, 1.00] n=10 |
| D | 0 | 600 | 1.00 | 0.77 | 0.77 | 7 | 1.00 | 1.00 [1.00, 1.00] n=10 |
| D | 1 | 600 | 1.00 | 0.77 | 0.77 | 7 | 1.00 | 1.00 [1.00, 1.00] n=10 |
| AC | 0 | 600 | 0.81 | 0.39 | 0.36 | 307 | 0.78 | 0.86 [0.75, 0.91] n=10 |
| AC | 1 | 600 | 0.81 | 0.39 | 0.36 | 307 | 0.77 | 0.86 [0.75, 0.91] n=10 |
| AD | 0 | 600 | 0.89 | 0.40 | 0.39 | 294 | 0.84 | 0.90 [0.81, 0.94] n=10 |
| AD | 1 | 600 | 0.88 | 0.40 | 0.39 | 294 | 0.83 | 0.90 [0.81, 0.94] n=10 |
| CD | 0 | 600 | 1.00 | 0.39 | 0.39 | 289 | 1.00 | 0.99 [0.99, 1.00] n=10 |
| CD | 1 | 600 | 0.99 | 0.39 | 0.39 | 289 | 0.99 | 0.99 [0.99, 1.00] n=10 |
| ID | 0 | 600 | 0.96 | 0.39 | 0.38 | 321 | 0.95 | 0.98 [0.92, 0.99] n=10 |
| ID | 1 | 600 | 0.97 | 0.39 | 0.38 | 321 | 0.97 | 0.98 [0.92, 0.99] n=10 |
| AI | 0 | 600 | 0.94 | 0.42 | 0.38 | 305 | 0.95 | 0.90 [0.74, 0.94] n=10 |
| AI | 1 | 600 | 0.93 | 0.42 | 0.38 | 305 | 0.93 | 0.90 [0.74, 0.94] n=10 |
| CI | 0 | 600 | 0.78 | 0.40 | 0.31 | 303 | 0.76 | 0.64 [0.37, 0.79] n=10 |
| CI | 1 | 600 | 0.79 | 0.40 | 0.31 | 303 | 0.77 | 0.64 [0.37, 0.79] n=10 |
| ACI | 0 | 600 | 0.82 | 0.66 | 0.46 | 156 | 0.94 | 0.92 [0.54, 0.95] n=10 |
| ACI | 1 | 600 | 0.81 | 0.66 | 0.46 | 156 | 0.94 | 0.92 [0.54, 0.95] n=10 |
| ACD | 0 | 600 | 0.60 | 0.41 | 0.26 | 299 | 0.50 | 0.44 [0.30, 0.55] n=10 |
| ACD | 1 | 600 | 0.59 | 0.41 | 0.26 | 299 | 0.47 | 0.44 [0.30, 0.55] n=10 |
| AID | 0 | 600 | 0.83 | 0.48 | 0.37 | 294 | 0.77 | 0.53 [0.32, 0.76] n=10 |
| AID | 1 | 600 | 0.85 | 0.48 | 0.37 | 294 | 0.82 | 0.53 [0.32, 0.76] n=10 |
| CID | 0 | 600 | 0.77 | 0.44 | 0.37 | 316 | 0.66 | 0.64 [0.43, 0.82] n=10 |
| CID | 1 | 600 | 0.83 | 0.44 | 0.37 | 316 | 0.76 | 0.64 [0.43, 0.82] n=10 |
| ACID | 0 | 600 | 0.76 | 1.00 | 0.60 | 0 | – | – |
| ACID | 1 | 600 | 0.78 | 1.00 | 0.60 | 0 | – | – |

## Failure categories (eval5, pooled over seeds): masked | unmasked

| arm | class | n | masked | unmasked |
|---|---|---|---|---|
| RT-canon | train | 48000 | ok:0.97 horizon:0.03 wrong halt:0.00 no halt at goal:0.00 | ok:0.96 horizon:0.03 invalid: no match:0.01 invalid: out of world:0.00 wrong halt:0.00 no halt at goal:0.00 |
| RT-canon | held-out | 42000 | ok:0.69 horizon:0.31 wrong halt:0.00 no halt at goal:0.00 | ok:0.62 horizon:0.23 invalid: no match:0.10 invalid: out of world:0.04 wrong halt:0.00 no halt at goal:0.00 |
| RT-post | train | 48000 | ok:0.99 horizon:0.01 wrong halt:0.00 no halt at goal:0.00 | ok:0.98 horizon:0.01 invalid: no match:0.00 invalid: out of world:0.00 wrong halt:0.00 no halt at goal:0.00 |
| RT-post | held-out | 42000 | ok:0.77 horizon:0.23 wrong halt:0.00 no halt at goal:0.00 | ok:0.73 horizon:0.19 invalid: no match:0.06 invalid: out of world:0.01 wrong halt:0.00 no halt at goal:0.00 |
| RT-all | train | 14400 | ok:0.97 horizon:0.03 wrong halt:0.00 | ok:0.97 horizon:0.03 invalid: no match:0.00 invalid: out of world:0.00 wrong halt:0.00 |
| RT-all | held-out | 12600 | ok:0.94 horizon:0.06 wrong halt:0.00 | ok:0.93 horizon:0.06 invalid: no match:0.01 invalid: out of world:0.00 wrong halt:0.00 |
| RT-absolute | train | 24000 | ok:0.92 wrong halt:0.05 horizon:0.03 no halt at goal:0.00 | ok:0.91 wrong halt:0.05 horizon:0.02 invalid: no match:0.02 no halt at goal:0.00 invalid: out of world:0.00 |
| RT-absolute | held-out | 21000 | ok:0.59 horizon:0.32 wrong halt:0.08 no halt at goal:0.00 | ok:0.50 horizon:0.20 invalid: no match:0.15 invalid: out of world:0.09 wrong halt:0.06 no halt at goal:0.00 |
| EI16 | train | 24000 | ok:0.96 horizon:0.04 wrong halt:0.00 no halt at goal:0.00 | ok:0.95 horizon:0.04 invalid: no match:0.01 wrong halt:0.00 invalid: out of world:0.00 no halt at goal:0.00 |
| EI16 | held-out | 21000 | ok:0.67 horizon:0.33 wrong halt:0.00 | ok:0.57 horizon:0.23 invalid: no match:0.11 invalid: out of world:0.08 wrong halt:0.00 |
| PF | train | 14400 | ok:0.99 horizon:0.01 | ok:0.99 horizon:0.01 invalid: no match:0.00 |
| PF | held-out | 12600 | horizon:0.63 ok:0.37 wrong halt:0.00 | horizon:0.60 ok:0.37 invalid: out of world:0.02 invalid: no match:0.01 wrong halt:0.00 |
| RT-examples | train | 14400 | ok:0.96 horizon:0.04 wrong halt:0.00 | ok:0.95 horizon:0.04 invalid: no match:0.01 invalid: out of world:0.00 wrong halt:0.00 |
| RT-examples | held-out | 12600 | ok:0.62 horizon:0.38 wrong halt:0.00 | ok:0.53 horizon:0.20 invalid: no match:0.17 invalid: out of world:0.10 wrong halt:0.00 |
| RT-shuffled | train | 9600 | ok:0.96 horizon:0.04 wrong halt:0.00 | ok:0.78 invalid: out of world:0.19 horizon:0.03 invalid: no match:0.00 wrong halt:0.00 |
| RT-shuffled | held-out | 8400 | ok:0.79 horizon:0.21 wrong halt:0.00 | ok:0.60 invalid: out of world:0.26 horizon:0.09 invalid: no match:0.04 wrong halt:0.00 |

## P1 (reported): {A, C} at k = 7–8, size ≤ 5 (the symbolic rungs are at their floor here — there is no reference; a rate is reported, not compared)

| arm | n/seed | masked | unmasked | d=1 | d=2 | d=3 |
|---|---|---|---|---|---|---|
| RT-canon | 600 | 0.53 [0.32, 0.62] n=10 | 0.52 [0.32, 0.62] n=10 | 0.000 | 0.002 | 0.020 |
| RT-post | 600 | 0.72 [0.67, 0.80] n=10 | 0.71 [0.65, 0.79] n=10 | 0.000 | 0.002 | 0.020 |
| RT-all | 600 | 0.57 [0.36, 0.70] n=3 | 0.55 [0.36, 0.70] n=3 | 0.000 | 0.002 | 0.020 |
| RT-absolute | 600 | 0.61 [0.01, 0.66] n=5 | 0.59 [0.01, 0.63] n=5 | 0.000 | 0.002 | 0.020 |
| EI16 | 600 | 0.76 [0.64, 0.81] n=5 | 0.76 [0.64, 0.81] n=5 | 0.000 | 0.002 | 0.020 |
| PF | 600 | 0.91 [0.73, 0.95] n=3 | 0.91 [0.72, 0.95] n=3 | 0.000 | 0.002 | 0.020 |
| RT-examples | 600 | 0.38 [0.28, 0.59] n=3 | 0.37 [0.27, 0.56] n=3 | 0.000 | 0.002 | 0.020 |
| RT-shuffled | 600 | 0.41 [0.38, 0.45] n=2 | 0.30 [0.28, 0.31] n=2 | 0.000 | 0.002 | 0.020 |

## Verdicts under the amended pass rules (DESIGN §17)

| prediction | verdict | reading |
|---|---|---|
| P2 | not met, not falsified | above the non-composing floor in more than 3/10 seeds, short of the pair conditions in 6/10 (or of the triples' clause) |
| P3 | structural: falsified; absolute: falsified |  |
| P5 | holds |  |
| P6 | does not hold |  |
| P7 | holds |  |
| P8 | falsified; unmasked: holds | under the mask the action set, not the presence tokens, carries the world; without the mask the tokens do |

**P2** — held-out binding instances that need the unseen combination. Pairs n = 209: greedy 0.40, lookahead-3 0.73, training-world reference 0.53 (floor 0.53), glued reference 0.75; triples n = 509: greedy 0.15, lookahead-3 0.51, training-world reference 0.21 (floor 0.21), glued reference 0.50. A masked pass says the decisions learned in the training worlds, applied state by state under the executor's mask, prove these instances; it does not separate planning over the unseen combination from single-world plans glued by greedy moves (the glued reference). The rows that bear on that — the unmasked reading, and the instances the glued reference does not solve — are below, unscored.

| reading | RT-canon, pairs | instance bootstrap 95 % (seeds' envelope) | ≥ greedy + 0.2 and ≥ 0.7 | > floor (pairs) | RT-canon, triples | > floor (triples) | RT-all, pairs | RT-all's median meets the pair conditions | pass conditions met |
|---|---|---|---|---|---|---|---|---|---|
| masked | 0.47 [0.28, 0.67] n=10 | [0.23, 0.73] | 0/10 | 4/10 | 0.38 [0.27, 0.56] n=10 | 10/10 | 0.94 [0.93, 0.95] n=3 | yes | no |
| unmasked | 0.37 [0.16, 0.56] n=10 | [0.11, 0.64] | 0/10 | 1/10 | 0.31 [0.17, 0.51] n=10 | 9/10 | 0.93 [0.93, 0.95] n=3 | yes | no |

RT-all's seed-median, masked: pairs 0.94, triples 0.85 — §6's ceiling clause (≥ 0.9 on both) not met. Beyond the glued reference (unscored): pairs n = 52 (lookahead-3 0.56) — RT-canon masked 0.45 [0.25, 0.60] n=10, unmasked 0.37 [0.12, 0.48] n=10; triples n = 256 (lookahead-3 0.36) — masked 0.30 [0.22, 0.46] n=10, unmasked 0.24 [0.11, 0.43] n=10.

| world | n needing the combination | RT-canon masked | unmasked | greedy | lookahead-3 |
|---|---|---|---|---|---|
| AI | 47 | 0.64 [0.19, 0.77] n=10 | 0.50 [0.09, 0.64] n=10 | 0.53 | 0.74 |
| CI | 162 | 0.48 [0.14, 0.66] n=10 | 0.32 [0.05, 0.55] n=10 | 0.36 | 0.72 |
| ACI | 5 | 0.30 [0.00, 1.00] n=10 | 0.10 [0.00, 1.00] n=10 | 0.00 | 0.40 |
| ACD | 218 | 0.37 [0.27, 0.50] n=10 | 0.34 [0.22, 0.45] n=10 | 0.03 | 0.36 |
| AID | 91 | 0.26 [0.03, 0.48] n=10 | 0.24 [0.01, 0.43] n=10 | 0.10 | 0.37 |
| CID | 195 | 0.49 [0.18, 0.77] n=10 | 0.39 [0.02, 0.69] n=10 | 0.32 | 0.73 |

**P3, clause 1** (RT-canon, masked, training worlds): matched ratio R against 0.9 × ρ, the size-blind ratio at the router's own size-≤ 5 success (interpolated between the rungs, at most 1 / success) — per seed R 0.44 0.56 0.56 0.55 0.33 0.50 0.51 0.50 0.58 0.53; size-≤ 5 success 0.97 0.97 0.99 0.96 0.98 0.98 0.99 0.98 0.96 0.98; ρ 1.04 1.03 1.01 1.04 1.02 1.02 1.01 1.02 1.04 1.02; R ≥ 0.9 ρ in 0/10 seeds (the unnormalised R ≥ 0.9 of §6: 0/10) → falsified. Single-axiom worlds R 0.54 0.64 0.66 0.66 0.42 0.62 0.60 0.61 0.68 0.63; training pairs R 0.33 0.47 0.46 0.44 0.25 0.39 0.41 0.38 0.48 0.43 against ρ 1.07 1.06 1.03 1.07 1.04 1.03 1.03 1.03 1.08 1.03.

**P3, clause 2** (RT-absolute, unmasked, training worlds, sizes 6–7; groups by the slots its head was ever trained on): must fail (n 986) success 0.00 0.00 0.00 0.00 0.00 (≤ 0.05 in 5/5); only a detour on labelled slots (n 1021, reported apart) 0.00 0.00 0.00 0.00 0.00; a shortest proof on labelled slots (n 2793) success 0.00 0.00 0.00 0.00 0.01, matched ratio R 0.00 0.00 0.00 0.00 0.01 against ρ 1.11 1.03 1.03 1.04 1.02; R ≥ 0.9 ρ in 0/5 → falsified.

**P5** (held-out binding stratum, n 1673, masked): RT-canon's seed-median 0.63, EI16's 0.68 (difference +0.05), greedy rung 0.33; unmasked 0.55 / 0.49; on the 718 that need the unseen combination 0.42 / 0.50 → holds.

**P6** (first decision, multi-class training instances, n 2213): RT-post's TV to the target 0.12 0.12 0.13 0.12 0.13 0.12 0.14 0.12 0.12 0.12 against the uniform-over-shortest-actions reference 0.10 (below it in 0/10); RT-canon's canonical-first share 0.93 0.93 0.93 0.92 0.92 0.93 0.92 0.92 0.90 0.92 (≥ 0.9 in 9/10); RT-canon's TV 0.60 0.60 0.60 0.60 0.59 0.60 0.60 0.59 0.59 0.60 → does not hold.

**P7** (unmasked): training binding stratum (n 1408) — RT-examples 0.89 0.90 0.92 against ceiling 0.95 × RT-canon's median 0.93 = 0.89 (± 0.1) → holds; held-out binding stratum (n 1673) — RT-examples 0.44 0.38 0.45 against RT-canon's median 0.55 (below by more than 0.05 in every seed) → holds. Clause 1 is a consistency band that can hardly fail; what the examples add over no cue at all is the unscored comparison with RT-shuffled on the same stratum, unmasked: 0.58 0.58.

**P8** (training worlds' binding stratum, n 1408): masked — RT-canon's median 0.94, RT-shuffled 0.91 0.90 (gaps 0.03 0.04) → falsified; unmasked — 0.93, 0.58 0.58 (gaps 0.36 0.35) → holds. A gap of more than 0.05 in every shuffled seed holds; none falsifies.
