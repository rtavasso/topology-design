# Study 4 automatic report

80 Stage-1 runs. Selection: pk0000 lr 0.016 (stable, grid edge), pk0100 lr 0.004 (stable), pk0110 lr 0.002 (stable), pk0111 lr 0.004 (stable), pk0120 lr 0.016 (unstable-at-best, grid edge), pk0121 lr 0.002 (unstable-at-best), pk0122 lr 0.001 (unstable-at-best), pk0123 lr 0.002 (unstable-at-best)

## Verdicts (Holm over H1–H3)

- **H1:** SUPPORTED (Holm p = 0.0003)
- **H2:** NOT SUPPORTED (Holm p = 1.0000)
- **H3:** SUPPORTED (Holm p = 0.0003)

## H1 (direct attribution): short paths' share of gen minus their share of mem (> 0)
pooled D = 0.117 [0.101, 0.134], seeds = 10, one-sided p = 0.0001; excluded {}; not grokked {'pk0100': 0, 'pk0110': 0, 'pk0111': 0, 'pk0120': 0, 'pk0121': 0, 'pk0122': 6, 'pk0123': 5}

| graph | D | 95% CI | n |
|---|---|---|---|
| pk0100 | 0.024 | [-0.007, 0.065] | 10 |
| pk0110 | 0.073 | [0.053, 0.091] | 10 |
| pk0111 | -0.014 | [-0.015, -0.013] | 10 |
| pk0120 | 0.596 | [0.535, 0.655] | 10 |
| pk0121 | -0.002 | [-0.002, -0.001] | 10 |
| pk0122 | 0.004 | [0.002, 0.005] | 4 |
| pk0123 | -0.000 | [-0.001, 0.000] | 5 |

## H1 (total attribution): short paths' share of gen minus their share of mem (> 0)
pooled D = 0.119 [0.108, 0.130], seeds = 10, one-sided p = 0.0001; excluded {}; not grokked {'pk0100': 0, 'pk0110': 0, 'pk0111': 0, 'pk0120': 0, 'pk0121': 0, 'pk0122': 6, 'pk0123': 5}

| graph | D | 95% CI | n |
|---|---|---|---|
| pk0100 | 0.036 | [0.021, 0.057] | 10 |
| pk0110 | 0.065 | [0.050, 0.079] | 10 |
| pk0111 | 0.027 | [0.025, 0.028] | 10 |
| pk0120 | 0.477 | [0.437, 0.515] | 10 |
| pk0121 | 0.045 | [0.035, 0.054] | 10 |
| pk0122 | 0.059 | [0.056, 0.061] | 4 |
| pk0123 | 0.046 | [0.034, 0.058] | 5 |

## H2: an extra short reader lowers the utility of existing long blocks (A − B < 0 in both)
intersection-union p = 1.0000

| contrast | blocks | U(A) | U(B) | mean A−B | 95% CI | pairs | not both grokked | undefined |
|---|---|---|---|---|---|---|---|---|
| H2a: pk0120 vs pk0121 | [2, 3] | 0.184 | 0.846 | -0.662 | [-0.737, -0.585] | 10 | 0 | 0 |
| H2b: pk0100 vs pk0110 | [2] | 0.948 | 0.458 | 0.490 | [0.452, 0.524] | 10 | 0 | 0 |

## H3: composition delays generalization (τ(C, delay) > 0)
τ_b = 0.447 [seed bootstrap 0.387, 0.514], permutation p = 0.0001, n = 80, censored = 11
median delay: pk0000 430, pk0100 2870, pk0110 7550, pk0111 4170, pk0120 398, pk0121 9700, pk0122 cens., pk0123 500006170
lr·wd-normalized delay (exploratory): τ_b = 0.388, p = 0.0001

## Without flagged graphs ['pk0120', 'pk0121', 'pk0122', 'pk0123']
- H1 direct D = 0.028 [0.015, 0.042]; total D = 0.042 [0.035, 0.050]
- H2a: mean A−B -0.662 [-0.740, -0.586] (involves a flagged graph)
- H2b: mean A−B 0.490 [0.452, 0.525]
- H3 τ = 0.627, p = 0.0001

## Shared learning rate (Stage 0, 3 seeds, exploratory)

| lr | H3 τ | H3 p | H2a A−B | H2b A−B |
|---|---|---|---|---|
| 0.00025 | 0.551 | 0.001 | — | — |
| 0.0005 | 0.551 | 0.002 | — | — |
| 0.001 | 0.511 | 0.002 | — | 0.240 |
| 0.002 | 0.022 | 0.447 | -0.013 | 0.198 |
| 0.004 | 0.154 | 0.190 | -0.039 | 0.584 |
| 0.008 | 0.178 | 0.146 | -0.047 | 0.457 |
| 0.016 | 0.302 | 0.038 | -0.159 | 0.022 |

## Exploratory (no claims)

### Q-C descriptors

| graph | C | Q | lr | grok | stable | median delay | lr·wd·delay | peak mem_total | persistence | final rule frac |
|---|---|---|---|---|---|---|---|---|---|---|
| pk0000 | 1 | 4 | 0.016 | 1.00 | 0.90 | 430 | 6.88 | 20.78 | 420 | 0.965 |
| pk0100 | 2 | 3 | 0.004 | 1.00 | 0.70 | 2870 | 11.48 | 35.20 | 1800 | 0.978 |
| pk0110 | 2 | 2 | 0.002 | 1.00 | 0.60 | 7550 | 15.10 | 38.83 | 5500 | 0.973 |
| pk0111 | 2 | 1 | 0.004 | 1.00 | 0.60 | 4170 | 16.68 | 34.83 | 3100 | 0.978 |
| pk0120 | 3 | 2 | 0.016 | 1.00 | 0.00 | 398 | 6.36 | 19.50 | 392 | 0.883 |
| pk0121 | 3 | 1 | 0.002 | 1.00 | 0.00 | 9700 | 19.40 | 45.20 | 8875 | 0.976 |
| pk0122 | 3 | 1 | 0.001 | 0.40 | 0.00 | cens. | cens. | 52.49 | 25200 | 0.965 |
| pk0123 | 4 | 1 | 0.002 | 0.50 | 0.00 | 500006170 | 500000012.34 | 37.77 | 5100 | 0.972 |

### Role map (mean over seeds; direct: emb, b1..b4; total-effect: b1..b4)

- pk0000: gen@end [-0.002, 4.855, 5.037, 4.7, 4.772] | tshap [4.855, 5.037, 4.7, 4.772]; mem@t_mem [0.846, 4.987, 4.952, 4.993, 4.94] | tshap [4.987, 4.952, 4.993, 4.94]
- pk0100: gen@end [-0.0, 0.188, 35.076, 0.806, 0.828] | tshap [17.731, 17.532, 0.806, 0.828]; mem@t_mem [0.107, 0.843, 33.973, 0.002, 0.002] | tshap [16.993, 17.823, 0.002, 0.002]
- pk0110: gen@end [-0.001, 0.39, 17.581, 17.019, 3.26] | tshap [17.705, 8.784, 8.501, 3.26]; mem@t_mem [0.119, 0.553, 19.544, 19.529, 0.247] | tshap [19.007, 10.313, 10.305, 0.247]
- pk0111: gen@end [-0.0, 0.069, 12.622, 13.347, 12.653] | tshap [19.396, 6.305, 6.668, 6.321]; mem@t_mem [0.098, 0.555, 11.47, 11.373, 11.51] | tshap [16.574, 6.124, 6.071, 6.139]
- pk0120: gen@end [-0.01, 0.06, 0.238, 3.348, 10.387] | tshap [1.095, 1.239, 1.312, 10.387]; mem@t_mem [0.439, 0.402, 1.562, 13.576, 3.375] | tshap [3.9, 5.285, 6.354, 3.375]
- pk0121: gen@end [-0.0, 0.03, 0.914, 34.75, 5.593] | tshap [14.676, 11.918, 11.87, 2.822]; mem@t_mem [0.068, 0.124, 2.05, 41.818, 1.003] | tshap [14.121, 14.995, 15.284, 0.593]
- pk0122: gen@end [-0.002, 0.141, 1.31, 13.226, 13.233] | tshap [10.095, 8.595, 4.609, 4.612]; mem@t_mem [0.054, 0.099, 1.374, 25.052, 25.524] | tshap [16.104, 17.376, 9.197, 9.373]
- pk0123: gen@end [-0.001, 0.028, 0.143, 1.53, 32.482] | tshap [9.153, 7.718, 8.348, 8.966]; mem@t_mem [0.058, 0.054, 0.269, 2.316, 34.581] | tshap [8.242, 8.889, 9.877, 10.213]

### Destruction

- pk0000: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0100: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0110: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0111: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0120: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0121: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0122: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_grok 1.00, t_end 1.00
- pk0123: runs with a destroyer at end 0.00; cancellation ratio t_mem 1.00, t_end 1.00, t_grok 1.00

### Effective depth (comp_dep_state of long blocks, end)

- pk0100: blocks [2] → [0.99]
- pk0110: blocks [2, 3] → [0.971, 0.971]
- pk0111: blocks [2, 3, 4] → [0.981, 0.976, 0.985]
- pk0120: blocks [2, 3] → [3.676, 1.334]
- pk0121: blocks [2, 3, 4] → [0.567, 0.991, 0.95]
- pk0122: blocks [2, 3, 4] → [0.862, 0.987, 0.987]
- pk0123: blocks [2, 3, 4] → [0.677, 0.939, 0.995]

### Post-fit long-path learning (medians)

- pk0100: U_long@t_end 0.972, U_long@t_fit nan, U_long@t_grok 0.382, U_long@t_mem 0.980, long_gen_share@t_end 0.974, long_gen_share@t_fit 0.620, long_gen_share@t_grok 0.382, long_gen_share@t_mem 0.988
- pk0110: U_long@t_end 0.901, U_long@t_fit nan, U_long@t_grok 0.786, U_long@t_mem 0.983, long_gen_share@t_end 0.903, long_gen_share@t_fit 0.851, long_gen_share@t_grok 0.786, long_gen_share@t_mem 0.978
- pk0111: U_long@t_end 0.999, U_long@t_fit nan, U_long@t_grok 0.997, U_long@t_mem 1.026, long_gen_share@t_end 0.999, long_gen_share@t_fit 0.933, long_gen_share@t_grok 0.997, long_gen_share@t_mem 1.003
- pk0120: U_long@t_end 0.201, U_long@t_fit nan, U_long@t_grok 0.169, U_long@t_mem nan, long_gen_share@t_end 0.219, long_gen_share@t_fit 0.590, long_gen_share@t_grok 0.169, long_gen_share@t_mem 0.691
- pk0121: U_long@t_end 0.999, U_long@t_fit nan, U_long@t_grok 0.999, U_long@t_mem 1.006, long_gen_share@t_end 0.999, long_gen_share@t_fit 0.964, long_gen_share@t_grok 0.999, long_gen_share@t_mem 0.998
- pk0122: U_long@t_end 0.996, U_long@t_fit nan, U_long@t_grok 0.997, U_long@t_mem 1.001, long_gen_share@t_end 0.996, long_gen_share@t_fit 0.972, long_gen_share@t_grok 0.997, long_gen_share@t_mem 0.997
- pk0123: U_long@t_end 0.999, U_long@t_fit nan, U_long@t_grok 0.999, U_long@t_mem 1.001, long_gen_share@t_end 0.999, long_gen_share@t_fit 0.978, long_gen_share@t_grok 0.999, long_gen_share@t_mem 0.998

### Gradient allocation at matched train loss (median mean-block grad norm, short / long)

- pk0000: loss≤1.0: 0.0177 / —; loss≤0.1: 0.0177 / —; loss≤0.01: 0.0028 / —
- pk0100: loss≤1.0: 0.1107 / 0.2090; loss≤0.1: 0.0104 / 0.0211; loss≤0.01: 0.0035 / 0.0068
- pk0110: loss≤1.0: 0.1468 / 0.1289; loss≤0.1: 0.0557 / 0.0458; loss≤0.01: 0.0076 / 0.0067
- pk0111: loss≤1.0: 0.2098 / 0.0910; loss≤0.1: 0.0284 / 0.0119; loss≤0.01: 0.0163 / 0.0071
- pk0120: loss≤1.0: 0.2331 / 0.2783; loss≤0.1: 0.0458 / 0.0456; loss≤0.01: 0.0060 / 0.0079
- pk0121: loss≤1.0: 0.5648 / 0.3352; loss≤0.1: 0.1315 / 0.0847; loss≤0.01: 0.0249 / 0.0151
- pk0122: loss≤1.0: 0.4595 / 0.2732; loss≤0.1: 0.0511 / 0.0288; loss≤0.01: 0.0169 / 0.0101
- pk0123: loss≤1.0: 0.8334 / 0.7156; loss≤0.1: 0.1583 / 0.1366; loss≤0.01: 0.0248 / 0.0206

### Preservation (embedding path, end medians)

- pk0000: gen0 -0.005, mem0 0.012, rule_cancelled 0.094, mem_cancelled 1.954
- pk0100: gen0 -0.000, mem0 0.001, rule_cancelled -0.037, mem_cancelled 0.071
- pk0110: gen0 -0.002, mem0 0.004, rule_cancelled -0.236, mem_cancelled 0.825
- pk0111: gen0 -0.000, mem0 0.000, rule_cancelled 0.027, mem_cancelled 0.021
- pk0120: gen0 -0.011, mem0 0.028, rule_cancelled 1.558, mem_cancelled 1.617
- pk0121: gen0 -0.001, mem0 0.001, rule_cancelled -0.206, mem_cancelled 0.195
- pk0122: gen0 -0.002, mem0 0.005, rule_cancelled 0.044, mem_cancelled 0.253
- pk0123: gen0 -0.001, mem0 0.002, rule_cancelled -0.022, mem_cancelled 0.183
