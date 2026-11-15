# Which signal learns a program of learned blocks — a toy (EXPLORATORY)

Definitions, and the predictions written before the run, in `routing_toy.py`. K = n blocks, step size 0.5, 16 sampled routes per step for the sampled signals, 10 seeds, budgets {2: 4000, 3: 20000, 4: 60000} steps. A run has learned when the exact success probability p exceeds 0.9.

| passes n | p at the start (≈ n⁻ⁿ) | signal | runs that learned | steps to learn, median [min, max] | among runs that did not: final p, median | … and the router's smallest top probability over passes, median |
|---|---|---|---|---|---|---|
| 2 | 0.2500 | EXACT-R | 10/10 | 109 [92, 141] | – | – |
| 2 | 0.2500 | EXACT-L | 10/10 | 55 [51, 64] | – | – |
| 2 | 0.2500 | FIX | 0/10 | – | 0.251 | 0.52 |
| 2 | 0.2500 | RF | 10/10 | 109 [93, 152] | – | – |
| 2 | 0.2500 | RF-B | 10/10 | 114 [87, 152] | – | – |
| 2 | 0.2500 | OWN-S | 10/10 | 54 [47, 74] | – | – |
| 2 | 0.2500 | RF-01 | 10/10 | 99 [84, 131] | – | – |
| 2 | 0.2500 | OWN-01 | 10/10 | 48 [44, 61] | – | – |
| 2 | 0.2500 | SEARCH | 10/10 | 38 [38, 39] | – | – |
| 3 | 0.0370 | EXACT-R | 10/10 | 540 [433, 736] | – | – |
| 3 | 0.0370 | EXACT-L | 10/10 | 99 [96, 109] | – | – |
| 3 | 0.0370 | FIX | 0/10 | – | 0.037 | 0.35 |
| 3 | 0.0370 | RF | 10/10 | 431 [229, 10737] | – | – |
| 3 | 0.0370 | RF-B | 10/10 | 572 [475, 644] | – | – |
| 3 | 0.0370 | OWN-S | 10/10 | 98 [91, 126] | – | – |
| 3 | 0.0370 | RF-01 | 10/10 | 460 [334, 546] | – | – |
| 3 | 0.0370 | OWN-01 | 10/10 | 99 [83, 106] | – | – |
| 3 | 0.0370 | SEARCH | 7/10 | 79 [79, 79] | 0.250 | 1.00 |
| 4 | 0.0039 | EXACT-R | 10/10 | 4507 [3608, 6039] | – | – |
| 4 | 0.0039 | EXACT-L | 10/10 | 145 [139, 175] | – | – |
| 4 | 0.0039 | FIX | 0/10 | – | 0.004 | 0.26 |
| 4 | 0.0039 | RF | 7/10 | 9290 [948, 17211] | 0.250 | 1.00 |
| 4 | 0.0039 | RF-B | 10/10 | 4664 [3584, 6597] | – | – |
| 4 | 0.0039 | OWN-S | 10/10 | 140 [134, 151] | – | – |
| 4 | 0.0039 | RF-01 | 10/10 | 3211 [2391, 3972] | – | – |
| 4 | 0.0039 | OWN-01 | 7/10 | 271 [164, 370] | 0.250 | 1.00 |
| 4 | 0.0039 | SEARCH | 4/10 | 119 [119, 119] | 0.250 | 1.00 |
