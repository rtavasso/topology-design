# Q2 — does the drift direction matter? (I13)

10 base runs, 160 jump runs.

### drift vs radial (primary null)

| jump at | H | eligible pairs | median steps saved vs base: drift / null | median D = t(null) − t(drift) | +/− | one-sided p | Holm p |
|---|---|---|---|---|---|---|---|
| 1000 | 250 | 10 | 87.5 / -25.0 | 112.5 | +9/−1 | 0.0107 | 0.0215 |
| 1000 | 1000 | 10 | 225.0 / -75.0 | 300.0 | +9/−1 | 0.0107 | 0.0215 |
| 2000 | 250 | 9 | 125 / 25 | 100 | +9/−0 | 0.0020 | 0.0078 |
| 2000 | 1000 | 9 | 375 / -100 | 475 | +9/−0 | 0.0020 | 0.0078 |

Pooled median D = 137.5 over 38 pairs.

### drift vs random (secondary)

| jump at | H | eligible pairs | median steps saved vs base: drift / null | median D = t(null) − t(drift) | +/− | one-sided p | Holm p |
|---|---|---|---|---|---|---|---|
| 1000 | 250 | 10 | 87.5 / 25.0 | 75.0 | +9/−1 | 0.0107 | 0.0215 |
| 1000 | 1000 | 10 | 225.0 / 187.5 | -25.0 | +3/−6 | 0.9102 | 0.9102 |
| 2000 | 250 | 9 | 125 / 0 | 100 | +9/−0 | 0.0020 | 0.0078 |
| 2000 | 1000 | 9 | 375 / -50 | 400 | +9/−0 | 0.0020 | 0.0078 |

Pooled median D = 100.0 over 38 pairs.

### drift vs rescale (secondary)

| jump at | H | eligible pairs | median steps saved vs base: drift / null | median D = t(null) − t(drift) | +/− | one-sided p | Holm p |
|---|---|---|---|---|---|---|---|
| 1000 | 250 | 10 | 87.5 / -50.0 | 137.5 | +9/−1 | 0.0107 | 0.0107 |
| 1000 | 1000 | 10 | 225.0 / -175.0 | 400.0 | +10/−0 | 0.0010 | 0.0039 |
| 2000 | 250 | 9 | 125 / 0 | 125 | +9/−0 | 0.0020 | 0.0059 |
| 2000 | 1000 | 9 | 375 / 25 | 350 | +9/−0 | 0.0020 | 0.0059 |

Pooled median D = 162.5 over 38 pairs.

## Verdict (preregistered)
- **P5 (direction matters, beyond norm changes):** drift jumps reach sustained grokking earlier than radial-matched null jumps in ≥ 3 of 4 cells (one-sided sign test, Holm p < 0.05) with pooled median D > 0; falsified if pooled median D ≤ 0: 4 cells significant, pooled median D = 137.5 → SUPPORTED
- *Secondary:* the random and rescale-only nulls above; random vs base and E2 replication in the json.
