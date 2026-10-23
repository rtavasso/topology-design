# E3 — credit by speed

180 runs. Main arms: branch 0 (blocks 1–3) steps ρ× faster than branch 1 at equal lr·wd. Controls: lr-only and wd-only at ρ = ¼ and 4.

| graph | arm | ρ | eligible | share of gen (end, median) | share of mem (peak) | collapses (median) |
|---|---|---|---|---|---|---|
| asym | main | 0.25 | 10/10 | 0.25 | 0.97 | 0.0 |
| asym | main | 0.5 | 9/10 | 0.36 | 0.99 | 2.0 |
| asym | main | 1 | 8/10 | 0.84 | 0.98 | 3.0 |
| asym | main | 2 | 6/10 | 0.97 | 1.00 | 3.5 |
| asym | main | 4 | 6/10 | 1.00 | 1.00 | 5.5 |
| asym | lr-only | 0.25 | 5/10 | 0.82 | 0.99 | 2.0 |
| asym | lr-only | 4 | 6/10 | 0.86 | 0.99 | 5.0 |
| asym | wd-only | 0.25 | 10/10 | 0.56 | 0.97 | 0.5 |
| asym | wd-only | 4 | 0/10 | — | — | 3.5 |
| sym | main | 0.25 | 10/10 | 0.01 | 0.01 | 0.0 |
| sym | main | 0.5 | 10/10 | 0.10 | 0.09 | 0.0 |
| sym | main | 1 | 9/10 | 0.50 | 0.50 | 0.0 |
| sym | main | 2 | 10/10 | 0.87 | 0.91 | 0.0 |
| sym | main | 4 | 10/10 | 0.98 | 0.99 | 0.0 |
| sym | lr-only | 0.25 | 10/10 | 0.14 | 0.34 | 0.0 |
| sym | lr-only | 4 | 10/10 | 0.92 | 0.64 | 0.0 |
| sym | wd-only | 0.25 | 10/10 | 0.48 | 0.00 | 0.0 |
| sym | wd-only | 4 | 10/10 | 0.50 | 1.00 | 0.0 |

## Verdicts (preregistered)
- **asym, P1** shares rise with ρ (testable ρ: [0.25, 0.5, 1.0, 2.0, 4.0]): τ(gen) = 0.845 (p = 0.0001), τ(mem) = 0.486 (p = 0.0005) → **supported**
- **asym, P2** co-location persists (logit gen − logit mem): trend τ = 0.799 (p = 0.0001), shifts vs ρ = 1: {'0.25': '-2.41', '0.5': '-2.31', '1': '0.00', '2': '1.62', '4': '2.14'} → **FALSIFIED (speed induces a role split)**
- **asym, P3** step size alone moves credit (lr-only arm, ρ = 4 vs ¼, paired): gen +2/−1 (p = 1.0000), mem +0/−0 (p = nan) → **untestable** (fewer than 6 eligible pairs)
- *asym, diagnostic:* logit-share effect ρ 4 vs ¼ — main 4.99, lr-only 0.32, wd-only —
- **sym, P1** shares rise with ρ (testable ρ: [0.25, 0.5, 1.0, 2.0, 4.0]): τ(gen) = 0.890 (p = 0.0001), τ(mem) = 0.940 (p = 0.0001) → **supported**
- **sym, P2** co-location persists (logit gen − logit mem): trend τ = -0.357 (p = 0.0030), shifts vs ρ = 1: {'0.25': '0.00', '0.5': '0.32', '1': '0.00', '2': '-0.41', '4': '0.00'} → **inconclusive**
- **sym, P3** step size alone moves credit (lr-only arm, ρ = 4 vs ¼, paired): gen +10/−0 (p = 0.0020), mem +10/−0 (p = 0.0020) → **supported**
- *sym, diagnostic:* logit-share effect ρ 4 vs ¼ — main 7.78, lr-only 4.23, wd-only 0.07
