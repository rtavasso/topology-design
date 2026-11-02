# Rung-3 results: 48 runs

## Phase 1: commitment (clean evaluation; the gate of DESIGN §3)

| arm | n | learned | E / L / mixed-or-other / R | criterion | deterministic | gate passes |
|---|---|---|---|---|---|---|
| EI16_p1 | 7 | 7 | 0 / 7 / 0 / 0 | 800 [700–1000] | 7 | – |
| EX_p1 | 7 | 5 | 5 / 0 / 0 / 0 | 800 [700–1000] | 5 | – |
| PFIS16_p1 | 5 | 4 | 0 / 4 / 0 / 0 | 1500 [1200–1700] | 0 | – |
| PF_p1 | 3 | 0 | 0 / 0 / 0 / 0 | 1300 [1100–1400] | 0 | – |

## Core B: exploration for the reward signals (from scratch, constant lr, 5k)

| arm | n | learned | program | criterion | distinct routes / beam hit (last 1k) | ESS / explained variance |
|---|---|---|---|---|---|---|
| AC16B_d1 | 2 | 0 | – | > 5000 | – / 0.99…0.99 | – / -0.00…0.07 |
| ACMC16B_d1 | 2 | 0 | – | > 5000 | – / 0.99…0.99 | – / 0.07…0.07 |
| PFB16_d0 | 3 | 3 | L/L/L | 800 [800–800] | – / 1.00…1.00 | – / – |
| PFB16_d1 | 5 | 5 | L/L/L/L/L | 1100 [900–1400] | – / 1.00…1.00 | – / – |
| PFB16_d1p4 | 3 | 2 | L/L | 3800 [2800–>5000] (1/3 > 5000) | – / 1.00…1.00 | – / – |
| PFIS16_d1p4 | 3 | 2 | mix/E | 2800 [2600–4500] | 13.81…13.83 / – | 0.80…0.81 / – |
| RL16B_d1 | 2 | 0 | – | > 5000 | – / 0.99…1.00 | 0.01…0.01 / – |
| RL16H0.01_d1 | 2 | 0 | – | > 5000 | 1.13…1.13 / – | – / – |
| RL16H0.03_d1 | 2 | 0 | – | > 5000 | 7.16…7.32 / – | – / – |
| RL16H0.1_d1 | 2 | 0 | – | > 5000 | 13.51…13.51 / – | – / – |
