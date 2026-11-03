# Rung-3 results: 168 runs

## Phase 1: commitment (clean evaluation; the gate of DESIGN §3)

| arm | n | learned | E / L / mixed-or-other / R | collapsed (learned, then left for halting) | criterion | deterministic | gate passes |
|---|---|---|---|---|---|---|---|
| EI16_p1 | 7 | 7 | 0 / 7 / 0 / 0 | 0 | 800 [700–1000] | 7 | 7 |
| EX_p1 | 7 | 5 | 5 / 0 / 0 / 0 | s0 (criterion at 700, collapsed by 2000) | 800 [700–1000] | 5 | 5 |
| PFIS16_p1 | 5 | 4 | 0 / 4 / 0 / 0 | 0 | 1500 [1200–1700] | 0 | 0 |
| PF_p1 | 3 | 0 | 0 / 0 / 0 / 0 | 0 | 1300 [1100–1400] | 0 | 0 |

What each committed router's phase-2 targets can see (DESIGN §5.3): shift A = the δ0 oracle route on P-words; shift B = a shortest correct completion on PP words. Coverage on the P-words the committed route does not already solve (stage-2 review B1), with the all-words value in brackets where it differs; n = words / unsolved.

| router | start | n | beam ε0.1 keeps oracle / oracle among completions / kept shortest (set_late) | beam ε0.3 keeps oracle | 16 samples | own posterior | margin HOP−PREV; p(PREV) | margin HOP−SET; p(SET) | B: n, beam ε0.1 / samples |
|---|---|---|---|---|---|---|---|---|---|
| cm1_EI16_p1_s0 | L | 271 / 258 | 0.58 [0.60] / 0.74 [0.75] / 0.67 [0.69] (0.89) | 0.59 [0.61] | 0.00 [0.05] | 0.01 [0.06] | -22.0 [-43.0, +13.8]; 0.23 | +7.9 [-9.6, +11.9]; 0.10 | 284 / 262, 0.73 [0.75] / 0.00 [0.08] |
| cm1_EI16_p1_s1 | L | 271 / 258 | 0.64 [0.65] / 0.82 [0.83] / 0.74 [0.76] (0.88) | 0.64 [0.66] | 0.00 [0.05] | 0.00 [0.05] | -21.0 [-23.2, -18.8]; 1.00 | +9.5 [-11.0, +12.2]; 0.10 | 284 / 263, 0.71 [0.74] / 0.00 [0.08] |
| cm1_EI16_p1_s2 | L | 271 / 258 | 0.64 [0.66] / 0.83 [0.84] / 0.75 [0.76] (0.88) | 0.64 [0.66] | 0.00 [0.05] | 0.01 [0.06] | -24.6 [-28.5, -20.9]; 1.00 | +8.7 [-8.8, +11.1]; 0.10 | 284 / 263, 0.70 [0.72] / 0.01 [0.08] |
| cm1_EI16_p1_s3 | L | 271 / 258 | 0.64 [0.66] / 0.83 / 0.75 [0.76] (0.88) | 0.64 [0.66] | 0.02 [0.07] | 0.20 [0.23] | -21.0 [-24.7, -18.4]; 1.00 | +5.4 [-9.2, +8.1]; 0.10 | 284 / 263, 0.72 [0.74] / 0.01 [0.08] |
| cm1_EI16_p1_s4 | L | 271 / 258 | 0.62 [0.64] / 0.81 [0.82] / 0.73 [0.75] (0.87) | 0.62 [0.64] | 0.00 [0.05] | 0.02 [0.07] | -20.4 [-22.5, -18.4]; 1.00 | +8.2 [-14.9, +13.5]; 0.10 | 284 / 263, 0.72 [0.74] / 0.00 [0.07] |
| cm1_EI16_p1_s5 | L | 271 / 258 | 0.66 [0.67] / 0.86 [0.87] / 0.70 [0.72] (0.95) | 0.66 [0.67] | 0.00 [0.05] | 0.05 [0.10] | -15.8 [-20.2, -11.8]; 1.00 | +6.9 [-10.7, +10.2]; 0.10 | 284 / 263, 0.68 [0.70] / 0.00 [0.07] |
| cm1_EI16_p1_s6 | L | 271 / 258 | 0.64 [0.65] / 0.83 [0.84] / 0.74 [0.76] (0.88) | 0.64 [0.65] | 0.00 [0.05] | 0.00 [0.05] | -24.6 [-25.0, -22.9]; 1.00 | +9.5 [-12.4, +12.9]; 0.10 | 284 / 263, 0.72 [0.74] / 0.00 [0.07] |
| cm1_EX_p1_s1 | E | 271 / 271 | 0.73 / 0.90 / 0.79 (0.99) | 0.74 | 0.00 | 0.00 | +26.0 [+24.6, +27.4]; 0.00 | -15.2 [-17.8, +20.0]; 0.99 | 284 / 278, 0.74 [0.75] / 0.00 [0.02] |
| cm1_EX_p1_s2 | E | 271 / 271 | 0.00 / 0.00 / 0.78 (0.04) | 0.00 | 0.00 | 0.00 | +21.2 [+21.2, +21.2]; 0.00 | -15.0 [-16.6, -11.5]; 1.00 | 284 / 278, 0.78 [0.79] / 0.00 [0.02] |
| cm1_EX_p1_s3 | E | 271 / 271 | 0.01 / 0.01 / 0.77 (0.10) | 0.06 | 0.00 | 0.00 | +26.2 [+24.3, +26.7]; 0.00 | -15.7 [-16.9, -10.7]; 1.00 | 284 / 278, 0.76 / 0.00 [0.02] |
| cm1_EX_p1_s5 | E | 271 / 271 | 0.47 / 0.56 / 0.77 (0.72) | 0.57 | 0.00 | 0.00 | +31.8 [+30.9, +32.6]; 0.00 | -15.7 [-17.0, +15.1]; 1.00 | 284 / 277, 0.78 [0.79] / 0.00 [0.02] |
| cm1_EX_p1_s6 | E | 271 / 271 | 0.02 / 0.03 / 0.77 (0.15) | 0.22 | 0.00 | 0.00 | +21.4 [+21.3, +24.0]; 0.00 | -15.9 [-17.1, -11.7]; 1.00 | 284 / 278, 0.78 [0.79] / 0.00 [0.02] |

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

## Phase 2: movement after commitment (shortest-correct on P-words under the new data, standard-configuration evaluation; time from the end of warm-up)

Outcomes per run: moved (≥ 0.95 at in4 and in8) / locked (first deviation is the relabeled decision, oracle PREV → router HOP) / collapsed (halting where the oracle continues; stage-2 review B3) / other. Mover counts are over runs that did not collapse; collapses are reported separately. Time to move also in relabel events (P-word queries seen = steps × batch × Qn × realized P-word fraction).

| arm | n | moved | moved~ | moved-R | locked | drifted | halted | other | time to move (steps) | (relabel events) | final in4 / in8 (mean) | program at end (E/L/mix/other) | pp8 at end | init routers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EI16_EA08 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 1100 [0–2800] | 22176 [0–56448] | 0.97 / 0.95 | 0/3/0/0 | 0.94 | 3 |
| EI16_EA15 | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 600 [0–>2900] (1/3 > 2900) | 23712 [0–>114608] (1/3 > 114608) | 0.67 / 0.67 | 0/2/0/1 | 0.66 | 3 |
| EI16_EA30 | 5 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 500 [0–>2900] (1/5 > 2900) | 34160 [0–>198128] (1/5 > 198128) | 0.80 / 0.80 | 0/5/0/0 | 0.80 | 5 |
| EI16_LA30 | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1900 [1100–>2900] (1/3 > 2900) | 129808 [75152–>198128] (1/3 > 198128) | 0.66 / 0.66 | 0/2/0/1 | 0.64 | 3 |
| EI16_d0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 300 [200–300] | 20496 [13664–20496] | 1.00 / 1.00 | 0/3/0/0 | 1.00 | 1 |
| EI16e3_EA30 | 3 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 500 [0–>2900] (1/3 > 2900) | 34160 [0–>198128] (1/3 > 198128) | 0.66 / 0.66 | 0/3/0/0 | 0.64 | 3 |
| EX_EA08 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | > 2900 | > 58464 | 0.00 / 0.00 | 0/0/0/3 | 0.00 | 3 |
| EX_EA15 | 3 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | > 2900 (2/3 censored) | > 114608 (2/3 censored) | 0.33 / 0.33 | 0/1/0/2 | 0.33 | 3 |
| EX_EA30 | 5 | 3 | 0 | 0 | 0 | 0 | 1 | 1 | 2000 [100–>2900] (2/5 > 2900) | 136640 [6832–>198128] (2/5 > 198128) | 0.60 / 0.60 | 0/3/0/2 | 0.60 | 5 |
| EX_LA30 | 5 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 1800 [1200–>2900] (2/5 > 2900) | 122976 [81984–>198128] (2/5 > 198128) | 0.65 / 0.58 | 0/5/0/0 | 0.48 | 5 |
| EX_d0 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | > 2500 | > 170800 | 0.01 / 0.00 | 0/0/0/3 | 0.00 | 1 |
| IW16_EA08 | 3 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | > 2900 | > 58464 | 0.01 / 0.01 | 0/0/0/3 | 0.00 | 3 |
| IW16_EA15 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | > 2900 | > 114608 | 0.01 / 0.00 | 0/0/0/3 | 0.00 | 3 |
| IW16_EA30 | 5 | 0 | 0 | 0 | 0 | 1 | 3 | 1 | > 2900 (4/5 censored) | > 198128 (4/5 censored) | 0.08 / 0.04 | 0/0/0/5 | 0.00 | 5 |
| IW16_LA30 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | > 2900 | > 198128 | 0.01 / 0.00 | 0/0/0/3 | 0.00 | 3 |
| PFIS16_EA08 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | > 2900 | > 58464 | 0.02 / 0.01 | 0/0/0/3 | 0.00 | 3 |
| PFIS16_EA15 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | > 2900 | > 114608 | 0.02 / 0.01 | 0/0/0/3 | 0.00 | 3 |
| PFIS16_EA30 | 5 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | > 2900 (4/5 censored) | > 198128 (4/5 censored) | 0.21 / 0.20 | 0/1/0/4 | 0.20 | 5 |
| PFIS16_LA30 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | > 2900 | > 198128 | 0.01 / 0.00 | 0/0/0/3 | 0.00 | 3 |
| PFIS16_d0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 600 [600–700] | 40992 [40992–47824] | 0.99 / 1.00 | 0/3/0/0 | 1.00 | 1 |
| PF_EA08 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | > 2900 | > 58464 | 0.02 / 0.01 | 0/2/0/1 | 0.01 | 3 |
| PF_EA15 | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1400 [200–>2900] (1/3 > 2900) | 55328 [7904–>114608] (1/3 > 114608) | 0.67 / 0.65 | 0/2/0/1 | 0.65 | 3 |
| PF_EA30 | 5 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 1700 [1000–>2900] (2/5 > 2900) | 116144 [68320–>198128] (2/5 > 198128) | 0.61 / 0.59 | 0/3/0/2 | 0.59 | 5 |
| PF_LA30 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 1400 [900–>2900] (1/3 > 2900) | 95648 [61488–>198128] (1/3 > 198128) | 0.99 / 0.97 | 0/3/0/0 | 0.96 | 3 |
| PF_d0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 500 [400–600] | 34160 [27328–40992] | 0.99 / 0.99 | 0/3/0/0 | 0.97 | 1 |
| RT_EA30 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 [0–0] | 0 [0–0] | 1.00 / 1.00 | 0/2/0/0 | 1.00 | 2 |
| RT_LA30 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | > 2900 (1/2 censored) | > 198128 (1/2 censored) | 0.50 / 0.50 | 0/1/0/1 | 0.50 | 2 |

**Unchanged-stream controls and shift B (δ = 1; the oracle is E).** Outcomes: held-L / held-E (the committed program, still ≥ 0.95 on clean strata), to-E / to-L (switched program, ≥ 0.95), degraded (0.5–0.95), lost (< 0.5, not halting), halted.

| arm | n | held | switched | degraded | lost | halted | final in4 / in8 on clean strata (mean) | program at end (E/L/mix/other) | pp8 on clean strata | init routers |
|---|---|---|---|---|---|---|---|---|---|---|
| EI16_LB0 | 2 | 2 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 0/2/0/0 | 0.01 | 2 |
| EI16_LB100 | 3 | 0 | 0 | 3 | 0 | 0 | 0.82 / 0.71 | 0/3/0/0 | 0.63 | 3 |
| EI16_LS | 2 | 2 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 0/2/0/0 | 0.01 | 2 |
| EX_ES | 5 | 5 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 5/0/0/0 | 0.01 | 5 |
| EX_LB0 | 2 | 2 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 0/2/0/0 | 0.01 | 2 |
| EX_LB100 | 3 | 0 | 0 | 2 | 1 | 0 | 0.55 / 0.45 | 0/2/0/1 | 0.39 | 3 |
| PFIS16_LB0 | 2 | 0 | 0 | 1 | 1 | 0 | 0.46 / 0.48 | 0/1/0/1 | 0.01 | 2 |
| PFIS16_LB100 | 3 | 0 | 0 | 0 | 3 | 0 | 0.02 / 0.02 | 0/1/0/2 | 0.01 | 3 |
| PF_LB0 | 2 | 0 | 2 | 0 | 0 | 0 | 1.00 / 0.98 | 2/0/0/0 | 0.01 | 2 |
| PF_LB100 | 3 | 0 | 0 | 1 | 2 | 0 | 0.54 / 0.44 | 0/2/1/0 | 0.33 | 3 |
| RT_LB100 | 2 | 0 | 1 | 0 | 1 | 0 | 0.51 / 0.50 | 2/0/0/0 | 0.50 | 2 |

Per run: outcome; final first deviation at in8 (oracle>taken: queries of 250); the criterion curve's end state at the relabeled classes (p(PREV) at (own flag, 1, 1), p(SET) at (own flag, 0, 0), margins HOP−PREV and HOP−SET) and the curve's maximum of shortest-correct on P-words:

- EI16_EA08 s1 ← cm1_EX_p1_s1.pt: **moved~**; {'SET>HOP': 9, 'HOP>SET': 6}; p(PREV) 1.00, p(SET) 0.96, HOP−PREV -11.95, HOP−SET -6.67; max shortest-correct 1.00
- EI16_EA08 s2 ← cm1_EX_p1_s2.pt: **moved**; {'HOP>SET': 1, 'SET>HOP': 1}; p(PREV) 1.00, p(SET) 0.99, HOP−PREV -18.07, HOP−SET -5.46; max shortest-correct 1.00
- EI16_EA08 s3 ← cm1_EX_p1_s3.pt: **moved~**; {'SET>HOP': 14}; p(PREV) 1.00, p(SET) 0.97, HOP−PREV -24.61, HOP−SET -5.05; max shortest-correct 0.97
- EI16_EA15 s1 ← cm1_EX_p1_s1.pt: **moved**; {'SET>HOP': 2}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.63, HOP−SET -8.98; max shortest-correct 1.00
- EI16_EA15 s2 ← cm1_EX_p1_s2.pt: **locked**; {'PREV>HOP': 141, 'SET>HOP': 78}; p(PREV) 0.00, p(SET) 0.53, HOP−PREV 9.77, HOP−SET -0.05; max shortest-correct 0.03
- EI16_EA15 s3 ← cm1_EX_p1_s3.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.50, HOP−SET -7.38; max shortest-correct 1.00
- EI16_EA30 s1 ← cm1_EX_p1_s1.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -21.39, HOP−SET -10.24; max shortest-correct 1.00
- EI16_EA30 s2 ← cm1_EX_p1_s2.pt: **locked**; {'PREV>HOP': 137, 'SET>HOP': 82}; p(PREV) 0.00, p(SET) 0.46, HOP−PREV 9.03, HOP−SET 0.62; max shortest-correct 0.03
- EI16_EA30 s3 ← cm1_EX_p1_s3.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.53, HOP−SET -7.96; max shortest-correct 1.00
- EI16_EA30 s5 ← cm1_EX_p1_s5.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.63, HOP−SET -9.92; max shortest-correct 1.00
- EI16_EA30 s6 ← cm1_EX_p1_s6.pt: **moved**; {'SET>HOP': 1}; p(PREV) 1.00, p(SET) 0.99, HOP−PREV -12.87, HOP−SET -5.56; max shortest-correct 1.00
- EI16_LA30 s0 ← cm1_EI16_p1_s0.pt: **moved**; {'HOP>HALT': 2}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -26.54, HOP−SET -8.63; max shortest-correct 1.00
- EI16_LA30 s1 ← cm1_EI16_p1_s1.pt: **locked**; {'SET>HOP': 222}; p(PREV) 1.00, p(SET) 0.14, HOP−PREV -24.13, HOP−SET 1.83; max shortest-correct 0.02
- EI16_LA30 s2 ← cm1_EI16_p1_s2.pt: **moved**; {'HOP>SET': 1}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -9.80, HOP−SET -9.78; max shortest-correct 1.00
- EI16_LB0 s0 ← cm1_EI16_p1_s0.pt: **held-L**; {'SET>HOP': 217}; p(PREV) 0.29, p(SET) 0.00, HOP−PREV -11.11, HOP−SET 11.44; max shortest-correct 1.00
- EI16_LB0 s1 ← cm1_EI16_p1_s1.pt: **held-L**; {'SET>HOP': 222}; p(PREV) 1.00, p(SET) 0.00, HOP−PREV -25.33, HOP−SET 8.71; max shortest-correct 1.00
- EI16_LB100 s0 ← cm1_EI16_p1_s0.pt: **degraded**; {'SET>HOP': 184, 'HOP>SET': 28, 'HOP>PREV': 11}; p(PREV) 1.00, p(SET) 0.09, HOP−PREV -7.08, HOP−SET 5.15; max shortest-correct 0.62
- EI16_LB100 s1 ← cm1_EI16_p1_s1.pt: **degraded**; {'SET>HOP': 200, 'HOP>SET': 16, 'HOP>PREV': 9}; p(PREV) 1.00, p(SET) 0.10, HOP−PREV -6.82, HOP−SET 4.16; max shortest-correct 0.63
- EI16_LB100 s2 ← cm1_EI16_p1_s2.pt: **degraded**; {'SET>HOP': 195, 'HOP>SET': 19, 'HOP>PREV': 11}; p(PREV) 1.00, p(SET) 0.10, HOP−PREV -8.63, HOP−SET 4.86; max shortest-correct 0.62
- EI16_LS s0 ← cm1_EI16_p1_s0.pt: **held-L**; {'SET>HOP': 217}; p(PREV) 0.29, p(SET) 0.00, HOP−PREV -11.11, HOP−SET 11.44; max shortest-correct 1.00
- EI16_LS s1 ← cm1_EI16_p1_s1.pt: **held-L**; {'SET>HOP': 222}; p(PREV) 1.00, p(SET) 0.00, HOP−PREV -25.33, HOP−SET 8.71; max shortest-correct 1.00
- EI16_d0 s0 ← scratch: **moved**; {'SET>HOP': 1, 'HOP>SET': 1}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.11, HOP−SET -8.02; max shortest-correct 1.00
- EI16_d0 s1 ← scratch: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.63, HOP−SET -10.75; max shortest-correct 1.00
- EI16_d0 s2 ← scratch: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.63, HOP−SET -13.73; max shortest-correct 1.00
- EI16e3_EA30 s1 ← cm1_EX_p1_s1.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.63, HOP−SET -7.21; max shortest-correct 1.00
- EI16e3_EA30 s2 ← cm1_EX_p1_s2.pt: **drifted**; {'SET>HOP': 125, 'PREV>HOP': 94}; p(PREV) 0.00, p(SET) 0.45, HOP−PREV 5.45, HOP−SET 0.37; max shortest-correct 0.03
- EI16e3_EA30 s3 ← cm1_EX_p1_s3.pt: **moved**; {'SET>HOP': 7, 'HOP>SET': 3}; p(PREV) 1.00, p(SET) 0.97, HOP−PREV -12.32, HOP−SET -7.66; max shortest-correct 0.97
- EX_EA08 s1 ← cm1_EX_p1_s1.pt: **halted**; {'SET>HALT': 202, 'SET>HOP': 17, 'HOP>HALT': 3}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 0.97, HOP−SET -2.75; max shortest-correct 0.00
- EX_EA08 s2 ← cm1_EX_p1_s2.pt: **halted**; {'SET>HALT': 196, 'SET>HOP': 15, 'HOP>HALT': 4}; p(PREV) 0.00, p(SET) 0.01, HOP−PREV -0.41, HOP−SET -2.78; max shortest-correct 0.00
- EX_EA08 s3 ← cm1_EX_p1_s3.pt: **halted**; {'SET>HALT': 148, 'SET>HOP': 72, 'HOP>HALT': 3}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 12.34, HOP−SET 0.13; max shortest-correct 0.00
- EX_EA15 s1 ← cm1_EX_p1_s1.pt: **halted**; {'SET>HALT': 206, 'SET>HOP': 8, 'HOP>HALT': 8}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV -0.09, HOP−SET -0.03; max shortest-correct 0.00
- EX_EA15 s2 ← cm1_EX_p1_s2.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -16.63, HOP−SET -13.61; max shortest-correct 1.00
- EX_EA15 s3 ← cm1_EX_p1_s3.pt: **halted**; {'SET>HALT': 202, 'SET>HOP': 11, 'HOP>HALT': 9}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 0.44, HOP−SET -0.44; max shortest-correct 0.00
- EX_EA30 s1 ← cm1_EX_p1_s1.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -23.76, HOP−SET -21.06; max shortest-correct 1.00
- EX_EA30 s2 ← cm1_EX_p1_s2.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.62, HOP−SET -14.45; max shortest-correct 1.00
- EX_EA30 s3 ← cm1_EX_p1_s3.pt: **halted**; {'SET>HALT': 211, 'SET>HOP': 10, 'HOP>HALT': 1}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 22.82, HOP−SET 3.18; max shortest-correct 0.00
- EX_EA30 s5 ← cm1_EX_p1_s5.pt: **moved**; {'SET>HOP': 1}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.48, HOP−SET -22.34; max shortest-correct 1.00
- EX_EA30 s6 ← cm1_EX_p1_s6.pt: **other**; {'HOP>SET': 250}; p(PREV) 0.07, p(SET) 0.56, HOP−PREV 12.04, HOP−SET -8.10; max shortest-correct 0.03
- EX_ES s1 ← cm1_EX_p1_s1.pt: **held-E**; {}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 27.45, HOP−SET -24.21; max shortest-correct 1.00
- EX_ES s2 ← cm1_EX_p1_s2.pt: **held-E**; {}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 27.63, HOP−SET -12.12; max shortest-correct 1.00
- EX_ES s3 ← cm1_EX_p1_s3.pt: **held-E**; {}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 27.63, HOP−SET -15.53; max shortest-correct 1.00
- EX_ES s5 ← cm1_EX_p1_s5.pt: **held-E**; {}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 27.63, HOP−SET -16.31; max shortest-correct 1.00
- EX_ES s6 ← cm1_EX_p1_s6.pt: **held-E**; {'SET>HOP': 2}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 27.63, HOP−SET -16.94; max shortest-correct 1.00
- EX_LA30 s0 ← cm1_EI16_p1_s0.pt: **moved~**; {'SET>HOP': 10, 'HOP>SET': 1}; p(PREV) 1.00, p(SET) 0.97, HOP−PREV -20.89, HOP−SET -9.45; max shortest-correct 0.97
- EX_LA30 s1 ← cm1_EI16_p1_s1.pt: **locked**; {'SET>HOP': 197, 'HOP>SET': 1}; p(PREV) 1.00, p(SET) 0.29, HOP−PREV -27.61, HOP−SET 0.60; max shortest-correct 0.15
- EX_LA30 s2 ← cm1_EI16_p1_s2.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.56, HOP−SET -11.14; max shortest-correct 1.00
- EX_LA30 s3 ← cm1_EI16_p1_s3.pt: **other**; {'HOP>PREV': 31, 'SET>PREV': 9, 'SET>HOP': 6}; p(PREV) 1.00, p(SET) 0.95, HOP−PREV -27.63, HOP−SET -8.33; max shortest-correct 1.00
- EX_LA30 s5 ← cm1_EI16_p1_s5.pt: **locked**; {'SET>HOP': 205, 'HOP>HALT': 4}; p(PREV) 1.00, p(SET) 0.35, HOP−PREV -25.18, HOP−SET -0.20; max shortest-correct 0.18
- EX_LB0 s0 ← cm1_EI16_p1_s0.pt: **held-L**; {'SET>HOP': 217}; p(PREV) 0.35, p(SET) 0.00, HOP−PREV -15.89, HOP−SET 13.98; max shortest-correct 1.00
- EX_LB0 s1 ← cm1_EI16_p1_s1.pt: **held-L**; {'SET>HOP': 222}; p(PREV) 1.00, p(SET) 0.00, HOP−PREV -27.57, HOP−SET 16.07; max shortest-correct 1.00
- EX_LB100 s0 ← cm1_EI16_p1_s0.pt: **lost**; {'SET>HOP': 217}; p(PREV) 0.24, p(SET) 0.03, HOP−PREV -14.47, HOP−SET 8.52; max shortest-correct 0.00
- EX_LB100 s1 ← cm1_EI16_p1_s1.pt: **degraded**; {'SET>HOP': 193, 'HOP>SET': 20, 'HOP>PREV': 12}; p(PREV) 1.00, p(SET) 0.09, HOP−PREV -22.19, HOP−SET 5.21; max shortest-correct 0.61
- EX_LB100 s2 ← cm1_EI16_p1_s2.pt: **degraded**; {'SET>HOP': 189, 'HOP>SET': 20, 'HOP>PREV': 15}; p(PREV) 1.00, p(SET) 0.10, HOP−PREV -25.16, HOP−SET 7.01; max shortest-correct 0.64
- EX_d0 s0 ← scratch: **drifted**; {'SET>HOP': 217}; p(PREV) 0.00, p(SET) 0.22, HOP−PREV 17.77, HOP−SET 1.34; max shortest-correct 0.01
- EX_d0 s1 ← scratch: **drifted**; {'SET>HOP': 218, 'HOP>SET': 2, 'PREV>HOP': 2}; p(PREV) 0.00, p(SET) 0.25, HOP−PREV 17.01, HOP−SET 1.17; max shortest-correct 0.03
- EX_d0 s2 ← scratch: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.23, HOP−PREV 15.76, HOP−SET 1.31; max shortest-correct 0.02
- IW16_EA08 s1 ← cm1_EX_p1_s1.pt: **halted**; {'SET>HALT': 174, 'SET>HOP': 39, 'HOP>HALT': 11}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 26.26, HOP−SET -2.88; max shortest-correct 0.01
- IW16_EA08 s2 ← cm1_EX_p1_s2.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV -9.98, HOP−SET 22.89; max shortest-correct 0.01
- IW16_EA08 s3 ← cm1_EX_p1_s3.pt: **drifted**; {'SET>HOP': 145, 'SET>HALT': 74, 'HOP>HALT': 2}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV -1.68, HOP−SET 7.25; max shortest-correct 0.03
- IW16_EA15 s1 ← cm1_EX_p1_s1.pt: **drifted**; {'SET>HOP': 176, 'SET>HALT': 44, 'HOP>HALT': 3}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 24.31, HOP−SET 15.31; max shortest-correct 0.02
- IW16_EA15 s2 ← cm1_EX_p1_s2.pt: **halted**; {'SET>HALT': 127, 'SET>HOP': 81, 'HOP>HALT': 8}; p(PREV) 0.00, p(SET) 0.02, HOP−PREV -3.34, HOP−SET -1.15; max shortest-correct 0.00
- IW16_EA15 s3 ← cm1_EX_p1_s3.pt: **halted**; {'SET>HALT': 180, 'SET>HOP': 37, 'HOP>HALT': 5}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV -1.80, HOP−SET 2.15; max shortest-correct 0.02
- IW16_EA30 s1 ← cm1_EX_p1_s1.pt: **halted**; {'SET>HALT': 168, 'SET>HOP': 38, 'HOP>HALT': 19}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 18.79, HOP−SET 3.38; max shortest-correct 0.01
- IW16_EA30 s2 ← cm1_EX_p1_s2.pt: **other**; {'HOP>SET': 248, 'SET>HALT': 2}; p(PREV) 0.00, p(SET) 0.03, HOP−PREV 26.38, HOP−SET -3.23; max shortest-correct 0.41
- IW16_EA30 s3 ← cm1_EX_p1_s3.pt: **halted**; {'SET>HALT': 183, 'SET>HOP': 28, 'HOP>HALT': 12}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 17.72, HOP−SET 1.06; max shortest-correct 0.00
- IW16_EA30 s5 ← cm1_EX_p1_s5.pt: **drifted**; {'SET>HOP': 208, 'SET>HALT': 11, 'HOP>HALT': 2}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 24.90, HOP−SET 16.82; max shortest-correct 0.97
- IW16_EA30 s6 ← cm1_EX_p1_s6.pt: **halted**; {'SET>HALT': 175, 'SET>HOP': 45, 'HOP>HALT': 1}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 12.03, HOP−SET 11.01; max shortest-correct 0.01
- IW16_LA30 s0 ← cm1_EI16_p1_s0.pt: **drifted**; {'SET>HOP': 194, 'HOP>HALT': 21, 'SET>HALT': 9}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 21.52, HOP−SET 21.48; max shortest-correct 0.01
- IW16_LA30 s1 ← cm1_EI16_p1_s1.pt: **drifted**; {'SET>HOP': 222, 'HALT>HOP': 5}; p(PREV) 0.05, p(SET) 0.44, HOP−PREV 2.39, HOP−SET 0.16; max shortest-correct 0.02
- IW16_LA30 s2 ← cm1_EI16_p1_s2.pt: **drifted**; {'SET>HOP': 201, 'HOP>HALT': 17, 'SET>HALT': 5}; p(PREV) 0.00, p(SET) 0.01, HOP−PREV 0.48, HOP−SET 9.15; max shortest-correct 0.01
- PFIS16_EA08 s1 ← cm1_EX_p1_s1.pt: **drifted**; {'SET>HOP': 222}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 25.81, HOP−SET 24.69; max shortest-correct 0.02
- PFIS16_EA08 s2 ← cm1_EX_p1_s2.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 22.95, HOP−SET 1.15; max shortest-correct 0.01
- PFIS16_EA08 s3 ← cm1_EX_p1_s3.pt: **drifted**; {'SET>HOP': 222}; p(PREV) 0.01, p(SET) 0.27, HOP−PREV 3.92, HOP−SET 0.96; max shortest-correct 0.01
- PFIS16_EA15 s1 ← cm1_EX_p1_s1.pt: **drifted**; {'SET>HOP': 212, 'HOP>HALT': 11, 'SET>HALT': 3}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 19.26, HOP−SET 1.12; max shortest-correct 0.03
- PFIS16_EA15 s2 ← cm1_EX_p1_s2.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.25, HOP−PREV 22.53, HOP−SET 1.12; max shortest-correct 0.01
- PFIS16_EA15 s3 ← cm1_EX_p1_s3.pt: **drifted**; {'SET>HOP': 222}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 21.81, HOP−SET 1.16; max shortest-correct 0.01
- PFIS16_EA30 s1 ← cm1_EX_p1_s1.pt: **drifted**; {'SET>HOP': 222}; p(PREV) 0.04, p(SET) 0.29, HOP−PREV 2.77, HOP−SET 0.83; max shortest-correct 0.03
- PFIS16_EA30 s2 ← cm1_EX_p1_s2.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.25, HOP−PREV 19.88, HOP−SET 1.12; max shortest-correct 0.01
- PFIS16_EA30 s3 ← cm1_EX_p1_s3.pt: **moved**; {}; p(PREV) 0.69, p(SET) 0.93, HOP−PREV -3.50, HOP−SET -2.61; max shortest-correct 1.00
- PFIS16_EA30 s5 ← cm1_EX_p1_s5.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.00, HOP−PREV 0.48, HOP−SET 23.81; max shortest-correct 0.01
- PFIS16_EA30 s6 ← cm1_EX_p1_s6.pt: **drifted**; {'SET>HOP': 221}; p(PREV) 0.00, p(SET) 0.25, HOP−PREV 21.35, HOP−SET 1.11; max shortest-correct 0.01
- PFIS16_LA30 s0 ← cm1_EI16_p1_s0.pt: **drifted**; {'SET>HOP': 217}; p(PREV) 0.00, p(SET) 0.23, HOP−PREV 22.93, HOP−SET 1.19; max shortest-correct 0.01
- PFIS16_LA30 s1 ← cm1_EI16_p1_s1.pt: **drifted**; {'SET>HOP': 222}; p(PREV) 0.09, p(SET) 0.32, HOP−PREV 2.06, HOP−SET 0.77; max shortest-correct 0.02
- PFIS16_LA30 s2 ← cm1_EI16_p1_s2.pt: **drifted**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 20.87, HOP−SET 1.13; max shortest-correct 0.01
- PFIS16_LB0 s0 ← cm1_EI16_p1_s0.pt: **lost**; {'SET>HOP': 217}; p(PREV) 0.00, p(SET) 0.26, HOP−PREV 8.51, HOP−SET 1.02; max shortest-correct 1.00
- PFIS16_LB0 s1 ← cm1_EI16_p1_s1.pt: **degraded**; {'SET>HOP': 222}; p(PREV) 0.03, p(SET) 0.29, HOP−PREV 3.20, HOP−SET 0.89; max shortest-correct 1.00
- PFIS16_LB100 s0 ← cm1_EI16_p1_s0.pt: **lost**; {'SET>HOP': 217}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 18.09, HOP−SET 1.14; max shortest-correct 0.00
- PFIS16_LB100 s1 ← cm1_EI16_p1_s1.pt: **lost**; {'SET>HOP': 219, 'HOP>SET': 3}; p(PREV) 0.09, p(SET) 0.30, HOP−PREV 2.29, HOP−SET 0.84; max shortest-correct 0.28
- PFIS16_LB100 s2 ← cm1_EI16_p1_s2.pt: **lost**; {'SET>HOP': 219}; p(PREV) 0.00, p(SET) 0.24, HOP−PREV 20.54, HOP−SET 1.14; max shortest-correct 0.01
- PFIS16_d0 s0 ← scratch: **moved**; {}; p(PREV) 0.72, p(SET) 0.88, HOP−PREV -3.32, HOP−SET -2.07; max shortest-correct 1.00
- PFIS16_d0 s1 ← scratch: **moved**; {}; p(PREV) 0.72, p(SET) 0.88, HOP−PREV -3.20, HOP−SET -2.00; max shortest-correct 1.00
- PFIS16_d0 s2 ← scratch: **moved**; {'SET>HOP': 3}; p(PREV) 0.75, p(SET) 0.85, HOP−PREV -3.47, HOP−SET -1.77; max shortest-correct 1.00
- PF_EA08 s1 ← cm1_EX_p1_s1.pt: **locked**; {'PREV>HOP': 221, 'HOP>SET': 1}; p(PREV) 0.12, p(SET) 0.84, HOP−PREV 1.61, HOP−SET -1.70; max shortest-correct 0.04
- PF_EA08 s2 ← cm1_EX_p1_s2.pt: **locked**; {'PREV>HOP': 219}; p(PREV) 0.11, p(SET) 0.82, HOP−PREV 1.73, HOP−SET -1.54; max shortest-correct 0.04
- PF_EA08 s3 ← cm1_EX_p1_s3.pt: **locked**; {'PREV>HOP': 222}; p(PREV) 0.16, p(SET) 0.82, HOP−PREV 1.32, HOP−SET -1.56; max shortest-correct 0.02
- PF_EA15 s1 ← cm1_EX_p1_s1.pt: **locked**; {'PREV>HOP': 221, 'SET>HOP': 1}; p(PREV) 0.13, p(SET) 0.81, HOP−PREV 1.55, HOP−SET -1.46; max shortest-correct 0.04
- PF_EA15 s2 ← cm1_EX_p1_s2.pt: **moved**; {'HOP>SET': 6, 'SET>HOP': 1}; p(PREV) 0.63, p(SET) 0.83, HOP−PREV -1.72, HOP−SET -1.58; max shortest-correct 1.00
- PF_EA15 s3 ← cm1_EX_p1_s3.pt: **moved**; {'PREV>HOP': 8}; p(PREV) 0.67, p(SET) 0.85, HOP−PREV -1.88, HOP−SET -1.71; max shortest-correct 0.98
- PF_EA30 s1 ← cm1_EX_p1_s1.pt: **locked**; {'PREV>HOP': 218, 'SET>HOP': 4}; p(PREV) 0.14, p(SET) 0.80, HOP−PREV 1.42, HOP−SET -1.41; max shortest-correct 0.04
- PF_EA30 s2 ← cm1_EX_p1_s2.pt: **moved**; {'HOP>SET': 1}; p(PREV) 0.65, p(SET) 0.81, HOP−PREV -1.85, HOP−SET -1.48; max shortest-correct 1.00
- PF_EA30 s3 ← cm1_EX_p1_s3.pt: **moved**; {'PREV>HOP': 9}; p(PREV) 0.65, p(SET) 0.83, HOP−PREV -1.87, HOP−SET -1.59; max shortest-correct 0.98
- PF_EA30 s5 ← cm1_EX_p1_s5.pt: **locked**; {'PREV>HOP': 219}; p(PREV) 0.18, p(SET) 0.79, HOP−PREV 1.14, HOP−SET -1.36; max shortest-correct 0.03
- PF_EA30 s6 ← cm1_EX_p1_s6.pt: **moved**; {}; p(PREV) 0.63, p(SET) 0.81, HOP−PREV -1.81, HOP−SET -1.47; max shortest-correct 1.00
- PF_LA30 s0 ← cm1_EI16_p1_s0.pt: **other**; {'SET>HOP': 8, 'HOP>SET': 3, 'PREV>HOP': 1}; p(PREV) 0.63, p(SET) 0.81, HOP−PREV -1.88, HOP−SET -1.52; max shortest-correct 0.96
- PF_LA30 s1 ← cm1_EI16_p1_s1.pt: **moved**; {'HOP>SET': 8, 'PREV>SET': 2}; p(PREV) 0.63, p(SET) 0.80, HOP−PREV -1.83, HOP−SET -1.39; max shortest-correct 1.00
- PF_LA30 s2 ← cm1_EI16_p1_s2.pt: **moved**; {'HOP>SET': 2}; p(PREV) 0.63, p(SET) 0.83, HOP−PREV -1.86, HOP−SET -1.64; max shortest-correct 1.00
- PF_LB0 s0 ← cm1_EI16_p1_s0.pt: **to-E**; {'SET>HOP': 38, 'HOP>PREV': 1, 'PREV>HOP': 1}; p(PREV) 0.13, p(SET) 0.57, HOP−PREV 1.47, HOP−SET -0.27; max shortest-correct 1.00
- PF_LB0 s1 ← cm1_EI16_p1_s1.pt: **to-E**; {'SET>HOP': 31, 'HOP>SET': 4, 'HOP>PREV': 2}; p(PREV) 0.10, p(SET) 0.55, HOP−PREV 1.80, HOP−SET -0.21; max shortest-correct 1.00
- PF_LB100 s0 ← cm1_EI16_p1_s0.pt: **lost**; {'SET>HOP': 196, 'HOP>SET': 17, 'HOP>PREV': 7}; p(PREV) 0.13, p(SET) 0.37, HOP−PREV 1.60, HOP−SET 0.55; max shortest-correct 0.37
- PF_LB100 s1 ← cm1_EI16_p1_s1.pt: **degraded**; {'SET>HOP': 207, 'HOP>SET': 12, 'HOP>PREV': 6}; p(PREV) 0.15, p(SET) 0.38, HOP−PREV 1.43, HOP−SET 0.48; max shortest-correct 0.44
- PF_LB100 s2 ← cm1_EI16_p1_s2.pt: **lost**; {'SET>HOP': 161, 'HOP>SET': 46, 'HOP>PREV': 7}; p(PREV) 0.18, p(SET) 0.48, HOP−PREV 1.11, HOP−SET 0.09; max shortest-correct 0.50
- PF_d0 s0 ← scratch: **moved**; {'HOP>SET': 7}; p(PREV) 0.64, p(SET) 0.83, HOP−PREV -1.80, HOP−SET -1.60; max shortest-correct 1.00
- PF_d0 s1 ← scratch: **moved**; {'PREV>SET': 4, 'HOP>SET': 1}; p(PREV) 0.58, p(SET) 0.79, HOP−PREV -1.63, HOP−SET -1.35; max shortest-correct 1.00
- PF_d0 s2 ← scratch: **moved**; {'HOP>SET': 3}; p(PREV) 0.58, p(SET) 0.81, HOP−PREV -1.67, HOP−SET -1.49; max shortest-correct 1.00
- RT_EA30 s1 ← cm1_EX_p1_s1.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.43, HOP−SET -9.29; max shortest-correct 1.00
- RT_EA30 s2 ← cm1_EX_p1_s2.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -15.17, HOP−SET -7.78; max shortest-correct 1.00
- RT_LA30 s0 ← cm1_EI16_p1_s0.pt: **moved**; {}; p(PREV) 1.00, p(SET) 1.00, HOP−PREV -27.62, HOP−SET -11.59; max shortest-correct 1.00
- RT_LA30 s1 ← cm1_EI16_p1_s1.pt: **locked**; {'SET>HOP': 222}; p(PREV) 1.00, p(SET) 0.18, HOP−PREV -19.46, HOP−SET 1.53; max shortest-correct 0.02
- RT_LB100 s0 ← cm1_EI16_p1_s0.pt: **lost**; {'SET>HOP': 209, 'PREV>SET': 7}; p(PREV) 0.00, p(SET) 0.16, HOP−PREV 11.23, HOP−SET 1.60; max shortest-correct 0.01
- RT_LB100 s1 ← cm1_EI16_p1_s1.pt: **to-E**; {}; p(PREV) 0.00, p(SET) 1.00, HOP−PREV 14.75, HOP−SET -11.33; max shortest-correct 1.00

### EI16: coverage at the checkpoint against movement (shift A; coverage = the beam keeps the δ0 oracle route, on P-words the committed route does not already solve)

The in-run column is the same quantity on the run's own criterion set at step 0; `flip` is the first step at which the run's beam keeps the oracle route on ≥ 0.5 of criterion P-words (the merge tie reversing); `move` is the first confirmed step with shortest-correct ≥ 0.95.

| cell | seed | start | gate coverage | in-run (step 0) | flip (step) | move (step) | outcome | HOP−PREV at checkpoint |
|---|---|---|---|---|---|---|---|---|
| EI16_EA08 | s1 | E | 0.73 | 0.74 | 0 | 0 | moved~ | +26 |
| EI16_EA08 | s2 | E | 0.00 | 0.00 | 2500 | 2800 | moved | +21 |
| EI16_EA08 | s3 | E | 0.01 | 0.01 | 900 | 1100 | moved~ | +26 |
| EI16_EA15 | s1 | E | 0.73 | 0.74 | 0 | 0 | moved | +26 |
| EI16_EA15 | s2 | E | 0.00 | 0.00 | None | None | locked | +21 |
| EI16_EA15 | s3 | E | 0.01 | 0.01 | 300 | 600 | moved | +26 |
| EI16_EA30 | s1 | E | 0.73 | 0.74 | 0 | 0 | moved | +26 |
| EI16_EA30 | s2 | E | 0.00 | 0.00 | None | None | locked | +21 |
| EI16_EA30 | s3 | E | 0.01 | 0.01 | 1300 | 1400 | moved | +26 |
| EI16_EA30 | s5 | E | 0.47 | 0.43 | 500 | 500 | moved | +32 |
| EI16_EA30 | s6 | E | 0.02 | 0.01 | 0 | 0 | moved | +21 |
| EI16_LA30 | s0 | L | 0.58 | 0.60 | 0 | 1100 | moved | -22 |
| EI16_LA30 | s1 | L | 0.64 | 0.60 | 0 | None | locked | -21 |
| EI16_LA30 | s2 | L | 0.64 | 0.60 | 0 | 1900 | moved | -25 |
| EI16e3_EA30 | s1 | E | 0.74 | 0.74 | 0 | 0 | moved | +26 |
| EI16e3_EA30 | s2 | E | 0.00 | 0.00 | None | None | drifted | +21 |
| EI16e3_EA30 | s3 | E | 0.06 | 0.06 | 500 | 500 | moved | +26 |

Coverage exactly zero (< 0.005): 1/4 moved; coverage ≥ 0.005: 10/13 moved (the preregistered rule's thresholds were 0.05 and 0.3).

### The other signals from E: what the target saw and what happened

| cell | seed | 16-sample coverage | label-reaching samples (step 0) | own posterior | outcome | move (step) |
|---|---|---|---|---|---|---|
| EX_EA08 | s1 | 0.00 | – | 0.00 | halted | None |
| EX_EA08 | s2 | 0.00 | – | 0.00 | halted | None |
| EX_EA08 | s3 | 0.00 | – | 0.00 | halted | None |
| EX_EA15 | s1 | 0.00 | – | 0.00 | halted | None |
| EX_EA15 | s2 | 0.00 | – | 0.00 | moved | 1700 |
| EX_EA15 | s3 | 0.00 | – | 0.00 | halted | None |
| EX_EA30 | s1 | 0.00 | – | 0.00 | moved | 2000 |
| EX_EA30 | s2 | 0.00 | – | 0.00 | moved | 100 |
| EX_EA30 | s3 | 0.00 | – | 0.00 | halted | None |
| EX_EA30 | s5 | 0.00 | – | 0.00 | moved | 700 |
| EX_EA30 | s6 | 0.00 | – | 0.00 | other | None |
| EX_LA30 | s0 | 0.00 | – | 0.01 | moved~ | 1200 |
| EX_LA30 | s1 | 0.00 | – | 0.00 | locked | None |
| EX_LA30 | s2 | 0.00 | – | 0.01 | moved | 1500 |
| EX_LA30 | s3 | 0.02 | – | 0.20 | other | 1800 |
| EX_LA30 | s5 | 0.00 | – | 0.05 | locked | None |
| IW16_EA08 | s1 | 0.00 | 0.06 | 0.00 | halted | None |
| IW16_EA08 | s2 | 0.00 | 0.08 | 0.00 | drifted | None |
| IW16_EA08 | s3 | 0.00 | 0.08 | 0.00 | drifted | None |
| IW16_EA15 | s1 | 0.00 | 0.06 | 0.00 | drifted | None |
| IW16_EA15 | s2 | 0.00 | 0.08 | 0.00 | halted | None |
| IW16_EA15 | s3 | 0.00 | 0.08 | 0.00 | halted | None |
| IW16_EA30 | s1 | 0.00 | 0.06 | 0.00 | halted | None |
| IW16_EA30 | s2 | 0.00 | 0.08 | 0.00 | other | None |
| IW16_EA30 | s3 | 0.00 | 0.08 | 0.00 | halted | None |
| IW16_EA30 | s5 | 0.00 | 0.08 | 0.00 | drifted | 200 |
| IW16_EA30 | s6 | 0.00 | 0.08 | 0.00 | halted | None |
| IW16_LA30 | s0 | 0.00 | 0.11 | 0.01 | drifted | None |
| IW16_LA30 | s1 | 0.00 | 0.16 | 0.00 | drifted | None |
| IW16_LA30 | s2 | 0.00 | 0.11 | 0.01 | drifted | None |
| PFIS16_EA08 | s1 | 0.00 | 0.06 | 0.00 | drifted | None |
| PFIS16_EA08 | s2 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA08 | s3 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA15 | s1 | 0.00 | 0.06 | 0.00 | drifted | None |
| PFIS16_EA15 | s2 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA15 | s3 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA30 | s1 | 0.00 | 0.06 | 0.00 | drifted | None |
| PFIS16_EA30 | s2 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA30 | s3 | 0.00 | 0.08 | 0.00 | moved | 2700 |
| PFIS16_EA30 | s5 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_EA30 | s6 | 0.00 | 0.08 | 0.00 | drifted | None |
| PFIS16_LA30 | s0 | 0.00 | 0.11 | 0.01 | drifted | None |
| PFIS16_LA30 | s1 | 0.00 | 0.16 | 0.00 | drifted | None |
| PFIS16_LA30 | s2 | 0.00 | 0.11 | 0.01 | drifted | None |
| PF_EA08 | s1 | 0.00 | – | 0.00 | locked | None |
| PF_EA08 | s2 | 0.00 | – | 0.00 | locked | None |
| PF_EA08 | s3 | 0.00 | – | 0.00 | locked | None |
| PF_EA15 | s1 | 0.00 | – | 0.00 | locked | None |
| PF_EA15 | s2 | 0.00 | – | 0.00 | moved | 200 |
| PF_EA15 | s3 | 0.00 | – | 0.00 | moved | 1400 |
| PF_EA30 | s1 | 0.00 | – | 0.00 | locked | None |
| PF_EA30 | s2 | 0.00 | – | 0.00 | moved | 1000 |
| PF_EA30 | s3 | 0.00 | – | 0.00 | moved | 1700 |
| PF_EA30 | s5 | 0.00 | – | 0.00 | locked | None |
| PF_EA30 | s6 | 0.00 | – | 0.00 | moved | 1200 |
| PF_LA30 | s0 | 0.00 | – | 0.01 | other | None |
| PF_LA30 | s1 | 0.00 | – | 0.00 | moved | 900 |
| PF_LA30 | s2 | 0.00 | – | 0.01 | moved | 1400 |

### PF from E: the margin at the relabeled class drops to ≈ +1 within 100 steps and crosses later, or not

| cell | seed | HOP−PREV at checkpoint | at step 100 | at the end | p(PREV) at the end | crossing (first step with margin < 0) | move (step) | outcome |
|---|---|---|---|---|---|---|---|---|
| PF_EA08 | s1 | +26.0 | +1.3 | +1.6 | 0.12 | None | None | locked |
| PF_EA08 | s2 | +21.2 | +2.0 | +1.7 | 0.11 | None | None | locked |
| PF_EA08 | s3 | +26.2 | +1.2 | +1.3 | 0.16 | None | None | locked |
| PF_EA15 | s1 | +26.0 | +1.6 | +1.5 | 0.13 | None | None | locked |
| PF_EA15 | s2 | +21.2 | +2.2 | -1.7 | 0.63 | 200 | 200 | moved |
| PF_EA15 | s3 | +26.2 | +0.9 | -1.9 | 0.67 | 1100 | 1400 | moved |
| PF_EA30 | s1 | +26.0 | +0.9 | +1.4 | 0.14 | None | None | locked |
| PF_EA30 | s2 | +21.2 | +1.1 | -1.8 | 0.65 | 1000 | 1000 | moved |
| PF_EA30 | s3 | +26.2 | +1.0 | -1.9 | 0.65 | 1700 | 1700 | moved |
| PF_EA30 | s5 | +31.8 | +1.3 | +1.1 | 0.18 | None | None | locked |
| PF_EA30 | s6 | +21.4 | +1.5 | -1.8 | 0.63 | 1200 | 1200 | moved |
| PF_LA30 | s0 | -22.0 | +2.6 | -1.9 | 0.63 | 2700 | None | other |
| PF_LA30 | s1 | -21.0 | +1.9 | -1.8 | 0.63 | 800 | 900 | moved |
| PF_LA30 | s2 | -24.6 | +1.4 | -1.9 | 0.63 | 1200 | 1400 | moved |

### EX from E: the halting collapse by dose, against the unchanged-stream controls

- EX_EA30: halted 1/5; moved 3; moved-R 0; locked 0; drifted 0; other 1
- EX_EA15: halted 2/3; moved 1; moved-R 0; locked 0; drifted 0; other 0
- EX_EA08: halted 3/3; moved 0; moved-R 0; locked 0; drifted 0; other 0
- EX_ES: halted 0/5; moved 0; moved-R 0; locked 0; drifted 0; other 0
- EX_d0: halted 0/3; moved 0; moved-R 0; locked 0; drifted 3; other 0
- IW16_EA30: halted 3/5; moved 0; moved-R 0; locked 0; drifted 1; other 1
- IW16_EA15: halted 2/3; moved 0; moved-R 0; locked 0; drifted 1; other 0
- IW16_EA08: halted 1/3; moved 0; moved-R 0; locked 0; drifted 2; other 0

### From the L routers: outcome by router and signal (shift A)

| router | RT | PF | EX | EI16 | PFIS16 | IW16 | HOP−SET at checkpoint; p(PREV) at class |
|---|---|---|---|---|---|---|---|
| L s0 | moved (900) | other | moved~ (1200) | moved (1100) | drifted | drifted | +8; 0.23 |
| L s1 | locked | moved (900) | locked | locked | drifted | drifted | +10; 1.00 |
| L s2 | – | moved (1400) | moved (1500) | moved (1900) | drifted | drifted | +9; 1.00 |
| L s3 | – | – | other (1800) | – | – | – | +5; 1.00 |
| L s5 | – | – | locked | – | – | – | +7; 1.00 |

### Filling in (shift B): PA'1 endpoint pp8 shortest-correct ≥ 0.9 (clean evaluation), and set placement on the shortest-correct PP routes

Route types for a run of two P's from the placement counts (predecessor-first-second-third-elsewhere): E-double 2-0-0-0-0, E-E 1-1-0-0-0, E-L 1-0-1-0-0, L-double 0-2-0-0-0, L-L 0-1-1-0-0; longer runs and other patterns are left as recorded.

- EI16_LB0: filled 0/2 (pp8 0.00, 0.02); ordinary P-words at in8 on clean strata 1.00, 1.00; program L/L; placement {'2:0-1-0-0-0': 3, '3:0-1-0-0-1': 1}
- EI16_LB100: filled 0/3 (pp8 0.62, 0.64, 0.64); ordinary P-words at in8 on clean strata 0.71, 0.70, 0.72; program L/L/L; placement {'L-L': 296, '3:0-1-1-1-0': 178, '2:0-1-1-0-1': 1, '3:0-1-0-0-3': 1}
- EX_LB0: filled 0/2 (pp8 0.00, 0.02); ordinary P-words at in8 on clean strata 1.00, 1.00; program L/L; placement {'2:0-1-0-0-0': 3, '3:0-1-0-0-1': 1}
- EX_LB100: filled 0/3 (pp8 0.00, 0.60, 0.59); ordinary P-words at in8 on clean strata 0.00, 0.70, 0.67; program –/L/L; placement {'L-L': 191, '3:0-1-1-1-0': 105}
- PFIS16_LB0: filled 0/2 (pp8 0.00, 0.01); ordinary P-words at in8 on clean strata 0.00, 0.95; program –/L; placement {'2:0-1-0-0-0': 3, '2:0-0-0-0-0': 1}
- PFIS16_LB100: filled 0/3 (pp8 0.00, 0.02, 0.00); ordinary P-words at in8 on clean strata 0.00, 0.04, 0.01; program –/L/–; placement {'L-double': 3, '2:0-0-0-0-0': 2}
- PF_LB0: filled 0/2 (pp8 0.00, 0.02); ordinary P-words at in8 on clean strata 0.99, 0.98; program E/E; placement {'2:1-0-0-0-0': 3, '3:1-0-1-0-0': 1}
- PF_LB100: filled 0/3 (pp8 0.21, 0.38, 0.40); ordinary P-words at in8 on clean strata 0.39, 0.57, 0.34; program L/L/mix; placement {'L-L': 123, '3:0-1-1-1-0': 60, 'L-double': 20, '3:0-2-0-1-0': 13, 'E-L': 10, '3:1-0-1-1-0': 4, '2:1-0-2-0-0': 3, '3:0-1-2-0-0': 2, '2:0-0-1-0-1': 2, '3:1-0-2-0-0': 2, '2:0-0-1-0-0': 2, '3:0-2-0-2-0': 2, '2:0-1-2-0-0': 2, '3:0-0-1-1-0': 1, '3:0-0-2-0-0': 1, '2:0-2-0-0-1': 1, '3:1-0-1-1-1': 1}
- RT_LB100: filled 1/2 (pp8 0.01, 1.00); ordinary P-words at in8 on clean strata 0.01, 1.00; program E/E; placement {'E-E': 148, '3:1-1-1-0-0': 103, '2:0-0-0-0-0': 1}
