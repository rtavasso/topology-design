# Rung 3 (commitment): results

**Provenance.** Design: [DESIGN.md](DESIGN.md) revision 2 with the derive-step corrections of §7a / §14 (`ba5bec6`). Stage 1: 48 runs, `c7439b9` ([results_data/stage1_report.md](results_data/stage1_report.md)). Stage 2: 120 runs from the stage-1 checkpoints, grid [grids/cm_stage2.json](grids/cm_stage2.json), bundle `ba5bec6`, two T4s, 2 h 20 wall, tests 60 passed / 1 skipped on the box; records in `results/kaggle_route-cm_stage2/results/` (outside git), analysis in [results_data/stage2_report.md](results_data/stage2_report.md) from `analyze.py`. Claim types follow AGENTS rule 5; every count below is over the preregistered seeds, and the per-seed predictions of §7a are read per seed.

Two corrections to the analysis code were made after seeing the records and before any reading (§12): the `collapsed` rule of review 2 B3 fired on every run with answer accuracy below 0.5, and the first-deviation key `SET>HOP` means the committed L program intact from the L start and the lost SET (the all-H route) from the E start. Outcomes are now taken from the actual signatures: **moved** (shortest-correct on P-words ≥ 0.95 at in4 and in8 on the standard strata), **moved~** (reached the criterion on the curve, ended at 0.85–0.95), **locked** (the committed program intact: `PREV>HOP` from E, `SET>HOP` with PREV in place from L), **drifted** (left the committed program without reaching the oracle's: the lost SET from E, PREV lost from L), **halted** (≥ half the queries halt where the oracle continues), **other**.

## 1. Summary

1. **Plasticity after commitment is a property of the signal, and exactness is the dividing line, not coverage.** From the E-committed routers under the local-trigger shift (δ = 0, p_P = 0.3): RT 2/2 (inside 100 steps), EI16 4/5, EX 3/5 (+ 1 to an R-like program), PF 3/5, PF-IS16 1/5, IW16 0/5. The exact own posterior (EX) moved although its mass on the correct route at the checkpoint was below 0.005 on every router; the sampled own posterior (IW16) never moved and collapsed the router into halting in 3/5; the sampled fixed prior (PF-IS16) drifted to the all-H route in 4/5. The preregistered reading row "EX moves, IW16 does not → sampling, not self-consistency, is the lock" is the one that applies. *Optimization; mechanism for the exact / sampled split (§4).*
2. **Coverage predicts movement for search as zero against non-zero, and the threshold rule was wrong by two orders of magnitude.** Over the 17 EI16 runs, routers whose beam kept the oracle route on ≥ 0.5 % of P-words at the checkpoint moved in 10/13; the one router at exactly 0.000 moved in 1/4. The beam's merge tie is a knife-edge: the run's own coverage goes from 0.00 to 1.00 inside one 100–300-step window and the move follows at once (§4.1). PA3's per-seed rows for s3 and s6 (coverage 0.01–0.02 → "do not move") were falsified; its AUROC form is moot. *Mechanism.*
3. **PF is not slow; it is bistable.** Every PF run from E drops the HOP − PREV margin at the relabeled class from +21 to +32 logits to ≈ +1.3 within 100 steps and sits there; the movers jump to −1.8 (p(PREV) ≈ 0.65) at 1,000 / 1,200 / 1,700 steps (0.3) and 200 / 1,400 (0.15); the rest sit at +1.1 to +1.7 for 2,900 steps. The crossing rate falls with the dose: 3/5, 2/3, 0/3. PA1's "PF ≥ 4/5" and PA2's "PF ≥ 2/3 at 0.08" fail at the count; the curve says a two-state system with a dose-dependent escape, not a slow drift. *Optimization; mechanism open (§4.2).*
4. **The shift induces EX's halting collapse, and more so at lower dose:** 1/5, 2/3, 3/3 of EX-from-E runs at p_P 0.3 / 0.15 / 0.08 ended halting at flagged pairs, against 0/5 of the EX controls continued on the unchanged stream (and 0/2 EI16 controls). Phase 1's seed-0 collapse was the same attractor reached spontaneously once in 7 × 2,500 steps. *Optimization; mechanism proposed (§4.3).*
5. **Commitment inverts two signals relative to scratch.** On δ = 0 from scratch at the constant lr: EX 0/3 (the all-H stall of rung 2), EI16 3/3, PF 3/3, PF-IS16 3/3. From the committed E routers: EX 3/5, PF-IS16 1/5. PF-IS16 also degrades a committed L router on the *unchanged* stream in 1/2 (§7). A deterministic committed router hands the sampled fixed prior sixteen identical samples plus the 6–8 % of label-reaching decoy routes, which become the whole target (review 2 Q3, confirmed). *Mechanism.*
6. **From the L start the router decides, not the signal.** L s1 stayed locked under RT (teacher forcing: p(SET) at the flag at 0.2 from step 100 to 3,000), EI16 and EX, and moved under PF in 900 steps; L s0 and s2 moved under RT, PF, EI16 and EX. The PA4 replacement holds for PF, EI16, IW16 and PF-IS16 and fails for RT (1/2) and for "the L start is easier for EI16" (2/3 against 4/5). *Optimization.*
7. **Filling in (shift B) does not happen at the preregistered endpoint for any learned signal.** PP words at pp8 reach 0.60–0.64 under EI16 and EX (L-L completions) at the cost of 0.3 on ordinary P-words; PF 0.2–0.4; RT 1/2 (the other RT run unlearned L inside the first 100 steps and never reached E: a dead zone between two programs that differ in one decision). PF at f = 0 drifts L → E and stays correct, as §5.1 derived; PF-IS16 at f = 0 breaks 1/2. *Optimization; PA′1 fails except its PF clause.*
8. **Checks:** stability controls EX 5/5 and EI16 2/2 unchanged over 3,000 steps; from-scratch references reproduce rung 2; program R reappears (EX s6: one extra step on every stratum, answer accuracy 0.92; IW16 s2 partially). Checks C (PC1) were not run in this stage (the derive script is not written); the PF and PF-IS16 phase-1 checkpoints are kept for it.

## 2. Stage 1 (recap; [results_data/stage1_report.md](results_data/stage1_report.md), DESIGN §7a)

7/7 EI16 routers committed to L and 5/7 EX routers to E (criterion 800 [700–1,000], deterministic, not R); EX seed 0 learned E and collapsed into halting at step ≈ 2,000, seed 4 was at 0.90–0.93. Core B: PB1 and PB2 held (the entropy and forced-breadth reward arms 0/2 each), PB3 held (PFB16 5/5 on δ = 1, 3/3 on δ = 0, program L, stochastic; a tie with the matched PF-IS16 reference at the dose). The gate: at the E routers the shift relabels one product-state class (HOP → PREV at +21 to +32); the beam keeps the oracle route on 0.73 / 0.47 / 0.02 / 0.01 / 0.00 of P-words for s1 / s5 / s6 / s3 / s2 (ε 0.3: 0.74 / 0.57 / 0.22 / 0.06 / 0.00); 16 samples never contain it; the exact own posterior puts < 0.005 on it. From the L routers the beam keeps it on 0.58–0.66, samples 0.00, own posterior 0.00–0.01 (s3 0.20, s5 0.05); the missing SET at the flag sits at +5 to +10; L s0 has HALT, not PREV, winning at the relabeled class.

## 3. Shift A from the E routers, p_P = 0.3 (core)

| signal | n | moved | locked | drifted | halted | other | time to move (steps from the end of warm-up) | §7a row | held? |
|---|---|---|---|---|---|---|---|---|---|
| RT | 2 | **2** | 0 | 0 | 0 | 0 | ≤ 100, ≤ 100 | 2/2 | yes |
| PF | 5 | **3** | 2 | 0 | 0 | 0 | 1,000 / 1,200 / 1,700; 2 not by 2,900 | ≥ 4/5 | **no** (3/5) |
| EX | 5 | **3** | 0 | 0 | 1 | 1 (R-like) | 100 / 700 / 2,000 | ≤ 1/5 (blind) | **no** |
| EI16 (ε 0.1) | 5 | **4** | 1 | 0 | 0 | 0 | ≤ 100 (s1, s6) / 500 (s5) / 1,400 (s3); s2 not | per seed: s1, s5 move; s2, s3, s6 do not | **no** (s3, s6 moved) |
| EI16 (ε 0.3) | 3 | **2** | 0 | 1 | 0 | 0 | ≤ 100 (s1) / 500 (s3); s2 not | s1 moves, s2 not, s3 reported | yes |
| PF-IS16 | 5 | **1** | 0 | 4 | 0 | 0 | 2,700 (s3) | ≤ 1/5 | yes |
| IW16 | 5 | **0** | 0 | 1 | 3 | 1 (R-like) | – | ≤ 1/5 | yes |

Per seed (checkpoint margin HOP − PREV in brackets): s1 [+26] moved under RT, EI16 (both ε), EX (2,000), not PF (locked at +1.4), drifted under PF-IS16, halted under IW16; s2 [+21] moved under RT, PF (1,000), EX (100); locked under EI16 (both ε, at +7 to +9 with the SET at the flag half lost), drifted under PF-IS16, R-like under IW16; s3 [+26] moved under PF (1,700), EI16 (1,400 at ε 0.1, 500 at ε 0.3), PF-IS16 (2,700); halted under EX and IW16; s5 [+32] moved under EI16 (500), EX (700); locked under PF (+1.1); drifted under PF-IS16, halted under IW16; s6 [+21] moved under PF (1,200), EI16 (≤ 100); R-like under EX (answer accuracy 0.92, one extra step on every stratum); drifted under PF-IS16, halted under IW16.

Final programs of the movers: L in every case (the δ = 0 oracle's; E is inexpressible there). The two R-like runs are correct with one extra step on all-H words too (none8 committed length 9.0 against 8.0) and are not shortest, so they are neither movers nor locks: the relabeled decision was learned through program R (F38).

## 4. Mechanisms

### 4.1 Search: the merge tie is a knife-edge, and coverage is binary

The ε-mixed beam under an E-committed router keeps E's per-episode route (SET at the unflagged predecessor), which is inexpressible as a rule, or the oracle's L route, and the merge keeps the more probable prefix (code review 1, item 4). The stage-2 curves record the run's own coverage (`search.oracle_kept_P` on the criterion set) every 100 steps:

| router | gate coverage (ε 0.1) | coverage at step 100 | first step ≥ 0.5 | move | end state |
|---|---|---|---|---|---|
| s1 | 0.73 | 1.00 | ≤ 100 | ≤ 100 | moved |
| s6 | 0.02 | 1.00 | ≤ 100 | ≤ 100 | moved |
| s5 | 0.47 | – | 500 | 500 | moved |
| s3 | 0.01 | 0.00 (0.00 through step 1,300) | 1,300–1,600 | 1,400 | moved |
| s2 | 0.000 | 0.00 (0.00 throughout) | never | – | locked (margin +22 → +7, p(SET) at the flag 0.83 → 0.35) |

s3's margin drifted from +26 to +8–13 over 1,300 steps with coverage exactly 0.00 and the loss flat at 0.48–0.52, then coverage went to 1.00 and shortest-correct from 0.01 to 1.00 inside one 300-step window. s2 drifted the same way to +7–10 and did not flip in 2,900 steps at 0.3 or 0.15, and flipped at 2,500 at the 0.08 dose. The reading: a few covered words give a gradient toward PREV at the relabeled class; the tie reverses collectively once p(PREV | own flag, 1, 1) passes π(SET | predecessor), and from then on every word is covered. Coverage at the checkpoint therefore predicts whether the cascade can start (zero against non-zero), not when: s6 at 0.02 flipped inside 100 steps and s3 at 0.01 took 1,400. Over the 17 EI16 runs (both ε, both starts, the doses): coverage ≥ 0.005 → 10/13 moved (the three others: L s1, and two dose runs that ended at 0.93–0.94 after reaching 1.00); exactly zero → 1/4. The preregistered thresholds (0.05, 0.3) and the AUROC form of PA3 do not describe this. ε 0.3 helped only s3 (500 against 1,400 steps), as the §7a reading-table note expected.

### 4.2 The exact fixed prior: two states at the relabeled class

| cell | seed | HOP − PREV at the checkpoint | at step 100 | at the end | p(PREV) at the end | crossing | outcome |
|---|---|---|---|---|---|---|---|
| p_P 0.3 | s1 | +26.0 | +0.9 | +1.4 | 0.14 | – | locked |
| | s2 | +21.2 | +1.1 | −1.8 | 0.65 | 1,000 | moved |
| | s3 | +26.2 | +1.0 | −1.9 | 0.65 | 1,700 | moved |
| | s5 | +31.8 | +1.3 | +1.1 | 0.18 | – | locked |
| | s6 | +21.4 | +1.1 | −1.8 | 0.63 | 1,200 | moved |
| p_P 0.15 | s1 | +26.0 | +1.6 | +1.5 | 0.13 | – | locked |
| | s2 | +21.2 | +2.2 | −1.7 | 0.63 | 200 | moved |
| | s3 | +26.2 | +0.9 | −1.9 | 0.67 | 1,100 | moved |
| p_P 0.08 | s1, s2, s3 | +21 to +26 | +1.2 to +2.0 | +1.3 to +1.7 | 0.11–0.16 | – | locked ×3 |

The first 100 steps take every router from a +21–32 lock to ≈ +1.3 (the CPU sanity run showed the same: +16.9 → +1.4 in 100 steps). After that the margin hovers within ±0.3 of +1.3 until, in the movers, it jumps to −1.8 between two 100-step points and shortest-correct goes from 0.02 to 0.98 at once. PF's loss is 2.2 in both states (the flat prior's padding sets the floor), so the two states are not separated by the loss PF sees. Why a policy-independent target has a quasi-stable state at p(PREV) ≈ 0.15 is not derived here; the pooled derivation (`routefind/derive/pooled.py`) gives the δ = 0 target at the class as PREV-dominant, which matches the movers' end state (0.63–0.67), not the locked one. Open (ledger I34). What the data do say: PF's movement is a stochastic escape whose rate falls with the number of relabel events per step (3/5 at 64 per batch, 2/3 at 34, 0/3 at 19), not a slow deterministic crossing, so L31's "slower by k×" does not apply as written either — the budget was not the obstacle at the margin, the escape was.

### 4.3 The own posterior: exact moves, sampled halts

EX's posterior over routes given the label, p(r | y) ∝ π(r) p(y | r), puts < 0.005 of its mass on the oracle route at every E checkpoint (gate), and EX moved in 3/5 at 0.3 (steps 100, 700, 2,000), with s6 going to an R-like program instead. An exact posterior is not sample-limited: a mass of 10⁻⁴ on the correct route, computed by recursion, still contributes a consistent gradient toward PREV every step, where 16 samples contribute nothing on 16 of 16 draws. The halting collapses (EX 1/5, 2/3, 3/3 by falling dose; IW16 3/5, 2/3, 1/3) follow the other half of the same target: on a P-word whose committed route is wrong, every route the policy puts mass on is wrong, the likelihood term is flat at the readout floor across them, and the own posterior over them is the policy itself — self-consistency with no information — leaving the step cost λ·E[steps] as the only consistent gradient on those words, which halts the route at the first flagged pair. From scratch (EX_d0) the policy's mode is the all-H route and the same mechanism gives the all-H stall. The dose dependence is read as a race between the oracle route's exact gradient (which grows with the number of P-words per batch) and the step cost (which does not): 3/3 collapses at 19 relabel events per batch, 1/5 at 64. *Proposed, not derived; the controls (0/5 collapses on the unchanged stream) establish that the shift induces it.*

### 4.4 The sampled targets: the label-reaching decoy samples are the target

Review 2 (Q3) predicted it from the gate: on the E routers 6–8 % of P-word queries have a label-reaching sample at step 0 (`search_init.hit_P`), none of which is the oracle route; they reach the label through the decoy chain, and under PF-IS16's π₀/π weights or IW16's own-posterior weights such a sample is the whole target for that query and teaches HOP at the flag. Observed: PF-IS16 from E ends with the SET at the flag gone (p(SET) at (own flag, 0, 0) 0.00–0.29, first deviation `SET>HOP` on 219–222 of 250 queries) in 4/5 at 0.3 and 3/3 at each dose; from L in 3/3 (PREV at the class lost as well); IW16 the same or halting. From scratch the same estimator learns δ = 0 in 3/3, because a stochastic young policy gives it sixteen different samples. On the unchanged δ = 1 stream from a committed L router PF-IS16 lost the program in 1/2 and PF (exact) moved both routers cleanly to E (§7). *The sampled fixed prior needs a stochastic proposal; a committed deterministic router is the worst proposal it can be given.*

## 5. The dose (shift A from E, p_P 0.3 / 0.15 / 0.08; relabel events per batch ≈ 64 / 34 / 19)

| signal | 0.3 | 0.15 | 0.08 | time at 0.15 / 0.08 (seeds 1, 2, 3) | PA2 |
|---|---|---|---|---|---|
| PF | 3/5 | 2/3 | 0/3 | 200, 1,400, – / –, –, – | time grows ✓; "≥ 2/3 at 0.08" ✗ |
| EX | 3/5 | 1/3 | 0/3 | –, 1,700, – / –, –, – (halting 2/3, 3/3) | collapse rises as the dose falls |
| EI16 (0.1) | 4/5 | 2/3 | 1/3 + 2 moved~ (0.93–0.94 after reaching 1.00) | ≤ 100, –, 600 / ≤ 100, 2,800, 1,100 | moved at every dose; endpoint noisier |
| PF-IS16 | 1/5 | 0/3 | 0/3 | – | never ✓ |
| IW16 | 0/5 | 0/3 | 0/3 | – | never ✓ |

PA2's clauses: "a signal that does not move at 0.3 does not move at any dose" holds for IW16 and PF-IS16; "time to move grows as p_P falls" holds for PF at the cell level and for EI16 s3 (1,400 → 600 → 1,100: no) and s2 (not, not, 2,800) only loosely; its falsifier "a signal that moves at 0.08 but not at 0.3" is met per seed by EI16 s2 and not by any cell. With the standard-configuration evaluation (review 2 B2) the endpoint is the same ruler at every dose; the two EI16 runs at 0.08 that end at 0.93–0.94 reached 0.97–1.00 on the curve, so the dose costs precision, not the move.

## 6. From the L routers (δ = 0, p_P 0.3)

| router | RT | PF | EX | EI16 | PF-IS16 | IW16 | missing SET at the flag (HOP − SET); p(PREV) at the class |
|---|---|---|---|---|---|---|---|
| L s0 | moved (900) | at threshold (0.95 / 0.96) | moved~ (1,200; 0.95) | moved (1,100) | drifted | drifted | +8; 0.23 (HALT winning) |
| L s1 | **locked** (p(SET) 0.2 from step 100) | **moved (900)** | locked (p(SET) 0.29, creeping) | locked (p(SET) 0.17) | drifted | drifted | +10; 1.00 |
| L s2 | – | moved (1,400) | moved (1,500) | moved (1,900) | drifted | drifted | +9; 1.00 |
| L s3 (EX only) | – | – | other (0.81; damaged on all-H words: committed length 16.5 against 8) | – | – | – | +5; 1.00 |
| L s5 (EX only) | – | – | locked (p(SET) 0.35) | – | – | – | +7; 1.00 |

Teacher forcing on L s1: SET accuracy at the oracle's SET states 0.00 for 3,000 steps, loss flat at 0.51–0.55, p(SET) at the class 0.09 → 0.21 by step 700 and flat after; on L s0 the same quantities go 0.18 → 0.36 (700) → 0.99 (1,000). The exact fixed prior moved s1 in 900 steps (p(SET) 0.31 → 0.81 between 700 and 1,300). We do not have the mechanism: it is one router, and the quantity that distinguishes it from s0 and s2 at the checkpoint (margins +10 against +8, +9; p(PREV) 1.00 like s2) does not account for it. The EX cell on s3 and s5 (added for the within-EX coverage contrast at 0.20 and 0.05 own posterior): s3 reached 1.00 on the curve at 1,800 and ended at 0.81 with its all-H words damaged; s5 is locked with p(SET) 0.35. The contrast is uninformative at these counts.

PA4 replacement: RT 2/2 → **1/2 ✗**; PF ≥ 2/3 → 2/3 + one at the threshold ✓; EI16 ≥ 2/3 → 2/3 ✓; EX ≤ 1/3 → 1 moved + 1 at 0.95 (borderline, read as ✗); IW16 ≤ 1/3 ✓ (0/3); PF-IS16 ≤ 1/3 ✓ (0/3). "The L start is easier for EI16" (L ≥ 2/3 against E 2/5): E came out 4/5, so **✗** — the two starts are alike for EI16 once the cascade is understood (coverage is non-zero from every L router, and from four of five E routers).

## 7. Controls and references

- **Stability (unchanged δ = 1 stream, 3,000 steps):** EX on all five E routers unchanged (loss 0.07 flat, p(SET) 1.00 at the flag, margins intact); EI16 2/2 unchanged. No spontaneous collapse in 5 × 3,000 EX steps: phase 1's seed-0 collapse (1 in 7 × 2,500) and the phase-2 collapses under δ = 0 are the same attractor, reached spontaneously rarely and under the shift often.
- **From scratch on δ = 0 at the constant lr (seeds 0–2, same initial weights as the phase-1 routers of those seeds):** EI16 3/3 (300), PF 3/3 (500), PF-IS16 3/3 (600), EX 0/3 (the all-H stall of rung 2: `SET>HOP` on 217–219 of 250). The inversion by commitment: EX 0/3 → 3/5, PF-IS16 3/3 → 1/5 (from E) and 0/3 (from L).
- **f = 0 controls (unchanged δ = 1 stream from the L routers, 3,000 steps):** EI16 2/2 held L; EX 2/2 held L; **PF 2/2 moved L → E** (shortest-correct 0.99 / 0.98; `set_late` on isolated-P words 1.0 → 0.01 between steps 1,600 and 2,600, after a dip to 0 at step 100 and recovery by 600) — the §5.1 derivation (the flat prior's SET padding prefers E at any PP fraction, including 0) held; **PF-IS16 1/2 lost the program** (shortest-correct 1.00 to 0.01 between steps 700 and 1,000; the other at 0.95).

## 8. Shift B: filling in (δ = 1, pp_frac 1.0, from the L routers; standard-configuration evaluation)

| signal | n | pp8 shortest-correct (PA′1 endpoint ≥ 0.90) | ordinary P-words at in8 (clean strata; 1.00 at the checkpoint) | program | PP route types among shortest-correct completions |
|---|---|---|---|---|---|
| RT | 2 | 1.00; **0.01** | 1.00; **0.01** | E; – | E-E 148, three-P runs 103 |
| EI16 | 3 | 0.62, 0.64, 0.64 | 0.71, 0.70, 0.72 | L ×3 | L-L 296, three-P runs 178 |
| EX | 3 | 0.60, 0.59; 0.00 | 0.70, 0.67; 0.00 | L, L; – | L-L 191, three-P runs 105 |
| PF | 3 | 0.21, 0.38, 0.40 | 0.39, 0.57, 0.34 | L, L, mix | L-L 123, L-double 20, E-L 10, mixed |
| PF-IS16 | 3 | 0.00, 0.02, 0.00 | 0.00, 0.04, 0.01 | – | – |

Filled at 0.90: RT 1/2, every learned signal 0/3. The RT run that failed is L s0 (the router with HALT winning at the relabeled class): teacher forcing on the E oracle removed L's PREV decision inside the first 100 steps (shortest-correct on ordinary P-words 1.00 → 0.00 by the first criterion point) and had not installed E's SET at the flag by step 3,000 (p(SET) 0.16): a dead zone between two programs that differ in one decision, entered by unlearning the old decision before the new one is in place. EI16 and EX fill PP words to ≈ 0.6 with L-type completions (L-L; the three-P runs at 0-1-1-1-0, i.e. a SET at each P) while losing 0.3 on ordinary P-words to the PP stream's false pointers — on the *clean* strata, so the loss is in the router, not the context; the preregistered expectation that EX fills in ≥ 2/3 (its posterior is on the completions) fails. PF is worse on both (its f = 0 control shows it also drifting to E on ordinary words, so the PP cell combines the prior drift with the shift). PF-IS16 loses everything. PA′1's falsifier "every signal fills PP in" is the opposite of what happened; its PF clause ("fails pp8 at pp_frac 1.0" against the count-ray generalization) came out on the pooled derivation's side: 0.21–0.40.

## 9. Predictions: scorecard (DESIGN §7, §7a)

| # | prediction | outcome |
|---|---|---|
| PA1 | RT 2/2; PF ≥ 4/5; IW16 ≤ 1/5; EX ≤ 1/5 (open row); EI16 per seed s1, s5 move / s2, s3, s6 not; PF-IS16 ≤ 1/5 | RT ✓; **PF 3/5 ✗**; IW16 0/5 ✓; **EX 3/5 (+ 1 R-like) ✗**; **EI16 4/5: s3, s6 moved ✗** (s1, s5, s2 ✓); PF-IS16 1/5 ✓. Neither falsifier ("every signal ≥ 4/5"; "only RT and PF") met. |
| PA2 | time to move grows as p_P falls; PF ≥ 2/3 at 0.08; a non-mover at 0.3 does not move at any dose | first clause ✓ for PF (and the halting collapse of EX rises 1/5 → 2/3 → 3/3); **PF 0/3 at 0.08 ✗**; third clause ✓ (IW16, PF-IS16). Falsifier "moves at 0.08 but not 0.3" met per seed by EI16 s2 (locked at 0.3 and 0.15, moved at 2,800 at 0.08), not by any cell. |
| PA3 | coverage predicts movement per seed, AUROC ≥ 0.75 over ≥ 20 runs | **Not in that form.** The predictor is binary (zero against non-zero: 1/4 against 10/13 within EI16) and the mechanism is a cascade (§4.1). For EX, mass < 0.005 and 3/5 moved: for exact targets "coverage" is the wrong quantity. Falsifier "AUROC ≤ 0.6: plasticity is not about what the target sees" is *not* the reading: it is about what the target sees, for search and for samplers, with a threshold at zero. |
| PA4 (withdrawn before test; X33) and its replacement | from L: RT 2/2, PF ≥ 2/3, EI16 ≥ 2/3, EX ≤ 1/3, IW16 ≤ 1/3, PF-IS16 ≤ 1/3; L easier for EI16 | **RT 1/2 ✗**; PF ✓; EI16 ✓; EX borderline ✗; IW16, PF-IS16 ✓; **"L easier for EI16" ✗** (2/3 against 4/5). |
| PB1–PB3 | stage 1 | held (DESIGN §7a; F42, F43). |
| PA′1 | RT 2/2; EX ≥ 2/3; EI16, PF-IS16 open; PF open between the pooled derivation (fails pp8) and the count-ray generalization; ordinary-word program stays L except PF at f = 0 | **RT 1/2 ✗; EX 0/3 ✗**; EI16 0/3, PF-IS16 0/3; PF 0/3 on the pooled derivation's side ✓; PF at f = 0 drifts to E ✓ (and PF-IS16 at f = 0 breaks). |
| PC1 | checks C | not run (derive script not written); open. |
| reading-table rows | | "EX moves, IW16 does not → sampling, not self-consistency, is the lock" applies. "RT does not move → re-run longer" does not apply from E (≤ 100 steps) and does apply from L s1 and L s0 under shift B, where the obstacle is a decision the router cannot yet express, not Σ lr. The §7a collapsed-run row applies to 6 EX runs; the controls' 0/5 says the signal under the shift did it. |

## 10. What this means (reading against DESIGN §7's table and the ledger)

**Risk 8 (plasticity after commitment).** Commitment is not sticky for exact targets (RT, EX, PF: the committed router is a *better* start than scratch for EX) and for search with any non-zero coverage (EI16 4/5, with the cascade), and it is a trap for the own-policy samplers: IW16 and PF-IS16 do not merely fail to move, they destroy the committed solution (halting; the all-H route), and PF-IS16 does so even on unchanged data. The ordering of DESIGN's own table ("how its targets can see the new PREV at the committed policy") is confirmed with one correction: "sees" means a *consistent non-zero gradient toward the correct route*, which exact recursion provides at any mass and which a 16-sample estimator provides only when a sample lands there, which under a committed policy is never. The synthesis of Entry 70 (search supplies targets, a fixed prior breaks ties) survives as a statement about plasticity with a sharper form: search is plastic as long as its support touches the alternative at all, and a fixed-prior target is plastic when exact and poisonous when sampled from the committed policy.

**What was wrong in the preregistration.** The coverage thresholds (0.05 / 0.3) treated a knife-edge as a dial; the EX row treated a posterior mass as a coverage; PA4's replacement assumed the ceiling (RT) moves everything from L; PA′1 assumed EX's posterior on PP completions would fill them in. All four were set on derive-step quantities that were correctly measured and wrongly read: coverage *is* the switch for search (zero against non-zero), mass *is* irrelevant for an exact target, the L start's missing decision *is* harder (a new discrimination at the flag state against a flip of a settled class), and shift B's filling-in is bounded by what the PP stream does to ordinary words.

**Three new facts the design did not anticipate**, each one seed-count short of a finding and each with a mechanism we do not have: PF's two-state behaviour at the relabeled class (§4.2); teacher forcing failing on L s1 for 3,000 steps where the fixed prior succeeds (§6); and the dead zone under shift B (RT on L s0; PF-IS16 at f = 0), where a router between two programs that differ in one decision is wrong on everything.

**For the ladder.** Rung 4 (distillation; D110) is unaffected. The signal for later rungs that train a committed router is narrowed to the exact targets and to search with engineered non-zero coverage; the own-policy samplers are out as trainers of committed routers, and PFB16 (stage 1's F42, a soft target over a searched support) is the untested candidate that combines PF's exactness-like behaviour with EI16's support — its plasticity was not run in this rung and is the obvious first check of the next one that needs a scalable signal.

## 11. Limitations

Seeds are 3–5 per cell (AGENTS rule 4; count predictions on near-bimodal outcomes); the per-seed EI16 claim rests on five routers and the L-s1 anomaly on one. The mechanisms of §4.2 and §6 are observations with proposed readings, not derivations. Checks C were not run. Two routers' results at the 0.95 threshold (PF and EX on L s0) are reported as such and not counted as movers. The halting-collapse mechanism (§4.3) is proposed from the loss structure and the dose pattern, not isolated by an intervention (an EX arm with λ = 0 would be the test).

## 12. Analysis-code corrections made after seeing the records

(1) `collapsed()` as written for review 2 B3 had a second clause (`acc_P < 0.5 and q_steps_P < 0.5·oracle`) whose key does not exist in the records, so it read as true for every run with answer accuracy below 0.5, which labelled every locked and drifted run "collapsed" in the first automatic pass; the clause is removed and halting is read from the first-deviation histogram alone. (2) The outcome taxonomy is conditioned on the start, because `SET>HOP` is the intact L program from L and the lost SET from E, and `PREV>HOP` is the intact E program; the δ = 1 cells (controls and shift B) get their own labels (held / switched / degraded / lost). (3) `moved~` was added for runs that reached the criterion on the curve and ended at 0.85–0.95 (three runs, all at the 0.08 dose or at the threshold). None of these changes a preregistered threshold; they change which words describe the non-movers. The first automatic pass (with the broken clause) labelled 61 of the 94 shift-A and shift-B runs "collapsed"; the corrected pass labels 12 halted (6 EX, 6 IW16).
