# Diagnosing the walk — results

Scored against [PREREGISTRATION.md](PREREGISTRATION.md), committed before any run, with the analysis code ([`analyze.py`](analyze.py)) also committed first.

**Status**

| | |
|---|---|
| E1, E2 | done (23 Oct 2026), independently recomputed. |
| E3, E4 | done (23 Oct 2026), independently recomputed. |

**Run details**
- Kaggle, 2× Tesla T4 per job, torch 2.11.0+cu128, commit `d95cf1f`.
- The fleet tests passed on the GPU before each job (59 passed, 1 skipped).
- Raw results stay outside git, in `results/kaggle_e*/results/`.

**Independent check:** see *Verification* at the end.

---

## E1 — information gap (measurement)

**Setup**
- 40 stream runs: 140 pairs at the start, then +32 pairs every *b* steps until all 1,392 have arrived.
- 30 data-floor runs on fixed datasets.

**Streams.** Prequential code length is each pair's cross-entropy at its arrival step, before any training on it, summed over the stream. Values are medians over 5 seeds.

| graph | lr | steps per increment | gradient descent (nats) | program prior | uniform | GD / program | production point (examples) | censored |
|---|---|---|---|---|---|---|---|---|
| asym | 1e-3 | 400 | 38,146 | 537 | 5,676 | 71 | 1,392 | 5/5 |
| asym | 1e-3 | 1600 | 39,287 | 537 | 5,676 | 73 | 1,392 | 5/5 |
| asym | 2e-3 | 400 | 35,230 | 537 | 5,676 | 66 | 1,392 | 5/5 |
| asym | 2e-3 | 1600 | 38,165 | 537 | 5,676 | 71 | 1,356 | 2/5 |
| sym | 1e-3 | 400 | 26,064 | 537 | 5,676 | 49 | 1,392 | 3/5 |
| sym | 1e-3 | 1600 | 38,104 | 537 | 5,676 | 71 | 1,196 | 0/5 |
| sym | 2e-3 | 400 | 25,831 | 537 | 5,676 | 48 | 1,324 | 0/5 |
| sym | 2e-3 | 1600 | 29,773 | 537 | 5,676 | 55 | 1,100 | 0/5 |

**Data floor.** Seeds grokking per fixed training fraction, out of 3, up to 60,000 steps.

| training fraction (examples) | 0.15 (522) | 0.2 (696) | 0.25 (870) | 0.3 (1,044) | 0.4 (1,392) |
|---|---|---|---|---|---|
| sym | 0 | 0 | 1 | 3 | 3 |
| asym | 0 | 0 | 0 | 0 | 3 |

### What E1 measures

- **The program prior pins down the rule after k\* = 3 examples.** Its code length, 537 nats, is almost entirely the initial 140-pair block at uniform cost. This passes the calibration check.
- **Gradient descent is worse than not learning at all for most of the stream.**
  - Its code length is 4.5–7× that of a uniform guesser, and 48–73× the program prior's.
  - The mechanism shows in the per-increment numbers. Once the network has memorized the pairs it has seen, it predicts each new pair *confidently wrong*: 12–29 nats per pair at arrival, against 4.1 for a uniform guess, with arrival accuracy near 0. It recovers only in the last increments.
  - Example (sym, lr 1e-3, b = 400): 4.1, 12.5, 18.5, 18.6, 19.3, 21.4, 28.8, … nats per pair, falling to 3.6, 2.4, 0.1 over the final increments.
  - This is the anti-generalization of memorization (ledger F4: support on unseen inputs goes negative), now measured as a cost.
- **More steps per example: slightly earlier production, worse interim predictions.**
  - With 4× more steps per increment, sym produced the rule earlier: 1,100 vs 1,324 examples at lr 2e-3, and 1,196 vs ≥ 1,392 at lr 1e-3.
  - But its code length *rose* (sym lr 1e-3: 26k → 38k nats). More walking per example deepens confident memorization before the rule arrives.
- **The gap is mainly a data gap, not a step gap.**
  - The data floor, even with up to 60,000 steps, is 1,044 examples for sym (348× k\*). For asym it lies between 1,044 and 1,392 examples (≤ 464× k\*); 0.4 is both the top of the sweep and the main training fraction, so it is an upper bound. At 870 examples, only 1 of 3 sym seeds groks.
  - The stream production points (≈ 1,100–1,390) sit at that floor.
  - Extra steps per example barely move it. Gradient descent needs hundreds of times more examples than a learner with a prior over short programs.
- *Exploratory (wiring):* asym, the chain plus shortcut, is worse on every E1 measure: higher floor, later or censored production, larger code length.

**Claim type:** measurement and optimization. The program prior is one prior; "worse than" is relative to it.

---

## E2 — is the answer visible before the network produces it, and can a learner act on it?

**Setup**
- 20 base runs: sym at lr 2e-3, asym at lr 1e-3.
- 80 drift-jump runs.
- 20 H = 0 noise-floor runs.
- All 20 base runs reached sustained grokking (test accuracy ≥ 0.95 held for 2,000 steps).

| graph | model reaches 0.5 / 0.9 (median step) | refit readout | straight-line jump-ahead, H = 1,000 |
|---|---|---|---|
| sym | 2,550 / 2,850 | 2,500 / 2,800 | 2,100 / 2,500 |
| asym | 18,350 / 22,700 | 18,300 / 21,550 | 19,000 / 26,800 |

### Verdicts (preregistered)

**P1, readout visibility — NOT supported, as the pilot predicted.**
- The refit readout's median relative lead at test accuracy 0.9 is 0.006, against the ≥ 0.2 required.
- Leads were consistent but negligible: +13/−1 with 6 ties, sign test p = 0.002. A lead is mostly exactly one 100-step eval.
- The network's own readout is essentially optimal for its current features at every point. The walk's time is spent building features, not fitting the readout.

**P1b, drift visibility — NOT supported, pooled as preregistered.**
- Median relative lead 0.077, +11/−6, p = 0.33.
- *Exploratory split by graph,* which is where the effect is:
  - **sym:** the straight-line jump-ahead reaches 0.5 test accuracy 15–22% earlier than the network, in **10 of 10 seeds**;
  - **asym:** −7% to +1%, none.
- At H = 4,000 the extrapolation overshoots: median relative lead −1.4 at the 0.9 threshold.
- The decay-aware operator shows no lead anywhere (median 0.0).

**P2, actionability — supported.** All four sym cells qualify.

| sym jump | steps saved (median) | Holm p |
|---|---|---|
| at 1,000, H = 250 | 100 | 0.016 |
| at 1,000, H = 1,000 | 200 | 0.016 |
| at 2,000, H = 250 | 100 | 0.023 |
| at 2,000, H = 1,000 | 400 | 0.023 |

- That is 4–14% of the ≈ 2,850-step run, with no more collapses.
- The savings are coarse: 1–4 eval intervals of 100 steps.
- The jump fast-forwards less than its nominal horizon: about 0.2–0.4 × H for H = 1,000 and 0.4–0.8 × H for H = 250.
- Many s = 2,000 jumps came after test accuracy had started rising (0.09–0.33 before the jump in several seeds), not mid-plateau. At s = 2,000 with H = 1,000, median test accuracy went from 0.06 to 0.44 at the moment of the jump.
- The s = 2,000 cells have 9 runs, because seed 613's base run reached sustained grokking exactly at step 2,000.
- **Noise floor = 0, but only for floating-point noise.** The H = 0 arms match their bases exactly in train and test accuracy and collapse counts at every eval; weight norms differ by about 4e-6. The floor measures floating-point nondeterminism between batches, not how a run reacts to a perturbation, so "saving above the noise floor" is easily met.
- **Sign tests are two-sided** (the analysis as committed). Read one-sided, as "in the positive direction" in the preregistration could be read, the sym cells' Holm p becomes 0.008–0.012. One asym cell (s 8,000, H 1,000; +9/−1) would then pass the sign test (Holm p 0.043), but it still fails the collapse condition: median change +0.5, a fragile margin from an even-sized set.
- **asym:** median savings of 1,700–5,900 steps (larger than H in some cells), but inconsistent across seeds (Holm p 0.09–1.0). Within a single cell, savings range from −21,100 to +15,300 steps. asym's grokking time is chaotic under perturbation, because of repeated collapses. The −21,100 comes from one seed whose jump runs never grok and are censored at the 40,000-step horizon.

### What E2 says, and doesn't

- **The readout is not the bottleneck; the features are.**
- **In the stable all-parallel network, the destination is partly visible from the run's own recent drift, about a fifth of the way ahead, and acting on it saves a modest fraction of the walk.** Most of the walk is not skippable by straight-line extrapolation.
- **In the chain-plus-shortcut network, the trajectory is not extrapolatable this way.** That is a wiring effect on how predictable learning is (exploratory).
- **Not established: that the drift *direction* is what helps.** The design has no matched-norm random-direction jump, so "jumping along the drift helps" is not separated from "any kick helps".
  - That matters most for asym, whose large but inconsistent savings look like kicks out of collapse cycles.
  - It matters less for sym, whose savings scale with the horizon jumped and come with an immediate rise in test accuracy (0.06 → 0.44).
  - Follow-up: ledger I13.

**Claim type:** optimization and measurement.

---

## E3 — credit by speed

**Design.** 180 runs.
- Main arms: branch 0 (blocks 1–3) steps ρ× faster than branch 1 at equal lr·wd.
- Controls at ρ = ¼ and 4: lr-only and wd-only.
- sym at lr 2e-3 for 30,000 steps; asym at lr 1e-3 for 50,000 steps.
- Shares are branch 0's: memorization at the run's peak of total memorization, generalization over the last 2,000 steps. Eligible runs only.

| graph | arm | ρ | eligible | share of gen (end) | share of mem (peak) | collapses |
|---|---|---|---|---|---|---|
| sym | main | ¼ / ½ / 1 / 2 / 4 | 10 / 10 / 9 / 10 / 10 | 0.01 / 0.10 / 0.50 / 0.87 / 0.98 | 0.01 / 0.09 / 0.50 / 0.91 / 0.99 | 0 |
| sym | lr-only | ¼ / 4 | 10 / 10 | 0.14 / 0.92 | 0.34 / 0.64 | 0 |
| sym | wd-only | ¼ / 4 | 10 / 10 | 0.48 / 0.50 | 0.00 / 1.00 | 0 |
| asym | main | ¼ / ½ / 1 / 2 / 4 | 10 / 9 / 8 / 6 / 6 | 0.25 / 0.36 / 0.84 / 0.97 / 1.00 | 0.97 / 0.99 / 0.98 / 1.00 / 1.00 | 0 → 5.5 |
| asym | lr-only | ¼ / 4 | 5 / 6 | 0.82 / 0.86 | 0.99 / 0.99 | 2 / 5 |
| asym | wd-only | ¼ / 4 | 10 / 0 | 0.56 / — | 0.97 / — | 0.5 / 3.5 |

### Verdicts (preregistered)

**P1, shares rise with ρ — supported in both graphs.**
- sym: τ(gen) = 0.89, τ(mem) = 0.94.
- asym: τ(gen) = 0.85, τ(mem) = 0.49.
- Permutation p ≈ 10⁻⁴ for all four.

**P2, co-location persists:**
- **sym: inconclusive.** The trend is significant (τ = −0.36, p = 0.003), but every shift is under 0.5 logit units. Clipping at the extremes compresses it.
- **asym: FALSIFIED as written.** The trend is τ = 0.80, with shifts up to 2.4 logits. Generalization follows ρ (0.25 → 1.00) while memorization stays in the chain branch (0.97–1.00).

**P3, step size alone moves credit:**
- **sym: supported.** In the lr-only arm both shares move toward the faster branch: gen +10/−0 and mem +10/−0 in paired seeds, p = 0.002.
- **asym: untestable** (fewer than 6 eligible pairs).

**Diagnostic** (logit-share effect, ρ = 4 vs ¼):

| | main | lr-only | wd-only |
|---|---|---|---|
| sym | 7.8 | 4.2 | 0.1 |
| asym | 5.0 | 0.3 | — |

**Notes on the verdicts** (from the independent recomputation):
- "Groks" means the first crossing of 0.95, and the generalization window is step > T − 2,000.
- With *sustained* grokking (0.95 held for 2,000 steps, as in E2), 4 asym runs become ineligible. P1 and P2 do not change.
- asym's memorization leg of P1 is statistically positive but negligible in substance: s(mem) is 0.96–1.00 at every ρ, and 25 of 39 runs sit at the clip.
- sym's "inconclusive" rests on the logit clip: at ρ = ¼ and 4 almost every share is clipped, so d is 0 there by construction. With a looser clip ([0.005, 0.995]) the shifts reach 0.5–0.6, borderline "falsified".

### Matched-time analysis (exploratory; corrected after verification)

P2 compares memorization at its peak with generalization at the end: two different times. An exploratory reanalysis compares both shares at the same moment.

**Early checkpoints are uninformative.** At the peak of memorization, total generalization is at or below zero in most runs: memorization pushes unseen inputs toward wrong answers (F4). Its "share" then tracks memorization mechanically. Only later checkpoints are used.

**Gen / mem share of branch 0 at the plateau midpoint** (halfway between peak memorization and grokking):

| cell | gen / mem | median logit gap d |
|---|---|---|
| asym main ρ ¼ | −0.01 / 0.09 | −1.2 |
| asym main ρ ½ | 0.22 / 0.73 | −2.4 |
| asym main ρ 1 | 0.51 / 0.97 | −3.4 |
| asym main ρ 2, 4 | ≈1 / ≈1 | 0 (saturated) |
| asym lr-only ρ ¼ | 0.42 / 0.95 | |
| asym wd-only ρ ¼ | 0.10 / 0.47 | |
| sym lr-only ρ ¼ / 4 | 0.02 / 0.53 and 0.96 / 0.46 | |
| sym main, sym wd-only | coincide (mostly saturated near 0 or 1) | |

**What this shows:**
- **asym is part migration and part a structural split.**
  - At matched times, the *trend* of the gap with ρ mostly disappears (τ ≈ 0.23–0.29, p 0.03–0.10 at the midpoint, three-quarters of the plateau and grokking). The preregistered P2 trend is largely produced by measuring memorization early, when it always sits in the chain, and generalization late.
  - But a large ρ-independent gap remains at ρ ≤ 1, *including ρ = 1, where nothing is manipulated.* Generalization spreads across both branches while memorization stays in the chain.
  - That is a division of labor set by the wiring, not by speed. It is a lead for the original question, measured here only at the branch level and exploratory.
- **In sym, speed alone produces a genuine role split at matched time.** In the lr-only arm at mid-plateau, comparing ρ = 4 with ρ = ¼ in paired seeds:
  - generalization moves to the faster branch: +6.9 logits, 10/0 seeds, p = 0.002;
  - memorization moves slightly to the slower one: −0.25 logits, 0/10, p = 0.002.
  - (In this arm the faster branch also has the shorter memory: its lr·wd is 4× larger.)
  - This contradicts H1's claim that memorization and generalization move together, once both are compared at the same time. The preregistered P3 still holds, because memorization at its early peak did move with speed.
- **The branch with more weight decay carries the grokking jump, then hands it back.** In sym main and wd-only, generalization share swings to the higher-decay branch at the moment of grokking (median 0.83 at ρ ¼, 0.11 at ρ 4), then returns to the other branch within about 1,500 steps.
  - Example run, checked by hand (sym main ρ 4): at the jump, branch 1 carries 13.3 vs branch 0's 2.3; by step 4,500 it is 3.8 vs 22.4.
  - This does not happen in lr-only. So single-time shares near grokking are volatile, and the branch that ends with the credit is not necessarily the one that formed the circuit.

**What E3 says about H1 (credit by speed):**
- **Supported in part.** In the stable wiring, measured as preregistered, speed at equal memory moves both kinds of function toward the faster branch, and steeply: about 2.8 logit units per unit of log ρ, against the lazy prediction of 1.
- **But it is not the whole story:**
  - in sym, speed alone splits roles at matched time (the rule goes to the faster branch first);
  - in asym, the chain takes memorization regardless of nominal speed, and a speed-independent split remains.
  - The chain's *effective* speed (it compounds its updates; F2) may explain part of this. Kernels were not measured.
- So under shared error, credit is not set by step size alone. The wiring and the weight decay of each branch also shape which branch carries what, and when.

**Claim type:** optimization and mechanism, exploratory beyond the preregistered verdicts.

---

## E4 — stability regions

**Design.** 864 runs.
- Graphs pk0000 / pk0100 / pk0120 / pk0123, with longest chain C = 1–4.
- 8 learning rates × 3 weight decays × 3 seeds.
- Three parameterizations: std; P1, a uniform 1/√L output multiplier; P2, a per-block 1/√depth multiplier.
- Stability uses Study 2's window: train accuracy ≥ 0.9 for 6,000 steps after first fit.

**lr\*** (largest stable learning rate) at wd = 1:

| | C1 (pk0000) | C2 (pk0100) | C3 (pk0120) | C4 (pk0123) |
|---|---|---|---|---|
| std | 0.016 | 0.004 | 0.0005 | 0.0005 |
| P1 | 0.032 | 0.008 | 0.0005 | none |
| P2 | 0.016 | 0.008 | 0.001 | 0.0005 |

### Verdicts (preregistered)

- **P0, replicating Study 2 — supported.** Under std, C1's ceiling (0.016, now interior to the extended grid) is far above C4's. C4 is stable in at most one cell, the grid floor 5e-4, under every parameterization and weight decay. So the "5-step gap" really means C4 has no stable region on this grid.
- **H8, a parameterization aligns stability across wirings — NOT supported** under P1 (gap of at least 7 steps) or P2 (gap 5).
  - The uniform multiplier shifts the shallow graphs up and leaves the deep chain unstable.
  - The depth-aware multiplier helps C2–C3 by about one step and C4 not at all.
- *Silencing check:* in stable runs, removing pk0123's deepest block drops test accuracy to 0.01–0.02 under every parameterization. The deep block still carries the function, so the comparison is not hollow.
- *Robustness* (independent recomputation):
  - Almost every "unstable" run fits within 50–200 steps, then has brief drops below 0.9 train accuracy and recovers; no run diverged. The non-monotone maps for C2–C4 come from this collapse noise, combined with the 3-seed, 2-of-3 rule.
  - Four alternative summaries all keep the C1–C4 gap at ≥ 4 grid steps under every parameterization, so P0 and the H8 verdict hold under each:
    - the largest lr below the first unstable one;
    - a tolerant rule (≥ 95% of the window above 0.9);
    - stability over the last 2,000 steps;
    - the fraction of stable runs: C1 0.71 / 0.88 / 0.71 vs C4 0.08 / 0.08 / 0.12 for std / P1 / P2.
  - For C2 and C3, the choice of summary moves lr\* by up to 3 grid steps. Their ordering should be read from the fraction of stable runs, not lr\*.

**What E4 says:**
- Chain depth's instability is not removed by output scaling. The pre-registered falsifier is met: comparisons between wirings, and any search over structures, need per-structure settings, or a stabilizer that changes the model class.
- This strengthens F2: the stability ceiling replicates with a 6,000-step window, an extended grid and 3 parameterizations.

**Claim type:** optimization.

---

## Verification

**E1, E2.** An independent agent recomputed every E1 and E2 number from the raw run files with its own code. It did not read `analyze.py`; it reused only the program-prior module and the arrival schedule.
- **Reproduced:** every number above, exactly or to rounding: code lengths, ratios, k\*, production points, data floor, all leads and sign tests, per-cell savings and Holm p-values, and the noise floor.
- **No verdict changes.** Its caveats are folded in above:
  - asym's floor is an upper bound;
  - sidedness of the sign tests;
  - the noise floor measures only floating-point nondeterminism;
  - censoring in asym;
  - the sym savings are coarse, and many came from jumps made after accuracy had started rising;
  - P1's lead is essentially one eval interval.

**E3, E4.** A second independent recomputation, also without `analyze.py`.
- **Reproduced:** every eligibility count, share, τ, p-value, verdict and lr\*.
- **Robustness:** it checked the sustained-grokking definition, alternative clips, alternative stability summaries and window definitions. No preregistered verdict changes.
- **It corrected the exploratory matched-time reading, now rewritten above:**
  - early checkpoints are uninformative;
  - midpoint agreement is mostly saturation;
  - asym shows a speed-independent split as well as migration;
  - sym lr-only shows a genuine speed-induced split;
  - the higher-decay branch carries the grokking jump.
