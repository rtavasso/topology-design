# Diagnosing the walk — results

Scored against [PREREGISTRATION.md](PREREGISTRATION.md), committed before any run, with the analysis code ([`analyze.py`](analyze.py)) also committed first.

**Status**

| | |
|---|---|
| E1, E2 | done (23 Oct 2026). |
| E3, E4 | running on Kaggle; sections pending. |

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
  - The data floor, even with up to 60,000 steps, is 1,044 examples for sym (348× k\*) and 1,392 for asym (464× k\*). At 870 examples, only 1 of 3 sym seeds groks.
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
- Leads were consistent but tiny: +13/−1, sign test p = 0.002.
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
- The noise floor is 0: the H = 0 arms match their bases.
- The saving is roughly 0.1–0.4 of the horizon jumped. At s = 2,000 with H = 1,000, test accuracy went from 0.06 to 0.44 at the moment of the jump.
- **asym:** median savings of 1,700–5,900 steps (larger than H in some cells), but inconsistent across seeds (Holm p 0.09–1.0). asym's grokking time is chaotic because of repeated collapses.

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

## Verification

Pending (independent recomputation from the raw files).
