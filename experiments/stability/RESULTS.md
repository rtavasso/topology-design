# Results — Study 2: stability ceiling vs. composition depth

**Run:** 20 Oct 2026, natively on the researcher's MacBook Pro (10 workers, commit `e45c0df`): 504 runs, 0 errors. Scored against [PREREGISTRATION.md](PREREGISTRATION.md), which was committed before any runs. Data: [`results_data/compact.json`](results_data/compact.json) holds per-run summaries; the full trajectories stay on the Mac. The automatic report is [`results_data/summary.md`](results_data/summary.md).

**Independent check:** see *Verification* at the end.

## Headline

**All four preregistered predictions are supported.** Across all 24 four-block parent-selection graphs, the longer a graph's longest composition chain $C$, the lower the learning rate at which it trains stably.

| Prediction | Result |
|---|---|
| **P1** τ_b(C, log lr*) < 0, permutation p < 0.05 | **τ_b = −0.836, p < 1e-4. Supported.** |
| **P2** lr*(pk0123) ≤ lr*(pk0000)/4 | **pk0000: 2.4e-2. pk0123: below the grid (< 1.5e-3), so the gap is > 16×. Supported.** |
| **P3** chain length, not root readers $Q$, carries the effect | **Within C = 2, τ(Q, log lr*) = 0.00. Rank-regression coefficient of C = −0.95 [95% CI −1.31, −0.71]. Passes formally, but uninformative.** Within C = 3 it passes only vacuously, and the continuous endpoint shows a secondary Q effect (caveat 2). |
| **P4** ordering robust at weight decay 0.1 | **Spearman ρ = 0.82. Passes narrowly.** Three of the four values are tied at the floor (caveat 4). |

**In short:** the direction is robust. Longer chains have lower ceilings under every definition of lr* tried. **Chain length is the dominant descriptor, but not the only one.** The evidence does not support "C, not Q."

## Robustness: what the instability actually is

**The failure mode is collapse after fitting.** Across all 504 runs:
- no run diverged numerically;
- only 3 runs failed to fit (C = 3 at lr 4.8e-2);
- every other unstable run fit the training set, then repeatedly crashed and recovered.

These are transient dips, not a failure to learn: unstable runs have median final train accuracy 1.0 and median peak test accuracy 0.997.

The number of such crashes in 6,000 steps grows with C (medians over graphs and seeds, weight decay 1.0):

| C | lr 1.5e-3 | lr 3e-3 |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 0 | 0 |
| 3 | 1 | 13 |
| 4 | 16 | 21 |

**A continuous endpoint tells the same story without the noisy pass/fail threshold.** Median minimum train accuracy after the first fit, weight decay 1.0 (exploratory, not preregistered):

| C | 1.5e-3 | 3e-3 | 6e-3 | 1.2e-2 | 2.4e-2 | 4.8e-2 |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 0.79 | 1.00 | 0.97 | 0.94 | 0.14 |
| 2 | 0.93 | 0.92 | 0.80 | 0.16 | 0.04 | 0.02 |
| 3 | 0.53 | 0.17 | 0.09 | 0.05 | 0.03 | 0.02 |
| 4 | 0.16 | 0.04 | 0.04 | 0.02 | 0.03 | 0.01 |

At every lr, the order is C1 ≥ C2 > C3 > C4. The single exception is the noisy C=1 cell at 3e-3.

**Instability does not prevent generalization.** At lr 3e-3, 14 of 27 C = 3 runs still reach ≥ 0.99 test accuracy at some point, despite a median of 13 crashes. The ceiling limits how *smoothly* a graph can be trained at a given lr, not whether it can generalize.

**Root readers matter secondarily on the continuous endpoint.** Average each graph's post-fit minimum over the lr grid:
- Within C = 2: τ(Q) = +0.42 (p = 0.07).
- Within C = 3: τ(Q) = +0.45 (p = 0.14).
- Rank regression: β_C = −0.75 [−0.97, −0.52], β_Q = +0.34 [0.04, 0.59].

More blocks reading the embedding directly helps a little, holding chain length fixed. The C effect is much larger: the C = 2 graph means are 0.36–0.56, against 0.11–0.27 for C = 3. The binary lr* is too coarse to see the Q effect.

## Caveats

1. **Floor effect.** All C = 3 and C = 4 graphs are unstable at the lowest grid lr (1.5e-3), so their lr* is "below grid" and the size of the gap is only a lower bound. The next grid should extend down to ~1e-4.
2. **P3 cannot separate C from Q here.** Within C = 3 every graph is at the floor, so the within-group test passes vacuously. Within C = 2 the binary lr* shows no Q dependence, but the continuous endpoint does (above). C and Q are also correlated by construction (τ = −0.64), and C = 1 and C = 4 are single graphs. pk0000 drives much of the overall τ.
3. **The binary criterion is noisy at 3 seeds.**
   - 7 of 24 graphs are unstable at some lr below one where they are stable. Example: pk0000 is stable in 3, 1, 2, 3, 2, 0 of 3 seeds across the grid.
   - τ depends on how lr* is defined, though its sign never changes: −0.84 (≥2/3 seeds, preregistered), −0.80 (≥1/3), −0.60 (3/3), −0.61 (stable at all lower lrs too).
   - About 10% of runs fall within ±0.1 of the 0.9 threshold.
4. **"Stable" partly means "has not grokked yet."** At lr 1.5e-3, only 9% of stable runs grokked within 6,000 steps, and none of the C = 2 runs did. At 3e-3, 61% did. A slow run that is still memorizing never meets the post-grok dynamics where many collapses happen. So low-lr stability partly reflects how far training got. A confirmatory run should compare stability at matched training progress, or train long enough for every run to grok.
5. **Weight decay matters, and P4 is weak evidence.** At weight decay 0.1, every lr* falls: the parallel graph drops from 2.4e-2 to 3e-3, and the other three are below the grid. With three of four tied, ρ = 0.82 is effectively one comparison. AdamW's lr·wd coupling is not separated from the step-size explanation.
6. **Scope.** One task (modular addition, p = 59), one optimizer (AdamW), one width, 3 seeds. Transfer to another task is required before calling this a design principle (SPEC Gate 4).

## Interpretation

This is the program's **first preregistered, out-of-sample link from a graph descriptor to a learning-dynamics outcome.** Longest composition chain predicts the stable learning-rate ceiling across all 24 graphs in the family. Root readers add a smaller, opposite-signed effect.

This is a strong lead, not yet a confirmed result. The floor, the 3 seeds and the grokking-progress confound all need to be removed first.

The direction is what Phase 0 Prop. 3 predicts. In a chain, the effective step on the end-to-end map is scaled by the current singular values, so long chains amplify updates as weights grow and destabilize sooner. The data are consistent with this mechanism but do not test it directly.

**A direct test**, the natural next step, is to measure the effective step size or sharpness along training (e.g., the top Hessian eigenvalue, or the norm of the end-to-end update per unit lr) and check that it scales with C as Prop. 3 predicts.

It also reframes Study 1. The sequential chain's poor grokking there is plausibly this stability ceiling, which blocks it from the fast, strongly regularized regime, rather than memorization interference.

## Next steps

1. **Mechanism test:** track sharpness and effective step size against C on a subset of graphs.
2. **Confirmatory run:**
   - 10 seeds;
   - an lr grid from 1e-4 to 4.8e-2, at factor-√2 spacing near each graph's ceiling;
   - the continuous endpoint as a preregistered co-primary;
   - Q tested as its own predictor, not a nuisance;
   - stability scored at matched training progress (or long enough training that every run groks);
   - a small 2-D lr × wd grid to separate step size from decay.
3. **Transfer:** repeat on the composition task once it is calibrated.

## Verification

An independent agent recomputed every number from `compact.json` with its own code, without seeing this write-up. Its scripts were not committed.
- **Reproduced:** graph descriptors, all lr* values, all four verdicts, and permutation p = 1/10001.
- **Its findings, all incorporated above:**
  - P3 is uninformative because of the floor;
  - P4 rests on one comparison;
  - the "stable = not yet grokked" confound;
  - lr* is sensitive to its definition;
  - the secondary Q effect on the continuous endpoint, which I re-checked: within-C τ(Q) = +0.42 and +0.45.

These change the headline from "C, not Q" to "C dominant, Q secondary."
