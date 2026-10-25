# The "priors" round — preregistration

**Status:** written 24 Oct 2026, before any main run, and revised the same day after an independent pre-run review (changes listed at the end). The analysis code ([`analyze.py`](analyze.py)) and grids ([`make_grids.py`](make_grids.py)) are committed with this file and implement the revised rules. A CPU pilot section is appended, dated, before launch.

**Where this comes from:**
- the discussion after E1–E4 ([history Entry 26](../../docs/history.md));
- ledger I17 (prior or search), I18 (member efficiency), I13 (random-kick control) and the H3 note (masking as explaining away).

---

## Why

E1 found that gradient descent needs about 1,044 examples to learn modular addition where a program-prior learner needs 3 (F13). The discussion turned on one question: **is that gap the network's prior or its search?**

- **Prior:** the regularized objective really does prefer memorizing below about 1,000 examples, because a lookup table of a few hundred pairs costs less than the rule. If so, better search (credit routing, competing members, cleverer optimizers) cannot close the gap; only a different prior can.
- **Search:** the rule is already the preferred solution at much smaller sizes, but training from scratch doesn't find it. If so, improving the search inside gradient learning is worth pursuing.

There's a clean way to separate them. Start from the rule, already learned on 40% of the pairs, and keep training on a smaller subset. If the rule survives on a subset of size n, the objective prefers it there (or at least holds onto it), so failing to learn it from scratch at that n is a search failure. If the network drifts back to memorizing ("ungrokking"), the objective prefers memorizing at that n, which is the prior.

A second, smaller question closes a hole left by E2: the drift jumps saved steps, but there was no control kick of the same size in a random direction (L13, I13).

## Prior art (checked before design, lesson L2)

- **Varma et al. 2023, [Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390).** This is our Q1 protocol on a one-layer transformer (p = 113, AdamW, wd 1).
  - A grokked network trained on a smaller dataset *ungroks* below a critical size.
  - The transition is sharp, and it doesn't depend on weight decay.
  - The ungrokking threshold (≈ 800–1,000 of 12,769 pairs) sits about **2× below** the from-scratch threshold (≈ 1,500–2,050).
  - They did not vary model size or architecture.
- **Huang et al. 2024, [Unified View of Grokking, Double Descent and Emergent Abilities](https://arxiv.org/abs/2402.15175).**
  - The critical dataset size is *negatively* correlated with model size: larger models generalize from less data.
  - Memorization capacity grows with size.
- **Song & Ye 2026, [Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds](https://arxiv.org/abs/2605.09724).** Grokking onset across model sizes is predicted by when the memorization and generalization timescales cross: a search-order account. Not a threshold-on-data study.
- **Singh et al. 2025, [When Data Falls Short](https://arxiv.org/html/2511.04760v1).** Distillation from a grokked teacher makes a student generalize below the from-scratch threshold (25% → 20%). That's another sign that the from-scratch threshold is partly a search limit.

**What is new here:**
1. The prior/search decomposition set against an ideal-learner reference (k\* = 3), giving how much of the whole gap each accounts for.
2. Wiring dependence: does the chain-plus-shortcut graph's higher threshold (E1) come from its prior or its search?
3. Width dependence of *both* thresholds in one design, as a test of whether generic restriction buys data efficiency at the level of a member (I18). The prior art disagrees in direction.
4. Our residual-MLP setting.

## Setting (shared)

- Modular addition, p = 59, 3,481 pairs, AdamW, weight decay 1, softmax cross-entropy, full batch.
- The fleet trainer.
- Graphs as in E1–E3: sym = `pk000000` (all six blocks read the embedding), asym = `pk012000` (a chain of three plus shortcuts).
- Learning rates as in E1–E3: sym 2e-3, asym 1e-3.
- Widths (d, hidden):

| width | (d, hidden) | parameters | scope |
|---|---|---|---|
| narrow | (32, 64) | 30k | sym only |
| base | (128, 256) | 416k | sym and asym |
| wide | (256, 512) | 1.6M | sym only |

- Subsets are nested. `gk.split` takes a prefix of one permutation per data seed, so the subset at fraction f is exactly the training set a scratch run at `train_frac = f` sees, and it lies inside the 40% set. Ungrokking runs and scratch runs at the same f train on **identical pairs** (tested).

---

## Q1 — prior or search

**Design**
- **Ungrokking runs** (grid `q1u`, 176 runs):
  - train on the 40% set (1,392 pairs) until test accuracy has held ≥ 0.95 at every eval for 2,000 steps;
  - then train only on the subset at f ∈ {0.0125, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35}, i.e. 44 … 1,218 pairs;
  - the post-shrink phase lasts 30,000 steps for sym (60 weight-decay time constants 1/(lr·wd)) and 40,000 for asym (40);
  - same learning rate and weight decay throughout; 4 seeds;
  - sym at three widths, asym at base width.
  - A fixed learning rate is defensible on this side: what is measured is where AdamW with weight decay settles, not how fast it gets there.
  - **Deadline:** a run must start its shrink by phase-1 cap + 2,000 steps (sym 15,000; asym 72,000), or it stops and is excluded. The count is reported. One excluded seed removes all of that seed's levels.
- **Scratch runs** (grid `q1s`, 225 runs):
  - train from initialization on each subset;
  - sym f ∈ {0.1, …, 0.4} for 60,000 steps; asym f ∈ {0.25, 0.3, 0.35, 0.4} for 90,000 steps;
  - learning rates: sym 2e-3, 4e-3, 8e-3 and 1.6e-2 (the top of the all-parallel graph's stable range in E4, added so the best rate can be interior); asym 1e-3, 2e-3 and 4e-3 (already past its stability limit); 3 seeds;
  - same widths as above;
  - a run stops 5,000 steps after first reaching 0.95, so the sustained criterion can be read.

**Definitions** (computed by `analyze.py q1`). One matched criterion on both sides: **test accuracy ≥ 0.95 at every eval for 2,000 steps.**
- **An ungrokking run keeps the rule** if it meets the criterion over the last 2,000 steps of the post-shrink phase.
  - A run that diverges after the shrink does not keep the rule.
  - Not converged: the mean test accuracy over the last 2,500 steps differs by more than 0.1 from the 2,500 steps ending 15,000 steps earlier. A cell with more than a quarter of runs unconverged is flagged.
- **n_crit** (per graph and width): the smallest level at which at least half of the shrunk seeds keep the rule, there *and at every larger level*.
  - If that is the smallest level (44 pairs), n_crit is censored below.
  - If no level qualifies, it is censored above.
- **A scratch run succeeds** if its sustained onset t (the criterion, starting at t) satisfies t + 2,000 ≤ the step budget.
- **n_floor** (per graph and width): for each learning rate, the smallest level at which at least 2 of 3 seeds succeed, there and at every larger level. n_floor is the minimum over the learning rates.
  - **Flags**, which make P2–P4b *provisional* for that graph and width:
    - the best learning rate is the top of the grid;
    - n_floor at half the budget differs from n_floor at the full budget, i.e. the threshold is still moving with steps.
- **S = n_floor / n_crit:** the factor attributable to search.
- **Search share = log S / log(n_floor / k\*)**, with k\* = 3, is **reported as a measurement, not a verdict**. The review showed that a share threshold could not fail on this grid: the log scale anchored at k\* = 3 makes "mostly prior" nearly true by construction.

**Predictions and falsifiers**

| | prediction | falsified if |
|---|---|---|
| **P1** (replication: ungrokking exists here) | sym base: fewer than half the seeds keep the rule at 174 pairs, at least three quarters at 1,218 | either fails |
| **P2** (search accounts for at most ~3×) | sym base: S ≤ 3 | S ≥ 4. 3 < S < 4 is inconclusive; a censored threshold is untestable. The largest uncensored S is 1,392/87 ≈ 16 (44 pairs is the bottom level, so a threshold there counts as censored) |
| *Replication check* | S ≥ 1.5 (Varma: ≈ 2) | reported, not a prediction of ours: S near 1 would mean *less* search than in the transformer |
| **P3** (wiring acts on search) | at base width, asym's floor is above sym's (replicating E1) and n_crit(asym)/n_crit(sym) < n_floor(asym)/n_floor(sym) | the ratio of n_crit exceeds the ratio of n_floor. A tie is inconclusive; untestable if asym's floor isn't above sym's or a threshold is censored |
| **P4** (width; a contest, prediction from prior art) | n_crit ordering wide ≤ base ≤ narrow with at least one strict step (Huang et al.) | any other result: the opposite ordering (our 24 Oct conjecture), flat, or non-monotone. Censored thresholds count when the censoring still fixes the order |
| **P4b** (generic width cannot close the gap) | at every width, n_floor > 348 pairs (0.1) | some width groks from scratch at ≤ 348 pairs, i.e. ≥ 3× below base |

**P1 and P2 are coupled** (noted by the second review). If P1 passes, n_crit ≥ 261, so falsifying P2 needs n_floor ≥ 1,044. If the extended learning-rate search brings the floor down to 870 or below, P2 can be falsified only if P1 fails too (the rule survives on 174 pairs). That is a real possible outcome, so P2 can still fail, but the two verdicts should be read together.

Every prediction can fail on data this design can produce (L12):
- P1 if ungrokking doesn't happen;
- P2 if training from the rule holds it down to ≤ 1/4 of the scratch floor (the distillation result above suggests it might);
- P3 and P4 are directional on independently measured quantities;
- P4b if a narrow network groks at 0.1.

**Known biases, stated in advance**
- **Biases S down** (towards P2): the minimum over three learning rates on the scratch side, with no counterpart on the ungrokking side.
- **Biases S up:**
  - "at least half of 4 seeds" for n_crit against "2 of 3" for n_floor;
  - scratch runs stop 5,000 steps after first crossing, so a scratch run that later loses the rule still counts as a success, while an ungrokking run must hold at the end;
  - slow ungrokking not finished within the post-shrink phase. That's what the convergence flag is for.
- **Sensitivity:** the summary also reports S under the original, mismatched criteria (median final accuracy ≥ 0.5; first crossing of 0.95), and n_floor at half the budget.

**What each outcome changes**
- **P2 supported** (search ≤ 3× of a gap of several hundred): better search inside this model class cannot close most of the gap. Member efficiency has to come from different priors: structure matched to the task, or a library of blocks grown across tasks (I19). The credit machinery (I20) becomes secondary. The H3 note's reading gains support: masking as explaining away, under a prior that charges too little for lookup tables.
- **P2 falsified** (S ≥ 4): improving the search within gradient learning is a real lever, and I20/I21 move up.
- **P4 in the C direction:** generic restriction by width is the wrong kind of restriction. The R direction would make width a usable dial for member efficiency.
- **P4b falsified:** generic restriction is a strong lever after all, which revises the claim that efficient members need structure matched to the task.

**Exploratory, labelled as such in the summary:**
- accuracy on the dropped pairs;
- weight norm and rule fraction at the shrink (the shrink comes only about 4 time constants after sustained grokking, so some memorization may remain), and n_crit's sensitivity to them;
- semi-grokked runs (final 0.1–0.9);
- collapses after the shrink.

## Q2 — does the drift's direction matter? (I13)

**Design** (grid `q2`, 170 runs): E2's four sym jump cells (jump at 1,000 or 2,000 steps; H = 250 or 1,000) with E2's seeds (610–619).

Each seed has a base run, the drift jump (as in E2), and three null jumps with the same norm per tensor:
- **radial** (the primary null): keeps the drift's component along the current weights, and replaces the orthogonal part with a random direction orthogonal to the weights, of the same norm. The review found the drift is largely a per-tensor *rescaling* (at step 1,000 its cosine with θ is about +0.58). Weight norm is a known lever on grokking (Omnigrok), so a pure random direction would differ from the drift on that lever. The radial null matches it.
- **random:** a random direction of the same norm (secondary).
- **rescale:** the drift's radial component alone (secondary; *not* norm-matched, since its jump is smaller). Does norm reallocation by itself account for the saving?

Matching is per whole tensor. How the drift moves norm between rows or neurons *within* a tensor is not matched; the radial null's orthogonal part is isotropic. P5 is scoped accordingly: "beyond per-tensor norm changes".

Each cell and seed draws its own direction. Evals run every 25 steps up to step 6,000, so savings are resolved finer than E2's 100-step evals. Runs last 15,000 steps, as in E2.

**Outcome.** Sustained grokking time (≥ 0.95 held for 2,000 steps; E2's definition and code). Per seed, D = t(null) − t(drift).
- Pairs whose base had grokked (sustained) before the jump are excluded, as in E2.
- So are pairs where both arms are censored.

**Prediction P5** (drift vs radial null): the drift direction matters beyond norm changes.
- **Supported:** D > 0 by a one-sided sign test with Holm-adjusted p < 0.05 in at least 3 of the 4 cells, and pooled median D > 0.
- **Falsified:** only if pooled median D ≤ 0.
- Otherwise inconclusive, so low power doesn't read as a negative.

**What it changes:**
- **Supported:** F16 stands as "the drift direction is informative."
- **Falsified:** F16 is demoted to "a kick, or a norm change, of this size helps." The rescale arm then tells which.

## Caveats stated in advance

- **The "prior" measured here is the training dynamics' preference at equilibrium** (AdamW with decoupled weight decay, this architecture), not a Bayesian prior. Varma reports that the ungrokking threshold doesn't depend on weight decay; we don't vary it.
- **Level resolution is coarse:** 11 unevenly spaced levels from 44 to 1,218 pairs, so ratios like S come in coarse steps.
- **Few seeds:** 4 per level for n_crit, and 3 per learning rate for n_floor. These are threshold estimates, not effect sizes.
- **A finite post-shrink phase:** a slow ungrokking could be missed. That's what the convergence flag is for.
- **"Best of three learning rates"** is a modest search over settings. A better optimizer could lower n_floor, which would *raise* S. P2 is stated for the search we actually ran.
- **The program prior** is one reference learner (F13's scope).
- **Scope:** one task, two wirings, three widths.

## Compute

| job | runs | rough size |
|---|---|---|
| `q1u` | 176 | ~8M run-steps |
| `q1s` | 288 | ~15M run-steps (many small batches) |
| `q2` | 170 | ~2.6M run-steps |

Expected to fit in three Kaggle jobs on 2× T4. A job cut by the 12-hour cap resumes from its saved results when relaunched (the queue does this automatically).

---

## Pilot (CPU, before launch; 24 Oct 2026; not evidence)

Single seed (700), CPU, a shorter post-shrink phase (16,000 steps).

**Phase 1 at 40%, sym:**

| width | lr | first crossing of 0.95 | collapses |
|---|---|---|---|
| narrow (32, 64) | 1e-3 | 14,600 | 0 |
| narrow (32, 64) | 2e-3 | 3,900 | 0 |
| narrow (32, 64) | 4e-3 | 1,800 | 0 |
| base | 2e-3 | 2,500 | 0 |
| wide (256, 512) | 1e-3 | 4,600 | 0 |
| wide (256, 512) | 2e-3 | 2,300 | 0 |

All widths learn the rule at the base rate, well inside the 13,000-step phase-1 cap. The reviewer's own narrow check gave 3.9k and 8.7k at 2e-3.

**Ungrokking, sym base** (shrink at step 4,500). Test accuracy 2,000, 6,000 and 16,000 steps after the shrink:

| pairs kept | +2,000 | +6,000 | +16,000 |
|---|---|---|---|
| 174 | 0.00 | 0.01 | 0.01 |
| 522 | 0.95 | 0.96 | 0.88 (slowly drifting, 0.92–0.96 in between) |
| 1,044 | 1.00 | 1.00 | 1.00 |

- **The planted-effect check passes (L1).** Ungrokking happens, fast, at small subsets, and doesn't at large ones.
- **Near the threshold the network semi-groks and drifts slowly.** The matched criterion (≥ 0.95 throughout the last 2,000 steps) and the 15,000-step convergence comparison are therefore both needed. The full 30,000-step phase is kept.
- **The dropped pairs** (seen in phase 1, then removed) track test accuracy exactly, as at 174 pairs: they are held by the rule, not by memory. Exploratory.
- This single seed hints that n_crit for sym base may lie between 522 and 1,044, close to the E1 floor (1,044). If that holds, S could come out near 1–2, possibly below Varma's 2. Both P2 outcomes, and the replication check, remain open. Nothing in the design changes.

## Changes after the independent review (24 Oct 2026, before any main run)

The review ran CPU checks: shrink logic under batching was bit-identical to solo runs, and the subsets are exactly nested. It found four blocking problems, all fixed:
1. **P2 could not fail in the intended direction.** The search-share threshold was unreachable without censoring. P2 is restated on S (≤ 3 supported, ≥ 4 falsified), the share is reported as a measurement, ungrokking levels go down to 44 pairs, and "S < 1.5" is split out as a replication check.
2. **The Q2 null did not match the drift's radial component.** The radial-matched null is added as the primary comparison, with random and rescale-only as secondaries.
3. **n_floor risked measuring a step budget, with the best learning rate at the grid's top edge.** Scratch learning rates now go up to 4×. There are flags for a best learning rate at the edge and for a floor that moves between half and full budget; either makes the verdicts provisional.
4. **The two sides used different success criteria,** and the scratch stop rule made a matched criterion impossible afterwards. Both sides now use "≥ 0.95 held for 2,000 steps", scratch runs continue 5,000 steps after first crossing, and the mismatched pair is kept as a sensitivity check.

Non-blocking suggestions adopted:
- convergence judged against accuracy 15,000 steps earlier;
- intermediate subset levels (0.075, 0.125);
- order-preserving censoring in P4, and ties in P3 called inconclusive;
- 0.4 added to sym scratch;
- the min-over-learning-rate bias stated;
- dense evals around Q2's jumps;
- P5 falsified only by a non-positive pooled median;
- a separate random direction per cell;
- a shrink deadline (no wasted steps, and the exclusion count is correct);
- post-shrink divergence counted as losing the rule;
- asym phase-1 cap raised to 70,000;
- rule fraction and weight norm logged at the shrink.

Not adopted: fresh seeds to confirm the best-learning-rate floor (cost). The bias it would correct is stated instead.

**Second review (of the fixes, 24 Oct):**
- No blocking issues. P2 reaches all outcomes without censoring; the radial null matches the drift per tensor to about 1e-6 on a real jump (total norm after the jump: drift 95.11, radial 95.12, random 77.35); the matched criteria and the new code are consistent.
- Adopted:
  - corrected the maximum uncensored S (16, not 28);
  - stated the P1–P2 coupling;
  - stated that rescale is not norm-matched, and that matching is per tensor;
  - stated the scratch stop-rule asymmetry;
  - added sym learning rate 1.6e-2, so the best rate is less likely to sit at the grid's edge.
