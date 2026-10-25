# The "priors" round — preregistration

**Status:** written 24 Oct 2026, before any main run. The analysis code ([`analyze.py`](analyze.py)) and grids ([`make_grids.py`](make_grids.py)) are committed with this file. A CPU pilot section is appended, dated, before launch. Changes after the independent review are listed at the end, also dated.

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
- **Ungrokking runs** (grid `q1u`, 112 runs):
  - train on the 40% set (1,392 pairs) until test accuracy has held ≥ 0.95 at every eval for 2,000 steps;
  - then train only on the subset at f ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35}, i.e. 174 … 1,218 pairs;
  - the post-shrink phase lasts 30,000 steps for sym (60 weight-decay time constants 1/(lr·wd)) and 40,000 for asym (40);
  - same learning rate and weight decay throughout; 4 seeds;
  - sym at three widths, asym at base width.
  - Phase 1 is capped (sym 13,000 steps, asym 53,000). A run that never reaches sustained grokking is not shrunk and is excluded, with the count reported.
- **Scratch runs** (grid `q1s`, 198 runs):
  - train from initialization on each subset;
  - sym f ∈ {0.1, …, 0.35} for 60,000 steps; asym f ∈ {0.25, 0.3, 0.35, 0.4} for 90,000 steps;
  - three learning rates (½×, 1×, 2× the graph's base rate), 3 seeds;
  - same widths as above;
  - a run stops 2,000 steps after first reaching 0.95 test accuracy.

**Definitions** (computed by `analyze.py q1`)
- **Final test accuracy** of an ungrokking run: the mean test accuracy over the last 5,000 steps of the post-shrink phase.
  - Not converged: the means of the two halves of that window differ by more than 0.1.
  - A cell where more than a quarter of runs are unconverged is flagged.
- **n_crit** (per graph and width): the smallest subset level at which the median final test accuracy is ≥ 0.5 there *and at every larger level*.
  - If that is the smallest level, n_crit is censored below (≤ 174).
  - If no level qualifies, it is censored above (> 1,218).
- **n_floor** (per graph and width): for each learning rate, the smallest level at which at least 2 of 3 seeds grok (first crossing of 0.95, as in E1) there and at every larger level. n_floor is the minimum over the three learning rates: the best search we tried. Censoring is as for n_crit.
- **S = n_floor / n_crit:** the factor attributable to search.
- **Search share = log S / log(n_floor / k\*)**, with k\* = 3: the fraction of the whole data gap (on a log scale) attributable to search. The rest, log(n_crit / 3), is attributed to the prior: what AdamW, weight decay and this architecture prefer at equilibrium.

**Predictions and falsifiers**

| | prediction | falsified if |
|---|---|---|
| **P1** (replication: ungrokking exists here) | sym base: median final test accuracy < 0.5 at 174 pairs and ≥ 0.9 at 1,218 | either fails, e.g. the rule survives on 174 pairs |
| **P2** (main: the gap is mostly prior) | sym base: 1.5 ≤ S ≤ 3 (Varma: ≈ 2) **and** search share < 0.25 | S > 3 or S < 1.5, or share ≥ 0.25. If a threshold is censored at the grid edge, the verdict is *untestable* |
| **P3** (wiring acts on search) | at base width, asym's floor is above sym's (replicating E1) and n_crit(asym)/n_crit(sym) < n_floor(asym)/n_floor(sym): the chain's extra data cost is mostly search | the ratio of n_crit is ≥ the ratio of n_floor. Untestable if asym's floor is not above sym's, or a threshold is censored |
| **P4** (width; a contest, prediction from prior art) | n_crit ordering wide ≤ base ≤ narrow, not all equal (Huang et al.: larger models generalize from less data) | the opposite ordering ("narrower generalizes from less data", our 24 Oct conjecture), flat, or non-monotone |
| **P4b** (generic width cannot close the gap) | at every width, n_floor > 348 pairs (0.1) | some width groks from scratch at ≤ 348 pairs, i.e. ≥ 3× below base |

Every prediction can fail on data this design can produce (lesson L12):
- P1 fails if ungrokking does not happen;
- P2 fails if search accounts for a large factor (the distillation result above suggests it might);
- P3 and P4 are directional on quantities measured independently;
- P4b fails if a narrow network groks at 0.1.

**What each outcome changes**
- **P2 supported** (prior dominates): no amount of better search inside this model class closes most of the gap. Member efficiency has to come from different priors: structure matched to the task, or a library of blocks grown across tasks (I19). The credit machinery (I20) becomes secondary. The "masking is explaining away under a prior that undercharges lookup tables" reading (H3 note) gains support.
- **S > 3** (search matters): improving the search within gradient learning is a real lever, and credit between members (I20) and the growth criterion (I21) move up.
- **P4 in the C direction:** generic restriction by width is the wrong kind of restriction; it hurts the rule's efficiency more than memorization's. The R direction would make width a usable dial for member efficiency.
- **P4b falsified:** generic restriction is a strong lever after all, which revises the claim that efficient members need structure matched to the task.

**Exploratory, labelled as such in the summary:**
- accuracy on the dropped pairs (seen in phase 1, then removed);
- weight norms at the end;
- semi-grokked runs (final test accuracy 0.1–0.9), which Varma reports near the threshold;
- collapses after the shrink;
- scratch thresholds by the sustained-grokking criterion.

## Q2 — does the drift's direction matter? (I13)

**Design** (grid `q2`, 90 runs): E2's four sym jump cells (jump at 1,000 or 2,000 steps; H = 250 or 1,000) with E2's seeds (610–619).
- Each seed has a base run, a drift jump (as in E2), and a **random jump**. The random jump replaces each parameter tensor's jump vector by a Gaussian direction of the same norm, per run and per tensor (tested).
- 15,000 steps, as E2. The base runs also replicate E2's.

**Outcome.** The sustained grokking time (≥ 0.95 held for 2,000 steps; E2's definition and code). Per seed, D = t(random) − t(drift).
- Pairs whose base had already grokked (sustained) before the jump are excluded, as in E2.
- So are pairs where both jump arms are censored at 15,000 steps.

**Prediction P5:** the drift direction matters. D > 0 by a one-sided sign test with Holm-adjusted p < 0.05 in at least 3 of the 4 cells, and pooled median D > 0. Falsified if the pooled median D ≤ 0, or at most 1 cell is significant; 2 significant cells is inconclusive. Sidedness is stated (L13): one-sided, drift better.

**What it changes:**
- **Supported:** F16 stands as "the drift direction is informative."
- **Falsified:** F16 is demoted to "a kick of this size helps", and the weight-space horizon (already short) loses its main positive result.
- Exploratory: random jump vs base (does any kick help?), and whether E2's drift savings replicate on the same seeds.

## Caveats stated in advance

- **The "prior" measured here is the training dynamics' preference at equilibrium** (AdamW with decoupled weight decay, this architecture), not a Bayesian prior. Varma reports that the ungrokking threshold doesn't depend on weight decay; we don't vary it.
- **Level resolution is coarse:** steps of 174 pairs, so ratios like S come in coarse steps.
- **Few seeds:** 4 per level for n_crit, and 3 per learning rate for n_floor. These are threshold estimates, not effect sizes.
- **A finite post-shrink phase:** a slow ungrokking could be missed. That's what the convergence flag is for.
- **"Best of three learning rates"** is a modest search over settings. A better optimizer could lower n_floor, which would *raise* S. P2 is stated for the search we actually ran.
- **The program prior** is one reference learner (F13's scope).
- **Scope:** one task, two wirings, three widths.

## Compute

| job | runs | rough size |
|---|---|---|
| `q1u` | 112 | ~5M run-steps |
| `q1s` | 198 | ~9M run-steps (many small batches) |
| `q2` | 90 | ~1.4M run-steps |

Expected to fit in three Kaggle jobs of under 12 hours each on 2× T4.

---

## Pilot (CPU, before launch)

*To be appended before launch.*

## Changes after the independent review

*To be appended before launch.*
