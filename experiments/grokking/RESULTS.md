# Results — topology vs memorization interference in grokking (study 1)

**Run:** 20 Oct 2026, natively on the researcher's MacBook Pro (12 CPUs, 10 workers, torch 2.14.1, commit `662af2d`).
- 96 LR-tuning runs and 190 main runs, unbatched CPU trainer.
- Scored against [PREREGISTRATION.md](PREREGISTRATION.md) (amendments A1–A2 written before these runs).
- Figures: [`results_figures/`](results_figures/).
- Raw run files stay on the Mac in `results/grokking/` (26 MB, not in git); a compressed copy is in `results/grokking/all_results.tgz`.

**Independent check.** The verdicts below were recomputed from the raw files by a separate reviewer using independent code. It reproduced every number and led to two downgrades; see the *Verification* section.

> **Status under the preregistered rules: nothing here is claimable yet** (see also the A3 follow-up below). Every separable contrast awaits the 29-seed escalation, and the learning rates are at the grid boundary. Read the results below as preliminary.

## Headline

1. **The sink idea failed in this task, cleanly.**
   - A side branch with a fixed multiplier did **not** soak up memorization.
   - Memorization formed first in the *chain*. The sink, once strong enough to matter, ended up hosting more of the *rule* than of the memorization.
   - The localization index was never positive and was negative at s = 4.
   - **H_U is rejected.**
2. **A larger sink multiplier means faster grokking, but this may be a tuning artifact (H_mono only partially supported).**
   - Delay falls monotonically, though not strictly, as s rises: 6,500 → 5,875 → 5,875 → 4,375 → 3,500 → 3,375 steps for s = 0 → 4, at the tuned lr.
   - Per-seed Kendall τ(s, delay) = −0.69 [−0.82, −0.53].
   - But at s = 4 the sink's share of the final rule energy is **bimodal across seeds** (mean 0.57 [0.31, 0.82]; 6/10 seeds above 0.5). So H_mono's second criterion is borderline.
   - With AdamW, a branch multiplier acts like a per-branch learning-rate boost. All arms are tuned at the top of the LR grid, so they are probably under-tuned, and larger s may simply be compensating for too low a learning rate.
3. **The parallel network groks fastest by a wide margin.**
   - Median delay is 1,875 steps, against 6,500 for the sequential residual chain.
   - P(par faster than seq) = 1.00 in all 10 paired seeds.
   - In this task, routes that read the embedding directly host the generalizing circuit fastest. Composition is a cost, not a help.
   - This is consistent with the limitation flagged in the preregistration: the modular-addition circuit is shallow.
4. **H_depth (chain groks without weight decay) is untestable at this budget.** No arm grokked in 20k steps at weight decay 0.

## Caveats that bound these claims

- **The rule quantities leak memorization (found 21 Oct 2026 in the Study 4 design review).**
  - `rule_projection` averages logits over *all* pairs sharing a sum, training pairs included. So fitted train labels partly count as "rule".
  - A train-only lookup table gets substantial rule energy, and its rule part alone scores perfect test accuracy.
  - Affected: rule fractions, `restricted_test_acc`, per-branch rule energy, and therefore the localization index and the reading "the sink hosted more rule than memorization".
  - Grokking times, delays and accuracies are unaffected.
  - Study 4 uses leak-free definitions (`grokking.support`, `rule_projection_test`) and re-tests the role split as its H1.

- **LR optima are at the grid boundary.** Every arm chose lr = 3e-3, the largest value tried. This violates the SPEC rule that optima be interior. The tuned comparisons are "best within the grid", not best-vs-best. The same ordering holds at the shared lr 1e-3 (see the tables), but a higher-lr check (6e-3, 1e-2) is needed before calling any ranking robust to tuning.
- **Escalation pending.** Contrasts whose CI excludes 0.5 must be rerun at 29 seeds before being claimed (protocol step 4). All results here are at **n = 10**.
- **The A1 stability flag fired for every chain-containing arm, but it is mis-specified for chains.**
  - The flag measures *direct* logit energy. In a chain, early blocks act *through* later blocks, so the last block always dominates direct energy by construction.
  - Total-effect ablations show the early blocks are essential. Removing block 1 of `seq` drops test accuracy to 0.03; removing block 2 drops it to 0.29.
  - The flag is reported as preregistered, but it does not indicate a conditioning pathology here. A future amendment should define stability by total effect.
- **Analysis bug, found and fixed before reporting.** The first version of `analyze.py` paired arms across different learning rates in its contrast list. All contrasts below hold weight decay and lr fixed.
- **Not yet done.** The planned GPU replication of the tuning grid (to validate the batched trainer) has not been run.

## Learning-rate selection (protocol step 1)

Median t_grok by lr over 3 seeds; a run that never groks counts as 20,000.

| arm | 1e-4 | 3e-4 | 1e-3 | 3e-3 | chosen |
|---|---|---|---|---|---|
| seq | 20000 | 20000 | 20000 | **4750** | 3e-3 |
| par | 20000 | 20000 | 6250 | **1750** | 3e-3 |
| sink_0 | 20000 | 20000 | 20000 | **5250** | 3e-3 |
| sink_0.25 | 20000 | 20000 | 20000 | **6250** | 3e-3 |
| sink_0.5 | 20000 | 20000 | 20000 | **7250** | 3e-3 |
| sink_1 | 20000 | 20000 | 20000 | **5000** | 3e-3 |
| sink_2 | 20000 | 20000 | 20000 | **4000** | 3e-3 |
| sink_4 | 20000 | 20000 | 16750 | **4500** | 3e-3 |

## Main results, weight decay 1.0

Delay = t_grok − t_fit, right-censored at the budget. Medians over 10 paired seeds.

| arm | delay @ lr 3e-3 (tuned) | censored | delay @ lr 1e-3 (shared) | censored | localization index @ 3e-3 [95% CI] | test acc with block 3 removed @ 3e-3 | final rule share in block 3 @ 3e-3 |
|---|---|---|---|---|---|---|---|
| par | **1,875** | 0 | 6,625 | 0 | −0.023 [−0.051, +0.007] | 0.95 | 0.32 |
| seq | 6,500 | 2 | ≥19,751 | 10 | +0.033 [+0.025, +0.041] † | 0.07 | 1.00 † |
| sink_0 | 6,500 | 0 | ≥19,751 | 9 | 0.000 | 0.98 | 0.00 |
| sink_0.25 | 5,875 | 0 | ≥19,751 | 9 | −0.000 | 0.97 | 0.00 |
| sink_0.5 | 5,875 | 0 | ≥19,751 | 8 | −0.009 [−0.026, −0.000] | 0.87 | 0.02 |
| sink_1 | 4,375 | 0 | ≥19,751 | 6 | −0.006 [−0.013, −0.000] | 0.62 | 0.11 |
| sink_2 | 3,500 | 0 | 17,750 | 1 | −0.007 [−0.021, −0.000] | 0.69 | 0.10 |
| sink_4 | 3,375 | 0 | 12,625 | 0 | −0.028 [−0.068, −0.001] | 0.35 | 0.57 |

† For `seq` and `par`, block 3 is not a sink; it is the last block of the chain, or one of three parallel blocks. Their localization indices are shown only for reference.

**Key paired contrasts** at lr 3e-3, weight decay 1.0, n = 10. P(row faster), with bootstrap 95% CI:
- par vs seq: **1.00** [1.00, 1.00]. par vs sink_4: **1.00** [1.00, 1.00].
- sink_4 vs sink_0: **1.00** [1.00, 1.00]. sink_4 vs seq: **0.95** [0.85, 1.00].
- sink_4 vs sink_2: 0.60 [0.35, 0.85] (not separable). sink_2 vs sink_1: 0.90 [0.70, 1.00].
- seq vs sink_0: 0.30 [0.00, 0.60] (not separable).

**Weight decay 0** (seq, par, sink_1 at lr 3e-3): **0 of 30 runs grokked** within 20k steps.

## Verdicts by the preregistered rules

| Hypothesis | Verdict | Basis |
|---|---|---|
| **H_U** (U-shaped delay, memorization localizes in sink) | **Rejected** | Delay is monotone, with its minimum at the boundary s = 4. No sink arm has a positive localization index. |
| **H_mono** (speed/cost knob) | **Partially supported, preliminary** | Criterion 1 met: delay is monotone, Kendall τ = −0.69. Criterion 2 borderline: the s = 4 rule share is bimodal (mean 0.57 [0.31, 0.82]). A boundary-LR artifact is not excluded. |
| **H_0** (no effect of s) | Rejected | sink_4 is faster than sink_0 in 10/10 seeds. |
| **H_depth** (seq groks without weight decay, par doesn't) | Not supported / not testable | 0 of 30 runs grokked at weight decay 0 (floor effect). Their lr was never tuned at weight decay 0 (A2). |
| **Void rule** (t_fit ratio > 2×) | Not triggered, but uninformative | Median t_fit = 250, the first readout, for every arm, so the check has no resolving power. |
| **Escalation (29 seeds)** | Required, not yet run | par vs seq, seq vs sink_2/4 and the s trend all have CIs excluding 0.5 at n = 10. |

## Exploratory observations (not preregistered; treat as leads)

- **Memorization lands in the chain first.** In sink_1 seed 0 at peak memorization (step 750), the chain carried 100% of the memorization component while the sink had decayed to ≈0. The sink then regrew and carried 66–89% of *both* components before grokking, with no dissociation between them.
- **The sequential chain is markedly unstable at its tuned lr.**
  - At lr 3e-3, `seq` repeatedly collapses after fitting: the median over seeds of the *minimum* train accuracy after the fit is 0.27. Every other arm stays at ≥ 0.98.
  - 8 of 10 seeds cross the 0.95 test threshold at some point, but only 4 of 10 are still above it when their run stops, 1,000 steps after crossing.
  - It ends with the highest Fourier concentration (top-5 share 0.89).
  - So `seq`'s "tuned" lr sits at the edge of stability. Its true best lr may lie between 1e-3 and 3e-3, below the grid's resolution. This weakens the par-vs-seq ranking as a statement about best achievable behavior.
- **The chain memorizes less completely.** `seq` peaks at 0.92 excluded-train accuracy, against 0.98–1.00 for every other arm.

## What this means for the research question

The routing asymmetry tested here differed in *speed and weight-decay cost*, not in depth, and it did not separate memorization from generalization. Whichever route was fast and cheap hosted both. In modular addition the generalizing circuit needs no composition, so shallow routes win outright. That explains why `par` is fastest and why adding sink weight speeds things up.

This is a clean negative result for "static wiring can quarantine memorization." It agrees with the MemSinks finding that localization needs an input-dependent routing signal.

**The question still open** is whether a route that is *structurally* better suited to the generalizing circuit protects it. That requires a task whose generalizing solution needs composition (e.g. $(a+b+c) \bmod p$ or two-hop composition), so that the chain has a real structural advantage the sink lacks.

## Next steps (revised after A3)

1. ~~Upper-LR check~~ done (A3, below).
2. **Before any escalation:** readouts every 25 steps, plus a 2-D lr × wd grid that extends lr to 3e-2 for the non-chain arms. This separates step size from regularization.
3. **Stability-ceiling study** (cheap, predicted by Phase 0): the maximum stable lr vs. longest chain length across one-edge mutants.
4. **Composition task** (new preregistration, with lessons 2–3 built in) where the generalizing circuit requires depth. This is still the real test of the research question.
5. **29-seed escalation** of whatever survives step 2. Run it on the batched GPU trainer once the GPU replication validates it.

## Amendment A3 follow-up: upper-LR re-tuning and re-test (20 Oct 2026)

**Re-tuning.** lr ∈ {2e-3, 3e-3, 6e-3, 1e-2}, 3 seeds, independent data seeds 200–202. Median t_grok:

| arm | 2e-3 | 3e-3 | 6e-3 | 1e-2 | chosen |
|---|---|---|---|---|---|
| seq | 11750 | **4500** | 20000 | 20000 | 3e-3 (**interior**) |
| par | 3250 | 2000 | 1000 | **750** | 1e-2 (boundary) |
| sink_0 | 8000 | 5750 | 2750 | **1250** | 1e-2 (boundary) |
| sink_0.25 | 6750 | 6250 | 2250 | **1250** | 1e-2 (boundary) |
| sink_0.5 | 8000 | 5500 | 2500 | **1000** | 1e-2 (boundary) |
| sink_1 | 7000 | 4750 | 2500 | **1000** | 1e-2 (boundary) |
| sink_2 | 5750 | 3000 | 1750 | **750** | 1e-2 (boundary) |
| sink_4 | 3750 | 3000 | 1750 | **750** | 1e-2 (boundary) |

**Re-test**, 10 seeds at the A3-tuned lrs, weight decay 1.0. Median delay, with censored runs in brackets:

| arm | delay | localization index [95% CI] | test acc, blocks 1 / 2 / 3 removed |
|---|---|---|---|
| par @1e-2 | **500** (0) | −0.024 [−0.049, +0.003] | 0.88 / 0.88 / 0.90 |
| seq @3e-3 | 6,500 (2) | n/a | 0.03 / 0.29 / 0.08 |
| sink_0 @1e-2 | 1,250 (0) | 0.000 | 0.04 / 0.07 / 1.00 |
| sink_0.25 | 1,000 (0) | −0.004 [−0.011, −0.000] | |
| sink_0.5 | 1,000 (0) | −0.020 [−0.046, −0.000] | |
| sink_1 | 875 (0) | −0.000 | 0.52 / 0.57 / 0.42 |
| sink_2 | 750 (0) | +0.004 [−0.036, +0.058] | |
| sink_4 | 500 (0) | **−0.059 [−0.092, −0.026]** | 0.20 / 0.23 / 0.67 |

At s = 4 the sink's share of the final rule energy, per seed, is 1, 0, 0, .36, 0, .19, .20, 0, .03, .03 (mean 0.18).

**Best-vs-best contrasts** (paired, each arm at its own tuned lr): P(par / sink_0 / sink_1 / sink_4 faster than seq) = **1.00 [1.00, 1.00]**.

**Reproducibility check.** The `seq` re-test runs, with the same configuration as study 1 under a new tag, reproduced study 1's t_grok values exactly.

### A3 verdicts, by the rules fixed in A3

| Question | Verdict |
|---|---|
| Does the s-trend survive re-tuning? | **Yes, compressed.** Delays fall from 1,250 to 500 over s = 0 → 4; sink_4 is faster than sink_0 in 10/10 seeds. By the A3 rule it is **not** a tuning artifact and goes to escalation. But every sink arm is still at the upper lr boundary, so it **stays flagged**. |
| Does par-faster-than-seq survive? | **Yes, and it is the most robust result.** `seq` now has an interior optimum and `par` is at the boundary, so any further tuning can only *widen* the gap. |
| H_mono criterion 2 (sink holds most of the rule at the largest s) | **Fails at the new optima** (mean 0.18). H_mono as preregistered is therefore **not supported**. Only "larger s → faster" remains. |
| H_U | **Still rejected.** The localization index is never positive, and at s = 4 it is significantly *negative*. |

### New caveats from A3

- **Readout resolution.** At these learning rates, delays of 500–1,250 steps are resolved at only 250-step granularity, and t_fit always equals the first readout. Comparisons in this regime need readouts every ~25 steps.
- **The lr sweep also sweeps regularization.** In AdamW the decoupled decay is applied as (1 − lr·wd) per step, so raising lr from 3e-3 to 1e-2 also triples the effective weight-decay rate. This study cannot separate "larger steps" from "stronger regularization". A 2-D lr × wd tuning grid is needed.

### Interpretation (not a preregistered claim)

The most robust topology effect here is a **stability ceiling for the sequential chain**. `seq` fails to grok at lr ≥ 6e-3 and is erratic at 3e-3. Every graph whose generalizing route reads the embedding directly keeps improving up to 1e-2.

Phase 0 predicts this direction. In a chain, the effective step on the end-to-end map is multiplied by the current singular values (Prop. 3: rich-get-richer), so the effective learning rate grows as the weights grow, and the stable lr is lower. The parallel graph has no such amplification.

In this task, then, composition hurts grokking mainly through **conditioning/stability**. That limits access to the fast, strongly regularized regime. It does not do so through memorization interference. This is a testable, theory-linked lead: measure the **maximum stable lr as a function of longest chain length** across one-edge mutants.

## Verification

An independent reviewer recomputed the results from the raw run files with separately written code; `analyze.py` was not reused.
- **Reproduced:** every t_fit, t_grok, censored count, median delay, paired P(A faster than B) with its CI, and localization index in this document.
- **Confirmed:** the LR selection followed the preregistered rule, and every optimum is at the upper edge of the grid.
- **Additions:**
  - Only the s = 1 → 2 step is individually separable (P = 0.90 [0.70, 1.00]). The other adjacent steps are 0.55–0.70.
  - The LR-tuning seeds all shared data seed 0.
  - par's dominance value (0.858) comes entirely from step 0, where the embedding path dominates at initialization. Any future dominance diagnostic should exclude initialization.
  - At lr 1e-3, five arms' medians are simply "censored". Their ranking rests on mostly tied censored runs, so no point medians should be quoted for them.
  - Branch energy shares ignore cross terms between branches, which matters for the bimodal s = 4 shares.
- **Downgrades adopted here:** H_mono from "supported" to "partially supported, preliminary", and an explicit statement that nothing is claimable before escalation.
