# Results — topology vs memorization interference in grokking (study 1)

**Run:** 20 Oct 2026, natively on the researcher's MacBook Pro (12 CPUs, 10 workers, torch 2.14.1, commit `662af2d`).
- 96 LR-tuning runs and 190 main runs, unbatched CPU trainer.
- Scored against [PREREGISTRATION.md](PREREGISTRATION.md) (amendments A1–A2 written before these runs).
- Figures: [`results_figures/`](results_figures/).
- Raw run files stay on the Mac in `results/grokking/` (26 MB, not in git); a compressed copy is in `results/grokking/all_results.tgz`.

**Independent check.** The verdicts below were recomputed from the raw files by a separate reviewer using independent code. See the *Verification* section.

## Headline

1. **The sink idea failed in this task, cleanly.**
   - A side branch with a fixed multiplier did **not** soak up memorization.
   - Memorization formed first in the *chain*. The sink, once strong enough to matter, ended up hosting more of the *rule* than of the memorization.
   - The localization index was never positive and was negative at s = 4.
   - **H_U is rejected.**
2. **The multiplier acts as a speed/cost knob (H_mono supported at n = 10).**
   - Grokking delay falls monotonically as s rises: 6,500 → 5,875 → 5,875 → 4,375 → 3,500 → 3,375 steps for s = 0 → 4, at the tuned lr.
   - At s = 4 the sink carries a majority (0.57) of the final rule energy.
3. **The parallel network groks fastest by a wide margin.**
   - Median delay is 1,875 steps, against 6,500 for the sequential residual chain.
   - P(par faster than seq) = 1.00 in all 10 paired seeds.
   - In this task, routes that read the embedding directly host the generalizing circuit fastest. Composition is a cost, not a help.
   - This is consistent with the limitation flagged in the preregistration: the modular-addition circuit is shallow.
4. **H_depth (chain groks without weight decay) is untestable at this budget.** No arm grokked in 20k steps at weight decay 0.

## Caveats that bound these claims

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
| **H_mono** (speed/cost knob) | **Supported (n = 10; escalation pending)** | Monotone delay. At s = 4 the sink holds 0.57 of final rule energy. |
| **H_0** (no effect of s) | Rejected | sink_4 is faster than sink_0 in 10/10 seeds. |
| **H_depth** (seq groks without weight decay, par doesn't) | Not testable at this budget | Nothing grokked at weight decay 0. |
| **Void rule** (t_fit ratio > 2×) | Not triggered | Median t_fit = 250 for every arm. |

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

## Next steps

1. **Upper-LR check:** lr ∈ {6e-3, 1e-2} for all arms on 3 seeds, to find interior optima.
2. **29-seed escalation** of the separable contrasts (par vs seq, sink_4 vs sink_0, sink_4 vs seq), at interior optima. Run it on the batched GPU trainer once the GPU replication validates it.
3. **Composition task** (new preregistration) where the generalizing circuit requires depth.
4. **Amend the stability diagnostic** to use total-effect ablations rather than direct energy.

## Verification

(see below)
