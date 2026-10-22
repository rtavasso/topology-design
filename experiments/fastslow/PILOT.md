# Phase 5 pilot — results (22 Oct 2026)

**Run:** Kaggle, 2× Tesla T4 (one used), torch 2.11.0+cu128, commit `05943fe`.
- Grid: [`grids/pilot.json`](grids/pilot.json), 36 runs, fixed 40,000-step horizon, 54 minutes.
- The fleet tests passed on the GPU first: 59 passed, 1 skipped.
- Raw results stay outside git, in `results/kaggle_pilot/results/` (15 MB).
- Analysis: [`analyze_pilot.py`](analyze_pilot.py), plus the extra diagnostics below.
- Seeds 490–492. Graphs: asym = `pk012000` (L = blocks 1–3 as a chain, S = blocks 4–6 reading the embedding); sym = `pk000000`.

**Outcome: stage 1 is not launched.** The pilot did its job: it showed the stage-1 design, as written, would mostly measure instability and a saturated statistic.

## 1. The preregistered learning-rate rule selects an unstable regime

| lr | graph | grokked (of 3) | t_grok | collapses after fit | min test acc after grokking |
|---|---|---|---|---|---|
| 1e-3 | asym | 2 | 25,400 / — / 16,400 | 3, 2, 3 | 0.58, —, 0.62 |
| 1e-3 | sym | 3 | 7,200 / 7,400 / 5,600 | 0, 0, 0 | 0.96 |
| 2e-3 | asym | 3 | 5,600 / 7,000 / 5,300 | 6, 9, 3 | 0.49, 0.19, 0.53 |
| 2e-3 | sym | 3 | 3,400 / 3,300 / 2,500 | 0, 0, 0 | ≥ 0.95 |
| 4e-3 | asym | 3 | 1,600 / 1,900 / 1,400 | 30, 47, 61 | 0.09–0.15 |
| 4e-3 | sym | 3 | 1,800 / 1,600 / 1,300 | 0, 0, 0 | ≥ 0.87 |

- The rule (the largest lr at which both graphs grok in ≥ 2 of 3 seeds) picks **4e-3**.
- There, asym collapses 30–61 times after fitting, and one seed ends at test accuracy 0.48.
- **asym is unstable at every lr tried; sym is stable at every lr tried.** This replicates Studies 2 and 4: chain-containing graphs are less stable.
- **The rule's flaw:** "groks" means *first* crossing of 0.95 test accuracy, so it rewards collapse-and-recover regimes.

## 2. The headline measure saturates at the end of training

Composition at the end of training, $c_X = \text{gen}_X/(|\text{gen}_X|+|\text{mem}_X|)$, averaged over the last 2,000 steps:
- **≈ +0.95 for both branches, in every arm and at every lr.**
- R = c_L − c_S lies within ±0.08 everywhere.
- After grokking, weight decay removes nearly all train-specific function, so there is almost nothing left to split between branches.
- **★H5B-1 required ΔR ≥ 0.10.** That is nearly impossible by construction in this regime. The measure cannot move by the effect size it was meant to detect.

**During the plateau** (fit ≤ t < grok), composition varies from −0.15 to +0.46, and arms differ (exploratory, 3 seeds).
- Example: memory-4 at lr 1e-3. The long-memory branch L is *less* generalizing than S: c_L 0.10–0.25 vs c_S 0.37–0.43. That is the opposite of the complementary-learning-systems prediction.
- But plateau lengths differ by arm, so plateau comparisons need a matching rule decided in advance.

## 3. The scale-matching multipliers are ill-conditioned, and they expose a capacity confound

The multiplier α_X is the ratio of branch X's output RMS in base to that in the unscaled memory arm.
- **Read at a single step (20,000), as preregistered:** inside collapsing runs it ranges from 0.23 to 2,127 across seeds. It is noise.
- **Averaged over steps 10k–30k:** consistent across seeds.
  - At 2e-3: memory-4 gives α_L ≈ 0.93–1.02 and α_S ≈ 3.1–3.7. Memory-¼ gives α_S ≈ 0.7–1.5.
- **The confound:** at 1e-3, weight decay 2 shrinks S's output to about 1/20 of base (RMS 0.22 vs 4.4). The memory arms change a branch's capacity, not only its time constant.

## What this says

- **About the design:** the preregistered stage-1 grids would have spent about 7 GPU-hours measuring a saturated statistic in an unstable regime. They were deleted unlaunched.
- **About the research question:** the result is consistent with the shared-error account in [`docs/LEDGER.md`](../../docs/LEDGER.md) (H1, credit by speed). Under one shared error, giving branches different time constants changes speed and scale, not the kind of function each carries. That is a hypothesis to test, not a conclusion from 3 seeds.

**Status of Phase 5 B–D:** on hold. They all keep one objective on a summed output. See the ledger, X9 and I9.
