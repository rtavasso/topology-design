# Preregistration — Study 2: stability ceiling vs. composition depth

**Written 20 Oct 2026, before any Study-2 runs.** Changes after runs start are appended under *Amendments*, with dates.

## Motivation

In Study 1 (`experiments/grokking/RESULTS.md`, A3), the sequential 3-block chain had a hard learning-rate ceiling: it never grokked at lr ≥ 6e-3 and was erratic at 3e-3. Every graph whose generalizing route reads the embedding directly kept improving up to 1e-2.

Phase 0 Prop. 3 predicts the direction. In a balanced chain of depth $L$, the update to the end-to-end map is the plain gradient scaled mode-wise by $\gamma_{ab}=\frac1L\sum_j s_a^{2(j-1)/L}s_b^{2(L-j)/L}$. So the *effective* step grows with the learned singular values, faster for longer chains. Graphs whose computation passes through longer compositions should therefore destabilize at lower learning rates.

This study tests that as a **dose–response over all 24 four-block parent-selection graphs**. It is the first planned test of a graph descriptor predicting a learning-dynamics outcome out of sample.

## Setup

- **Task and model:** as in Study 1. Modular addition, $p=59$, train fraction 0.4, $d=128$, hidden 256, bias-free ReLU blocks, AdamW (0.9, 0.98).
- **Graphs:** $L=4$ blocks, so $h_i=h_{i-1}+F_i(h_{k_i})$ for all 24 parent vectors $k$ (arms `pk0000` … `pk0123`), with unit multipliers. Bit-identical parameters and initial weights across graphs for a given seed.
- **Graph descriptors**, computed before any runs (`grokking.block_depths`):
  - **longest chain $C$**: the maximum composition depth; block $i$ has depth $1+$ the deepest block contributing to its parent state.
  - **root readers $Q$**: the number of blocks reading $h_0$ directly.

  | $C$ | graphs | $Q$ |
  |---|---|---|
  | 1 | 1 | 4 |
  | 2 | 13 | 1–3 |
  | 3 | 9 | 1–2 |
  | 4 | 1 | 1 |

  $C$ and $Q$ are correlated, but $Q$ varies within $C=2$ and $C=3$.
- **Learning rates:** {1.5e-3, 3e-3, 6e-3, 1.2e-2, 2.4e-2, 4.8e-2}.
- **Weight decay:** 1.0 for all graphs.
  - **Robustness subset:** weight decay 0.1 for graphs `pk0000` ($C=1$), `pk0101` ($C=2$), `pk0120` ($C=3$), `pk0123` ($C=4$).
  - This partly separates step size from the AdamW decay rate lr·wd.
- **Seeds:** 3 paired model/data seeds (300–302).
- **Budget:** 6,000 steps, readouts every 50 steps, no early stopping (instability can appear after grokking).
- **Total:** 24×6×3 + 4×6×3 = 504 runs on CPU, with the unbatched trainer validated in Study 1.

## Endpoints

**Per run:**
- **stable** if the loss stays finite, training accuracy reaches ≥ 0.99, and after that first fit it never drops below 0.9 for the rest of the budget;
- `post_fit_min_train_acc`, the continuous version;
- **failure mode:** diverged / never fit / collapsed after fit.

**Primary endpoint, per graph:** $\text{lr}^*_G$ = the largest grid lr at which at least 2 of 3 seeds are stable.
- If no lr is stable, $\text{lr}^*_G$ is set below the grid.
- If the top lr is stable, it is right-censored at 4.8e-2.

**Secondary (exploratory):**
- the fastest grokking delay over each graph's stable lrs, as a function of $C$;
- the failure-mode mix as a function of lr and $C$.

## Hypotheses and decision rules

- **P1 (primary, directional).** $\text{lr}^*$ is non-increasing in $C$.
  - **Test:** Kendall τ between $C$ and $\log\text{lr}^*$ across the 24 graphs, with ties handled by τ-b.
  - **Supported if** τ < 0 with a one-sided permutation p < 0.05. The p-value comes from 10,000 random relabelings of $\text{lr}^*$ across graphs.
  - **Falsified if** τ ≥ 0, or p ≥ 0.05.
- **P2 (magnitude).** $\text{lr}^*(\texttt{pk0123}) \le \text{lr}^*(\texttt{pk0000})/4$, i.e. at least two grid steps apart.
- **P3 (chain length beats the rival descriptor $Q$).**
  - Within the $C=2$ and $C=3$ groups, $\log\text{lr}^*$ does not increase with $Q$: within-group Kendall τ($Q$, log lr*) ≤ 0.2 in both groups.
  - And across all graphs, a rank regression of log lr* on $C$ and $Q$ gives a negative coefficient for $C$, with 95% bootstrap CI below 0. The bootstrap resamples graphs.
  - **If instead $Q$ carries the effect** (positive within-group τ > 0.2 in both groups, and the CI for $C$ includes 0), the stability ceiling is attributed to direct-read fan-in, not composition depth.
- **P4 (robustness to weight decay).** For the 4-graph subset, the ordering of $\text{lr}^*$ by $C$ at weight decay 0.1 matches weight decay 1.0 (Spearman ρ ≥ 0.8). If not, the ceiling is attributed partly to the AdamW decay rate rather than to step size.

## Known limitations

- **Coarse grid.** A factor-2 grid resolves $\text{lr}^*$ only to within 2×.
- **Few seeds.** 3 seeds per cell is a screening design. If P1 holds, the confirmatory follow-up adds seeds and a finer grid around each graph's ceiling.
- **One task.** Modular addition only. Transfer to another task is required before calling this a design principle (SPEC Gate 4).
- **Correlated descriptors.** $C$ and $Q$ are correlated. P3 is the guard.

## Amendments

(none yet)
