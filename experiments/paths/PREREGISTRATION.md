# Preregistration — Study 4: what each path does

**Written 21 Oct 2026, before any Study-4 runs.** Any change after runs start is appended under *Amendments*, with its date.

## Why this study

Studies 1–3 drifted from the program's motivation toward optimization questions: grokking speed, stability ceilings, and calibrating a hard task. This study returns to the researcher's original questions ([`docs/00_questions_in_original_words.md`](../../docs/00_questions_in_original_words.md)). It asks all three on one shared set of runs:

- **Q-A, path roles.** *"Can we construct a parameter-free routing topology that simultaneously offers short paths for preserving information, short paths for destroying information, and short paths for constructing new information?"* What does each path actually do in the trained network?
- **Q-B, long-path learning under short-path friction.** *"Measure whether long computations learn anything useful while simple features are already strong."* *"We really need a way of measuring long path learning and utility."*
- **Q-C, memorization vs. generalization by topology.** *"Characterizing the topologies that affect memorization and generalization."*

It obeys the v0 constraints. Graphs differ only in parameter-free edges, with identical learned operators, parameter count, initial weights, data, softmax cross-entropy and optimizer. There are no gates, normalization or clipping.

It also obeys the warning *"If we probe for structure we expect, like a 2-layer circuit, we are blind to what we don't know to measure."* Every measurement is an exact decomposition of the network's output function. No hypothesized circuit or trained probe is assumed. The only task-specific object is the definition of the task's rule.

## Setup

- **Task and model:** as in Studies 1–2.
  - Modular addition, $p=59$, train fraction 0.4.
  - $L=4$ bias-free ReLU MLP blocks, $d=128$, hidden 256.
  - $h_i=h_{i-1}+F_i(h_{k_i})$ with unit multipliers, logits $=Uh_4$.
  - AdamW (0.9, 0.98), weight decay 1.0.
- **Graphs.** Six graphs, chosen so that neighbours differ by rewiring one block's input:

  | graph | block depths | $C$ (longest chain) | $Q$ (root readers) | one-edge neighbours |
  |---|---|---|---|---|
  | `pk0000` | 1,1,1,1 | 1 | 4 | `pk0100` (block 2) |
  | `pk0100` | 1,2,1,1 | 2 | 3 | `pk0000`, `pk0120` (block 3) |
  | `pk0111` | 1,2,2,2 | 2 | 1 | `pk0100` (blocks 3–4: two edges) |
  | `pk0120` | 1,2,3,1 | 3 | 2 | `pk0100`, `pk0122` (block 4), `pk0123` (block 4) |
  | `pk0122` | 1,2,3,3 | 3 | 1 | `pk0120`, `pk0123` |
  | `pk0123` | 1,2,3,4 | 4 | 1 | `pk0120`, `pk0122` |

  A block is **short** if its depth is 1 (it computes from the raw embedding), and **long** if its depth is ≥ 2.
  - The ladder `pk0000 → pk0100 → pk0120 → pk0123` lengthens the chain one edge at a time.
  - `pk0111` and `pk0122` vary $Q$ at fixed $C$.

### Stage 0: per-graph learning rate (3 seeds: 400–402)

Study 2 showed that long chains have lower stable learning rates, so a shared lr would compare graphs at unequal effective training. Each graph is therefore run at lr ∈ {5e-4, 1e-3, 2e-3, 4e-3, 8e-3}, for up to 40,000 steps (stopping 1,000 steps after grokking). The lr is then chosen by this fixed rule (`select_lr.py`):

1. **Stable tier.** Among lrs where ≥ 2/3 seeds grok (test accuracy ≥ 0.95) and ≥ 2/3 seeds are stable (Study 2 criterion), pick the lr with the smallest median grokking step. Ties go to the smaller lr.
2. **Unstable tier.** If there is none, apply the same rule among lrs where ≥ 2/3 seeds grok. The graph is flagged **unstable-at-best**.
3. **No-grok tier.** If there is still none, pick the lr with the highest median final test accuracy. The graph is flagged **no-grok**.

A choice at the grid edge (5e-4 or 8e-3) is flagged. Unstable-at-best graphs stay in all analyses, and every confirmatory result is also reported without them.

### Stage 1: main runs (10 fresh seeds: 500–509)

- Each graph runs at its chosen lr, for up to 40,000 steps.
- Each run continues **5,000 steps after grokking**, so the post-grokking clean-up phase is observed.
- Measurements are taken every 100 steps.
- Seeds are paired: the same seed gives bit-identical initial weights and the same train split in every graph.

## Measurements: exact function-space accounting (`grokking.path_measure`)

**Components and rule projection.** The output decomposes exactly into $L+1$ components: the embedding path $Uh_0$, plus each block's direct write $UF_j(h_{k_j})$. Let $R$ project centered logits onto functions of $(a+b) \bmod p$, i.e., the task's rule; let $M=Z-R$ be the remainder. $R$ is linear, so the following are **exact additive decompositions** over components.

**Per-component roles:**
- **gen** — mean over *test* points of $R_j[i,y_i]$. The component's support for the correct answer on unseen inputs through the rule.
  - gen > 0 means it **constructs** the rule; gen < 0 means it **interferes** with it.
- **mem** — mean over *train* points of $M_j[i,y_i]$. The component's support for the train label through non-rule structure.
  - mem > 0 means it **memorizes**; mem < 0 means it **cancels memorization written elsewhere (destroys)**.
- **noise** — mean over *test* points of $M_j[i,y_i]$.
- **Energy shares.** The Shapley value of the quadratic energy game, $\phi_j=\langle v_j,\sum_k v_k\rangle$, for the rule and for train-point memorization. These sum exactly to the total energy, and a negative share means the component cancels others.

**Composition dependence** (block $j$ with parent $k_j\ge1$): re-evaluate the same block on the raw embedding and compare.
- `comp_dep_state` $=\lVert w_j(h_k)-w_j(h_0)\rVert^2/\lVert w_j(h_k)\rVert^2$.
- The gen and mem of $U(w_j(h_k)-w_j(h_0))$.
- These measure whether a nominally deep block actually *uses* upstream computation (it composes) or recomputes from the embedding (in effect it preserves and ignores upstream). They are zero by definition for short blocks.

**Total-effect ablations:** remove one block, all short blocks, or all long blocks from the forward pass. Record train/test accuracy, gen and mem.

**Utility of a block set** $S$ (fraction of rule support lost on removal): $U_S = 1 - \text{gen}(\text{without } S)/\text{gen}_\text{total}$.

**Also recorded:** per-block gradient norms, the residual norm $\lVert p-e_y\rVert$, and all Study 1–2 measures (rule fraction, excluded-train accuracy, stability).

**Time points:**
- $t_\text{fit}$: first eval with train accuracy ≥ 0.99.
- $t_\text{mem}$: first eval at which excluded-train accuracy peaks.
- $t_\text{grok}$: first eval with test accuracy ≥ 0.95.
- $t_\text{end}$: last eval.

## Confirmatory hypotheses

There are three, each with a stated basis. Seeds are the unit of replication. "Supported" means the stated statistic's 95% bootstrap CI (10,000 resamples of seeds) lies on the predicted side. The family is Holm-corrected at α = 0.05: each hypothesis has a one-sided bootstrap p-value, i.e., the fraction of resamples on the wrong side.

**H1 — Roles split by depth: short paths construct the rule; long paths carry memorization.**

*Basis.* In Study 1, memorization formed first in the chain, while the embedding-reading side branch ended up hosting more of the rule than of the memorization.

*Statistic.* For each mixed graph (`pk0100`, `pk0111`, `pk0120`, `pk0122`, `pk0123`) and seed:
$$D = \frac{\sum_{\text{short}} \text{gen}_j}{\sum_{\text{blocks}} \text{gen}_j}\Big|_{t_\text{end}} - \frac{\sum_{\text{short}} \text{mem}_j}{\sum_{\text{blocks}} \text{mem}_j}\Big|_{t_\text{mem}}$$
- The sums are over blocks only, excluding the embedding path.
- If either denominator has magnitude below 0.05 × its graph's median, that run's $D$ is excluded and counted.

*Prediction.* The mean of $D$ over the five graphs, computed per seed and then bootstrapped, is > 0. Per-graph values are reported.

**H2 — Friction: adding a short reader reduces what the existing long blocks contribute.**

*Basis.* Additive branches share one error signal $p-e_y$, so a short path that fits first can starve the others (gradient starvation; v0 §5). This is the direct test of "long computations learning while simple features are strong".

*Statistic.* Two single-edge contrasts compare the *same* long blocks, with the same wiring, between a graph and the neighbour where one downstream block is rewired to read $h_0$:
- **H2a:** utility of blocks {2, 3}, $U_{\{2,3\}}$, in `pk0120` (block 4 short) versus `pk0123` (block 4 long), at $t_\text{end}$.
- **H2b:** utility of block 2, $U_{\{2\}}$, in `pk0100` (block 3 short) versus `pk0120` (block 3 long), at $t_\text{end}$.

*Prediction.* Utility is lower in the graph with the extra short reader. The paired difference ($A - B$ per seed) has a CI below 0 for **both** contrasts.

**H3 — Composition delays generalization (Q-C).**

*Basis.* Study 1: the parallel net's median grokking delay was 1,875 steps versus 6,500 for the chain, and P = 1.00. This tests whether that becomes a dose–response with tuned learning rates.

*Statistic.* Kendall τ between $C$ and grokking delay ($t_\text{grok}-t_\text{fit}$) over all Stage-1 runs. Runs that never grok are censored at the budget and enter as ties at the top. The p-value comes from a permutation test that permutes graph labels within each seed (10,000 permutations).

*Prediction.* τ > 0, with p < 0.05.

## Exploratory estimands (reported in full, no claims)

1. **Role map.** Mean gen, mem and noise per component at $t_\text{fit}$, $t_\text{mem}$, $t_\text{grok}$ and $t_\text{end}$, per graph. A figure of trajectories.
2. **Destruction.**
   - The fraction of runs in which some component has mem < −0.1·|mem_total|.
   - The cancellation ratio $\sum_j|\text{mem}_j| / |\text{mem}_\text{total}|$ over time.
   - Whether cancellation peaks in the clean-up phase (after $t_\text{grok}$), and at which depth.

   This tests whether memorization is removed by shrinking or by other paths cancelling it.
3. **Preservation.** The embedding path's gen and mem over time, and how much later components cancel it (its Shapley share minus its own energy).
4. **Effective depth.** `comp_dep_state` and the composed share of gen for long blocks: do nominally deep blocks compose, or recompute from the embedding?
5. **Post-fit long-path learning.** The long blocks' share of gen at $t_\text{fit}$ versus $t_\text{end}$, and $U_\text{long}$ over time. This is the direct "long path learning and utility" curve.
6. **Gradient allocation at matched progress.** Per-block gradient norms, short versus long, interpolated at train loss 1.0, 0.1 and 0.01.
7. **Q-C descriptors.** Peak excluded-train accuracy, memorization persistence ($t_\text{grok}-t_\text{mem}$) and final rule fraction, against $C$ and $Q$.

## What each outcome would mean

| Result | Reading |
|---|---|
| H1 ✓ | In this task, depth sorts function: short paths host the generalizing rule and long paths the memorization. That is a topology-controlled division of labor without gates. |
| H1 ✗ (D ≈ 0) | Roles are spread in proportion to capacity; topology does not sort them here. |
| H1 reversed | Long paths host the rule. That would contradict Study 1 and needs explaining. |
| H2 ✓ | Friction is real: a short reader takes over function the long blocks would otherwise carry. Topology allocates learning by path length. |
| H2 ✗ | Long blocks learn the same whatever short paths exist. Shared error does not starve them here. |
| H3 ✓ | Composition depth slows generalization even at each graph's best lr. Not just a stability artifact. |
| H3 ✗ | Study 1's gap was produced by the shared learning rate. |

## Known limits, stated before the data

- **The rule is shallow.** Modular addition can be computed by one ReLU layer from the embeddings, which favours short paths hosting it, so H1 may be task-specific. The deep-rule counterpart is the composition task (Study 3). It is paused until a parameter-free way to train long chains stably is found.
- **Unequal learning rates.** Per-graph lrs mean AdamW's lr·wd coupling differs between graphs. Best-vs-best is the fairest comparison available without new parameters.
- **Instability.** Long chains may be unstable even at their best lr (Study 2). Those graphs are flagged and also analysed without.
- **Escalation.** Ten seeds per graph. Under the SPEC protocol, any contrast whose CI comes within 0.1 standard deviations of its threshold is rerun at 29 seeds before being claimed.
- **No richness sweep.** The SPEC requires effects to vanish as branch scale → 0. That check is deferred to a follow-up, out_scale ∈ {0.25, 1, 4} on the ladder graphs.

## Amendments

(none)
