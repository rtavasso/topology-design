# Preregistration — Study 4: what each path does

**Written 21 Oct 2026, before any Study-4 runs.** It was revised once before any runs, after an independent design review (see *Pre-run revision*). Any change after runs start is appended under *Amendments*, with its date.

## Why this study

Studies 1–3 drifted from the program's motivation toward optimization questions: grokking speed, stability ceilings, and calibrating a hard task. This study returns to the researcher's original questions ([`docs/00_questions_in_original_words.md`](../../docs/00_questions_in_original_words.md)). It asks all three on one shared set of runs:

- **Q-A, path roles.** *"Can we construct a parameter-free routing topology that simultaneously offers short paths for preserving information, short paths for destroying information, and short paths for constructing new information?"* What does each path actually do in the trained network?
- **Q-B, long-path learning under short-path friction.** *"Measure whether long computations learn anything useful while simple features are already strong."* *"We really need a way of measuring long path learning and utility."*
- **Q-C, memorization vs. generalization by topology.** *"Characterizing the topologies that affect memorization and generalization."*

It obeys the v0 constraints. Graphs differ only in parameter-free edges, with identical learned operators, parameter count, initial weights, data, softmax cross-entropy and optimizer. There are no gates, normalization or clipping.

It also obeys the warning *"If we probe for structure we expect, like a 2-layer circuit, we are blind to what we don't know to measure."* Every primary measurement is an exact decomposition of the network's output on the true label. No hypothesized circuit, trained probe or rule-based basis is assumed.

**Claim type** (AGENTS rule 5): H1 and H2 are *mechanism* claims about how function is allocated across paths. H3 is an *optimization* claim.

## Setup

- **Task and model:** as in Studies 1–2.
  - Modular addition, $p=59$, train fraction 0.4.
  - $L=4$ bias-free ReLU MLP blocks, $d=128$, hidden 256.
  - $h_i=h_{i-1}+F_i(h_{k_i})$ with unit multipliers, logits $=Uh_4$.
  - AdamW (0.9, 0.98), weight decay 1.0.
- **Graphs.** Eight graphs, chosen so that neighbours differ by rewiring one block's input:

  | graph | block depths | $C$ (longest chain) | $Q$ (root readers) | role in the design |
  |---|---|---|---|---|
  | `pk0000` | 1,1,1,1 | 1 | 4 | all short |
  | `pk0100` | 1,2,1,1 | 2 | 3 | H2b, A |
  | `pk0110` | 1,2,2,1 | 2 | 2 | H2b, B (block 3 reads $h_1$ instead of $h_0$) |
  | `pk0111` | 1,2,2,2 | 2 | 1 | $Q$ varies at fixed $C$ |
  | `pk0120` | 1,2,3,1 | 3 | 2 | H2a, A |
  | `pk0121` | 1,2,3,2 | 3 | 1 | H2a, B (block 4 reads $h_1$ instead of $h_0$) |
  | `pk0122` | 1,2,3,3 | 3 | 1 | chain ladder |
  | `pk0123` | 1,2,3,4 | 4 | 1 | full chain |

  A block is **short** if its depth is 1 (it computes from the raw embedding), and **long** if its depth is ≥ 2.

### Stage 0: per-graph learning rate (3 seeds: 400–402)

Study 2 showed that long chains have lower stable learning rates, so a shared lr would compare graphs at unequal effective training.
- Each graph runs at lr ∈ {2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}, for up to 40,000 steps.
- Each run continues **5,000 steps after grokking**, the same window as Stage 1, so stability is judged over the same post-grokking exposure.

The lr is chosen by this fixed rule (`select_lr.py`):
1. **Stable tier.** Among lrs where ≥ 2/3 seeds grok (test accuracy ≥ 0.95) and ≥ 2/3 seeds are stable (Study 2 criterion), pick the lr with the smallest median grokking step.
2. **Unstable tier.** If there is none, look among lrs where ≥ 2/3 seeds grok. Pick the highest median post-fit minimum train accuracy (Study 2's continuous stability endpoint), then the smallest median grokking step. The graph is flagged **unstable-at-best**.
3. **No-grok tier.** If there is still none, pick the highest median final test accuracy. The graph is flagged **no-grok**.

Remaining ties go to the smaller lr. A choice at the grid edge is flagged. Selection refuses to run unless all 168 Stage-0 runs are present without errors.

Stage 0 also records the full path measurements. That gives an **exploratory shared-lr comparison** at every lr (3 seeds each).

### Stage 1: main runs (10 fresh seeds: 500–509)

- Each graph runs at its chosen lr, for up to 40,000 steps, and continues 5,000 steps after grokking.
- **Evaluation schedule:** every 10 steps below step 200, every 25 below step 1,000, then every 100. Training fits within about 100 steps, so the early time points need dense evals.
- Seeds are paired: the same seed gives bit-identical initial weights and the same train split in every graph.

## Measurements (`grokking.path_measure`)

Everything is computed on the full $59^2$ grid. The **components** are the embedding path $Uh_0$ and each block's direct write $UF_j(h_{k_j})$. Their centered logits sum exactly to the network's.

**Support decomposition** (exact and additive over any split of the logits):
- **gen** — mean over *test* points of the centered logit for the true label. Support for the correct answer on inputs never trained on.
- **mem** — mean over *train* points of the same, minus gen. Train-specific support.
- A pure function of the task rule has mem = 0. A lookup table that boosts only train labels has gen = 0. This is tested.
- **Signs carry roles:**
  - gen > 0 means the component **constructs** generalizing function; gen < 0 means it **interferes**.
  - mem > 0 means it **memorizes**; mem < 0 means it **cancels memorization written elsewhere (destroys)**.

**Two attributions:**
- **Direct.** Each component's own write, as above.
- **Total effect.** Every subset of blocks is ablated (multiplier set to 0). Let $v(T)$ be gen (or mem) of the network with only the blocks in $T$ present. Each block gets its exact **Shapley value** in that game (`gen_tshap`, `mem_tshap`).
- Direct attribution credits a deep block with upstream computation it merely reads. Total-effect attribution credits upstream blocks for what flows through them. A claim must hold under both (AGENTS rule 6).

**Energy accounting:** the Shapley values of the quadratic energy game, $\phi_j=\langle v_j,\sum_k v_k\rangle$.
- Rule energy uses a rule projection estimated *from test points only*, i.e., functions of $(a+b)\bmod p$. This avoids crediting fitted train labels to the rule.
- Memorization energy is the train-point remainder.
- A negative share means the component cancels others.

**Composition dependence** (block $j$ with parent $k_j\ge1$): the same block re-evaluated on the raw embedding, $w_j(h_k)-w_j(h_0)$. Recorded as a state-space energy share and as gen/mem. It is zero by definition for short blocks. *Caveat:* $h_0$ is an input the block never sees in training.

**Utility of a block set** $S$: $U_S = 1-\text{gen}(\text{without } S)/\text{gen}_\text{total}$, undefined if gen_total ≤ 0.

**Time points:**
- $t_\text{fit}$: first eval with train accuracy ≥ 0.99.
- $t_\text{mem}$: the eval at which total mem peaks.
- $t_\text{grok}$: first eval with test accuracy ≥ 0.95.
- $t_\text{end}$: **the mean over the last 5 evals**, so one eval landing mid-collapse cannot decide it.

## Confirmatory hypotheses

There are three. Seeds are the unit of replication. **H1 and H2 use only runs that grokked**; exclusions are reported. Each hypothesis has a one-sided p-value, and the family is Holm-corrected.

**Verdict rule:**
- **Supported** requires the stated CI condition and Holm p < 0.05.
- If 0.01 < Holm p < 0.05, the verdict is *supported, escalate*: rerun at 29 seeds before claiming (SPEC protocol).

**H1 — Roles split by depth: short paths construct generalizing function; long paths carry memorization.**

*Basis.* In Study 1, memorization formed first in the chain, while the embedding-reading side branch ended up hosting more of the rule than of the memorization. That reading used the leaky rule projection, so it needs this recheck.

*Statistic.* For each mixed graph (all except `pk0000`) and seed:
$$D = \frac{\sum_{\text{short}} \text{gen}_j}{\sum_{\text{blocks}} \text{gen}_j}\Big|_{t_\text{end}} - \frac{\sum_{\text{short}} \text{mem}_j}{\sum_{\text{blocks}} \text{mem}_j}\Big|_{t_\text{mem}}$$
- The sums are over blocks only.
- Each share is clipped to [−1, 2].
- A run is excluded if either denominator has magnitude below 0.05 × its graph's median.

*Prediction.* The per-seed mean of $D$ over graphs has a 95% bootstrap CI (10,000 resamples of seeds) above 0 **under both direct and total-effect attribution**. The p-value is the larger of the two one-sided p-values.

**H2 — Friction: an extra short reader reduces what existing long blocks contribute.**

*Basis.* Additive branches share one error signal $p-e_y$, so a short path that fits first can starve the others (gradient starvation; v0 §5). This is the direct test of "long computations learning while simple features are strong".

*Statistic.* Two single-edge contrasts. In each, one block reads $h_0$ in graph A and $h_1$ in graph B. In both graphs that block does not read the set $S$, so removing $S$ leaves its input unchanged and $U_S$ means the same thing in both.
- **H2a:** $U_{\{2,3\}}$ in `pk0120` (A, block 4 reads $h_0$) versus `pk0121` (B, block 4 reads $h_1$).
- **H2b:** $U_{\{2\}}$ in `pk0100` (A, block 3 reads $h_0$) versus `pk0110` (B, block 3 reads $h_1$).

Both are measured at $t_\text{end}$ and paired by seed; only pairs where both runs grokked are used.

*Prediction.* The 95% bootstrap CI of the mean of $U_A-U_B$ lies below 0 in **both** contrasts. The p-value is the larger of the two one-sided p-values (intersection-union test).

*Note.* A and B usually run at different lrs. The Stage-0 shared-lr comparison is reported alongside.

**H3 — Composition delays generalization (Q-C).**

*Basis.* Study 1: the parallel net's median grokking delay was 1,875 steps versus 6,500 for the chain, and P = 1.00. This tests whether that becomes a dose–response at per-graph-tuned lrs.

*Statistic.* Kendall τ_b between $C$ and grokking delay ($t_\text{grok}-t_\text{fit}$) over all Stage-1 runs.
- Runs that never grok are censored as ties at the top.
- The p-value comes from a permutation test that permutes graph labels within each seed (10,000 permutations).
- The CI comes from a bootstrap over seeds, keeping each seed's runs together.

*Prediction.* τ > 0 with a CI above 0.

*Reading.* A positive τ at tuned lrs means composition costs generalization speed under this optimizer at each graph's best stable setting. It does **not** separate step size from decay (lr·wd differs between graphs). An lr·wd-normalized delay and the shared-lr τ from Stage 0 are reported as exploratory checks.

**Robustness for all three:** each is also reported without graphs flagged unstable-at-best or no-grok. H2 contrasts that involve a flagged graph are marked.

## Exploratory estimands (reported in full, no claims)

1. **Role map.** Mean gen and mem per component, under both attributions, at $t_\text{fit}$, $t_\text{mem}$, $t_\text{grok}$ and $t_\text{end}$. Trajectory figures.
2. **Destruction.**
   - The fraction of runs with a component whose end mem is < −0.1 × the run's peak |mem_total|.
   - The cancellation ratio $\sum_j|\text{mem}_j|/|\text{mem}_\text{total}|$ at $t_\text{mem}$, $t_\text{grok}$ and $t_\text{end}$.
   - Whether cancellation rises in the clean-up phase.
3. **Preservation.** The embedding path's gen and mem, and how much of its energy later components cancel.
4. **Effective depth.** `comp_dep_state` of long blocks: do nominally deep blocks compose?
5. **Post-fit long-path learning.** Long blocks' share of gen, and $U_\text{long}$, at $t_\text{fit}$, $t_\text{mem}$, $t_\text{grok}$ and $t_\text{end}$. This is the "long path learning and utility" curve.
6. **Gradient allocation at matched progress.** Per-block gradient norms, short versus long, at the first eval where train loss falls below 1.0, 0.1 and 0.01.
7. **Q-C descriptors.** Peak mem_total, persistence ($t_\text{grok}-t_\text{mem}$), final rule fraction, stability, and lr·wd-normalized delay, against $C$ and $Q$.
8. **Shared lr.** H2 contrasts and H3 τ at each Stage-0 lr.

## What each outcome would mean

| Result | Reading |
|---|---|
| H1 ✓ | In this task, depth sorts function: short paths host generalizing function and long paths the memorization, under both attributions. That is a topology-controlled division of labor without gates. |
| H1 ✗ (D ≈ 0) | Roles are spread in proportion to capacity; topology does not sort them here. |
| H1 attributions disagree | The "role" depends on how credit for composed computation is assigned. Report as unresolved. |
| H1 reversed | Long paths host generalizing function, contradicting Study 1's reading. |
| H2 ✓ | Friction is real: a short reader takes over function the long blocks would otherwise carry. Topology allocates learning by path length. |
| H2 ✗ | Long blocks contribute the same whatever reader is added. Shared error does not starve them here. |
| H3 ✓ | Composition depth slows generalization even at each graph's best lr. |
| H3 ✗ | Study 1's gap was produced by the shared learning rate. |

## Known limits, stated before the data

- **The rule is shallow.** Modular addition can be computed by one ReLU layer from the embeddings, which favours short paths hosting it, so H1 may be task-specific. The deep-rule counterpart (Study 3) is paused.
- **Unequal learning rates.** Per-graph lrs mean AdamW's lr·wd coupling differs between graphs. Best-vs-best is the fairest comparison available without new parameters; shared-lr checks are exploratory, at 3 seeds.
- **Instability.** Long chains are likely unstable at every lr that groks (Study 2). They are flagged and also analysed without.
- **Small H2 contrasts.** H2 compares a depth-1 reader with a depth-2 reader, a small structural difference, so a null result is weak evidence.
- **SPEC requirements deferred:**
  - the richness sweep (effects must vanish as branch scale → 0), planned as out_scale ∈ {0.25, 1, 4} on the H2 graphs;
  - random-routing and operator-permutation nulls.

  Seed-to-seed variation within each graph, from the paired bootstrap, serves as the identical-graph null.

## Pre-run revision (21 Oct 2026, before any runs)

An independent review of the first draft found blocking problems. All were fixed before any run:

1. **gen used the full-grid rule projection, which credits fitted train labels as rule.** A train-only lookup table scored gen = 1.93. gen and mem are now defined directly on true-label logits, as above, and a test covers the lookup-table case. The same leak affects Study 1's rule-based quantities; this is noted in `experiments/grokking/RESULTS.md`.
2. **The H2 contrasts had the predicted answer built in.** In the original B graphs, the rewired block read the ablated set. They are replaced by `pk0121` and `pk0110`, which do not.
3. **The lr rule would have picked the most unstable lr for long chains.** The unstable tier is now ranked by the continuous stability endpoint before speed. The grid now covers 2.5e-4 to 1.6e-2. Stability is judged over the same post-grokking window as Stage 1. Selection checks that Stage 0 is complete.
4. **Direct attribution alone shifts credit along chains.** Total-effect Shapley is added, and H1 must hold under both.
5. **The early time points were unresolved.** Evals are now dense early, $t_\text{mem}$ is the peak of total mem, and $t_\text{end}$ is averaged over 5 evals.
6. **Analysis gaps.** H2 is added to the without-flagged analysis; verdicts and the escalation rule are printed; H3 has a CI; H1 and H2 are restricted to grokked runs, with counts reported.

## Amendments

(none)
