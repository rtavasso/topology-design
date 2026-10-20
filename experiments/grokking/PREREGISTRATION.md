# Preregistration — topology vs memorization interference in grokking

**Written 20 Oct 2026, before any arm-comparison results.** A calibration run (seq and par arms, one seed, p = 59) was used only to check that the baseline groks and to set the step budget. Its arm comparison is not used as evidence. Any change to this file after the main runs start is appended under "Amendments", with the date.

## Question

Can parameter-free routing keep the fast memorizing solution from interfering with formation of the generalizing ("grokked") circuit in modular addition?

Interference is assumed to run through two channels:
1. **Shared-loss starvation.** Memorization drives the shared softmax residual toward 0.
2. **Weight entanglement.** Memorization reuses the generalizing circuit's units.

## Setup

**Task.** $(a+b) \bmod p$, one-hot inputs, a random train/test split of the full $p\times p$ grid. Defaults: $p$ fixed by calibration (59 or 97); train fraction 0.4.

**Model** (`src/topology_design/grokking.py`):
- $h_0 = E_a[a]+E_b[b]$.
- $h_i = h_{i-1} + \alpha_i F_i(h_{k_i})$, where $F_i$ is a bias-free 2-layer ReLU MLP ($d=128$, hidden 256).
- logits $= U h_3$.

Every arm has bit-identical parameters and initial weights for a given seed.

**Optimizer.** Full-batch AdamW, betas (0.9, 0.98), lr from the per-arm tuning step, weight decay 1.0 unless stated.

**Arms** ($L=3$):

| Arm | Parents | $\alpha$ | Role |
|---|---|---|---|
| `seq` | (0,1,2) | (1,1,1) | standard residual chain |
| `par` | (0,0,0) | (1,1,1) | all blocks read the embedding: one wide 2-layer net |
| `sink_s` | (0,1,0) | (1,1,s) | 2-block chain plus a shallow "sink" branch on $h_0$ with fixed multiplier $s \in \{0, 0.25, 0.5, 1, 2, 4\}$ |

`sink_0` is the matched-parameter control: the sink exists but writes nothing, so its weights only decay.

### Known limitation

In modular addition the generalizing circuit is itself shallow: Fourier embedding, one multiplicative nonlinearity, readout. So the sink is **not** structurally better at memorizing than the chain. It differs from the chain only through its multiplier $s$. With AdamW, $s$ does two things:
- **Speed.** It scales how fast the sink's output moves; the change per step scales with $s$.
- **Cost.** It scales the weight-decay cost of implementing a given output through the sink. For a 2-layer block the minimum $\ell_2$ cost scales as $1/s$.

A high-$s$ sink is therefore both faster *and* cheaper. This study tests whether a speed and cost asymmetry between routes is enough to separate the two solutions. It does **not** test "composition favors generalization." That needs a task whose generalizing circuit requires depth, which is planned as follow-up (e.g. $(a+b+c) \bmod p$ or two-hop composition).

## Rival hypotheses (strong inference)

- **H_U — productive-friction sweet spot.**
  - Prediction: grokking delay is **U-shaped** in $s$, minimized at intermediate $s$.
  - At that minimum, memorization localizes in the sink while the rule lives in the chain.
  - Mechanism: a weak sink doesn't absorb memorization; a strong sink starves the chain.
- **H_mono — speed/cost knob.** A larger $s$ makes the sink the dominant branch for *both* solutions.
  - Prediction: delay changes **monotonically** in $s$, like changing one branch's effective learning rate and weight-decay strength (cf. Omnigrok's weight-norm account).
  - At large $s$ the rule ends up in the sink, with no localization benefit.
- **H_0 — no effect.** Delay and localization do not depend on $s$ beyond seed noise.

**Secondary hypothesis H_depth** (Notsawo et al. 2025; Phase 0 Prop. 2). Without weight decay, `seq` groks within budget more often than `par`.
- Rationale: depth carries an implicit low-rank bias that can replace explicit regularization; the parallel arm is a single wide 2-layer net.
- Tested only at $s=1$ and for seq/par, with weight decay 0.

## Primary endpoints (two only)

1. **Grokking delay** $= t_{\rm grok} - t_{\rm fit}$.
   - $t_{\rm fit}$: first evaluation with train accuracy ≥ 0.99.
   - $t_{\rm grok}$: first evaluation with test accuracy ≥ 0.95.
   - Runs that never grok are right-censored at the step budget. Comparisons use $P(A<B)$ over paired seeds, which handles censoring by ordering (ties count ½).
2. **Localization index at peak memorization.**
   - Peak memorization is the evaluation where `excluded_train_acc` is maximal: the remainder of the logits, after removing everything that depends only on $(a+b) \bmod p$, best fits the training labels.
   - At that checkpoint, compute the sink's share of total remainder energy (memorization) minus its share of total rule energy.
   - Values > 0 mean the sink carries memorization disproportionately.

**Secondary / exploratory** (Holm-corrected if tested):
- final test accuracy after ablating the sink;
- `rule_frac` trajectory;
- residual norm and per-branch gradient norms during the plateau (starvation);
- per-branch `rem_train_acc`;
- embedding Fourier concentration;
- parameter norms.

## Protocol

1. **Per-arm learning rate.** Each arm gets lr ∈ {3e-4, 1e-3, 3e-3} on 3 seeds. Pick the lr with the smallest median $t_{\rm grok}$, censored runs counted as budget. Report results at the per-arm optimum *and* at the shared lr = 1e-3.
2. **Main grid.** Weight decay 1.0, all arms (seq, par, sink_{0,0.25,0.5,1,2,4}), **10 paired seeds** (model seed = data seed index), at each arm's lr.
3. **H_depth grid.** Weight decay 0, arms seq, par, sink_1, 10 seeds.
4. **Escalation.** If a primary endpoint shows $P$ CI excluding 0.5 for any contrast, rerun those arms to **29 seeds** before claiming it.
5. **Readouts.** At least every 250 steps. All raw trajectories are saved.

## Decision rules and kill criteria

| Hypothesis | Supported if | Rejected if |
|---|---|---|
| **H_U** | Median delay at some interior $s$ is below both $s=0.25$ and $s=4$, each with bootstrap $P(\text{interior faster}) > 0.5$ (lower CI bound) **and** the localization index at that $s$ is > 0 (CI excluding 0) | The delay curve is monotone, **or** localization is never > 0. "Localization never > 0" would replicate the MemSinks finding that static structure alone does not localize memorization. That is a clean negative and is reported as such. |
| **H_mono** | Delay is monotone in $s$ **and** at the largest $s$ most of the final rule energy is in the sink | — |
| **H_depth** | `seq` groks without weight decay in more seeds than `par`, with $P(\text{seq faster}) > 0.5$ (CI lower bound) | — |

**Any arm comparison is void** if the arms differ in reaching $t_{\rm fit}$ by more than 2× in median. That would be an optimization-speed confound; report it as such.

## Amendments

**A1 (20 Oct 2026, after calibration, before any main or LR-tuning runs).** Calibration ran 1 seed of `seq` and `par`, with $p=59$, train fraction 0.4, lr 1e-3, weight decay 1.0, and 15k steps.

**Results:**
- **`par`:** fit the training set at step 250 and grokked at step 7,250. Rule fraction rose from 0.04 to 0.91, excluded-train accuracy fell from 1.00 to 0.18, and the top-5 Fourier share of the embeddings rose from 0.18 to 0.59.
- **`seq`:** fit at step 250 but did not grok in 15k steps. Block 3's direct logit energy was about 100× larger than blocks 1–2, which collapsed to ≈0, and the training residual showed instability spikes.

This looks like an optimization pathology at this learning rate, not evidence about topology, so it is not used as a result.

**Changes made:**
1. **Fix $p = 59$, 20,000 steps, readout every 250 steps.** About 10 CPU-minutes per run.
2. **Extend the per-arm LR grid to {1e-4, 3e-4, 1e-3, 3e-3}.** The lowest value was added because of the `seq` instability.
3. **Add a stability diagnostic to every report:** the max over training of the largest single branch's share of direct logit energy. If an arm's median at its tuned LR exceeds 0.9, its comparisons are flagged as conditioning-dominated.

No other change.
