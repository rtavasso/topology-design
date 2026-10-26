# Composing a frozen primitive library: design

**Status:** design, 26 Oct 2026. Not yet a preregistration. The decisions and the literature behind them are in [docs/generator_design.md](../../docs/generator_design.md) (§§1–14). This document consolidates them into one plan. It will be turned into `PREREGISTRATION.md` after a throughput test and an independent review.

**Ledger:** I25 (generator of worlds), following F19 (the data gap is the prior).

---

## 1. Question

Can a meta-model learn, from a distribution of generated worlds, to compose a frozen library of verified primitive blocks into programs inferred from examples in context? And does it generalize to compositions never seen in training better than a monolithic PFN of equal size trained on the same worlds?

**Sub-questions:**
1. **Composition data.** How does the ability to compose depend on how much composition the training mix contains: depth, diversity and concept mixing?
2. **Open-loop vs execution-conditioned.** Does seeing the execution trace help, especially on longer programs? This measures whether an open-loop composer learns an internal model of its primitives.
3. **The learned prior.** Does the composer learn the Bayes posterior over programs (it can be checked exactly)? How much of its competence is the first proposal ("seeing") and how much is verified search ("searching")?

**Not in this round:**
- discovering primitives jointly with the composer (later);
- a memory store or home world;
- randomizing symbols per world;
- a soft (unclean) interface (a later stability variant).

## 2. The world generator

**Domain:** values in Z_7.

**Machine:** two registers. At the start r0 = x and r1 = y, with x, y ∈ Z_7. The output is r0 after the program.

**Instruction set (32 instructions):**
- `dest ← src1 op src2` for op ∈ {add, sub, mul}, with dest, src1, src2 ∈ {r0, r1};
- `dest ← f(src)` for f ∈ {neg, inc};
- plus HALT.

**Concept groups:**
- additive: {add, sub, neg, inc}. Its programs form a closed family of 7³ = 343 affine maps;
- multiplicative: {mul}.

**Function enumeration** (`experiments/worlds/count_isa.py`): breadth-first search over register states, deduplicated, up to length 6. Each output function is a 49-entry table.

| minimal length ℓ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| new functions (mod 7) | 13 | 80 | 552 | 4,179 | 32,741 | 260,279 |

Cumulative at ℓ ≤ 6: 297,845 functions (60 s to enumerate on CPU). Among all of them, 8 examples identify a function 67% of the time and 12 examples 93%.

For each function, all minimal-length programs are stored.

**A world** is one function. **An episode** is a context of k examples ((x, y), f(x, y)), with k ≤ 16, plus query inputs, all drawn from the 49 inputs without replacement. Six to eight examples pin a function down about 50–90% of the time, and twelve about 98%.

**Program prior:**
1. pick a function uniformly within the training pool (per minimal length, with the cell's depth weights);
2. pick one of its minimal programs uniformly.

Value semantics are fixed across worlds.

### Held-out sets

All held-out sets are defined over *functions* and certified by enumeration: the minimal length and the membership of every held-out function are computed, not assumed.

| split | training sees | test | kind |
|---|---|---|---|
| **in-support** | 90% of functions at each ℓ ≤ 5 | the other 10% | in support of the training prior |
| **length** | ℓ ≤ 5 | ℓ = 6 | out of support |
| **concept mixing** | functions whose minimal programs use one concept group | functions whose every minimal program needs both | out of support |
| **new operation in composition** | `neg` only in ℓ = 1 functions | functions whose minimal programs all need `neg` inside a composition | out of support |
| **new addressing** | `inc` never applied to r1 | functions whose minimal programs all need `inc r1` | out of support |
| **order** | programs where every mul precedes every additive instruction | functions whose minimal programs all need an additive step before a mul | out of support |

The last four follow the ExeDec taxonomy, adapted. A split is used only if its certified test set has at least 200 functions.

**Which cells carry which split.** Every out-of-support split except length restricts what training may contain, so the restrictions can't all apply in every cell (concept mixing would remove all mixed programs from the mixed cells). Hence:
- the D and N cells train on everything except the in-support hold-out (ℓ = 6 is out of their pool by construction). They test in support and length;
- the **S5** cell carries the concept-mixing restriction;
- the **X5** cell carries the three structural restrictions together (neg only alone, no `inc r1`, mul before additive). Each of those splits is tested on functions that violate *exactly one* restriction, so failures are attributable.

## 3. Stage 1: the frozen library

- **One block per operation:** add, sub and mul take a 49-way one-hot input (the two operand values) and output 7 logits; neg and inc take a 7-way input.
- **Register addressing is done by the executor,** not the blocks: it reads the source registers and writes the destination register.
- **Training:** each block is trained on its full input domain until exhaustively correct (49 or 7 inputs), then verified on every input and frozen.
- **Clean-up:** the executor snaps each block's output to a one-hot value (argmax), so execution is exact. With exact execution the library behaves as a symbolic interpreter. Stage 1 is the positive control, and every composer error is an inference error.

## 4. Models

All models are sized for T4s, with parameters and training episodes matched across arms.

- **Context encoder (shared design):**
  - each example is one token: embed(x) + embed(y) + embed(f(x, y)), with no positional encoding (the context is a set);
  - a transformer encoder of width 128–256, 4–6 layers, about 1–5M parameters in total;
  - the final sizes are set by the throughput test.
- **Monolithic PFN:** a query token embed(x) + embed(y) + [?] attends to the encoded context and outputs a distribution over Z_7. Trained by cross-entropy on held-out query outputs: the standard PFN objective, whose optimum is the posterior predictive.
- **Open-loop composer:**
  - an autoregressive decoder over instruction tokens (32 + HALT, programs up to 6 + HALT);
  - each instruction cross-attends to the encoded context;
  - trained by cross-entropy on the sampled minimal program (latent supervision; optimum = the posterior over programs);
  - predictions come from executing the decoded program (greedy, or best of k verified samples) on the query inputs.
- **Execution-conditioned composer:**
  - the same decoder;
  - after each instruction, the library executes the partial program on every context input;
  - the current register values (r0, r1) for each example are embedded alongside that example's target output, and the decoder attends to these trace tokens at the next step.
  - Same objective.

## 5. The training-mix sweep (9 cells)

| cell | depth cap | distinct training functions | concepts |
|---|---|---|---|
| D2 | 2 | all (≈ 84 after hold-out) | mixed |
| D3 | 3 | all (≈ 580) | mixed |
| D4 | 4 | all (≈ 4.3k) | mixed |
| D5 | 5 | all (≈ 34k) | mixed |
| N1k | 5 | 1,000 (subsample) | mixed |
| N4k | 5 | 4,000 | mixed |
| N16k | 5 | 16,000 | mixed |
| S5 | 5 | all single-concept functions | single |
| X5 | 5 | all functions satisfying the three structural restrictions | mixed |

- Depth weights are equal per depth up to the cap: equal probability per minimal length, then uniform over functions within a length.
- Hold-out sets are removed from every cell's pool first.
- Grid: 3 arms (monolithic, open-loop, execution-conditioned) × 9 cells × 3 seeds = **81 runs.**

## 6. Evaluation

All results are stratified by minimal length (1–6) and by split.

**Metrics:**
- **Functional accuracy** on held-out query inputs.
- **Predict-then-learn code length** over a stream of k = 1 … 16 examples, against the exact Bayes learner. The Bayes learner is computed under (a) the cell's own training prior and (b) the generator prior over all functions with ℓ ≤ 6.
- **Posterior fidelity:**
  - for the composers, KL(exact posterior ‖ the composer's induced distribution over functions), estimated from sampled programs;
  - for the monolithic PFN, KL between predictive distributions.
- **Seeing vs searching:** accuracy of the first (greedy) program against the best of k = 8 and k = 64 samples verified on the context.
- **Program decodability:** a linear probe on the composer's state predicts the true function's minimal program on held-out worlds. Decodable but wrong means an execution-model error (open-loop); not decodable means an inference error.

**Instrumentation**, logged throughout training (lesson L6):
- loss and accuracy by stratum;
- gradient and update norms per layer;
- activation norms;
- attention entropy;
- the decoder's per-step entropy;
- for the execution-conditioned arm, attention to trace tokens.

## 7. Preliminary predictions (to be fixed in the preregistration)

1. **In support:** all arms approach the exact Bayes code length in the high-diversity cells, and fail in D2 / N1k. This is a diversity threshold, the Kirsch / Raventós phenomenon.
2. **Length 6:** the execution-conditioned composer beats the open-loop one, which beats the monolithic PFN. Open-loop's gap grows with length.
3. **Concept mixing:** S5-trained models fail on mixed functions, and fail less with execution conditioning.
4. **Depth cap:** composition beyond the training depth appears only above some cap and diversity, and the D-cells and N-cells separate depth from diversity.
5. **Seeing vs searching:** the gap between first proposal and best of 64 verified samples is small in support and large out of support.

## 8. Throughput and batching

The models are tiny and the sequences short (≤ 16 context tokens, ≤ 7 program tokens). On a T4, single runs would be bound by launch overhead and latency, not compute. So the design maximizes concurrency.

1. **Stack runs into one batched model,** as the fleet trainer does. Every run of the same architecture and size keeps its parameters along a leading run dimension, and every layer is a batched matmul (`einsum`/`bmm`) over runs. All 9 cells × 3 seeds of one arm (27 runs) train in one process, one optimizer step and one kernel per layer.
2. **Mix cells differ only in data.** Each run samples from its own function pool (an index tensor), so cells co-batch without padding.
3. **Data generation and execution on the GPU:**
   - all function tables (≈ 38k × 49 int8, plus ℓ = 6 for evaluation) and minimal-program lists (padded) live on the GPU;
   - episodes are built by indexing;
   - execution traces come from 7 × 7 operation tables (exact, because of clean-up);
   - no CPU round trips per step.
4. **Fixed shapes:** context 16 (with a mask for smaller k), programs padded to 7, a fixed batch size per run. No recompilation, and CUDA graphs or `torch.compile` are possible.
5. **Mixed precision:** fp16 autocast with gradient scaling. T4 tensor cores are about 8× faster than fp32. Embedding and attention widths should be multiples of 8.
6. **Two GPUs per Kaggle session:** one arm (or one stack) per GPU. The three arms take two sessions, or one session with the monolithic and open-loop stacks sharing a GPU.
7. **Batched evaluation:**
   - exact posteriors are a matrix test of context consistency (functions × context examples);
   - sampling k programs and verifying them is one batched execution;
   - probes are fitted on stored states.
8. **The throughput test comes before the preregistration.** Measure run-steps per second against stack size (1, 9, 27, 54), model width and fp16/fp32 on one T4. Then set the model sizes and training episodes so the grid fits two or three Kaggle sessions.

**Implemented and checked so far:**
- `src/topology_design/stacked.py`: stacked layers and a stacked monolithic PFN.
- `tests/test_stacked.py`, which checks that:
  - a stacked run equals the same model alone (to 1e-5);
  - a loss on one run gives exactly zero gradient to every other run;
  - queries can't see each other.
- **CPU check** (2 cores, width 64, 2 layers, not representative of a GPU): stacking gives 1.8× more run-steps per second than separate models at R = 9 and 27. On a CPU the work is compute-bound; on a T4 these models are launch-bound, which is where stacking should pay much more. The GPU number decides.
- **Kaggle throughput job:**

  ```
  python experiments/fleet/kaggle/launch.py --user riley9 --bench-worlds
  ```

  It runs the stacked tests on the GPU, then benchmarks:
  - width 128 and 256;
  - fp32 and fp16;
  - stacks of 1–108 runs;
  - 16–256 episodes per run;
  - against separate models.

## 9. Before the main runs (lessons L1, L4, L11, L12, L17, L18)

- **Planted checks:**
  - every library block is exhaustively correct;
  - the exact-Bayes code reproduces known posteriors on small hand cases;
  - an oracle composer (the exact posterior MAP) scores as computed;
  - each held-out split's certificate is recomputed independently.
- **Every prediction must be able to fail,** given the counts. Check that the grid can produce both verdicts.
- **Dip-robust criteria** for any threshold.
- **An independent pre-run review** of the design and code.

## 10. Still to fix in the preregistration

- final model sizes and training lengths (from the throughput test);
- exact verdict rules and effect thresholds;
- the minimum certified test-set size per split;
- ~~whether the ℓ = 6 enumeration fits~~: it does (297,845 functions × 49 int8 ≈ 15 MB).

---

## Independent review (26 Oct 2026, before preregistration)

The reviewer's check scripts are in `review/`. It confirmed:
- the counts (additive-only saturates at 343; 297,845 functions at ℓ ≤ 6, *including* the identity at ℓ = 0);
- the cell sizes;
- the stacked-model tests.

**Blocking findings:**
1. **The "in-support" split is not in support.** Held-out functions have zero prior under the cell's training prior. An ideal PFN trained on the pool fails them: Bayes on the pool gets 0.003 accuracy at 16 examples, against 1.000 under the generator prior. Prediction 1 means opposite things under the two references.
2. **Composers cannot express programs longer than the training cap.** The decoder learns HALT after `cap` steps, so the length-6 and beyond-depth predictions are fixed by the architecture.
3. **Monolithic vs composer is not a fair test of composition.** The composer alone gets latent program supervision and an exact interpreter. The open-loop composer never touches the library during training, so with exact clean-up the library plays no causal role. This is the synthesis-vs-induction comparison RobustFill already made. Parameters are matched, but compute is not.
4. **The structural splits are ambiguous and leak.**
   - The X5 pool is only 3,879 functions.
   - The "order" restriction excludes 90% of ℓ = 5.
   - Many test functions also have a compliant program of length ≤ 5.
   - "inc r1" is ambiguous (source or destination).
   - "Exactly one" violation is ambiguous.
5. **Concept mixing can only come out one way.** The S5 pool has 389 functions and the test set covers 99% of ℓ ≤ 5.
6. **The D and N cells don't separate depth from diversity.** Subsampling is unstratified, and repetitions per function covary with diversity.
7. **Posterior-fidelity KL is unestimable from samples** at small k, and the program-decodability label is undefined: only 1% of ℓ = 5 functions have a unique minimal program.
8. **3 seeds and one learning rate per stack** violate AGENTS rule 4 and L8; the stacked optimizer needs a per-run learning rate.

**Non-blocking:**
- seeing vs searching saturates at Bayes at 16 examples;
- ℓ = 6 weight under prior (b) and the identity are unspecified;
- inconsistent identifiability numbers (line 49 quotes ℓ ≤ 5 figures);
- CSR storage for 13.9M minimal programs at ℓ = 6;
- tiny strata at ℓ ≤ 2;
- `stacked.py` efficiency (embedding gather, mask broadcasting, padding mask for k < 16);
- the tests run on CPU, not GPU;
- the bench covers the monolithic arm only;
- no episode budget (about 1.5M episodes to see each ℓ = 5 function 10 times);
- missing controls (Bayes on the pool, the nearest-pool floor, a monolithic PFN with a program head).

**Status:** the design is under revision. Items 2, 3 and 5 change the science and go to the researcher. The rest are fixed in the revision.
