# Dynamic graphs over primitives vs a static-graph PFN: design (revision 2)

**Status:** design, 26 Oct 2026. Not yet a preregistration. Revision 1 (a composer trained on programs, against a monolithic PFN) was reviewed independently. Its review is kept at the end, with how each finding is handled here. The researcher then reframed the round (history Entries 30–31). This document replaces revision 1. It becomes `PREREGISTRATION.md` after a second independent review, a GPU throughput test and the pilot gates in §11.

**Ledger:** I25 (a generator of worlds), I26 (exact step classification; commitment vs averaging), following F19 (the data gap is the prior).

**Scope note (AGENTS hard rule 1).** The primary-comparison family in AGENTS (single-pass DAGs, no learned gates, no recurrence) is the Phase 0–1 family. This round deliberately compares a static graph with a dynamic, recurrent one, at the researcher's direction (LEDGER, Direction). Every claim below is labeled by type: expressivity, optimization or mechanism.

---

## 1. Question

A PFN is a **static graph**. For every query it runs the same computation and outputs an average over the worlds consistent with its context. The alternative is a **meta-model that builds a dynamic graph** for each context: a serial path through a library of primitive computations, ending in HALT. The path is then executed on the query, and its output is the prediction.

The working assumption is that the data come from reusable, composable, serializable primitives. If so, a learner that searches over compositions should extrapolate *which world it is in* better than one that averages in output space.

**Main question.** Both models see the same generated worlds and are trained on the same next-token loss. Does the dynamic-graph meta-model generalize to worlds outside its training distribution better than a static-graph PFN of equal size? Specifically:
- functions never seen in training;
- longer functions;
- held-out compound patterns.

**Sub-questions:**
1. **Which prior does each model learn?** On functions it never saw, does a model behave like Bayes over its training pool (memorization) or like Bayes over the generator (generalization)? How does that depend on the diversity and depth of the training mix?
2. **Does seeing the execution help?** Execution-conditioned paths see the partial program's register values on the context examples. Open-loop paths must predict them internally.
3. **How good are the branch decisions?** On Z_7, every step of every path can be classified exactly (§6.3). Where do the meta-model's errors come from: dead branches, wasted steps, ambiguity, or halting?
4. **When does it commit?** Does the meta-model keep many worlds alive while the context is ambiguous and narrow to one as it fills in, as Bayes does?

**What is known already.**
- *Neural program synthesis vs induction* (RobustFill 2017; DeepCoder; execution-guided synthesis, Chen 2019; ExeDec 2024): a synthesizer trained on programs vs an induction network trained on outputs.
- *Learning programs from denotations*, i.e. from outputs only, by marginal likelihood or REINFORCE (Liang 2017/2018, MAPO; Guu 2017).
- *Looped transformers* (Yang 2024; Fan 2025) and the fixed-depth limits of static transformers (Merrill & Sabharwal).
- *PFNs and meta-learned Bayes* (Müller 2022; Kirsch 2022; Raventós 2023; Learning Universal Predictors 2024).

**What is not known (to our reading; literature check pending for the preregistration).** We have not found work combining all of the following:
- a dynamic graph over primitives and a static PFN, trained on *the same* next-token objective, so the two losses are the same prequential code length;
- comparison to *exact* Bayes references under both the training prior and the generator prior;
- every branch decision classified exactly.

The second literature check before the preregistration must confirm this.

**Not in this round:**
- neural primitives (the library is a symbolic interpreter, by decision; later rounds learn it);
- discovering primitives;
- memory stores;
- multi-token outputs;
- refining a path across positions instead of rebuilding it.

## 2. Worlds and the task

**Machine.** Values in Z_7, two registers. At the start r0 = x and r1 = y; the output is r0 after HALT.

**Instructions (32 + HALT):**
- `dest ← src1 op src2` for op ∈ {add, sub, mul}, with dest, src1, src2 ∈ {r0, r1};
- `dest ← f(src)` for f ∈ {neg, inc}.

**Execution is exact.** A register pair takes 49 values, so each instruction is a 49-entry transition table. Executing a path on any batch of inputs is a chain of table lookups on the GPU.

**Functions** are 49-entry output tables, enumerated by breadth-first search over register states (`count_isa.py`, confirmed by the reviewer's `review/isa_enum.py`):

| minimal length ℓ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| functions | 13 | 80 | 552 | 4,179 | 32,741 | 260,279 |
| minimal programs | 16 | 254 | 3,956 | 63,002 | 936,732 | ≈13.9M |
| share with a unique minimal program | 77% | 13% | 6% | 2% | 1% | — |

The identity (ℓ = 0) is excluded from every pool and test set. It is one function, and with equal weight per length it would take a whole stratum.

**A world** is one function f.

**An episode** is a sequence of n = 16 pairs ((xᵢ, yᵢ), zᵢ = f(xᵢ, yᵢ)), with inputs drawn from the 49 without replacement.

**The task is next-token prediction of each zᵢ** from the earlier pairs and (xᵢ, yᵢ). Training loss summed over positions is the episode's prequential (predict-then-learn) code length. That makes the loss of every arm directly comparable with exact Bayes (§6.1).

**The generator prior.** Equal weight per minimal length ℓ = 1…6, then uniform over the functions of that length. A cell's **training prior** restricts this to its pool and depth cap, keeping equal weight per length up to the cap. Programs are never sampled for training: worlds are functions. A minimal program is drawn uniformly only for the path-supervised control arm (§4).

## 3. Test sets

Every test set is a set of *functions*, certified by enumeration.

| test | functions | which Bayes is the right reference |
|---|---|---|
| **pool** | fresh episodes of training-pool functions | Bayes on the cell's training prior (a) |
| **unseen in range** | 10% of functions at each ℓ ≤ 5, removed from every pool | Bayes on the generator prior (b); (a) shows memorization |
| **length** | ℓ = 6 (a sample of 5,000, fixed) | (b) |
| **pattern A / B** | functions certified to need a held-out compound pattern (below) | (b) |

**Unseen in range** answers sub-question 1. A model at the optimum of its training loss behaves like Bayes(a), which puts zero mass on unseen functions. (The reviewer found Bayes(a) at 0.003 accuracy where Bayes(b) is at 1.000.) Moving from (a) toward (b) is exactly the generalization in question. The primary endpoint is each arm's excess code length over Bayes(b). Bayes(a) assigns zero probability to outputs no pool function gives, so its code length on unseen functions is infinite. The memorization index is therefore defined on accuracy (§7.2). This replaces revision 1's "in-support" split, which was not in support (review item 1).

**Compound patterns** follow DBCA (CFQ), at the level of functions:
- A **pattern** P = (A, B) is a data-dependent bigram: an instruction of op A writes a register, and a later instruction of op B reads that register while it still holds A's output.
- A program is **compliant** if it never executes P.
- `certify_patterns.py` enumerates compliant programs over (register state, freshness flag) to get each function's minimal compliant length l_c. The last layer (length 7) is not stored, only checked against the candidates.
- Each function falls into one class:
  - **Pattern cell pool:** functions with ℓ ≤ 5 and l_c ≤ 5, i.e. trainable without P.
  - **P-test:** l_c > 7. Every program the meta-model can build (T_max = 7) uses P. The certificate holds at the model's full step budget.
  - **P-minimal:** ℓ ≤ 5 and l_c ∈ {6, 7}. Every *minimal* program uses P, but a longer compliant program exists. This is a secondary test: does a model trained without P find the longer route?
- **The confound, and the remedy.** Certified P-test functions are almost all ℓ = 6 (table below). Pattern is therefore confounded with length. The reported effect is a difference in differences, compared against the size-matched control cell C (§5):

  DiD = [acc_C(P-test) − acc_C(compliant ℓ = 6)] − [acc_P(P-test) − acc_P(compliant ℓ = 6)]

  Here "compliant ℓ = 6" means functions with l_c = 6. A positive DiD means that removing P from training costs more on functions that need P than on equally long functions that don't.

Certified counts (`certify_patterns.py 7`):

| pattern | pool (share of ℓ ≤ 5) | P-test, ℓ ≤ 5 | P-test, ℓ = 6 | P-minimal (ℓ ≤ 5) |
|---|---|---|---|---|
| **mul→inc** | 33,239 (88.5%) | 4 | 9,555 | 4,322 |
| **inc→mul** | 30,988 (82.5%) | 16 | 4,492 | 6,561 |
| inc→inc | 36,653 (97.6%) | 0 | 1,213 | 912 |
| mul→add | 31,305 (83.3%) | 0 | 836 | 6,260 |
| sub→mul | 27,271 (72.6%) | 0 | 4,963 | 10,294 |
| add→mul | 26,156 (69.6%) | 0 | 5,604 | 11,409 |
| mul→sub | 21,309 (56.7%) | 0 | 21,946 | 16,256 |
| mul→mul | 21,262 (56.6%) | 695 | 54,061 | 15,608 |
| add→add | 33,707 (89.7%) | 0 | 34 | 3,858 |
| neg→mul | 37,547 (100%) | 0 | 0 | 18 |
| mul→neg | 37,451 (99.7%) | 0 | 0 | 114 |

Pools are counted before the unseen-in-range hold-out; ℓ ≤ 5 has 37,565 functions. `check_certify_bruteforce.py` checks the search against brute force over all 32⁴ programs: 4,825 functions, 0 mismatches.

**What the certification shows.** A function that needs a pattern for its *minimal* program almost always has a longer compliant program within 7 steps. Certified P-test functions are therefore ℓ = 6 almost without exception, which is why the difference in differences is needed. neg→mul and mul→neg never need certification: neg is always replaceable.

**The chosen patterns** are **A = mul→inc** and **B = inc→mul**: the same two atoms composed in opposite orders. They are the only patterns meeting the size criteria below other than inc→inc, whose pool would barely differ from Nall. The atom-divergence check below is still to be run on them.

**The criteria.** Each pattern must have:
- ≥ 1,000 certified P-test functions;
- a pool that keeps ≥ 80% of ℓ ≤ 5;
- similar operation frequencies (atoms) in the pool and in the P-test minimal programs (DBCA's low atom divergence), measured before choosing.

One pattern per cell, so a failure is attributable to one held-out pattern (review item 4).

## 4. Arms

Every arm shares the **context encoder**:
- each pair (xⱼ, yⱼ, zⱼ) is one token from a single joint embedding table of 343 entries;
- causal attention, so one pass encodes every prefix of the sequence;
- width d and depth set by the throughput test, ≈ 1–3M parameters in total;
- parameters matched across arms, within 5%.

### Primary arms

**S: static PFN.** For each position i, a query token embed(xᵢ, yᵢ) attends to the encoded pairs before i. A head outputs a distribution over Z_7. Loss: cross-entropy on zᵢ at all 16 positions. Its optimum is the posterior predictive (Bayes(a)).

**O: dynamic graph, open-loop.** For each trained position i:
1. A small step decoder (2–3 layers) builds a path a₁, a₂, …, HALT autoregressively, up to T_max = 7 instructions. Each step attends to the encoded context and to the previous steps.
2. The path is executed on (xᵢ, yᵢ).
3. The prediction is the mixture over paths, q(z | context) = Σ_paths π(path | context) · [exec(path, xᵢ, yᵢ) = z].

**E: dynamic graph, execution-conditioned.** The same as O, except that after each step the partial path is executed on every context input xⱼ, yⱼ (j < i). Each example's current registers and target form one trace token, embedded from a 343-entry joint table of (r0, r1, zⱼ). The next step attends to these trace tokens as well. With the joint embedding, "register matches target" is linear in the token, so the trace is cheap: the trace tokens don't attend to each other.

**Paths are blind to the query.** In both O and E, the path depends only on the context, never on (xᵢ, yᵢ):
- E's trace covers context examples only.
- This is what makes a path a *committed world*. A path that could see the query could choose a constant program per query, which is a PFN in disguise.
- It also means one sampled path can be evaluated on all 49 inputs, which gives a full sampled function (§7).
- Both arms can still reach the posterior predictive: a posterior over programs, mixed and executed on the query.

**Training objective for O and E: marginal likelihood through execution.**
- Sample K paths per trained position.
- A path *hits* if its output equals zᵢ.
- The loss estimates −log q(zᵢ | context). Its gradient is −Σₖ wₖ ∇ log π(pathₖ), with wₖ = hitₖ / Σ hits (zero if nothing hits).
- This is maximum marginal likelihood from samples (Guu 2017): the same objective as S, estimated by sampling.
- Programs and distances are never given.
- At initialization a random path hits with probability about 1/7, so the signal exists from the first step; the pilot gate checks it is enough (§11).
- O and E train on P random positions per episode (default 4) with K samples (default 8). S trains on all 16 positions. Control S4 (below) removes this difference.

**Evaluating O and E.** q(z | context) is computed by best-first enumeration of paths in order of probability, until the covered mass is ≥ 0.999 or 4,096 paths. The residual mass δ is spread uniformly over Z_7 and reported. This bounds the code length from above, with no smoothing parameter to tune.

**Halting.** HALT is a token like any other.
- E can learn to halt once its trace matches the targets on every context example. That rule transfers to any length.
- O cannot verify, so on lengths beyond its training cap it may stop early because of halting, not because of inference.
- Diagnostic (§7): evaluate with HALT masked until step L, for L = cap … 7. If accuracy rises when halting is delayed, the failure was in halting (expressivity), not inference.

### Controls (in 3 cells: N400, Nall, Pattern A)

- **SL: static, compute-matched.** The PFN's query pathway is a weight-tied block looped T_max = 7 times, with the encoded context re-injected at each loop, as in looped transformers. It has the same per-path serial compute as a dynamic path, but the graph is static.
  - If SL matches E, extra compute and not dynamic routing explains any advantage.
  - It does not match the K samples. Those buy a better estimate of the mixture, not more computation per world.
- **S4: static, positions-matched.** S trained only on the same P random positions per episode that the dynamic arms train on.
- **EP: execution-conditioned with path supervision.** E plus cross-entropy on a minimal program sampled uniformly for the true function: the amortized posterior over programs (inference compilation).
  - It is given privileged information, so it is labeled as such.
  - It shows how much of any gap is credit assignment rather than inductive bias.

A dense-distance arm (exact step values as targets, I26) is deferred: it would make sense only if EP shows a large credit-assignment gap.

## 5. Training mixes (9 cells)

Equal weight per minimal length up to the cap, then uniform over the cell's functions of that length. The unseen-in-range hold-out is removed from every pool first. After the hold-out, ℓ ≤ 3 has ≈ 580 functions; ℓ = 4 has ≈ 3,761; ℓ = 5 has ≈ 29,467.

| cell | cap | ℓ ≤ 3 | ℓ = 4 | ℓ = 5 | purpose |
|---|---|---|---|---|---|
| D3 | 3 | all | — | — | depth axis |
| D4 | 4 | all | 400 | — | depth axis |
| N100 | 5 | all | 100 | 100 | diversity axis |
| N400 | 5 | all | 400 | 400 | diversity axis; depth axis at cap 5 |
| N1600 | 5 | all | 1,600 | 1,600 | diversity axis |
| Nall | 5 | all | all | all | diversity axis |
| PA | 5 | pattern-A pool | | | pattern A held out |
| PB | 5 | pattern-B pool | | | pattern B held out |
| C | 5 | Nall minus a random, length-stratified removal of the same size as PB's (6,577 functions before hold-out) | | | size-matched control for PA and PB |

**Depth vs diversity (review item 6).** Small strata (ℓ ≤ 3) are always complete. Diversity varies only at ℓ = 4 and 5, with the same per-length weights in every cell, so cells on the diversity axis differ only in how many distinct functions each length offers.

The depth axis (D3, D4, N400) holds the per-stratum count at 400. Total pool size still grows with cap (≈ 580 / 980 / 1,380), and this is stated, not hidden.

Every cell trains on the same number of episodes, so repetitions per function fall as diversity grows. Checkpoints let cells be compared at matched repetitions per function as well as at matched episodes.

**Grid:**
- primary arms S, O, E × 9 cells × 10 seeds = 270 runs;
- controls SL, S4, EP × 3 cells × 10 seeds = 90 runs;
- 360 runs in total, before the pilot.

If the throughput test says this doesn't fit, cut in this order: controls to 2 cells; then N1600 and D4. Never go below 10 seeds.

## 6. Exact references

### 6.1 Bayes predictive

For a context, the version space is the set of functions consistent with it: one matrix test over the function table.

The Bayes predictive under:
- (a) the cell's training prior;
- (b) the generator prior.

The prequential code length of each is computed exactly for every evaluation episode. Bayes(b) needs all 297,844 non-identity functions at ℓ ≤ 6: ≈ 15 MB as int8.

### 6.2 Bayes-optimal step policy

Under a prior over functions, with a minimal program chosen uniformly per function, every prefix has an exact posterior mass: the posterior-weighted share of minimal programs that extend it. The Bayes step policy is

π*(a | prefix, context) = mass(prefix + a) / mass(prefix).

It is computed along the meta-model's own sampled prefixes, using stored minimal programs (≈ 1.0M at ℓ ≤ 5, ≈ 13.9M at ℓ = 6, in CSR form).

When the model's prefix leaves the support of minimal programs, π* is undefined there. That step is counted as off-minimal (it cannot be on a minimal path), and the KL is reported only on support.

### 6.3 Branch classes (I26)

For a step from register state s to s′, with t steps used and budget T_max = 7, let d(·) be the remaining distance to the target. The step's regret is 1 + d(s′) − d(s).

| class | definition |
|---|---|
| optimal | regret 0 |
| lateral | regret 1: one step lost |
| away | regret ≥ 2, but t + 1 + d(s′) ≤ T_max |
| budget-fatal | t + 1 + d(s′) > T_max, but a completion exists at some length |
| dead | s′ maps two inputs to the same registers although the target separates them; nothing can ever undo this |

Distance can fall by at most one per step, so every step that is not optimal loses the minimal length for good. Two targets:
- **Oracle:** the true function, on all 49 inputs.
- **Version space:** any function consistent with the context; the BFS then checks only the context inputs, which is much cheaper.

"Dead relative to the context" (two *context* inputs merged) can be seen through execution, so E can see it. "Dead only on unseen inputs" needs the prior. The difference between the two answers sub-question 2 mechanistically.

**Measured on the starting state** (`branch_classify.py`, 25 targets of minimal length 4, 32 first moves each):

| class | share of first moves |
|---|---|
| optimal | 5.9% |
| lateral | 24.9% |
| away or worse | 69.3% |
| dead (included in "away or worse") | 42.2% |

**Computing distances** is a breadth-first search from s′ with the remaining budget, deduplicated on the GPU, stopping at the first hit. The number of distinct states grows about 9× per step: 376,202 at 5 steps from the start (measured), so an estimated ≈ 3.4M at 6 and ≈ 30M at 7. Dead steps need no search. Results are cached by (state, target). Classification runs on a fixed sample of evaluation episodes at the final checkpoint and at three earlier ones.

## 7. Metrics

All metrics are stratified by test set, by minimal length, and by context size k = 0…15.

1. **Code length** (nats per position) against Bayes(a) and Bayes(b): the primary endpoint. Accuracy (argmax) is reported as well.
2. **Which prior the model tracks.** On unseen-in-range functions:
   - the excess code length over Bayes(b), ΔL_b (primary);
   - a memorization index on accuracy at k ≥ 8:

     ρ = (acc_b − acc_arm) / (acc_b − acc_a)

     Here acc_a is the accuracy of Bayes(a), falling back to a uniform guess once no pool function is consistent with the context.
   - ρ = 1 means it behaves like a memorizer of the pool; ρ = 0 means it behaves like Bayes over the generator.

   Reported per cell, to get the curve over diversity and depth (the Kirsch / Raventós transition, measured against exact references).
3. **Commitment and coherence (O and E only).** A sampled path is a whole function: evaluate it on all 49 inputs.
   - **Valid-world rate:** the share of sampled functions consistent with the context. Bayes gives 1.
   - **Effective number of worlds:** the exponentiated entropy of the sampled-function distribution, against the same for the Bayes(b) posterior, as k grows.
   - S has marginals only, so it gets the corresponding predictive-entropy curve.
4. **Step decisions (O and E):**
   - class rates (§6.3) by step index, length stratum, test set and k, for both targets;
   - dead-relative-to-context vs dead-only-on-unseen;
   - KL(π* ‖ π) on support, and the off-minimal rate.
5. **Halting:** accuracy with HALT masked until L = cap … 7; path-length distribution against minimal length.
6. **Probes on S** with labels intrinsic to the function, not to programs: minimal length; whether the output depends on y; whether f is affine; f at a held-out input. Linear probes, with label-permutation nulls. (Revision 1's program-decodability label was undefined; review item 7.)
7. **Instrumentation** throughout training (L6):
   - loss by stratum;
   - gradient, update and activation norms per layer;
   - attention entropy;
   - the step decoder's per-step entropy;
   - the hit rate of the K samples;
   - attention mass on trace tokens (E).

## 8. Preliminary predictions

These are to be fixed in the preregistration with thresholds and bootstrap rules.

| # | prediction | falsifier | claim type |
|---|---|---|---|
| P1 | On pool test sets, every primary arm comes within 0.05 nats per position of Bayes(a) at k ≥ 8 in Nall. | S, O or E stays further away at the end of training. If so, this is an optimization failure, and P2–P4 are reported only as conditional. | optimization (sanity) |
| P2 | On unseen in range, ΔL_b(E) < ΔL_b(S) and ΔL_b(O) < ΔL_b(S) in every cell, with the same order in ρ. The gap is largest at N100 and N400. | ΔL_b(S) ≤ ΔL_b(E) at N100 and N400 (paired bootstrap, P > 0.9). | mechanism (inductive bias) |
| P3 | On length (ℓ = 6), E > O > S in accuracy at k = 15 for Nall and N400. O's gap shrinks when HALT is delayed. | S ≥ E. Or O's gap does not change with delayed HALT (then halting is not the cause). | E vs O: mechanism; O halting: expressivity |
| P4 | DiD(S) > DiD(E) for both patterns. | DiD(E) ≥ DiD(S) for both. | mechanism |
| P5 | E's rate of dead-relative-to-context steps is far below O's. Most remaining errors of E are dead-only-on-unseen steps at k ≤ 6. | E's dead-relative-to-context rate is ≥ half of O's. | mechanism |
| P6 | Valid-world rate → 1 as k grows, and the effective number of worlds tracks Bayes(b) within a factor of 2 on pool functions. | Effective number collapses to 1 at small k (premature commitment) or stays high at large k. | mechanism |
| P7 | SL performs like S on P2–P4. | SL matches E on P2 or P3. Then compute, not dynamic routing, explains the advantage. | mechanism (control) |

**Before preregistering, check that every prediction can fail (L17).** Bayes(a) and Bayes(b) are computable now. Planted models give both verdicts of every rule:
- an oracle path sampler from π*;
- a memorizer of the pool;
- a uniform predictor.

## 9. Fairness and seeds (AGENTS rule 4, L8)

- **Learning rate per arm.** A pilot sweep of 3 learning rates × 3 seeds on D3 and N400 picks each arm's learning rate. The stacked optimizer takes a per-run learning rate tensor, so one stack can hold several learning rates.
- **Seeds.** Pilot null distribution first: identical arms, different seeds. Then 10 paired seeds per cell × arm. Verdicts use the run-level bootstrap P(A > B).
- **Matching.** Results are compared at matched training episodes, at matched training loss, and at matched repetitions per function. Compute (FLOPs and wall time) is reported per arm, because the dynamic arms cost more per episode by construction.
- **Equal objective.** Every primary arm minimizes the same next-token loss on the same episodes. Only EP gets privileged information, and it is labeled.
- **Capacity.** Every arm must fit its pool at the end of training (P1). An arm that doesn't fit is reported as such, not compared.

## 10. Throughput and batching

The models are small and the sequences short, so a single run on a T4 is bound by launch overhead.

1. **Stack runs.** All runs of one arm (cells × seeds × learning rates) share one batched model. Parameters carry a leading run dimension and every layer is a `bmm` (`src/topology_design/stacked.py`, tested for exact equivalence and independence). Cells differ only in their function-pool index tensors.
2. **Everything on the GPU:**
   - function tables, pools, hold-out masks and minimal programs (for EP);
   - episode sampling by indexing;
   - path execution by gathering from the 33 × 49 transition table, with no CPU round trip.
3. **Cost of the dynamic arms per episode:** P × K × T_max decoder steps (default 4 × 8 × 7 = 224 single-token steps), each attending to ≤ 15 context tokens and ≤ 15 trace tokens. Against 32 tokens for S, that is roughly an order of magnitude more per episode. Trace tokens are embedding lookups with no per-token layers. The K samples of one position batch together, and so do positions.
4. **Fixed shapes:** sequences of 16, paths padded to 8, masks for k < 16 (padding mask added; review). fp16 autocast with gradient scaling; widths are multiples of 8.
5. **Efficiency items from the review:**
   - `F.embedding` with per-run offsets instead of gather on an expanded table;
   - masks broadcast instead of expanded;
   - a padding mask;
   - stacked tests also run on the GPU.
6. **The throughput job (`launch.py --bench-worlds`) is extended to all arms** (S, SL, O, E). It measures run-steps per second against stack size, width, P and K, then sets the model sizes and the episode budget.
   - Target: ≈ 1.5M episodes per run, about 10 sightings of each ℓ = 5 function in Nall.
   - The grid should fit in a handful of two-GPU Kaggle sessions.
7. **Offline exact references:** certification, the function table, minimal programs and Bayes code lengths are computed once on CPU and shipped with the job. Branch-class searches run on the GPU after training, on a fixed sample.

## 11. Before the main runs

**Planted checks:**
- The executor's transition tables equal the symbolic semantics on all 49 × 33 cases.
- The Bayes code reproduces hand-computed posteriors on small cases.
- An oracle path sampler drawn from π* reaches the Bayes(b) code length under the best-first estimator. This validates the estimator.
- The branch classifier agrees with brute-force enumeration for budgets ≤ 3.
- The pattern certificates are recomputed independently.

**Pilot gates:**
1. **Learning gate.** O and E, trained by marginal likelihood alone, must come within 0.1 nats per position of Bayes(a) at k ≥ 8 on D3 within the pilot budget. If they don't, the training estimator is revised (K, an entropy term, a leave-one-out baseline) and the change is recorded. Path supervision is *not* added to the primary arms; it stays the EP control.
2. **Null gate.** The between-seed spread of identical arms sets the minimum detectable effect for P2–P4.
3. **Predictions can fail.** The planted models of §8 give both verdicts.

**An independent pre-run review** of this design and of the code.

## 12. Still open for the preregistration

- Model sizes, P, K and the episode budget (from the throughput test).
- Thresholds and bootstrap rules for P1–P7.
- The atom-divergence check on patterns A and B (§3).
- The size of the evaluation sample for the branch-class diagnostics.
- Whether SL's loop count should also be swept (1, 3, 7).

---

## Independent review of revision 1 (26 Oct 2026), and how revision 2 handles it

The reviewer's check scripts are in `review/`. It confirmed:
- the counts (additive-only saturates at 343; 297,845 functions at ℓ ≤ 6 including the identity);
- the cell sizes of revision 1;
- the stacked-model tests.

| # | finding (blocking) | handled in revision 2 |
|---|---|---|
| 1 | The "in-support" split was not in support: Bayes on the pool gets 0.003 accuracy, against 1.000 for Bayes on the generator. | Renamed **unseen in range**. Both references are reported, and a memorization index on accuracy measures where each arm sits between them (§3, §7.2). Fresh episodes of pool functions are the in-support test. |
| 2 | Composers could not express programs longer than the training cap. | T_max = 7 in every cell. HALT can be learned by verification (E). A delayed-HALT diagnostic separates halting from inference. Halting claims are labeled expressivity (§4, P3). |
| 3 | Unfair: only the composer got program supervision and an exact interpreter; the library played no causal role; compute was not matched. | Every primary arm trains on the same next-token loss. The dynamic arms learn *through execution*, so the library is causal. Path supervision is only the EP control. SL matches serial compute; S4 matches trained positions. Compute is reported (§4, §9). |
| 4 | Structural splits were ambiguous and leaked; pools too small. | Replaced by DBCA-style compound patterns, certified at the model's full step budget (l_c > 7). One pattern per cell, a size-matched control, and a difference in differences against equally long compliant functions (§3, §5). |
| 5 | Concept mixing could only come out one way (pool of 389, test covering 99%). | Dropped. Patterns keep ≥ 80% of the pool by construction. |
| 6 | D and N cells didn't separate depth from diversity. | Strata ℓ ≤ 3 are always complete; diversity varies only at ℓ = 4 and 5 with fixed per-length weights; depth axis at a fixed per-stratum count; comparisons at matched repetitions (§5). |
| 7 | Posterior KL unestimable at small k; program-decodability label undefined. | Predictive KL against exact Bayes per query; function-level coherence from sampled paths; step-level KL against the exact π*; probes on S use function-intrinsic labels (§6, §7). |
| 8 | 3 seeds and one learning rate per stack. | 10 paired seeds after a null pilot; a learning-rate sweep per arm; a per-run learning-rate tensor in the stacked optimizer (§9). |

**Non-blocking findings, handled:**
- The identity is excluded.
- ℓ = 6 weight under (b): equal per length.
- Identifiability numbers are corrected (§2 no longer quotes them).
- CSR storage for minimal programs.
- `stacked.py` efficiency items and GPU tests (§10.5).
- The bench is extended to all arms.
- Episode budget: ≈ 1.5M.
- Controls: Bayes on the pool is reference (a); the pool memorizer is a planted model.

**Seeing vs searching** saturated at Bayes at 16 examples, so it is replaced by the commitment and step-decision metrics.
