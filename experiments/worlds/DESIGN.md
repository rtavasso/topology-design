# Committed paths vs superposed programs over shared primitives: design (revision 3)

**Status:** design, 26 Oct 2026. Not yet a preregistration. Revision 3 answers the independent review of revision 2 (kept at the end) and the researcher's decisions in history Entries 33–34:
- give the static model the same primitives;
- add a per-step cost;
- mask the held-out pattern in every arm that uses the library;
- fewer cells, more seeds.

It becomes `PREREGISTRATION.md` after a third independent review, a GPU throughput test of all arms and the pilot gates in §12.

**Ledger:** I25 (a generator of worlds), I26 (exact step classification; commitment vs averaging), following F19 (the data gap is the prior).

**Scope note (AGENTS hard rule 1).** The primary-comparison family in AGENTS is the Phase 0–1 family. This round deliberately compares static and dynamic graphs, at the researcher's direction. Every claim is labeled by type: expressivity, optimization or mechanism.

---

## 0. What changed from revision 2

| revision 2 | revision 3 | why |
|---|---|---|
| A static PFN that must learn mod-7 arithmetic, against dynamic paths over exact primitives | A factorial. The no-library PFN (N) is a reference. Superposed programs (SO, SE) and committed paths (CO, CE) use the same primitives. | Revision 2 confounded access to primitives with commitment (review item 2; Entry 33). |
| Different objectives per arm | One objective for every arm: held-out log loss given a context prefix | Fairness |
| Nothing favors short programs | A per-step cost | It breaks ties among equivalent programs and makes compute allocation measurable (review; Entry 34). |
| Best-first evaluation estimator | Monte Carlo over sampled paths, each executed on all 49 inputs | Depends only on the induced distribution over functions (review item 1). |
| Each sampled path scored on one point | Each path scored on every remaining point of the episode; the initial path length biased short | Spurious hits and the constant-output attractor (review item 4). |
| Pattern held out from the pool only | Pattern **masked** in every library arm during training, unmasked at test | It had leaked into the dynamic arms (review item 3). |
| ρ on accuracy | A fitted memorization weight, measured only where it is defined | ρ sat at chance (review item 5). |
| 9 cells × 3 arms × 10 seeds | 6 cells × 5 arms × 10 seeds, plus 29 seeds for the headline pair | Feasibility and AGENTS rule 4 (review). |

## 1. Question

The researcher's motivation (Entry 34): a library of primitives serialized into paths gives a combinatorial family of deep computations, |primitives|^L of them. The parameter cost is linear and the compute per token is linear. A dynamic graph also lets compute go where a world needs it.

A static network can contain combinatorially many paths too, but it **sums** them in superposition; a dynamic graph **selects** one. Given the same primitives, the question is whether selection beats superposition.

**Main question.** All arms get the same primitives, the same generated worlds and the same objective. Does a meta-model that **commits** to one path per context generalize better and track Bayes more closely than one that **superposes** all paths in a fixed graph?

**Sub-questions:**
1. **What do the primitives buy?** Each library arm against a PFN that has to learn the arithmetic in its weights. This is a prior by construction, so it is labeled as such. Its size is the measurement.
2. **Memorization vs generalization.** On functions never seen in training, does each arm behave like Bayes over its training pool or like Bayes over the generator? How does that change with diversity?
3. **Feedback.** Does seeing the partial program's registers on the context help, within each kind of composition?
4. **Compute allocation.** Do the steps used track each world's minimal length?
5. **Branch decisions and commitment.** On Z_7, every step can be classified exactly (§7.3). When does a committed model narrow from many worlds to one?

**What this round cannot test.** Whether depth helps *representation*. On Z_7 with 49 inputs, every function is a 49-entry table, and composing maps on 49 states has shallow shortcuts. Testing that claim needs unbounded, inherently serial inputs; it is deferred to a sequence-input domain (Entry 34, D55).

**Prior work, and what is new.**
- **Module networks:** NMN (Andreas 2016), IEP (Johnson 2017), N2NMN (Hu 2017), Stack-NMN (Hu 2018), NS-CL (Mao 2019).
  - With the right layout given, modular networks generalize systematically (Bahdanau 2019, SQOOP).
  - Learning layouts from answers is where it fails: induced layouts did not converge to tree layouts, and soft parametrizations blurred (SQOOP). CLEVR-trained models, an NMN given ground-truth programs included, fail CLOSURE.
  - Relaxed program search loses to discrete solvers (TerpreT, Gaunt 2016).
  - DARTS relaxes architecture choice into a softmax over operations, which is what our superposed arm does over primitives.
- **Two Bayes references along a diversity axis:** Raventós 2023.
- **Amortized program posteriors with symbolic execution:** DreamCoder (Ellis 2021).

What we have not found combined: superposed and committed composition over *exact* primitives, trained on the *same* objective, measured against *exact* Bayes references under two priors, with every branch decision classified exactly. The contribution is framed as a measurement. The second literature check before the preregistration must confirm it.

**Not in this round:** neural primitives; discovering primitives; memory stores; multi-token outputs; refining a path across positions.

## 2. Worlds and the task

**Machine.** Values in Z_7, two registers. At the start r0 = x and r1 = y; the output is r0 at the end.

**Instructions:** 32, plus a 33rd token that ends the path:
- `dest ← src1 op src2` for op ∈ {add, sub, mul}, with dest, src1, src2 ∈ {r0, r1};
- `dest ← f(src)` for f ∈ {neg, inc}.

The 33rd token is **HALT** in committed paths, where it ends the path, and **NOP** in superposed programs, where it is an identity step. A superposed program that puts all its weight on one token per step is exactly a committed path with its NOPs removed.

**Execution.** Each instruction is a 49-entry transition table on register pairs.

**Functions** by minimal length (enumerated; `rev3/enum6.py` reproduces the table in 30 s):

| ℓ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| functions | 13 | 80 | 552 | 4,179 | 32,741 | 260,279 |
| minimal programs | 16 | 254 | 3,956 | 63,002 | 936,732 | 13,866,340 |

The identity is excluded. **A world** is one function.

**An episode** is 16 pairs ((xᵢ, yᵢ), zᵢ) with distinct inputs, each pair one token from a 343-entry joint table.

**The generator prior.** Equal weight per length ℓ = 1…6, then uniform within a length. A cell's **training prior** is the generator prior restricted to its pool and depth cap.

## 3. Test sets

| test | functions | right reference |
|---|---|---|
| **pool** | fresh episodes of training functions | Bayes(a): the cell's training prior |
| **unseen in range** | 10% of each length ≤ 5, removed from every pool (581 / 3,761 / 29,467 functions remain at ℓ ≤ 3 / 4 / 5) | Bayes(b): the generator prior; Bayes(a) for the memorization weight |
| **length** | ℓ = 6, a fixed sample of 5,000 | Bayes(b) |
| **pattern A** | functions every program of which, within 7 steps, uses mul→inc | Bayes(b), as a difference in differences |

**The pattern hold-out** (`certify_patterns.py`; reproduced independently by the reviewer):
- A pattern A→B means an instruction of op A writes a register, and a later op B reads it while it still holds A's output.
- For mul→inc:
  - the pattern cell's pool keeps functions with a compliant program of length ≤ 5 (33,239 before the hold-out);
  - the P-test has 9,555 certified functions at ℓ = 6 (and 4 at ℓ ≤ 5);
  - a secondary P-minimal set has 4,322 functions at ℓ ≤ 5 whose minimal programs all use mul→inc, but which have a compliant program of length 6–7.
- **Masking:** during training in the pattern cell, every library arm is forbidden from executing mul→inc.
  - In committed paths, a freshness flag per register is tracked along the path. The instruction that would complete the pattern gets logit −∞, and the softmax renormalizes over the rest.
  - In superposed programs, the belief is tracked over (register state × freshness flags) = 49 × 4 states per input. In each flag state, the step's instruction weights are renormalized over the allowed instructions.
  - The two arms therefore apply the same masked, renormalized policy.
  - At test the mask is lifted. Every atom has been used in other contexts; only the compound is new.
- The no-library PFN cannot be masked, so for it this is a hold-out of a class of functions. Its result is reported separately and kept out of the masked comparisons.
- **Effect measure.** The control cell C_A removes 4,326 functions at random, stratified by length (the same number as the pattern pool removes), with no mask. The effect is

  DiD = [acc_C(P-test) − acc_C(compliant ℓ = 6)] − [acc_P(P-test) − acc_P(compliant ℓ = 6)],

  where "compliant ℓ = 6" means functions with a compliant program of length 6. This separates the pattern from length, since certified P-test functions are almost all ℓ = 6.

## 4. Arms

**Shared context encoder.** Causal attention over the pair tokens, so one pass encodes every prefix. Width and depth are set by the throughput test, ≈ 1–3M parameters. Parameters are matched across arms within 5%.

**One objective for every arm.** Per episode, draw P split points i (default P = 2). Each arm predicts every remaining output zⱼ, j ≥ i, from the context c<i and the input (xⱼ, yⱼ). The loss is

  L = mean over splits and j ≥ i of −log q(zⱼ | c<i, xⱼ, yⱼ).

Its optimum for every arm is the Bayes posterior predictive under the training prior. Evaluation uses the j = i terms: the prequential code length, Σᵢ −log q(zᵢ | c<i, xᵢ, yᵢ). At evaluation, every arm's predictive gets the same floor: q ← (1 − 10⁻³) q + 10⁻³ / 7.

| arm | uses the library | composition | feedback | how q is computed |
|---|---|---|---|---|
| **N** | no | none (a PFN) | — | exact (softmax head) |
| **SO** | yes | superposed, static graph | open | exact |
| **SE** | yes | superposed, static graph | soft execution trace | exact |
| **CO** | yes | committed paths, dynamic graph | open | Monte Carlo |
| **CE** | yes | committed paths, dynamic graph | execution trace | Monte Carlo |

**N, no-library PFN.** Query tokens embed(xⱼ, yⱼ) attend to c<i; a head outputs logits over Z_7. It keeps revision 2's role as the reference for what the primitives buy.

**SO, superposed program, open-loop.**
- A step decoder, the same module as the committed arms', produces weights w_t ∈ Δ³³ for t = 1…7. Its inputs are the encoded context and a step-index embedding only.
- For every one of the 49 inputs, a belief over the 49 register states starts at the input's own state and is updated as b ← b · Σₐ w_t(a) Tₐ.
- The prediction for input x is the r0-marginal of the final belief.
- Every primitive is computed at every step, so the graph is static and nothing is committed.
- Its prediction is a mixture over paths with a step-factorized distribution Πₜ w_t(aₜ). That family cannot represent "program A or program B" without the junk cross-terms (A's first step, B's second).
- It trains on exact cross-entropy.

**SE, superposed program with feedback.** As SO, but the weights at step t also attend to **soft trace tokens**. For each context example j < i, the token is the belief-weighted embedding Σₛ b_t[xⱼ](s) · E[s, zⱼ], using the 343-entry table. The graph stays static.

**CO, committed paths, open-loop.**
- The step decoder samples a path a₁, a₂, …, HALT autoregressively (at most 7 instructions), attending to the encoded context and to its own previous steps.
- Paths never see the query, so a sampled path is a whole function, executed on all 49 inputs at once.
- Its prediction is q(z | x) = Σ_paths π(path | c<i) · [exec(path, x) = z].

**CE, committed paths with feedback.** As CO, plus a hard trace token per context example after each step, embedded from the 343-entry table as (r0, r1, zⱼ).

**Step cost** (library arms). A cost λ_step per executed instruction, NOP and HALT excluded. The default λ_step = 0.01 nats; the pilot checks 0.003 and 0.03.
- In committed arms the cost enters inside the marginal likelihood, as a factor e^(−λ_step·len) on each path's score. The optimal path distribution then puts its mass on minimal programs.
- In superposed arms it is added as λ_step · E[len], computed exactly.
- It breaks ties among equivalent programs and makes "steps used" mean something.

**Training the committed arms.**
- K = 16 paths are sampled per split.
- Each path is executed on all 49 inputs and scored on every remaining point j ≥ i.
- The gradient is −Σⱼ Σₖ wₖⱼ ∇log π(pathₖ), with wₖⱼ = hitₖⱼ e^(−λ len_k) / Σₖ′ hitₖ′ⱼ e^(−λ len_k′). Points with no hit are skipped.
- This is maximum marginal likelihood per point, and it is the same objective as the other arms. A path matching many points gets weight on many points.
- **Pilot alternatives,** tried in this order if the learning gate fails (§12):
  - noisy-channel joint scoring, Πⱼ[(1 − ε) if hit, else ε/6], with ε ∈ {0.1, 0.3};
  - β-tempered weights;
  - uniform exploration with importance correction;
  - K = 64.

  Program supervision is never added to the primary arms.

**Initialization.** The HALT logit (committed) and the NOP logit (superposed) start with a bias that gives an expected initial path length of ≈ 3. The reason is that constant-output paths, which carry no information, grow with length (§5.2).

**Evaluating the committed arms.** M = 1,024 paths are sampled per context prefix, and each is executed on all 49 inputs:

  q̂(z | x) = (1/M) Σₘ [execₘ(x) = z].

This estimator depends only on the distribution over *functions* the sampled paths induce. It is therefore invariant to how probability spreads over equivalent programs, the failure of revision 2's best-first enumeration. It is validated before use (§12, G1).

**Control EP** (N400 only, 10 seeds): CE plus cross-entropy on a minimal program of the true function drawn uniformly. It has privileged information and is labeled as such. It measures how much of any gap is credit assignment.

## 5. Derived before measuring (AGENTS rule 2)

Scripts are in `experiments/worlds/rev3/`.

### 5.1 How much superposition can lose (`superpos.py`)

For each context, a superposed program (7 steps × 33 weights) is fitted directly to the exact Bayes predictive on all 49 inputs: 3 restarts, 400 Adam steps. The prior is equal weight per length over all functions of ℓ ≤ 5, standing in for Bayes(a) in Nall. 20 episodes per k. A committed-path mixture can match Bayes exactly at every k (a mixture of minimal programs).

| k | effective number of consistent functions | best superposed KL to Bayes (nats per input), median | max | share > 0.05 |
|---|---|---|---|---|
| 0 | 3,006 | 0.001 | 0.001 | 0.00 |
| 1 | 416 | 0.022 | 0.032 | 0.00 |
| 2 | 70 | 0.039 | 0.102 | 0.35 |
| 3 | 13.6 | 0.053 | 0.209 | 0.55 |
| 4 | 1.8 | 0.031 | 0.638 | 0.35 |
| 6 | 1.1 | 0.005 | 1.383 | 0.30 |
| 8 | 1.0 | 0.001 | 1.363 | 0.25 |
| 12 | 1.0 | 0.001 | 1.363 | 0.25 |

**Two separate effects:**
- **Expressivity:** while a handful to a hundred worlds remain (k ≈ 1–3), the factorized family stays 0.02–0.05 nats per input from Bayes even when fitted directly.
- **Optimization:** once the world is identified (k ≥ 6), a one-hot program reaching zero KL always exists, yet in 25–30% of contexts the fit lands in a local optimum (up to 1.4 nats). That replicates TerpreT's finding in this setting.

These are fits per context, not amortized learning, so they bound what the arms can do rather than predicting it. They set the effect sizes in P3 and P4.

### 5.2 The training signal at initialization (`initsig.py`)

With a uniform policy (most paths run to 7 steps), on targets of length 4–5, K = 16, scored on all remaining points:
- the mean agreement of a random path with the target is 0.18;
- 33.5% of random paths have a constant output;
- 7.5% of episodes contain a path exactly right on every remaining point.

| scoring | weighted agreement | weight on constant paths | weight on exact paths |
|---|---|---|---|
| per-point marginal (primary) | 0.351 | 0.266 | 0.065 |
| noisy-channel, ε = 0.1 | 0.445 | 0.342 | 0.068 |
| noisy-channel, ε = 0.3 | 0.419 | 0.360 | 0.056 |

Two conclusions:
- Scoring on all remaining points roughly doubles the agreement of the paths that receive gradient.
- Constant-output paths keep a quarter to a third of the weight. They are the collapse risk.

| random path length | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| share with constant output | 0.07 | 0.13 | 0.19 | 0.23 | 0.30 | 0.33 | 0.37 |

Constant outputs come from long, information-destroying paths: dead branches. Hence the short-length initialization, the step cost, and a pilot gate on the constant-output share (§12).

### 5.3 Where the memorization weight is defined (`memo.py`)

Unseen functions of length 4–5, 300 episodes per cell. Each entry is the share of contexts where the pool's version space is non-empty, then the share also having TV(Bayes(a), Bayes(b)) > 0.1 at the next query:

| cell | k = 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| N100 | 1.00 / 0.34 | 1.00 / 0.90 | 0.90 / 0.82 | 0.45 / 0.40 | 0.19 / 0.12 | 0.09 / 0.03 | 0.02 / 0.01 |
| N400 | 1.00 / 0.07 | 1.00 / 0.63 | 0.97 / 0.87 | 0.58 / 0.51 | 0.22 / 0.15 | 0.10 / 0.06 | 0.02 / 0.01 |
| Nall | 1.00 / 0.03 | 1.00 / 0.07 | 1.00 / 0.29 | 1.00 / 0.62 | 0.86 / 0.70 | 0.44 / 0.35 | 0.10 / 0.06 |

The memorization weight is estimable at k ≈ 2–4 in N100 and N400, and at k ≈ 3–6 in Nall. In Nall the two references differ less: the pool is nearly the generator up to length 5.

## 6. Training mixes (6 cells), grid and seeds

| cell | cap | ℓ ≤ 3 | ℓ = 4 | ℓ = 5 | mask | purpose |
|---|---|---|---|---|---|---|
| D3 | 3 | all | — | — | — | depth: training only on short programs |
| N100 | 5 | all | 100 | 100 | — | diversity |
| N400 | 5 | all | 400 | 400 | — | diversity; headline |
| Nall | 5 | all | all | all | — | diversity; headline |
| PA | 5 | pattern-A pool | | | mul→inc | pattern |
| C_A | 5 | Nall minus 4,326 at random, length-stratified | | | — | control for PA |

Every cell gives equal weight per length up to its cap, then uniform within a length. Every cell trains on the same number of episodes; comparisons are made at matched episodes and at matched repetitions per function.

**Grid:**
- 5 arms × 6 cells × 10 seeds = 300 runs;
- the headline pair CE and SE, raised to 29 seeds in N400 and Nall: +76;
- EP in N400, 10 seeds: +10.
- **386 runs,** plus the pilot.

## 7. Exact references

### 7.1 Bayes predictive

The version space is one matrix test over the function table (297,844 functions × 49, 14.6 MB). Bayes(a) and Bayes(b) give exact predictives and prequential code lengths for every evaluation episode.

### 7.2 Bayes-optimal steps

With the step cost, the optimal committed policy puts its mass on minimal programs. Among those it is still not unique, so π* (uniform over minimal programs, weighted by the posterior) is a reference distribution, not *the* optimum.

KL(π* ‖ π) and the off-minimal rate are diagnostics. They are not preregistered endpoints.

### 7.3 Branch classes: a partition with precedence

For a step from state s to s′, with t steps used and T_max = 7, the classes in order of precedence are:
1. **dead:** s′ maps two inputs the target separates to the same registers. This is sufficient, not necessary, for unreachability.
2. **no completion within budget:** t + 1 + d(s′) > T_max, or no completion found within the remaining budget.
3. **away:** regret 1 + d(s′) − d(s) ≥ 2.
4. **lateral:** regret 1.
5. **optimal:** regret 0.

Two targets:
- **Version space (primary):** the state is distinguished only on the context inputs, so the search is cheap.
- **Oracle:** all 49 inputs, on a smaller sample. A search from a step-1 state reaches ≈ 3.3M states.

"Dead, visible in the context" (two *context* inputs merged) is what execution feedback can see.

## 8. Metrics

All stratified by test set, minimal length and context size k.

1. **Code length** (nats per position) against Bayes(a) and Bayes(b): primary. Accuracy too.
2. **Memorization weight.** For each run, fit λ_mem ∈ [0, 1] minimizing the mean KL(q_arm ‖ λ·Bayes(a) + (1 − λ)·Bayes(b)) over unseen-in-range positions where the pool's version space is non-empty and TV > 0.1 (§5.3). Report λ_mem and the residual KL (how well the mixture explains the arm).
3. **Commitment and coherence.** Paths sampled from any library arm are whole functions:
   - committed arms: from π;
   - superposed arms: one token per step drawn from w_t.

   On these, report:
   - **valid-world rate:** the share of sampled functions consistent with the context. Bayes gives 1; the junk cross-terms of superposition lower it.
   - **effective number of worlds:** the exponentiated entropy of the function distribution from M samples, compared with the same estimator on M samples from the Bayes(a) posterior, so both are capped at M.
4. **Compute allocation:** steps used (committed) or expected non-NOP steps (superposed) against the true function's minimal length ℓ, by k.
5. **Branch decisions** (committed arms): class rates (§7.3) by step, length and k; dead-visible-in-context vs dead-only-unseen.
6. **Halting:** the path-length distribution against ℓ; accuracy with HALT masked until L = cap … 7. The second is readable only where early halting is visible.
7. **Probes on N** with labels intrinsic to the function: minimal length, dependence on y, affinity, f at a held-out input. Linear, with label-permutation nulls.
8. **Instrumentation:**
   - loss by stratum;
   - gradient, update and activation norms per layer;
   - attention entropy;
   - step-decoder entropy;
   - hit rate and **share of constant-output paths** (committed);
   - entropy of the per-step weights (superposed);
   - attention to trace tokens.

## 9. Preliminary predictions

Thresholds are to be fixed in the preregistration. Run-level paired bootstrap; "P > 0.9" means P(A better than B) > 0.9 across seeds.

| # | prediction | falsifier | type |
|---|---|---|---|
| P1 | Every arm comes within 0.05 nats per position of Bayes(a) on pool functions at k ≥ 8 in Nall. | An arm stays further away. Its comparisons are then reported as conditional. | optimization (sanity) |
| P2 | Every library arm beats N (lower excess over Bayes(b)) on unseen in range and on length, in N100, N400 and Nall. | N is better than or equal to any library arm at N400 (P > 0.9). | expressivity (prior by construction) |
| **P3** | **Headline.** Where 3–300 functions remain consistent (k ≈ 1–3), CE beats SE by ≥ 0.02 nats per position, on pool and unseen in range, in N400 and Nall. Derived best case: 0.02–0.05 (§5.1). | CE − SE > −0.01 nats per position in both cells (P > 0.9). | expressivity of the path distribution |
| P4 | Where the world is identified (≤ 1.5 effective functions), CE and SE agree within 0.02 on average. SE has a tail of failures (excess > 0.05 in ≥ 10% of contexts), and CE's tail is smaller. Derived: superposed fits fail in 25–30% of identified contexts (§5.1). | SE's failure share is no larger than CE's. | optimization |
| P5 | On length (ℓ = 6) and pattern DiD, CE beats SE by ≥ 0.02 nats per position at k ≥ 8. | SE beats CE. | mechanism |
| P6 | Within each composition, feedback helps (E better than O) on ℓ ≥ 4 and on length. CE's dead-visible-in-context step rate is at most half CO's. | O ≥ E, or the dead-step ratio is > 0.5. | mechanism |
| P7 | λ_mem falls from N100 to Nall in every arm. Library arms have lower λ_mem than N in N100 and N400. | λ_mem(N) ≤ λ_mem(library) in either cell. | mechanism |
| P8 | In CE at k ≥ 8 on pool functions, steps used track ℓ: slope ≥ 0.8, mean excess ≤ 0.5 steps. | Slope < 0.8 or excess > 0.5. | mechanism (compute allocation) |
| P9 | Committed arms reach a valid-world rate ≥ 0.9 at k ≥ 4. Superposed arms have a lower valid-world rate than committed ones at k = 1–3. | Superposed ≥ committed at k = 1–3. | expressivity |

**Planted models give both verdicts of every rule (L17):**
- an oracle committed sampler (π*);
- the best per-context superposed fit (§5.1);
- a pool memorizer;
- a uniform predictor;
- a random-path predictor (AGENTS' random-routing control).

## 10. Fairness and seeds (AGENTS rule 4, L8)

- **Learning rate per arm:** a pilot sweep of 3 learning rates × 2 seeds × 5 arms in N400. The stacked optimizer takes a per-run learning-rate tensor.
- **Seeds:** a null pilot first. Then 10 paired seeds per cell and arm, and 29 for CE and SE in N400 and Nall (the headline).
- **Matching:** results compared at matched episodes, matched training loss and matched repetitions per function. FLOPs and wall time are reported per arm.
- **Same objective and data** for every primary arm. Only EP is privileged.
- **The library is shared** by all four library arms, so the commitment comparison is not affected by the library being the generator's own instruction set. That fact affects only P2, which is labeled as a prior by construction.
- **Not done, and why:**
  - an initialization-scale sweep (no closed-form richness parameter for these arms);
  - an operator-label permutation (all library arms share the library, so it would not separate them).
- **Capacity:** every arm must fit its pool (P1).

## 11. Throughput and the feasibility gate

- **Stacking.** All runs of one arm share a batched model, with a leading run dimension (`src/topology_design/stacked.py`). Pools are index tensors, and data, execution and masks live on the GPU.
- **Cost per episode:**
  - **N:** 16 context tokens and 2 × 16 query tokens.
  - **SO/SE:** 7 steps of a single-token decoder per split. Belief propagation is (P, 49 inputs, 49 or 196 states) × 33 tables per step, which is negligible. Soft trace tokens are one matrix product per step.
  - **CO/CE:** P × K × 7 single-token decoder steps (2 × 16 × 7 = 224), plus gather-based execution.
- **Feasibility gate.** The throughput job is extended to all five arms. The grid of 386 runs must fit in about three weeks of Kaggle quota (≈ 90 GPU-hours). If it doesn't, cut in this order:
  1. the episode budget, guided by where the pilot's learning curves plateau;
  2. K;
  3. CO and SO in the PA and C_A cells.

  Seeds are never cut below 10, and the headline's 29 are kept.

## 12. Before the main runs

**Planted checks (tests first):**
- The transition tables match the symbolic semantics.
- The superposed belief update equals the exact path mixture on small cases.
- The masked policies of superposed and committed arms are identical on small cases.
- The Bayes code reproduces hand cases.
- The branch classifier agrees with brute force for budgets ≤ 3.
- The pattern certificate (done; reproduced by the reviewer).

**Pilot gates:**
- **G1 estimator.** The Monte Carlo estimator with M = 1,024 gives the same code length, within 0.005 nats per position, for:
  - an oracle sampling minimal programs;
  - the same oracle spread over equivalent padded paths.

  Its gap to the exact Bayes predictive is also reported as a function of M.
- **G2 learning.** CO and CE, on N400 functions of length 4–5, come within 0.1 nats per position of Bayes(a) at k ≥ 8 within the pilot budget. The constant-output share must fall below 10%. If either fails, the estimator alternatives of §4 are tried in order, and the change is recorded.
- **G3 superposed fit.** SO and SE reach P1 on the pool. Failure is reported (it bears on P4), and their comparisons are then conditional.
- **G4 null.** The spread between seeds of identical arms sets the minimum detectable effect for P3–P5.
- **G5.** The planted models of §9 give both verdicts of every rule.

**An independent pre-run review** of the design and of the code.

## 13. Still open for the preregistration

- Model sizes, P, K and the episode budget (from the throughput test).
- λ_step and the threshold values.
- The atom-divergence check for mul→inc (pool vs P-test operation frequencies).
- The sample size for branch-class diagnostics.
- Whether a second pattern cell (inc→mul) fits the budget.

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

---

## Independent review of revision 2 (26 Oct 2026)

The reviewer's scripts are in `review2/`. **Verified correct:**
- the function and minimal-program counts (13,866,340 minimal programs at ℓ = 6; 3,321,056 register states at depth 6);
- the mul→inc and inc→mul rows of the pattern table, reproduced exactly with an independent search and an independent layer-7 expansion;
- that the layer-7 shortcut is sound, and the freshness-flag semantics;
- the hold-out arithmetic: 581 / 3,761 / 29,467 functions remain at ℓ ≤ 3 / 4 / 5;
- that the pattern removals fall almost entirely at ℓ = 5.

**Blocking findings:**
1. **The best-first evaluation estimator is wrong, and it is not an upper bound.** The training objective gives no reason to prefer one of many equivalent programs, and any minimal program padded with r1-writes is still correct, so probability spreads over many 7-step paths.
   - An oracle sampling only minimal programs is 0.003 nats per position from Bayes(b), sometimes below it.
   - The same Bayes-optimal oracle spread over equivalent 7-step paths is 0.94 nats per position from Bayes(b) (≈ 1.9 at k ≥ 8 for ℓ ≤ 3).
   - P1 would fail from the estimator alone. Fix: Monte Carlo over sampled paths, each executed on all 49 inputs (possible because paths ignore the query), validated on both oracles. A tie-break prior over equivalent programs (a step cost or a length prior) is needed if π*, branch classes and halting are to mean anything. (`estimator.py`, `estimator_junk.py`)
2. **The library is the generator itself, so P2 and P3 come out largely by construction.** O and E execute the generator's own instructions with enough steps, so their output space *is* the generator's support, and they never learn mod-7 arithmetic; S must. Nothing separates dynamic routing and commitment from a hard-coded true hypothesis class, and SL does not control for it. AGENTS rule 4 also asks for an operator-label permutation and a random-routing control. Suggested:
   - a mismatched or label-permuted library arm;
   - a **soft-path static arm**, with per-step instruction distributions that don't depend on earlier choices. Its mixture is an exact Markov chain over the 49 register states, so it trains on exact cross-entropy, with the same library and no commitment.

   Until then, P2 and P3 are "prior by construction", not "mechanism".
3. **The held-out pattern is not held out from the dynamic arms.** Worlds are functions, so O and E can emit the pattern whenever it hits. In PA, 19.1% of ℓ = 5 pool functions have a minimal program using mul→inc, and 3.5% of training mass is on functions whose every minimal program uses it. For PB the figures are 30.6% and 5.0%. Also, C matches PB's removal size, not PA's. Fix: mask the pattern online in the O and E decoders during PA and PB training, or drop the DBCA framing. (`pools.py`)
4. **Sampled maximum marginal likelihood will probably not learn from a random start.**
   - The initial hit rate is 0.175, but 97% of hits are spurious (95% in D3).
   - 27% of random paths output constant 0, which hits 24% of ℓ = 5 positions: a collapse attractor.
   - Self-normalized weights give a lone spurious hit full weight.

   Guu 2017, Johnson 2017 and Hu 2017 all hit this problem. Fix: score each path on all remaining points of the episode (same expected loss, because inputs are exchangeable; use the same held-out form for S). With 8 points the hit rate is 0.6%, and about 84% of hits are exact. Add β-meritocratic weights or ε-exploration, and run the learning gate on ℓ = 4–5, not D3. (`init_hits.py`)
5. **ρ doesn't measure memorization.** On unseen ℓ = 4/5 functions at k = 8, the pool version space is empty in 98% of N100/N400 contexts, so acc_a is at chance and a uniform predictor scores as a perfect memorizer. Fix: measure where the pool version space is non-empty (k ≤ 4–6), on predictive distributions, for example by the mixture weight λ in q ≈ λ·Bayes(a) + (1 − λ)·Bayes(b). (`rho.py`)

**Non-blocking findings:**
- **Branch classes are not a partition.** Use the precedence dead > no completion within budget > away > lateral > optimal. Merging two inputs is a *sufficient* condition for unreachability, not a necessary one (the comment in `branch_classify.py` is now corrected). A completion "at some length" can't be certified.
- **π\* is not unique,** since any spread over equivalent programs is Bayes-optimal. KL(π* ‖ π) and the off-minimal rate are diagnostics only, and halting gets no learning signal.
- **P6** should use Bayes(a) on pool functions. The effective number of worlds from M samples is capped at M, so compare against the same estimator applied to matched-size Bayes samples.
- **Prediction wording.** P2 says "every cell" but is falsified only at N100 and N400. P7 needs an equivalence margin.
- **AGENTS rule 4:** 10 seeds against ≈ 29 for headline claims; no initialization-scale sweep; no random-routing control.
- **Small strata.** The unseen strata at ℓ = 1 and 2 hold 1 and 8 functions.
- **Feasibility.**
  - O, E and SL have no code and no bench yet.
  - `StackedPFN` is still revision 1's model.
  - Estimated at ≈ 20–60 min per dynamic run on a T4, i.e. 70–200 GPU-hours for O, E and EP, against about 30 GPU-hours a week on Kaggle.
  - The pilot is outside the 360-run count.
  - Suggested cuts: D4, N1600, O in the pattern cells, controls at N400 only, EP in one cell. Spend the savings on ≈ 29 seeds for S vs E at N400 and Nall.
- **Novelty.** Prior work already covers much of this:
  - Raventós 2023 measures against two Bayes references along a diversity axis;
  - Bahdanau 2019 (CLOSURE), Johnson 2017 and Hu 2017 compare dynamic module graphs with static networks, trained from answers, and learned layouts did not generalize better;
  - Li et al. 2024 compare induction and transduction on ARC;
  - ExeDec's concept-order split resembles the A/B choice;
  - Hewitt 2020 learns program posteriors from data.

  The combination (exact Bayes, the same prequential loss, exact step classes) may still be new, framed as a measurement contribution.

**Status:** items 2 and 3 change what the experiment can claim and go to the researcher. Items 1, 4 and 5 and the non-blocking items will be fixed in revision 3.

**How revision 3 handles the review of revision 2:**

| # | finding | revision 3 |
|---|---|---|
| 1 | The best-first estimator fails when probability spreads over equivalent paths | Monte Carlo over sampled paths, executed on all 49 inputs, which is invariant to equivalent programs. Gate G1 validates it on both oracles. A step cost breaks ties (§4, §12). |
| 2 | The library is the generator, so P2/P3 come out by construction | The static model gets the same primitives (superposed arms SO and SE). The headline compares commitment against superposition over a shared library. P2 (library vs no library) is labeled a prior by construction (§0, §4, §9). |
| 3 | The held-out pattern leaked into the dynamic arms; C was mismatched to PA | The pattern is masked in every library arm during training. C_A is matched to PA's 4,326 removals. Only pattern A this round (§3, §6). |
| 4 | Spurious hits and the constant-output attractor | Paths are scored on all remaining points, initialized short, with a step cost; there is a gate on the constant-output share and an ordered list of estimator alternatives. The learning gate is on ℓ = 4–5 in N400 (§4, §5.2, §12). |
| 5 | ρ sat at chance | A fitted memorization weight, measured only where it is defined (§5.3, §8). |
| non-blocking | Branch classes not a partition; π* not unique; P6 reference; prediction wording; seeds; feasibility; novelty | A precedence order; π* demoted to a diagnostic; Bayes(a) and matched-sample estimators for commitment; falsifiers per cell; 29 seeds for the headline; a feasibility gate with an ordered cut list; the contribution framed as a measurement against the module-network literature (§1, §7, §9–§11). |
