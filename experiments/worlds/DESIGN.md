# A PFN over model composition: design (revision 4)

**Status:** design, 26 Oct 2026. **The independent review of revision 4 judged it not worth running as designed (at the end of this document). Direction goes to the researcher.** Revision 4 follows the researcher's reframing (history Entries 36–40) and the papers read in Entry 39. The previous revisions and their three reviews are kept at the end. Next steps: an independent review of this design and its experimental value, then code, a throughput test, the pilot gates, and the preregistration.

**Ledger:** I25 and I26, following F19 (the data gap is the prior).

**Scope note (AGENTS hard rule 1).** This round deliberately studies dynamic, recurrent composition, at the researcher's direction. Claims are labeled expressivity, optimization or mechanism.

---

## 0. What changed, and why

| from revision 3 | to revision 4 | source |
|---|---|---|
| The path framed as a *hypothesis about the world*, with commitment vs superposition as the headline | The router framed as an **amortized posterior over the model's own composition**: a PFN whose output is a composition, not a prediction | Entries 36, 40 |
| "Committed vs superposed" as the headline | Headline: **can the composition posterior be learned from next-token sequences alone**, and how close does it get to a structure-supervised composition PFN? Soft vs committed becomes a secondary axis, with a static mixture-of-chains control. | Review 3 finding 1; Entry 40 |
| Committed arms trained by sampled marginal likelihood, shown not to learn (0 of 30 at ℓ ≥ 3) | Committed arms trained by **expert iteration** (AlphaZero): a policy and a value head, short search over branches with exact symbolic execution, outcomes judged by consistency with the context and by the next token | Review 3 finding 3; Entry 39 |
| No structure-supervised arm | **CP:** the router trained on compositions sampled from the prior. This is literally a PFN at the structure level, and the upper reference for learnability. | Entry 40 |
| Different step-cost forms per arm | One form: a factor e^(−λ) per executed instruction, inside every library arm's score | Review 3 finding 4 |
| A Monte Carlo evaluator only for committed arms | The same evaluator for every library arm, with sample counts and floors set by the derived bias table | Review 3 finding 5 |
| Memorization weight against the uncapped Bayes(b) | Bayes(b) restricted to ℓ ≤ 5, and a 3-component fit (Bayes(a), Bayes(b), uniform) | Review 3, non-blocking |
| New endpoints | Structure-level fidelity against the exact posterior; learned value against exact distance; search vs policy; routing-only adaptation | Entries 38–40; Neural Interpreters; AlphaZero |

## 1. Thesis and questions

**Thesis (Entry 40).** A PFN amortizes the posterior predictive over *outputs*. The proposal is to amortize the posterior over *how the model composes its primitives* for each token, conditioned on the context. The model learns where it wants to go structurally, and the output is produced by running the composition.

This is Solomonoff's decomposition made explicit: a prior over programs, with prediction by execution. On the training support, it has the same optimum as an output-level PFN. It differs in what must be learned (only *which* composition), and in how it can extrapolate (to compositions never seen whole, whose parts still execute exactly).

**Questions:**
1. **Learnability.** Can a composition posterior be learned from next-token sequences alone, with no compositions ever shown? How close does it come to a composition PFN trained on compositions sampled from the prior?
2. **Soft vs committed composition.** Soft routing, as in Neural Interpreters, superposes primitives and gets exact gradients. Committed routing trained by search, as in AlphaZero, gets one path per sample and a learned value. A static mixture of soft chains separates "factorized vs correlated path distribution" from "commitment" (review 3, finding 1).
3. **Composition level vs output level.** A composition PFN against an output-level PFN with no library. This comparison is advantaged by construction, because the library is the generator's own instruction set. It is labeled as such, and its size is the measurement.
4. **Seeing vs searching.** How much does test-time search over branches add to the learned policy? Does the learned value agree with the exact distance on worlds never seen?
5. **Structure-level fidelity.** How close is the router's distribution over compositions to the exact posterior over compositions, including compositions never seen whole in training?

**What this stage cannot test,** with the reason for each:
- **Routing at every token, and primitives that read the context.** In pair worlds, x and y are uniform and inputs never repeat within an episode. Only the z predictions carry information, and computing z needs only the current pair. Stage B (§13) uses worlds where each token depends on earlier tokens.
- **Whether depth helps representation.** Functions on 49 inputs have shallow shortcuts (Entry 34). Stage B.
- **Learned primitives,** and the reuse × selection 2×2 with neural layers. A later stage.

## 2. Prior work, and what is new

| work | what it does | what it lacks relative to this design |
|---|---|---|
| PFNs and meta-learned in-context learning (Müller 2022; Kirsch 2022; Raventós 2023) | Amortize the posterior predictive over outputs. Raventós measures against two Bayes references along a diversity axis. | No composition. |
| Schug et al. 2024, *When can transformers compositionally generalize in-context?* | Tasks built from shared modules. A transformer whose output feeds a hypernetwork, separating task inference from execution, generalizes to held-out module combinations; a plain transformer does not, even though the task code is linearly decodable from it. | Soft, linear module mixing per task, not serialized per-token routes. No exact Bayes or structure-level references. The closest result in spirit. |
| Schug et al. 2025, *Attention as a hypernetwork* | Attention scores across heads act as a latent code configuring a value network. A modified attention (HYLA) improves compositional generalization on symbolic Raven matrices and fuzzy logic. | Composition stays implicit. |
| Neural Interpreters (Rahaman et al. 2021) | Soft per-token routing through shared, code-programmed functions, reused across depth. Routing-only fine-tuning recovers R² 0.986 of 0.995. | No in-context world prior. No exact references. Correspondence between functions and primitives never verified. Out-of-distribution gains modest. |
| Compositional Recursive Learner (Chang et al. 2019) | A controller picks modules from a library step by step, with halting and a step cost. PPO on a sparse reward, modular arithmetic. Generalizes to longer expressions. | Not meta-learned in context. Needed a curriculum. No exact references. |
| Modular meta-learning (Alet et al. 2018) | Learns a module library across tasks and *searches* compositions for each new task (simulated annealing). Meta-learning at the composition level. | Search per task, not amortized. |
| Module networks (Andreas 2016; Johnson 2017; Hu 2017; Bahdanau 2019) | With given layouts, systematic generalization. Learned layouts fail or need program supervision. | Not in-context. No exact references. |
| MoEUT (Csordás 2024); Mixture-of-Recursions (Bae 2025); Dr.LLM (2025) | Shared layers with routing in language models. Dr.LLM trains per-layer skip/execute/repeat routers on labels from offline MCTS. | Efficiency-oriented. No world prior. No exact references. |
| DreamCoder (Ellis 2021); AlphaDev and AlphaTensor | An amortized program posterior with symbolic execution; AlphaZero-style search over instruction-level programs. | AlphaDev and AlphaTensor search per task. DreamCoder's recognition model is the nearest relative of CP. |

**What we have not found combined:**
1. a PFN whose amortized object is the posterior over its *own* compositions, trained on a prior over worlds with the next-token loss;
2. compared, at matched size, with a composition PFN trained on prior samples, with soft routing (single and mixture of chains), and with an output-level PFN;
3. measured against *exact* posteriors at both the output level and the structure level, with an exact value for every branch decision;
4. with the router trained by expert iteration and context verification. CRL used PPO on the final answer; Dr.LLM used offline search labels.

The contribution is framed as a measurement. A final literature check precedes the preregistration.

## 3. Worlds, tokens and the task

- **Worlds:** unchanged. Functions of Z_7 × Z_7 → Z_7 with minimal programs over 32 instructions (2 registers). Counts by length: 13 / 80 / 552 / 4,179 / 32,741 / 260,279. The generator prior gives equal weight per length, then uniform within a length.
- **Tokens:** each pair (x, y, z) becomes three tokens, so an episode of 16 pairs is a sequence of 48 tokens. The loss is the next-token loss over the whole sequence.
- **Which positions carry information.** The next token after a z or an x is a fresh uniform input, so every arm predicts it with the shared output head (exact Bayes there is uniform). The next token after a y is z = f(x, y). That prediction is the one the composition makes: at each y position, the router builds a composition from the preceding tokens and executes it on (x, y).

  On pair worlds this is the only place composition matters, and §1 states why.

## 4. Test sets

As in revision 3; certificates are reproduced by two reviewers.

| test | functions | right reference |
|---|---|---|
| pool | fresh episodes of training functions | Bayes(a) |
| unseen in range | 10% of each length ≤ 5, held out of every pool | Bayes(b) |
| length | ℓ = 6, a fixed sample of 5,000 | Bayes(b) |
| pattern A | every program of length ≤ 7 uses mul→inc (9,555 certified); masked in every library arm during training | difference in differences, in nats, against the size-matched control C_A |

The memorization weight uses Bayes(b) restricted to ℓ ≤ 5, and a three-component fit: Bayes(a), capped Bayes(b), uniform.

## 5. Arms

All arms share the context encoder: causal attention over tokens, with a joint 343-entry embedding per pair or per-token embeddings. Router and decoder sizes are matched within 5%.

| arm | composition | how it learns | information beyond next-token data |
|---|---|---|---|
| **N** | none (an output-level PFN) | next-token loss | none |
| **CP** | committed, autoregressive router over the 32 instructions + HALT | cross-entropy on a minimal program of the true function, drawn uniformly | compositions sampled from the prior (the PFN-at-structure-level target) |
| **CX** | committed, autoregressive router with a policy head p and a value head v | expert iteration | none |
| **CS-1** | soft, one superposed chain of 7 steps | exact next-token loss | none |
| **CS-8** | soft, a static mixture of 8 chains | exact next-token loss | none |
| *CO* (pilot only, N400) | committed | DAgger toward the optimal next steps relative to what the context entails | privileged (exact distances) |

**CP, the composition PFN.** Its optimum is the posterior over minimal programs given the context. That target is unique because the prior over compositions is defined: uniform over the minimal programs of each function. Predictions come from sampled compositions executed on the query.

**CX, expert iteration.**
- At each y position, S simulations of PUCT search (default 64) run over the 33 actions, up to depth 7. Execution is exact and symbolic.
- Dirichlet noise is added at the root, with α scaled to the branching factor (AlphaZero uses α ≈ 10 / branching).
- Leaves are evaluated by v. A terminal score is the fraction of context examples the composition reproduces, times e^(−λ·len).
- Values are averaged in the search, not maximized (Entry 39).
- Training targets:
  - p is trained toward the root visit counts;
  - v is trained toward the outcome: whether the composition's output equals the actual next token, which is the ordinary next-token label, combined with context consistency.
- No composition is ever shown.
- At test, CX is evaluated both policy-only ("seeing") and with search S ∈ {16, 64, 256} ("searching").

**CS-1 and CS-8, soft composition.** Each step mixes all 33 instruction tables with weights from the router, acting on a distribution over register states for each input (as Neural Interpreters mixes functions by compatibility). CS-8 runs 8 such chains and mixes them at the output.
- CS-1 cannot represent correlations between steps.
- CS-8 can, approximately, without committing.

The gap CS-1 → CS-8 is the "factorized vs correlated" effect. Only CX − CS-8 may be credited to commitment.

**Common to all library arms:**
- the step cost factor e^(−λ_step) per executed instruction, inside the score (default 0.01; the pilot checks 0.003 and 0.03);
- pattern-A masking in PA cells: freshness flags on committed paths, flag-augmented states for soft chains (equivalent; review 3).

**Collapse.**
- Root noise and visit-count sampling during CX training.
- An entropy floor in the pilot for CS.
- The share of constant-output compositions is monitored in all arms (§12).

## 6. Derived before measuring

Scripts are in `rev3/`, `rev4/` and `review3/`.

### 6.1 The foothold: what search alone can find (`rev4/mcts_search.py`)

PUCT with a uniform prior and no learned policy or value; random rollouts; k = 8 context examples; 12 targets per length (13 at ℓ = 1). A success is a program consistent with all 8 examples.

| ℓ | S = 64 | S = 256 | S = 1,024 |
|---|---|---|---|
| 1 | 10/12 | 12/12 | 12/12 |
| 2 | 3/12 | 5/12 | 9/12 |
| 3 | 1/12 | 2/12 | 2/12 |
| 4 | 0/12 | 0/12 | 0/12 |
| 5 | 1/12 (not the true function) | 0/12 | 0/12 |

Search alone gets a foothold at ℓ ≤ 2, which carries 40% of the training mass under equal weight per length, and almost none beyond. Expert iteration must bootstrap: the learned prior and value have to extend search to ℓ = 3–5. Nothing yet shows that it does. This is gate G2.

### 6.2 Soft composition (review 3 `superpos2.py` and `superpos_mix.py`, scored on non-context inputs like the objective)

Best per-context fit, median KL to Bayes in nats per input:

| | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| one chain | 0.010 | 0.021 | 0.042 |
| static mixture of 8 chains | 0.001 | 0.004 | 0.016 |

At identified contexts, single-chain fits fall into local optima in 15–30% of cases, up to ≈ 1.4 nats.

### 6.3 Where the memorization weight is defined

Revision 3 §5.3: k ≈ 2–4 in N100 and N400, k ≈ 3–6 in Nall. It is recomputed with the capped Bayes(b) before preregistering.

### 6.4 Evaluator bias (review 3 `mc_tail.py` and `mc_bias.py`)

With M = 1,024 samples and a 10⁻³ floor, the bias reaches +0.55 nats where the model's probability on the true value is ≈ 10⁻³. Hence:
- M = 8,192 sampled compositions at k ≥ 4, with a floor of 10⁻²;
- the same estimator for every library arm, including CS (its paths are sampled from its step weights);
- a mis-specified planted model in G1.

## 7. Exact references

- **Output level:** Bayes(a) and Bayes(b), exact predictives and prequential code lengths at the z positions.
- **Structure level.** Two compositions are equivalent when they compute the same function, so the exact posterior over composition classes is the posterior over functions given the context. Within a class, π* is uniform over minimal programs.
- **Branch values:** the distance to the version space (primary) and to the true function (on a sample), with classes in precedence order: dead > no completion within budget > away > lateral > optimal.

## 8. Endpoints

| # | endpoint | arms |
|---|---|---|
| E1 | Code length at z positions against Bayes(a) and Bayes(b), by test set, length and k | all |
| E2 | Structure fidelity: the router's distribution over functions against the exact posterior, measured where Bayes's effective number of functions is ≤ M/20; also mass on the true function and the valid-world rate | library arms |
| E3 | **Learnability:** CX − CP and CS-8 − CP on E1 and E2 | CX, CS, CP |
| E4 | Seeing vs searching: CX policy-only vs S = 16 / 64 / 256 at test | CX |
| E5 | Learned value vs exact distance to the version space: rank correlation and calibration on unseen-in-range and ℓ = 6 | CX |
| E6 | Routing-only adaptation (Neural Interpreters): after training in PA, adapt only the router on 64 / 256 episodes of pattern-test functions with the mask lifted; N gets full fine-tuning on the same episodes | library arms, N |
| E7 | Branch-class rates; steps used against ℓ (compute allocation) | committed arms |
| E8 | Memorization weight (§4) | all |

## 9. Preliminary predictions

Thresholds are fixed at preregistration. Verdicts use a run-level paired bootstrap.

| # | prediction | falsifier | type |
|---|---|---|---|
| H1 (headline) | CX comes within 0.05 nats per z position of CP at k ≥ 4, on pool and unseen-in-range, in N400 and Nall | CX − CP > 0.15 in both cells, or CX fails G2 | optimization (learnability) |
| H2 | At k = 1–3, CS-1's structure KL (E2) is above CS-8's, and CS-8's is near CX's. At identified contexts, CS-1 has a failure tail (excess > 0.05 in ≥ 10% of contexts) that CX and CP lack. | CS-8 ≥ CS-1, or CX's tail ≥ CS-1's | expressivity (H2a), optimization (H2b) |
| H3 | CP beats N on unseen-in-range and on length in N100 and N400. Labeled advantaged by construction; the magnitude is the result. | N ≥ CP at N400 | expressivity (prior by construction) |
| H4 | CX with search (S = 64) beats CX policy-only, by more on ℓ = 6 and pattern A than on pool. Learned-value rank correlation with exact distance ≥ 0.7 on unseen-in-range ℓ ≤ 5, and lower on ℓ = 6. | No gain from search, or correlation < 0.5 on unseen ℓ ≤ 5 | mechanism |
| H5 | With the router alone adapted, library arms reach a lower pattern-test code length than N fully fine-tuned, at 64 adaptation episodes | N ≤ the best library arm | mechanism |
| H6 | The memorization weight on Bayes(a) falls from N100 to Nall in every arm, by at least 0.1 | A fall of < 0.1 in any arm | mechanism |

**Planted models give both verdicts of every rule:**
- an oracle composition sampler (π*);
- the best per-context soft fit (§6.2), single and mixture;
- a pool memorizer;
- a uniform predictor;
- a random-composition predictor;
- a mis-specified sampler (G1).

## 10. Cells, grid and seeds

- **Cells:** N100, N400, Nall, PA (mask mul→inc), C_A (size-matched control). D3 is dropped: no prediction uses it (review 3).
- **Grid:**
  - 5 arms (N, CP, CX, CS-1, CS-8) × 5 cells × 10 seeds = 250 runs;
  - CX and CP raised to 29 seeds in N400 and Nall: +76;
  - CO pilot: 5 runs in N400.
  - 331 runs, plus the pilot.
- **Seeds:** a null pilot first; paired seeds; learning rate tuned per arm in the pilot (3 values × 2 seeds, N400).

## 11. Cost and the feasibility gate

- CX's cost is dominated by search: about S node evaluations per y position in training, ≈ 9× CP at S = 64.
- Evaluation with search is run on a subset only (1,000 episodes, k ∈ {1, 2, 3, 4, 8, 12}).
- The throughput job is extended to all arms. The grid must fit in ≈ 90 T4 GPU-hours. If it doesn't, cut in this order:
  1. S, to 32;
  2. CS-8, to N400 and Nall only;
  3. the N100 cell.

  Never fewer than 10 seeds; the headline's 29 are kept.

## 12. Before the main runs

**Planted checks (tests first):**
- execution tables;
- the soft-chain update equals the exact path mixture on small cases;
- masking is equivalent across arms;
- the Bayes code;
- the branch classifier against brute force for budgets ≤ 3;
- the search implementation reproduces §6.1.

**Pilot gates:**
- **G1, evaluator.** The estimator agrees within 0.005 nats across the minimal-program oracle and its padded twin, and detects the mis-specified sampler.
- **G2, bootstrap.** Within the pilot budget, CX's search with the learned prior (S = 64) finds context-consistent compositions for ≥ 50% of ℓ = 3 and ≥ 20% of ℓ = 4 targets in N400. Search alone manages 2 of 12 and 0 of 12 (§6.1). The constant-output share must fall below 10%.

  If G2 fails, that is itself a result (expert iteration does not bootstrap from the ℓ ≤ 2 foothold here). CX is then replaced by **CP→CX**: CP warm-started, then trained by expert iteration only. This is labeled, and H1 becomes conditional.
- **G3, soft fit.** CS-1 and CS-8 fit the pool (E1 within 0.05 of Bayes(a) at k ≥ 8 in Nall), or their comparisons are reported as conditional.
- **G4, nulls.** The spread between seeds of identical arms sets the minimum detectable effect.
- **G5.** The planted models give both verdicts.

**An independent pre-run review** of the code.

## 13. Stage B (next round, recorded so this stage stays scoped)

Worlds where every token depends on earlier tokens. For example, recurrent worlds zᵢ = f(xᵢ, yᵢ, zᵢ₋₁), or running programs over a sequence. The library adds context-reading primitives, e.g. "read register from the previous token" and "find the earlier token with the same input". Every token is routed. Serial depth grows with sequence length, which is where Entry 34's depth claim can fail. Learned (neural) primitives, and the reuse × selection 2×2, come after that.

## 14. Open for the preregistration

- Model sizes, S, M, λ_step, the episode budget (throughput test).
- The thresholds in H1–H6.
- Recomputing the memorization window with the capped Bayes(b).
- The atom-divergence check for mul→inc.
- Whether CO runs beyond the pilot.

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

---

## Independent review of revision 3 (26 Oct 2026)

The reviewer's scripts and outputs are in `review3/`.

**Verified correct:**
- the function counts;
- the §5.1–5.3 tables, reproduced exactly from the scripts;
- the hold-out arithmetic;
- the local-optimum finding of §5.1, robust to 8× more restarts;
- that sampling paths from SE is well defined;
- that the masking rule is equivalent in superposed and committed arms.

**Blocking findings:**
1. **CE vs SE measures how expressive the path distribution is, not commitment.**
   - Under log loss, CE's predictive is itself a mixture over sampled paths in output space.
   - A *static* mixture of 8 factorized chains, fitted like §5.1, closes most of the gap: median KL 0.001 / 0.004 / 0.016 at k = 1 / 2 / 3, against 0.010 / 0.021 / 0.042 for one chain.
   - The §5.1 gap is the mean-field vs structured family limitation, and TerpreT's FMGD is essentially SO.
   - Fix: an SE-mix arm with C chains matched to CE, crediting only CE − SE-mix to commitment. Otherwise relabel P3 as "factorized vs correlated path distribution". (`superpos_mix.py`, `spmix_C8.txt`)
2. **§5.1 scored a different quantity from the objective.** `superpos.py` averaged the KL over all 49 inputs, the context inputs included; the objective scores only the remaining ones.
   - Scored like the objective: 0.010 / 0.021 / 0.042 at k = 1 / 2 / 3 (0.008 / 0.016 / 0.035 with 24 restarts × 800 steps). The stated numbers were 0.022 / 0.039 / 0.053.
   - The derived best case is ≈ 0.01–0.04, so P3's 0.02 threshold sits at the bound.
   - Under an N400-like prior, the "3–300 functions" window is k ≈ 1–2, and at k = 3 it mixes with optimization failures.
   - Fix: rerun per cell with the objective's mask and define windows by effective-function count. (`superpos2.py`, `sp_*.txt`)
3. **No evidence the committed arms can learn.**
   - A per-context committed learner (tabular autoregressive policy, the design's per-point estimator, K = 16, the step cost, 1,200 iterations) found **0 of 30** targets of ℓ = 3–5, and 10–20% at ℓ = 2. It collapses to short partial matchers.
   - The fallbacks also failed:
     - noisy-channel ε = 0.1 collapses to a deterministic wrong path;
     - K = 64 found 0 of 30 at ℓ ≥ 3;
     - uniform init with 16 points also failed at ℓ ≥ 3.
   - §5.2 overstated the signal: of the "7.5% of episodes with an exact path", only 0.015% contain the true function; the rest are coincidences on 1–3 points.
   - §5.2 was measured with a uniform policy, not the designed short init (with the short init, the constant share is 0.17 and the weight on constants 0.21).
   - Amortization plus the 40% of training mass at ℓ ≤ 2 might act as a curriculum, but nothing shows it.
   - Fix: run G2 before preregistering anything, and preregister what happens if it fails. P3 and P4 are expressivity *and* optimization: SE gets exact gradients, CE gets 16-sample score-function gradients. (`commit_search.py`, `commit_search_out*.txt`, `initsig_exact.py`, `initsig_len3.py`)
4. **"One objective for every arm" is false.**
   - The step cost sits inside the log for committed arms and outside it for superposed arms, so the optima differ by O(λ), the same order as the thresholds.
   - No arm's optimum is the Bayes predictive once λ > 0, and SE's isn't even at λ = 0 (§5.1).
   - Fix: multiply each non-NOP transition by e^(−λ) inside SE's belief update, and take −log of the unnormalized r0 marginal.
5. **The Monte Carlo evaluator is biased against CE where its probability on the true value is small.**
   - With M = 1,024 and a 10⁻³ floor, the bias is +0.56 / +0.55 / +0.22 / +0.05 nats at q(true) = 3·10⁻⁴ / 10⁻³ / 3·10⁻³ / 10⁻²: the ℓ = 6 and pattern regime of P5.
   - For posterior-like models it is +0.001–0.015 nats (0.015 for Bayes(b) at k = 4).
   - G1's oracles never enter the low-q regime.
   - Fix: evaluate superposed arms with the same sampled-path estimator in every comparison with CE; use M = 8,192 or a 10⁻² floor at k ≥ 4; add a mis-specified planted model to G1. (`mc_tail.py`, `mc_bias.py`)

**Non-blocking findings:**
- **Predictions contradict each other.** The §5.1 tails imply that SE fails P1 if it confirms P4, and P4's "within 0.02 on average" contradicts its own tail clause.
- **λ_mem is confounded in Nall.** Bayes(b) puts 0.17–0.31 posterior mass on ℓ = 6, beyond every cap. Use Bayes(b) restricted to ℓ ≤ 5 and a 3-component fit (Bayes(a), Bayes(b), uniform). P7's first clause has no falsifier. (`memo_cap.py`)
- **P8 is unlikely to pass at λ ≤ 0.03.** The tilt per step (≈ 0.99) is weak against the entropy of equivalent paddings.
- **Masked logits get no gradient,** so unmasking at test moves an arbitrary amount of mass in PA arms only. Report PA pool performance with the mask both on and off at test.
- **Inconsistencies:** P2 is falsified only at N400; DiD is defined on accuracy while P5 is in nats.
- **Planted models** can't give the pass verdicts of P6 or P7, and the random-path predictor is not AGENTS' trained random-routing control.
- **The effective-world estimator** saturates near M at k ≤ 1.
- **The ℓ = 1–2 unseen strata** (1 and 8 functions) are still unaddressed.
- **Feasibility.**
  - Evaluating CE costs about as much as training (16 prefixes × 1,024 paths × 7 steps per episode).
  - 168 committed runs in ≈ 70 GPU-hours is not plausible.
  - Suggested cuts: D3 (no prediction uses it); CO and SO outside N400; evaluation only at k ∈ {1, 2, 3, 4, 8}.
- **Literature undercutting the novelty framing:**
  - TerpreT FMGD (= SO);
  - Neural Programmer, against hard-program learners trained from denotations (NSM/MAPO, Guu 2017);
  - DreamCoder's unigram vs bigram recognition models;
  - soft vs sparse mixture-of-experts;
  - ACT/PonderNet for compute allocation.

**Status:** findings 1 and 3 change what the round can claim and go to the researcher. Findings 2, 4 and 5 and the non-blocking items are fixable in revision 4.

---

## Independent review of revision 4: design and experimental value (26 Oct 2026)

The reviewer's scripts and outputs are in `review4/`.

**Verdict: not worth running as designed.**
- On pair worlds with a library equal to the generator's instruction set, the posterior over compositions, taken up to equivalence, *is* the posterior over 49-entry output tables (§7 says so). So the thesis's distinction between structure and output is zero by construction.
- Exact inference over all 297,845 functions is a 5 ms table lookup per context, so there is nothing hard for "find vs verify" or "seeing the minima" to amortize.
- What remains is amortized program induction over a known DSL:
  - CP is inference compilation, NeSymReS, or DreamCoder/MWS fantasy training;
  - CX is expert iteration for program synthesis (REPL synthesis, AlphaNPI, MWS replay, CodeIt).

  `generator_design.md` §11 had already called this a positive control.
- Most predictions are foregone or derivable now, and every outcome leads to the same next step.

**Blocking findings:**
1. **The thesis is not tested.** Label Stage A a program-induction positive control, cut it to a pilot, and test the thesis where the posterior over structure is not the posterior over outputs: Stage B, or a library that is not the generator's basis.
2. **CX's training rule doesn't have Bayes as its optimum.**
   - The terminal score is nearly length-neutral, at ≈ 1% per step, against the prior's ≈ 7× per step.
   - Visit-count targets seek the mode.
   - The next-token signal never reaches terminal nodes.

   Excess over Bayes (which itself scores 0.125 nats per z position, at k = 4–15): uniform over consistent functions 0.149; commit to the MAP function 0.199; uniform over the shortest consistent functions 0.030. The 0.149 sits at H1's falsifier. Fix: train on the marginal likelihood, with search only as a proposal (generator_design §10, objective 2), or judge H1 at k ≥ 8 only; and specify the open details.
3. **H2/E2 for the soft arms are decided by derivation, and CS-8 is not a fair control.** At the level of functions, a best-fit 8-chain mixture puts only 0.48–0.69 of its mass on context-consistent functions (TV 0.51–0.58), although its marginal KL is ≈ 0.001–0.012. §6.2 scored a marginal quantity (L20 again). CX − CS-8 confounds expressivity, estimator, objective and compute.
4. **Feasibility.** The Python search costs ≈ 38 µs per simulation, so ≈ 16 CPU-hours of tree bookkeeping per CX run at the planned budget. A GPU-batched search is needed and is not specified. "≈ 9× CP" ignores 64 sequential network calls per search.

**Non-blocking findings:**
- G2 should be scored on unseen targets, since pool functions can be memorized.
- §6.1's foothold reflects the 33-way needle, not averaging (max-backup doesn't help).
- E2's estimator is biased at k = 2.
- Missing from §2:
  - REPL synthesis (Ellis 2019);
  - AlphaNPI (Pierrot 2019);
  - memoised wake-sleep (Hewitt 2020; replay vs fantasy is CX vs CP);
  - CodeIt (Butt 2024);
  - NeSymReS (Biggio 2021) and Kamienny 2023;
  - AVICI (Lorch 2022);
  - the Latent Program Network (Bonnet & Macfarlane 2024);
  - inference compilation (Le 2017);
  - He et al. 2024.

**Suggested sequence:**
1. Entry 38's learned-value test on CPU (exact version-space distances → rank correlation and search nodes saved on unseen and ℓ = 6 functions).
2. CP vs N in N400 and Nall (10 seeds) to size the by-construction advantage.
3. A G2 pilot for CX with the corrected objective (3 seeds).
4. Then Stage B, or a mismatched library. Skip the 331-run grid.
