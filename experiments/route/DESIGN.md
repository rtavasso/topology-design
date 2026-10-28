# Rung 1: a learned per-token router over exact generic circuits (revision 1)

**Status:** design, 28 Oct 2026, for independent review before implementation finishes. Rung 1 of the ladder (D72): symbolic generic circuits, routing only (Q1, Q2); the formalism of D75 (latent-route mixture, step-cost prior, soft routing as the first E-step). Carries rung 0's constraints (Entry 60).

**Ledger:** thesis Entry 48; what the toy is for Entry 45; generic circuits Entry 44; residual signals Entry 46; setup credit Entry 47; pooling Entry 50; ragged depth Entry 52; module-network prior work Entry 33; rung 0's results Entries 55–59 (F22–F27).

**Scope note (AGENTS hard rule 1).** This round compares a learned router over fixed exact circuits with static and looped learned models, at the researcher's direction. Every claim is labeled expressivity, optimization or mechanism.

---

## 1. The question

**Q1 (B2).** From next-token loss alone, does a per-token router learn *order-dependent serial chains of distinct generic circuits*, where the chain needed varies by token and context?

**Q2 (B3).** Do the learned chains generalize to longer chains and to orders never seen, better than a static stack of learned layers and a looped learned block at matched per-token compute?

**The bet that makes this different from module networks.** Learned layouts inferred *from the input* fail systematic generalization (SQOOP, CLOSURE; Entry 33). The router here infers *each step from the computational state*: "my register holds a key whose pair says ←, so the next circuit is PREV". That rule is local, depth-free and content-free, so a router that learns it generalizes to any word over the circuits and any length. The static alternative must host every circuit at every depth and learn each instance from the tokens that need it there (pooling for selection, Entry 50). The test is whether the learned router actually finds the local rule rather than a memorized table of word shapes.

**Also asked.** B1 (the step-cost prior: does it buy minimal routes?), B4 (iterate until done, inherited from rung 0), the residual study of D73 (which signals track the exact remaining distance), and the setup-credit question of D74 in its soft form (PREV steps change nothing in the prediction; does the exact chain rule credit them?).

**Not asked here.** Learned primitives (rung 2–3), hard routing with search and a learned value (rung 4; B5, B6), transfer to new symbol alphabets (empty by construction with frozen bases, as in rung 0's H8).

## 2. The task

**Symbols.** 48 keys, 16 answers, one marker ← ("go back"), one dead symbol ⊥. Frozen orthonormal embeddings in disjoint subspaces of the stream, as in rung 0: key slot, value slot, **prev slot**, register, scratch (d = 5·64 + 8 = 328; the neural baselines use the same bases).

**Pair tokens.** Each pair token carries (key, value, prev): the key, its value (a key, an answer, or ←), and the key of the pair that *precedes it in context order*. The prev slot is the previous-token head's output, provided in the token so that every circuit in the library is a one-layer match-and-read (§3); the baselines get the same tokens.

**Chains.** A chain is a sequence of states s₀ → s₁ → … → s_k with s_k an answer. Each step is one of two circuits, written in the data:
- **H (hop):** pair(s_i).value = s_{i+1} (a key, or the answer on the last step);
- **P (prev):** pair(s_i).value = ← and pair(s_{i+1}) is the pair immediately before pair(s_i) in context order.

The last step is always H (an answer is reached only as a value). The **word** of a chain is its step sequence, e.g. HHPHPH. The word is written in the data (← marks exactly the P-steps) and is readable only by following the chain: nothing in the query token says what the word is (F7).

**An episode.** A forest over the 48 keys: Qn = 6 queried chains plus distractor chains, P = 48 pair tokens, shuffled subject to the adjacency that P-steps force (a run of j P-steps is a block of j + 1 pairs in reverse order). Queries: (start key, answer) token pairs, 61 tokens in all, as in rung 0. The loss is the cross-entropy over the 16 answers at the query positions.

**Lengths and words (training).** k ~ Zipf(1.5) truncated at k_max = 8. Each non-final step is P with probability p_P = 0.25, independently, **except that P-runs are limited to length 1 in training** (no two consecutive P's). So the training family W_train = words over {H, P} of length ≤ 8 ending in H with no PP. The fraction of chains with at least one P, the count of P-steps at each depth, and the number of distinct training words are computed in `derive/` (§5).

**Evaluation strata** (one forced query per episode plus Zipf fillers, 250 episodes per stratum, exact references for every query):
- **In range:** k ∈ {1, …, 8}, words from W_train.
- **Length:** k ∈ {10, 12, 16, 20, 24, 32}, words with the training statistics (p_P = 0.25, no PP).
- **New orders, within length:** k ≤ 8 with (i) PP and PPP runs (never seen), (ii) P-dense alternation (HPHPHP…, p_P = 0.5), (iii) P at depth ≥ 6 (rare in training; counts in §5).
- **New orders, long:** k ∈ {16, 32} with PP runs and alternation.

The no-hop baseline (uniform over the terminals present) is reported per stratum.

**Read-only context, ragged depth.** As in rung 0 (D79): context tokens are single-step; a query's route attends to the context's depth-1 states. The router has its own attention over the context (Entry 52's deferred variant, needed here because the circuits are symbolic and there is no block to read through).

## 3. The library and the exact executor

Three actions. On a register holding symbol s with pair(s) = (s, v, q) when s is a key:
- **HOP:** register ← v. If s is not a key (an answer, ←, ⊥): register ← s if s is an answer (answers are absorbing), else ⊥.
- **PREV:** register ← q (the previous pair's key); ⊥ if s is not a key or pair(s) is the first pair. Answers absorbing.
- **HALT:** stop; read out.

Both circuits are one-layer generic match-and-read circuits: match the register against the key slot, read the value slot (HOP) or the prev slot (PREV). The library is the two of them and HALT; the router chooses. A wrong circuit does not raise an error: HOP on a ← pair gives ⊥ (dead, absorbing, uniform readout); PREV on an H-state gives a real key and the route *wanders* (it may reach some answer, usually the wrong one). So wrong routes are not detectable step by step; they are detected only by the likelihood of the target. This is the untyped semantics on purpose: it is what makes the E-step a real inference rather than a lookup.

**Readout.** Register holding answer a: p(y = a) = 1 − ε, the rest ε/15, with ε = 0.02 fixed; register holding anything else: uniform over the 16 answers.

**The executor is symbolic** (`exec.py`): integer tensors, gathers over the episode's pair table. No neural circuits run in the routed arms.

## 4. The model: a Markov router and the exact marginal

**Router.** π_θ(a | s, c), a ∈ {HOP, PREV, HALT}, from the register symbol s and the context c. Architecture: the state token = frozen register embedding of s + a learned marker; one cross-attention layer (2 heads) from the state token over the context's pair tokens (frozen slot embeddings + learned positions, through one learned pre-norm self-attention layer so that a pair can also see its neighbours), then an MLP and a linear head to 3 logits. The router sees **no step index** and **no route history**: it is Markov in the symbolic state, which is the form that can pool across depth (Entry 50). Its logits are standardized by a LayerNorm on the MLP input (L24), and the planted router (§6) is within one unit of the init.

**Exact marginal likelihood.** Because the state is one of 66 symbols and π is Markov, the mixture over all routes is a forward recursion over symbols: with alive mass b_t(s) (b₀ = δ_{s₀}),
- halt mass h_t(s) = b_t(s) · π(HALT | s, c),
- b_{t+1}(s') = Σ_s b_t(s) · Σ_{a ∈ {HOP, PREV}} π(a | s, c) · [exec_a(s) = s'],
- forced halt at T: h_T(s) += b_T(s),
- q(y | c) = Σ_{t ≤ T} Σ_s h_t(s) · readout_s(y).

This is the latent-route mixture of Entry 48, q(y | c) = Σ_r π(r | c) p_r(y | c), summed exactly over all 3^T routes in O(T · |S| · 3) with |S| ≤ 66, because routes that share a state at a step are merged (π is Markov). The router is evaluated only at states with mass above 10⁻⁶; in practice a handful per step. Gradients flow to π at every live state in proportion to the posterior mass of the routes through it, so setup steps (PREV, which changes no prediction) get the chain-rule credit of Entry 47 exactly, and no route has to be *found*: short chains dominate the gradient (Zipf), the local rule is learned on them, and it applies at every depth.

**Training loss.** −log q(y | c) + λ · E[steps], with E[steps] = Σ_t t · Σ_s h_t(s) (the expected number of circuit applications), λ = 0.02 per step (so that one extra step costs about the log-likelihood gap of a 2 % readout error; swept in the pilot). T_train = 12 = k_max + 4; T_eval = 40. AdamW, lr 1e-3, 500 warm-up, cosine, 6k steps, batch 64 (pilot: lr ∈ {3e-4, 1e-3, 3e-3}).

**Test-time routing.** Two readings of the same model: **committed** (at each state take argmax_a π; the route the system would actually run, its steps are the compute) and **marginal** (q itself, the Bayes-mixture prediction, no compute saving). Route accuracy = the committed route equals the exact oracle route up to the halting step; answer accuracy = argmax of the committed route's readout; steps = circuit applications before HALT.

## 5. Derived before measuring (`derive/`)

- **The distinguishing quantity (L22)** (`derive/coverage.py`, 24,000 training queries). A state-blind policy (a fixed program, or per-step mixing that depends only on the step) succeeds on a query only if the query's word is a prefix of its program. The best fixed program of any length is all-H: it covers **0.768** of training queries (the 76.8 % whose word has no P) and **0** of every word with a P; the most common word covers 1.00 / 0.74 / 0.55 / 0.43 / 0.32 / 0.24 / 0.15 / 0.13 of the chains of length 1–8. There are 87 distinct training words. A Markov state-reading policy reaches 1.0 everywhere. So the gap is 0.23 of the in-range stream (all of it on words with a P) and all of the new-order strata: every in-range table reports accuracy on words with a P separately, which is where the arms differ.
- **Counts per (circuit, depth)** (`derive/counts.py`, per 100 training steps of 38,400 queries): P-steps at depth 1–8 are 4,690 / 2,124 / 1,576 / 1,014 / 677 / 421 / 163 / 0 (the last step is always H), H-steps 33,710 / 16,415 / 10,028 / 6,700 / 4,483 / 2,932 / 1,799 / 845; 23.1 % of queries contain a P and 12.2 % of steps are P.
- **The wandering mass** (`derive/wander.py`). Under a uniform router the oracle route's share of the mass halting on the target is 0.30 / 0.22 / 0.12 / 0.05 / 0.01 / 0.00 at k = 1 / 2 / 3 / 4 / 6 / 8 (the rest is routes halting on a non-answer, whose uniform readout gives 1/16); the gradient for long chains is small at init and the local rule is learned on short chains first (Zipf), then applies at every depth.
- **Static reach, and the static arm's budget.** With read-only context a static depth-L learned model reaches k ≤ L (rung 0's argument), and every layer must host HOP, PREV and the ← test. Rung 0 found that S8 learns hops one layer at a time (k = 3 at ≈ 6.6k steps, k = 4 not by 25k; F26), so within the 25k-step budget its reach is k ≤ 3–4 whatever the word. Predictions about the static arm are therefore made within that reach (P4), and its learning time per (circuit, depth) is the endpoint, not its accuracy at depth.
- **Compute.** The oracle route costs k + 1 router calls per query (k circuit applications and a HALT decision); a router call is one cross-attention over 48 tokens, comparable to one block application in rung 0.
- **The wandering mass.** Under a uniform router the posterior mass on the oracle route of length k is ≈ 3⁻ᵏ relative to the mass of all routes; the gradient still identifies the correct action at every state because wandering routes reach the target answer only by coincidence (computed: the fraction of length-k wandering routes that end on the right answer, per episode, `derive/wander.py`).

## 6. Exact references and planted models (L17)

- **Oracle route** for every query: the word written in the data; remaining distance at every state: exact.
- **Planted router (tested):** cross-attention with the register matched against key slots (score β on the unique match, a BOS sink otherwise), reading the value slot into the MLP input; the MLP classifies the value slot into {answer, key, ←} by three fixed directions and maps to {HALT, HOP, PREV}. Under the exact executor it reproduces the oracle route on every query at every k ≤ 32 and every word, committed and marginal (q(y) = 1 − ε on the target). Weights are one unit from the init (L24).
- **Planted failures:** a step-index router with the best fixed program (fails new orders and long chains); a router that never halts (forced halt at T; fails short chains under λ > 0 by cost and at k > T by reach); a myopic router that halts whenever the readout does not change (halts at the first P-step: fails every word with a P).
- **The looped baseline's expressivity (argued, not planted):** HOP and PREV fit in one block as two heads that match the register against the key slot and read the value slot and the prev slot respectively, with a marker-gated MLP that moves the ←-selected read into the register (rung 0's planted block does this for one head; the ← gate is a second gated linear map). So the looped learned block can express one oracle step per loop at d = 344 with 2 heads; the pilot's in-range LF accuracy (G4) is the empirical check. Planting it is deferred (it is a fairness check, not a measure).

## 7. Arms

| arm | what is learned | role | seeds |
|---|---|---|---|
| **RM** | π_θ(a ∣ s, c), exact marginal, λ = 0.02 | **the thesis arm** | 10 |
| RM-λ0 | as RM, λ = 0 | B1: the step cost | 5 |
| RM-idx | π_θ(a ∣ t) only (a learned fixed program) | the state-blind null; exact-circuit static | 5 |
| RM-T | π_θ(a ∣ s, c) trained by teacher forcing on the oracle route | recognition upper bound (NPI-style traces) | 5 |
| **S8** | 8 unique learned layers (rung 0's model, d = 328, prev slot in the token), lr 1e-4, 25k steps | static learned stack; reach k ≤ 8 | 5 |
| **LF / LR** | one looped learned block, T = 10, the stability exit at test (rung 0's best) | the m = 1 learned alternative with internal gating | 10 |
| S2 | 2 learned layers | compute-matched static sanity | 3 |
| oracle, fixed programs | — | exact references (§5, §6) | — |

Not run, with reasons: a soft superposed stack with context-conditioned mixing (its best case is the fixed-program coverage of §5 plus whatever a static encoder can infer of the word, i.e. a static model that outputs the program; it is dominated by S8 on the same depth argument); per-layer head selection (MoH: a static stack with routing *within* a layer; its failure on new depths is S8's); transfer to new alphabets (empty by construction).

## 8. Cells and grid

| cell | arms × seeds | steps | purpose |
|---|---|---|---|
| pilot | RM × lr {3e-4, 1e-3, 3e-3} × 2 seeds; RM λ ∈ {0.005, 0.02, 0.1} × 2; LF lr {3e-4, 1e-3} × 2; S8 lr {3e-5, 1e-4} × 2 | 4k (routed), 6k (LF), 15k (S8) | rates, λ, the gate (§11) |
| main | RM 10, RM-λ0 5, RM-idx 5, RM-T 5, S8 5, LF 10, S2 3 | routed 6k, LF 6k, S8 25k, S2 8k | P1–P8 |

Cost: a routed run is far cheaper than a looped neural run (one cross-attention per live state per step; ≤ ~10 live states); measured in the pilot. The neural arms cost as in rung 0 (S8 ≈ 0.055 s per step at d = 328, LF ≈ 0.07): main ≈ S8 5 × 25k × 0.055 + LF 10 × 6k × 0.07 + S2 + routed ≈ 2.0 + 1.2 + 0.1 + ~1 = **≈ 4.5 T4-hours**; one session with the pilot.

Batching: the routed arms stack over runs like rung 0's models (a leading run dimension on the router; the executor is shared), the neural arms with `stacked.py`.

## 9. Endpoints

By stratum (in range; length; new orders within length; new orders long), run-level:
1. Route accuracy (committed route = oracle route), answer accuracy (committed), marginal accuracy, steps; for the neural arms answer accuracy and steps (LR's steps).
2. **Pooling for selection:** S8's accuracy on words with a P at depth d against the training count of P-steps at depth d (§5), and RM's and LF's on the same words.
3. **Residual study (D73):** along the committed route, the router's entropy, p(HALT) and the state-change indicator against the exact remaining distance; the KL from the committed prediction to the exact posterior predictive at each step; a linear probe for remaining distance from the router's MLP input at live states (privileged labels; diagnostic).
4. **Step cost:** steps and route accuracy with and without λ; the halting excess over k.
5. Time to criterion by (circuit, depth) for S8 and by k for every arm, as in rung 0.
6. The fixed-program coverage and the oracle, as references on every table.

## 10. Predictions (thresholds fixed here; run-level paired bootstrap, per-seed counts)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **Learnability (B2).** RM reaches route accuracy ≥ 0.95 and answer accuracy ≥ 0.98 in range (k ≤ 8, W_train) in ≥ 9/10 seeds within 6k steps; so does RM-T. | RM < 0.8 route accuracy in range in ≥ 5/10 seeds while RM-T ≥ 0.95: the marginal objective does not find the rule that traces teach (an E-step problem, not an architecture problem). Both < 0.8: the router architecture cannot read the local rule. | optimization |
| **P2** | **Length (B3).** RM committed answer accuracy ≥ 0.9 at k = 16 and ≥ 0.8 at k = 32 in ≥ 8/10 seeds, steps ≤ k + 1.5. | ≥ 5/10 seeds < 0.6 at k = 16: the learned rule is not the local one (then the entropy and probe diagnostics say what it reads). | mechanism (what the router conditions on), conditional on P1 |
| **P3** | **New orders (B3, the headline).** On PP/PPP runs and on alternation within k ≤ 8, RM ≥ 0.9 in ≥ 8/10 seeds; on the long new-order strata (k = 16, 32) ≥ 0.8. | RM < 0.6 on PP runs within range in ≥ 5/10 seeds while in-range accuracy ≥ 0.95: the router memorized word shapes (the module-network outcome). | mechanism |
| **P4** | **The static arm.** Within its reach at 25k steps (k ≤ 3, F26), S8 handles P-words as well as all-H words of the same length (≥ 0.9 on PP runs at k = 3: the ← gate is learned per layer from ≥ 1.5k examples per 100 steps at depths 1–3), and its time to criterion for k = 1 / 2 / 3 replicates rung 0 (each ≥ 2× the previous); its accuracy at k ≥ 5 is at chance for every word. The pooling-for-selection contrast (accuracy on P-at-depth-d words against the count of P at depth d) is reported for RM and LF/LR as a diagnostic (prediction: flat at ≥ 0.9 for every d ≤ 7), and for S8 only as time to criterion by (circuit, depth). | S8 < 0.6 on PP runs at k = 3 while ≥ 0.95 on all-H words at k = 3: selection within a layer is what fails, not depth. RM or LF/LR below 0.8 at some d ≤ 7 while ≥ 0.9 at others: the selection rule did not pool across depth. | optimization (depth) and mechanism (pooling), separately labeled |
| **P5** | **The m = 1 learned alternative.** LF/LR extrapolates in length (≥ 0.9 at k = 16, as in rung 0) **and** on PP runs and alternation (≥ 0.8), since its block learns the gated local rule once. | LF/LR < 0.6 on PP runs: a single learned block does not learn two gated circuits well enough to compose them in unseen orders; then explicit selection has a measured advantage over internal gating. | mechanism; the library question's first data point |
| **P6** | **Step cost (B1).** RM-λ0 halts late (steps ≥ k + 3 in range) with in-range accuracy unchanged; its length extrapolation is no better than RM's. | RM-λ0 halts at k + 1 anyway (the cost is redundant here), or extrapolates better (minimal routes are not what generalizes). | optimization / mechanism |
| **P7** | **The null.** RM-idx reaches exactly the fixed-program coverage of §5 in range (within 0.05) and chance on new orders. | RM-idx above coverage by > 0.1: the strata leak the word. | expressivity |
| **P8** | **Residual signals (D73).** Along the committed route the router's p(HALT) is < 0.05 at every state with remaining distance ≥ 1 and > 0.5 at distance 0, in and out of range; the linear probe recovers remaining distance from the MLP input with R² ≥ 0.8 in range and ≥ 0.5 at k = 16–32 (the state carries progress, cf. Entry 47). | p(HALT) > 0.3 at distance ≥ 1 on P-states (myopic halting on "no change"), or R² < 0.3 out of range. | mechanism (diagnostic) |

**Reading the outcomes.** P1 ∧ P2 ∧ P3: the recognition model learns a local, depth-free, order-free rule from next-token likelihood; Q1 and Q2 answered yes with exact circuits; rung 2 (neural primitives) is next. P1 ∧ ¬P3: the module-network outcome at the step level; the next test is what the router reads (entropy/probe) and whether traces (RM-T) generalize where the marginal did not. ¬P1 with RM-T passing: the E-step is the problem even when it is exact; the objective's basin matters (as in rung 0's halting). P5 passing alongside P3: a single gated block matches explicit selection on this task; the library's advantage, if any, is not generalization at m = 2 and must be sought in pooling at larger m or in compute (rung 2–3). P4: the static stack's failure on new orders is a count effect, not a capacity effect.

## 11. Gates before the main grid

- **G1 planted:** the planted router reproduces the oracle route on every stratum (committed and marginal); the planted block for LF is exact; the three planted failures fail where stated.
- **G2 exact marginal:** the forward recursion equals brute-force enumeration over all routes for T ≤ 6 on random routers (test).
- **G3 derive:** coverage, counts and wandering mass computed and inserted in §5 before the pilot.
- **G4 pilot:** learning rates and λ; RM must pass P1 in the pilot (2 seeds) or the router/objective is revised before the main grid; cost measured.
- **G5 reviews:** an independent review of this design and of the code before the main grid.

## 12. Fairness (AGENTS rule 4)

- Learning rates per arm from the pilot; S8 bracketed (3e-5, 1e-4); the same frozen symbol bases and tokens for every arm; the same seeds (paired episode streams).
- Compute: steps per query (router calls or loops); S2 is compute-matched; the exact executor's cost is not counted against the routed arms because the neural arms' blocks are also one layer per step — stated as a limitation: the routed arms are given the circuits.
- 10 seeds for the headline arms (RM, LF), 5 elsewhere, 3 for S2; per-seed counts in every prediction; paired bootstrap.
- Time-to-criterion comparisons on the same stream per seed; schedules differ by arm (stated).

## 13. What this rung does not answer

- Whether neural primitives can be made to act on one shared state without drift (rung 2, Q3), or learned jointly with the router without the damage of F22–F23 (rung 3, Q4).
- Hard routing, search and a learned value (rung 4): here the E-step is exact because the state is symbolic; nothing is said about finding routes when it is not.
- Whether the local rule survives when the trigger is not written in the token (← in the value slot) but must be inferred from several tokens.
- m > 2 circuits and the basis-size frontier (Entry 51 level 1 applies; not run).

## 14. Prior work (to be checked by the reviewer)

Neural Module Networks and layout induction (Andreas 2016; Hu 2017 N2NMN; Johnson 2017 IEP; Bahdanau 2019 SQOOP; CLOSURE): layouts inferred from the input fail systematic generalization unless given. Neural Programmer-Interpreter (Reed & de Freitas 2016) and its recursive version (Cai 2017): a controller over exact programs, trained on traces (RM-T is this form). Neural Programmer (Neelakantan 2016), NRAM (Kurach 2016), Neural GPU, differentiable Forth: soft selection over exact or differentiable modules trained from outputs, with length generalization as the test; the closest prior art to RM. RASP-L and length generalization (Zhou 2023): tasks with short RASP programs generalize in length. Transformer circuits (Elhage 2021, Olsson 2022): HOP and PREV are the copy/induction and previous-token heads. What none of these do, to our reading: a Markov router over context circuits on a token sequence, trained by the exact route marginal from next-token loss, tested for extrapolation in *order* against a static learned stack and a looped learned block with exact references.
