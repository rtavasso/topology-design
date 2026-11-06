# The rewrite carrier — a proof-shaped toy for the road to Lean (design, revision 1, 6 Nov 2026)

**Status: draft for discussion.** Supersedes [programs/DESIGN.md](../programs/DESIGN.md) revision 1 (the same questions, moved off the route carrier) after the researcher fixed the real-model target: **Lean** (history Entry 75). Open choices in §12; independent design review after them. Nothing here is implemented.

> **Correction, 6 Nov (same day, after the researcher's reading of §1).** "Proof length = depth" is wrong for the thesis. Proof length is the *external* loop, which the kernel runs and a scratchpad externalizes. The thesis's depth is **per decision**: at each next token the model branches over candidate next steps, rolls each forward in the latent stream with its own routing and halting (breadth B and lookahead depth D, both elastic), and selects — internal search that beats chain-of-thought at matched exploration because a latent rollout step costs one expert application rather than a forward pass plus decoding. Consequences for revision 2: the strata are **per-decision difficulty (B\*, D\*)** — the breadth of indistinguishable candidates and the lookahead needed to tell them apart — not proof length (which stays as a closure check); the library arm has the explicit **propose → per-branch latent rollout → value → select** structure; the CoT opponent is matched in exploration budget (best-of-B with D-step written lookahead) and compared at matched forward passes and matched tokens; the Lean-level endpoint is kernel calls and tokens per proof at matched success. **Held open (researcher, 6 Nov, 10:53):** whether a branch's rollout is latent only (A), executor-stepped (B: tree search with a learned prior and value), or split (C: the executor validates and applies candidates one step, the latent rollout supplies the lookahead). Rung 4 does not depend on it (it tests the prior; under B that is the whole claim about worlds, under A / C half of it, with the learned dynamics the other half — rung 4's result is reported under both readings). Rung 5 holds it as a measurement: all three library arms on the same (B\*, D\*) strata, with the **latent-against-exact lookahead curve** — the share of the exact-search gain the latent lookahead captures, per decision difficulty, at what cost in forward passes — as the endpoint that decides it.

## 0. Why a new carrier, and what it must look like

The route carrier (rungs 1–3) was built to isolate one thing — a four-action library with a setup step over an exact executor — and it did its job: programs are closures of per-state decisions (F30), the signal chooses among tied programs (F37), plasticity after commitment splits by exactness (F45). Its results transfer to Lean only by analogy. The real model will predict **tactics**: a proof is a sequence of tactic applications, each transforming an explicit **proof state**, each checked by an exact **kernel**; depth is proof length; the ambient theory and the file's local definitions are the **world**; premise selection is the **lookup** part, structurally separated from tactic choice. A toy that has that shape lets rungs 4 and 5 and the scaled run share one structure, so that what the toy finds about worlds, depth, labels and search is about the thing we will build.

The rewrite carrier is **equational rewriting**: a state is a term, a primitive is a rewrite rule applied at a position, a proof is a route of rewrites from a term to a goal term, the executor is exact, the oracle is breadth-first search, the world is the set of axioms in force — stated in the context, as Lean's hypotheses and imports are. It keeps everything the ladder's method depends on: an exact verifier (`check_episode`), exact oracle routes, an exact posterior over routes computed by dynamic programming on the reachable graph (small terms), planted routers, and the tie structure (several shortest proofs) that made rungs 2–3 informative.

## 1. The carrier

**Signature.** Atoms a, b, c, d and the constant 0; one binary operation ⊕ (optionally a unary inverse ⁻ in a later revision). Terms are binary trees over the atoms, written in prefix order as tokens (`⊕ ⊕ a b 0`); a position is a node index in prefix order. Term size (leaves) ≤ 5 in training, ≤ 7 in evaluation.

**Rules (the primitives; each a tactic with a direction).** A→ (x⊕y)⊕z → x⊕(y⊕z) and A← ; C x⊕y → y⊕x; I→ x⊕0 → x, 0⊕x → x and I← (x → x⊕0, bounded by the size cap); D→ x⊕x → x and D← (bounded). Seven directed rules; an action is (rule, position) or HALT; with ≤ 7 leaves there are ≤ 13 positions, so ≤ 92 actions. A rule that does not match at the position is a no-op that costs a step (untyped semantics, as the route carrier's wrong PREV: errors are detected only by the goal).

**Worlds.** A world is the set of **axioms in force**, W ⊆ {A, C, I, D}, W ≠ ∅: a rule may be applied only if its axiom is in W (applying one outside W is a no-op that costs a step and is marked invalid by the verifier). W is **written in the context** as axiom tokens, the way a Lean file states its hypotheses — the world is read, not guessed; inferring it from example proofs is a separate arm (§4). Fifteen worlds; training uses a subset, the rest are held out (§3).

**Instances.** Sample a term t of size ≤ 5; apply k valid random rewrites under W to obtain s, re-sampling until the BFS distance from t to s under W is exactly k (the *depth*); k ~ Zipf(1.5) on 1 … 6 in training, 1 … 10 in evaluation. The task: given (W, s, t), rewrite t into s. An episode is one instance (a proof state and a goal); batches mix worlds.

**Executor and oracle.** Exact rewriting on term trees; the verifier checks each step's validity and the final equality; BFS from t under W gives the distance, the set of shortest proofs (the ties) and the reachable graph within T steps. **Exact posteriors** over routes — under the router's own policy (EX) or under a flat prior over valid actions (PF) — are forward–backward passes over the reachable graph (≤ 10⁴ nodes at size 5; a per-episode sparse computation on CPU, used for the derive step, the PF arm and the checks).

**The router.** A small transformer encoder (d 128, 3 layers, 4 heads) over the token sequence `[axioms of W] [goal s] [SEP] [current term t]`, with a head over (rule, position) + HALT, applied once per step; the executor applies the chosen rewrite and the router is re-run on the new term. It is Markov in (t, context). This is the "PFN over tactics": attention over the stated axioms and the goal chooses the next primitive. Teacher forcing on the oracle proof, beam search with the verifier, the exact posteriors — the signals of rungs 2–3 — all transfer.

**Readout and loss.** Success = the committed route reaches s and halts, within T = 2k_max + 2. Per-step cross-entropy on labels (RT); the route-level signals as in rung 2.

## 2. The questions on this carrier (rung 4, symbolic executor)

- **Q1 (core) — recombination across worlds.** Trained on some axiom sets, tested on axiom sets never seen: does the router plan with a combination of rules it never saw together? Non-derivable: a per-rule decision conditioned on "this axiom is listed" composes only if the planning is compositional.
- **Q2 (core) — term-size extrapolation.** Terms of size 6–7 at test: positions and subterm shapes never seen. Rules against tables: a rewrite decision learned as a pattern over subterms extrapolates; one learned as a table over whole terms does not.
- **Q3 (check) — depth extrapolation** within training worlds: proofs of length 7–10 from training ≤ 6 (closure, F30).
- **Q4 (reference) — labels against search.** Routes found by beam search with the verifier (EI16) against routes labelled by the oracle (RT): does the recombination result depend on where the routes came from? In Lean the labels are human proofs and the search is expert iteration; this is the toy's version of that choice.
- **Q5 (ties).** Worlds with C and A have several shortest proofs. RT on a canonical choice learns one strategy; RT on the posterior over shortest proofs (the tie-aware label) and PF spread. Reported per world: which strategy each signal commits to, and whether the committed strategy transfers to held-out worlds.
- **Q6 (PFN inference, one arm).** Axioms *not* stated; two example proofs from the same world in the context instead. Does attention over examples identify the world? The in-context-identification question, posed the way a Lean file poses it (earlier proofs show which lemmas exist).

## 3. Held-out structure

Fifteen worlds. **Training: eight** — the four singletons {A}, {C}, {I}, {D}, and four pairs {A,C}, {A,I}, {C,D}, {I,D}. **Held out: the two pairs {A,D}, {C,I}, the four triples, and the full set** — every held-out world is a recombination of axioms seen individually and in pairs. Term size: train ≤ 5, test 6–7. Depth: train ≤ 6, test 7–10. Evaluation strata: world (train / held-out pair / triple / full) × depth × size.

## 4. Arms

| arm | signal | role | seeds |
|---|---|---|---|
| **RT-canon** | teacher forcing on one canonical shortest proof per instance | core (the labelled corpus) | 10 |
| **RT-post** | teacher forcing on the posterior over shortest proofs (soft targets; the tie-aware label) | core | 10 |
| **EI16** | beam (W 16, ε 0.1) over valid actions with the verifier, hard EM | reference Q4 (the Lean search regime) | 3 |
| **PF** | exact flat-prior posterior over routes reaching s (DP on the reachable graph) | reference Q5 | 3 |
| **RT-examples** | RT with axioms unstated and two example proofs in context | Q6 | 3 |
| **RT-shuffled-axioms** | axiom tokens permuted across episodes | null: the axioms are read | 2 |
| planted router | hand-written policy: match rule patterns at positions, prefer the rule that reduces BFS distance (from the oracle's table) | expressivity reference (L17) | – |

## 5. Derived before measuring (`derive/`)

- Reachable-graph sizes and BFS distances per world and term size; the tie multiplicity per instance (how many shortest proofs); the share of instances whose shortest proof's first step is unique.
- The closure table for Q3 (which (subterm pattern, rule) decisions the held-out depths need; all seen) and the recombination table for Q1 (which decisions the held-out worlds need and in which training worlds each was taken).
- A planted router that is exact on every world, size and depth (expressivity), and a planted **table router** (lookup over whole training terms) that fails size 6–7 — both outcomes of Q2, planted.
- Wandering mass at init under a uniform policy per depth (the gradient the answer-only signals start with).
- Greedy-vs-shortest: the fraction of instances where a greedy "reduce distance now" policy is optimal; the complement is where lookahead is needed — the quantity rung 5 is built on.

## 6. Predictions (fixed here)

| # | prediction | falsifier |
|---|---|---|
| P1 (check) | Within training worlds, RT-canon reaches ≥ 0.95 success at depths 7–10 in ≥ 9/10 seeds. | ≤ 7/10 |
| **P2 (core, Q1)** | RT generalizes to the held-out **pairs** in ≥ 6/10 seeds (≥ 0.9 success at in-range depth and size) and to the **triples** in fewer; the full set is the hardest. | ≤ 3/10 on the pairs: no composition across axiom sets; the prior must cover combinations. |
| **P3 (core, Q2)** | Size 6–7 extrapolation in ≤ 3/10 seeds (tables), with failures at positions never seen. | ≥ 7/10: pattern rules; the stronger PFN claim. |
| P4 (reference) | EI16 generalizes to held-out worlds in no fewer seeds than RT-canon (3/3 against the RT rate); search-found routes compose as well as labelled ones. | EI16 0/3 where RT ≥ 6/10: labels matter for composition. |
| P5 (ties) | RT-canon commits to the canonical strategy in every world; RT-post and PF keep a mixture; the committed strategies differ by world. | — (reported) |
| P6 (Q6) | RT-examples identifies the world from two example proofs on training worlds (≥ 0.9) and generalizes to held-out worlds in fewer seeds than the stated-axiom arm. | RT-examples ≥ stated-axiom RT on held-out worlds: attention reads the generator's signature directly. |
| P7 (null) | RT-shuffled-axioms is at the single-world baseline (it cannot know which rules are valid). | — |

## 7. Rung 5 on the same data: the neural basis and the internal depth (outline; its own design after rung 4)

What the symbolic rung cannot test is the thing a scaled run would die of: learning the **primitives and the router together** (risk 2; F22–F23, F32). Rung 5 keeps the data and the verifier and changes two things.

**The depth variable moves inside the step.** In Lean the kernel supplies the external loop; the model's internal depth is the lookahead it needs to choose the *next* tactic. The derive step above measures where greedy fails; rung 5 constructs instances whose correct first rewrite is determined only by lookahead of depth k_la (garden paths: the step that reduces distance now is wrong). A model that predicts the next tactic from the state must simulate k_la rewrites internally — or write them out.

**Arms.** Dense transformers of depth L predicting the next tactic from (W, s, t) with no scratchpad; dense **with a scratchpad** (allowed to emit lookahead tokens — the chain-of-thought opponent, which is what Lean puts in front of us); the **library** — m small experts (attention and MLP), a router with an explicit state, internal iteration to a cap, halting — trained on labels, by verifier-guided search, and by **end-to-end gating with load balancing** (the SUT / MoEUT regime, the only arm that reproduces how the existing architectures are trained). Planted experts and router as the ceiling.

**Endpoints.** Next-tactic accuracy and proof success stratified by k_la (never the mean); compute per solved problem in the lookahead tail at matched accuracy (where the library should beat the scratchpad by about a layer count); parameters at matched tail accuracy against the dense family; a functional probe of what each learned expert computes (is it a rewrite?); **distillation calibration** — distill the dense-L arms and the planted library into libraries and recover m(L), which validates the instrument before it is used on a real teacher (Entry 75).

**Predictions, to be fixed in its own design.** Dense-L cliffs at a lookahead depth set by L; the scratchpad arm extrapolates; the library extrapolates under labels and search; the gating-trained library matches in range and does not extrapolate, with blended experts under the probe — the chains-against-bags explanation owed for MoEUT.

## 8. The road after the toys

Synthetic symbolic math with derivation traces for the first real model (controllable depth, exact grading, labels free); **Lean tactic traces** (Mathlib via LeanDojo) as the real-data step, where RT is supervised traces, EI16 is verifier-guided proof search, premise selection is the lookup part, and the ambient theory is the world; natural-language math last, as a benchmark. The distillation instrument, calibrated in rung 5, measures m(L) on a math-trained dense teacher first.

## 9. Budget

Rung 4: 31 runs; the router is small and the executor cheap, but the executor runs in the training loop (batched term rewriting on CPU or a vectorized tree representation on GPU; the pilot measures it); ≈ 10–20k steps per run; one Kaggle session if the executor is vectorized, two if not. Building the carrier (terms, rules, BFS, verifier, DP posteriors, generator, router, beam, analysis) is the main cost: on the order of a week of agent work with tests first.

## 10. Gates

G1 tests: rewriting on hand-built terms; BFS distances against brute force; the verifier rejecting invalid and out-of-world steps; DP posteriors summing to one and matching enumeration on tiny instances; the planted router exact on every world; the table router failing size 6–7 as predicted; generator strata. G2 derive (§5). G3 pilot (lr, steps, executor cost; one seed; reads nothing). G4 independent design review of this document after §12, then a code review before the grid.

## 11. What this rung does not answer

Learning the primitives (rung 5); chains against bags in a trained model (rung 5); where labels come from at scale (Lean's are given, the search is expert iteration; the open question there is the prior over proofs, which this carrier's PF arm can only gesture at); worlds as hidden latents with piecewise constancy (later).

## 12. Open choices for the researcher

1. **The axiom set and directions.** {A, C, I, D} with both directions for A and I and D, C self-inverse — or add the unary inverse (⁻) and cancellation now, which makes the worlds group-like and the proofs longer, at the cost of a larger reachable graph.
2. **Worlds stated or inferred as the core.** This draft makes stated axioms the core (Lean-like) and inference from example proofs one arm; the alternative makes inference the core, which is the PFN question proper but stacks two difficulties.
3. **The labelled core.** RT on a canonical shortest proof, or on the posterior over shortest proofs (the tie-aware label); this draft runs both at 10 seeds.
4. **Order.** Rung 4 (symbolic) first, then rung 5 (neural) on the same data — the recommendation, since it is cheap once the carrier exists and settles routing generalization before primitives are added — or straight to rung 5 with rung 4's cells as a subset of its arms.
5. **Size caps.** Train ≤ 5 leaves, test ≤ 7 (reachable graphs stay small; DP posteriors exact); or larger, with the DP arms dropped.
