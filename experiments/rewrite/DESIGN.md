# The rewrite carrier — a proof-shaped toy for the road to Lean (design, revision 2, 6 Nov 2026)

**Status: revision 2, for independent design review.** Revision 1 (same day) is in git; its §12 choices were fixed by the researcher at the recommended defaults (§12 below records them), and its reading of depth was corrected the same day (§0). Supersedes [programs/DESIGN.md](../programs/DESIGN.md). Nothing here is implemented.

## 0. What the carrier is for, and what "depth" means here

**The real-model target is Lean** (history Entry 75): a proof is a sequence of tactic applications, each transforming an explicit proof state, each checked by an exact kernel; the file's hypotheses and imports are the world; premise selection is the lookup part, structurally separate from tactic choice. The toy has that shape so that rungs 4 and 5 and the scaled run share one structure.

**Depth is per decision, not proof length.** Proof length is the *external* loop — the kernel runs it, and a scratchpad externalizes it. The thesis's depth is what happens inside one decision: at each next step the model branches over candidate next steps, rolls each forward in the latent stream with its own routing and halting (breadth and lookahead both elastic), and selects — internal search that is meant to beat chain-of-thought at matched exploration because a latent rollout step costs one expert application rather than a forward pass plus decoding. So the carrier's primary strata are **per-decision difficulty** (§1, (B\*, D\*)); proof length is a closure check.

**Held open (the researcher, 6 Nov; ledger I36):** whether a branch's rollout is latent only (A), executor-stepped (B: tree search with a learned prior and value), or split (C: the executor validates and applies candidates one step, the latent rollout supplies the lookahead). Rung 4 (this document's core) does not depend on it: it tests the *prior* — under B that is the whole claim about worlds, under A / C half of it, with the learned dynamics the other half — and its results are reported under both readings. Rung 5 (§7) measures it instead of assuming it.

**Why not the route carrier.** Rungs 1–3 did their job on it (F30, F37, F45); their results transfer to Lean only by analogy. The rewrite carrier keeps everything the method depends on — an exact verifier, exact oracle routes, exact posteriors over routes by dynamic programming on the reachable graph, planted routers, ties — in the Lean shape.

## 1. The carrier

**Signature.** Atoms a, b, c, d and the constant 0; one binary operation ⊕. Terms are binary trees over the atoms and 0, written in prefix order as tokens (`⊕ ⊕ a b 0`); a position is a node index in prefix order. Term size (leaves) **≤ 5 in training, ≤ 7 in evaluation** (§12.5).

**Rules — the primitives (§12.1).** A→ (x⊕y)⊕z → x⊕(y⊕z) and A←; C x⊕y → y⊕x (self-inverse); I→ x⊕0 → x and 0⊕x → x, and I← x → x⊕0 (bounded by the size cap); D→ x⊕x → x and D← x → x⊕x (bounded). Seven directed rules. An action is (rule, position) or HALT; with ≤ 7 leaves there are ≤ 13 positions, ≤ 92 actions. A rule that does not match at its position is a no-op that costs a step (untyped semantics, as the route carrier's: errors are detected only by the goal).

**Worlds (§12.2).** A world is the set of axioms in force, W ⊆ {A, C, I, D}, W ≠ ∅ — fifteen worlds. A rule may be applied only if its axiom is in W; applying one outside W is a no-op that costs a step and is marked invalid by the verifier. **W is written in the context as axiom tokens**, as a Lean file states its hypotheses: the world is read, not guessed. Inferring it from example proofs is one arm (§4, RT-examples).

**Instances.** Sample a term t of size ≤ 5; apply k valid random rewrites under W to obtain s; keep the instance if the BFS distance from t to s under W is exactly k; k ~ Zipf(1.5) on 1 … 6 in training, 1 … 10 in evaluation. The task: given (W, s, t), rewrite t into s and HALT. An episode is one instance; batches mix worlds uniformly.

**Per-decision difficulty (the primary strata).** At a state t with goal s under W, let d(·) be the BFS distance to s; the correct next actions are those with d(a(t)) = d(t) − 1. Let σ(u, s) be a **syntactic progress proxy** that needs no search — the tree-edit distance between u and s (the derive step fixes it, §5). Define, for the decision at t:
- **B\*** — the number of next actions whose one-step σ is minimal (how many candidates look equally good greedily);
- **D\*** — the smallest lookahead d ≥ 1 at which choosing by "min σ over the best d-step continuation" picks only correct actions (D\* = 1: greedy by σ is right; D\* ≥ 2: a garden path — the right move looks worse than a wrong one for d − 1 steps).
Both are instance properties computed by the generator; evaluation reports success by (B\*, D\*) of the *first* decision and by the maximum D\* along the oracle proof. Proof length k is reported too, as the closure stratum.

**Executor, oracle, posteriors.** Exact rewriting on term trees; the verifier checks each step's validity (match, in-world) and the final equality; BFS from t under W gives the distance, the set of shortest proofs (the ties), and the reachable graph within T steps. **Exact posteriors** over routes — under the router's own policy (EX) or under a flat prior over valid actions (PF) — are forward–backward passes over the reachable graph (small at size ≤ 5; a per-episode sparse CPU computation for the derive step, the PF arm and checks).

**The router.** A small transformer encoder (d 128, 3 layers, 4 heads) over `[axioms of W] [goal s] [SEP] [current term t]`, with a head over (rule, position) + HALT, applied once per step; the executor applies the rewrite and the router runs again on the new term. Markov in (t, context); no rollouts in rung 4 — this is the amortized prior ("the PFN over tactics"), and how far it reaches in D\* without search is one of rung 4's readings.

**Loss.** Per-step cross-entropy on labels (RT); the route-level signals as in rung 2. Success: the committed route reaches s and halts within T = 2·k_max + 2.

## 2. Rung 4's questions (symbolic executor; the prior)

- **Q1 (core) — recombination across worlds.** Trained on some axiom sets, tested on axiom sets never seen: does the router plan with rules it never saw together?
- **Q2 (core) — term-size extrapolation.** Sizes 6–7 at test: positions and subterm shapes never seen. Rules against tables: a rewrite decision learned as a pattern over subterms extrapolates; one learned as a table over whole terms does not.
- **Q3 (core, new in revision 2) — the prior's reach in lookahead.** Success by D\* with no search: how much garden-path lookahead the amortized prior absorbs from data. This is the baseline rung 5's rollouts have to beat, and under reading B it is the quantity that sets how much search the prior saves.
- **Q4 (check) — depth closure.** Proofs of length 7–10 within training worlds (F30).
- **Q5 (reference) — labels against search.** EI16 (beam with the verifier) against RT: do search-found routes compose across worlds as well as labelled ones? The toy form of Lean's human-proofs-versus-expert-iteration choice.
- **Q6 (ties).** Worlds with A and C have several shortest proofs. RT on a canonical proof commits to one strategy; RT on the posterior over shortest proofs and PF keep a mixture. Which strategy each signal commits to per world, and whether it transfers to held-out worlds.
- **Q7 (PFN inference, one arm).** Axioms unstated, two example proofs from the world in the context: does attention over examples identify the world?

## 3. Held-out structure

**Training: eight worlds** — {A}, {C}, {I}, {D}, {A,C}, {A,I}, {C,D}, {I,D}. **Held out: {A,D}, {C,I}, the four triples, {A,C,I,D}** — every held-out world recombines axioms seen singly and in pairs. Size: train ≤ 5, test 6–7. Length: train ≤ 6, test 7–10. Strata: world class (train / held-out pair / triple / full) × (B\*, D\*) × size × length.

## 4. Arms (§12.3–4)

| arm | signal | role | seeds |
|---|---|---|---|
| **RT-canon** | teacher forcing on one canonical shortest proof per instance (lexicographically first) | core | 10 |
| **RT-post** | teacher forcing on the posterior over shortest proofs (soft targets; the tie-aware label) | core | 10 |
| **EI16** | beam (W 16, ε 0.1) over valid actions with the verifier, hard EM | reference Q5 | 3 |
| **PF** | exact flat-prior posterior over routes reaching s (DP on the reachable graph) | reference Q6 | 3 |
| **RT-examples** | RT, axioms unstated, two example proofs in context | Q7 | 3 |
| **RT-shuffled-axioms** | axiom tokens permuted across episodes | null: the axioms are read | 2 |
| **RT-single-world** × 8 | RT on one training world each | reference: each world's program unmixed | 1 each |
| planted router | hand-written: match rule patterns at positions; choose by the oracle's distance table | expressivity reference (L17) | – |
| planted table router | lookup over whole training terms | the "tables" outcome of Q2, planted | – |

Same router, same parameter count, in every learned arm; lr from the pilot per arm.

## 5. Derived before measuring (`derive/`)

- Reachable-graph sizes and BFS distances per world and term size; tie multiplicity per instance; the share of instances whose correct first action is unique.
- **The (B\*, D\*) distribution per world** under the chosen σ, and the proxy's choice: σ must make D\* ≥ 2 common enough to populate the strata (if garden paths are rare under {A, C, I, D}, the generator oversamples them or the proxy is reconsidered — decided here, before the pilot).
- The closure table (Q4) and the recombination table (Q1): which (subterm pattern, rule) decisions each held-out world needs and in which training worlds each was taken.
- The planted router exact on every world, size and depth; the planted table router failing size 6–7 as predicted.
- Wandering mass at init under a uniform policy per depth (the answer-only signals' starting gradient).

## 6. Predictions (fixed here; counts over seeds)

| # | prediction | falsifier | type |
|---|---|---|---|
| P1 (check) | Within training worlds, RT-canon ≥ 0.95 success at lengths 7–10 in ≥ 9/10 seeds. | ≤ 7/10 | expressivity / optimization (derivable) |
| **P2 (core, Q1)** | RT generalizes to the held-out **pairs** in ≥ 6/10 seeds (≥ 0.9 success at in-range length and size), to the triples in fewer, to the full set least. | ≤ 3/10 on the pairs: no composition across axiom sets; the labelled prior must cover combinations. | optimization |
| **P3 (core, Q2)** | Size-6–7 extrapolation in ≤ 3/10 seeds, failures at unseen positions (the table signature). | ≥ 7/10: pattern rules; the stronger PFN claim. | optimization |
| **P4 (core, Q3)** | Success falls with D\*: ≥ 0.9 at D\* = 1, ≤ 0.5 at D\* ≥ 3 for RT-canon within training worlds — the prior absorbs one step of lookahead, not three. | flat in D\* (≥ 0.9 at D\* = 3): the amortized prior does the search's work from data, and rung 5's rollouts have less to add. | optimization |
| P5 (reference, Q5) | EI16 generalizes to held-out worlds in no fewer seeds than RT-canon's rate. | 0/3 where RT ≥ 6/10 | reported |
| P6 (Q6) | RT-canon commits to the canonical strategy in every world; RT-post and PF keep a mixture; committed strategies differ by world. | — | reported |
| P7 (Q7) | RT-examples ≥ 0.9 on training worlds; fewer held-out seeds than stated-axiom RT. | RT-examples ≥ stated RT on held-out worlds: attention reads the generator's signature directly. | reported |
| P8 (null) | RT-shuffled-axioms at the single-world baseline. | — | check |

**Reading under the two rollout readings (I36).** Under B (exact lookahead at inference), P2–P4 bound how much the prior saves the search and recombination is nearly free from the executor's side; under A / C, P2 and P4 are half the claim and the latent dynamics' generalization is the other half, measured in rung 5.

## 7. Rung 5 on the same data (outline; its own design after rung 4)

What the symbolic rung cannot test: learning the primitives and the router together (risk 2; F22–F23, F32), and the per-decision search itself. Rung 5 keeps the data, the verifier and the (B\*, D\*) strata and adds:

- **Arms.** Dense transformers of depth L predicting the next step from (W, s, t) with no scratchpad; dense **with a scratchpad** (best-of-B chains with D-step written lookahead — the chain-of-thought opponent, matched in exploration budget); the **library**: m small experts (attention and MLP), a router with explicit state, and the **propose → per-branch rollout → value → select** structure, in the three rollout forms of I36 — (A) latent-only, (B) executor-stepped, (C) split — each trained on labels, by verifier-guided search, and by end-to-end gating with load balancing (the SUT / MoEUT regime). Planted experts and router as the ceiling. In the toy the latent rollouts can be supervised against the executor's real consequences; an unsupervised variant measures what that supervision is worth, since at scale the kernel is available at training time only.
- **Endpoints.** Next-step accuracy and success by (B\*, D\*), never the mean; **the latent-against-exact lookahead curve** — the share of (B)'s gain over the no-rollout prior that (A) and (C) capture, per D\*, at what cost in forward passes — which is the measurement that decides I36; compute per solved problem in the lookahead tail at matched success, against the scratchpad at matched forward passes and matched tokens; parameters at matched tail accuracy against the dense family; a functional probe of each learned expert; **distillation calibration** — distill the dense-L arms and the planted library into libraries and recover m(L) before the instrument is used on a real teacher.
- **Predictions, to be fixed in its own design.** Dense-L cliffs at a D\* set by L; the scratchpad extrapolates at token cost; the library extrapolates under labels and search; the gating-trained library matches in range and does not extrapolate, with blended experts under the probe (the chains-against-bags explanation owed for MoEUT); (A) tracks (B) at small D\* and loses to it as D\* grows, with the crossover the number that matters; the value head does not collapse because breadth is built in (F40's missing exploration) — preregistered, not assumed.

## 8. After the toys

Synthetic symbolic math with derivation traces for the first real model; Lean tactic traces (Mathlib via LeanDojo) as the real-data step — RT is supervised traces, EI16 is verifier-guided search, premise selection is the lookup part, the ambient theory is the world, and the Lean-level endpoint is **kernel calls and tokens per proof at matched success**; natural-language math as a benchmark only. The distillation instrument, calibrated in rung 5, measures m(L) on a math-trained dense teacher first.

## 9. Budget

Rung 4: 10 + 10 + 3 + 3 + 3 + 2 + 8 = 39 runs; small router, cheap executor, but the executor runs in the training loop — vectorized term rewriting (or a CPU worker pool) is the pilot's first measurement; ≈ 10–20k steps per run; one Kaggle session if vectorized, two otherwise. Building the carrier (terms, rules, BFS, verifier, DP posteriors, generator with (B\*, D\*), router, beam, analysis), tests first, is the main cost: on the order of a week of agent work.

## 10. Gates

G1 tests: rewriting on hand-built terms; BFS distances against brute force; the verifier rejecting invalid and out-of-world steps; DP posteriors summing to one and matching enumeration on tiny instances; (B\*, D\*) on hand-built garden paths; the planted router exact on every world; the table router failing size 6–7 as predicted; generator strata populated. G2 derive (§5), including the σ decision. G3 pilot (lr, steps, executor cost; one seed; reads nothing, L30). G4 independent design review of this revision, then a code review before the grid.

## 11. What this rung does not answer

Learning the primitives and the per-decision search (rung 5); chains against bags in a trained model (rung 5); the rollout form (I36; rung 5); where labels come from at scale beyond what Q5 says; worlds as hidden latents with piecewise constancy; the prior over proofs at Lean scale (this carrier's PF arm can only gesture at it).

## 12. Choices fixed (the researcher, 6 Nov, "good with defaults")

1. Axiom set {A, C, I, D} with the directions of §1; inverse / cancellation deferred.
2. Worlds stated in the context as the core; inference from example proofs as one arm.
3. Labelled core: both RT-canon and RT-post at 10 seeds.
4. Rung 4 (symbolic) before rung 5 (neural, same data).
5. Term size: train ≤ 5 leaves, test ≤ 7; exact-posterior arms kept.

## 13. Prior work

PFNs and in-context learning of algorithms; equational reasoning and rewriting-based theorem proving; neural tactic prediction and search in Lean (LeanDojo / ReProver; HTPS; AlphaProof; the DeepSeek-Prover line); compositional generalization benchmarks; model-based search in latent spaces (MuZero's value-equivalent models) for the rung-5 rollouts; DreamCoder's recognition model; the routed, layer-shared, halting transformer family ([literature.md](../../docs/literature.md)). New here, to our reading: per-decision difficulty (B\*, D\*) as the stratification of a proof task, worlds as stated axiom sets with recombination held out, and a toy in which the three forms of internal search can be compared against an exact executor.
