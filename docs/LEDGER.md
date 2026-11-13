# Ledger: findings, failures, predictions, ideas

> "Rather than fighting to make our hypothesis true, we should be aware of the honest things our data says about our hypothesis. There is no pride in this. If I am wrong, let me be wrong fast." (the researcher, 22 Oct 2026)

**What this is.** One place where every claim the program makes is listed with its honest status. It is the first file to read before designing anything, and the first to update after any result.
- [history.md](history.md) records *why* decisions were made.
- This ledger records *what we currently believe and how strongly*.

**Rules**
1. **Every entry carries:** an ID, a status, its evidence (with a link) and its scope. Hypotheses also carry their prediction, falsifier and cheapest test.
2. **Statuses**
   - Findings:
     - **established**: preregistered and replicated, or derived and tested;
     - **strong lead**: preregistered, not yet replicated;
     - **suggestive**: exploratory, or few seeds.
   - For anything resting on reasoning alone:
     - **argument**: derivation or reasoning, untested;
     - **prior art**: answered or partly answered in the literature.
   - Labels for entries that leave the main lists:
     - **failed**: a prediction we made, contradicted by data;
     - **refuted**: an idea killed by data or derivation;
     - **inconclusive**: the test could not discriminate.
3. **Status changes are appended with a date, never overwritten.** Example: "suggestive → failed (2026-11-02, link)". A failed prediction is moved to *Failed predictions and refuted ideas*, never deleted.
4. **Before proposing an experiment,** check this file and [literature.md](literature.md) / [literature_review.md](literature_review.md) for whether it is already answered (lesson L2).
5. **Claim types,** as in AGENTS rule 5: expressivity, optimization or mechanism. Most of what we know so far is optimization.

## Direction

The researcher's end goal, in his words:

> "A real learner shouldn't need to walk the loss landscape, it should be able to see the minima on the horizon by reasoning over the structure of the data it has learned, which generalizes to unseen examples. All of the experiments I am interested in are related to that end-goal. But we must make small steps towards that vision to understand what is happening and why."

### Current orientation (29 Oct 2026): the architecture thesis — read this first

The researcher's framing (history Entry 65), which the rungs serve:

- **Depth buys capability but cannot be trained as unique layers.** A model with a million unique layers is infeasible: no signal reaches most of them, and no token needs most of them.
- **The basis claim.** What those layers would compute is a serial composition of a small basis of primitive layer functions. A layer is a high-dimensional function that can be approximated by a basis expansion, like a Taylor or Fourier series whose terms are primitives. So a router over m primitives can give the expressivity of a very deep unique-layer model at tractable cost.
- **Per-token elasticity.** The router allocates computation per token: a token that needs to go back through the context can, and the rest do not pay for a static graph.
- **What the router is.** It is like a PFN: amortized inference that assigns, per token and per step, which primitive to apply next. Routing is query–key–value assignment over layers, and also learning the **contextual value of each layer** given the input sequence and the token's current computational state.
- **Composition is a search process.** Even if some primitives commute as functions, the route is chosen adaptively from intermediate states, so composition cannot be treated as commuted. Commutation only shrinks the search, from sequences toward multisets.

**Derived reading (Claude, agreed in discussion; Entry 65):**
- The layer update is a state-indexed basis expansion of a residual vector field, x_{t+1} = x_t + Σᵢ cᵢ(context, x_t) φᵢ(x_t). The router supplies the coefficients from the state, not from the depth index, so the field is autonomous; that is why rung 1's Markov router generalized in depth (F30).
- Composition over a series is worth exactly the non-commutativity of the primitives (Magnus / BCH), and adaptivity over a fixed program is worth the value of intermediate information (rung 1's state-blind gap: 0.84 → 1.00, F34).
- "Contextual layer value" is a value function Q(context, state, primitive), defined by a Bellman recursion over the downstream policy. Myopic signals (one-step loss reduction, matching pursuit on the loss gradient, confidence) assign zero value to setup steps (D74, SET).

**Risks to the thesis, ordered (what the rungs exist to retire):**
1. **The basis claim itself.** Untested: the fidelity of a trained deep model approximated by a router over m primitives, as a curve in m (Entry 51's distillation test).
2. **Learnability of generic, reusable primitives.** The program's strongest negatives are F22–F23 (joint training damages the block) and F32 (a second gated circuit is not learned).
3. **A router training signal that generalizes without exact inference.** Sampled vs search vs value vs amortized-posterior signals (rung 2). *Retired for the toy, 30 Oct (F35):* the sampled current-policy posterior, the sampled fixed-prior posterior and search + imitation all reach the program within 1.6× the exact marginal; reward-only signals collapse (F40).
4. **Contextual value of computation.** Value functions with bootstrapping vs myopic signals, which decides setup steps and elasticity. *Not readable yet (30 Oct, F40):* every reward-only signal collapsed into a deterministic policy before learning; an exploration term is the prerequisite.
5. **Pooling as the efficiency mechanism.** One decision learned at every depth (D100).
6. **Depth extrapolation of allocation** (F24's confirmation, D93).
7. **Interference of primitives in a shared stream** (Entry 51's basis frontier).

**Experiment-design rule (L27).** Each rung retires one of these risks, and its core must be cells whose outcome we cannot derive. Derived or already-observed outcomes are checks, at minimal seeds. The value of a rung is measured by the decision it gates in the list above.

- **The original question:** what structure do network topologies promote during learning, and can wiring support the generalizing solution and keep the memorizing shortcut from interfering?
- **How the frame has moved (22 Oct):** from "which wiring" towards "how learning is organized":
  - the gradient as a single-path, averaging learner;
  - shared error versus separate errors;
  - search over structures versus walking;
  - topology as a hypothesis about how the data is generated.
- **Scope principle:** toy tasks isolate one piece; realism is added back step by step. A fixed dataset is not the learning system we care about in the limit.

- **How the frame moved again (24 Oct discussion, no new data):**
  - In a gradient-trained network the prior is not a separate object: it is the bias of the search (optimizer, topology, settings together). E1's predict-then-learn code length measures how well a learner's prior fits the data.
  - The horizon is short in weight space (E2) and 3 examples long in hypothesis space (E1), so "seeing the minima" points to holding hypotheses explicitly, not extrapolating a walk.
  - **Generality at the population, efficiency at the member:** a general network is data-hungry for the same reason it is general. A population of restricted members can be general overall while each learns from few examples, at a cost of about log N nats of evidence for the selection, paid in compute rather than data.
  - The open problems this exposes: where the members' structure comes from (selection across tasks, growth, compressed generators), how credit flows *between* members (responsibility by out-of-sample prediction, I20), and composition versus competition (sums share error; mixtures do not compose).
  - The two claims everything rests on: I18 (member level) and I19 (transfer).

### Update (1 Nov 2026, after rung 2 and the 30 Oct – 1 Nov discussions): sharpenings, and the risk list revised

History Entries 69–70. Each line carries its status: *argument*, *supported* (by a finding), or *open*.

**Sharpenings of the thesis.**
- **Resolution comes from iteration, not parameters.** *Argument, supported.* The researcher's picture of the data distribution as fractal (coarse structure, detail at every zoom) supports the thesis for a precise reason: a fractal is a short generator iterated, and its detail at a zoom costs iterations, not parameters. A static-depth circuit resolves the distribution to a fixed zoom; the only ways past it are more parameters or more iterations. Supported: F30 (trained at length ≤ 8, exact at 32), rung 0; theory: fixed-depth transformers are expressively bounded and iteration lifts the bound.
- **The distribution is grammar-like, not self-similar.** *Argument.* Phonology, syntax, discourse and facts are different structures at different scales, so the basis needs scale-specific primitives. Risk 1 is "how many generators," not "is there one."
- **The loss splits into a generated part and a particular part.** *Argument.* Facts have no generator; memory (parameters or retrieval) is the right mechanism for them, and much of the scaling gain is there. The thesis addresses the generated part. The particular part should be large and shallow (lookup), the generated part small and deep (iteration); a static model spends the same depth on both, and the router's first job is to tell which kind of token it stands on.
- **Routing time with state, not routing width.** *Argument.* Mixture of experts already routes with learned parameters "about navigation" and has not produced resolution beyond parameter count, because it routes width: one step, no state, no iteration. The claim must be about iterated composition with a carried state, or MoE is its counterexample. Falsifiable form: at fixed parameters, on a task with a depth variable, a routed-iterated model's error falls with allowed iterations and a static model's does not.
- **What a routed model has over a loop, with evidence.** An explicit writable state that carries setup credit (F29, F31, against F32: a loop with the same compute never learned the second circuit); a choice among primitives (a loop is a router with one option); elastic halting (rung 0, fragile); per-step commitment, which keeps long compositions exact where a soft state accumulates error (F39). **Branching** is a hypothesis in three forms: superposition in one state (latent-reasoning work; interference, risk 7), explicit parallel states with a merge (needs a judge), and training-time search (the slime mold with the label as food). Only the third has evidence: search + imitation was fastest and most robust in every rung-2 cell, and forced breadth is the fix for F40's collapse.
- **The control loop.** *Open.* In a static graph the estimate at layer ℓ changes nothing about what later layers do; the model can represent "I don't know" and spends the same compute. Routing closes the loop. The toy router did not close it either: it routed on the symbolic state, never on its own prediction or policy entropy.
- **Epistemic, not aleatoric.** *Argument.* Sharpening is right only when more computation would change the answer; next-token entropy is mostly aleatoric and a model trained to sharpen it would learn overconfidence. The quantity to act on is the value of computation ("would more computation change my answer"), with the known failure modes: myopic (a setup step changes nothing immediately, D74) and confidently wrong (no spread to act on). A single pass cannot measure it; **breadth estimates it** (disagreement among computations) and **depth reduces it**; the router's own policy entropy marks the forks worth splitting. In the toy, the committed routers have no forks once trained (the marginal over routes beat the committed route by 0.007 on the state-blind reference and by nothing on the state router); the stochastic routers of rung 2 (PF-IS16: right answer on 3 of 4 samples) are the case where disagreement carries information.
- **The program manifold is smaller than the data manifold; that is the bet, not a free lunch.** *Argument, supported.* The navigation prior generalizes in the ordinary sense over the space of programs (F30: exactly the closure of per-state decisions), and it inherits the inductive-bias problem: among programs the data do not separate, the procedure chooses (F37), and the choice shows only off-support. The researcher's position: more varied data break any given tie, and at scale undisambiguated ties are rare. Claude's: the ties that remain after all the data are the off-support behaviour the thesis is about, and data break a tie only if the learner can still move (F36 is the from-scratch evidence). **This is the live disagreement; rung 3 tests it from committed starts.**
- **PFN and AlphaZero, reconciled.** *Argument, supported in parts.* The value the router needs at training time is the backward message of the route posterior (the probability of reaching the outcome from a state under the prior and the cost); PF computed it exactly, EI16 searched it. At inference the outcome is unknown and the router carries its amortized expectation over worlds, which is the "prior over the data distribution" folded into the policy. A fixed prior cannot lock in but is capped by the prior (SET padding, dose fragility: F41); a self-generated prior improves but can lock in (F36, EI16B) and needs search to stay honest. Synthesis: **search supplies the targets; a fixed prior over programs breaks the ties the data leave.** A learned value head earns its place only with search at inference or targets too deep to compute, and it must see the query (F40's critic: explained variance 0.05 without it).
- **Library growth by abstraction.** *Argument (DreamCoder).* Reused compositions should become primitives: depth traded for width over time, which keeps inference tractable as programs deepen and gives "continual memory" a role beyond facts (programs worth naming). Without it a routed model pays full compositional depth at every zoom forever.
- **Full coverage.** *Argument.* Grant the primitives universal coverage; a universal basis is tiny, so all difficulty moves into composition, which is program synthesis. The router is amortized program synthesis; what makes it tractable is the learned prior over compositions and the abstraction cycle above.

**Risks to the thesis, revised (the 29 Oct numbering kept).**
1. **The basis claim.** Unchanged, and now with two curves to measure (rung 4 target, D110): fidelity of a routed basis to a trained deep model as a function of the number of primitives m, and as a function of *iterations allowed past the teacher's depth*. The second is the fractal claim's direct test: if a distilled basis gains resolution by iterating further than the teacher could, the fine structure was generated; if it plateaus at the teacher's depth, the picture was a metaphor.
2. **Learnability of generic, reusable primitives.** Unchanged; still the strongest negatives (F22–F23, F32). The thesis presupposes the basis; the toy hands it over.
3. **A router training signal without exact inference.** Retired for the toy (F35). Carry-over: which signal rungs 4+ should use depends on risk 8.
4. **Contextual value of computation.** Reframed: the job is *to rank partial computations without the answer*, non-myopically. Prerequisite: an exploration term, since reward-only signals collapse before any value question arises (F40). Candidate estimator that needs no learned head: disagreement among branches. Rung 3 runs the prerequisite.
5. **Pooling as the efficiency mechanism** (D100). Unchanged.
6. **Depth extrapolation of allocation** (D93). Unchanged; F30 is partial support.
7. **Interference of primitives in a shared stream.** Unchanged; now also the cost of the superposition form of branching.
8. **Plasticity after commitment** (new). A committed router moves to the program later data require only if its targets can see that program; by how the targets are made (exact fixed prior, search under its own policy, importance-sampled fixed prior, own posterior), with coverage at the committed policy as the predictor. Rung 3's core. *3 Nov (rung 3 read; F45–F50): committed routers move under exact targets and under search; the own-policy samplers take the committed solution apart, PF-IS16 even on unchanged data. The predictor was wrong in form: for search, coverage decides whether the beam's tie can reverse, not when; for the exact own posterior the gate's mass was not the quantity (its moves pass through a halting state); the exact fixed prior sits on a plateau and escapes stochastically. The trainer of a committed router in later rungs is an exact target or a searched support; PFB16's plasticity is untested.*
9. **Abstraction of reused compositions into new primitives** (new, far). Untested; no toy yet.

**The next experiments.** Rung 3 ([commit/DESIGN.md](../experiments/commit/DESIGN.md)): commitment. Core A: plasticity after commitment (risk 8; the live disagreement). Core B: exploration for the reward signals, so that risk 4 becomes readable (entropy and forced-breadth variants of RL16 / AC16). Checks C: disagreement among sampled routes against the exact marginal as a predictor of committed-route error, and where the router's policy entropy sits (derivable from the saved routers; cheap). Then rung 4: the distillation test with two curves (risk 1).

### Update (6 Nov 2026, history Entries 74–77a): the frame restated by the researcher; rung 4 re-based on the rewrite carrier

- **The thesis, restated (Entry 74).** The router is a **PFN over programs under the data prior**: given the primitives and a prior over their compositions, test-time inference is amortized routing of programs; the ladder is not a study of a walk over routes. Under this frame rungs 2–3 read simply — PFN-like signals (the exact posterior under a fixed prior; search approximating it) stay plastic and carry ties as mixtures; self-training signals (the router's own policy as its prior) commit and cannot hear the data. The 1 Nov "live disagreement" about ties dissolves as the researcher said: a PFN carries a tie until data break it. Rung 3's leftovers I34–I35 drop in priority (they concern the walk).
- **The decomposition (the researcher's).** (1) Assume the corpus comes with the optimal program per token, train the PFN on it, test whether it generalizes across programs; (2) if it does, ask where the labels come from. "Optimal program" is a prior in disguise (programs are many-to-one onto outputs), so the honest label is the posterior over programs and the PFN is tested for calibration over ties. Step (1) splits into closure (a check, F30), **recombination of decisions across worlds under a cue** (the core), new decisions (a check, F49) and **rules against tables** (size extrapolation). Step (2)'s candidates: a synthetic prior, search on real tokens, a teacher model (rung 5), an exogenous reasoner, circuits read off a trained model; self-training excluded.
- **The architectural family and the scaling claim (Entry 75).** At the block-diagram level the proposal is a Sparse Universal Transformer / MoEUT / Mixture-of-Recursions (Universal Transformers with ACT behind them; Routing Networks and Modular Networks for the training-signal result) — [literature.md](literature.md). The boundary is not the architecture: it is what the router is (an amortized posterior over programs on an explicit state), how it is trained (posterior or search, against gating) and what it does (F30, F37, F45–F49). The researcher's scaling claim: a dense transformer's parameters are linear in depth and a token needing k > L steps is unreachable at any width; a library of m primitives is constant in k *provided* compositions at depth k use the primitives of depth 8 — m(L) saturating is the instrument (rung 5). Evaluate stratified by required depth, never the mean (MoEUT's flat average is consistent with the claim). Domain: math; the opponent is dense with chain of thought; **Lean is the real-model target** (tactic = primitive, proof state = computational state, kernel = exact executor, proof length = depth, ambient theory = world, premise selection = the lookup part). The researcher corrected the depth: proof length is the external loop; the thesis's depth is per decision (branch over candidates, roll each forward in the latent stream, select). Whether a rollout is latent-only, executor-stepped or split is **held open** and becomes rung 5's arms; rung 4 is reported under both readings (D122, proposed).
- **Rung 4 re-based (D119–D121, proposed; Entries 75–77).** The rewrite carrier ([rewrite/DESIGN.md](../experiments/rewrite/DESIGN.md), revision 3): terms, eight directed rules, fifteen axiom-set worlds stated in context, BFS oracle, exact verifier, exact posteriors. Q1 recombination across held-out axiom sets on the **binding stratum** (a missing axiom lengthens the proof — the Lean situation; P2), Q2 size extrapolation as rules against tables, a structural pointer router against a parameter-matched absolute control (P3), Q3 how much lookahead the amortized prior absorbs, read against a symbolic ladder (P4); EI16, RT-post, PF, RT-examples and RT-shuffled as references and checks (P5–P8). Then rung 5 (the neural basis with depth inside the step; distillation as calibration), a first real model on synthetic symbolic math, Lean. The programs design (revision 1) and the route carrier retire. Two independent reviews rebuilt the design (Entry 76: six broken items; Entry 77: five blocking items in the code, L34–L35); implemented, dataset built and derived, pilot ready (Entry 77a). The derive step found P4's preregistered cells (GP1 instances by (world, k)) unreadable — the greedy and lookahead-2 rungs tie near zero there — and the cells were amended before any run to every (world, binding, GP1, k) cell whose rungs separate (DESIGN §16, D124 proposed; L36). Pilot read 9 Nov (Entry 78): lr per arm set, 12,000 steps, main grid on four shards with a session budget (27 process-hours, 12 of them EI16's); the launch waits on the pass-rule audit (Entry 79). Audit 4 (9 Nov, Entry 79): six of the seven other pass rules amended before the main grid (DESIGN §17) — P2 scored on the 209 / 509 held-out binding instances that need the unseen combination, against a non-composing floor; cue questions scored on the unmasked reading beside the masked one, with the masked falsifiers kept; verdicts computed in code with planted tests; every run saves its router.

### Update (10 Nov 2026, history Entries 80–81): the rung-4 main grid, scored and independently reviewed

- **Verdicts under DESIGN §17** ([rewrite/RESULTS.md](../experiments/rewrite/RESULTS.md), [review 6](../experiments/rewrite/review/review6_results.md); findings F51–F55, failed predictions X42–X45). P2 (recombination, the rung's core) **not met, not falsified** — and not falsified by two instances in one seed. P3 (size extrapolation) **falsified on both clauses, neither of which tested its claim**. P4 **falsified**: the no-search router beats the proxy-guided lookahead-3 policy, also on pairs it never trained on. P5 holds (low power), P6 does not hold, P7 holds, P8 masked falsified / unmasked holds.
- **What the rung says about the thesis's step (1)** (the PFN generalizes across programs given labels). *Depth inside the trained range:* yes, far past the ladder (F53). *Recombination across worlds under a stated cue:* not shown; RT-canon scores like a non-composing policy on the instances that need an unseen combination, most apparent held-out generalization is instances that need none and a third of it is recall of trained rows (F51); and the cue at an unseen pattern costs success relative to ignoring it under the mask (F52, two seeds). *Rules against tables:* not asked cleanly (F54).
- **What the rung does not license.** A claim that the router cannot recombine (low power, unconverged curves, one arm at one lr; RT-post's seeds reach the bar in 4/10 unmatched). A claim that it learned a table over terms. A count of "steps of lookahead" (the ladder is proxy-guided and stops at three).
- **Defects of the instrument found by the run and its review, all mine:** the eval sets were never deduplicated against the training stream (36 % of training-world eval instances are training pairs; L45); P3's clause for the absolute control was unpassable and its structural clause mixed the feature clamp, growth past the training cap and position extrapolation (L43); P6's reference equals the target on 45 % of its instances (L42); a curve statistic of mine manufactured a late decline (L46).
- **Open with the researcher:** whether Q1 gets a sharper reading from the 41 saved routers before rung 5 — an enriched needs-the-combination set, deduplicated eval sets — which needs no training and can move P2 off its coin flip but cannot produce a pass; and what the recombination result means for the rung-5 design, which assumed the labelled prior composes across worlds.

### Update (12 Nov 2026, history Entries 86–88): a fixed question and a design — belief updating on a hidden-world carrier

- **The question the program now asks** ([belief/DESIGN.md](../experiments/belief/DESIGN.md), revision 0, unreviewed). Scored: after kernel verdicts on its own proposals, does a looped prover's distribution over next steps move as the exact posterior does, beyond a mask (Q1); and on held-out worlds is its belief the generalising posterior or the memorising one, as a function of the number of training worlds (Q2). This is the original thesis's "amortized posterior over programs" turned into a measurement, and rung 4's recombination question asked with coverage as the variable.
- **Why a hidden world.** With the world known a kernel verdict carries 0 bits (Entry 87). Hiding the eight rules leaves 6.2 bits after the problem is seen; the exact update differs from a mask on a third of post-verdict decisions; the generalising and memorising posteriors differ on nearly half the decisions at 8–32 training worlds ([prelim_info.md](../experiments/belief/prelim_info.md)). Solve rates differ by a few points only: the claims will be about beliefs.
- **What it is not.** Not an architecture contribution (every component exists, Entry 86), not a prover, not a comparison with published systems, nothing about real Lean.
- **Power (Entry 89, [power.md](../experiments/belief/power.md)).** On the existing executor the share of the exact update a model makes is estimated to ±0.005 from 2,000 instances, and its place between the memorising and generalising posteriors to ±0.01 at 8–32 training worlds. A larger algebra is not needed. The coverage axis is short (the two posteriors nearly coincide from 64 training worlds on); the unseen risks are seed spread, structured model error and learnability. Design revision 1 scores projection coefficients, not vote counts (L48).
- **Decided (D134, the researcher, 13 Nov):** the scored questions are Q1 and Q2. **D136 (the researcher, 13 Nov):** the latent arm is kept at one coverage level. **Defaults in force:** H8 on the existing executor; on-policy data only on pilot evidence. **Next:** independent design review of revision 2 (running), then stage 0.

### Update (11 Nov 2026, history Entry 83): the program's object changes — looped encoders and soft programs; contribution not yet identified

- **The researcher's decision (D132).** Discrete designed primitives are set aside. The object is latent transformation in looped transformers — soft programs as paths of a latent state under one iterated rule — in a Lean-like domain, with the latent computation in an encoder and tokens at the boundary.
- **What the literature settles** ([literature.md](literature.md), 11 Nov section). A looped block is one rule driven by a re-injected input; looping raises manipulation and not storage (Ouro); experts in shared layers buy storage at constant compute, with a router that matters little (MoEUT, ModernALBERT, Tying the Loop); partial sharing beats full sharing; coverage and test-time search were measured for theorem proving in 2021 (INT).
- **What it leaves open, as far as a brief search went.** Whether experts inside a loop take on transformations as well as facts; whether a learned soft program has reusable structure across problems; where extra compute should go off-support (loops, search, scratchpad); latent paths against a proof system's intermediate states.
- **Honest status.** The architecture is occupied ground and rungs 0–4 (routers over handed-over discrete primitives) lose most of their bearing on the new object. No contribution is established; the candidates are measurements. A novelty search precedes any design.

### Update (10 Nov 2026, history Entry 82): after rung 4 — the thesis restated against prior work

- **The researcher's critique of rung 4 (accepted).** Eight training worlds give a PFN no support for a reading on recombination; the cue was a name, not a description; the router was routed on a signal that was uninformative about the world. Raventós et al. (2023) predict the result: below a task-diversity threshold a transformer acts as the Bayesian estimator over its finite training tasks. Recombination is a threshold in coverage to be measured, not a yes/no ([literature.md](literature.md), the pushback review).
- **Three cheaper alternatives stand between the thesis and its payoff:** depth — a scratchpad (unbounded serial depth at token cost); reuse — one looped block (TRM / HRM: no library, no router); routing — a fixed or hashed assignment (matches learned MoE routing). The thesis is the conjunction: a library with a posterior- or search-trained router on an explicit state beating all three **off-support**, where dense shortcuts and amortization break. Four rungs studied the router with primitives handed over; none of the three comparisons has been run.
- **Hypothesis from rung 4 (post hoc):** the prior wins in support, search wins off it (0.94 against 0.72; 0.47 against 0.73 for lookahead-3); the no-search router was asked to do what the thesis assigns to per-decision search.
- **Route labelling (discussion, no result):** geometry in the latent gives a test for arrival, not a compass; learning routes and primitives together has our own negatives (F22–F23, F32) and a catalogued literature (Rosenbaum et al. 2019); designed primitives need a designed state and a trace, which mathematics has; a dense teacher's layer path as per-step targets is risk 1, still first and untested.
- **Open (D131, the researcher's):** which comparison leads the next design.

### Update (25 Oct 2026, after the priors round)

- **The data gap is the prior, not the search (F19).**
  - A rule the network already has survives on fewer examples only about where it would be found from scratch (S ≈ 1).
  - So credit routing, competing members, growth criteria or better optimizers inside this model class can't close the gap.
- **Generic restriction (width) doesn't help (F20).** The chain's extra cost looks like prior too (F21, suggestive).
- **E2's jump result survives a matched null (F16 note),** in a narrower form.
- **What this points to:** changing the *prior*. Structure matched to the task; priors or blocks learned across a family of tasks (I19, the amortized rival I23); search over priors rather than over weights. The open question is no longer "how does the walk fail" but "what prior makes the rule cheaper than a lookup table from a few examples, and where does that prior come from".

### State of the evidence (23 Oct 2026, after E1–E4)

A short synthesis for discussion. Each line points to the entry that carries the evidence.

**What we can now say**
1. **The large inefficiency is in data, not steps.** Gradient descent needs ≈ 1,044 examples where a program-prior learner needs 3; extra steps per example barely help (F13).
2. **Memorization is actively harmful to prediction before the rule arrives.** On a stream, the network predicts new pairs confidently wrong, worse than guessing, for most of the stream; more steps per example make this worse (F14).
3. **During the plateau, the network is building features, not waiting on a readout.** The current features never contain the answer early (F15, X11).
4. **The destination is partly visible from the run's own motion, but only in the stable wiring, and only a short way ahead** (F16, X12). Acting on it skips 4–14% of the walk. Whether the *direction* matters, rather than any kick, is untested (I13).
5. **Under shared error, step size is a steep lever on which branch learns what** (F18). It moves both kinds of function in the stable wiring as preregistered, but it is not the only lever: at matched times speed splits roles transiently, the chain holds memorization regardless of speed, and the higher-decay branch carries the grokking jump (X14, I15, I16).
6. **Chain depth carries an instability that output scaling does not remove** (F17, X15). The deep chain can learn, but its learning-rate window is empty: too slow below, collapsing above.

**What the results do not say**
- Nothing about learners other than AdamW-trained networks on modular addition, apart from the one program-prior reference. "Hundreds of times more data" is relative to that prior.
- Nothing about *why* the data gap exists: whether averaging (the statistical-query argument), the shortcut (masking, H3), or the parameterization. No experiment yet separates these.
- Nothing about whether wiring can separate memorization from generalization *by design* (X10 stands; I15 is an exploratory lead).
- The data floor was measured at one learning rate and weight decay per graph. Regularization is known to move the training fraction at which grokking happens (Power et al. 2022), so "1,044" is a floor for this setting, not for gradient descent in general.
- Nothing about kernels: "speed" was the nominal step size, not each branch's effective speed.
- No random-direction control for the jumps, and no replication of the sym/asym splits beyond 10 seeds on one task.

**Open questions these results sharpen**
- Is the data gap closable inside gradient learning (perturbation credit I12, priors, structure), or is it the signature of an averaging learner (I6, H9, H10)?
- Is the drift's direction informative (I13), and why is the chain's trajectory unpredictable (I14)?
- Is there a wiring-set division of labor (I15), and is a circuit formed in one branch and copied to another (I16)?
- Since structures need their own settings (X15), how would a search over structures compare them fairly (I2, L8)?

---

## Findings

### Established or strong

**F1 — First-order equivalence of graphs at zero.** *Established (derived and tested), expressivity/optimization.*
- Every graph in the parent-selection family has the same first-order learning step at zero branch weights. Topology enters only at second order in branch weight.
- The parallel arm (I+A+B) is a single matrix trained at doubled lr.
- A sequential chain trained from identity keeps det W > 0, so it can never reverse an orientation; the parallel graph can.
- Evidence: [phase0_linear_theory.md](phase0_linear_theory.md); history, Pivot 16.

**F2 — Longer composition chains have lower stable learning-rate ceilings.** *Strong lead (preregistered, one task), optimization.*
- Across all 24 four-block graphs, τ_b(C, log lr*) = −0.836, p < 1e-4. Root readers Q have a smaller, opposite-signed effect.
- The failure mode is collapse after fitting, not divergence.
- Caveats:
  - the floor effect;
  - "stable" partly means "not yet grokked";
  - 3 seeds, one task.
- Evidence: [Study 2](../experiments/stability/RESULTS.md).
- Replicated qualitatively in Study 3 (all 8 chain runs unstable), Study 4 and the Phase 5 pilot (asym unstable at every lr tried).

**F3 — Parallel routes grok fastest on modular addition.** *Established within the task (best-vs-best, 10/10 seeds), optimization.*
- par vs seq: P = 1.00 at each arm's tuned lr.
- Reason: the rule is shallow and needs no composition.
- Evidence: [Study 1](../experiments/grokking/RESULTS.md), A3.

### Suggestive (exploratory, but robust or repeated)

**F4 — Memorization and generalization co-locate.** *Suggestive, strong; mechanism.*
- In 6 of 7 mixed graphs, the blocks carrying memorization at its peak later carry generalization, in the same proportions: cosine 0.98–0.999, against a permutation null of 0.27–0.76.
- While memorizing, a block's support for the true label on unseen inputs goes *negative*. At grokking it flips positive in the same block: the generalizing function grows in the path that memorized.
- Evidence: [Study 4](../experiments/paths/RESULTS.md), Q-A.
- Not yet tested at a shared stable lr, or on a deep-rule task.

**F5 — No path destroys another's memorization.** *Suggestive, strong; mechanism.*
- The cancellation ratio is 1.00 at every time point in every graph. Memorization is removed by shrinking in place.
- Evidence: Study 4.

**F6 — Topology decides which path carries the function; the learning rate moves it.** *Suggestive (3 seeds per cell), optimization.*
- At high lr, function migrates onto an independent short reader, and only where one exists (`pk0120`: share 0.43 → 0.87 from lr 1e-3 to 1.6e-2).
- At moderate lr, the composed path wins even against two short readers.
- Evidence: Study 4, Q-B.

**F7 — Composition's grokking delay is mostly the learning-rate assignment.** *Suggestive, optimization.*
- Within a graph, delay × lr is roughly constant. τ(C, delay) = 0.447 ≈ τ(C, 1/lr) = 0.445; at the shared lr 2e-3, τ = 0.02.
- A residual cost per unit of lr·wd·steps remains: about 6.5 for short graphs vs 11–19 for chains. Untested.
- Evidence: Study 4, Q-C.

**F8 — Chain instability does not come from large weights.** *Suggestive (2 seeds), optimization.*
- The chain collapses at about half the parameter norm of the stable parallel net.
- Stronger weight decay makes it *worse*: more collapses at wd 3, no fit at wd 10.
- Evidence: [Study 3 calibration, round 4](../experiments/composition/CALIBRATION_NOTES.md).

**F9 — End-of-run composition saturates after grokking.** *Suggestive (3 seeds), measurement.*
- c ≈ +0.95 for both branches in every arm and at every lr; R within ±0.08.
- During the plateau, composition does vary (−0.15 to +0.46).
- Evidence: [Phase 5 pilot](../experiments/fastslow/PILOT.md).

**F10 — Per-branch weight decay changes branch capacity, not only time constant.** *Suggestive (3 seeds).*
- Weight decay 2 shrinks a branch's output about 3× at lr 2e-3, and about 20× at lr 1e-3.
- Evidence: Phase 5 pilot.

**F11 — In a chain, the stream does not build the intermediate as a stage.** *Suggestive (2 seeds, chain unstable throughout).*
- On ((a+b) mod p)·c: the parallel net represents s = a+b explicitly in every block. The chain writes a joint function of (s, c) at its last block, and s is never linearly decodable.
- This cannot be read as a topology effect until the chain trains stably.
- Evidence: Study 3 calibration, round 4.

### From "diagnosing the walk" (E1, E2; 23 Oct 2026; [RESULTS](../experiments/walk/RESULTS.md), independently recomputed)

**F13 — Gradient descent needs hundreds of times more data than a program-prior learner, and more steps barely help.** *Strong (measurement, preregistered design, recomputed).*
- A Bayes mixture over short arithmetic programs pins down modular addition after **3 examples**.
- Gradient descent needs **1,044 examples** (sym) before it groks at all, even with up to 60,000 steps. asym needs between 1,044 and 1,392.
- On streams, its production point sits at that floor (≈ 1,100–1,390). Four times more steps per example lowers it only slightly.
- Scope: one task, one optimizer, two wirings.

**F14 — Under predict-then-learn scoring, gradient descent is worse than not learning for most of a stream.** *Strong (measurement).*
- After memorizing what it has seen, it predicts each new pair confidently wrong: 12–29 nats per pair against 4.1 for uniform. Its total code length is 4.5–7× uniform.
- More steps per example made this *worse* (sym, lr 1e-3: 26k → 38k nats).
- This is the anti-generalization of memorization (F4) measured as a cost.

**F15 — The readout is not the bottleneck; the features are.** *Strong (E2 P1, as predicted).* A readout re-solved on frozen features leads the network by about one eval interval at most (median relative lead 0.006).

**F16 — In the all-parallel network, the destination is partly visible from the run's own drift, and acting on it helps modestly. In the chain network, it isn't visible.** *Suggestive (exploratory split, plus preregistered P2 in sym).*
- sym: extrapolating the recent drift reaches 50% test accuracy 15–22% earlier than training, in 10/10 seeds.
- Jumping along it saves 100–400 steps (4–14%), with no more instability.
- asym: no lead; jump outcomes are chaotic.
- Not separated from "any kick helps": there was no matched random-direction control (I13).

- *2026-10-25 (priors Q2, preregistered P5 supported):* the drift jump beats a null matched in per-tensor norm change in all 4 sym cells (pooled median 137.5 steps; 9/10 or 9/9 seeds per cell).
  - So the drift's *direction* is informative, beyond "any kick helps".
  - Qualifications: the effects are small (100–475 steps on runs of about 2,900); the rescale-only arm *costs* steps, so part of the margin is the null hurting; and a plain random kick matched the drift in one cell (jump 1,000, H 1,000).
  - F16 stands in a narrower form: a short-range, informative direction.
- *2026-10-25 (the researcher's reading, adopted):* **Q2 is essentially a result about momentum, not about seeing the destination.**
  - The drift is an exponential average of the parameter updates over about 500 steps. A jump of H steps along it is a one-off, long-timescale heavy-ball extrapolation, closely related to Lookahead and Grokfast-EMA (prior art: amplifying the slow component of the gradient accelerates grokking).
  - What the nulls add: the gain comes from the averaged update *direction*, not from the norm change it carries, nor from any kick. That is the premise of momentum, now confirmed in this setting.
  - The saving (4–14% of the walk) is no larger than what a modestly higher learning rate gives (Q1 scratch: higher rates grok sooner in sym).
  - It says nothing about a horizon beyond the current heading. H5's "visible destination" reading of F16 is withdrawn: what is visible is the heading, for a few hundred steps.

**F17 — The chain's stability ceiling survives output rescaling.** *Established (preregistered E4 P0, and the H8 falsifier met; recomputed with four alternative summaries).*
- The longest-chain graph (C4) has essentially no stable region at learning rates from 5e-4 to 6.4e-2, under std, a uniform 1/√L multiplier, or a per-block 1/√depth multiplier.
- C1 is stable up to 0.016–0.032.
- Strengthens F2.
- "No stable region" does not mean "cannot learn." At lr 2e-3, C4 groks in 3/3 seeds under std and P1 (2/3 under P2), reaching test accuracy ≈ 0.95–0.97, but with 14–22 collapses in 12,000 steps. Its window is squeezed from both sides: below about 2e-3, C2–C4 do not fit within 12,000 steps; above it, C4 collapses repeatedly. For C4 the two limits meet. (Read from `experiments/walk/results_data/e4_runs.json.gz`, wd = 1.)

**F18 — Under shared error, step size moves credit steeply, but it is not the only thing that sets credit.** *Mixed (E3: preregistered P1 and P3 supported in sym; P2 falsified as written in asym; matched-time reanalysis exploratory).*
- In the stable wiring, at equal memory time, the faster branch takes both kinds of function: about 2.8 logit units per unit of log speed ratio, steeper than the lazy-regime prediction.
- Compared at matched times, three things show up beyond speed:
  1. **Speed alone splits roles transiently** (sym lr-only, at mid-plateau): the rule goes to the faster branch first, while memorization stays put or moves slightly the other way.
  2. **In the chain-plus-shortcut graph, memorization sits in the chain regardless of nominal speed,** and a speed-independent split remains even at equal speeds: generalization spreads across both branches while memorization stays in the chain.
  3. **The higher-decay branch carries the grokking jump, then hands it back** within about 1,500 steps.

### From the "priors" round (25 Oct 2026; [RESULTS](../experiments/priors/RESULTS.md), independently recomputed)

**F19 — The data gap is the prior, not the search.** *Strong (preregistered P2, recomputed, robust to every matched criterion tried; one task, AdamW).*
- In the all-parallel network, a learned rule survives on a subset only down to about where training from scratch finds it: both thresholds are 870 pairs, so the search factor S ≈ 1.
- At grid resolution S is 0.8–1.25. It is at most 1.5 under matched variants, and 2 only when semi-grokked runs (50–85% test accuracy) count as keeping the rule.
- On a log scale, search accounts for 0–12% of the gap to the program-prior learner (870 vs 3). The prior (the equilibrium preference of AdamW, weight decay and this architecture) accounts for the rest.
- **Consequence:** better search inside this model class cannot close the gap. That covers credit routing, competing members, growth criteria, or optimizers that settle at the same equilibrium. Only a different prior can.
- **Prior art:** Varma et al. 2023 found S ≈ 2 in a transformer under a lenient criterion; we match that only under the same leniency.

**F20 — Width is not a lever on data efficiency here.** *Strong for the scratch floor (P4b), suggestive for the rest.*
- Narrowing 14× or widening 4× leaves the scratch floor at 870 pairs.
- The ungrokking threshold does not decrease with width (the opposite of Huang et al.'s transformer result; X17). Narrow is lowest or tied (522, borderline).
- Base against wide is decided by single-eval dips, so no ordering is claimed.
- Generic capacity restriction does not buy efficiency at the level of a member (I18, generic version).

**F21 — In the chain-plus-shortcut network, the extra data cost shows up on both sides.** *Suggestive; the preregistered P3 was a tie and is uninformative.*
- Ungrokking and scratch thresholds are both 1,218 pairs (sym: 870). That places the wiring's cost in the prior at least as much as in the search.
- Fragile: the asym floor rests on 2 of 3 seeds at one learning rate, and 24 asym scratch runs cross 0.95 without sustaining it.

### From rung 0 of the dynamic-depth ladder (27–28 Oct 2026; [REV3](../experiments/ptr/REV3.md), [PILOT1](../experiments/ptr/PILOT1.md), [PILOT2](../experiments/ptr/PILOT2.md))

**F22 — A shared block trained with a fixed loop or oracle halting iterates 4× beyond its training depth; the same block trained jointly with a learned halting policy does not.** *Strong (10 paired seeds per arm, preregistered arms, one task); mechanism.*
- LR (the stability exit) on the fixed-loop block: 1.00 to k = 16, 0.93 at 32 (trained on k ≤ 8); the oracle-trained block 0.92 at 32. On every learned-halting arm's own block it fails at the arm's own failure point (0.26 / 0.74 / 0.36 / 0.23 at k = 16 / 16 / 24 / 8 for LH-s / LH-Δ / LH-s24 / LH-s-k4).
- The heads are not the limit: chasing λ = 0.000 from step 2k in every arm; LH-s24's head continues to 36 loops.
- The failure point scales with the chain lengths seen (≈ 1.5·k_max) and moves out with the loop horizon (T = 24 → ≈ 22).
- Evidence: Entry 59; REV3 tables "LR@arm", "fixed@arm", λ by loop.

**F23 — Extrapolation under a learned halting policy is reached mid-training and then lost.** *Strong (8/10 seeds in each of two arms); mechanism (late-training degradation of the block).*
- At 4k steps LH-s and LH-Δ reach 0.89 and 0.88 at k = 16 (8/10 seeds ≥ 0.9 each); at 8k, 0.22 and 0.65 (0/10, 1/10). LH-s24 keeps it (5/5); LO and LF never lose it.
- Pilot 2's "fails beyond ≈ 1.5× the horizon" was this state at the end of training.
- Evidence: Entry 59; REV3 "Extrapolation over training".

**F24 — A change-reading halting head with a fast-enough head learning rate halts at k + 1 and extrapolates 4× with no decay.** *Exploratory (2 seeds, the gate's ×10 cell; not the preregistered cell); mechanism if confirmed.*
- 1.00 at every k ≤ 32, steps k + 1.1 (cheaper than the one-parameter exit's k + 1.3), LR on its own block 1.00, 0.85 → 1.00 at k = 16 over training. Heads at w_KL ≈ −3 to −5 against −0.66 at ×1.
- Confirmatory run: D93.

**F25 — Oracle halting with a dominant one-hop share does not learn the hop.** *Strong (0/10 runs at tails ≤ 10⁻¹; 1/10 at factor 1); optimization.*
- Training accuracy at k = 2 stays at chance for 6k steps while the loss (71–76 % one-hop queries) falls to 0.05: the one-loop map wins and the reusable hop is never learned. A block pools signal across depths only when training iterates it on every example.

**F26 — Unique-layer depth learns hops sequentially; the shared block does not.** *Strong (preregistered P5, 5–10 seeds); optimization.*
- S8 (lr 1e-4, bracketed): k = 1 / 2 / 3 / 4 at 500 / 2,500 / 6,600 / > 25k steps; LO all eight by 3,600; LF by 3,200. The static stack is 2.6× slower to k = 3 when the k ≥ 3 share drops 7×; the shared block under LH-s 1.3×.

**F27 — On this task the one-parameter stability exit beats every learned policy that was preregistered.** *Strong (10 seeds); performance.*
- Geiping-style KL exit on the fixed-loop block: 1.00 to k = 16, 0.97 / 0.93 at 24 / 32, k + 1.1–1.6 steps, flat across thresholds 10⁻⁴–10⁻². Only the exploratory ×10 change head (F24) is cheaper.

### From rung 1 of the ladder (28 Oct 2026; [route/RESULTS.md](../experiments/route/RESULTS.md), [PILOT.md](../experiments/route/PILOT.md), [DESIGN.md](../experiments/route/DESIGN.md) revision 2)

**F28 — The exact route marginal learns a four-action program with a setup step from next-token likelihood alone, ≈ 100 steps behind supervision on oracle traces.** *Strong (10 seeds RM, 5 RM-T, preregistered P1); optimization.*
- Deferred trigger: SET (a step that changes nothing in the prediction) flips at 600 [500–600] steps of batch 64 against RM-T's 500; HALT/HOP ≤ 100, PREV 100; every k at criterion by 600–1,200. Committed route = the oracle's on every in-range stratum, steps equal to the oracle's; head weights within the AdamW displacement bound (L24).
- Evidence: Entry 63; RESULTS §1–2.

**F29 — Setup credit arrives through the explicit state: the same objective does not find the setup step when its payoff decision must re-read the context at a state the init phase has labelled otherwise.** *Strong (5/5 seeds RM-d0 fail at 6k steps while 10/10 RM succeed; the planted deferral-0 program is a stable optimum under training); optimization / mechanism.*
- In the first 50 steps both routers learn, from the mass that strays into SET at init, the state-only rule "set here ⇒ HOP, carried count ⇒ PREV". With the deferred trigger that is the oracle's rule, so SET's credit is large once learned. With the local trigger the oracle's PREV sits at the set-here state (flagged), against a 0.99 HOP, and SET pays only if that PREV exists: the two decisions wait for each other through 6,000 × 64 steps.
- The local trigger was designed as the easy control; it is the hard case because SET and PREV share a symbol and cannot be separated by the state.
- Evidence: RESULTS §3, `experiments/route/diag_deferral0.py`, `results_data/diag/`.
- *30 Oct (rung 2, F36):* the basin is caused by training on the router's own posterior: with the same likelihood, fixed-prior and searched targets learn the local trigger (PF 2/2, PF-IS16 3/3, TB16 2/3, EI16 2/2) and current-policy targets do not (EX-AL 0/3, IW16 0/2). F38 adds a seed-dependent escape (program R) that routes around the labels.
- *Qualified 29 Oct (rung 2 review 3, Entry 66):* EX's step-0 target, pooled by state class, already points the right way in every class (no stray-SET push: +0.7 vs HOP +189 at unflagged states), so "learned from the mass that strays into SET at init" describes the first-50-step dynamics, not the initial target. Whether the basin comes from self-consistency or the readout's credit for halting at a key is rung 2's P2 (EX vs EX-AL vs PF).

**F30 — A Markov router over an explicit product state generalizes exactly to the closure of its per-state decisions under the state encoding's geometry.** *Strong (10 seeds, preregistered P2, P3a, P3b, with counterfactual-state probes); mechanism.*
- Length 4× (1.00 at k = 32, steps = oracle), every training word at every depth (alternation, deep P: 1.00); the unseen count value 2 handled (probe (a) 1.00 in 10/10: the count enters the state token linearly and the PREV decision extrapolates along that ray); the unseen conjunction "flagged ∧ pending" not (probe (b) 0.03; PP runs 0.00 in 10/10 with first deviation SET>PREV 2,411 / 2,497): no training example constrains the precedence of two learned rules and no geometry decides it. Teacher forcing on traces (RM-T) fails PP the same way.
- New orders fail for a state-routed learner only where the data leave a decision undetermined — the module-network outcome at the level of rules, bounded exactly.
- *30 Oct (rung 2):* the bound does not depend on the training signal: no signal that learned δ = 1 passes PP (E programs fail by precedence, L programs by the read), and every signal that learned δ = 0 passes.

**F31 — The explicit richer state buys a deterministic program at oracle cost, not Bayes accuracy.** *Strong (5 seeds RM-s vs 10 RM, preregistered P9); mechanism.*
- The symbol-only router: committed route accuracy 0 on every P-word and answer accuracy equal to the all-H share; yet the argmax of the exact route mixture is 1.00 on in-range P-words and at k = 32, with p(target) 0.77 (0.60 at k = 32), one sampled run right 77 % of the time, +1.2 actions per P-word — a stochastic setup step whose mass the mixture sorts out (review 2's prediction, number for number). Three readings of one model: 1.00 / 0.77 / 0.00.
- Consequence for later rungs: answer accuracy under a route marginal does not distinguish a system that knows where it is in a computation from one that gambles well; steps, route determinism and sampled-run reliability do.

**F32 — The looped learned block and the static stack learn the hop chain and nothing of the second gated circuit under either trigger in the same budget.** *Strong (10 LF, 5 LF-d0, 3 S8; preregistered P5 in its strong form); optimization.*
- All-H words 1.00 at every in-range k at the training depth and under the stability exit; P-words at chance (0.11–0.21) at every depth of the deepest P, at every evaluation, at the oracle depth and at any loop (0.5 = the chance of 16 answers over 40 loops); LF-d0 = LF (−0.00 / −0.01). ≈ 380k P-word queries seen. The obstacle is not the two-match read of the deferred flag (the local one is not learned either).
- Caveat (review 1 A.iii): given circuits + learned selection vs learned circuits + learned selection; D99 separates gate from circuit.

**F33 — The step-cost prior selects between exact programs of different length.** *Suggestive (pilot, 2 seeds per cell, not preregistered); mechanism.*
- At λ = 0.005 (2/2) and lr 3e-3 (1/2) the router keeps the stray-taught rule and sets at every state (exact answers in both readings, never the oracle route, k extra actions); at λ ≥ 0.02 the carried-bit route wins (10/10 in the main grid). B1 in its weakest form.

**F34 — The state-blind reference holds in both readings.** *Strong (5 seeds RM-idx vs the derivation; P7); expressivity.*
- 0.843 committed / 0.850 marginal, Zipf-weighted in range, against 0.83–0.84 / 0.85 derived; RM 1.00; the whole gap is on words with a P. With decoys the Bayes mixture of a state-blind policy no longer rescues P-words (revision 1's 0.94 came from wandering mass dying on ⊥).

### From rung 2 of the ladder (30 Oct 2026; [routefind/RESULTS.md](../experiments/routefind/RESULTS.md), [PILOT.md](../experiments/routefind/PILOT.md), [DESIGN.md](../experiments/routefind/DESIGN.md) revision 4)

**F35 — Three signals that never enumerate the state space learn the deferred program within budget: the sampled current-policy posterior (IW16, VIMCO), the sampled fixed-prior posterior (PF-IS16, self-normalized importance weights) and beam search + imitation (EI16).** *Strong (5 seeds each: 4/5, 5/5, 5/5; preregistered success = shortest correct program within 3× the exact marginal's criterion; medians 1,300 / 1,200 / 900 vs 800); optimization.*
- 5.5–8.0 M sampled or searched routes against 51k exact recursions. Risk 3 is retired for this task family.
- The one IW16 miss is program R (F38), not a collapse. Trajectory balance (TB16) is slower by ≈ 3× and the 5k schedule cuts it off at 0.97–0.98 (δ = 0: 2/3 at 3,500).

**F36 — F29's basin is a property of training on the router's own posterior, not of the likelihood or the task.** *Strong as an intervention (every current-policy signal stalls on the local trigger — EX 0/2 + 0/5, EX-AL 0/3, IW16 0/2 — while every fixed-prior or search signal learns it — PF 2/2, PF-IS16 3/3, TB16 2/3, EI16 2/2); mechanism (causal).*
- EX-AL (the exact marginal with PF's answer-only likelihood) has PF's step-0 gradient to 1 % and matches EX on both triggers; the only thing that differs between EX-AL and PF is whose posterior the target is.
- Qualifies F29: the init phase's labels persist because the target is recomputed under the policy that carries them. A fixed target (PFN-style), exact or sampled, or a searched target does not reinforce them.

**F37 — Which program a signal lands in is a property of the signal.** *Strong for the counts (current-policy posteriors and exact PF → E: 3/3, 2/2, 4/4, 2/2; imitation → L in 10/10 across three cells, whatever the init beam returned); mechanism (inductive bias); EI16's L mechanism open.*
- PF-IS16 lands in L-leaning mixtures (`set_late` 0.59–0.78), not exact PF's E: the self-normalized estimator with the router as proposal carries a current-policy component the exact one lacks.
- On δ = 0 every learned run is L (the only expressible program), as derived.

**F38 — A third correct program, R: SET once at the start, then the oracle's rule one count higher.** *Suggestive (2 runs: IW16 δ = 1 seed 0, EX-AL δ = 0 pilot seed; inferred from endpoints — exactly +1 step on every word including all-H words, answers 1.00, deterministic, PREV at c = 2 by probe); mechanism.*
- R builds the program around the stray-phase labels in the spare count capacity: the PREV decision sits at a state the init phase never labeled. It is perfect on answers, one step longer than the oracle, and fails PP like E.
- A shortest-route criterion scores it as a failure; an answer criterion as a success. Rung 3's metrics must keep both.

**F39 — Posterior-matching signals are stochastic at their optimum; EM and imitation signals are deterministic.** *Strong and derived (PF, PF-IS16, TB16: committed route shortest-correct, sampled route shortest-correct on 0.05–0.18 with 0.8–1.8 extra steps; EX, EX-AL, IW16, EI16: 1.00); by construction, confirmed.*
- A router trained toward a soft posterior is a program only under its argmax; its committed accuracy decays on 32-long words (−0.1 to −0.8). TB16's samples get the answer right on 0.99 on δ = 0 while taking non-shortest routes.

**F40 — Reward-only signals collapse into a deterministic policy within 200–400 steps and never learn, whatever the baseline or value.** *Strong (RL16, ACMC16, AC16: 0/5, 0/5, 0/5 on δ = 1, 0/3 on δ = 0 and at the dose; 16 identical samples per query from step 200–400; critic explained variance 0.05–0.09); optimization.*
- Exploration, not credit assignment, is the first failure of value-based router signals here. Pure TD(0) collapses into halting at the start instead (AC16T0 0/2; §5.5 of the design). Contextual value (risk 4) is unreadable until an exploration term is added.

**F41 — The spurious-density dose (answers drawn per episode from a pool of 4, readout fixed) slows every signal and hurts the fixed-prior target most.** *Suggestive (2–3 seeds: EX +19 %, IW16 +31 %, EI16 +44 %; PF-IS16 from 1,200 steps to not at criterion, in a mixed program); optimization.*
- The fixed-prior posterior spreads over the coincident routes (0.99 of its mass at k = 8); the current-policy posterior and the beam's MAP route do not. EI16 is the fastest signal in every cell.

### From rung 3, stage 1 (2 Nov 2026; [commit/DESIGN.md](../experiments/commit/DESIGN.md) §7a, [stage1_report.md](../experiments/commit/results_data/stage1_report.md); phase 2 not yet run)

**F42 — A soft fixed-prior target restricted to the ε-mixed beam's completions (PFB16) learns the deferred program and the local one, in L, stochastic.** *Suggestive (5/5 on δ = 1 at criterion 1,100 [900–1,400]; 3/3 on δ = 0 at 800; sampled shortest-correct 0.3–0.4; preregistered PB3); optimization.*
- At rung 2's spurious-density dose it reaches criterion in 2/3 (3,800) against the matched constant-lr PF-IS16 reference's 2/3 (2,800): not worse in count, slower, so F41's fragility is in the fixed prior, not in the sampled proposal. The reference lands in mix / E, PFB16 in L (the beam's merge rule keeps one path per state, so the prior's padding cannot act). PF-IS16 at the dose reaches 2/3 on a constant schedule where rung 2's cosine schedule recorded 0/3 (L31).

**F43 — Forced breadth under self-normalized own-policy weights is not exploration, and an entropy bonus is not either.** *Suggestive (RL16-B, ACMC16-B, AC16-B 0/2 each over the beam's 16 completions with the weights' effective sample size at 0.01 of the support; RL16+H 0/2 at β ∈ {0.01, 0.03, 0.1}, keeping 1.1 / 7.2 / 13.5 distinct routes per query and learning nothing; preregistered PB2 and PB1); optimization (H14).*
- The beam supplies breadth; weights w(r) ∝ π(r) over the support throw it away as soon as π is peaked, which is the from-scratch collapse of F40 over a wider support. A reward signal that keeps exploring has to weight unsampled routes by something other than its own probability of them, and every such weighting is a posterior target (Entry 71, D113).

**F44 — Committed routers are what the design assumed, with one surprise: EX's own signal can leave a committed solution on its own.** *Strong for the commitment (7/7 EI16 → L and 5/7 EX → E at criterion 800, deterministic, not R); the collapse is one seed; the coverage facts are derive-step measurements on 271 P-word queries per router.*
- EX seed 0 learned E by step ≈ 700–1,000 and collapsed at ≈ 2,000 into halting at flagged pairs (loss 0.08 → 0.50; all-H words still solved); seed 4 was dipping at 2,500. Under δ = 0 the E routers hold exactly one relabeled product-state class at +21 to +32 logits; the own-posterior target puts 0.00 on the new route, 16 samples never contain it, and the beam keeps it on 0.73 / 0.47 of P-words for two routers and ≈ 0 for three — the merge tie breaks per seed, not by ε. From every L router the beam keeps the oracle's route (0.58–0.66) and the own-posterior target sees it on 0.00–0.01 of the words that must move (0.20 on one seed). L seed 0 has HALT, not PREV, winning at the relabeled class.

### From rung 3, stage 2 (3 Nov 2026; [commit/RESULTS.md](../experiments/commit/RESULTS.md); independently recomputed and reviewed, [review3_results.md](../experiments/commit/review/review3_results.md))

**F45 — Plasticity after commitment is a property of the signal: exact targets and search move a committed router; the own-policy samplers do not, and they take the committed solution apart.** *Strong for the split (from the E-committed routers under the local-trigger shift at p_P 0.3: RT 2/2, EI16 4/5, EX 3/5 + 1 R-like, PF 3/5, against PF-IS16 1/5 and IW16 0/5 at the end; the same split at p_P 0.15 and 0.08 and from the L routers); optimization.*
- The preregistered reading row that applies: "EX moves, IW16 does not → sampling, not self-consistency, is the lock." One IW16 run held the moved program for 1,000 steps before the all-H collapse, so the sampled target can see the alternative; it does not keep it.
- Commitment inverts two signals against scratch on the same data: EX 0/3 from scratch → 3/5 committed; PF-IS16 3/3 from scratch → 1/5 from E, 0/3 from L, and it loses or degrades a committed L router on the *unchanged* stream (0/2 held). A deterministic committed router gives the sampled fixed prior sixteen identical samples plus the 6–8 % label-reaching decoy routes; the outcomes are consistent with review 2's decoy-sample mechanism (not measured).

**F46 — From the E start, search's movement is a collective reversal of the beam's merge tie, and the checkpoint's coverage predicts whether it can happen, not when.** *Mechanism (E start, 14 EI16 runs); from the recorded in-run coverage curves.*
- The run's own coverage goes 0.00 → 0.92–1.00 inside one 100-step window and the move follows at once; s6 (gate coverage 0.02) reversed inside the warm-up, s3 (0.01) after 1,300 steps, s5 (0.47) after first losing its coverage to 0.00, s2 (0.000) at 2,500 at one dose and never at two. From the L start coverage is 0.72–0.84 from step 0 on every router and one of three never moves, so coverage is not the limiting quantity there. PA3's AUROC: within EI16 0.705 (between its falsifier 0.6 and its bar 0.75), pooled over the learned signals without PF / RT 0.821 (holds, carried by EI16 against three signals with zero coverage everywhere).

**F47 — The exact fixed prior parks every router, from either start, at one intermediate state within 100 steps and leaves it by a sudden escape, or not within the budget.** *Suggestive (14 runs); optimization; mechanism open (I34).*
- From E the margin at the relabeled class drops from +21–32 to ≈ +1.3 (p(PREV) ≈ 0.15) in 100 steps; from L, PF first removes the correct PREV (p 1.00 → 0.05–0.15) and parks at the same state. Escapes to p(PREV) ≈ 0.65 at 1,000 / 1,200 / 1,700 steps (p_P 0.3), 200 / 1,100 (0.15), 0/3 at 0.08, 800 / 1,200 / 2,700 from L. The plateau's loss is higher than the escaped state's (2.41 against 2.21): a metastable state of PF's own objective. No dose effect between 0.3 and 0.15 (the movers were faster at 0.15); a fixed budget censors a stochastic escape (L31).

**F48 — Under the shift, the own posterior's first act is to lose the SET at the flag; movement is an escape from the halting state.** *Suggestive (10/11 EX-from-E runs lose the SET within 0–500 steps at every dose; escapes to L or R 4/5, 1/3, 0/3 by falling p_P; 0/5 controls on the unchanged stream in 3,000 steps; 6/11 end halting against 1/12 without the shift); optimization; mechanism proposed, not isolated.*
- The event is sudden in p(SET at the flag) and invisible in the loss (the committed E route is already wrong on every P-word under δ = 0). One run escaped, relapsed and escaped again. The gate's posterior mass on the oracle route (< 0.005) was therefore not the operative quantity in either direction, and the design's own falsifier for the EX coverage reading (an EX move from L s0–s2) fired (X38). Proposed reading: with every weighted route wrong the own posterior is the policy itself and the step cost is the only consistent per-word gradient; it does not explain the dose dependence of the escape as written; an EX arm with λ = 0 would test it.

**F49 — From the L start the missing decision is of a different kind, and one router resisted teacher forcing for 3,000 steps.** *Suggestive (3 routers × 6 signals; one router for the anomaly); optimization.*
- L s1: RT (SET accuracy 0.00, p(SET) at the flag 0.15–0.22, loss flat), EI16 (own coverage 0.72–0.81 throughout) and EX (creeping) did not move it; PF left its plateau at 800. L s0 and s2 moved under RT, PF, EI16 and EX as 600–800-step sigmoids. By the preregistered RT rule the L-s1 cell is unreadable at this budget. Reading to test on the checkpoints (I35): under δ = 1 the L router never needs the own-flag feature at the flag state, so its logits there may not separate flagged from unflagged states at all (p(SET) at the flag 0.08–0.20 on every L router at step 0 under every signal) — a representation problem, against the E start's decision problem.
- Under shift B teacher forcing on L s0 removed L's PREV inside the warm-up and had not installed E's SET by 3,000: a dead zone between two programs that differ in one decision (also PF-IS16 at f = 0).

**F50 — Filling in undetermined decisions does not happen at the preregistered endpoint under any learned signal.** *Suggestive (3 seeds each); optimization.*
- pp8 shortest-correct 0.60–0.64 under EI16 and EX (L-L completions) at a cost of 0.3 on ordinary P-words on clean strata; PF 0.2–0.4; PF-IS16 0; RT 1/2. PF at f = 0 drifts L → E and stays correct, as derived (DESIGN §5.1); the pooled derivation's "PF fails pp8" held.

### From rung 4 on the rewrite carrier (10 Nov 2026; [rewrite/RESULTS.md](../experiments/rewrite/RESULTS.md); independently recomputed and reviewed, [review6_results.md](../experiments/rewrite/review/review6_results.md))

**F51 — Zero-shot recombination of stated axioms across held-out axiom sets was not shown; what looks like held-out generalization is mostly instances that need no unseen combination, and partly recall.** *Not met at the preregistered level in any of ten seeds; not falsified, by two instances in one seed (P2, DESIGN §17). One arm (RT-canon) at one learning rate; 209 pair instances in two worlds; curves still rising slowly at the end.*
- On the held-out pair instances with no proof inside a contained training world: RT-canon 0.47 [0.28, 0.67] masked, 0.37 unmasked; a non-composing reference 0.53; glued single-world plans 0.75; lookahead-3 0.73; the same router trained on all fifteen worlds 0.94. Above the floor in 4/10 seeds (falsifier ≤ 3/10); instance resampling returns the falsifier in half the draws. Triples: 0.38 against a floor of 0.21 (10/10) and a glued reference of 0.50.
- On held-out binding instances provable inside a contained training world: 0.93 (pairs) and 0.75 (triples). About 30 % of held-out eval instances are verbatim rows trained in a contained world; RT-canon scores 0.89–0.97 on those and 0.45–0.71 on the rest.
- The rule as first preregistered (§6) would have passed or sat on the line (0.77; 7/10 seeds). The pass-rule audit (Entry 79) changed the outcome of the rung's core prediction.
- No world shows recombination by this instrument: the glued reference is at or above RT-canon's median in each. RT-post (lr 3e-4, more distinct rows; unmatched) meets the pair conditions in 4/10 seeds and is above the floor in 8/10 — a hypothesis for a matched test, not a reading.

**F52 — Under the executor's mask the stated axioms are not what carries the world, and at an unseen pattern the true cue costs success relative to ignoring it.** *P8's masked falsifier (two shuffled seeds against ten; gaps 0.03 / 0.04 on the training binding stratum, 0.040 / 0.041 on pairs never trained on); the held-out comparison is post hoc.*
- Masked, the token-blind router is ahead of RT-canon over all held-out instances (0.78 / 0.80 against 0.68 [0.63, 0.79]) and above every RT-canon seed in the full world, on the triples' non-binding instances and on the triples' binding instances that need no combination.
- Unmasked the tokens carry the world on training worlds (0.93 against 0.58; the shuffled router fails by out-of-world actions) and on the needs-the-combination strata (0.37 against 0.15); over all held-out instances the two arms are level, and in the full world, where no action is out of world, the token-blind router leads by about 0.17 in 10/10 seeds.
- Consequence for F51: part of RT-canon's held-out shortfall is the unseen token pattern as an input, which this design does not separate from failing to compose decisions.

**F53 — Inside the trained term sizes, a router trained on shortest-path labels with no search at test time beats the proxy-guided lookahead-3 policy by a wide margin, and solves half the instances at proof length 7–8.** *Strong within the trained sizes (P4 falsified toward "three steps of search's work": ≥ the d = 3 rung in 10/10 seeds; 0.94 against 0.72 on the six separated cells, 0.93 against 0.73 on the 150 pairs never trained on, 0.91 against 0.76 on goals never trained on). P1: {A, C} at k = 7–8, no train overlap, 0.53 [0.32, 0.62] against 0.02.*
- Every other arm that learned is above the d = 3 rung on those cells too, the search-trained EI16 and the fixed-prior PF included (PF 0.91 at k = 7–8).
- Not a count of lookahead steps: the router's success is unrelated to the ladder's instance by instance (0.93 where lookahead-3 succeeds, 0.93 where it fails), the rungs are limited by a tree-edit proxy, and the ladder stops at d = 3.
- At k = 5–6 the pooled training-world rates include recall: 0.88 and 0.73 on new pairs against 0.97 and 0.99 on repeated ones.

**F54 — Size extrapolation was not answered; the records locate the structural router's loss at the feature clamp and at growth steps past the training cap, and where neither applies one-step rules transfer.** *P3 falsified as scored (structural matched ratio 0.33–0.58, 0/10; absolute 0.00–0.01, 0/5); the localisation is post hoc and the mechanism was not confirmed by replay.*
- At sizes 6–7 the structural router is below the greedy rung at every k in 10/10 seeds (k = 1: 0.65 against 1.00); a quarter of episodes end in a wrong halt, 95 % of the one-step ones at step 0.
- One-step instances with no node past depth 4 and no growth (n = 199): 0.95 [0.92, 0.97] in every seed on terms of 11–13 tokens never seen. With growth 0.79, with a node past depth 4 0.77, with both 0.35; with the rewritten node at depth ≥ 4, 0.25 and a wrong halt in 0.72.
- At k = 2–4 the in-range, non-growing instances are still below greedy (0.86 / 0.76 / 0.66 against 0.97 / 0.86 / 0.72).
- The absolute control's clause could not have held: every size-6–7 input is longer than any it trained on.

**F55 — Search-found routes composed as well as labelled ones at the median, and two example proofs in place of stated axioms did worse on held-out worlds.** *P5 holds with little power (EI16 0.68 [0.42, 0.77] over five seeds against RT-canon's 0.63; three of five within 0.1; EI16's held-out curves the least stable in the grid). P7 holds (training binding 0.89–0.92; held-out 0.38–0.45 against 0.55; clause 1 can hardly fail).*
- On the needs-the-combination pairs RT-examples is at 0.14 masked and 0.08 unmasked, below the token-blind router (0.60 / 0.15).
- P6: RT-canon commits to the canonical class (first action canonical in 0.90–0.93); RT-post spreads (0.32–0.43) and leans toward its target but is not at the derived mixture, and PF, never trained on that target, is closer to it.

### Derived constraints (arguments checked numerically during design)

**F12 — Under AdamW, decoupled weight decay is itself the time filter.** *Established (derivation plus check).*
- The memory time constant is τ = 1/(lr·wd).
- AdamW cancels most pre-optimizer gradient scaling.
- A linear time-invariant filter of the gradient cannot select between memorization and generalization: it rescales every persistent component by the same factor.
- Evidence: [fastslow/DESIGN.md](../experiments/fastslow/DESIGN.md) §2 and [DESIGN_checks.md](../experiments/fastslow/DESIGN_checks.md).

**The rewrite carrier's binding stratum has structural holes (6 Nov 2026, history Entry 77a).** *Derived constraint (counted on the built dataset, `data/rewrite/meta.json`).*
- Binding (d_full < d_W) is empty in the full world by construction (W is the whole rule set), so P2's second row and P5's held-out stratum are read on the pairs and the triples.
- In {A, C, I} binding lives on terms of size ≤ 2 (134 of 156 eval instances at sizes ≤ 5): the duplication-against-identity growth race (0 → (0+0)+(0+0) is two Db steps outside W and three I steps inside), which the size cap removes at sizes 6–7 (0 / 600). The stratum measures "D would be shorter", and D is only shorter on terms with room to grow.
- The eval binding target (0.50) is infeasible at the 5 % acceptance floor in {A, C, I} at sizes ≤ 5 (0.26) and in five of the eleven multi-axiom worlds at sizes 6–7 (realized 0.37–0.42 in four of them; {A, C, I} has none there); the realized rates travel with every results table (derive §1).

**On the rewrite carrier most held-out "binding" instances need no unseen combination, and the executor's mask gives the world away (9 Nov 2026, history Entry 79; [audit4](../experiments/rewrite/review/audit4_pass_rules.md), recounted in `derive/derived.md` §9).** *Derived constraint (dataset and symbolic policies; no router).*
- 0.66 of the held-out pairs' binding instances (AI 0.85, CI 0.47) and 0.52 of the triples' have a proof within the horizon inside a *training* world contained in the held-out one. Only 209 of 608 and 509 of 1,065 need the unseen combination; a policy that plans only inside training worlds and lets the mask choose scores 0.84 / 0.60 on the full strata and 0.53 / 0.21 on those.
- Under the masked reading a world-blind full-world planner exceeds the derived "cue-blind ceiling" by more than 0.05 in 14 of 14 worlds; unmasked, a token-blind fewest-axioms planner does too, and solves 0.61 of the training worlds' binding stratum against the cue-aware lookahead-3's 0.69: the pair (t, s) itself says much about the world.
- Consequence: a claim about the *cue* is scored on the unmasked reading, a claim about *composing decisions* on the masked one, and the stratum for recombination is the instances that need the unseen combination (DESIGN §17).
- The categorical (absolute) head is trained on 52 (prefix position, rule) slots — all eight rules at positions 0–4, six at 5–6, none at 7–12 — so of the training worlds' 4,800 size-6–7 instances 986 have no route within the horizon on labelled slots, 1,021 only a detour, and 2,793 a shortest proof (the design had said rows 9–12 and counted 318; L41).
- A masked pass of P2 cannot separate planning over an unseen combination from single-world plans glued by greedy moves: that glued symbolic policy scores 0.751 / 0.497 on the 209 / 509 instances, above the pass bar on the pairs. Only the unmasked reading (token-blind policies reach 0.41 there) and the 52 / 256 instances the glued policy fails bear on the difference.
- Sizes 6–7 are easier than size ≤ 5 on matched (world, k, binding) cells: the symbolic rungs' own ratio is 1.06–1.13 (1.12–1.32 on the training pairs), so "≥ 0.9 of the size-5 success" is not a bar for extrapolation without that normaliser.
**Branching in the rewrite carrier, in bits (12 Nov 2026, history Entry 87; [branching_info.md](../experiments/rewrite/analysis/branching_info.md)).** *Exact computation on eval5; no model.*
- Branching entropy (uniform over shortest proofs) is front-loaded: about 1.2 bits per step in the first half of a length-6 proof, 0.10 bits at the last step; 4.96 bits for the whole path. 69 % of instances have several shortest proofs, and only 11 % of the path entropy is the order of independent steps.
- With the world known, a kernel verdict on a tried step carries 0 bits and the exact update is a mask (a proposition). Data of this kind cannot test updating on evidence.
- With the world hidden (prior uniform over consistent worlds, 2.68 bits): one trial yields 0.34 bits about the world; the exact update differs from a mask by TV 0.087 (more than 0.05 on 47 % of instances); belief on optimal first steps 0.774 → 0.817 (mask) → 0.885 (exact).

**The rewrite carrier's eval sets overlap its training stream (10 Nov 2026, history Entry 81; [review6](../experiments/rewrite/review/review6_results.md) finding 1, recounted in `results_data/review6_checks.txt`).** *Counted on the built dataset.*
- The build never excluded training instances from the eval sets. 1,716 of the 4,800 training-world eval5 instances (0.357) are verbatim training (t, s) pairs — 0.55 at k = 1, 0.28 / 0.22 / 0.32 at k = 4 / 5 / 6; 90 % of eval instances with a one-leaf start term, 0.6 % at five leaves. The k = 7–8 set has none.
- 30 % (pairs), 27 % (triples) and 32 % (full world) of held-out eval5 instances are (state, goal) rows RT-canon was trained on inside a contained training world, with only the presence tokens different.
- Consequence: every training-world rate on this dataset is read on the never-trained pairs beside the pooled one, and a held-out rate is not a rate on unseen (state, goal) rows unless it is restricted to them.


---

## Failed predictions and refuted ideas

These are the negative results. They rule things out, and are kept so that nobody rediscovers them.

**X1 — A side "sink" branch quarantines memorization (Study 1, H_U).** *Failed.*
- Memorization formed first in the chain. Once strong, the sink hosted both components. The localization index was never positive, and was significantly negative at s = 4 (−0.059).
- Lesson: static wiring under shared error does not quarantine memorization. This agrees with MemSinks, where localization needed an input-dependent routing signal.

**X2 — The sink hosts most of the rule at large multipliers (Study 1, H_mono criterion 2).** *Failed* at the re-tuned optima: mean share 0.18. Only "larger multiplier → faster" survived, and that one is confounded with step size.

**X3 — A chain groks without weight decay, a parallel net doesn't (Study 1, H_depth).** *Inconclusive:* 0 of 30 runs grokked at wd 0. It was a floor effect, not a test.

**X4 — Short paths construct, long paths memorize (Study 4, H1).** *Nominally supported, substantively failed.*
- The pooled effect came almost entirely from one unstable grid-edge graph. Elsewhere, D = 0.02–0.05.

**X5 — An extra short reader lowers the long blocks' utility ("friction", Study 4, H2).** *Failed.*
- Both contrasts were also structurally flawed: sibling additivity, and an 8× lr difference.
- The only friction-like signal is small and exploratory: `pk0120` vs `pk0121` at a shared lr.

**X6 — Composition delays generalization as a topology effect (Study 4, H3).** *Nominally supported, substantively refuted:* it is the learning-rate assignment (F7).

**X7 — Chain instability is the Adam "slingshot" from weight-norm growth.** *Refuted in that form* (F8).

**X8 — The low-frequency part of the gradient is the generalizing part.** *Refuted by derivation* (Phase 5 design, 22 Oct discussion).
- With full batches, a fixed training set makes the memorizing signal static. It points the same way every step until fitted, so it is "low frequency" too.
- The axis that separates the two is agreement across examples, not frequency in time (F12).
- Minibatch noise is a third, separate component.

**X9 — Per-branch time constants give branches different roles, as measured at the end of training (Phase 5B as designed).** *Measure failed before any test* (F9).
- The preregistered statistic could not move by the preregistered effect size.
- The underlying idea is on hold, not refuted (see H2).

**X10 — Static forward wiring separates memorization from generalization under shared-error training.** *Not supported anywhere it was tested:* Studies 1 and 4, and the pilot.
- This is the program's original bet in its simplest form. Across every wiring tried, topology changed *speed, stability and which path carries everything*, never *what each path carries*.
- Scope: one shallow task (modular addition), AdamW, parent-selection graphs.
- *2026-10-23:* the first possible exception is exploratory: E3's matched-time reanalysis in asym finds generalization spread across both branches while memorization stays in the chain, even at equal speeds (F18, I15). Not preregistered, branch-level only. X10 stands until I15 is tested.

### From rung 4, the main grid (10 Nov 2026, [rewrite/RESULTS.md](../experiments/rewrite/RESULTS.md), reviewed)

**P2 — "RT-canon exceeds the greedy rung by ≥ 0.2 and reaches ≥ 0.7 on the held-out pairs' binding instances that need the unseen combination in ≥ 6/10 seeds" (the rung's core).** *Not met (0/10); not falsified (4/10 above the non-composing floor against ≤ 3/10, by two instances).* See F51.

**X42 — "The structural router extrapolates to sizes 6–7 at ≥ 0.9 of its matched size-5 success; the absolute router fails exactly the unaddressable instances and no more" (P3).** *Falsified as scored on both clauses (0/10; 0/5) — and neither clause tested its claim:* clause 1's eval set mixes the feature clamp, growth past the training cap and position extrapolation; clause 2 conditioned on action slots while every input was out of the absolute router's range. The preregistered reading of the falsifier ("also learned a table") is not supported (F54). A failed prediction and a failed instrument.

**X43 — "Within training worlds RT-canon matches the lookahead-2 rung and not the lookahead-3 rung" (P4).** *Falsified (≥ the d = 3 rung in 10/10 seeds, also on pairs never trained on).* I expected an amortized prior worth two steps of search; the labelled prior is past the top of the ladder I built to measure it (F53).

**X44 — "RT-shuffled is well below RT on the binding stratum" (P8's second clause, masked reading as preregistered).** *Falsified masked (gaps 0.03 / 0.04, two seeds); holds unmasked (0.36 / 0.35).* Under the mask the action set carries the world (F52); the audit had derived that a token-blind policy could do this (X40) and the pilot's sanity table had shown it.

**X45 — "RT-post ends at the derived pooled fixed point" (P6).** *Does not hold under the amended rule (0/10), nor under the auditor's alternative bar (distance 0.12–0.14 against 0.05).* The amended rule was itself defective (L42); the reading that survives is "leans toward the target", which PF does as well.

### From rung 4, before the main grid (9 Nov 2026, [rewrite/DESIGN.md](../experiments/rewrite/DESIGN.md) §17, [audit4](../experiments/rewrite/review/audit4_pass_rules.md))

**X40 — "A router that ignores the presence tokens cannot exceed the cue-blind ceiling" (rung-4 P8, first clause; DESIGN §5).** *Withdrawn before test, by derivation:* the ceiling assumed a planner that follows a uniformly chosen full-world shortest proof over all 105 slots. Symbolic token-blind policies exceed it by more than 0.05 in 14 of 14 worlds — under the mask because the action set supplies the world, unmasked because the fewest-axioms world in which the goal is reachable is usually the right one. P8's second clause and falsifier are kept and scored on each reading (§17).

**X41 — Rung-4 pass rules that could not carry their claims as preregistered (P2's stratum, P3's "≥ 0.9", P7's "within 0.1 of the ceiling", P4's GP1 cells).** *Amended before any main-grid run (§16, §17):* P4's cells had tied lower rungs; P2's stratum was two-thirds solvable without the unseen combination; P3's ratio was passed by a router that lost 15 % at sizes 6–7; P7's bar was a perfect planner's. None is a result about a router; all four are results about how the rules were written (L36, L39). Review 5 added a fifth of the same kind before the launch: P3's second clause counted the categorical head's artefact from the wrong rows (L41).

### From rung 3 (2 Nov 2026, [commit/DESIGN.md](../experiments/commit/DESIGN.md) §7a, before stage 2)

**X33 — "The L start is harder than the E start for every signal: two decisions, the first with no in-support read" (rung-3 PA4).** *Withdrawn before test, at the derive step:* at the seven committed L routers PREV at set-here is already in place on six (p(PREV) 1.00; the seventh, seed 0, has HALT winning there), and the missing SET sits at +5 to +10 logits against the E routers' single relabeled decision at +21 to +32; the beam keeps the oracle's route from every L router (0.58–0.66 of unsolved P-words) and from two of five E routers. Replaced by per-signal counts from the L start and an EI16-only "L is easier" prediction (DESIGN §7a). Withdrawn on derive-step quantities before any phase-2 outcome existed, which is what the open-row rule is for (review 2, Q2).

### From rung 3, stage 2 (3 Nov 2026, [commit/RESULTS.md](../experiments/commit/RESULTS.md) §9)

**X34 — Rung-3 PA1's "PF ≥ 4/5" and the open EX row "≤ 1/5 (blind)", and the per-seed EI16 rows for s3 and s6.** *Refuted:* PF 3/5 (a plateau and a stochastic escape, F47); EX 3/5 + 1 R-like, two of the moves starting from the halting state (F48); EI16 s3 and s6 moved from gate coverage 0.01–0.02 (F46).
**X35 — PA2's "PF ≥ 2/3 at p_P 0.08" and "time to move grows as p_P falls".** *Refuted / not observed:* PF 0/3 at 0.08 and no dose effect between 0.3 and 0.15 (the movers were faster at 0.15); its falsifier ("moves at 0.08 but not at 0.3") met per seed by EI16 s2. Its third clause (a non-mover at 0.3 does not move at any dose) held for IW16 and PF-IS16.
**X36 — The coverage rule's thresholds (0.05 / 0.3) as a graded predictor of movement.** *Not the right form:* for search the checkpoint number decides whether the tie can reverse, not when (F46); PA3's own statistic came out between its bar and its falsifier within EI16 (0.705) and above the bar pooled (0.821).
**X37 — The PA4 replacement's "RT 2/2 from L" and "the L start is easier for EI16".** *Refuted:* RT 1/2 (F49: the L-s1 cell unreadable by the design's own rule); EI16 2/3 from L against 4/5 from E.
**X38 — The coverage reading for the exact own posterior ("EX from L s0–s2 would falsify it"; DESIGN §7a).** *Its falsifier fired:* EX from L s2 moved at 1,500 (s0 confirmed at 1,200, ending at 0.949). Together with F48, the gate's posterior mass is not the quantity that governs EX.
**X39 — PA′1's "RT 2/2" and "EX fills PP in ≥ 2/3 (its posterior is on the completions)".** *Refuted:* RT 1/2 (the dead zone, F49), EX 0/3 at 0.60 with a 0.3 loss on ordinary words (F50).

### From rung 2 (30 Oct 2026, [routefind/RESULTS.md](../experiments/routefind/RESULTS.md))

**X27 — EI16's learned program follows the program its init beam returns (rung-2 P5).** *Refuted 10/10:* imitation lands in L whatever the init beam's MAP route (E on 4 of 5 δ = 1 seeds). Mechanism open; the beam's merge rule is a candidate.

**X28 — The sampled fixed-prior target (PF-IS16) does not degrade at the spurious-density dose because its pooled argmax does not move (rung-2 P6).** *Refuted:* it degrades most (5/5 → 0/3 at criterion), while IW16 and EI16 lose 31–44 % in time. The pooled argmax is not what a K = 16 self-normalized estimator sees when 0.99 of the target's mass is on coincident routes.

**X29 — PF-IS16 reaches exact PF's fixed point (program E on δ = 1).** *Not observed:* exact PF lands in E as derived, PF-IS16 in L-leaning mixtures (F37). The sampled estimator carries a current-policy component.

**X30 — Rung 2's two-program analysis (E and L exhaust the correct programs that matter).** *Incomplete:* program R (one extra SET at the start, decisions one count higher) appeared in 2 of 20 learned runs of the current-policy signals (F38).

**X31 — Revision 3's PF arm as a causal test of F29, and revision 3's "PF fails δ = 0 ⇒ the task is the attractor" reading.** *Withdrawn before running (review 3):* PF's fixed point is derivable and its step-0 gradient equals EX-AL's; the causal test became EX vs EX-AL vs PF / PF-IS16, which then gave F36.

**X32 — AC16 with TD(0) and a zero critic "starts as RL16" (revision 3 §5).** *Refuted on CPU before running:* TD(0) pushes toward halting from step 0 and collapses within 100 steps even after a critic warm-up (AC16T0 0/2 at the grid); TD(0.9) avoids the trap and then collapses like RL16 (F40).

### From rung 1 (28 Oct 2026, [route/RESULTS.md](../experiments/route/RESULTS.md))

**X22 — "New orders never seen in training" (rung-1 revision 1, P3).** *Withdrawn before running:* with the trigger on the pair the P-step leaves from, every order stratum was in-distribution for a Markov router; the reviewer's CPU run passed P1–P3 at 7 % of the budget (L25).

**X23 — The count extrapolates to the unseen value 2, so PP runs pass (rung-1 revision 2, P3 as first written).** *The count does extrapolate (probe (a) 1.00), but PP fails first on the untrained precedence* (F30); restated before the pilot as P3a / P3b after review 2 and the CPU sanity run, then observed as restated.

**X24 — The local trigger is the easy control: RM-d0 at its in-range level, PP local there (rung-1 P3's control line).** *Falsified, 5/5:* RM-d0 never learns the P-rule in 6k steps (F29). The design's outcome row "P1 ∧ RM-d0 failing: a bug or a λ artefact; stop and inspect" fired; the inspection found neither — a basin created by the stray-action phase.

**X25 — The looped block must carry the pending bit across a loop; LF-d0 beats LF by ≥ 0.2 (rung-1 P5 as first written).** *Withdrawn before running* (review 2: the deferred flag is readable in one layer by a value-slot match); LF-d0 = LF observed (F32).

**X26 — P6 (B1: the step cost buys minimal routes), P8's probe (remaining distance is linearly decodable from the router's input).** *Removed before running* as untestable / failing by construction (review 1); B1's weakest form appeared unasked in the pilot (F33).

### From rung 0 (28 Oct 2026, [REV3](../experiments/ptr/REV3.md))

**X18 — "What the halting policy reads" decides extrapolation (rev-3 P1, P2).** *Falsified as preregistered.* LH-Δ (change features) at the gate-chosen head rate: 0.65 at k = 16, 0.15 at 32; LH-s24 (state head, T = 24): 0.96 at 16, 0.44 at 24. In both the stability exit on the arm's own block fails at the same point (F22): the block lost the signal; the head did not misread it. The ×10 head (F24) is the open exception.

**X19 — A learned change head is cheaper than the stability exit (rev-3 P3).** *Falsified at ×1* (k + 2.3 vs k + 1.3; the head commits late, post-answer λ 0.3–0.8). *Exploratory counter-evidence* at ×10 (k + 1.1).

**X20 — The oracle-halted block's time to learn a depth is insensitive to how rare that depth is (rev-3 P6a).** *Refuted in the opposite direction* (F25): at tails ≤ 10⁻¹ the oracle arm never learns k = 2.

**X21 — The state head's failure tracks the loop horizon T (pilot-2 reading, rev-3 P2's LH-s-k4 line).** *Falsified:* with k_max = 4 and T = 10 the failure is at k ≈ 6–7 (≈ 1.5·k_max), not ≈ 15; with T = 24 it is at ≈ 22. The block's trained range, not a fixed horizon.

### From "diagnosing the walk" (E1–E4, 23 Oct 2026)

Preregistered predictions that failed, and ideas the runs killed. Details in [walk/RESULTS.md](../experiments/walk/RESULTS.md).

**X11 — The answer is readable from the current features before the network produces it (E2 P1, readout visibility).** *Failed, as the pilot predicted.*
- A logistic readout refit on frozen features leads the network by a median 0.006 of the time to 0.9 test accuracy, against ≥ 0.2 required. The lead is mostly exactly one 100-step eval.
- Kept here because it was a preregistered prediction. It was expected to fail; its failure is the positive finding F15 (the features, not the readout, are the bottleneck).

**X12 — The run's own drift points at the destination well ahead of time, in both wirings (E2 P1b, pooled).** *Failed pooled* (median relative lead 0.077, +11/−6, p = 0.33).
- Holds in sym alone (10/10 seeds, 15–22% earlier), and not at all in asym. That split is exploratory (F16).
- Extrapolating further (H = 4,000) overshoots: median relative lead −1.4.

**X13 — Co-location persists under any speed ratio in the chain-plus-shortcut graph (E3 P2, asym).** *Falsified as written* (τ = 0.80; shifts up to 2.4 logits).
- Generalization followed the speed ratio while memorization stayed in the chain.
- The exploratory matched-time reanalysis reads this differently: most of the *trend* came from comparing memorization early with generalization late (L14). What remains is a speed-independent gap: generalization spreads, memorization stays in the chain (I15).
- In sym, P2 was inconclusive: significant but small, and dependent on the logit clip.

**X14 — Credit is set by speed, so memorization and generalization always move together (H1, strong form).** *Refuted, on exploratory evidence.*
- At matched times in sym lr-only, speed sent the rule to the faster branch (+6.9 logits, 10/0 seeds) while memorization moved slightly the other way (−0.25, 0/10).
- In asym, the chain holds memorization at every speed ratio.
- The weak form survives (F18): speed is a steep determinant of credit (≈ 2.8 logits per unit log ρ in sym).

**X15 — A depth-aware output parameterization aligns stability regions across wirings (H8).** *Failed* (E4; F17).
- Neither a uniform 1/√L nor a per-block 1/√depth multiplier gives C4 a stable region. Under P1 the C1–C4 gap is ≥ 7 grid steps; under P2, 5.
- Consequence: structure comparisons and any search over structures need per-structure settings, or a stabilizer that changes the model class.

**X16 — A decay-aware extrapolation of the drift predicts where the weights are going.** *Refuted.*
- Proposed by the pre-run review as a fix to the linear operator. On real runs it predicted nothing (R² ≈ 0 in the operator pilot; median lead 0.0 in E2). It had been checked only on a weight with no gradient (L11).
- The plain linear extrapolation, capped at H ≤ 1,000, is what showed the sym lead.

**X17 — Wider networks generalize from less data (priors P4, prediction from Huang et al. 2024).** *Failed* in our MLP setting, under every variant tried.
- The ungrokking threshold did not decrease with width, and the scratch floor did not move with width at all (F20).
- Our 24 Oct conjecture, the opposite direction (narrower generalizes from less data), isn't supported either. Narrow was lowest or tied on the ungrokking side only, and borderline.

---

## Open hypotheses

Each has a prediction and a falsifier. Most are arguments from the 22 Oct discussion and have not been tested.

**H1 — Credit by speed.** *Argument (first-order kernel reasoning); consistent with F4 and F6.*
- **Claim:** under one shared error, component k's share of fitting a function-space mode m is η_k λ_km / Σ_j η_j λ_jm, i.e. learning rate × kernel eigenvalue along that mode.
- **Predicts:**
  - co-location (F4);
  - learning-rate migration (F6);
  - "fast memorizer plus slow generalizer" being self-defeating (H2).
- **Falsifier:** measured shares are not ordered by η·λ; a slower component wins a mode it is slower on.
- **Cheapest test:** during training, measure each branch's credit share along the rule direction and along per-example directions, across learning rates and wirings. Compare with the formula on Study 4 graphs.
- *2026-10-23:* **E3 results → partly supported, partly falsified** (F18; [RESULTS](../experiments/walk/RESULTS.md)).
  - Speed moves credit steeply, and in the stable wiring it moves both kinds of function together when measured as preregistered.
  - But at matched times speed alone splits roles transiently, and in asym the wiring sets a speed-independent split.
  - H1's strong form ("credit is set by speed, so memorization and generalization always move together") is **refuted**.
  - What survives: speed is a strong, steep determinant of credit, and kernel-level effective speed is untested.
  - The refuted strong form is recorded as X14.
- *2026-10-22:* argument → **preregistered as E3** ([walk/PREREGISTRATION.md](../experiments/walk/PREREGISTRATION.md)): direct manipulation of the branch speed ratio ρ at fixed lr·wd.
  - Revised after review: lr-only and wd-only controls separate speed from weight scale.
  - P2: co-location persists (trend test in logit space).
  - P3: step size alone moves credit.

**H2 — Under shared error, a fast memorizer takes every mode, the rule included.** *Argument (corollary of H1).*
- **Claim:** complementary learning systems work in brains because the cortex learns from replay, not from an error it shares with the hippocampus.
- **Falsifier:** a fast flat-kernel component and a slow component, under shared error, end up dividing rule and exceptions.
- **Cheapest test:** a planted two-branch case with known kernels.
- *2026-10-23:* **not directly tested; weakened.** E3's branches had the same kernel, so it is not H2's test. As preregistered, the faster branch took both kinds of function (consistent with H2). But at matched times, speed alone sent the rule to the faster branch first while memorization stayed or moved the other way (X14). So "the fast component takes every mode" does not hold at every moment, even with identical kernels. The flat-kernel test is still open.

**H3 — Masking.** *Argument; partial support from pilot plateau dynamics.*
- **Claim:** once the memorizing component fits the training set, it removes the error the generalizer needs. Weight decay then drives the plateau.
- **Open sub-question:** is masking harmful, or the mechanism by which grokking happens at all?
- **Falsifier:** removing masking (e.g. the unmasked arm) leaves the generalizer's trajectory unchanged.
- *2026-10-24 (interpretation, from discussion):* summed branch logits are a product of experts, and shared error is the correct credit for that kind of conjunctive composition. Masking is then "explaining away": once one explanation (memorization) accounts for the data, the others get no signal. Bayesian explaining-away ranks explanations by prior × likelihood; gradient descent ranks them by which arrives first, and weight decay slowly corrects towards the prior. On this reading, grokking is explaining-away arriving late, and masking is a symptom of a prior that undercharges lookup tables (I17) as much as a credit problem.
- *2026-10-23:* **still untested directly** (the unmasked arm was never run). Consistent observations, none decisive:
  - at the peak of memorization, total generalization is at or below zero in most E3 runs: memorization pushes unseen inputs toward wrong answers;
  - E1 measures that as a cost, 12–29 nats per new pair (F14);
  - E2 shows the plateau is spent building features, not fitting a readout (F15).
  - None of these separates "memorization removes the generalizer's error" from "the generalizer is slow anyway".

- *2026-10-25:* **F19 supports the "prior" reading of masking.** Below about 870 pairs, the equilibrium itself prefers memorizing (a learned rule ungroks), so masking there is not a failure of credit or search. Explaining-away, as read in the 24 Oct note, follows the prior. The open question shifts from "how to stop memorization masking the rule" to "what prior would make the rule cheaper than a lookup table at small data".

**H4 — The grokking delay is the price of overcompleteness.** *Prior art (partly answered).*
- **Claim:** a hypothesis that can express the rule but cannot memorize finds it without a plateau. Masking needs two ingredients: shared error, and a hypothesis that offers the shortcut.
- **Prior art:** [Omnigrok](https://arxiv.org/abs/2210.01117) (Liu, Michaud & Tegmark) eliminated grokking on algorithmic data by constraining weight norm. The lazy-to-rich account (Kumar et al.) is related.
- **Open part:** whether *structural* constraint (wiring, capacity) does the same; and whether constrained hypotheses get stuck instead, i.e. how much of overparameterization's value is easy optimization.

**H5 — The answer is visible long before it is reached.** *Open; the researcher's "minima on the horizon".*
- **Claim:** the final generalizing solution is predictable early in training from signals the learner already has: cross-example gradient agreement, internal representations, the direction of drift.
- **Falsifier:** prediction quality stays near chance until close to grokking. That would mean the walk is doing irreducible work.
- **Why it matters:** it decides whether "seeing instead of walking" is possible from information a learner already holds. It is the most decision-relevant open question.
- *2026-10-22:* **partly prior art.** [Two Speeds of Learning](https://arxiv.org/abs/2605.27078) finds that representations improve before the test-accuracy jump, using linear probes on frozen features. [Canalization Before Generalization](https://arxiv.org/abs/2608.25813) detects the coming generalization about 4,000 epochs early, from weight-decay perturbation responses, while static progress measures lag.
- *2026-10-22:* **preregistered as E2**, adding what those papers lack:
  - the timing of a lead, from cheap static operators (logistic readout refit, drift extrapolation);
  - an intervention: jumping ahead along the drift.
  - Instrument pilot (p = 23, 2–3 seeds; not evidence): the refit readout tracks the network with no lead, which points to the features rather than the readout as the bottleneck. Drift jumps saved 100–400 steps in sym and nothing in asym.
- *2026-10-23:* **E2 results** ([RESULTS](../experiments/walk/RESULTS.md)):
  - **P1 failed, as expected:** the readout version of H5 is refuted (F15).
  - **P1b failed** pooled as preregistered, but holds in sym alone (10/10 seeds) and not in asym (F16).
  - **P2 supported** in sym, modestly.
  - H5 status: argument → **suggestive, wiring-dependent.** The horizon is partly visible in the stable network, but only a fraction of the walk can be skipped by extrapolation. The large gap is in data (F13), not in steps.

**H6 — Agreement across halves separates the signals without labels.** *Argument (derivation).*
- **Claim:** for disjoint halves A and B of the training set, E⟨G_A, G_B⟩ measures the shared (population) part of the gradient, and ‖G_A‖² − ⟨G_A, G_B⟩ measures the memorizing part.
- **Falsifier:** the estimate disagrees with the test-label-based gen/mem gradient split.
- **Cheapest test:** compare it against the existing 5A measurements.
- *2026-10-23:* untested; E1–E4 did not touch it.

**H7 — Chain instability comes from compounding Jacobians.** *Argument (Phase 0 Prop. 3); open since Study 2.*
- **Claim:** sharpness, or the effective step on the end-to-end map, grows with chain length C.
- **Falsifier:** sharpness does not scale with C at matched training progress.
- **Cheapest test:** track the top Hessian eigenvalue on a subset of Study 2 graphs.

**H8 — A depth-aware parameterization aligns stability regions across structures.** *Argument (μP literature).*
- **Claim:** with μP-style depth scaling, the optimal lr is shared across structures, so any remaining instability differences are structural.
- **Falsifier:** optimal lrs still differ by structure under that parameterization.
- **Cheapest test:** a learning-rate sweep per structure, with and without the scaling.
- *2026-10-23:* **E4 → NOT supported** (F17). Neither uniform 1/√L nor per-block 1/√depth output scaling aligns stability across wirings: the deepest chain has no stable region either way. Structure comparisons and structure search need per-structure settings, or a stabilizer that changes the model class.
  Recorded as X15.
- *2026-10-22:* **preregistered as E4.** Stability regions for C = 1–4 under std, uniform 1/√L and per-block 1/√depth output multipliers, on an lr × wd grid. Depth-μP's known result is transfer across depth; whether it transfers across wirings at fixed depth is the open part.

**H9 — Competing hypotheses scored by prequential surprise avoid masking.** *Argument.*
- **Claim:** if hypotheses are not summed or trained jointly, each predicts alone and is scored by its surprise on each example *before* learning from it. Then memorization earns nothing at the level of which structures survive. Masking inside a branch disappears if the branch cannot express the shortcut, and true exceptions go to a separate memory store.
- **Falsifier:** prequential selection still favors overcomplete or memorizing structures; or constrained structures cannot be optimized.
- *2026-10-23:* **the criterion has teeth (E1).** Scored prequentially, a memorizing gradient learner costs 12–29 nats per new pair against 4.1 for a uniform guess, and 4.5–7× uniform over a stream (F14). A selection by prequential score would reject it long before it produces the rule. Whether such selection finds better structures, rather than merely rejecting bad ones, is untested. E4 adds a constraint: structures differ in their stable settings (X15), so a selection must not score hyperparameter luck.

**H10 — Schema assimilation separates a real learner from gradient descent.** *Argument, with an animal analogue (Tse et al. 2007).*
- **Claim:** once a rule is learned, new consistent examples should be absorbed almost immediately and generalize, while exceptions should be stored without disturbing the rule. Standard gradient training will walk for both and may overwrite old examples.
- **Falsifier:** standard training already assimilates consistent examples in one or two steps without interference.

---

**H11 — Joint training with a halting policy degrades the shared block through the policy's objective, and the degradation goes to whichever part can move.** *Argument from rev-3 data (F22–F24); untested.*
- **Claim:** the PonderNet KL-to-prior term (and more generally any pressure from the halting loss) reaches the block through the head's input when the head's own parameters move slowly; it pushes intermediate states to look "settled"; inside the trained range the cross-entropy holds the block to the answer, outside it the pressure wins and iteration fails. A head that moves fast absorbs the pressure itself.
- **Predicts:** LH-Δ ×10 at 10 seeds extrapolates (≥ 8/10 ≥ 0.9 at k = 32) with no decay; LH-Δ ×1 with its input detached from the block (stop-gradient) extrapolates; β = 0 at ×1 extrapolates; LH-s ×10 extrapolates too (the input does not matter when the head absorbs the pressure).
- **Falsifiers:** stop-gradient still decays (the damage is not through the head's input); β = 0 still decays (not the prior); ×10 at 10 seeds < 6/10 at k = 32.
- **Cheapest test:** D93, ≈ 4 T4-hours.

**H12 — Plasticity after commitment is set by how the targets are made.** *Argument from F36 and EI16B; rung 3 core A.*
- **Claim:** a router committed to one of two tied programs moves to the other when the data start to require it in the order exact fixed-prior target > search under its own policy > importance-sampled fixed prior ≈ own posterior, and search's plasticity is predicted by its route coverage at the committed policy.
- **Falsifiers:** every signal moves when 16 % of queries relabel the decision (then commitment is not sticky and F36 was a from-scratch artefact); only the exact target moves (then plasticity is about enumeration, not priors).
- *1 Nov, design review 1 of rung 3:* the relabeling shift is the local trigger from a committed deferred-trigger router (one in-support decision, HOP → PREV, relabeled by every P-word; dose = p_P). PP words do not break the E/L tie (both programs complete them at the oracle's length), so the earlier "PP words favour E" is withdrawn; PP words test filling in, not overturning.
- *2 Nov (Entry 72, rung 3 stage 1 and the derive step):* 7/7 EI16 routers committed to L and 5/7 EX routers to E (EX seed 0 learned E and then **collapsed spontaneously** at step ≈ 2,000 into halting at flagged pairs under its own signal at the constant lr — so an EX run that ends at zero under the shift is read as *collapsed*, not as locked, and EX's spontaneous-collapse rate is measured on all five stability controls). At the committed E routers the shift relabels exactly one product-state class (HOP → PREV at +21 to +32 logits), the own-posterior target has 0.00 mass on the new route, 16 samples never contain it, and the ε-mixed beam keeps it on 0.73 / 0.47 of P-words for two routers and ≈ 0 for three (the merge tie breaks per seed, not by ε). The open rows are set from these per seed (DESIGN §7a); stage 2 is 120 runs.
- *3 Nov (Entry 73, rung 3 stage 2; F45–F49):* **supported in its structure, corrected in its order and its predictor.** How the targets are made decides plasticity, but the line is exact-or-searched against sampled-from-the-own-policy: RT 2/2 ≥ EI16 4/5 ≥ EX 3/5 ≈ PF 3/5 ≫ PF-IS16 1/5 ≈ IW16 0/5 from E; the claimed "exact fixed prior > search" is not seen, and the own posterior is plastic when exact (by way of the halting state, F48). Search's coverage predicts whether the merge tie can reverse, not when (F46); PA3's AUROC 0.705 within EI16, 0.821 pooled. Neither falsifier met. Status: *supported (narrowed)*.

**H13 — Disagreement among sampled routes estimates the value of computation.** *Argument; rung 3 check C, a carrier question later.*
- **Claim:** per query, disagreement among k sampled routes predicts the committed route's error better than the prediction's entropy (which is constant in the toy) and approaches the exact marginal's predictive power as k grows; the router's policy entropy concentrates at the setup and payoff decisions, not at HOPs.
- **Falsifier:** disagreement is spread uniformly by SET padding and carries no information about which queries the committed route gets wrong.

**H14 — Reward signals for the router fail by exploration before credit.** *Supported by F40 as the failure; the fix is untested; rung 3 core B.*
- **Claim:** with an entropy term or forced breadth, RL16 learns the deferred program; then and only then does AC16 vs ACMC16 vs RL16 (bootstrapped vs Monte-Carlo value) become readable.
- **Falsifier:** the exploration-fixed reward arms still fail at every dose (then value signals have a credit problem as well as an exploration one).
- *1 Nov (Entry 71, CPU, one seed):* forced breadth with self-normalized weights under the router's own policy collapses too (effective sample size 0.004 of the beam's completions by step 200; all-H). The entropy form is derived to fail (rung 3 §5.2). Both run as checks in rung 3. Reposed: a reward signal that keeps exploring weights unsampled routes by something other than its own probability of them, and every such weighting is a posterior target; the value question needs per-step credit inside a posterior target.
- *2 Nov (Entry 72, rung 3 stage 1, F43):* both forms failed at the grid (0/2 each; effective sample size 0.01; the entropy arm keeps 7–13 routes and learns nothing), as predicted. The claim's first clause is refuted for these two fixes; the hypothesis stands in its reposed form (per-step credit inside a posterior target, I33), untested.

## Ideas (not yet hypotheses)

| ID | idea | origin | status |
|---|---|---|---|
| I1 | The backward graph as a design variable: one-way coupling, branch dropout, cross-fitting, stop-gradients | 22 Oct discussion | **Deprioritized.** The researcher: patching the backward graph ignores the observation that the structure of learning is flawed. Branch dropout is the least ad hoc variant. |
| I2 | A population of structures (e.g. wirings) in VRAM, refined by gradients, selected by prequential surprise (sequential Monte Carlo over models) | researcher, 22 Oct | Open. The fleet trainer can already hold the population. Needs H8 and stability regions first, or selection rewards whatever fits the hyperparameters. |
| I3 | A two-system learner: a fast memory store, plus a slow learner trained from replay, with no shared error | complementary learning systems discussion | Open. Matters only for streams, not fixed datasets. |
| I4 | Amortized / in-context learner across a task family (moduli, group operations): learning moved into inference | 22 Oct discussion | Open. "Gradients at the meta level, reasoning at the task level." |
| I5 | A rule-plus-exceptions task, so memorization persists and composition stays measurable | pilot analysis | Open. Implemented as `modadd_exc`; the exception fraction is uncalibrated. |
| I6 | Sparse parity as a testbed with a proven separation between averaging and reasoning learners (statistical-query lower bounds) | 22 Oct discussion | Open. |
| I7 | Group-product composition task (abelian control vs S4), gated on all-short networks failing | Phase 5 discussion | Open. |
| I8 | Topology as a hypothesis about how the data is generated: a learner discovers its own wiring from per-example surprise | 22 Oct discussion | Open. Crosses the repo's "no architecture search" rule; that boundary is the researcher's to redraw. |
| I9 | Phase 5 B–D as designed: per-branch time constants, update routing, consolidation, within one shared loss | Phase 5 design | **On hold** (X9, H1–H2). Stage-1 grids deleted unlaunched. |
| I10 | Measure how much information a training run has versus what it uses: steps or examples needed to determine the function, versus to produce it | 22 Oct discussion | **Measured in E1** (22 Oct design; results 23 Oct, F13–F14: 3 examples vs about 1,044; worse than uniform for most of a stream). Prequential code length of gradient descent on streams vs a program-prior learner (k* = 3 examples), plus the data floor on fixed datasets. It became a measurement, not a hypothesis test, after the review showed the original primary prediction could not fail. Nearest prior art: weight-compression complexity of grokking (no ideal-learner reference); Power et al. on training fraction. |
| I11 | Identity-path credit as a backward-graph experiment: each block learns from the error at the top of the residual stream only, dropping the terms that pass back through later blocks. If co-location (F4), migration (F6) and credit by speed (H1) are unchanged, the wiring acts on learning almost entirely through what blocks read (the forward graph), not through how error reaches them. | Q Labs "Noise" (Oct 2026): a community record uses "identity-path credit" and trains well | Open. Parameter-free; keeps one loss. |
| I12 | Test the claim that backprop "explores the loss landscape poorly", and that this is why networks need so much data, on our toy, where data efficiency is exactly measurable: perturbation-based credit (Dust-style node perturbation, per-example noise) vs backprop, compared on the E1 scoreboard (data floor, prequential code length) and the grokking delay. | [Noise](https://qlabs.sh/noise), [Dust](https://qlabs.sh/research/dust/) | Open. Prediction from the statistical-query argument: perturbation credit is still an averaging learner, so any gain is a constant factor or regularization, not reasoning. It should not beat gradient descent on sparse parity. |
| I13 | Matched-norm random-direction jump control for E2: does jumping along the drift help more than an equally large kick in a random direction? Needed before "the drift direction helps" can be claimed, especially for the chaotic chain network. | E2 verification, 23 Oct | Open. Cheap. *2026-10-24:* **preregistered as Q2**, with a radial-matched null (the drift is largely a per-tensor rescaling, found by the pre-run review) plus random and rescale-only nulls. *2026-10-25:* **done** (priors Q2, P5 supported, with qualifications; see the F16 note). |
| I14 | Why is the chain network's trajectory not extrapolatable when the parallel one is? Curvature of the path, or collapse cycles (F2)? | E2, 23 Oct | Open. |
| I15 | The wiring-set division of labor in asym (E3, exploratory): generalization spreads across both branches while memorization stays in the chain, even at equal speeds. Preregister it at matched times, block-level, with a kernel measure of each branch's effective speed. | E3 matched-time reanalysis, 23 Oct | Open. The first positive lead on the original question since Study 1, and still exploratory. |
| I16 | The higher-decay branch carries the grokking jump, then hands it back (E3, exploratory). Is the circuit formed in the strongly regularized branch and then copied? | E3 recomputation, 23 Oct | Open. |
| I17 | Decompose the data gap into prior vs search. At each training-set size, compare what the regularizer charges (weight norm) for the rule solution and for the memorizing solution. Rule already cheaper below the data floor but not found → search failure. Memorizing genuinely cheaper there → prior failure. | 24 Oct discussion | Open. Decides between "fix the prior inside gradient learning" and "change the learning mechanism". Check the grokking-at-low-training-fraction literature first (L2). *2026-10-24:* **preregistered as Q1** of the priors round ([priors/PREREGISTRATION.md](../experiments/priors/PREREGISTRATION.md)), via the ungrokking protocol of Varma et al. 2023 (prior art: S ≈ 2 in a transformer): the threshold at which a learned rule survives on a subset vs the from-scratch threshold on the same pairs. *2026-10-25:* **answered (F19):** S ≈ 1 (matched criteria, 1.0–1.5); the gap is the prior. |
| I18 | The member-level claim: a restricted member (e.g. one whose primitive is a rotation by a learned frequency), chosen from a population by predict-then-learn scoring, learns modular addition from a handful of examples where the general network needs ≈ 1,044. | 24 Oct discussion | Open. One of the two claims the library/population program rests on. *2026-10-24:* its generic version (does width restriction lower the thresholds?) is preregistered as Q1 P4/P4b; prior art (Huang et al. 2024) predicts the opposite direction to our conjecture. *2026-10-25:* **generic version answered: no** (F20). Efficient members would need structure matched to the task, not less capacity. |
| I19 | The transfer claim: blocks grown or selected on one family of tasks make learning on held-out tasks efficient. Without it, a library is a lookup table of past solutions (memorization one level up). | 24 Oct discussion | Open. The other load-bearing claim. The choice of task family *is* the world prior and must be explicit. |
| I20 | Mixture of competing members with responsibility assigned by out-of-sample prediction, not current fit, plus a one-shot memory member that predicts uniformly on anything unseen. Prediction: the rule member claims rule examples once it predicts them, exceptions go to memory (the rule member is confidently wrong, worse than uniform: F14), and no member wins by memorizing. | 24 Oct discussion; Jacobs et al. 1991 | Open (argument). In-sample responsibility produces fragmentation (one expert per case) and starvation (masking by responsibility). *2026-10-24 revision:* only needed when members are trained jointly **and** the data is a mixture of regimes (e.g. rule plus exceptions). With one rule, independent members weighted by predictive evidence need no coupling at all. *2026-10-25:* **deprioritized by F19.** Credit between members is a search mechanism, and search is not where the gap is. |
| I21 | Growing a network by held-out residual fitting: for each (block, link, site), insert it without changing the function, fit it on half A of the error signal (several inits in parallel), score the code-length reduction on half B minus the cost of describing the addition. Prediction: held-out scoring adds a rotation-like block before any lookup-like one; training-set scoring does the reverse, and growth stalls once the training set is memorized. | 24 Oct discussion; Cascade-Correlation, GradMax, boosting | Open (argument). Greedy growth is itself an averaging procedure and fails on parity; library blocks set how far one growth step can jump. *2026-10-25:* **deprioritized by F19** as a route to data efficiency; still of interest as a way to search over priors (structures), which is where F19 points. |
| I22 | Library size has an MDL optimum (library cost + solution cost), not diminishing-but-positive returns; an oversized library memorizes the task suite. Measure held-out-task code length vs library size. | 24 Oct discussion | Open (argument). |
| I23 | The amortized rival: a network meta-trained by gradient descent across a task family to learn in context, compared with an explicit population of restricted members on held-out tasks (examples needed, code length, extrapolation outside the family). It uses exactly our tooling and is the approach validated at scale, so the explicit-population program must beat or complement it. | 24 Oct discussion | Open. |
| I24 | Express structural diversity as masks or primitives inside one shared tensor shape, so a population of structures batches as dense matmuls (the fleet trainer already batches same-shape independent runs). Heterogeneous topologies break batching. | 24 Oct discussion | Open (design constraint from tooling). |
| I25 | A generator of worlds (PFN-style) for a library of uniform, composable blocks with a separate memory store: random symbol bijection per world, length-weighted programs over modular arithmetic and group operations, an exact Bayes reference from `program_prior.py`, a sweep of world diversity, splits on held-out compositions and lengths, and rule-plus-exception worlds. Claims C1–C5 and what each needs: [generator_design.md](generator_design.md). | 26 Oct discussion; literature review of generator design | Open. Not yet designed as an experiment; decisions in §5 of the note are the researcher's. *2026-10-26 decisions (researcher):* hand-chosen primitives with exact solutions; chunking allowed and measured; no memory store yet; monolithic PFN vs routed library of uniform blocks; instrumentation is a primary deliverable (generator_design.md §6). *2026-10-26:* consolidated into [worlds/DESIGN.md](../experiments/worlds/DESIGN.md): 3 arms (monolithic PFN, open-loop and execution-conditioned composers) × 9 training-mix cells × 3 seeds; mod-7 two-register instruction set (297,845 functions at ℓ ≤ 6); stacked-run batching implemented and tested; GPU throughput test pending. |
| I26 | Exact step classification on the toy, plus the commit-vs-average framing. From any partial program on Z_7, the remaining distance to the target (or to the version space consistent with the context) is computable by enumeration. So every step the meta-model takes has an exact regret, 1 + d(s') − d(s): 0 = optimal, ≥1 = suboptimal, budget-fatal when it exceeds the cap, and dead when the state merges two inputs the target separates (irreversible at any length). The Bayes-optimal step policy (posterior mass of minimal programs extending each prefix) is also exact. A static PFN averages worlds in output space; the dynamic meta-model averages committed, self-consistent worlds in path space, and the two have the same optimum under log loss. The differences are in errors (a wrong valid world vs an invalid blend), joint coherence across queries, verifiability against the context, and where novelty is extrapolated. Commitment is also irreversible: most of the branch space is dead. | 26 Oct discussion; `experiments/worlds/branch_classify.py` | Open. *Measured 26 Oct* (25 targets of minimal length 4 × 32 first moves): optimal 5.9%, lateral 24.9%, away or beyond 69.3%, of which provably dead 42.2%. To be folded into the redesign as diagnostic, exact step-level reference, and an optional privileged-signal arm. *26 Oct, revision 2 review:* the step-level references need a tie-break prior over equivalent programs (π* is not unique), and branch classes need a precedence order (history Entry 32). |
| I27 | Rung 0 of the dynamic-depth ladder (history Entries 45–54): pointer chasing with hidden, heavy-tailed chain length; a shared block with learned halting against static stacks and an oracle; headline = halting extrapolation beyond the training depth with the oracle separating block from policy; plus the signal-pooling test. Design, review and code: [ptr/DESIGN.md](../experiments/ptr/DESIGN.md). | 27 Oct; review 1 recorded | Open. Implemented with run-stacking. *Pilot 1 (27 Oct):* G2 passes (oracle-halted block extrapolates to k = 32; injection essential); the fixed loop overthinks beyond its horizon; LH-s collapsed from a halting-init asymmetry (fixed, L23); static arms slow. *Pilot 2:* G4 passes in a narrow region (lr 3e-4, weak prior); the state-reading halting policy extrapolates only ≈ 1.5× its horizon (its reading of the state drifts with depth) while a change-based exit reaches 0.97 at 4×; unique-layer depth learns hops one at a time (k = 3 at ~10k steps vs all eight by 4k for the shared block). *Revision 3 (28 Oct, Entry 58):* the question is now what the halting policy reads (LH-Δ on standardized change features; LH-s24 with T = 24; LR on every arm's own block); reviewed twice and revised; a gate (pilot 3) then a 93-run confirmatory grid, preregistered P1–P7. *Ran 28 Oct (Entry 59, F22–F27, X18–X21, H11):* P1–P4 falsified as preregistered, P5 holds, P6a refuted the other way; the limit is the block trained jointly with the policy, not the head, and it is lost late in training; a ×10 head learning rate (2 seeds, exploratory) removes it entirely. D93 deferred to rung 3 (D95, Entry 60); rung 0 closed. |
| I28 | Rung 1 of the ladder (history Entry 61): symbolic generic circuits (HOP, PREV) with a learned Markov router over an explicit product state (symbol × pending × set-here), trained by the exact route marginal from next-token loss; the P-trigger is **deferred** (flag one state earlier: a SET setup step, a bit carried across a hop; PP runs reach a pending count never seen in training); arms: RM, the symbol-only null (F4), the local-trigger control (δ = 0), teacher forcing, the state-blind index table, the looped block (read at the oracle depth) and static stacks on the same tokens. Preregistered P1–P9; design, reviews and code: [route/DESIGN.md](../experiments/route/DESIGN.md), [route/review/](../experiments/route/review/). | 28 Oct; review 1 recorded | Open. Revision 1 (local trigger) found by review to be in-distribution for a Markov router (the reviewer's 1,800-step CPU run passed P1–P3); revision 2 implemented and tested (222 tests); review 2 applied (counterfactual probes for precedence and count; P5 relabelled; the symbol-only router's marginal trick). *Pilot (28 Oct, Entry 62, [PILOT.md](../experiments/route/PILOT.md)):* rates unchanged; at 2 seeds RM is exact in range, at length and on alternation, SET flips 100 steps after teacher forcing, PP fails on the untrained precedence with the count read ordinally, the symbol-only router is 0 committed / 1.00 marginal (p(target) 0.75) on P-words, the looped block learns hops and nothing of the P-rule; λ selects between the carried-bit route and an always-SET locally-readable program. *Main grid (28 Oct, Entry 63, [RESULTS.md](../experiments/route/RESULTS.md); F28–F34, X22–X26):* P1, P2, P3a, P3b, P4, P5, P7, P8, P9 all hold at full seed counts; the local-trigger control fails 5/5 and the diagnosis (setup credit arrives through the state) is F29. **Closed** (D102); carry-over to rung 2 in Entry 63. |
| I29 | Rung 2 (history Entry 64): the rung-1 router trained without the exact route marginal — IW (sampled bound), ELL (K = 1 bound), RL (REINFORCE on p(y \| r)), SI (sample-then-imitate), EI (beam expert iteration), EI16B (EI started inside the exact marginal's δ = 0 basin); success = a non-exact procedure reaches the rule on the deferred trigger within 4× EX's episodes; headline that can fail = EI16B. Design, reviews, code: [routefind/DESIGN.md](../experiments/routefind/DESIGN.md). | 29 Oct; review 1 (design + code) recorded | Open. Revision 2 implemented and tested; CPU sanity: EI learns δ = 0 from scratch, IW collapses on δ = 1 at one seed; pilot (13 runs) and main (55) grids ready. 29 Oct (Entry 66): re-scoped to the router's training signal (D106 decided); revision 4 after reviews 3: references EX / EX-AL / PF, candidates IW16 / PF-IS16 / TB16 / RL16 / ACMC16 / AC16 (TD(0.9)) / EI16, answer-pool dose; CPU sanity: every candidate but EI16 in all-H at 7.5 % budget, AC16 TD(0) collapses into halting, PF leaves δ = 0's all-H while EX and EX-AL do not; pilot (26) and main (91) grids ready pending review 4. 30 Oct (Entry 68): pilot ran (27 runs, seed 100): IW16, PF-IS16 and EI16 learn δ = 1, PF-IS16 and TB16 learn δ = 0 (where EX stalls), the reward arms collapse into a deterministic policy within 400 steps (P4 not readable; D108 demotes them to checks), EX-AL δ = 0 reaches a correct count-shifted program one step longer than the oracle; lr 1e-3 everywhere; main grid 72 runs ready. **Closed 30 Oct (Entry 69):** main grid ran (93 runs, the full preregistered grid); success — IW16 4/5, PF-IS16 5/5, EI16 5/5 within 1.6× EX; F35–F41, X27–X32. Risk 3 retired for the toy; risk 4 blocked on exploration (F40). |
| I30 | Branch disagreement as the estimator of epistemic uncertainty (Entry 70): sample or beam k routes at forks the router's policy entropy marks; agreement ⇒ commit and halt, disagreement ⇒ this token is worth more computation. Needs no learned value head; in the toy it is a Monte Carlo estimate of the exact marginal, so it is a check there and a candidate controller for a carrier with real epistemic uncertainty. | 1 Nov 2026 |
| I31 | Routing on the model's own estimate (Entry 70): close the loop the static graph lacks by giving the router the current prediction (logit-lens state) and its own policy entropy as inputs, so that computation depends on uncertainty. The toy router never had either. | 1 Nov 2026 |
| I32 | Library growth by abstraction (Entry 70): detect reused subcompositions in the router's programs and promote them to primitives (DreamCoder's sleep phase), trading depth for width; the mechanism behind "continual memory" of programs. No toy yet; needs a composition space with reusable subroutines. | 1 Nov 2026 |
| I36 | **Latent or exact lookahead inside a decision** (Entry 75; held open by the researcher). Three readings of the thesis's per-token search: latent-only rollouts (the model's own value-equivalent simulation; beats CoT on cost, drifts with depth, must learn the axioms' semantics), executor-stepped rollouts (tree search with a learned prior; exact, no architectural claim, generalizes to new worlds through the executor), or split (executor for breadth, latent for depth; the likely deployable form). Decided by rung 5's latent-against-exact lookahead curve, not by assumption. "Generalize across worlds" means prior-only under the second reading and prior-plus-dynamics under the others. | 6 Nov 2026 |
| I34 | PF's plateau (F47): derive the exact fixed-prior target's per-state marginal at (own flag, 1, 1) under δ = 0 with the committed policy's route visitation, and what makes p(PREV) ≈ 0.15 (loss 2.4) metastable against the escaped state (p(PREV) ≈ 0.65, loss 2.2) — the λ term, the readout floor, or the shared trunk; then the escape rate against the dose. Derive step on the saved checkpoints; no training. | 3 Nov 2026 |
| I35 | The L start as a representation problem (F49, review 3): probe the saved L checkpoints' logits at the (own flag, 0, 0) states against unflagged states of the same symbol class; if they do not separate, the L router's missing SET is a feature to be formed, not a decision to be flipped, which predicts sigmoids under every signal and explains L s1. Also the λ = 0 EX arm for F48's reading. Derive step first. | 3 Nov 2026 |
| I33 | Exploration-fixed reward signals for the router (Entry 69–70): an entropy bonus, or routes supplied by an ε-mixed beam with importance weights, so that RL16 / AC16 keep sampling distinct routes; the prerequisite for reading contextual value (P4). Rung 3 core B. *1 Nov (Entry 71): both forms fail on CPU or by derivation; checks in rung 3. Next form: a bootstrapped value as per-step credit inside a posterior target (PFB16 / PF-IS16 with TD-weighted steps instead of whole-route weights).* *3 Nov: PFB16 is now also the candidate trainer of committed routers (F45 puts the own-policy samplers out); its plasticity is the first check of the next rung that needs a scalable signal.* | 1 Nov 2026 |

---

## Lessons about our process

| ID | what went wrong | rule adopted |
|---|---|---|
| L1 | Phase 5's headline measure could not move by the target effect size; found only by the pilot, after a day of instrumentation | Before building or running a study, plant the effect in a synthetic case and confirm the measure detects it. |
| L2 | A "constrained structure without plateau" experiment was proposed as new; Omnigrok had largely answered it, and our own literature review cites it | Check the literature docs and this ledger before proposing an experiment. |
| L3 | Each confound became the next study (Study 3: five calibration rounds) | When a confound appears, ask whether it changes the question before building around it. |
| L4 | Answers built into contrasts (Study 4 H2), a leaky rule measure (Study 1), an lr rule picking unstable settings | An independent pre-run review for every study (it caught these); measures checked on planted cases. |
| L5 | Learning-rate rules defined by first crossing a threshold rewarded collapsing regimes (Study 4, pilot) | Selection rules must include stability, e.g. a post-fit minimum or a collapse count. |
| L6 | End-of-run readings discarded the dynamics where the effect lives | Observe training trajectories and the gradient's components as primary data; end-of-run values are summaries. |
| L7 | Proposed patching the backward graph to rescue a hypothesis | When an observation says the frame is wrong, change the frame; don't patch around the observation. |
| L8 | Structure comparisons at fixed settings mixed "bad structure" with "bad settings" | Compare structures by their stability regions, or at each one's best, or under a parameterization that aligns them. |
| L9 | Implementation ran ahead of the question several times; the researcher had to pull the discussion back to the central goal | Before any implementation, state the question, what each outcome would change, and the cheapest decisive test. |
| L10 | E2's first readout operator (least-squares refit) was a weaker classifier than the network's own readout (0.88–0.91 vs 0.99–1.0 after grokking), so it could never show a lead. Caught by a sanity run before Kaggle (22 Oct). | Check every "what could the network do now" operator against the network itself on a grokked checkpoint: it must at least match it. |
| L11 | The pre-run review of E1–E4 (22 Oct) found six blocking problems: a primary prediction that could not fail (censoring); a stability grid whose ceiling sat below the known optimum and whose definition differed from Study 2; a manipulation (E3) confounded with weight scale; and lesson L5 repeated in E2. One of its proposed fixes was also wrong: a decay-aware drift operator that the review checked only on a weight with no gradient, which predicted nothing on real runs. | An independent review before every run, and check each fix too, by measuring the operator against what actually happens. |
| L12 | Before the review, several predictions could only come out one way: E1 P2 ("gradient descent is ≥ 5× worse"); E3 P1, with speed and scale pushing the same way. | For each prediction, write down a concrete result that would make it fail, and check that the design can produce that result. |
| L13 | E2's noise floor (H = 0 jump) measured only floating-point nondeterminism, not how a run responds to a perturbation, so "saving above the noise floor" was close to automatic. Also, the sidedness of the sign tests was ambiguous in the preregistration. | A null for an intervention must be an intervention of the same size in an uninformative direction. State the sidedness of every test. |
| L14 | E3's co-location measure compared memorization at its peak with generalization at the end, so migration over time read as a role split. My first matched-time reanalysis then used early checkpoints, where generalization is below zero and its share is meaningless, and read agreement that was only saturation at 0 or 1. Both were caught by independent recomputation. | Compare role measures at the same time, at checkpoints where both quantities are well away from zero. Don't read agreement between saturated shares as evidence. Have exploratory reanalyses verified too, not just preregistered numbers. |
| L15 | E1–E4's raw results lived only on the Mac and in a gitignored folder, so RESULTS.md's numbers could not be rechecked from the repo. Fixed 23 Oct with `experiments/walk/results_data/`. | Commit a compact per-run copy, the analysis output and the verification scripts with every results report. |
| L16 | E4's "no stable region" for C4 read as "the chain can't learn" until the per-lr counts were checked: it groks at lr 2e-3, but collapses 14–22 times. | When a headline statistic is a threshold rule, also report what the runs did beneath it (fit, grokked, collapses). |
| L17 | The priors round's first draft had a headline condition (search share < 0.25) that could not fail on its own grid, because the log scale anchored at k\* = 3 makes "mostly prior" nearly automatic; and a random-direction null that differed from the drift on a known lever (weight norm). Both caught by the pre-run review (24 Oct). | Before preregistering a ratio or share, enumerate every value the grid can produce and check both verdicts are reachable. A null must match the intervention on every known lever except the one under test. |
| L18 | Near a threshold, a strict "≥ 0.95 at every eval" criterion mostly counted single-eval dips: base vs wide, and P3, turned on one dip in 21 evals (priors round, caught by recomputation). | Preregister a dip-robust criterion as primary, or at least as a planned sensitivity check (e.g. the window mean ≥ 0.95, or ≤ 1 dip), and check orderings against it before reporting them. |
| L19 | The worlds design (26 Oct) called a split of held-out functions "in support", though holding a function out gives it zero prior. It also tested composers on program lengths their decoder could never emit (HALT is forced at the training cap). Caught by the pre-design review. | Before naming a split or writing a prediction, check (a) the test target's probability under each arm's *training* prior, and (b) that every arm can *express* the target at all. |
| L20 | The worlds revision-3 derivation of superposition's gap (§5.1) averaged the KL over all 49 inputs, the context inputs included, while the training objective scores only the remaining inputs. This overstated the gap by about 2× and put the headline threshold at the bound. Caught by the independent review. | A derivation that sets a preregistered threshold must score exactly what the objective and the endpoint score: the same inputs, the same prior per cell, the same normalization. Check by reproducing one endpoint number through the derivation code. |
| L21 | Revision 3 derived the expected failures of the superposed arm (local optima), and none for the committed arm, then predicted that the committed arm's failure tail would be smaller. A per-context committed learner using the design's own estimator found 0 of 30 programs of length ≥ 3. | Derive or pilot trainability for *every* arm, especially the one the hypothesis favors, before writing predictions that compare their failures. |
| L22 | Four design revisions of the worlds round went to review before anyone checked that the toy could separate the thesis from known work. On pair worlds with the generator's own library, the posterior over composition classes equals the posterior over output tables, and exact inference is a 5 ms lookup. The thesis's distinguishing quantity was zero by construction. | Before designing arms, write down the quantity that distinguishes the thesis from the nearest known result, and compute it on the toy. If it is zero, or trivially cheap to get exactly, change the toy first. Ask the experimental-value question (what each outcome would change) at the first review, not the fourth. |
| L23 | Rung-0 pilot 1: the learned-halting arm collapsed in every configuration because its halting head started at λ ≈ 0.5 while the control's started at λ ≈ 0.12. With PonderNet's expected loss, early halting mass starves deep steps of gradient, the block never learns them, and early halting becomes self-confirming. | Initialize any learned halting or routing policy so that its probability mass is spread over the depths the task needs, and give every arm the same policy init. A halting arm that fails must be checked for this before it counts as a result (gate G4). |
| L24 | Rung-0 revision 3 (Entry 58): the new headline arm's head read raw log features spanning 18 units with a bias at −2; under AdamW the whole step budget moves a parameter by ≈ 1.1, so every working policy was unreachable, and the arm would have failed in a way that mimicked the mechanism under test. Caught by the design review's arithmetic, not by the tests (the planted head passed because its weights were set by hand). | Before an arm runs, check that a sufficient solution lies within what the optimizer can reach from the init in the step budget (Σ lr for Adam-type optimizers); standardize a policy head's inputs as the state head's are; and put the planted solution's parameters within that reach, so the planted test also tests reachability. |
| L25 | Rung-1 revision 1 (Entry 61): "new orders never seen in training" were defined on the data's surface (PP runs, alternation), but the router's receptive field was the matched pair, for which every stratum was in-distribution; the independent reviewer's 1,800-step CPU run passed the headline predictions at 7 % of the budget before any GPU step. | Define out-of-distribution relative to what the model's state and receptive field actually see, not the data's surface; before preregistering a generalization prediction, run the cheapest configuration of the thesis arm on CPU and check that the prediction is not already true there. |
| L26 | Rung-1 main grid (Entry 63): the arm designed as the easy control (the local trigger) was the one that failed, and the failure mode — the init phase teaches a state-only rule that happens to be the oracle's for one trigger placement and the opposite for the other — was invisible to the design, both reviews and the pilot, which did not run the control. | A control arm that "passes trivially" is a prediction like any other: run it in the pilot, and before predicting which of two configurations is easier for an optimizer, write down what the first hundred steps teach from the init's stray mass and check whether it agrees with the target rule. |
| L27 | Rungs 0–2 (Entries 59–64): the preregistered cores kept turning out foregone or derivable (rung 1 revision 1 passed at 7 % of budget on CPU; rung 2 revision 1's headline was decided by a CPU seed), while the findings came from unplanned failures and side cells (F22–F23, F29, F33, program B). | Build each rung's core from cells whose outcome cannot be derived or cheaply observed, tied to a named risk in the Direction section; run every derivable cell on CPU first and demote it to a check at minimal seeds. |
| L28 | Rung 2 review 3 (Entry 66): an arm trained toward a FIXED target (PF) was put in the core as uncertain, but its end state is the target pooled over the features the model can distinguish, which is computable; and at step 0 it equalled another arm with a different likelihood, so the contrast it was meant to isolate was absent. | Before calling a fixed-target arm uncertain, compute its pooled fixed point; compare arms' step-0 gradients class by class to see which ingredient a contrast actually varies. |
| L29 | Rung 2 CPU sanity (Entry 66): three sampled signals had absorbing degenerate states that only showed up in training — identical samples zero IW's leave-one-out advantages (rev 2), a free HALT term let TB drive routes into dead, and TD(0) against a poor policy's value collapses into halting. | Run every new estimator on CPU before preregistering it, and look for states where the loss or its variance is zero regardless of reward; fix or demote before the grid. |
| L30 | Rung 2 pilot → main (Entry 68–69): the one-seed pilot produced a striking program (R) for EX-AL on δ = 0, and a reading was drafted around it; at 3 main seeds EX-AL stalled like EX, and R turned up instead in a different arm. | A pilot sets learning rates and catches bugs; a pilot surprise is a hypothesis for the main grid's checks, never a reading. Write the surprise into the analysis as a detectable category and count it. |
| L31 | Rung 2 TB16 (F35, RESULTS §3H): a matched step budget on a cosine schedule records a slow-but-converging signal as a failure (0/5 at 0.97–0.98, still rising when the lr hit 15 %). | Report every arm's curve with its count, and say "slower by k×" when the curve says so; before a grid, check that the budget covers the slowest arm's CPU-estimated time at the decayed lr, or use a constant lr for the comparison of speeds. |
| L32 | Rung 3 derive step (Entry 72, review 2): three of the open rows were set from a coverage averaged over words the committed router already solved, a stage-1 run was described from its final evaluation alone ("sat in all-H") when its curve showed it had learned and then collapsed, and the dose cells' evaluation strata shrank with the dose. All three were caught by the re-review, none by the author. | A derive-step quantity that sets a prediction is conditioned on the cases the prediction is about (here: the words that must change). A run's outcome is read from its curve, not its endpoint, and the analysis carries an outcome category for every way a run can end at zero. An evaluation set is generated on the standard configuration whenever the treatment changes the data; the dose must not change the ruler. Re-review the derive step before stage 2 of any two-stage design; it is where preregistration is quietly rewritten. |
| L33 | Rung 3 stage 2 (Entry 73): the analysis code's "collapsed" rule read a key that does not exist in the records (`q_steps_P`), so its test on synthetic records passed and the first automatic table called 61 of 94 runs collapsed; the same first draft also read the end-state histogram as if halting and moving were alternative outcomes, when the curves show 10 of 11 EX runs halting first and the movers escaping from it, and said "never moved" of a signal that had held the moved program for 1,000 steps. All caught by the independent reviewer from the curves. | Test analysis code on fixtures cut from a real record (or assert that every key it reads exists in one); an end-state histogram is a projection of a trajectory, so classify from the curve (state sequence per checkpoint), not from the endpoint, and say "ended in" rather than "was" when only the endpoint is known; hand the reviewer the records, not the tables. |
| L34 | Rung 4 code review 2 (Entry 77): the size-extrapolation prediction P3 would have been decided in *both* directions by untrained embedding rows — the structural router's depth/size/path rows beyond the training range (cost ≈ 0.1 of size-6–7 success in a 2,000-step probe) and the control's learned positions past the training length (100 % of size-6–7 inputs) — after review 1 had already moved the artefact out of the output head (B4). | An extrapolation prediction must be checked on the *input* side as well as the head: every categorical feature read at the extrapolation sizes must have a trained row (clamp to the training range, or use a fixed encoding), and the check is mechanical: list the rows each eval set touches and the rows the training stream labels. |
| L35 | Same review: the Kaggle test gate would have failed on the uploaded bundle (the tests load a module and fixtures the include list omitted) — the second time a stage was lost to the bundle differing from the tree (Entry 72). | Before any launch, reconstruct the bundle from the include list and run the gate inside it (a reviewer did it in minutes); treat the include list as code under test. |
| L36 | Rung 4 derive step (Entry 77a): review 2 had said the lower rungs might tie near zero on P4's GP1 stratum; the design answered with a fallback clause ("if no cell is separated, not readable") instead of the computation, and the computation on the built eval sets showed all eleven preregistered cells degenerate (pooled rungs 0.03 / 0.07 / 0.30), so the rung's Q3 core would have been unreadable before a single run. The cells were amended from the ladder alone before the pilot (§16). | When a review says a preregistered cell may be degenerate, compute that cell's floors on the real or smoke data before fixing it; a "not readable" fallback is a report, not a design. Run the derive step before the preregistration is called final, not after. |
| L37 | Rung 4 pilot, 8 Nov (Entry 77a): the Kaggle test gate failed because Kaggle's dataset pipeline auto-extracts the code archive and gunzips its nested `.gz` members — the fixtures and data files arrived as `.json` / `.jsonl`. The local bundle check (L35) had passed because it unpacked the tar itself, so it never saw the layout the kernel sees. | A bundle check must reproduce the platform's unpacking, not just the archive's contents: for Kaggle, also run the gate on the auto-extracted, gunzipped layout (`run_on_kaggle.py` now restores `.gz` members and logs the counts). Any file whose name the loaders match by suffix is a liability on a platform that rewrites suffixes. *Same day, second failure:* the simulated layout was the one I assumed (bare gunzipped files), not the one Kaggle produces (a directory named after the member, holding the file) — simulate from an observed layout, and when the platform's behaviour is unknown, ship the job with a manifest and a self-check that prints what it found, so one failed run is enough to learn it. |
| L38 | Rung 4, 9 Nov (Entry 78): the kernel launched each shard as `python … \| tee \| sed` and read the pipeline's exit status — `sed`'s. A shard killed for memory in a local simulation was reported as `rcs=[0, 0]`; only the missing result file showed it. | Read the worker's status, not the pipeline's (`pipefail`), and make every job summary compare what it produced with what it was asked for (`results n of N`). A status that cannot be non-zero is not a check. |
| L39 | Rung 4, 9 Nov (Entry 79): after P4's cells were found unreadable, an independent audit walked the other six pass rules against the derive step's numbers and found one unreadable, three that could pass or fail for reasons unrelated to their claims, and two with no decision rule — after two design reviews, a code review and the author's own reading. What found them was not reading the rules but *scoring policies on them*: a symbolic policy that should fail each rule and one that should pass it, on the real strata. | Before a preregistration is called final: for every pass rule, (i) score at least one symbolic policy that embodies the claim's negation and one that embodies the claim, on the built strata, and require the rule to separate them; (ii) for every reference, write down its action set and the information it assumes, and score the rule on the reading that matches; (iii) give every clause a statistic over seeds, a count, and a name for the outcome in between; (iv) compute the verdict in code with a planted pass and a planted fail. Do this at the derive step, before the pilot — amendments made after a pilot have to argue their independence from it. |
| L40 | Rung 4, 9 Nov (Entry 79): the main grid as first prepared saved outcomes only. The audit's strata could be rescored from the per-instance records, but an eval set enriched for the 209-instance stratum could not have been scored without retraining 27 GPU-hours. | Save the trained model of every reported run (it was 2.4 MB a run), with a test that the file alone reproduces the run's final evaluation. |
| L41 | Rung 4, 9 Nov (Entry 79, review 5): the absolute control's designed artefact was stated from the architecture — "the categorical head's rows 9–12 are never labelled at size ≤ 5" (a term of 5 leaves has 9 nodes) — and never counted. No action is valid at the last two prefix positions of a full-size term, nor an A rule at the two before: the stream labels nothing at rows 7–12. The wrong set went through the implementation choices, the derive step, two reviews, an audit and my own amendment; the rule built on it would have called a head that failed exactly its artefact "a table beyond the artefact". The fifth reader found it by brute force over all 47,155 terms. | A property of the training stream is counted on the stream, not inferred from the model's shape: which labels, slots, positions, lengths and worlds actually occur, as a table in the derive step. Every "never" and "always" in a design gets a count beside it. |
| L42 | Rung 4 main grid (Entry 80): P6's amended rule asked RT-post's distance to its target to be below that of a uniform-over-shortest-actions reference. On 1,003 of the 2,213 instances the target *is* uniform, so the reference scores zero there and no learned policy can beat it; the rule returned "does not hold" in 0/10 seeds while on the other 1,210 instances RT-post was closer than uniform in 10/10. The rule had planted pass-and-fail tests (exact planted outcomes) and two blind reviews. | When a rule compares a learned policy with a reference, count the instances on which the reference equals the target and score the rule off them; and make the planted "pass" a noisy version of the claim (the target plus estimation error of a realistic size), not the exact target. |
| L43 | Rung 4 main grid (Entry 80): after L34 the structural features were clamped to the size-≤ 5 range so that no untrained embedding row is read at sizes 6–7. The clamp makes the nodes larger terms add share features; at sizes 6–7 the router fell below the *greedy* rung on one-step instances (0.65 against 1.00), ended a quarter of its episodes in a wrong halt, and its success fell along nesting depth past the clamp. P3 was falsified, but its preregistered reading ("learned a table") is not separable from the encoding's aliasing — the prediction's input side failed a second time. | Clamping an input feature at the edge of the training range trades untrained rows for aliasing. Before an extrapolation grid: count the inputs at the extrapolation sizes whose encoded features collide, and check one-step (k = 1) instances at those sizes on a quickly trained model — if the router is below greedy there, the instrument measures the encoding, not rule against table. Prefer a fixed encoding, or training sizes that leave a margin inside the feature range. |
| L44 | Rung 4 main grid (Entry 80): EI16 took 272–281 min a run against 148 projected from the 3,000-step pilot, the grid 44.8 process-hours against 27; the session-budget guard stopped session 1 cleanly at 29 of 41 runs and the resume worked, but the second session's fetch overwrote the first session's logs. | Project an arm's cost from the pilot's last interval, not its average, for arms whose per-step cost depends on the policy (search, beams), and carry a 2× margin on them; fetch each session into its own directory. |
| L45 | Rung 4 results review (Entry 81): the dataset build drew the eval sets from the same sampler as the training stream and never excluded training instances; 36 % of the training-world eval instances are verbatim training pairs and about 30 % of held-out ones are rows trained in a contained world. The write-up said "the instances are new pairs". Two design reviews, a code review, an audit, review 5 and I all took disjointness for granted; the sixth reader streamed the training files and counted. P4 survived the correction; the per-k rates and the picture of held-out generalization did not survive unchanged. | A dataset build ends with a counted overlap table — eval against train at every level the model could recall (instance, input row, goal) — written into the dataset's meta file and asserted by a test; every result is printed on the never-trained subset beside the pooled one. Never write "new", "unseen" or "held-out" of instances without the count behind it. |
| L46 | Same review: I reported that RT-canon's held-out curve "peaks before the end in 10/10 seeds and finishes 0.01–0.10 below its maximum". "Final − max" is negative for any flat noisy curve (expected −0.034 from binomial noise alone at that n), and the final point was on a different instance set from the curve points. In window means the curves were still rising. | A statistic of a curve needs its null before it is read: compute it on a flat curve with the measurement's own noise, and never difference points taken on different instance sets. Trends are read from window means. |
| L47 | Same review: the first write-up said the absolute control's collapse was "confounded with its clamped positions"; that router has fixed sinusoidal positions and no clamp. I wrote it from my memory of an earlier state of the code (L34's fix), on the day I recorded L41 about not inferring properties from the architecture. | A sentence that attributes a result to a property of the code is written with the code line open; a reviewer's brief asks for every such sentence to be checked against the source. |
| L48 | Belief experiment, design revision 0 (Entries 88–89): I wrote a pass rule ("nearer the exact update than the mask on two thirds of the informative decisions") into a design the same day I recorded why such rules need scoring first. Simulated under it, a model making three quarters of the exact update scores 0.65–0.73 at realistic noise — on the bar — and the share is a step function without noise. A projection coefficient on the same data tracks the true share to ±0.005. | Before a pass rule is written into a design, run stand-in models through it: points along the line between the references it is meant to separate, at several noise levels. Prefer a statistic that estimates a quantity (how much of the update, with an interval) to one that counts wins against a bar. |
