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

- **The original question:** what structure do network topologies promote during learning, and can wiring support the generalizing solution and keep the memorizing shortcut from interfering?
- **How the frame has moved (22 Oct):** from "which wiring" towards "how learning is organized":
  - the gradient as a single-path, averaging learner;
  - shared error versus separate errors;
  - search over structures versus walking;
  - topology as a hypothesis about how the data is generated.
- **Scope principle:** toy tasks isolate one piece; realism is added back step by step. A fixed dataset is not the learning system we care about in the limit.

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

### Derived constraints (arguments checked numerically during design)

**F12 — Under AdamW, decoupled weight decay is itself the time filter.** *Established (derivation plus check).*
- The memory time constant is τ = 1/(lr·wd).
- AdamW cancels most pre-optimizer gradient scaling.
- A linear time-invariant filter of the gradient cannot select between memorization and generalization: it rescales every persistent component by the same factor.
- Evidence: [fastslow/DESIGN.md](../experiments/fastslow/DESIGN.md) §2 and [DESIGN_checks.md](../experiments/fastslow/DESIGN_checks.md).

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
- *2026-10-22:* argument → **preregistered as E3** ([walk/PREREGISTRATION.md](../experiments/walk/PREREGISTRATION.md)): direct manipulation of the branch speed ratio ρ at fixed lr·wd. P1: shares rise with ρ. P2: co-location at every ρ.

**H2 — Under shared error, a fast memorizer takes every mode, the rule included.** *Argument (corollary of H1).*
- **Claim:** complementary learning systems work in brains because the cortex learns from replay, not from an error it shares with the hippocampus.
- **Falsifier:** a fast flat-kernel component and a slow component, under shared error, end up dividing rule and exceptions.
- **Cheapest test:** a planted two-branch case with known kernels.

**H3 — Masking.** *Argument; partial support from pilot plateau dynamics.*
- **Claim:** once the memorizing component fits the training set, it removes the error the generalizer needs. Weight decay then drives the plateau.
- **Open sub-question:** is masking harmful, or the mechanism by which grokking happens at all?
- **Falsifier:** removing masking (e.g. the unmasked arm) leaves the generalizer's trajectory unchanged.

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

**H6 — Agreement across halves separates the signals without labels.** *Argument (derivation).*
- **Claim:** for disjoint halves A and B of the training set, E⟨G_A, G_B⟩ measures the shared (population) part of the gradient, and ‖G_A‖² − ⟨G_A, G_B⟩ measures the memorizing part.
- **Falsifier:** the estimate disagrees with the test-label-based gen/mem gradient split.
- **Cheapest test:** compare it against the existing 5A measurements.

**H7 — Chain instability comes from compounding Jacobians.** *Argument (Phase 0 Prop. 3); open since Study 2.*
- **Claim:** sharpness, or the effective step on the end-to-end map, grows with chain length C.
- **Falsifier:** sharpness does not scale with C at matched training progress.
- **Cheapest test:** track the top Hessian eigenvalue on a subset of Study 2 graphs.

**H8 — A depth-aware parameterization aligns stability regions across structures.** *Argument (μP literature).*
- **Claim:** with μP-style depth scaling, the optimal lr is shared across structures, so any remaining instability differences are structural.
- **Falsifier:** optimal lrs still differ by structure under that parameterization.
- **Cheapest test:** a learning-rate sweep per structure, with and without the scaling.
- *2026-10-22:* **preregistered as E4.** Stability regions for C = 1–4 under std, uniform 1/√L and per-block 1/√depth output multipliers, on an lr × wd grid. Depth-μP's known result is transfer across depth; whether it transfers across wirings at fixed depth is the open part.

**H9 — Competing hypotheses scored by prequential surprise avoid masking.** *Argument.*
- **Claim:** if hypotheses are not summed or trained jointly, each predicts alone and is scored by its surprise on each example *before* learning from it. Then memorization earns nothing at the level of which structures survive. Masking inside a branch disappears if the branch cannot express the shortcut, and true exceptions go to a separate memory store.
- **Falsifier:** prequential selection still favors overcomplete or memorizing structures; or constrained structures cannot be optimized.

**H10 — Schema assimilation separates a real learner from gradient descent.** *Argument, with an animal analogue (Tse et al. 2007).*
- **Claim:** once a rule is learned, new consistent examples should be absorbed almost immediately and generalize, while exceptions should be stored without disturbing the rule. Standard gradient training will walk for both and may overwrite old examples.
- **Falsifier:** standard training already assimilates consistent examples in one or two steps without interference.

---

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
| I10 | Measure how much information a training run has versus what it uses: steps or examples needed to determine the function, versus to produce it | 22 Oct discussion | **Preregistered as E1** (22 Oct): prequential code length of gradient descent on a stream vs a program-prior learner; primary question is whether more steps per example reduce gradient descent's sample need. Nearest prior art (weight-compression complexity of grokking) has no ideal-learner reference. |

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
