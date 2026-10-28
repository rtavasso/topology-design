# History and decisions

This is a condensed record. The full v0 reasoning, including every pivot in the researcher's original discussion, is preserved unedited in [`archive/v0/docs/01_intellectual_history.md`](../archive/v0/docs/01_intellectual_history.md) and [`10_decision_log_and_hypotheses.md`](../archive/v0/docs/10_decision_log_and_hypotheses.md). Do not rewrite past entries; append.

## v0 (to 19 Oct 2026, written with ChatGPT): pivots 1–15 in brief

1. **Reverse residuals → feed-forward DAGs only.** Recurrence confounds topology with extra inference compute.
2. **Residuals shift the parameter origin; they don't add degrees of freedom.** Expressibility, accessibility from initialization, and robustness are distinct.
3. **Topology as a polynomial in weights.** Graphs with equal function classes can differ in first-order trainability (the $(1+bc+abc)$ vs. $(1+c+abc)$ example).
4. **No learned gates.** They change the operator class.
5. **Smoothness ≠ convexity ≠ useful learning.** Per-example transfer via gradient dot products.
6. **Fixed softmax cross-entropy.** Additive branches share the residual $p-e_y$; a fast short route can starve a long one. An exact factorized softmax was rejected as the main design because it imposes token groups.
7. **"Productive frustration."** Unresolved error the shallow route cannot fix but a deep route could. Do not maximize disagreement.
8. **Cross-entropy is the training loss, not the research outcome.** Target memorization/generalization mechanisms.
9. **Discover functions before presuming circuits.**
10. **Scale has hidden covariates** (feature load, sparsity, resolution).
11. **Delay architecture search.**
12–15. **Adversarial self-review.** Separate expressivity, optimization and mechanism. Adopt the matched linear pair $(I+A+B)$ vs. $(I+B)(I+A)$ as the flagship. Measure functions and one-example transfer before circuits. Add decision gates.

## Pivot 16 (19 Oct 2026, with Claude): derive before measuring; theory-first restructure

**Premise.** An external review found that the v0 flagship (Study 1) is answered largely in closed form by deep-linear-network theory that v0 never cited. Every claim below was verified numerically, and all are now tests ([phase0_linear_theory.md](phase0_linear_theory.md)):
- The parallel arm $(I+A+B)$ conserves $A-B$; it is a single matrix trained at doubled learning rate, i.e. convex softmax regression.
- From $A=B=0$ the composed arm is a balanced depth-2 linear network. Its end-to-end flow is Arora et al.'s closed-form preconditioner.
- Its tangent kernel, and hence first-order one-example transfer, is a closed-form function of the current learned map. The two v0 "primary measurements" were largely the same measurement.
- The first single-example step of the composed graph is the parallel step times a scalar $1-\eta\kappa/2$. The $BAx$ term adds no new direction for a single example.
- Both arms reach the same function on non-separable data, so differences are transient.
- Identity initialization removes the Saxe-style plateaus v0 cited as motivation.
- A sequential chain trained from identity keeps $\det W>0$: it can never fully erase a direction or reverse orientation. The parallel graph can (tested on a reversal task).
- First-order equivalence at zero holds for **every** graph in the parent-selection family, not just the hand-built pair. Topology enters only at second order in branch weights.

**Decisions:**
- **D16.** The linear pair becomes a Phase 0 fixture, not the flagship.
- **D17.** Organize the program around the chain *topology → parameterization symmetries → conserved quantities → preconditioner → learning order and implicit bias → emergent structure* ([SPEC §2](SPEC.md)).
- **D18.** Richness (initialization scale × fixed branch multiplier × required distance from identity) is the primary experimental axis, because topology effects are higher order in branch weight.
- **D19.** "Productive frustration" is formalized as saddle/plateau dynamics. It is measured with plateau length, escape direction and hidden-progress measures, using existing theory (leap complexity, saddle-to-saddle).
- **D20.** Gated deep linear networks (fixed gates; Saxe et al. 2022) are the analyzable bridge from linear to nonlinear (Phase 1). Whether fixed input-dependent gates count as "parameter-free routing" is left to the researcher. Phase 1 uses them, labeled.
- **D21.** New flagship (Phase 2): does sequential composition lower the effective leap of composite features relative to learned intermediates?
- **D22.** Methodology:
  - directional pre-registered predictions;
  - per-arm learning-rate tuning (a shared learning rate is not a fair control when preconditioners differ);
  - over-provision width instead of restricting targets;
  - compare at matched steps and at matched loss;
  - use existing validated instruments (LLC, coherence, Tracr/InterpBench) before building new ones.
- **D23.** Documentation:
  - v0 docs moved to `archive/v0/`, unedited;
  - active docs are the SPEC, the Phase 0 theory, the literature map, this file and the researcher's original questions;
  - no new parallel copies of the same derivation.

**Rejected alternatives:**
- *Running v0 Study 1 as specified.* Its outcome is predictable from theory, and its own Gate 1 stop rule ("reducible to a known reparameterization effect") is met on paper.
- *Deleting v0.* The original questions and pivots still carry the motivation.

**Exploratory finding recorded, not claimed.** In the mixed graph $k=(0,1,0,2)$ (a chain plus a shortcut), the shortcut carries more of the change in directions being erased than in directions being amplified (subspace shares 0.40–0.42 vs 0.24; 4 seeds). Per-mode behavior is non-monotone, and one mode has the shortcut opposing the change. See Phase 0 Prop. 7.

**Expected falsifiers for this direction:**
- Prop. 7 fails to derive cleanly or the exceptions dominate;
- the Phase 2 lag effect is absent at per-arm-tuned learning rate or does not scale with richness;
- the GDLN race predictions do not transfer to ReLU networks.

## Pivot 17 (19 Oct 2026): literature review tightens novelty and protocol

**Premise.** A five-track literature review covered theory, empirical topology studies, measurement methods, inference methodology, and emergent-structure targets. It is in [literature_review.md](literature_review.md), with notes in [literature_review_notes/](literature_review_notes/).

**Evidence that changed decisions:**
- **Local invariants are ruled out.** Residual blocks share conservation laws with skip-free blocks (Marcotte et al. 2025). So topology acts through global composition, which reinforces D17–D18.
- **Several Phase 0 results are anticipated by prior work:**
  - Prop. 2 is an instance of permutation-symmetry collapse (Ziyin 2024).
  - Corollary 3a is anticipated by Bartlett, Helmbold & Long 2018 (checked against the arXiv abstract).
  - The weight-decay bias of residual chains is known (Boix-Adsera 2025).
- **The Phase 2 parallel arm is one two-layer network of width $L\cdot m$ plus a skip.** The sample-complexity version of H2 is occupied (Nichani 2023; Allen-Zhu & Li).
- **Structure metrics are fragile.** CKA can be manipulated and is biased; unrestricted causal-abstraction maps are vacuous; modularity metrics need topology-preserving nulls.
- **Variance studies imply more seeds.** About 29 runs per arm are needed for headline claims (Bouthillier et al. 2021).

**Decisions:**
- **D24.** Phase 0 novelty bets are DAG implicit bias (including commuting liftings) and RLCT of residual/parallel parameterizations. Restatements of prior results are attributed in the theory doc.
- **D25.** Phase 2 is reframed openly: the parallel arm is anchored to two-layer theory and its plateaus are predicted before training. Novelty is the lag dynamics, richness scaling and one-edge-mutant dose–response, plus induce/eliminate reversal.
- **D26.** The inference protocol is upgraded (SPEC §3):
  - rival-explanation table;
  - pilot null, ≥10 paired seeds, ≈29 per arm for headlines with $P(A>B)$;
  - richness phase diagram whose effect must vanish as $\alpha\to0$;
  - three clocks;
  - tuning-budget curves;
  - preregistered primary endpoints with kill criteria;
  - mask-preserving and density-matched nulls.
- **D27.** The measurement stack is tiered by validity across topologies (SPEC §5), with function space first. Causal abstraction uses restricted maps and random-init baselines. Representation similarity is used only at symmetry-matched sites, with nulls.
- **D28.** Emergent-structure targets are prioritized (SPEC §6). Phase 4 adds a fixed decaying identity coefficient and induction-head emergence in sequential vs parallel attention.

**Expected falsifiers:**
- the pilot shows structural endpoints too noisy to power at feasible seed counts;
- the paired effect does not vanish as $\alpha\to0$, which would mean a confound;
- the parallel arm deviates from its two-layer prediction, which would mean the anchor itself is wrong.

## Pivot 18 (21 Oct 2026): back to the motivating questions

**Premise.** The researcher flagged that Studies 1–3 had drifted from the program's motivation.

**Evidence of drift:**
- The outcomes became learning dynamics, not learned structure. Grokking speed (Study 1) and stability ceilings (Study 2) say how fast or how stably a graph trains, not what it learns.
- Each confound became the next study. Study 3 spent five calibration rounds searching for a task setting where long chains grok.
- Round 5 proposed gradient clipping and LayerNorm as the setup, which v0 §2 allows only as labeled diagnostics.
- The composition task presupposed a circuit (a+b computed as a stage), against v0 §7's warning about probing for expected structure.

**Decisions:**
- **Study 4** ([experiments/paths/PREREGISTRATION.md](../experiments/paths/PREREGISTRATION.md)) asks the three original questions on one shared run set: path roles, long-path learning under short-path friction, and memorization vs generalization.
- It uses modular addition, which already works, and eight graphs linked by single-edge rewirings.
- Each graph gets its own learning rate in its stable range. No stabilizers.
- All measurements are exact decompositions of the output function (`grokking.path_measure`): per-path support for the rule on unseen inputs and for train labels through non-rule structure, with signs giving construct, interfere, memorize or destroy; composition dependence by re-evaluating a block on the raw embedding; and total effects of any block subset.
- The composition study (Study 3) is paused. Round 5 is shelved. Studies 1–2 stand as context: Study 2's ceiling motivates the per-graph lr.

**Rejected alternative:** stabilizing long chains with normalization or clipping. That changes the model class the questions are about.

**Pre-run review (same day).** An independent review found blocking problems, all fixed before any run:
- `gen` leaked memorization through the full-grid rule projection; the same leak affects Study 1's rule quantities, so their caveat is updated;
- the H2 contrasts had the predicted answer built in;
- the lr rule would have chosen the most unstable lr.

The graph set grew to eight: `pk0110` and `pk0121` were added as clean H2 controls. H1 must now hold under both direct and total-effect (Shapley) attribution. Details are in the preregistration's *Pre-run revision*.

**Expected falsifiers:**
- H1 fails: depth does not sort rule vs memorization.
- H2 fails: an added short reader does not reduce the long blocks' utility.
- H3 fails: Study 1's composition cost was produced by the shared lr.

## Entry 19 (21 Oct 2026): Study 4 results — roles co-locate; learning rate decides which path carries the function

**Premise.** Study 4 ([RESULTS](../experiments/paths/RESULTS.md)) asked the three motivating questions on 248 runs.

**Evidence:**
- The preregistered verdicts were H1 supported, H2 not supported, H3 supported. An independent recomputation reproduced them.
- H1 rests on one unstable, grid-edge graph (`pk0120`). Elsewhere, D is 0.02–0.05.
- The H2 contrasts were structurally flawed: sibling additivity, and an lr confound.
- H3 is mostly the learning-rate assignment: τ(C, delay) ≈ τ(C, 1/lr).
- **Robust (exploratory):** in 6 of 7 mixed graphs, memorization and generalization occupy the same blocks in the same proportions (cosine 0.98–0.999, against a permutation null of 0.27–0.76). The generalizing function grows in the path that memorized.
- No path ever cancels another's memorization.
- At high learning rates, function migrates onto independent short readers (`pk0120`: share 0.49 → 0.87).

**Argument.** In this setting, topology does not separate memorization from generalization. It decides which path carries both, and the learning rate moves that choice.

**Decisions:**
- The next confirmatory studies are:
  - co-location at a shared stable lr, plus a deep-rule task;
  - migration dose–response, with per-path stability tracked;
  - friction at matched lr (`pk0120` vs `pk0121`).
- Lesson for designs: single-edge contrasts must not change sibling structure, and must be compared at a shared lr as well as tuned lrs.

**Expected falsifiers:**
- Co-location fails at a shared lr, or on a deep-rule task.
- Migration does not track per-path instability.

## Entry 20 (21 Oct 2026): Phase 5 design — fast and slow paths

**Premise.** Study 4 showed that, in a family where every block sees all information, topology acts only as a preconditioner of first-order learning. Memorization and generalization co-locate, and generalization runs on the weight-decay clock.

The researcher proposed separating them by time constant: a fast path that memorizes, a slow path that generalizes, without masking each other on the shared stream. Possibly with low-frequency gradient routed to the long branch, and consolidation from fast to slow.

**Design:** [experiments/fastslow/DESIGN.md](../experiments/fastslow/DESIGN.md), revision 3, reviewed three times (two independent reviews of revision 1, one of revision 2).

**Decisive findings from the reviews and checks:**
- Linear time filters cannot select between memorization and generalization.
- AdamW cancels most pre-optimizer gradient scaling. Two-sided routing reverses the other branch.
- **Decoupled weight decay is itself the time filter,** with τ = 1/(lr·wd). So per-branch (lr, wd) is the direct implementation of "different time constants".
- Decay also sets branch weight scale, so role measures must be scale-invariant (the composition c_X).
- The first split statistics were biased or unstable, and were replaced.

**Decisions:**
- Study order: measurement of signal and masking (5A); per-branch time constants with a CLS prediction and scale-matched arms (5B, the headline); direction-only routing of the preconditioned update, gated by 5A (5C); function-space consolidation (5D); sequential arrival (5E).
- All are labeled diagnostics outside the parameter-free family. Their structural factor (chain vs parallel) stays parameter-free.
- Current repo priority (AGENTS.md: Phase 0) is superseded by the researcher's direction.

**Expected falsifiers:**
- Composition is unchanged by memory length, or changes only without scale matching.
- Removing signal masking does not change L's composition.
- No routed split beats its matched null.

## Entry 21 (21 Oct 2026): Phase 5 instrumentation in the fleet trainer

**Premise.** The Phase 5 studies need features the batched trainer lacked: per-branch time constants, the unmasked arm, update routing, consolidation, arrivals and probes. Each must be checked before any result is read.

**Evidence.**
- Code: `src/topology_design/fleet.py` and `fleet_signal.py`.
- Tests: `tests/test_fleet_p5.py`, 46 tests. Each mechanism is checked against a direct single-model computation, and float64 bitwise tests cover the exactness claims.
- Documentation: [experiments/fleet/DESIGN.md](../experiments/fleet/DESIGN.md), section "Phase 5 extensions".

**Independent review of the first version: six confirmed bugs, all fixed with tests that fail on the old code.**
1. The 5A gradient projection lacked the lr factor. H5A-4's decay-versus-gradient comparison would have been decided by units.
2. Consolidation averaged over classes. The design squares the norm, so κ was effectively 59 times weaker than preregistered.
3. Consolidation was silently dropped when combined with group routing. The combination is now rejected.
4. One consolidation EMA was initialized from the first (to, from) pair's logits for every run.
5. Compaction advanced the consolidation EMA twice in a step.
6. Exactness:
   - a changed loss layout made plain training differ in the last bits;
   - plain runs batched with routed or measured runs took different arithmetic;
   - measurement shifted the routing nulls' random draws.

**Also changed after the review:**
- the 5A time splits now include the current step;
- agreement groups are drawn from the present examples only;
- the permutation null is a derangement;
- the masking measure uses the CE gradient alone;
- added `S_u_topk`.

**Added from the design:** the Grokfast positive control, a fixed-angle rotation null (matched to the time arm's measured angle), path efficiency, branch-alone losses, per-component mem of new pairs, and protocol-matched probes.

**Decisions.**
- The rotation null's angle is passed in from the time arm's run (`route_rot_angle`). Matching it inside one batch would couple independent runs.
- The first-order imprint keeps the centered true-label logit, consistent with the supports.

## Entry 22 (22 Oct 2026): the pilot, a change of frame, and the ledger

**Premise.** The Phase 5 pilot ([PILOT.md](../experiments/fastslow/PILOT.md)) was run on Kaggle before stage 1.

**Evidence:**
- The preregistered lr rule picks 4e-3, where asym collapses 30–61 times. asym is unstable at every lr tried; sym is stable at every lr.
- End-of-run composition saturates at about +0.95 for every branch and arm, so ★H5B-1's effect size is unreachable.
- The multipliers, read at a single step, are noise.

Stage 1 was not launched; its grids were deleted.

**Argument (discussion with the researcher):**
- End-of-run readings discard the dynamics. The gradient during training is the object.
- Under one shared error, credit for each function-space mode goes to whichever component is fastest along it (ledger H1). So "fast memorizer plus slow generalizer" is self-defeating under shared error. Complementary learning systems need a different organization of learning, such as replay.
- The researcher: patching the backward graph would ignore that observation, so the frame changes instead.
- The direction is restated as the researcher's end goal: a learner that sees the minimum by reasoning over the structure of its data, instead of walking the loss landscape. Toy tasks remain the way to isolate pieces.
- Sharper framing from the discussion:
  - the gradient is an averaging, single-path learner;
  - statistical-query lower bounds show reasoning over relations between examples can be exponentially better for some structures;
  - prequential surprise as an unmaskable criterion;
  - structure search with competing (not summed) hypotheses;
  - stability regions to separate structure from settings.

**Decisions:**
- **D29.** [LEDGER.md](LEDGER.md) is created and becomes the first file to read. It records every finding with an honest status, every failed prediction and refuted idea, open hypotheses with falsifiers, ideas, and process lessons. AGENTS.md now requires updating it.
- **D30.** Phase 5 B–D are on hold. The negative results of Studies 1–4 and the pilot are consolidated in the ledger as results (X1–X10).
- **D31.** The next step is chosen from the ledger's open hypotheses. The candidates judged most decision-relevant are:
  - H5: how early the answer is visible inside training;
  - H1: the credit-by-speed law;
  - the information-gap metric for the vision (I10).

**Rejected alternative:** fixing the pilot's problems and running Phase 5 B–D as designed. That would test time constants inside the shared-error structure that the pilot and H1 suggest cannot produce different roles.

**Expected falsifiers:**
- H1's shares are not ordered by η·λ.
- The destination of training is not predictable early (H5 fails), which would mean walking does irreducible work.

## Entry 23 (23 Oct 2026): E1 and E2 results — the gap is data, and the readout is not the bottleneck

**Premise.** The first two preregistered "diagnose the walk" experiments ran on Kaggle (2 × T4; tests passed on the GPU). They were analysed with the committed code and independently recomputed: every number reproduced, and no verdict changed. See [walk/RESULTS.md](../experiments/walk/RESULTS.md).

**Evidence:**
- **E1** (measurement):
  - a program-prior learner needs 3 examples; gradient descent needs about 1,044 (sym; 1,044–1,392 for asym), even with ample steps;
  - on streams it is worse than uniform for most of the stream, at 12–29 nats per new pair, because it predicts confidently wrong after memorizing;
  - more steps per example barely lower its sample need, and raise its interim surprise.
- **E2:**
  - P1, the readout leads: failed, as predicted. The readout is essentially optimal for the current features.
  - P1b, the drift extrapolation leads: failed pooled. It holds in sym in 10/10 seeds and not in asym.
  - P2, jumping along the drift helps: supported in sym, at 100–400 steps saved. asym is chaotic.

**Argument.** For the researcher's goal of seeing instead of walking:
- The large gap between gradient descent and a learner with structural prior knowledge is in **data**, hundreds of times, not in steps.
- Within a run, the walk spends its time on features. In the stable wiring the destination is partly visible from the trajectory, and only a modest fraction of the walk can be skipped.
- Wiring changes how predictable learning is.

**Decisions:**
- **D32.** Record F13–F16, H5 → suggestive and wiring-dependent, and ideas I13 (random-direction jump control) and I14.
- **D33.** The lesson on intervention nulls (L13) applies to future designs.

**Expected falsifiers (open):**
- A matched random kick saves as much as a drift jump (I13).
- The data floor drops sharply with a different optimizer or with perturbation-based credit (I12).

## Entry 24 (23 Oct 2026): E3 and E4 results — credit is not set by speed alone; chain instability survives rescaling

**Premise.** E3 (credit by speed) and E4 (stability regions) ran on Kaggle. They were analysed with the committed code and independently recomputed; every preregistered number and verdict reproduced. See [walk/RESULTS.md](../experiments/walk/RESULTS.md).

**Evidence:**
- **E4:**
  - P0 is supported: the deepest chain has no stable region on the grid, while the all-parallel graph is stable up to 0.016.
  - H8 is not supported under either rescaling.
- **E3:**
  - P1 (shares rise with speed) is supported in both graphs.
  - P3 (step size alone moves credit) is supported in sym.
  - P2 (co-location persists) is falsified as written in asym and inconclusive in sym, a verdict that rests on the clip.
- **The matched-time reanalysis** (exploratory, corrected after verification):
  - speed alone splits roles transiently in sym;
  - asym shows a speed-independent split, with generalization spread across both branches while memorization stays in the chain;
  - the higher-decay branch carries the grokking jump, then hands it back.

**Argument.**
- Under shared error, step size is a strong, steep determinant of which branch carries function. But it is not the only one: the wiring and each branch's weight decay also shape credit, and roles can separate transiently.
- This revises the 22 Oct argument that shared-error training can only decide credit by speed (H1, H2). Its strong form is refuted.
- The chain's stability ceiling is robust to output rescaling, so structure comparisons need per-structure settings.

**Decisions:**
- **D34.** Record F17 and F18; H1 is partly refuted; H8 is not supported; add ideas I15 (preregister the wiring-set split) and I16 (where the circuit forms), and lesson L14.
- **D35.** The one positive lead on the original question since Study 1 (I15) is exploratory. It gets a preregistered, matched-time, block-level test with a kernel measure before any claim.

**Expected falsifiers (open):**
- I15's split disappears at matched times when measured per block, or is explained by the chain's effective speed.

## Entry 25 (23 Oct 2026): record audit after E1–E4

**Premise.** Before discussing what E1–E4 mean, the researcher asked whether everything was recorded for posterity. An audit found gaps:
- the failed preregistered predictions had not been moved to the failures list;
- several hypotheses had no status note after E1–E4;
- there was no synthesis;
- the raw results and analysis outputs were only outside git.

**Decisions:**
- **D36.** Ledger: X11–X16 record E2 P1 and P1b, E3 P2 (asym), H1's strong form, H8 and the decay-aware operator. Status notes are added to H2, H3, H6 and H9, and a dated "state of the evidence" synthesis sits under Direction. Ideas and lessons are renumbered in order, and L15–L16 are added.
- **D37.** `experiments/walk/results_data/` holds the analysis outputs, a compact per-run copy (13 MB: configs, summaries, thinned trajectories) and the independent verification scripts. The full raw results stay on the Mac and in the Kaggle kernel outputs (repo convention: large artifacts outside git).
- A detail surfaced while auditing E4 is now in F17: the deepest chain *does* grok at lr 2e-3, with 14–22 collapses. "No stable region" means a window that is empty, not a structure that cannot learn.

## Entry 26 (24 Oct 2026): discussion after E1–E4 — priors, populations of restricted members, growth

No runs. A conceptual discussion with the researcher about what E1–E4 mean for the end goal. Recorded so later rounds can build on it.

**Arc.**
1. *What the results say* (23 Oct). The gap is data, not steps; gradient descent is confidently wrong on new examples; the plateau builds features; the horizon in weight space is short; credit is not set by speed alone; chain instability is structural under AdamW.
2. *What a prior is.* Universal approximation concerns support, not inductive bias or reachability (Wilson & Izmailov). In the kernel regime the prior is the eigenspectrum and sample efficiency is task–model alignment (Canatar, Bordelon & Pehlevan). Random networks favor simple functions (Valle-Pérez et al.). In trained networks the prior is the search's bias. No free lunch: useful world priors are about recurring properties (symmetry, locality, composition, sparsity), not specific functions. Solomonoff induction is the formal version of "branch over all programs and prune"; a Bayes mixture over N structures costs at most log N nats over its best member.
3. *Generality at the population, efficiency at the member* (the researcher found this compelling). The artificial neuron's generality is the death of its efficiency. Efficient members have restriction, symmetry, reuse and commitment. Self-organizing routes: competition (Jacobs et al. 1991, who hit our co-location problem with summed experts and switched to competing ones), selection plus learning (the Baldwin effect; Hinton & Nowlan 1987), development from a compressed code (genomic bottleneck; Zador 2019), and factorized schemas (Tolman-Eichenbaum Machine).
4. *How competing experts specialize.* The mixture log-likelihood's gradient weights each expert by its posterior responsibility: that is the filter, and it follows from assuming each example came from one source. Failure modes: fragmentation, starvation (masking by responsibility), collapse. Proposed fix: responsibility by out-of-sample prediction with a uniform-predicting memory member (I20). Unsolved: composition versus competition.
5. *The researcher's three loops* (evolve circuits; select circuits and topology per task; train a hyper learner). Pushback: generic blocks are relative to a task family; blocks must be scored by their contribution to compositions, so loops 1–2 interleave; library size has an optimum and can memorize the task suite; the library needs combinators, not only blocks; the useful hyper learner amortizes selection; continuous relaxations (DARTS) reintroduce shared-error credit and collapse to skip connections (Fair DARTS); the payoff exists only across many related tasks. Reframe: one recursive loop (compose, score predictively, compress winners into blocks, amortize proposals).
6. *How to grow a network.* Function-preserving insertion plus fitting the candidate to the error signal its link defines (Cascade-Correlation, GradMax, gradient boosting); score on held-out data with a description cost, which also gives a stopping rule. Fahlman's "herd effect" is co-location under another name. Greedy growth is an averaging procedure and fails on parity; library blocks set how far one step can jump.

**Decisions.**
- **D38.** Record ideas I17–I22 and the frame update under Direction. No experiment chosen yet.
- **D39.** Any move to populations, libraries or growth crosses the "no architecture search" rule (I8). That boundary is the researcher's to redraw, explicitly.

**Expected falsifiers (open):** I18 fails if no restricted member, selected predictively, beats the general network by a large factor on examples needed; I19 fails if grown blocks do not transfer to held-out tasks.

## Entry 27 (24 Oct 2026): the priors round, designed and reviewed

**Premise.** The discussion (Entry 26) ended on one question: is the data gap prior or search (I17)? The researcher asked for the next experiments.

**Evidence before design.** Varma et al. 2023 had already run the ungrokking protocol on a transformer. The rule survives down to about half the from-scratch threshold (S ≈ 2). Huang et al. 2024 report that larger models generalize from less data, against our 24 Oct conjecture that restriction helps.

**Design** ([priors/PREREGISTRATION.md](../experiments/priors/PREREGISTRATION.md)):
- **Q1:** ungrokking vs scratch thresholds on identical nested subsets; sym and asym at base width, sym at three widths. S = n_floor / n_crit.
- **Q2:** E2's drift jumps against radial-matched, random and rescale-only nulls.

**Review.** An independent pre-run review found four blocking issues:
- a headline condition that could not fail;
- a null that was mismatched on weight norm;
- a floor that could be a step budget;
- mismatched success criteria.

All were fixed before launch (listed in the preregistration). New lesson L17.

**Decisions:**
- **D40.** Run Q1 and Q2 as preregistered. P2 is stated on S (≤ 3 supported, ≥ 4 falsified), and the search share is reported, not judged.
- **D41.** Width enters as a two-sided contest (P4) with a bound (P4b): the generic test of "efficiency at the member" (I18).

**Expected falsifiers:** S ≥ 4 (search matters); a narrow network groking at ≤ 348 pairs; pooled median D ≤ 0 against the radial null.

## Entry 28 (25 Oct 2026): priors round results — the gap is the prior

**Premise.** Q1 and Q2 ran on Kaggle (634 runs). They were analysed with the committed code and independently recomputed: every preregistered number matched. See [priors/RESULTS.md](../experiments/priors/RESULTS.md).

**Evidence:**
- **Q1:** in the stable network, the ungrokking threshold equals the scratch floor (870 pairs, S = 1.0; 1.0–1.5 under matched variants).
  - P1 and P2 supported; P4b supported; P4 not supported; P3 an uninformative tie.
  - The Varma-style S ≈ 2 appears only under a lenient criterion.
- **Q2:** the drift direction beats a matched-norm null in all 4 cells (P5 supported), with small effects and qualifications.

**Argument.**
- Below about 870 pairs, the training equilibrium itself prefers memorizing, so the gap is not a failure of the walk to find what it would prefer.
- This removes the motivation for search-side fixes (credit routing, competing members, growth by residual) *as routes to data efficiency*. It points to changing the prior.

**Decisions:**
- **D42.** Record F19–F21 and X17. Note F16 (narrower form), H3 (the prior reading of masking) and I17/I13/I18 (answered). Deprioritize I20/I21. New lesson L18.
- **D43.** The next round's question is about priors: what makes the rule cheaper than a lookup table from a few examples, and can that prior be learned across tasks (I19, I23)? To be discussed with the researcher before any design.

**Correction.** My first message to the researcher said "S ≈ 1–2" and gave a width ordering. The recomputation showed S ≈ 1 under matched criteria, and that the ordering turns on single-eval dips. Corrected in RESULTS.md and the ledger (L18).

## Entry 29 (26 Oct 2026): from the gap-is-prior result to a generator of worlds (discussion)

**Premise.** F19 says the data gap is the prior. The discussion moved through four steps:
1. learned, context-dependent depth;
2. a composable library of uniform blocks routed by the state (a learned interpreter);
3. PFN-style training on generated worlds as the source of the prior;
4. the researcher's framing: blocks as circuits whose serial composition defines a function; the data generator decides which circuits must be learned; real-world grounding supplies memorized facts.

**Literature read** for how generators were designed: PFN/TabPFN, Learning Universal Predictors, Kirsch 2022, Raventós 2023, Garg 2022, MLC, PCFG SET, CFQ/DBCA, LIME, edge of chaos, formal-language pre-pretraining, and shortcuts to automata. The recurring principles (G1–G10), what carries over, and claims C1–C5 are in [generator_design.md](generator_design.md).

**Decisions:**
- **D44.** Record I25.
- **D45.** No experiment is designed until the researcher decides the open questions in §5 of the note: the primitives, how compositional the worlds are, how the real world enters, and the architecture's state.

## Entry 30 (26 Oct 2026): the worlds round designed

**Premise.** The discussion converged on a design ([worlds/DESIGN.md](../experiments/worlds/DESIGN.md); decisions and literature in [generator_design.md](generator_design.md) §§6–14):
- a frozen library of verified primitive blocks (mod 7: add, sub, mul, neg, inc over two registers);
- a composer trained as a PFN over programs, open-loop and execution-conditioned, against a monolithic PFN of equal size;
- training mixes swept over depth, diversity and concept mixing;
- held-out sets certified by enumeration (in support, length 6, concept mixing, and three structural splits adapted from ExeDec).

**Evidence used:**
- function counts by minimal length: 13 / 80 / 552 / 4,179 / 32,741 / 260,279;
- the closure trap measured: additive-only programs saturate at p³ affine maps;
- identifiability: 12 examples pin a function down 93% of the time among all functions at ℓ ≤ 6.

**Decisions:**
- **D46.** 9 cells × 3 arms × 3 seeds = 81 runs. The structural splits get their own cells (S5, X5), because their training restrictions conflict.
- **D47.** Stacked-run batching (every run of an arm in one batched model, with data generation and execution on the GPU) is the throughput strategy. Implemented and tested for exact equivalence and independence. The GPU throughput test (`launch.py --bench-worlds`) precedes the preregistration.

## Entry 31 (26 Oct 2026): exact branch classification, and commitment vs averaging (discussion)

**Premise.** The researcher noted two things.
- On Z_7 we can prove, for every sampled next primitive, whether it moves toward the shortest branch set, laterally, or away; whether it makes the optimal path impossible; or whether it makes any solution impossible within the program-length cap.
- A static PFN averages worlds in its output, while the dynamic meta-model selects a graph that commits to one world. Attention's ability to combine unseen tokens is thereby moved from the output logits to the branch selection.

**Evidence.** `experiments/worlds/branch_classify.py` sampled 25 target functions of minimal length 4 and tried all 32 first moves for each:

| Class | Share of first moves |
|---|---|
| Optimal (distance −1) | 5.9% |
| Lateral (distance unchanged; optimum lost by one step) | 24.9% |
| Away or worse | 69.3% |
| Provably dead (the state merges inputs that the target separates) | 42.2% |

The provably dead moves are a subset of "away or worse". The branch space is mostly irreversible death: overwriting a register destroys information, and nothing recovers it.

**Argument.**
- **Step classes.** Each step's regret is 1 + d(s′) − d(s). Distance falls by at most one per step but can rise without bound, so every non-optimal step makes the minimal length unreachable.
- **What "target" means.** With k context examples, the target is the version space consistent with the context. The exact Bayes-optimal step policy is the posterior mass of minimal programs that extend each prefix.
- **Commitment vs averaging.** Under log loss, a mixture over paths has the same optimum as the PFN: the posterior predictive. What differs:
  - the meta-model's errors are wrong but valid worlds, never blends;
  - a committed path gives coherent joint predictions across queries;
  - a committed path can be checked against the context;
  - novelty is extrapolated in path space rather than in logit space.
- **The price of commitment.** It is irreversible. Breadth (keeping a branch set while the version space is wide) or verification is required.

**Decisions.**
- **D48.** Record I26.
- **D49.** The redesign uses exact step classification and the exact step-level Bayes policy as diagnostics and references. Dense distance or value targets are privileged information and appear only as an optional arm.

## Entry 32 (26 Oct 2026): worlds design revision 2, and its independent review

**Premise.** Revision 2 replaced revision 1 ([worlds/DESIGN.md](../experiments/worlds/DESIGN.md)). It compares a static PFN with two dynamic-graph meta-models (open-loop and execution-conditioned) that build query-blind paths over the symbolic instruction set. All are trained on the same next-token loss, against exact references: Bayes under the training prior and under the generator, the step-level Bayes policy, and the branch classes. Held-out compound patterns are certified at the full step budget (`certify_patterns.py`; brute-force check in `check_certify_bruteforce.py`): mul→inc 9,555 and inc→mul 4,508 certified functions, almost all ℓ = 6.

The certification also showed that a function needing a pattern for its minimal program nearly always has a longer compliant program, so pattern and length are confounded. The design measures pattern by a difference in differences.

A visual walkthrough of revision 2 was published as an artifact (Paths Over Primitives).

**The independent review** (recorded in DESIGN.md; scripts in `experiments/worlds/review2/`) verified the counts, the certification and the hold-out arithmetic. It found five blocking issues:
1. the best-first evaluation estimator fails when probability spreads over equivalent paths;
2. the library is the generator itself, so the dynamic arms' advantage on P2 and P3 is partly by construction;
3. the held-out patterns can still be emitted by the dynamic arms;
4. sampled marginal likelihood faces 97% spurious hits at initialization and a constant-0 attractor;
5. the memorization index ρ sits at chance.

The review also raised a novelty point: dynamic module graphs against static networks, trained from answers, were compared before (Johnson 2017, Hu 2017, Bahdanau 2019), and learned layouts did not generalize better.

**Decisions:**
- **D50.** Record the review.
- **D51.** Items 2 and 3 change what the round can claim and go to the researcher. The others are fixed in revision 3.
- **D52.** Correct the comment in `branch_classify.py`: merging inputs is sufficient for unreachability, not necessary.

## Entry 33 (26 Oct 2026): giving the static PFN the same primitives (discussion)

**Premise.** The researcher pointed out that revision 2's static PFN has to learn mod-7 arithmetic while the dynamic arms are handed the primitives, which confounds two differences: access to the primitives, and static averaging vs a committed path. A static model should be able to use the same symbolic primitives.

**Argument.** The static counterpart is **superposed execution**:
- T layers. In each, all 33 primitive tables are applied to a belief over the 49 register states of every input.
- Each layer's mixing weights are computed from the context by the same encoder.
- The output is the final distribution of r0.

Its topology is fixed and it never commits. Its prediction is exactly a mixture over paths whose distribution factorizes across steps, so it trains on exact cross-entropy. A variant conditions each layer's weights on the current belief at the context inputs, which gives execution feedback while keeping the topology fixed.

This holds the library and depth fixed and isolates commitment and step-correlated path distributions. A factorized mixture cannot represent "program A or program B" without the junk cross-terms A₁B₂. So any dynamic advantage is predicted where the context is ambiguous, or where it must choose a composition it never saw. Where the world is already identified, a deterministic soft path is a committed path.

The round becomes a factorial: library (learned arithmetic vs given primitives) × composition (superposed vs committed) × feedback (open vs execution).

**Prior work on module networks**, read for this:
- With the right layout given, modular networks generalize systematically (Bahdanau 2019, SQOOP).
- Learning the layout from answers alone is where it fails. IEP and N2NMN bootstrapped from program supervision. In SQOOP, induced layouts often failed to converge to tree layouts, and soft parametrizations blurred.
- CLEVR-trained models, including an NMN given ground-truth programs, fail CLOSURE's novel combinations.
- Relaxed program search loses to discrete solvers (TerpreT).

**Decisions.**
- **D53.** Revision 3 replaces "S vs O/E" with the factorial above. The original S stays as the no-library reference.

## Entry 34 (26 Oct 2026): why a dynamic graph, stated as separate claims (discussion)

**Premise (researcher).** A dynamic graph makes effective computational depth scale combinatorially with the number of primitives and the program length, while a static graph's depth is just its layer count, plus whatever hypotheses each layer can hold in its own space. Dynamic depth lets compute go where it is needed. Reusable primitives avoid training astronomically many unique layers: they serialize into effectively deep models.

**Argument (the motivation split into four claims, each testable separately):**
1. **Combinatorial family, linear cost.** What scales combinatorially is the number of distinct deep computations available (|primitives|^L), at linear parameter cost and linear per-token compute. The depth of any one executed computation is still L.
   - Static nets also contain combinatorial path counts: a residual net with L blocks has 2^L paths (Veit 2016), but it sums them in superposition rather than selecting them.
   - The real axis is superposition vs selection: the composition axis of the factorial (Entry 33).
2. **Compute allocation.** Steps used should track the depth a world needs. This requires a per-step cost or length prior. Without one, nothing favors short programs: this is the review's tie-break finding, which turns out to *be* the compute-allocation objective (shortest program, MDL). On the toy it is measurable exactly: steps used against minimal length ℓ.
3. **Reuse across depth.** One library is serialized at every depth (unlike per-layer experts), so each primitive is trained by every path through it. In neural form this needs a shared interface between primitives (norms, representation drift). The symbolic toy assumes the interface away.
4. **Depth for representation.** On Z_7 with 49 inputs, every function is a 49-entry table, and composing maps on 49 states has shallow shortcuts (parallel prefix, as in Liu 2023). The toy therefore *cannot* test depth's representational advantage, only inference over compositions and compute allocation. Testing it needs an unbounded input, such as a world program iterated over a variable-length sequence (state tracking), where a fixed-depth static model must fail as the sequence grows (TC⁰ limits; non-solvable groups).

**The open risk.** The combinatorial family is only as good as the selector's inference over it. The selector is itself a static network, iterated once per step. Module-network results locate generalization failures there.

**Decisions.**
- **D54.** Revision 3 adds a per-step cost or length prior (which serves as both tie-break and compute-allocation objective), and steps used against ℓ as a metric.
- **D55.** The representational-depth claim is recorded as untestable on the current toy, and is deferred to a sequence-input domain.

## Entry 35 (26 Oct 2026): worlds design revision 3, and its independent review

**Premise.** Revision 3 ([worlds/DESIGN.md](../experiments/worlds/DESIGN.md)) implemented the researcher's decisions:
- a factorial: no-library PFN, superposed programs (SO, SE) and committed paths (CO, CE) over shared primitives;
- one held-out objective for all arms;
- a step cost;
- the pattern masked in the library arms;
- 6 cells and 386 runs.

The derivations in `experiments/worlds/rev3/` gave three numbers:
- superposition's best-case gap to Bayes: 0.02–0.05 nats at k ≈ 1–3, and 25–30% local optima at identified contexts;
- the initial training signal under each scoring rule;
- where the memorization weight is defined.

**The independent review** (`experiments/worlds/review3/`) found:
1. **The headline does not isolate commitment.** Under log loss, a committed arm's predictive is itself an output-space mixture over sampled paths, and a *static* mixture of 8 factorized chains closes most of the §5.1 gap. The comparison measures factorized vs correlated path distributions, a known mean-field limitation; TerpreT's FMGD is essentially SO.
2. **My §5.1 derivation scored the KL on all 49 inputs,** context included, unlike the objective. Objective-matched, the gap is 0.01–0.04, so P3's threshold sits at the bound.
3. **A per-context committed learner** using the design's estimator found **0 of 30** targets of length 3–5. The fallbacks failed too, and §5.2's "exact paths" were almost all coincidences on 1–3 points.
4. **The step cost** enters the two arm types differently, so "one objective" was false.
5. **The Monte Carlo evaluator** is biased by up to +0.56 nats where the model's probability on the true value is small, which is where P5 is tested.

**Decisions:**
- **D56.** Record the review.
- **D57.** Findings 1 and 3 go to the researcher before revision 4:
  - what "commitment" can mean when the predictive is a mixture either way;
  - how the committed arms are trained, if search from scratch fails.
- **D58.** Lesson L20 (derive under the objective's own scoring), and L21 (derive trainability for every arm, not only the one expected to lose).

## Entry 36 (26 Oct 2026): the reframe, per-token routing through a shared library (discussion)

**Premise (researcher).** The task structure has been wrong. What is meant:
- an ordinary autoregressive sequence model in which *every token's* path through the model is chosen dynamically from the preceding tokens;
- a library of primitive layers stores information and computation for all token types;
- context-dependent routing composes them per token, and serialization lets the model represent information in ways a static graph cannot;
- the route depends on the context, not on a "world" in the sense used so far;
- training is the standard next-token loss over the whole sequence, not supervision on an answer.

**Argument.**
- **What revisions 1–3 did instead.** The path was a *hypothesis*: a program over the task's own registers, whose output was read off as the prediction. That is what created the commitment-vs-averaging problem (review 3, finding 1) and the hit-or-miss credit (review 3, finding 3).
- **In the researcher's framing, the route is a computation over the model's hidden state.** The prediction comes from the final hidden state through a softmax. The route at an output position implements *inference*: it maps the context and current token to a predictive distribution. Averaging over worlds happens in the representation, as in a PFN; what is dynamic is the computation graph. The commitment question dissolves.
- **The sparse-credit failure came from symbolic exact primitives, not from sequence training.** With neural primitives, a wrong route still gives a graded loss, and gradients reach the primitives along the chosen route. The router's discrete choice still needs credit assignment, as in mixture-of-experts routers (gate-weighted outputs), with the known risks of collapse and load imbalance.
- **On the information argument.** In the mod-7 stream, x and y tokens are uniform, so the informative targets are still the z tokens. What changes the learning signal is the graded, differentiable loss, not the number of targets.
- **A tension with the earlier decision to use symbolic primitives.** Routing over hidden states needs primitives that act on hidden states. Symbolic primitives over a symbolic state bring back exactly the 0/1 credit problem, unless routing is soft.
- **The structural question is a 2×2:** reuse of layers across depth (no/yes) × per-token selection (no/yes).
  - neither: a static transformer;
  - selection without reuse: per-layer mixture-of-experts;
  - reuse without selection: universal or looped transformers;
  - both: the researcher's library routing.
- **Closest prior work:**
  - MoEUT (Csordás 2024: shared layers across depth with mixture-of-experts; slightly better than parameter-matched transformers);
  - Mixture-of-Recursions (Bae 2025: per-token recursion depth over a shared block; better perplexity at equal FLOPs);
  - Neural Interpreters (Rahaman 2021: tokens routed through shared functions, trained end-to-end);
  - SUT; Mixture-of-Depths; routing networks.

  None, to our reading, measures per-token library routing against exact Bayes on a generator with known compositional structure.

**Decisions.**
- **D59.** Revisions 1–3 (path = program over task registers) are superseded as a test of the researcher's idea. They remain a valid but different question: amortized program induction. Revision 4 is to be redesigned around per-token routing through a shared library, trained on the full-sequence next-token loss, after the researcher settles neural vs symbolic primitives.

## Entry 37 (26 Oct 2026): correction to Entry 36, symbolic primitives are not the problem

**Premise.** The researcher challenged Entry 36's claim that per-token routing needs neural hidden states: symbolic primitives are abstract functions and compose exactly as layers do.

**The challenge is right. Entry 36 conflated three separate properties:**
1. **Composition.** Identical for symbolic and neural primitives.
2. **A state that can hold mixtures between steps.** This is what gives the router a gradient. In mixture-of-experts, a gate-weighted sum of expert outputs is meaningful because the state is a vector. A symbolic state held as a *distribution over symbols* has the same property; that is the superposed belief of SO/SE. With a single exact symbol as the state, routing must be sampled and credit is hit-or-miss (review 3, finding 3), whether the primitives are symbolic or neural.
3. **Primitives that can read the context.** In revisions 1–3, the primitives acted only on the current pair's registers, and the context entered only through the choice of route. So the route at a z position *had* to compute f(x, y): it was a hypothesis about the world. For the route to be an inference procedure, primitives must be able to read other tokens, e.g. "find the context pair with this x and copy its z". Symbolic primitives can do that too.

**Neural primitives additionally make the primitives learnable.** That is a separate choice about what the experiment studies.

**Decision.**
- **D60.** Entry 36's "neural vs symbolic" tension is withdrawn. The design question for revision 4 is (a) whether the inter-step state holds mixtures (soft or top-k gated routing) or single symbols (sampled routing), and (b) a primitive set that includes context-reading operations.

## Entry 38 (26 Oct 2026): exact per-step progress as the router's signal (discussion)

**Premise (researcher).** On the toy we know a priori the branching space toward the next token. So we can tell, for each selected primitive, whether it constrains the prediction and moves optimally along the shortest path. That gives the meta-model a clear reward at every serial step.

**Argument.**
- **What it is.** An exact optimal value function: V*(state) = −(remaining distance). The per-step progress is a potential-based shaping reward (Ng 1999), which leaves the optimal policy unchanged. Better still, the oracle can be queried at *any* state, including states off the optimal path. So the router can be trained by imitating the oracle on the states it actually visits, as in DAgger: cross-entropy on the set of optimal next primitives. This replaces sparse, sampled credit with dense, exact credit.
- **Caveat 1: the target must be what the context makes knowable, not the true token.** An oracle that knows the true z never needs to read the context. Imitating it would teach the router never to use context-reading primitives: the privileged-expert imitation gap. Distances must be measured to the *version space*, the states that predict what the context entails. Reading the context is then part of the optimal path.
- **Caveat 2: under ambiguity the target is a set.** Several worlds remain, each with its own shortest paths. The right per-step target is the posterior-weighted share of optimal continuations (π*), or progress toward the version space. With equivalent programs, train on the whole *set* of optimal actions, not one of them.
- **Caveat 3: what the claim becomes.** Oracle-trained routing answers "can dynamic routing represent and execute efficient inference?" (expressivity), not "can a learner discover routing from data". For a comparison with static models, give them the matching dense target (intermediate states along optimal paths), or label the oracle arm as privileged.
- **The transferable piece.** Real data has no oracle. A *learned* value model, trained on the exact distances of training worlds and tested on whether its estimate of "does this step move toward the answer" transfers to unseen worlds and longer programs, is a model that "sees the minima on the horizon". The toy makes that both trainable and exactly checkable.

**Decision.**
- **D61.** Revision 4 treats exact per-step progress three ways:
  - as a diagnostic, in every arm;
  - as a training signal, measured against the version space, for an oracle-imitation arm labeled as privileged, with matched dense targets for the static arms;
  - as the training target for a learned value model whose generalization is itself an endpoint.

## Entry 39 (26 Oct 2026): what Neural Interpreters and AlphaZero say about the per-token routing design

**Premise.** The researcher asked what the two papers imply for the design of Entries 36–38: per-token routing through a shared library of symbolic primitives, with context-reading primitives and exact per-step progress. Both were read in full; notes are in [notes_neural_interpreters_alphazero.md](notes_neural_interpreters_alphazero.md).

**Lessons from Neural Interpreters** (soft, per-token routing through shared functions, reused across depth: the researcher's quadrant):
1. **Re-routing a frozen library recovers almost everything.** On the fuzzy-Boolean toy, R² is 0.986 when only routing is tuned, against 0.920 for the output tokens alone and 0.995 for everything. This is direct support for "the library stores computation; routing composes it", and it suggests a routing-only adaptation endpoint.
2. **Collapse is the main failure mode, and freezing signatures at a random, diverse initialization is a cheap remedy.**
3. **Context-reading lives inside primitives,** as attention gated by both tokens' compatibility with the function. That matches the requirement of Entry 37.
4. **Routing is soft** (a compatibility-weighted sum), which is how its router gets gradients without a credit-assignment problem.
5. **Out-of-distribution gains are modest.** The hard PGM splits stay near chance: 20.0% on held-out triples and 19.4% on extrapolation, against 12.5% chance. Expected effects should be calibrated accordingly.
6. **Function–primitive correspondence was never verified,** only inferred from plots. Our exact toy can verify it.

**Lessons from AlphaZero:**
1. **Policy plus value plus search is the answer to sparse credit without an oracle.** Search turns a sparse outcome into dense, improved policy targets: the router is trained toward visit counts, the value toward outcomes.
   - On our toy, the outcome can be verification on the *context* (whether the partial computation is consistent with what the context entails), so no privileged information is needed.
   - The exact oracle distances then evaluate the learned value, plus a labeled privileged arm.
2. **Symbolic primitives give a free, exact simulator,** as game rules do for AlphaZero. Lookahead over 33 branches at depth ≤ 7 is cheap. This supports the researcher's symbolic choice.
3. **Average value errors; don't maximize them.** Subtree averaging cancels value errors, while max-propagation amplifies them.
4. **Exploration noise scaled to the branching factor** is the standard remedy for the premature collapse review 3 found.
5. **A learned prior and value make search selective:** 1,000× fewer positions than Stockfish, and better scaling with thinking time. This is the "see the minima on the horizon" mechanism, measurable on our toy as policy-only vs with-search performance, the "seeing vs searching" gap.
6. **Caveat: AlphaZero's success rests on enormous self-play volume.** Our tree is tiny, but per-token search during training multiplies cost by the number of simulations.

**Decisions:**
- **D62.** Revision 4 considers two training regimes for the router:
  - soft compatibility-weighted routing (Neural Interpreters);
  - hard routing trained by expert iteration (AlphaZero), with a policy and value head per token, short lookahead over primitive branches using exact symbolic execution, and outcomes judged by consistency with the context.
- **D63.** Endpoints to add:
  - routing-only adaptation to held-out worlds;
  - verified function–primitive correspondence;
  - the gap between the learned policy and search;
  - the learned value's agreement with exact distances on unseen worlds.
- **D64.** Calibrate predictions to modest out-of-distribution effects. Name AlphaDev and AlphaTensor (AlphaZero over instruction-level programs with exact execution, searching per task) as the nearest program-search precedent. Our difference is amortization across worlds in context.

## Entry 40 (26 Oct 2026): a PFN over model composition (discussion)

**Premise (researcher).** A traditional PFN meta-learns at the level of the data set: it amortizes the posterior predictive over *outputs*. The proposal is to meta-learn at the level of *model composition*: amortize the posterior over how the model composes its primitive layers for each token. The model knows from context where it wants to go structurally, but cannot know the output of a composition before running it. A possible analogy to P vs NP.

**Argument.**
- **The decomposition.** This is the Solomonoff decomposition made explicit. The universal prior is a prior over *programs*, and the prediction is what those programs produce. A PFN approximates the induced predictive directly. A composition-level PFN approximates the posterior over compositions and lets execution produce the prediction.
- **Same optimum, different extrapolation.** On the training support, both have the same optimum under log loss. They differ in what must be learned: the PFN must absorb every function's input-output behavior into its weights, while the composer only learns *which* composition, because execution supplies the behavior. Off support, the composer can give mass to compositions it has never seen whole, whose parts still execute correctly.
- **Knowledge separates into two places:** *what* the primitives compute lives in the shared library, and *how to combine* them, the prior, lives in the router. A PFN puts both in one set of weights.
- **The analogy sharpened into two ideas:**
  1. **Find vs verify (the P vs NP intuition).** Finding a composition consistent with the context is search in an exponential branch space. Checking a candidate against the context examples is one cheap execution. The meta-model amortizes the finding, as a learned heuristic, and execution does the checking. It exploits the structure of the training distribution (the average case), so it says nothing about worst-case P vs NP.
  2. **Structure vs output (computational irreducibility).** For general compositions, the only way to know the output is to run them. So the model commits to structure, and the output is revealed by execution.
- **Measurable on the toy.** The structure-level analog of "PFN vs Bayes" is the router's distribution over compositions against the exact posterior over compositions, taken over equivalence classes of compositions that compute the same thing.
- **The caveat.** The amortized structure posterior is still a learned network, and whether its support extends to unseen compositions is exactly what module networks failed at when layouts were learned.

**Decision.**
- **D65.** Revision 4 states the round's thesis as "a PFN over model composition". The primary structure-level endpoint is the router's composition distribution against the exact posterior over composition equivalence classes, alongside the output-level code length against exact Bayes.

## Entry 41 (26 Oct 2026): revision 4 judged not worth running as designed

**Premise.** Revision 4 ("a PFN over model composition") had five arms:
- N, an output-level PFN;
- CP, a composition PFN trained on prior samples;
- CX, expert iteration with exact symbolic search;
- CS-1 and CS-8, soft routing.

It also had the search-alone foothold (`rev4/mcts_search.py`: ℓ = 1 12/12, ℓ = 2 9/12, ℓ = 3 2/12, ℓ ≥ 4 0/12 at 1,024 simulations) and a prior-work table. The researcher asked for a review of the design *and* of its experimental value.

**The review** (DESIGN.md; `experiments/worlds/review4/`):
- On pair worlds with the generator's own library, the posterior over composition classes *is* the posterior over output tables, and exact inference is a 5 ms lookup. So the thesis's structure-vs-output distinction is zero by construction, and the round reduces to amortized program induction over a known DSL, already well studied:
  - inference compilation, NeSymReS, DreamCoder and memoised wake-sleep (for CP);
  - REPL synthesis, AlphaNPI and CodeIt (for CX).
- CX's training rule doesn't have Bayes as its optimum.
- The soft-arm predictions were decided by derivation, scored on marginals again (L20).
- The search cost needs a GPU-batched implementation that was never specified.
- Suggested sequence: a learned-value test on CPU; CP vs N; a CX bootstrap pilot; then Stage B or a mismatched library.

**Assessment.** The central point is correct, and it was available earlier: `generator_design.md` §11 had called the composer "a positive control". Testing the thesis needs a world where the posterior over structure carries information the output posterior does not, and where exact inference is not a cheap lookup:
- unbounded or sequence inputs (Stage B);
- or a library whose basis differs from the generator's.

**Decisions:**
- **D66.** Record the review. The 331-run grid is not run.
- **D67.** Lesson L22.
- **D68.** Direction goes to the researcher: the suggested sequence, or straight to designing Stage B.

## Entry 42 (26 Oct 2026): dropping Z_7, and what input space the idea needs (discussion)

**Premise (researcher).** Scrap mod 7. Assume abstract, independent primitives that compose serially:
- overdefined (many compositions give the same output);
- fully expressive (any token can be mapped to any token).

A transformer trains matrices to produce context-dependent next-token transformations. Here, the *selection* is trained: given teacher forcing, how to compose primitives for the query token so as to produce the next token. Which data structure and input space suit this?

**Argument.**
1. **A new alphabet alone changes nothing.** What matters is a set of properties (below), and the reviews and L22 show which ones.
2. **Primitives acting only on the current token make composition degenerate.** If the state is just the token (n values), a composition is a walk on an n-node graph. With routing per token and teacher forcing, any walk to the observed next token fits, so the route encodes nothing beyond its destination. At test the router would need to know the answer first.

   Composition is meaningful only when the primitives transform a state *richer than the output*: current token plus values read from the context, as a transformer's residual stream is much larger than its logits. A route is then a *procedure*, reusable at other positions and in other contexts, and the output is a projection of the transformed state.
3. **Properties the input space needs:**
   - **P1.** Data generated by serial compositions of primitives: overcomplete, and fully expressive. The full transformation monoid on n tokens is generated by 3 maps (two generators of S_n plus one rank-(n−1) idempotent), an exact instance of "any token to any token".
   - **P2.** The same procedure is reused across positions with different inputs, and the per-token depth varies, so that knowing the composition is worth more than predicting one output.
   - **P3.** A regime where fixed-depth static predictors are known or provably limited, with a recurrence-matched static baseline, so the advantage isn't recurrence alone.
   - **P4.** Exact references (Bayes over procedures by enumeration; branch values by graph search).
   - **P5.** No cheap table lookup: the input is the whole prefix, unbounded.
4. **A candidate: sequence-program worlds.** Primitives in the style of RASP:
   - reads and selects from the context (attention-like);
   - elementwise token maps;
   - counting;
   - each episode a random short program, with sequences generated by running it.

   The route at each position is the program's per-position computation. A simpler variant: an in-context lexicon of named compositions (meanings shown by their effects, nestable), acting on a carried state.
5. **Prior work for these worlds:**
   - RASP (Weiss 2021) and Tracr;
   - Learning Transformer Programs (Friedman 2023: discrete RASP-like programs learned inside a transformer, the nearest architecture);
   - the RASP-L conjecture (Zhou 2023: transformers length-generalize when the task has a short RASP-L program);
   - shortcuts to automata (Liu 2023);
   - DeepCubeA (Agostinelli 2019: a learned cost-to-go guiding search, precedent for Entry 38's learned value).

**Decision.**
- **D69.** Before any arms are designed (L22), the next design step computes the distinguishing quantity for the candidate worlds:
  - how much Bayes over procedures beats a lookup or marginal baseline;
  - at which lengths fixed-depth and recurrence-matched static predictors are limited.

## Entry 43 (26 Oct 2026): the informational lens on attention plus routed primitives (discussion)

**Premise (researcher).** Use ordinary attention over the token sequence to build a context-dependent query state, which then goes through a routed composition of primitives that outputs the ideal next token.

**Argument.**
- **Two jobs.** Attention gathers: it is where Shannon information about the next token enters, from the prefix. Composition computes: it turns gathered information into the prediction.
- **Data processing.** Primitives cannot add information. Whatever reaches the output came from the query state or from the route choice. The route choice is itself a channel of at most T·log₂|library| bits, set from the context. So the query state must be a sufficient statistic of the prefix for the next token.
- **What composition adds is usable information.** In the sense of V-information (Xu et al. 2020, information extractable by a constrained function class), routing enlarges the decoder's function class from one fixed head to compositions. Composition matters exactly when the information is present in the query but not extractable by the fixed part.
- **Dependent reads.** One attention pass before routing fixes what can be read before any computation happens. Multi-hop dependencies (read A, then use A to read B) need sequential reads. So a hybrid fits best: attention for broad gathering, plus read primitives *inside* routes for dependent gathering. Neural Interpreters puts attention inside its functions.
- **The shortcut risk.** If the static attention stack can compute the answer itself, the routes become decorative (Entry 42 again). The output head must read only the route's final state, the static part's computation must be limited, and ablations must show where the computation actually happens.
- **Symbolic primitives need a bridge.** They require discretizing the query state into registers: a neural front end and a symbolic back end. The discrete bottleneck may help systematic generalization but can lose information.
- **Teacher forcing gives at most log₂|V| bits per position about a route space of size |library|^T.** Routes are identified only through regularities reused across positions and contexts, as a grammar is learned from strings.
- **Information metrics to measure:**
  - sufficiency of the query state (a probe for the Bayes predictive);
  - usable-information gain from routes over the best fixed decoder;
  - mutual information between routes and the latent procedure;
  - route description length against the shortest generating program.

**Decision.**
- **D70.** The candidate architecture: attention front end, then a routed composition that includes read primitives, then an output read only from the route's final state. It comes with ablations that locate the computation, and the information metrics above.

## Entry 44 (27 Oct 2026): the wall, primitives must be generic context circuits (discussion)

**Premise (researcher).** The toy obscures what we want to measure. The primitives should be things like attention context lookups and their compositions, not independent computational operations in a branching space. The toy made the primitives specific to the hypothesis, which bypasses the generic context circuits transformers learn. That is why it keeps falling apart.

**Argument (agreed).**
- **Every revision so far took the primitives from the generator's own basis** (add, mul, …). So composing primitives *was* enumerating hypotheses about the world: program induction, with the library equal to the generator (reviews 2–4, L22).
- **Transformers divide the labor differently:**
  - generic, task-agnostic circuits route information (previous-token, copy, induction, select-by-match, aggregate, count, compare);
  - task-specific content lives in the context (keys, values, mappings shown in-context) or in memorized lookups.

  Induction heads, for example, arise from *composing* a previous-token head with a copying head (Elhage et al. 2021, K-composition).
- **So the library should be generic context circuits** (RASP-like select/aggregate with a small set of predicates, positional shifts, matching, counting, a few generic elementwise ops). Routing should compose them per token. Specific content must be read from the context, not built into a primitive.
- **The data should not be generated by the library itself.** It should come from diverse sequence processes (formal languages, automata or state tracking, associative recall, in-context mappings, multi-hop lookups, interleaved tasks). The library's sufficiency is then a checkable fact, as in RASP-L (Zhou 2023), not true by construction.
- **What gets measured:** whether composing generic circuits *dynamically per token* beats a static arrangement of the same circuits (a fixed-layer transformer, or a fixed program) where the needed circuit chain varies by token and by context: k-hop lookups with varying k, interleaved tasks, length growth.
- **Exact references** survive for processes with known posteriors, such as known programs or automata. Branch values become distances in circuit-program space.
- **Prior work:**
  - transformer circuits and head composition (Elhage 2021; Olsson 2022);
  - RASP, Tracr, Learning Transformer Programs;
  - per-token head routing within fixed layers: Mixture of Attention Heads (Zhang 2022), MoH (Jin 2024);
  - Neural Interpreters (attention inside routed functions);
  - Zoology/MQAR for recall circuits.

  None composes generic circuits in a per-token *serial* route measured against exact references, to our reading.

**Decision.**
- **D71.** The Z_7 program-induction toy is retired as a test of the thesis; it is kept only as a solved positive control. The next design starts from a generic context-circuit library and data from diverse sequence processes that are not written in the library. The L22 check comes first: the distinguishing quantity (dynamic vs static circuit composition) must be non-zero and measurable.

## Entry 45 (27 Oct 2026): what the toy is for, working backward from the end system (discussion)

**Premise (researcher).** The toy exists to build knowledge on the smallest part of the end system we cannot answer now: to make observations, build intuitions and explore on a simple problem. It must be designed so it actually probes the questions about the end goal, a neural, serialized LLM.

**The end system.** A shared library of generic primitive layers (attention circuits and MLP-like transforms). A per-token router chooses, from context, which primitive comes next or whether to halt. It is trained autoregressively, possibly with a learned value and search.

**Open questions about it, and what is already known:**

| question | already known | open |
|---|---|---|
| Does per-token routing train at all? | Yes, in soft or top-k form (MoE, MoA/MoH, Neural Interpreters, MoR) | — |
| Can depth vary per token over a shared block, with learned halting? | Yes (MoR, Universal Transformer/ACT) | — |
| **Q1.** From next-token loss alone, does a router learn *order-dependent serial chains of distinct generic circuits*, where the chain needed varies by token and context? | — | open |
| **Q2.** Do learned chains generalize to longer chains and new orders of known circuits, better than static stacks, looped blocks and per-layer head selection? | Module networks warn that learned layouts don't | open |
| **Q3.** Can arbitrary orders of neural primitives act on one shared state without drift? | — | open: the interface |
| **Q4.** Do generic primitives form and stay generic when learned jointly with routing? | — | open: library formation |
| **Q5.** Can a learned value or search make routing selective? | — | open |

**A constraint from theory.** Static transformers solve k-hop induction in ⌊log₂k⌋ + 2 layers by parallel doubling. Trained depth-L models reach k ≈ 2^(L−2) (Sanford, Hsu & Telgarsky 2024), and associative compositions such as state tracking have parallel-prefix shortcuts. So serial routing is unlikely to win on expressivity or parallel compute in these tasks. Its possible advantage is learnability and generalization: one primitive, iterated as needed, extrapolates in k and to new orders, while circuits fixed at a given depth do not.

**Fidelity requirements for the toy:**
- **F1.** Generic, hypothesis-agnostic primitives (Entry 44).
- **F2.** Per-token routing in an autoregressive sequence, with teacher-forced next-token loss as the signal.
- **F3.** Chains that depend on order, and that vary by token and context.
- **F4.** A state richer than the output, with the output read only from the route's final state.
- **F5.** Baselines matched on parameters and per-token compute: a standard transformer; a fixed stack of the same primitives; a looped block (reuse without selection); per-layer head selection (MoA/MoH); a soft superposed stack.
- **F6.** The ground-truth chain known for every token, so route correctness and branch classes are exact.
- **F7.** The chain not inferable from the token alone.

**What can be abstracted away:** natural language, scale, a large vocabulary, and, at first, learned primitives.

**Decision.**
- **D72.** A ladder of toys, each isolating one unknown:
  1. symbolic generic circuits with routing only (Q1, Q2);
  2. neural primitives initialized or distilled from the circuits (Q3);
  3. primitives learned jointly with routing (Q4);
  4. learned value and search (Q5);
  5. more natural data.

  The first rung: sequences needing mixed, order-dependent chains of generic circuits (e.g. match-and-read-next, previous token, copy, select-by-predicate), with chain length and order varying per query, and the chain inferred from context. Trained on short chains; tested on longer chains and new orders.

## Entry 46 (27 Oct 2026): residuals as the signal for composition and depth (discussion)

**Premise (researcher).** Learn the PFN over model composition by looking at the residuals of layers, and generalize at test time by monitoring them. What "residual" means is open. The idea is hierarchical refinement and pathfinding, so a token can signal through its residual that it needs more depth.

**Argument: four candidate meanings of "residual".**
1. **The update.** The change a primitive makes to the token's state, ‖Δh‖ (convergence and fixed-point views; DEQ).
2. **Prediction change.** The shift in the next-token distribution between steps (logit-lens stability; the researcher's earlier "unstable between layers" intuition; CALM-style consistency exits).
3. **Error.** Target minus prediction. It is known at training time (teacher forcing) and unknown at test time, so at test time it must be *estimated*. A learned estimate of remaining loss is exactly the value / cost-to-go of Entries 38–39. Stagewise fitting of residuals is the boosting view of residual networks.
4. **Uncertainty.** The entropy of the current prediction.

**The crucial distinction.** Some residual is *irreducible*: the context genuinely leaves the next token uncertain, and more depth cannot help. The rest is *computational*: the answer is determined by the context but not yet computed. Only the second should earn more depth.
- Entropy and confidence mix the two. Small update norms can mean "stuck" as well as "done". A stable prediction can be stably wrong.
- The right definition is the reducible gap: the KL from the current prediction to the Bayes predictive.
- On an exact toy it is computable. Elsewhere it can be approximated by the gap to the model's own deepest prediction, or learned as a value.

**Routing by residual.** The residual's *direction* can address the library: a primitive is chosen by matching the token's current error representation to primitive signatures (Neural Interpreters-style type matching). Primitives learn which residuals they reduce. Coarse-to-fine refinement follows if large reductions come first.

**The myopia risk.** Some chains need preparatory steps that reduce nothing by themselves, like stepping back before matching. A router that is greedy on immediate residual reduction fails there; it needs a cost-to-go (lookahead or a value). The rung-1 toy's mixed chains contain exactly such steps.

**Prior work:**
- ACT (Graves 2016) and PonderNet (Banino 2021);
- Universal Transformers;
- CALM (Schuster 2022);
- early-exit and LayerSkip;
- logit lens and tuned lens (Belrose 2023);
- DEQ (Bai 2019);
- residual networks seen as boosting;
- predictive coding (prediction errors drive processing level by level).

**Decision.**
- **D73.** Rung 1 adds a residual study:
  - per step, measure the candidate signals (update norm, step-to-step KL, entropy, learned remaining-loss value) against the exact reducible residual (KL to Bayes) and against the true remaining chain length;
  - compare halting and routing policies (fixed depth; norm threshold; stability; confidence; learned value; residual-matched routing; greedy vs cost-to-go) against the oracle;
  - test specifically on chains with preparatory steps.

## Entry 47 (27 Oct 2026): setup moves, delayed reward and multi-layer circuits (discussion)

**Premise (researcher).** How does AlphaZero handle moves whose payoff appears only after several more actions, where a single move looks weak but sets up the rest? The same happens in circuits: three sequential layers may be needed for a behavior while no single layer moves the needle.

**Argument.**
- **AlphaZero has no per-move reward.** Only the final outcome exists.
  - A move is judged by the value of the position it leads to, v(s) ≈ the expected final outcome. That is cost-to-go, not immediate gain. A setup move is good because the resulting position is good.
  - v is trained on the final outcomes of self-play games, so positions full of potential learn high value.
  - Search expands follow-up sequences and averages deep values back to the root. Setups within the search horizon get credit directly.
  - The policy is distilled from search visit counts, so setup moves become intuitive over iterations.
  - Root noise and sampling find them in the first place.
- **AlphaZero's limits:**
  - combinations beyond the search horizon rely on v's intuition (horizon effects; ladders were a known weakness of AlphaGo-style systems);
  - learned values have exploitable blind spots (adversarial policies against KataGo, Wang et al. 2023).
- **The key difference from our setting.** In games every random playout ends in a win or a loss, so a learning signal exists from the start. In composition, a random route almost never reaches a target that needs a specific multi-step chain (the needle in Entry 41's foothold table). Remedies:
  - curriculum through short chains;
  - potential-based shaping from a value;
  - hindsight relabeling (CodeIt, HER, DreamCoder dreams): any route computes *something*, so treat it as a correct example for its actual output. For sequence models this is valid for learning route→behavior models, but changing the target without changing the context makes inconsistent sequences.
- **How transformers already solve setup credit.** An induction head is a previous-token head, which alone does nothing to the logits, composed with a copying head. Gradient descent credits the preparatory layer through the chain rule, because the composition is fixed and differentiable. The abrupt emergence of induction heads after a plateau (Olsson 2022) is the signature of a multi-step circuit discovered by gradient.

  The difficulty appears only when the composition is a *discrete* routing choice, which gets no gradient. Hence:
  - soft routing gets chain-rule credit for setup steps, like a transformer;
  - hard routing needs a value or search;
  - a hybrid (soft while learning, hardened later, or straight-through estimates) is an option.
- **The residual to monitor is a value over states, not the change in prediction.** After a setup step the prediction is unchanged, but the state is one step closer. On the exact toy this is the distance to the target: a setup step is optimal exactly when the distance drops, even with zero change in logits.

**Decision.**
- **D74.** Rung 1 tests whether learned values over states recognize setup progress (distance drops with unchanged predictions), and compares credit for setup steps under soft routing (chain rule), hard routing with value and search, hindsight relabeling, and a soft-to-hard hybrid.

## Entry 48 (27 Oct 2026): the unifying formalism, amortized EM over routes (discussion)

**Premise (researcher).** The idea circles three things: a PFN / Bayesian posterior over model composition; AlphaZero-style search over the branch space of layers; serialized composition for autoregressive prediction. The formalism joining them is missing, and disproving wrong parts would be as valuable as confirming right ones.

**The formalism.**
- Position t, context c = x_{≤t}, target y = x_{t+1}. State space S, initial state h_0 = embed(c). Primitives ℓ_a: S → S (they may read c). HALT, then a readout p(y | h).
- Route r = a_1 … a_T, with p_r(y | c) = readout(ℓ_{a_T} ∘ … ∘ ℓ_{a_1}(h_0)).
- Router π_θ(r | c) = Π_k π_θ(a_k | h_{k−1}, c).
- **Model:** q_θ(y | c) = Σ_r π_θ(r | c) · p_r(y | c). A latent-variable autoregressive model with the route as the latent. Training loss: −log q_θ over the sequence.

**The three pieces are one object.**
1. **PFN over composition = the amortized route posterior.** On data from a prior over worlds, the optimum of the next-token loss is the posterior predictive p*(y | c). If the library can realize each world's predictive by some route, then π*(r | c) = Σ_w p(w | c) · Unif(MinRoutes(w, c)) attains it. What is learned at the structure level is the map c → posterior over *procedures*. It is non-identifiable from the output loss alone (any π with the same mixture is optimal); a route prior e^(−λ|r|), Solomonoff/MDL, makes it well defined.
2. **AlphaZero = Monte Carlo EM, with search as the E-step.** Maximizing the marginal likelihood is EM. E-step: p(r | c, y) ∝ π_θ(r | c) · p_r(y | c) · e^(−λ|r|): a search over the branch space with terminal reward log p_r(y | c) − λ|r| and the current policy as prior. MCTS with a policy prior and a value head is an approximate E-step (Grill et al. 2020: MCTS ≈ KL-regularized policy optimization against the prior). The value V(s) is the soft cost-to-go, the log-partition over completions from s. M-step: distill the E-step distribution into π_θ(· | c), i.e. the visit-count cross-entropy. In expectation over y ~ p*(· | c), distilling the label-informed posterior p(r | c, y) into a label-blind policy gives p(r | c). Label-informed search at training time is therefore sound: it is the wake-sleep / amortized-inference fixed point, not a leak.
3. **Serialized autoregression = test time.** *Seeing* is π_θ(r | c). *Searching* refines it under a label-free objective, the expected log score E_{y~p*}[log p_r(y | c)] = −H(p*) − KL(p* ‖ p_r). The irreducible residual is H(p*); the reducible residual is the KL; the value head estimates the reducible part.

In one line: **amortized variational EM over per-token routes, with MCTS as the E-step, the router as the recognition model, the value as the partition estimate, and a step-cost prior as the regularizer.** The pieces are standard (planning as inference, Levine 2018; wake-sleep, Hinton 1995, Hewitt 2020; Grill 2020; PFNs, Müller 2022). Per-token model composition with exact references is the new application.

**What could be wrong: six bets, each falsifiable, in the order to test.**
- **B1 (identifiability / MDL).** Without a route prior the learned π spreads over equivalent routes (seen in review 3). The bet: minimal routes generalize better out of distribution than arbitrary equivalent ones. Test: step cost on vs off, measured on longer chains and new orders.
- **B2 (learnability of procedures).** The recognition model must learn order-dependent chains of generic circuits from next-token loss. Soft routing with the chain rule is the cheapest test. If this fails, the thesis fails regardless of search.
- **B3 (the extrapolation claim, central).** The map context → route is simpler than the map context → output, so π_θ should generalize to longer chains and new orders where a static model stops. Module-network history says learned layouts often don't. Test: chain-length and order extrapolation against static, looped and head-selection baselines at matched compute.
- **B4 (depth vs parallelism).** Static models solve k-hop in log k layers; a router with a hop primitive should reach any k at cost k. If the router cannot learn "iterate until match", the depth argument fails.
- **B5 (the E-step needle).** For long chains, random routes rarely hit, so hard routing needs a curriculum, hindsight or a soft warm-start. Test: whether expert iteration bootstraps from short to long chains.
- **B6 (value learnability).** The value must track the reducible residual, including setup steps, on unseen worlds. Test: rank correlation with the exact remaining distance out of distribution. If near zero, search is blind beyond training and "seeing the minima" fails.

**The narrow path.** B2 and B3 first, with soft routing and a step cost, on the rung-1 circuit worlds: cheap, gradient-based, exact references. Only if they pass do B5 and B6 (hard routing, search, value) matter. Soft vs hard buys nothing at the predictive optimum (review 3); what hard routing can buy is per-token compute, coherence and a discrete object for MDL and search, and that is measured at matched compute.

**Decision.**
- **D75.** This formalism is the frame for rung 1: the model is the latent-route mixture, the loss is the marginal likelihood with the step-cost prior, soft routing is the first E-step, and the six bets are the preregistration's hypothesis list in order.

## Entry 49 (27 Oct 2026): the thrust restated as dynamic depth, and the minimal toy (discussion)

**Premise (researcher).** Depth in a static transformer is prohibitive because compute per token grows linearly with layers while the computation each token needs is dynamic. In a 1,024-layer model most tokens barely change through most layers while a few need deep refinement; at that depth residuals are lost (hence residual-injection tricks) and gradients make the depth hard to train, since few tokens and events demand deep computation. Looped transformers share parameters but not compute. The thrust: learn a policy that allocates computational depth per token, a dynamic looped transformer. Define the toy around the most minimal unanswered question.

**What is already known.**
- Per-token recursion depth over a shared block with a learned router exists: Mixture-of-Recursions (Bae 2025; better perplexity at equal FLOPs, recursion depths ≤ 4, no ground truth for demand). Mixture-of-Depths (Raposo 2024) skips blocks per token at a fixed capacity.
- Learned halting exists: ACT (Graves 2016), Universal Transformer, PonderNet (Banino 2021; better extrapolation than ACT on parity). ACT is known to be finicky and to collapse toward constant depth.
- Looped transformers need input injection to stay stable over many loops (Yang 2024), and adaptive loop counts for length generalization have relied on *known* step counts (Fan 2025).

**What is not known (the minimal question).** Whether a depth policy learned from next-token loss plus a compute cost, with no step-count supervision:
1. allocates depth matching the true per-token demand when demand is heavy-tailed, so that deep tokens are rare and carry rare gradient signal;
2. extrapolates to demands deeper than any seen in training;
3. can be driven by an internal signal (the reducible residual, Entry 46) rather than a ponder cost;
4. trains stably through many loops without the residual drifting.

This is rung 0 of the ladder: depth allocation with one shared block, before any selection among primitives.

**The toy: pointer chasing with data-dependent termination.** A sequence holds key→value pairs. A query names a start key; the value is the next key; the chain ends at a terminal marker, whose value is the answer. The chain length k is hidden until the chain is executed; its distribution is heavy-tailed and controlled. Several queries per sequence need different k.
- Required depth per token is exact (k hops), serial by construction, and unknown a priori.
- The only static shortcut is doubling: an L-layer static model reaches k ≤ 2^(L−2) (Sanford, Hsu & Telgarsky 2024). So the static ceiling and the dynamic advantage are derivable before any run (L22): static pays L per token and fails beyond its ceiling; an oracle-halting loop pays k + 1 and has no ceiling.
- Exact references: deterministic answers; remaining hops as the exact branch value; the reducible residual is 0/1 per step.

**Baselines, matched on parameters and on average and maximum compute per token:** a static transformer of depth L; a fixed-loop block; a loop with oracle halting (upper bound, uses ground truth); learned halting with a ponder cost (ACT/PonderNet); residual-triggered halting (update norm, prediction stability, learned value); an MoR-style router. Input injection on and off.

**Endpoints:** accuracy by k; depth used against k (calibration slope and excess); extrapolation to k beyond training; accuracy on rare deep tokens against their frequency; compute per token; trainability against loop count (gradient norms through loops).

**Falsifiers of the thrust:**
- learned halting collapses to constant depth (no compute saved);
- rare deep tokens are under-served because their gradient is rare, in which case "few tokens need deep computation" is the disease rather than the motivation;
- no extrapolation beyond the training depth, so dynamic depth buys nothing a static stack cannot;
- the doubling static model matches up to its ceiling at far less serial cost, which is expected: the win must appear beyond the ceiling and in average compute.

**Decision.**
- **D76.** Rung 0 is pointer chasing with hidden, heavy-tailed chain length and a single shared block with learned halting. The distinguishing quantity (static ceiling 2^(L−2) and compute L vs k + 1) is derived, nonzero and exact. Primitive selection waits for rung 1.

## Entry 50 (27 Oct 2026): two intuitions made into predictions, signal pooling across depth and a world-agnostic routing signal (discussion)

**Premise (researcher).** Two intuitions:
1. there is a residual or signal in the prediction task on which a router can learn a PFN that decides routing and generalizes across worlds;
2. the useful layers can be treated as composable functions and the desired behavior elicited by composition, instead of needing gradient signal to learn 1,024 independent layers from few examples. Serial composition of fewer unique layers has the effect of a very deep model with unique layers.

**Intuition 2 as a learning-signal argument.**
- In a static model of depth D, layer d's parameters are trained only by tokens whose computation is still changing at depth d. Under heavy-tailed demand those are rare at large d, so deep layers get little informative gradient: data per deep parameter is scarce.
- With m shared primitives composed over T steps, primitive p receives gradient from every token that uses it *at any depth*. The signal pools across depth: data per parameter grows by roughly the number of depths at which the primitive is used.
- The *expressive* equivalence holds only under a compositional assumption: the deep computations needed are compositions of a small basis (m^T distinct deep computations from m primitives). Evidence that real transformers are partly like this: cross-layer sharing (ALBERT, Universal Transformer, MoEUT) keeps most quality at the same depth with far fewer parameters, with a modest loss at matched parameters; middle layers are partly redundant (layer-pruning results, e.g. Gromov et al. 2024 removing a large fraction of layers with small loss). The honest status: sharing across depth is established; the gain is in parameters and length generalization, not in-distribution quality.
- Where it fails: when different depths genuinely need different functions (early detokenization-like work, late output formatting). Then the basis must have m > 1, which is rung 1.

**Intuition 1 as a transfer condition.**
- The router is p(a_next | state, context). For it to generalize across worlds, the signal it reads must be a function of the *computational situation* ("an unresolved pointer sits in the register", "the prediction is still spread over candidates matching key X"), not of the world's content. This is Neural Interpreters' type matching, with the type required to be world-agnostic.
- The same condition decides whether the halting policy inherits the rarity problem. "Continue" at step k gets gradient only from tokens needing more than k steps, which are rare. If halting is a function of the state type, the same decision recurs at every depth and the signal pools; if it is a function of the step index, it does not, and it becomes a memorized depth table.

**Predictions this adds to rung 0 (Entry 49):**
- **P-pool.** On rare deep tokens, the static model's accuracy at depth k scales with the number of training examples needing depth k; the looped model's accuracy is nearly independent of that count, because one block serves all depths. Falsifier: the looped model's deep accuracy tracks example count as closely as the static model's.
- **P-signal.** A halting head that sees only the state extrapolates to deeper chains; a head that also sees the step index does not. Falsifier: no difference, or the reverse.
- **P-transfer.** A routing (halting) policy trained on worlds with one key/value alphabet and chain structure transfers to new alphabets with no adaptation, and transfers to new structures only if the state types are shared. Falsifier: no transfer across alphabets.

**Decision.**
- **D77.** Rung 0 gains the three predictions above and the design axis they require: halting from state only vs state plus step index; controlled counts of deep examples; worlds that vary alphabet and structure separately.

## Entry 51 (27 Oct 2026): the circuit-equivalence belief, and a provable way to test it (discussion)

**Premise (researcher).** The underlying belief: in a sufficiently trained very deep transformer, any unique layer can be accurately approximated by a serial composition of simpler layers, a circuit equivalence. Define a computational language of primitives rather than requiring unique layer functions. Asked for the toy and a structured, minimal, provable test.

**Three versions of the belief, of increasing strength.**
1. *Universality* (trivially true): with enough width and depth, anything approximates anything. Not worth testing.
2. *A small fixed basis with short compositions covers the structured layer class* (provable by enumeration in a finite setting).
3. *The layers that training actually produces are short compositions over such a basis* (empirical; falsifiable; the version that matters for the end system).

Version 3 is the one the belief is really about, and there is evidence on both sides:
- for: cross-layer sharing keeps most quality (ALBERT, Universal Transformer, MoEUT); layer pruning removes many layers cheaply; Relaxed Recursive Transformers (Bae 2024) convert a pretrained unique-layer LLM into a looped block plus small per-depth adapters and recover most of its performance, i.e. each layer ≈ shared block + small correction;
- against: trained transformers often implement shortcut circuits rather than the natural serial program (Liu 2023, shortcuts to automata; Fourier circuits for modular addition), so a trained layer may not be a short word in the *natural* basis even when it is in *some* basis. Tracr-compiled and learned models look different inside.

**A subtlety the toy must respect.** A transformer layer runs heads in parallel. Serial composition of single-operation primitives emulates parallel heads only if the state carries the intermediate results side by side. With a product state (registers), parallel is serial on different registers; the state must be wide enough, and the basis must include writes to distinct registers.

**The test, in three levels.**

*Level 1, provable by enumeration (no training).* A finite symbolic layer language: a state of a few registers over an alphabet Σ, plus the context; a basis of RASP-like primitives (token maps, select-by-predicate, aggregate, positional shifts, register moves). The set of layer-functions generated under composition is a finite transformation monoid, so for a candidate basis one can compute exactly:
- **coverage:** which fraction of the structured layer class is reachable at all;
- **length:** the distribution of minimal composition lengths and the diameter (the longest minimal word), i.e. the circuit complexity of layers over the basis;
- **redundancy:** the number of words per function (non-identifiability; the MDL prior's job);
- **the Pareto frontier** basis size vs. composition length: the real trade, a tiny basis buys long serial compositions, more steps and more routing decisions.

The Cayley-graph BFS from the Z_7 round (`count_isa.py`) is exactly this tool. The classical anchor: three generators suffice for every map on a finite set, with long words.

*Level 2, decompilation with ground truth (diagnostic).* Train deep unique-layer transformers on sequence tasks whose generating programs are known words over the basis. For each trained layer, fit the shortest basis word that reproduces the layer's input→output behavior on that layer's own input distribution (behavioral equivalence, or causal patching). Measure fidelity against word length. This is where version 3 can fail by shortcuts: a layer that computes a parallel doubling step is not a short serial word unless the basis admits it.

*Level 3, distillation (the direct test of version 3).* Take a trained deep unique-layer model and distill it into m shared primitives routed over T steps. Measure fidelity to the original as a function of (m, T), with m = 1 (looped block with input injection, the Relaxed-Recursive setting) and m = D (no sharing) as the ends. The belief holds for that model if a small m with moderate T reproduces its function; it fails if fidelity needs m ≈ D. This finds the basis by optimization rather than enumeration, which sidesteps the natural-basis objection: the question becomes whether *any* small basis exists for the layers training produced.

**Prediction and falsifier.** On the pointer/match task family (rung 0/1), the distillation curve reaches ≥ 95% fidelity at m ≤ 4 and T ≤ 2D for models trained on compositional data; it needs m close to D on data without compositional structure (random lookup tables), which is the control. The decompilation finds short words for most layers, except where the static model learned a doubling shortcut, which should show up as layers needing parallel bundles. Falsifier: fidelity needs m ≈ D even on compositional data, or decompiled words are as long as the model is deep.

**Decision.**
- **D78.** The circuit-equivalence belief is tested at three levels on the same toy family as rungs 0–1: enumeration of a symbolic basis (coverage, length, redundancy, the size-vs-length frontier); decompilation of trained unique-layer models into basis words; distillation of trained deep models into m routed primitives, with m = 1 and m = D as the ends. Level 1 reuses the Z_7 enumeration tooling.

## Entry 52 (27 Oct 2026): what a token at depth t attends to when depths are ragged (discussion)

**Premise (researcher).** With per-token dynamic depth, token i may halt at depth 4 while token j > i needs depth 10. What does the router for token j at depth 9 condition on, and what does token j attend to there?

**Argument.** Dynamic depth turns the causal structure into a 2-D grid (position × depth). The options, and who uses them:
1. **Depth-aligned with frozen finals** (Universal Transformer/ACT; Mixture-of-Recursions with per-depth KV caches; rung 0): at depth t, token j attends to each earlier token i's state at depth min(t, halt_i). A halted token's final state is copied forward. Train and test agree if generation caches K/V per depth.
2. **Expected halted state** (Sparse Universal Transformer): the soft version, K/V = Σ_t p_t h_t.
3. **Finals only** (generation-time natural order: earlier tokens are complete before a later token starts): token j at any depth attends to earlier tokens' final states. In teacher-forced training this is sequential across positions, or needs sweeps; it is the semantics the other options approximate.
4. **Whole histories**: token j attends to all depths of earlier tokens (reading their scratchpads). Richer, costlier; not used here.
5. **Skip semantics** (Mixture-of-Depths): a token that skips a layer is absent from that layer's attention and reappears, unchanged, at the next.

**The router's conditioning.** In rung 0 the halting head reads the token's own state after each block application. That state already contains what attention gathered at that depth (data-processing argument, Entry 43), so the router conditions on context through the block, not on context directly. The first step is applied to every token (the first layer is static); selection begins from the state after it. A router with its own attention over the depth-aligned context is the more expressive variant, deferred.

**Decision.**
- **D79.** Rung 0 uses option 1 (depth-aligned, frozen finals), with context tokens single-step so their states are their depth-1 states in every arm. The semantics are stated in the design.

## Entry 53 (27 Oct 2026): novelty assessment, rung 0 and the core idea (discussion)

**Premise (researcher).** Do these experiments provide value the literature hasn't, and is the core idea novel?

**Rung 0.** Narrow but real. Not in the literature, as far as three reviews and our reading found: learned, unsupervised per-token halting on *hidden* serial demand in a causal transformer, scored against exact demand, with LO separating block extrapolation from policy extrapolation; and the signal-pooling test (deep accuracy against the count of deep examples, shared block vs static stack). The rest of the grid is confirmatory or a control. Its value is mainly as the foundation gate of the ladder; as a result it is small-to-moderate.

**The core idea, piece by piece:**
- per-token dynamic depth over a shared block: exists (ACT, PonderNet, UT, MoR, Ouro, recurrent-depth LMs);
- per-token selection among shared primitives reused across depth: exists in pieces (Neural Interpreters, MoEUT, SUT, CRL, routing networks);
- routing trained by search with policy and value: exists for programs (AlphaNPI, REPL, CodeIt) and as offline labels for layer skipping (Dr.LLM); not as online expert iteration over a primitive library in a sequence model;
- residual-driven refinement and exit: exists (predictive coding, CALM, Geiping's KL exit);
- "a PFN over model composition" (Entry 48: amortized posterior over the model's own composition, trained on a prior over worlds, with EM/AlphaZero as the training loop): a synthesis of known pieces (planning as inference, wake-sleep, Grill 2020, PFN theory). Not stated this way in the literature we found, but a synthesis is not a result; it earns its place only through predictions the pieces don't make: the MDL/identifiability bet, extrapolation in structure space, structure-level fidelity against an exact posterior over procedures;
- the circuit-equivalence belief (Entry 51): partially supported (Relaxed Recursive Transformers, layer pruning, cross-layer sharing); the distillation test (fidelity of a trained deep model distilled into m routed primitives, as a curve in m) appears not to have been done.

**Where novelty could actually land:**
1. the measurement methodology: exact references for routing (Bayes over procedures, exact branch values, structure fidelity) on generators with known compositional structure;
2. the combination reuse-across-depth × per-token selection × learned allocation, trained from next-token loss with value and search, evaluated for extrapolation in composition space (rung 1 onward);
3. the distillation curve for circuit equivalence.

**Risk.** The field is converging on this space (MoR, Ouro, MoEUT, Dr.LLM, recurrent-depth LMs, all 2024–2025). Dynamic depth alone will not differentiate; the library axis and the exact-reference methodology will.

**Decision.**
- **D80.** Run rung 0 lean (main cell and P-pool first; the other cells only if cheap), then move to rung 1 and the distillation test, which carry the novelty.

## Entry 54 (27 Oct 2026): rung 0 designed, reviewed, revised and implemented

**Premise.** The researcher asked for the rung-0 design, an independent review, and the implementation.

**Design** ([ptr/DESIGN.md](../experiments/ptr/DESIGN.md)): pointer chasing with hidden, heavy-tailed chain length; one shared block with learned halting against static stacks, a fixed loop, oracle halting and an index-only null; the headline is halting extrapolation (LH-s against LO) and the signal-pooling test (deep accuracy against the count of deep examples).

**Derivations before measuring:** oracle allocation costs 1.12 steps per token against 8 for S8; the deep-example counts per tail factor; the basis-size vs word-length frontier for token maps (3 generators cover all maps on 6 symbols at mean word length 15; 45 generators at 4.3 with 400 minimal words per map).

**Review 1** found the static ceiling wrong for pairs-as-tokens (doubling at pair tokens; simulated causal reach S4 0.51 at k = 8), the planted block as *described* destroying its answer, incoherent halting semantics, an empty transfer hypothesis, a confounded state-vs-index test, a saturating P-pool test, and a cost underestimate. **Revision 2:** context tokens read-only and single-step by construction (static reach exactly L; coherent semantics), an index-only null, direct subsampling of the deep tail over four decades, equal width primary, a no-hop baseline, two hypotheses dropped.

**Implementation** (`src/topology_design/ptr/`, 17 tests): numpy and vectorized GPU generators with an exact executor; the planted hop block (exact at every k ≤ 32, idempotent after termination, context tokens fixed points); static and looped models with fixed / oracle / learned / index-only / value halting; run-stacked models with per-run AdamW and clipping, paired per-seed data, equivalence to separate runs to float precision; stratified evaluation with residual traces, a step-index probe and the no-hop baseline; grids (307 runs + 60 pilot), a resumable stacked runner, analysis, and a Kaggle job tested end to end on CPU.

**CPU pilot (gate G2, inconclusive).** With oracle halting and d = 264, the block solves k = 1 within 500 steps and is still near chance on k ≥ 2 after 2,500 steps (loss 1.31 → 1.14, consistent with partial elimination rather than a second hop). Without input injection it moves slightly faster. 2,500 CPU steps is far short of the 6,000–20,000 planned; the GPU pilot decides G2. A no-injection oracle arm was added to the pilot.

**Decisions.**
- **D81.** Nothing in the grid runs before the GPU pilot settles G2 and the learning rates; if the block cannot learn the second hop, the block (or the injection) is revised and recorded first.
- **D82.** Run order: pilot → main → P-pool; the other cells only with budget left over.

## Entry 55 (27 Oct 2026): rung-0 pilot 1

**Runs.** 60 runs, 6,000 steps each, run-stacked on Kaggle's two T4s in 3.2 hours ([ptr/PILOT1.md](../experiments/ptr/PILOT1.md); raw records in `experiments/ptr/results_data/`). A first launch failed on the first optimizer step from a mixed-precision scaler misuse (fixed: per-run manual loss scaling, halting math in fp32, fp32 fallback).

**Results.**
1. **Gate G2 passes.** With oracle halting and input injection, lr 1e-3, the shared block learned all eight hops by 4,000 steps; trained on k ≤ 8 it is 1.00 at every k ≤ 24 and 0.89 at k = 32. lr 3e-3 diverges. **Without injection the block never learns the second hop at any learning rate** (the CPU hunch in Entry 54 was wrong; Yang 2024's finding holds here).
2. **The fixed loop overthinks.** LF at its training depth is 1.00 for k ≤ 8; at 40 loops its accuracy rises with k, from 0.38 at k = 1 to 0.99 at k = 32. The answer state is not a fixed point beyond the training horizon. Oracle halting has no such decay. This is direct evidence for the halting question.
3. **LH-s collapsed in all 24 configurations:** λ → 1 at step 1, perfect at k = 1, chance elsewhere. The index-only null LH-x learned a constant depth of 8 (perfect for k ≤ 8). The cause is an initialization asymmetry of ours: the state head started at λ ≈ 0.5 (halting mass on steps 1–2, so deep steps never carried enough weight for the block to learn them, after which early halting was optimal), the index head at λ ≈ 0.12. PonderNet's rich-get-richer failure, triggered by us. Fixed: every halting head starts at λ ≈ 0.12 (`halt_bias_init`).
4. **The static arms barely learn in 6,000 steps:** S8 (7.3M parameters) 0.24 at k = 2, S2 at chance for k ≥ 2, while the shared block learns the hop once for all depths. This is the pooling effect of Entry 50 appearing early, but the fairness rule (over-provision so every arm fits) requires longer, gentler static runs before any conclusion.
5. **Cost:** ≈ 0.06 s per run-step for a looped arm (T = 10, batch 64); the main cell at 10k steps is about one two-GPU session.

**Decisions.**
- **D83.** Record the pilot; gate G2 is passed with injection on; `LOni`/injection-off becomes a one-cell ablation as designed.
- **D84.** Pilot 2 (22 runs): LH-s with the corrected init over lr × β × λ_p at 8,000 steps; S8 at lr 1e-4 and 3e-4, and S2, at 15,000 steps. The main grid waits for G4 (LH-s within 0.02 of LO in range) and for the static arms' budget.
- **D85.** Lesson L23.

## Entry 56 (27 Oct 2026): exploratory, residual-triggered halting on the pilot-1 fixed loop

**Not preregistered.** Computed after the fact from the per-step traces that pilot 1's fixed-loop runs saved (2 runs × 64 queries per k).

**Result.** Halting the fixed loop (trained at 10 loops, evaluated at up to 40) at the first step where the readout's step-KL falls below 0.01 gives accuracy 1.00 / 1.00 / 1.00 / 0.99 / 0.99 / 0.97 at k = 1 / 4 / 8 / 16 / 24 / 32, at 2.6 / 5.1 / 9.3 / 17.9 / 26.1 / 34.3 steps. Running 40 loops regardless gives 0.07 / 0.02 / 0.02 / 0.25 / 0.87 / 0.96 (overthinking). Thresholds 10⁻² to 10⁻⁴ all work; 10⁻¹ stops too early on long chains. The rule beats stopping at exactly k on these traces because the readout settles one step after the hop.

**Reading.** On this task, prediction stability is a world-agnostic completion signal that generalizes four times beyond the training depth with no learned halting at all (Geiping 2025's zero-shot KL exit, reproduced on exact demand). It confirms Entry 46's intuition for the *completion* case: the reducible residual here is 0/1 with a crisp absorbing state. It says nothing yet about predicting demand, or about tasks whose residual is graded.

**Consequences for the design.**
- The bar for learned halting is no longer the oracle alone: LH-s must be compared with the free rule (LR), in accuracy *and* in compute (LR spends ≈ k + 1.5 steps; a learned policy could stop at k).
- LR is promoted from diagnostic to a primary arm (LF + KL exit), and H2 is restated as LH-s vs LR vs LO.
- If learned halting does not beat the free rule, that is the rung-0 result: on crisp-completion tasks, a stability exit suffices, and the case for a *learned* depth policy moves to tasks where completion is not a fixed point (graded residuals, Stage B / rung 1).

**Decision.**
- **D86.** Record as exploratory; promote LR to a primary arm; restate H2; thresholds for H7 are set from these numbers only in the sense of being falsifiable in the main grid, not confirmed by them.

## Entry 57 (27 Oct 2026): rung-0 pilot 2

**Runs.** 22 ([ptr/PILOT2.md](../experiments/ptr/PILOT2.md); records in `experiments/ptr/results_data/`).

**Results.**
1. **G4 passes, narrowly.** With the spread halting init, LH-s at lr 3e-4 and a weak prior (β = 0.003) matches the oracle in range: 1.00 for k ≤ 10, steps ≈ k + 1. At lr 1e-3, or with β = 0.01 and λ_p = 0.2, it collapses to immediate halting again. PonderNet-style halting has a narrow basin here.
2. **H2 is falsified for the state-reading policy, with the mechanism identified.** Beyond ≈ 1.5× the training horizon the policy halts regardless of the register (chance by k = 16, steps capped near 15). Its λ stays 0 through the training horizon and then ramps while the register still holds a key: the state's composition drifts beyond the trained depths (the step index is linearly decodable from the state, 0.35 vs 0.03 chance), and an absolute reading of the state drifts with it. The same block halted by the readout's *change* (LR) reaches 0.97 at k = 32. The block extrapolates; the policy, as parameterized, does not; a change-based rule does.
3. **Unique-layer depth learns hops sequentially.** S8 at lr 1e-4 learns k = 1 / 2 / 3 at 0.5k / 2–4k / 9–11k steps and not k = 4 by 15k; the shared block learned all eight by 4k. Each layer rediscovers the hop, after the layer below. The pooling claim of Entry 50 appears as learning *time*, not only as data.

**What this changes.**
- The interesting comparison is no longer "learned halting vs oracle" but **what the halting policy reads**: absolute state (fails beyond the horizon) vs change (extrapolates). A learned change-based head (the learned counterpart of LR), and a state head trained with a horizon well beyond k_max (T = 24), test the mechanism directly.
- The static comparison should be measured as **time to learn each depth** (steps to criterion), and the pooling test as time-to-criterion at a depth both arm types reach (k = 3) against the count of such examples, rather than accuracy at a fixed budget.
- Effects are large; 10 seeds suffice everywhere. 29 for the headline is not needed.

**Decisions.**
- **D87.** Record pilots 1–2 as results of rung 0 (not preregistered; the preregistration follows for the confirmatory grid).
- **D88.** Revision 3 of the rung-0 design: fixed LH-s constants (lr 3e-4, β 0.003, λ_p 0.1); two new arms, LH-Δ (halting head on change features: readout step-KL and relative ‖Δh‖) and LH-s24 (state head, T = 24); static arms at lr 1e-4 with a long budget, measured by steps to criterion; the pooling test as steps-to-criterion at k = 3 vs example count (k_deep = 3), arms S8, LO, LH-s; 10 seeds; the main grid lean.

## Entry 58 (27–28 Oct 2026): rung-0 revision 3 designed, implemented and reviewed; the confirmatory grid

**Premise.** The pilots (Entries 55–57) answered revision 2's headline and invalidated two of its cells (the learned-halting arm at a learning rate that collapses; the static arm at a budget it cannot finish). D88 asked for a revision before the main grid.

**Design (DESIGN.md revision 3).** The question is now *what the halting policy reads*. Arms: LH-Δ (PonderNet head on change features only: log readout step-KL and log relative ‖Δh‖, h₀ = e₀), LH-s24 (the state head trained with T = 24 for chains ≤ 8), LR as a primary arm (the stability exit computed on every LF run's full evaluation set, threshold 10⁻² preregistered from pilot 1), LO, LF, LH-s, LH-x, S8 (lr 1e-4, 25k steps), S2. Endpoints add steps to criterion by k (the first 100-step training window with accuracy ≥ 0.95 at k, confirmed by the next; the halted readout for learned arms) and pooling as time at k = 3 (k_deep = 3; factors 1, 10⁻¹, 10⁻²; arms LO, LH-s, S8). Predictions P1–P7 with thresholds, falsifiers and an outcome table. After the reviews (below): a gate grid (pilot 3, 6 runs, ≈ 1.3 T4-hours) and the confirmatory grid (93 runs, ≈ 14.8 T4-hours in two balanced shards of ≈ 7.4 h): two Kaggle sessions.

**Implementation.** `halt_input="delta"` in the single and stacked models (fp32 features, standardized with centres and scales from the pilot-1 traces, clamped; same init as every other head; a head learning-rate multiplier); `halted_logits` (the policy's own prediction) logged as training accuracy next to the T-loop readout; `kl_exit` on the full evaluation set for fixed arms and, at the final evaluation, on every looped arm's own block run as a fixed loop; per-step states kept at query positions only (memory); balanced shard assignment by estimated cost; analysis rows for LR (on LF's block and on each arm's own), the threshold frontier, steps to criterion with example-count windows and censoring-aware medians, pooling by time, λ by loop with the head's parameters, and the preregistered comparisons as paired seed bootstraps. Tests: 24 in `tests/test_ptr.py`, including the planted LH-Δ (exact halting at k + 1 to k = 32 with a hand-set head) and the planted-LR caveat.

**Derived before running.**
- The stability exit relies on the readout *moving while the chain is followed*. On the exact planted block the readout is uniform while the register holds a key, so LR halts at loop 2 for every k ≥ 3 (tested). LR therefore measures a property of trained blocks; LH-Δ's ‖Δh‖ feature does not share the blind spot.
- Tail shares at k_deep = 3: k = 3 is 0.100 / 0.014 / 0.0014 of queries at factors 1 / 10⁻¹ / 10⁻² (≈ 54 examples per 100-step window at 10⁻²).

**Reviews** (two fresh agents; `experiments/ptr/review/review3_design.md`, `review3_code.md`; handling table in DESIGN.md §12). The design review's main finding was quantitative and would have cost the whole grid: with AdamW, Σ lr over the 8k-step schedule is ≈ 1.13, so no head parameter can move more than ≈ 1.1 units; the change head as first parameterized (raw log features in [−14, 4], bias −2) could not reach a policy whose chasing λ is small enough, and would have halted prematurely at ≈ 15–35 loops in a way indistinguishable, in the accuracy table, from the horizon failure it was meant to explain. Handled by standardizing the features (chasing ≈ +3, settled ≈ −3 on the KL feature, measured on the pilot-1 traces; the state head already reads through a LayerNorm), by a head learning-rate multiplier, by a gate run (pilot 3) with a selection rule fixed in advance, and by naming the λ-by-loop signature that tells a flat floor from a post-horizon ramp. The second finding: P1's falsifier compared the learned head with the stability exit on a *different* block (LF's), although the head shapes its own block; the exit is now computed on every arm's own block. Also: P7's LR line contradicted pilot 1 (restated with k + 1.1–2.3), P6 was unconfirmable as written (restated one-sided and censoring-aware), dead zones in P1/P2 (per-seed fractions), the outcome table was incomplete (extended), the paired bootstrap was named but not implemented (implemented), S8's rate was not bracketed (3e-5 added), and three cheap cells were worth adding (LH-Δ in the pooling cell; LH-s with k_max = 4 to ask whether the state head's failure follows T or k_max; ‖h_t‖ in traces). The code review found no blocking defect and four should-fix items: LH-x beyond T used *untrained* logits rather than repeating the last trained one (fixed; the pilot numbers stand since it halted before T), the pooling ratio ignored censoring (fixed), the bootstrap (above), and the memory claim for a 10-stack at T = 24 was wrong (≈ 15 GB; corrected, and a failing stack now splits in two instead of retrying in fp32). The trace-measurement that fed the standardization also showed that on the trained fixed-loop block the relative ‖Δh‖ does **not** separate chasing from settled states (medians 0.78 vs 0.87 in log) while the step-KL does (−1.07 vs −5.6): the block keeps moving after the answer, which is the over-running of Entry 55 seen from the inside. The KL feature carries the signal on trained blocks; the ‖Δh‖ feature carries it on the planted one.

**Decisions.**
- **D89.** Run pilot 3 (the gate for the two untrained arms, with the selection rule fixed in DESIGN §10) and then `grids/rev3.json` as the confirmatory grid of rung 0; the pilots stay labeled not preregistered.
- **D90.** The doubling cell and S8-pm are deferred, with reasons (DESIGN §11); LH-Δ in the pooling cell is run now.
- **D91.** Any learned policy head in this program reads *standardized* inputs, and its reachable set under the optimizer and step budget is checked against a planted solution before the arm runs (L24).

**Expected falsifiers.** P1: ≥ 8/10 seeds of LH-Δ < 0.6 at k = 16 while LR on its own block ≥ 0.9. P2: ≥ 4/5 seeds of LH-s24 < 0.6 at k = 16. P6: LO's time to k = 3 grows ≥ 3× at factor 10⁻².

## Template for future entries

Date, premise, evidence (with run or test links), argument, decisions, rejected alternatives, expected falsifiers.
