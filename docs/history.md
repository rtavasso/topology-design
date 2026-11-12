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
- **D92.** The gate and the grid run as one unattended Kaggle session: `gate.py` applies the G4b rule between them (`launch.py --ptr-gate … --ptr-grid …`), so no one chooses the multiplier after seeing out-of-range numbers.

**Expected falsifiers.** P1: ≥ 8/10 seeds of LH-Δ < 0.6 at k = 16 while LR on its own block ≥ 0.9. P2: ≥ 4/5 seeds of LH-s24 < 0.6 at k = 16. P6: LO's time to k = 3 grows ≥ 3× at factor 10⁻².

## Entry 59 (28 Oct 2026): rung-0 revision 3 ran — the block, not the head

**Runs.** 99 (gate 6 + grid 93), one unattended Kaggle session (D92), 8.4 h; [ptr/REV3.md](../experiments/ptr/REV3.md); records in `experiments/ptr/results_data/`.

**Preregistered verdicts (DESIGN.md §8).** P1 falsified (LH-Δ at the gate-chosen ×1 head rate: 0.65 at k = 16, 1/5/4 seeds; 0.15 at 32), P2 falsified on its k = 24 line (LH-s24: 0.96 at 16, 0.44 at 24), P3 falsified (the one-parameter stability exit is cheaper, k + 1.3 vs k + 2.3, and far more accurate out of range), P4 falsified on excess, P5 holds (S8 learns hops at 500 / 2,500 / 6,600 / > 25k steps; LO all eight by 3,600), P6a refuted in the opposite direction (the oracle-halted block never learns k = 2 when k = 1 dominates: 0/10 runs at tails ≤ 10⁻¹), P6b holds (S8 2.6× slower at 10⁻¹), P6c mixed and not isolating the policy, P7 two narrow misses.

**The reading that the preregistration did not anticipate.** In every learned-halting arm the stability exit applied to the arm's *own block* fails at exactly the arm's own failure point (LR@LH-s 0.26 at 16; LR@LH-Δ 0.74 at 16; LR@LH-s24 0.36 at 24; LR@LH-s-k4 0.23 at 8), while on the fixed-loop block it reaches 0.93 at 32 and the oracle-trained block 0.92. The halting *head* is not the limit: chasing λ is 0.000 from step 2,000 in every arm; LH-s24's head keeps going to 36 loops at k = 32 with slope 1.06; LH-Δ's ramp begins only after its block has settled. **The block trained jointly with a halting policy loses the ability to iterate beyond its training range, and it loses it late:** at 4,000 steps LH-s and LH-Δ extrapolate to k = 16 in 8/10 seeds each (0.89, 0.88 — the oracle's level); at 8,000 steps 0/10 and 1/10. Pilot 2's result was the late-training state of this. The failure point scales with the chain lengths seen (LH-s-k4 fails at ≈ 1.5·k_max = 6–7) and moves out with the loop horizon (LH-s24: ≈ 22), so it is the block's trained range, not a fixed number.

**The exception, exploratory.** The two gate runs of LH-Δ with a ×10 head learning rate: 1.00 at every k ≤ 32, steps k + 1.1 (cheaper than the exit rule), LR on their own block 1.00 everywhere, and no decay through 8k steps. Their heads reached w_KL ≈ −3 to −5 (×1: −0.66). The preregistered gate rule, blind to out-of-range accuracy by design, chose ×1 for the confirmatory cell because both multipliers passed in range; so this is a 2-seed pilot result, and the rule did its job (the headline is not claimed from 2 seeds that were not preregistered).

**Candidate mechanism (H11).** The PonderNet KL-to-prior term is absorbed by whichever part can move. At ×1 the head barely moves (|w| ≈ 0.7–1.9 after 8k steps, within the Σ lr bound of L24) and the term reaches the block through the head's input, pushing intermediate states to look "settled"; inside the trained range the cross-entropy holds the block to the answer, outside it the pressure wins. At ×10 the head absorbs the term (its KL term is 2.1 nats, the largest; its block is clean). Consistent with LH-x (an index head cannot absorb it; its block fails even in range: LR@LH-x 0.36 at k = 8) and with LH-s24 (λ ≈ 0.17 after the answer spreads p closer to the prior; its block holds to ≈ T).

**On pooling (Entry 50).** The oracle arm, meant as the clean block test, failed for a reason that is itself about pooling: a shared block pools signal across depths only when training iterates it on every example; oracle halting never iterates the dominant k = 1 queries, and the one-loop map wins (0/10 runs learn k = 2 at tails ≤ 10⁻¹; 1/10 fails at factor 1). The static stack's time to k = 3 grows 2.6× when the k ≥ 3 share drops 7×; the shared block under LH-s grows 1.3×. At a 70× drop both learned-halting arms slow ≈ 7× at the T-loop readout too, so the policy's data dependence could not be separated from the block's.

**Decisions.**
- **D93.** The next round is *decoupling*, one short session (≈ 4 T4-hours): LH-Δ ×10 at 10 seeds (confirmatory of the exploratory result; preregistered: ≥ 8/10 seeds ≥ 0.9 at k = 32, steps ≤ k + 1.5, no decay between the 4k and 8k evaluations); LH-s ×10 (does a fast state head also leave the block clean?); LH-Δ ×1 with the head's input detached from the block (stop-gradient) and with β = 0 (which of the two is the damage); evaluations every 1,000 steps to time the collapse. Oracle arms iterate k = 1 queries at least twice, or are dropped as a pooling control.
- **D94.** Pilot results and the confirmatory grid are reported as they fell: the preregistered P1–P4 are falsified, and the ×10 result is a pilot result until D93 runs.

**Rejected alternatives.** Re-running the ×10 cell now as "the real P1": the rule chose ×1 before the data, and choosing afterwards on out-of-range accuracy is the selection the gate was built to prevent. Reading the ×1 failure as "change features do not transfer": LR on LH-Δ's own block fails at the same point, so the signal is gone from the block, not misread by the head.

**Expected falsifiers of H11.** LH-Δ ×1 with the stop-gradient still decays (then the damage is not through the head's input); β = 0 still decays (then not the prior); LH-Δ ×10 at 10 seeds below 6/10 at k = 32 (then the 2-seed result was luck).

## Entry 60 (28 Oct 2026): rung 0 closed; D93 deferred to rung 3 (discussion)

**Premise (researcher).** Concern that the program was narrowing onto the halting mechanics of rung 0; unless the proposed decoupling round (D93) could be argued to be worth running before the planned sequence, move on.

**Argument (agreed).** D93 tests whether joint training of a halting head and its block damages the block, and what prevents it. That is a rung-3 question (Q4, primitives learned jointly with routing). Rung 1 as defined (D72) has fixed symbolic circuits, so the mechanism of F22–F23 cannot occur there and no D93 outcome would change rung 1's design or reading. What rung 0 settled that rung 1 needs carries over without another run:
- one primitive iterated per token extrapolates 4× in chain length when its training does not pass through a halting objective (F22);
- a router's policy must be reachable by the optimizer within the step budget (L24): standardized inputs, a planted solution within reach;
- halting from change works, but the readout-stability rule is blind on exact circuits (the readout is constant while chasing); the state-change signal is not;
- signal pooling across depths requires the training procedure to iterate the primitive on every example (F25);
- unique-layer depth learns hops one at a time (F26), the baseline rung 1 inherits.

**Decisions.**
- **D95.** Rung 0 is closed with its record (Entries 54–59). D93 is deferred to rung 3, where joint learning is the question; the ×10 result (F24) stays exploratory until then. The 10-seed ×10 confirmation may ride on a spare GPU during a later session but gates nothing.
- **D96.** Rung 1 is designed next, per D72 and D75, with the carry-over rules above as design constraints.

## Entry 61 (28 Oct 2026): rung 1 designed, reviewed, revised (the deferred trigger), implemented

**Premise.** D96: rung 1 per D72/D75 — symbolic generic circuits (HOP, PREV) with a learned Markov router trained by the exact route marginal; Q1 (order-dependent chains learned from next-token loss) and Q2 (generalization to longer chains and new orders vs static and looped learned models).

**Revision 1** ([route/DESIGN.md](../experiments/route/DESIGN.md) at `de85390`): the P-trigger (←) in the value slot of the pair the P-step leaves from; three actions; 66 symbols; P1–P8.

**Review 1** ([review1_design.md](../experiments/route/review/review1_design.md), [review1_code.md](../experiments/route/review/review1_code.md); independent, with CPU checks). Verdict "run with changes", and the decisive finding: **with a local trigger every "new order" is in-distribution for a Markov router** — the rule is a function of the matched pair, the router's receptive field; the reviewer's 1,800-step CPU run (7 % of the budget) already passed P1–P3 (route accuracy 0.93–1.00 in range, 0.95/0.93 at k = 16/32, 1.00/0.98/0.93 on PP runs never seen). Six of eight predictions were decided before any GPU step. Also: P7's marginal reading was mis-specified (a state-blind Bayes mixture reached 0.94 because wandering mass died on ⊥ and had no vote); P8's probe had a ceiling of {0, 1, ≥ 2} (R² ≤ 0.61, failing by construction); P6 (B1) is untestable with absorbing answers; alternation and deep-P are training words; time-to-criterion on training batches risks memorization with a 60k pool; RM vs LF measures learned circuits, not selection vs gating (the planted-gate block would). Prior work: NPI (recursive), NRAM, Neural Program Lattices, CRL predict the local-trigger result. The code review found no numerical bug (marginal vs brute force 7e-7; planted router exact; hand-written reference agrees with the stacked router to 1e-6) and asked for all-H strata, P/no-P splits and depth-of-P curves for every arm.

**Revision 2 (this entry; DESIGN.md §15 has the item-by-item handling).**
- **The deferred trigger (D98).** The flag for a P-step sits one state *earlier* in the chain (δ = 1); the route must take a setup action there (SET: pending count +1, set-here = 1), carry the bit across a HOP and spend it at the next state (PREV requires pending ≥ 1). The state is a product (symbol × pending ∈ {0,1,2} × set-here), 390 values; four actions; P-steps carry a **decoy** value (HOP wanders into another chain instead of dying). PP runs reach pending = 2, a state value never visited in training. δ = 0 (the local trigger with the same library) is the control arm; the symbol-only router (state token without pending / set-here) is the F4 null.
- **Derived (revision 2, §5):** 54 training words; all-H covers 0.837; the state-blind reference is now 0.84 (committed answer) / 0.85 (marginal) — with decoys the Bayes mixture no longer rescues P-words (≈ 1/16); P at depth 2–7 per 100 steps 3,366 / 1,572 / 1,252 / 795 / 490 / 208; the oracle route's share of the target mass under a uniform router 0.66 / 0.37 / 0.13 / 0.03 at k = 1–4. A symbol-only router cannot produce the oracle route deterministically (SET then a move at the same symbol) but can succeed stochastically in the marginal reading (SET with probability a, then HOP; a of the mass carries the bit) at a cost in E[steps] — predicted as P9's two lines.
- **Predictions P1–P9 (§10):** P1 learnability with the setup step (SET and PREV flip together; RM-T the speed reference); P2 length by construction; P3 the headline: does the count extrapolate to the unseen value 2 (moderate confidence; both outcomes informative); P4 pooling (by construction for RM); P5 the open one: the looped block must hold the bit in its stream (LF ≤ 0.6 on P-words at k ≥ 4, its oracle-depth reading no better, LF-d0 ≥ 0.2 better); P6 removed; P7 per reading with the 0.84/0.85 references; P8 halting only; P9 the F4 null (committed 0 by construction, marginal above 0 by the stochastic trick).
- **Held-out criterion set (D97):** 256 episodes per run, evaluated every 100 steps: route accuracy by k, accuracy by deepest-P depth, teacher-forced accuracy by action (flip times); training accuracy kept as the memorization reference. The probe is dropped. LF is also read at the oracle depth (loop k, k + 1, k + n_P, any). The first deviation from the oracle route is recorded per failure.
- **Implemented:** `route/data.py`, `exec.py`, `router.py`, `train_route.py`, `train_neural.py`, `pool.py`; the flag on a frozen free direction in the ptr models; `tests/test_route.py` (12 tests: generator with flags and decoys, executor against the numpy oracle incl. the step-indexed marginal path, brute-force enumeration over 3^L routes, the planted router on every stratum for both deferrals, the planted failures, run independence, record fields); the full suite 220 passed. The planted router is exact in the committed reading at a few units from the init and sharp in the marginal (p(target) ≥ 0.958 at k = 32). Grids: pilot 20 runs ≈ 2 T4-hours, main 51 runs ≈ 6 (routed cost estimated at 6× revision 1's; the pilot measures it).

**Decisions.**
- **D97.** Time-to-criterion is measured on a fixed held-out set per run (every 100 steps), never on training batches, in every rung from here on.
- **D98.** Rung 1 runs with the deferred trigger (δ = 1) as the thesis setting, δ = 0 as the control, decoys at P-steps, and the symbol-only router as the F4 null; the committed route is the thesis reading, the marginal is reported.
- **D99.** The planted-gate looped block (circuits planted, gate learned; review 1's apples-to-apples selection arm) is deferred: run if P5 fails, to say whether the block's failure is the gate or the circuits.

**Rejected alternatives.** Running revision 1 as a replication (its result is predicted by NPI/NRAM and the reviewer's CPU run already shows it). The neighbour-trigger variant (flag on pair(s_{i+1}), readable with one self-attention layer among pairs): still a function of the local context, no setup step, no state extrapolation. A third circuit with a count condition: more library, same local reading.

**Expected falsifiers.** P1: `route_acc_P` < 0.5 on in8 in ≥ 5/10 RM seeds while RM-T ≥ 0.95 (the setup credit is not found by the likelihood in 6k steps). P3: pp4 < 0.5 while in4 ≥ 0.95 (the count is read categorically). P5: LF `acc_P` ≥ 0.9 on in8 in ≥ 5/10 seeds (the block holds the bit as well as the router). P9: RM-s `route_acc_P` > 0 anywhere (a leak).

**Addendum (same day): review 2 and a CPU sanity run.** Review 2 ([review2.md](../experiments/route/review/review2.md); independent, with hand-built policies on the repository executor): "run with changes", all applied. (1) Under δ = 1 a flagged state entered with a pending count *is* a PP run, so training never shows the conjunction of the two learned conditions (flag ∧ ¬set-here ⇒ SET; pending ⇒ PREV); the oracle's precedence (SET first) carries no training signal, and PP fails on that decision before the count value 2 is ever reached. P3 is split into P3a (ordinal count) and P3b (precedence), each measured by a counterfactual-state probe on ordinary P-words (the router is evaluated at all 390 states, so the probes are free). (2) The looped block need not carry the bit across a loop: the flagged pair is the one whose value slot holds the P-leaving key, readable in one layer by a value-slot match; P5 is relabelled "a two-match gate in one block vs an explicit register" and the LF-d0 margin is dropped; a carried bit needs δ ≥ 2. (3) In the marginal reading the symbol-only router's stochastic SET trick gives argmax accuracy ≈ 1 for any SET probability ≥ 0.3 (the wrong mass is spread), so P9 is restated in p(target) and a sampled-route reading is recorded. (4) Reachability is not the obstacle: PREV is sharp from step 100, SET flips at ≈ 1,000 steps of batch 16. Code paths correct; memory ≈ 4 GB per 10-run stack. A 2-seed CPU run (batch 16, 1,500 steps, 6 % of the budget) observed before the pilot: in-range route accuracy 0.85–1.00 including P-words, len16 0.88, alternation 0.80–1.00, p(HALT) exact by distance, **pp 0.00 with first deviation SET>PREV in 71 of 80 failures** — P3b's expected failure mode. The design records these as observed, not predicted (§10, §11).

- **D100.** If P3b holds in the main grid, the follow-up candidate is a dose–response in the PP training rate with a depth-pooling readout (how many conjunction examples at any depth fix the precedence at every depth, against the static stack); recorded, not scheduled — it competes with rung 2 under the researcher's rule (Entry 60).

## Entry 62 (28 Oct 2026): rung-1 pilot ran — rates unchanged; the precedence failure, the F4 null and the block's silence on P-words, at two seeds

**Runs.** 20 (`grids/pilot.json`), one Kaggle session, 1.2 h; [route/PILOT.md](../experiments/route/PILOT.md); records in `experiments/route/results_data/pilot/`.

**G3.** RM lr 1e-3, λ 0.02; LF lr 1e-3; S8 1e-4 — all the defaults; the main grid runs as written. Cost measured: routed 0.09 s per run-step, LF 0.113, S8 0.095; main ≈ 7.9 T4-hours.

**Observed (2 seeds; the main grid gives counts).** RM: route accuracy 1.00 in range, at length to 32 (steps = the oracle's), on alternation and deep-P; SET flips at step 600 vs 500 under teacher forcing (the setup credit costs ≈ 6,400 episodes); p(HALT) exact by distance. PP runs 0.00 for every RM configuration and RM-T, first deviation `SET>PREV`; probe (a) (count forced to 2 → PREV) 1.00 in all 12 routed runs, probe (b) (flagged ∧ pending → SET) ≤ 0.06: the count is ordinal, the precedence untrained. RM-s: committed 0 on P-words (answer accuracy = the all-H share), marginal argmax 1.00 with p(target) 0.75 (0.60 at len32), sampled 0.75 — review 2's reading of F4 at this rung. LF: all-H words 1.00 at T = 10 and under LR, P-words at chance at every depth and every evaluation; S8 reaches k ≤ 2 by 8k (F26).

**Not anticipated: two basins, selected by λ.** At λ = 0.005 (2/2) and lr 3e-3 (1/2) the router converges to "SET at every key state, PREV where the predecessor's pair is flagged" — exact answers in both readings, never the oracle route, k extra actions: the locally-readable solution (a value-slot match on the predecessor's pair) with the count as a free resource. At λ ≥ 0.02 the carried-bit route wins. The step cost therefore selects between two exact programs of different length here (B1 in its weakest form; an observation, not a preregistered result).

**Decision.** D101: the main grid runs unchanged (the preregistered cell); the always-SET basin is read from the first-deviation and steps columns wherever route accuracy is 0 with answer accuracy 1.

## Entry 63 (28 Oct 2026): rung 1 ran — the deferred rule is learned, the control is not; the ladder's generalization bound; rung 1 closed

**Runs.** 51 (`grids/main.json`), one Kaggle session, 3.7 h; [route/RESULTS.md](../experiments/route/RESULTS.md); records in `experiments/route/results_data/main/`; the deferral-0 diagnostic in `experiments/route/diag_deferral0.py` and `results_data/diag/`.

**Preregistered verdicts (DESIGN.md §10, revision 2).** All hold at full seed counts: P1 (10/10; SET flips at 600 vs RM-T's 500), P2 (1.00 at k = 32, steps = oracle), P3a (count probe 1.00, 10/10), P3b (precedence probe 0.03; PP 0.00 with first deviation SET>PREV 2,411/2,497), P4 (RM flat at 1.00 across P depth; the neural arms at chance everywhere), P5 (LF at chance on P-words at every reading, 10/10; LF-d0 identical), P7 (RM-idx 0.843 / 0.850 vs the derived 0.84 / 0.85), P8 (p(HALT) 1.00 / 0.00 / 0.00), P9 (RM-s committed 0 / marginal 1.00 / p(target) 0.77 / sampled 0.77 / +1.2 actions). Findings F28–F34.

**The control that failed.** RM-d0 (the local trigger: SET then PREV at the same symbol) never learns the P-rule in 6,000 steps, 5/5 seeds, loss plateau 0.47. The design's row for this ("a bug or a λ artefact; stop and inspect") fired. The inspection (CPU, logits at the four decision states every 50 steps, both deferrals; a planted-init run) found neither: the planted deferral-0 program is a stable optimum; from a random init both routers learn in fifty steps, from the stray-SET mass, the state-only rule "set here ⇒ HOP, carried count ⇒ PREV" — the oracle's rule under the deferred trigger and the opposite of what the local trigger needs at the set-here state. Under the local trigger SET and PREV then wait for each other. **F29: setup credit arrives through the explicit state.** The intuition that the local trigger is the easy case was wrong for a reason the state makes exact: its two actions share a symbol. L26.

**What rung 1 says, in the ladder's terms.**
- Q1 (order-dependent chains with a setup step from next-token loss): yes with exact circuits and an exact E-step, at near-trace speed, *when the setup's payoff is a state feature* (F28, F29).
- Q2 (generalization to new lengths and orders vs the learned carriers): the state-routed learner generalizes exactly to the closure of its per-state decisions under the state encoding's geometry — length and depth by construction, an unseen count value by linearity, an unseen conjunction of conditions not at all (F30); the looped block and the static stack do not acquire the second circuit (F32). The explicit state's value is a deterministic program at oracle cost, not Bayes accuracy (F31); the step cost selects the shorter of two exact programs (F33).
- The novelty position (Entry 61, the 28 Oct discussion of Mixture-of-Recursions): a sharpening of NPI / NRAM / Neural Program Lattices / CRL with four modest additions (setup credit measured against traces and shown to flow through the state; the three-readings result; the step-cost selection; the exact generalization bound with probes); not a stand-alone result. Rung 2 carries the novelty.

**Decisions.**
- **D102.** Rung 1 is closed with its record. D99 (the planted-gate block) and D100 (the PP dose–response) stay recorded, not scheduled; neither outranks rung 2 under the researcher's rule (Entry 60).
- **D103.** Constraints rung 2 inherits: (i) the state the router reads must make the decisions a task needs *determined* — ordinal and compositional encodings for quantities the rule depends on, and a library whose semantics do not put a setup action and its payoff at the same state unless the state records the setup; (ii) a neural state must keep the geometry that carried the count extrapolation (a direction, not a category), which is the first thing to measure once the state is neural; (iii) endpoints must include steps, route determinism and sampled-run reliability, never answer accuracy alone (F31); (iv) the step cost is part of the objective, not a regularizer to drop (F33); (v) a control arm runs in the pilot (L26).

**Rejected alternatives.** Re-running RM-d0 with a curriculum, a different init or longer training to "fix" the control: it would answer how to break one basin on one toy, not a ladder question; recorded as a follow-up with D99 / D100. Reading RM-d0's failure as evidence against the exact marginal: RM-d1 and RM-T succeed with the same objective and architecture; the failure is specific to where the setup's payoff lives.

**Expected falsifiers of the carry-over (D103).** A neural state at rung 2 that extrapolates an unseen count value without an ordinal encoding (then (ii) was unnecessary); a local-trigger variant that learns under a richer state that records the setup (then F29's "through the state" reading is confirmed in the other direction — the cheapest test, ≈ 0.3 T4-hours, if rung 2's design needs it).

## Entry 64 (29 Oct 2026): rung 2 designed — finding routes without the exact E-step; reviewed; revised

**Premise (researcher, 28–29 Oct).** Stay symbolic (neural carriers confound the mechanism); design rung 2, review it with two independent agents (design, code), implement the fixes.

**Design** ([routefind/DESIGN.md](../experiments/routefind/DESIGN.md)). Same router, task and data as rung 1; only the training procedure changes, to procedures that see only the routes they sample or search, each scored exactly: the K-sample importance-weighted bound (IW16, VIMCO), the expected log-likelihood (ELL16, the K = 1 bound), REINFORCE on p(y | r) (RL16), sample-then-imitate (SI16), beam expert iteration / hard EM (EI16), and EI16B, expert iteration started inside the exact marginal's δ = 0 basin. Success: a non-exact procedure reaches the exact router's rule on the deferred trigger within 4× its episodes. Code: `route/estimators.py`; the estimator switch, per-run init and sampling seeds, warm start and held-out search statistics in `train_route.py`; `tests/test_routefind.py` (16 tests, including exact unbiasedness at K = 2 by enumerating all route pairs).

**Reviews** ([design](../experiments/routefind/review/review1_design.md), [code](../experiments/routefind/review/review1_code.md)). Code: correct (unbiasedness against full enumeration; the beam against brute force on 40 more cases; the exact and teacher paths byte-identical to rung 1; cost at or below the exact marginal). Both reviewers independently found the blocking flaw: revision 1's derived beam numbers at "init" were computed at an exactly uniform policy, where every prefix ties and the answer is set by sort order. At the real initialized routers, the beam's oracle share on short P-words is seed-dependent (0.10–0.87) under both deferrals, so the predicted δ = 0 / δ = 1 contrast does not exist. The design reviewer also showed that (i) expert iteration from scratch on δ = 0 is already decided (the CPU sanity run learns it; the expected hard-EM result), and (ii) the expected log-likelihood is not "the RL objective": its per-route reward prefers halting to guessing and points into a halt-at-the-flag basin. Both were confirmed in the derivation.

**Revision 2.** §5 re-derived at real init; the headline that can fail is now EI16B (search inside the basin; derived to fail, because the beam's MAP correct route at the basin policy is never the oracle's); RL16 and SI16 added; ELL16 relabeled with the halting prediction; EI4 and IW4 cut; teacher forcing on δ = 0 added as the speed reference; collision-free grid names; every review item mapped in DESIGN §13.

**CPU sanity (L25), observed before preregistration.** EI16 on δ = 0 learns (held-out 0.97 at 6 % of the budget; the beam's kept route is the oracle's on 0.95–1.00 throughout; the stray rule never forms). IW16 on δ = 1 collapses by step 300 into "SET at every state, never halt" — identical samples give zero leave-one-out advantages — so P1 is stated as open.

**Decision.** D104: spurious-route density as a dose (fewer answers, heavier length tails) is the first follow-up of rung 2, not run in this grid, for budget.

## Entry 65 (29 Oct 2026): the architecture thesis restated; the rungs re-oriented (discussion)

**Premise (researcher, in his words, condensed).** Computational value is important for the elasticity of a model, but learning a million different layers at train time is infeasible. The layers of a very deep model are each learning a function that could be composed serially from basis functions, so an effective model with millions of unique layers can be obtained tractably from primitive layer functions. The router also allocates computation per token: a token that needs to attend back to the context can, without every token paying for one static graph. A layer is a high-dimensional function approximable like a Taylor or Fourier series whose terms are primitives. The router is like a PFN. Layers may commute, but their composition is a search process and cannot be treated as commuted. Routing is query and value assignment, and also learning the contextual value of a layer given the input sequence and the token's current layer.

**Discussion (Claude).**
- *Series vs composition.* A series is a linear span of m terms; composition of m primitives k deep reaches up to m^k functions (depth separation: Telgarsky 2016; compositional sparsity: Poggio et al. 2017; Kolmogorov–Arnold). The residual stream joins the two: each step adds a router-weighted sum of primitives, and depth composes the steps. Neural-ODE work already expands weights *in depth* in a basis (Massaroli et al. 2020; Queiruga et al. 2021). The thesis indexes the coefficients by *state*, which makes the field autonomous — rung 1's Markov result.
- *Commutation (researcher's correction, accepted).* The Magnus/BCH argument concerns a fixed route. The router's route is adaptive on intermediate states, so order matters even for commuting primitives. Commutation shrinks the search (the beam's state merging). The value of adaptivity is the state-blind gap rung 1 measured (F34).
- *Router signal (literature).*
  - Differentiable gates: soft MoE; Rosenbaum et al. 2019 and Mittal et al. 2022 on collapse and failed specialization.
  - EM over module choices: Kirsch et al. 2018; Neural Program Lattices.
  - Search-generated labels: expert iteration; Dr.LLM.
  - A matching-pursuit / functional-gradient oracle: dense and state-based, but blind to setup steps.
  - Local confidence for depth: CALM, LayerSkip, Mixture-of-Depths.
  - Value of computation: Russell & Wefald; Hay et al. 2012.
  - PFN-style amortized inference: Müller et al. 2022; DreamCoder's recognition model.
- *Pushback.* Contextual layer value is a Bellman quantity: setup steps have zero immediate value. A PFN generalizes within its prior's support, which is the same closure as F30. Posterior (which primitive is on the true route) and value (which is worth applying under cost) diverge under cost and ambiguity (the pilot's two basins, F33).

**Decisions.**
- **D105.** The framing is recorded at the head of the ledger's Direction section, with the ordered risk list and the experiment-design rule (L27). Future rungs are justified by the risk they retire.
- **D106 (proposed, not yet decided).** Rung 2 is re-scoped from "estimators" to "the router's training signal". The core is the cells whose outcome we cannot derive: current-policy vs fixed-prior posterior targets, each exact and sampled; reward; search. Spurious density is a second axis. Derivable cells become checks. The distillation curve (risk 1) is the leading candidate for the rung after.

## Entry 66 (29 Oct 2026): rung 2 revision 3 designed and reviewed; revision 4; CPU sanity runs set P3–P4

**Premise (researcher).** "Design and review it": rung 2 re-scoped to the router's training signal (D106), designed, reviewed by two independent agents (design, code), fixes implemented.

**Revision 3** (`f34b73d`; [DESIGN.md](../experiments/routefind/DESIGN.md)). Arms by signal: EX (exact current-policy posterior), IW16 (sampled), PF (exact fixed-prior posterior: the PFN signal, with an answer-only likelihood), TB16 (trajectory balance toward it), RL16 vs AC16 (Monte-Carlo vs bootstrapped value), EI16 (search + imitation); a 4-answer spurious-density dose; program identity E/L (two equally short programs per trigger, with an exact expressivity table).

**Reviews 3** ([design](../experiments/routefind/review/review3_design.md), [code](../experiments/routefind/review/review3_code.md)).
- *Design (do not run as written).*
  - PF is derivable: its fixed point, the target pooled over the one-lookup features, is the oracle's program in every cell, stochastic when sampled.
  - At step 0 PF's gradient equals EX's with PF's likelihood to 1 % per class. So P2 was not a causal test of F29: "whose posterior" contributes nothing at step 0, and the EX–PF difference is the likelihood. EX's own step-0 target is already correct in every class, so F29's basin is built by the dynamics. (Reproduced here.)
  - Success could be met by exact enumeration.
  - The critic was not zero at init.
  - On-policy VarGrad TB is the reverse KL (entropy-regularized), not "PF sampled".
  - The 4-answer dose also changed the readout.
  - PP precedence is an open endpoint, not a check.
- *Code.* `posterior_counts` exact (independent brute force to 3·10⁻¹⁵); `ac_loss` and `set_late` correct. Two bugs:
  - TB's loss was dominated by no-op moves at answers (log π₀ = log 10⁻¹²);
  - the value head's init shifted the RNG and was not zero.

**Revision 4** (`77ca591`–`445a9b9`).
- **Added:** EX-AL (exact current-policy marginal with the answer-only likelihood); PF-IS16 (PF's forward KL by self-normalized importance weights); ACMC16 (λ = 1); RL16 on δ = 0.
- **Demoted:** PF to a 2-seed check.
- **Relabeled:** TB16.
- **The dose** became a per-episode answer pool with the readout fixed.
- **Success** is restricted to candidates (no state enumeration), judged on the committed route within 3× EX in the same grid. Determinism is reported apart.
- **Program identity** is classified positively (early / late / other) on shortest-correct routes.
- **Other:** 5,000 steps; pilot seed 100; the value head zero-initialized behind a stop-gradient.

**CPU sanity (batch 16, 1,500 steps, seed 0; DESIGN §9).**
- *δ = 0.* EX and EX-AL stay in all-H; exact PF leaves all-H at ≈ 1,000 steps toward L.
- *δ = 1, exact PF* heads for L first, though its derived fixed point is E.
- *δ = 1, value arms.*
  - RL16: all-H.
  - AC16 with TD(0) collapses into halting at the start within 100 steps, even after a 200-step critic-only warm-up. This is a myopic trap: under the init policy the start state's value is below the 1/16 a halt at a key earns, and once all routes halt nothing explores.
  - TD(0.9) and ACMC16 avoid it and sit in all-H. The query-blind critic explains ≈ 5 % of the return's variance.
- *δ = 1, posterior-matching arms.*
  - TB16's first canonical form, which kept a log π(HALT | arrival) term, collapsed into PREV-to-dead. TB could shrink that term freely, making dead routes look under-weighted.
  - Corrected (cut routes end with no action; a separate −log π(HALT | arrival) loss), TB16 sits in all-H with its gradient clipped on ≈ 95 % of steps.
  - PF-IS16 sits in all-H, with a first L SET at 1,300 steps.

**Decisions.**
- **D106 decided:** rung 2 is "the router's training signal", as revision 4.
- **D107:**
  - AC16 bootstraps with TD(λ = 0.9); pure TD(0) is the check AC16T0.
  - The actor-critic arms train the critic alone for 200 steps first.
  - TB16 and PF-IS16 score canonical routes that end on arrival, with a local halt loss.
- **P2 restated:** where F29's basin comes from. EX-AL ≤ 1/3 on δ = 0 (the likelihood alone does not prevent it); PF-IS16 ≥ 2/3 (a sampled fixed target is not trapped).
- **P3:** PF-IS16 beats TB16 (forward over reverse KL).
- **P4:** no advantage from the learned value at this horizon. It is uninterpretable if RL16 sits at ceiling or floor.

**Rejected.**
- Keeping PF as a core arm (derivable).
- Reading "PF fails δ = 0" as "the task is the attractor" (a fixed target has no attractor).
- The 4-answer dose (confounded by the readout).
- AC16 with TD(0) as the bootstrapping test (it tests the myopic trap, not contextual value).

**Expected falsifiers.**
- EX-AL learning δ = 0: the basin was the readout's credit for halting, not self-consistency.
- TB16 ≥ PF-IS16.
- A learned value beating RL16 by more than a quarter of seeds.
- Every candidate stuck in all-H at 5k: rung 2 fails, and "learned from next-token loss" needs exact inference.

## Entry 67 (29 Oct 2026): rung 2 revision 4 re-reviewed (review 4); pilot ready

**Review 4** ([review4.md](../experiments/routefind/review/review4.md)): launch the pilot; three items to fix before the main grid. Verified independently:
- `canonical_routes` on six hand cases;
- the no-HALT cut route is the correct objective (the router's probability of a cut route is its prefix's);
- PF-IS16's gradient at large K matches exact PF's (cosine 0.996–0.999);
- the grids match §6.

**Fixed** (DESIGN §15):
- P3 gets a both-fail clause and a one-seed inconclusive band;
- P4 gets 5 ACMC16 seeds, ±1-seed bands and a "worse" falsifier;
- sampled shortest-correct accuracy is logged (determinism was not recoverable);
- PP is read on shortest-correct accuracy (L runs failed by construction);
- EX-AL's program is open in P5;
- the PF check row is split;
- IW16 is open in P6, with fraction margins;
- distinct sampled routes per query are logged (ESS hides collapse);
- EX-AL δ = 0 at 3e-3 is added to the pilot (27 runs; main 93).

**Noted for the ledger by the reviewer.** The reward arms (RL16, ACMC16, AC16) share the rung-1 readout's 1/16 credit for halting at a key, the same dilution P2 tests and the cause of AC16T0's halting trap. A refuted P2(a) would change how their failures read.

## Entry 68 (30 Oct 2026): rung 2 pilot ran — reward arms collapse, fixed-prior targets learn both triggers, EX-AL finds a count-shifted program

**Evidence.** 27 runs, seed 100, two T4s, ≈ 0.55 h wall per shard; [routefind/PILOT.md](../experiments/routefind/PILOT.md), [results_data/pilot/](../experiments/routefind/results_data/pilot/). One seed: nothing is a finding.

**Outcomes on δ = 1.**
- IW16 and EI16 reach a shortest correct program at lr 3e-4 and 1e-3 (criterion 1,600–1,800 and 800–1,300 steps), IW16 in E and EI16 in L, both deterministic, both failing PP (IW16 by F30's SET>PREV precedence; EI16 because L cannot read the flag at a PP arrival).
- PF-IS16 reaches it at every lr (1,400–2,500), in L, stochastic at its optimum as derived (sampled shortest-correct 0.13), with 12–13 distinct routes per query throughout.
- TB16 climbs to 0.97 / 0.84 (k = 8) and then flattens as the lr decays; its gradient is clipped on 65–85 % of steps.
- RL16, ACMC16 and AC16 become deterministic within 200–400 steps at every lr (16 identical samples per query), freeze in all-H or in halting, and never move again.

**Outcomes on δ = 0 (lr 1e-3).** PF-IS16 learns L at 1,100 steps and TB16 at 3,800, both passing PP. EX-AL reaches answer accuracy 1.00 on in4, in8, none8 and len32 with exactly one extra step on every word: a count-shifted program (SET once at the start; the oracle's decisions one count higher; PREV at c = 2), inferred from the endpoints. RL16 collapses into halting, AC16 into all-H.

**Argument.**
- The reward arms fail by exploration collapse, not credit assignment: once the samples are identical, the leave-one-out and critic advantages are zero. P4's preregistered floor condition is met; contextual value cannot be read without an exploration term.
- The fixed-prior targets learn where the exact current-policy marginal stalls (F29), as P2(b) predicted, and keep their sample diversity because their optimum is a soft posterior.
- EX-AL's count-shifted program is F29's mechanism from the other side: the init phase's labels are kept and the program is built around them in the spare count capacity. P2(a) holds as stated (not shortest), but its reading was binary where the outcome is not.

**Decisions.**
- lr 1e-3 for every signal (the pilot rule).
- **D108:** the reward arms run as 2-seed δ = 1 checks; their δ = 0 and dose cells are dropped (93 → 72 runs, ≈ 9 T4-hours). The design's predictions are unchanged.
- The first follow-up for risk 4 is an exploration term for the reward arms (entropy bonus or ε-mixing), after a CPU sanity run and review; not added to this grid.

**Expected falsifiers at the main grid.** EX-AL δ = 0 seeds that reach the shortest program (then the likelihood alone suffices, P2(a) refuted); EX-AL seeds that stay in all-H (then the count-shifted program is a seed accident); PF-IS16 seeds that collapse like IW16 at 3e-3; TB16 reaching criterion on δ = 1 in a majority (P3 refuted); the PF check landing in E (the pooled fixed point stands and PF-IS16's L is a transient).

## Entry 69 (30 Oct 2026): rung 2 ran — three non-enumerating signals learn the program; F29's basin is self-consistency; rung 2 closed

**Premise.** "Check results": the main grid ran overnight on the full preregistered 93-run grid at `705e345` (the researcher's checkout predated the D108 cut; the cut was not applied, which is the cleaner record), lr 1e-3 everywhere per the pilot rule, 5.7 h wall on two T4s, no failures. [routefind/RESULTS.md](../experiments/routefind/RESULTS.md); data in [results_data/main/](../experiments/routefind/results_data/main/).

**Evidence (counts per cell; details in RESULTS §1).**
- δ = 1: EX 3/3 (800 steps, E); EX-AL 2/2 (800, E); PF 2/2 (1,050, **E**); IW16 4/5 (1,300, E; the fifth is program R); PF-IS16 5/5 (1,200, L-leaning mixtures); EI16 5/5 (900, L); TB16 0/5 (0.97–0.98 and rising at 5k); RL16, ACMC16, AC16 0/5 each (collapsed by step 200–400).
- δ = 0: EX 0/2, EX-AL 0/3, IW16 0/2 (all-H); PF 2/2, PF-IS16 3/3, TB16 2/3, EI16 2/2 (L, passing PP); RL16 0/3 (halting), AC16 0/3.
- Dose (answer pool 4): EX +19 %, IW16 +31 %, EI16 +44 %; PF-IS16 0/3 at criterion (0.91–0.94, mixed program); TB16, RL16, AC16 0/3.
- Checks as derived: SI16, ELL16, EI16B 0/2; AC16T0 0/2 (halting).
- PP: no δ = 1 learner passes (E by precedence, L by the read); every δ = 0 learner passes. Determinism: EM and imitation 1.00; posterior-matching 0.05–0.18.

**Predictions.** P1 held (4/5, E). P2 held in both halves (EX-AL 0/3; PF-IS16 3/3 in L). P3 held (5/5 vs 0/5), TB16 slow rather than dead. P4 not read (reward arms at the floor, as preregistered). P5: E/L lines held, PF resolved to E (the pooled fixed point), PF-IS16 to L-leaning mixtures, **EI16's init-beam line refuted 10/10**. P6: EI16 clause held, **PF-IS16 clause refuted** (it degrades most).

**Argument.**
- Risk 3 is retired for the toy (F35): the router's signal does not need exact inference; sampled EM, a sampled fixed-prior posterior and search + imitation all work within 1.6× of the exact marginal.
- F29's cause is identified by intervention (F36): with the likelihood held fixed (EX-AL vs PF share their step-0 gradient), every target computed under the router's own policy stalls on the local trigger and every fixed-prior or searched target learns it. The basin is self-consistency.
- The signal decides the program (F37): current-policy posteriors and exact PF → E; imitation → L always; the sampled fixed-prior estimator → mixtures, because self-normalized weights with the router as proposal are partly current-policy.
- A third correct program exists (F38, R): one SET at the start, decisions one count higher, built around the stray-phase labels in the spare count; found by IW16 (1/5) and by EX-AL on the pilot seed, not at the main seeds (L30).
- Posterior matching gives a stochastic router whose argmax is a program (F39); reward-only signals die of exploration collapse before any credit question arises (F40); the dose hurts the fixed-prior target most (F41).

**Decisions.**
- **Rung 2 closed.** I29 closed; F35–F41; X27–X32; L30–L31.
- **Risk 4 (contextual value) is blocked on exploration**, not on value estimation: the first follow-up, when the program returns to it, is an entropy or ε term for the reward arms, with a CPU sanity run and a review. Not run now.
- **Metrics carried to rung 3:** program identity with R as a category, shortest-correct and answer accuracy both, committed and sampled readings both.
- **Next rung** per the Direction's risk list: risk 1 (the basis claim) or risk 2 (learnable generic primitives), to be designed; D106's note stands.

**Rejected.**
- Reading TB16's 0/5 as failure (its curves say 3× slower; L31).
- Reading EI16's L as a beam-init effect (10/10 L whatever the init beam returned).
- Reading the dose as "spurious density" in Min et al.'s sense: for a Markov router the only consistent spurious program is all-H, and what the dose did was spread the fixed-prior posterior.

**Expected falsifiers of the carry-over.** A neural-state rung in which a fixed-prior target also stalls on a local trigger (then F36's mechanism is specific to the explicit state); an exploration-fixed reward arm that still fails (then value signals have a credit problem, not only an exploration one); EI16 landing in E under a different beam merge rule (then the merge is the mechanism of X27).

## Entry 70 (30 Oct – 1 Nov 2026): what rung 2 means for the thesis; the fractal picture; branching and uncertainty; the risk list revised (discussion)

**Premise.** Four conversations after rung 2 closed, recorded here because they changed the Direction section (ledger, 1 Nov update) and set rung 3.

**1. Rung 2 against the thesis (30 Oct).** The inference half of the router (amortized "which primitive next") survived its first test without exact inference (F35). The PFN framing became a mechanism: the only thing separating the signals that escape F29's basin from those that do not is whose posterior the target is under (F36). The PFN target has two measured costs: a stochastic router whose committed route is the program (F39), and fragility to coincident routes (F41), which reversed the imported intuition that averaging is the robust signal under spurious solutions. The value half could not be read because exploration failed first (F40); posterior targets did the setup-step credit assignment with no value function. Program identity became a variable set by the signal (F37, R). F30's bound held under every signal.

**2. Ties and value (30 Oct).** The researcher: two programs identical on every training world are identical for every purpose that matters; more varied data break ties, and at scale undisambiguated ties are rare. Claude conceded the strong form of its earlier claim ("the data cannot decide" should be "the distribution we gave it cannot decide") and kept two reservations: the ties that survive all the data are the off-support behaviour the thesis is about, and data break a tie only if the learner can still move (F36). On value: a PFN has no value function but computes one (the backward message β of the route posterior) and never isolates it; AlphaZero keeps a value head because it searches at play time, its prior is self-play (self-consistent, can lock in), and its targets are search-improved. Synthesis: search supplies the targets, a fixed prior over programs breaks the ties the data leave; a learned head only with search at inference or targets too deep to compute.

**3. The fractal picture (31 Oct).** The researcher, condensed: current practice fits the data distribution into parameters; the distribution is fractal, with detail at every zoom; bigger models fit finer detail; that path is near-doomed because the distribution is huge; the alternative is to spend parameters on how computation composes (a compressed prior over composition) rather than on the data itself, keeping high-dimensional functions as the basis. Claude: the metaphor supports the thesis for a reason not stated: a fractal is a short generator iterated, so resolution costs iterations, not parameters, which is elasticity in its purest form and what F30 and rung 0 measured; the theory of fixed-depth limits agrees. Against: the distribution is grammar-like, not self-similar; the loss has a particular part (facts) with no generator, where memory is correct and much of scaling's gain lies; "doomed" is too strong; MoE already routes with learned parameters and bought nothing beyond parameter count, so the claim must be about iterated composition with state, not routing as such; the navigation prior generalizes over a smaller space but inherits inductive bias (F30, F37); and the picture presupposes the basis, which is where the program's strongest negatives are (F22–F23, F32). The researcher let the "compression" angle go.

**4. Chain of thought, loops, branching, uncertainty (31 Oct – 1 Nov).** The researcher: CoT gives the model a computational register in context and lets it explore the distribution by emitting tokens; a looped transformer does the same internally but its pathfinding is worse, perhaps because it cannot branch the way a slime mold does; a routed serial graph with a fork could run both branches and keep the better one. Then: the model's uncertainty about its next token is never a function of anything; the model is blind to it and so are we; on hard tokens a static graph is confidently wrong or diffuse and should be able to sharpen itself; with a full-coverage basis, intelligence is composition quality; the irreducible memory is not that big but needs to be continual. Claude: what CoT provides that a loop lacks is a writable register and a per-step commit, not branching (F29, F31 against F32; F39 for commitment). Branching has three forms, and only training-time search has evidence; at inference "keep the one we like" needs a judge, and every candidate judge is the contextual value function. On uncertainty: the overstated half is that models do represent it (calibration, probes, ACT / CALM / MoD; rung 0's halting head); the sharp half is that in a static graph the estimate changes nothing downstream, so there is no control loop, and the toy router never closed it either. Sharpening is right only for epistemic uncertainty ("would more computation change my answer"), which is the value of computation and not the prediction's entropy; a single pass cannot measure it, breadth (disagreement among computations) estimates it, depth reduces it, and the router's policy entropy marks the forks. The missing piece of "continual memory" is abstraction of reused compositions into primitives (DreamCoder), which keeps inference tractable as programs deepen.

**Decisions.**
- **D109.** Rung 3 = *commitment* ([commit/DESIGN.md](../experiments/commit/DESIGN.md)). Core A: plasticity after commitment, the live disagreement, from committed routers under two data shifts with the training signal as the treatment (risk 8). Core B: exploration for the reward signals, so that risk 4 becomes readable (H14). Checks C: disagreement among sampled routes against the exact marginal, and policy-entropy localization (H13), from saved routers. Phase 1 saves checkpoints; the derive step (route coverage at the committed policies) sits between phases.
- **D110.** Rung 4 target: the distillation test of risk 1 with two curves, fidelity vs number of primitives and fidelity vs iterations past the teacher's depth.
- **D111.** The Direction section carries the sharpenings and the revised risk list (risks 8 and 9 added; 3 retired; 4 reframed); I30–I33; H12–H14.

**Rejected or let go.**
- "Fitting in parameters is doomed" (too strong; the particular part of the loss is real and large).
- Composition as compression as the organizing angle (the researcher set it aside).
- Branching at inference as the reason for routed over looped (reduces to the value problem).
- Entropy of the prediction as the control signal (aleatoric; sharpening it is wrong).

**Expected falsifiers.** Every signal moving under a minority of disambiguating data (then ties do not matter and F36 was a from-scratch artefact); exploration-fixed reward arms still failing (a credit problem, not only exploration); disagreement among routes uninformative about committed error (then breadth is not an uncertainty estimator in this carrier); a distilled basis plateauing at the teacher's depth (the fractal reading fails).

**Addendum (1 Nov, later).** Design review 1 of rung 3 ([commit/review/review1_design.md](../experiments/commit/review/review1_design.md)) found that revision 1's core shift did not do what it claimed: PP words do not break the E/L tie (a PP run admits at least five shortest route types, and an L policy with three off-support decisions completes 0.90 of them), the fixed-prior target drifts from L to E with no shift at all (the flat prior's SET padding; also derived here, [commit/derive/pooled_pp_120.txt](../experiments/commit/derive/pooled_pp_120.txt)), the doses were misstated five-fold (PP words sit only on chains of length ≥ 4: 0.7 % and 4.6 % of queries, not 5 % and 25 %), core B lacked the arm that answers its question, and nothing defined "committed". Revision 2 makes the local-trigger shift from committed deferred-trigger routers the core (one in-support decision relabeled by every P-word, dose = p_P), demotes PP words to a filling-in test with f = 0 controls, adds the commitment gate, margins, a teacher-forced ceiling and matched references, runs the beam-supported reward arms unconditionally and reads P4 there, and sets the open rows from per-seed coverage at the checkpoints. **D112:** rung 3 as revision 2; implementation next, then a code review and a CPU sanity run before stage 1.

## Entry 71 (1 Nov 2026): rung 3 implemented, reviewed, sanity-checked; stage 1 ready

**Implementation** (`3715b02`, fixes `3489290`, `7990048`): checkpoints (`save_state` / `init_from`, `run_route --ckpt-dir`, Kaggle upload and fetch), a constant schedule with a short phase-2 warm-up, clean-context evaluation beside the mixture strata, program identity on isolated-P words, PP set placement keyed by run length, every completion of the ε-mixed beam with three losses over that support (PFB16 soft target, RL16-B policy gradient with self-normalized weights, AC16-B with the value head), an entropy bonus, `gate.py` (the commitment gate, coverage in three forms, both margins, EX's posterior mass), grids and analysis; 11 tests, the rung-1/2 suites green. **Code review 1** ([commit/review/review1_code.md](../experiments/commit/review/review1_code.md)) verified the completion set, the PFB weights, the RL-B gradient and TD(λ) independently, and found three bugs (the gate script missing from the Kaggle bundle, an import-order crash in the analysis, the PP classifier over-counting), all fixed; it also found that **the beam under an E-committed router never keeps the δ = 0 oracle route**: E's per-episode route and the oracle's each have one ε-floored action and end at the same product state, and the merge keeps the more probable prefix, decided by π(SET | predecessor) ≈ 2·10⁻⁶ against π(PREV | set-here) ≈ 10⁻⁷. So from the E start ε is not a coverage knob and imitation's target is a route the router cannot learn as a rule.

**CPU sanity** (DESIGN §9; one seed, batch 16, 1,500 steps). An EX router committed to E (margin HOP − PREV at the relabeled state +16.9 logits; the oracle route in no beam, no sample, and at 2·10⁻⁵ of its own posterior). Continued under δ = 0: **PF moves** (the margin falls to +1.4 in 100 steps, negative by 500; shortest-correct on P-words 1.00 at 500, program L); **EI16 is stuck** (0.00–0.02 throughout, margin 8.5–12.8). RL16-B from scratch on δ = 1: the self-normalized support weights put all their mass on the policy's own route from step 200 (effective sample size 0.004 of the completions) and it sat in all-H.

**Decisions.**
- **D113.** The forced-breadth reward arms (RL16-B, ACMC16-B, AC16-B) are 2-seed checks; PFB16 (the forced-breadth posterior) is the core-B arm; rung 2's P4 is not read in rung 3. The reason is conceptual as well as empirical: a reward signal that keeps exploring has to weight unsampled routes by something other than its own probability of them, and every such weighting is a posterior target. Risk 4 is reposed as per-step credit inside a posterior target (ledger H14, I33).
- Stage 1 = 48 runs (phase 1: EI16 7, EX 7, PF 3, PF-IS16 5; core B: PFB16 5 + 3 + 3, PF-IS16 dose reference 3; checks). Stage 2 is generated from the gate report after the derive step.

**Expected falsifiers at the grid.** The CPU contrast (PF moves, EI16 stuck) failing to replicate at five seeds; RT not moving (reachability after all); PFB16 landing in E (the merge-rule derivation wrong); a reward arm learning over the beam support (then P4 is read in a follow-up at five seeds).

## Entry 72 (2 Nov 2026): rung 3 stage 1 ran; the derive step; the pre-stage-2 re-review corrected three of its readings; stage 2 ready

**Stage 1** (48 runs, 2.7 h on two T4s; [commit/results_data/stage1_report.md](../experiments/commit/results_data/stage1_report.md)). Phase 1 committed 7/7 EI16 routers to L and 5/7 EX routers to E (criterion 800 [700–1,000], deterministic, not R); the gate excluded EX seed 4 (0.90–0.93, margin falling) and EX seed 0, which **learned E by step ≈ 700–1,000 and collapsed at ≈ 2,000** into halting at flagged pairs under its own signal at the constant lr (ledger F44). Core B: PB2 held (RL16-B, ACMC16-B, AC16-B 0/2 with the support weights' effective sample size at 0.01; RL16+H 0/2 at every β) and PB3 held (PFB16 5/5 on δ = 1, 3/3 on δ = 0, program L, stochastic; a tie in count with the matched PF-IS16 reference at the dose) — ledger F42, F43; H14's first clause refuted for both exploration fixes.

**The derive step** (`gate.py`, 256 episodes per router, 271 P-word queries; [commit/results_data/gate_stage1.json](../experiments/commit/results_data/gate_stage1.json)). At the committed E routers the shift relabels exactly one product-state class (HOP → PREV at +21 to +32 logits, SET already in place): the own-posterior target puts 0.00 of its mass on the new route, 16 samples never contain it, and the ε-mixed beam keeps it on 0.73 and 0.47 of P-words for two routers and ≈ 0 for three — **the merge tie breaks per seed, not by ε** (ε moves one router from 0.02 to 0.22, none across 0.3). From every L router the beam keeps the oracle's route (0.58–0.66); on the words that must move, the own posterior sees it on 0.00–0.01 (L s3 0.20, s5 0.05) and sampling on 0.00. L seed 0 has HALT, not PREV, as the argmax at the relabeled class (p(PREV) 0.23 against 1.00 on the other six). The open rows of §7 were set from these per seed ([DESIGN §7a](../experiments/commit/DESIGN.md)); PA4 was withdrawn before test (X33).

**Re-review 2** ([commit/review/review2_stage2.md](../experiments/commit/review/review2_stage2.md); an independent agent with the checkpoints) found three blocking faults in the first draft of §7a, all mine: (B1) the L-start rows for EX, IW16 and PF-IS16 were set from a coverage averaged over the 13 of 271 P-words every L router already solves under δ = 0 (own posterior ≈ 1 there), which turned ≤ 1/3 rows into "reported" and inverted the PA4 replacement (an EX move from L s0–s2 would *falsify* the coverage reading, not confirm it); (B2) the dose cells' strata were generated at the training p_P, so `in4` held 33 P-words at p_P 0.08 and the movement endpoint would have failed a truly moved router a quarter of the time, in PA2's predicted direction; (B3) EX seed 0 was described from its endpoint as "sat in all-H for all 2,500 steps" when its curve shows it learned and collapsed, which changes what the EX stability control is for and requires a *collapsed* outcome. Handling ([DESIGN §14](../experiments/commit/DESIGN.md)): coverage conditioned on unsolved words in `gate.py` and the gate rerun at 256 episodes; the three rows set to ≤ 1/3 and the PA4 replacement rewritten as per-signal counts with "L is easier" for EI16 only; the standard-configuration evaluation (p_P 0.3, pp_frac 0) recorded for every non-standard stream and read by the endpoint, the dose cells' criterion set at p_P 0.3, with a test; outcomes moved / locked / collapsed / other in the analysis with the first-deviation histogram per run, movers counted over non-collapsed runs; the EX stability control on all five E routers; EX from L s3 and s5 added as per-seed reported rows; p(PREV) and p(SET) at the relabeled classes on the criterion curve; the pool seed offset for continued runs (the controls had been set to replay their own phase-1 batches); the AUROC read within EI16 and pooled without PF / RT; filling in at pp8 ≥ 0.90 with the route-type split; relabel events beside steps for the dose. Tests: 61 (`tests/test_commit.py` 13; rung-1/2 suites green).

**Decisions.**
- **D114.** Stage 2 = 120 runs ([commit/grids/cm_stage2.json](../experiments/commit/grids/cm_stage2.json)): shift A from E s1, 2, 3, 5, 6 at p_P 0.3 (RT 2; PF, EX, EI16, PF-IS16, IW16 × 5; EI16 ε0.3 × 3) and at 0.15 / 0.08 (5 signals × 3); from L s0–2 (RT 2; 5 signals × 3) and EX from L s3, s5; stability EX × 5, EI16 × 2; from-scratch δ = 0 references 4 × 3; shift B from L (RT 2; 4 signals × 3; f = 0 controls 4 × 2). ≈ 10 T4-hours, one Kaggle session, from the stage-1 checkpoints.
- **D115.** A run that leaves its committed solution for halting is *collapsed*, excluded from mover counts and reported with the stability controls' collapse rate; if the controls collapse at a similar rate, phase-2 EX is unreadable at this lr and is rerun lower.
- Process: L32 (a derive-step quantity is conditioned on the cases the prediction is about; outcomes are read from curves; the dose must not change the ruler; re-review the derive step).

**Launch note (2 Nov, 16:41–16:54).** The first stage-2 launch stopped at the Kaggle test gate without running a single run: the new `test_phase2_outcomes_on_synthetic_records` imports `experiments/commit/analyze.py`, which the Kaggle bundle does not ship (it carries `gate.py` only; analysis runs locally), so the test failed with `FileNotFoundError` and `run_on_kaggle.py` refused the grid (`TESTS FAILED: grid not run`). My omission: the test was written after the bundle contents were last checked. Fixed by skipping that test when the file is absent (verified in a bundle-shaped tree: 1 passed, 1 skipped); relaunched. Cost: ≈ 4 min of session time, no results affected. **The relaunch (17:00) failed the same way with the OLD test file:** the launcher pushes the code as a new *version* of an existing Kaggle dataset, version processing is asynchronous, and `kaggle datasets status` answers "ready" for the previous version, so the kernel mounted the stale bundle (the traceback shows the pre-fix test; the Mac's file had the fix). Never seen before because every earlier launch used a fresh grid name, hence a fresh dataset. Fixed in the launcher: a fresh dataset slug per launch (kernel name + launch time; nothing to be "ready" but the new one), and the bundle's commit hash and time written into `job.json` and printed at the top of the kernel log, so a log says which code ran. `tests/test_fleet.py` green.

**Expected falsifiers at stage 2** (DESIGN §7, §7a). PA1: every signal moving from E in ≥ 4/5, or only RT and PF; IW16 ≥ 3/5. The per-seed EI16 row: any E router on the wrong side of its coverage (s1, s5 stuck or s2, s3, s6 moving). PA2: a signal moving at 0.08 but not 0.3; PF ≤ 1/3 at 0.08. PA3: AUROC ≤ 0.6 within EI16. From L: EI16 ≤ 1/3 (or ≥ 3/5 from E); EX, IW16 or PF-IS16 ≥ 2/3 from L s0–2. RT not moving anywhere (reachability; rerun longer). The stability controls collapsing at EX's phase-1 rate or above.

## Entry 73 (2–3 Nov 2026): rung 3 stage 2 ran; the results were independently reviewed and four of my readings overturned; rung 3 read

**The runs.** Stage 2 launched three times: the first stopped at the Kaggle test gate on a test of mine that imported a file the bundle does not ship; the second ran the previous bundle because the launcher pushed a new *version* of an existing dataset and Kaggle's status call answers "ready" for the old one (both in Entry 72's launch note; both fixed in the launcher, which now creates a fresh dataset per launch and prints the bundle's commit at the top of the kernel log). The third ran 120 runs in 2 h 20 (bundle `ba5bec6`; 60 tests passed, 1 skipped on the box). Records are outside git; [commit/RESULTS.md](../experiments/commit/RESULTS.md) and [commit/results_data/stage2_report.md](../experiments/commit/results_data/stage2_report.md) carry the analysis.

**Two bugs in my analysis before any reading, and four readings the reviewer overturned.** The review-2 `collapsed` rule had a clause on a key that does not exist in the records, so it was true for every run below 0.5 answer accuracy and the first automatic table called 61 of 94 runs collapsed; and the first-deviation key `SET>HOP` means the intact L program from the L start and the lost SET from the E start, so the taxonomy had to be conditioned on the start. I fixed both, wrote a first draft, and sent the records and the draft to an independent reviewer ([commit/review/review3_results.md](../experiments/commit/review/review3_results.md)), who recomputed every cell and time from the records (all reproduce) and found, from the curves: that EX's halting is a **transit** that 10 of 11 EX-from-E runs enter within 0–500 steps and that the EX "moves" are escapes from it (I had read the end-state histogram as if halting and moving were alternatives); that one IW16 run held the moved program for 1,000 steps before collapsing (I had written "never moved"); that the E-start EI16 cascade does not generalize to the L start, where coverage is 0.72–0.84 from step 0 and one router never moves; and that PA3's preregistered AUROC had not been computed (0.705 within EI16, between its bar and its falsifier; 0.821 pooled without PF / RT, which holds). Plus ten should-fix items (two 0.949 runs counted differently; PA2's PF "time grows" not supported — the movers were faster at the lower dose; PF's plateau has a higher loss than its escaped state and PF reaches the plateau from L by destroying the correct PREV; IW16 s2 is not program R; the preregistered "RT does not move → unreadable" rule applies to L s1 and to L s0 under shift B; a mechanism for the L start I had not considered). All incorporated; RESULTS §12 lists them. Ledger L33.

**What rung 3 says** (RESULTS §1, §10; ledger F45–F50, X34–X39, H12, risk 8).
- Plasticity after commitment is a property of the signal: from the E-committed routers under the local-trigger shift, RT 2/2, EI16 4/5, EX 3/5 (+ 1 R-like), PF 3/5, PF-IS16 1/5, IW16 0/5 at the end. The own-policy samplers do not merely fail to move; they take the committed solution apart (halting; the all-H route), PF-IS16 even on unchanged data. Commitment inverts two signals against scratch: EX 0/3 → 3/5, PF-IS16 3/3 → 1/5 and 0/3. The preregistered reading row "sampling, not self-consistency, is the lock" applies.
- The gate's numbers predicted less than the design assumed, and in a different form: for search, coverage decides whether the beam's merge tie can reverse (collectively, inside one 100-step window), not when; for the exact own posterior the gate's mass was not the quantity — its moves pass through a halting state the shift induces (6/11 against 1/12 on unchanged data); the exact fixed prior parks every router, from either start, on one plateau (p(PREV) ≈ 0.15, loss 2.4) and escapes stochastically to the target state (0.65, 2.2), 0/3 at the lowest dose.
- From the L start one router resisted teacher forcing for 3,000 steps (p(SET) at the flag pinned at 0.15–0.22), which by the design's own rule makes that cell unreadable; the reviewer's reading — the L routers never formed the own-flag feature at the flag state, so their missing SET is a feature to form, not a decision to flip — is cheap to test on the checkpoints (I35).
- Filling in (shift B) did not happen at 0.90 under any learned signal; PF's f = 0 drift L → E held as derived.
- Scorecard: PA1 fails for PF (3/5), EX (3/5) and the per-seed EI16 rows (s3, s6); PA2 fails for PF at 0.08 and its time clause; PA3 neither held nor falsified within EI16, held pooled; the PA4 replacement fails for RT and "L easier"; the §7a falsifier for the EX coverage reading fired; PA′1 fails except its PF clauses; PB1–PB3 held (stage 1); PC1 not run.

**Decisions.**
- **D116.** Rung 3 is read and closed in its stage-2 form; checks C (PC1) remain open on the saved PF / PF-IS16 checkpoints and are not a gate for rung 4.
- **D117.** The trainer of a committed router in later rungs is an exact target or a searched support; the own-policy samplers (IW16, PF-IS16) are out for that role. PFB16's plasticity (F42) is the first check of whichever later rung needs a scalable signal; it was not run in rung 3.
- **D118.** Two derive-step questions on the saved checkpoints before any new training on this carrier: the L routers' logit separation at the flag states (I35) and PF's plateau (I34); the λ = 0 EX arm if F48's reading is to be tested. None of these diverts from rung 4 (D110), which is unaffected.

**Rejected or let go.** "The router decides, not the signal" (my first headline for the L start; the outcome on L s1 is signal-dependent). "EX moved although blind" (the moves did not start from the committed policy). "Coverage is the switch for search" as a general statement (E start only). The §7a rule "movers counted over non-collapsed runs" (halting is a transit).

**Expected falsifiers for what is now believed.** PFB16 failing to move a committed router where EI16 does (then the searched support is not what makes EI16 plastic); the L checkpoints' logits separating flagged from unflagged states at the flag (then I35's reading is wrong and L s1 is a budget question after all); an EX arm with λ = 0 still losing the SET under the shift (then the step cost is not the mechanism of F48); PF's plateau absent under a different lr or batch (then it is an optimizer artefact, not a property of the target).

## Entry 74 (3 Nov 2026): the guiding star restated by the researcher; the two-step decomposition; chains against bags; rung 4 redefined (design draft)

**The correction.** After rung 3 I had been reading the ladder against "see the minima on the horizon instead of walking the loss landscape" as a claim about the learning process. The researcher corrected the emphasis: the thesis is that **the router is a PFN over programs under the data prior** — given the primitives and the prior over their compositions, inference at test time is amortized routing of programs; we are not studying a walk over routes. Under that frame rungs 2–3 read more simply than they were written: the signals that are PFN-like (PF, the exact posterior under a fixed prior; search, the same posterior approximated) stayed plastic and carry ties as mixtures; the self-training signals (the router's own policy as its prior: EX, IW16, PF-IS16 from a committed policy) commit and cannot hear the data. Rung 3's halting transits and plateaus are the failure mechanics of the design not to use plus the amortization lag of the one to use. The "live disagreement" about ties (ledger 1 Nov) dissolves the way the researcher said: a PFN carries a tie until data break it.

**The decomposition** (the researcher's): *(1)* assume the corpus comes with the optimal program per token, train the PFN on it, test whether it generalizes across programs; *(2)* if it does, ask where the labels come from. Discussion: "optimal program" is a prior in disguise (programs are many-to-one onto outputs; rung 2 showed what imposing one label does), so the honest label is the posterior over programs under the prior and the PFN should be tested for calibration over ties; most tokens of a real corpus have a lookup as their optimal program (the generated / particular split), so a labelled corpus must make that split or it teaches "retrieve". Step (1) already has a small version in the ladder: teacher forcing on oracle routes generalized exactly in depth (F30) because programs are closures of per-state decisions, and rung 3 showed the case that does not come from labels (a decision never taken at a state, F49). So step (1) splits into closure (a check), **recombination of decisions across worlds under a cue** (the amortized-inference question), new decisions (a check) and **rules against tables** (count extrapolation). Step (2)'s candidates, with what rungs 2–3 say about each: a synthetic prior (PFN / DreamCoder dreaming; capped by the prior, F41 and §5.1), search on real tokens (rung 2's most robust signal; plastic where its support reaches, F45–F46), a teacher model (rung 5), an exogenous reasoner (Entry 73's blog discussion), circuits read off a trained model; self-training excluded.

**Chains against bags.** The researcher's picture: everything a model learns is a heuristic; deep layers must do many *linking* steps to produce a working heuristic rather than a bigram or a key–value lookup. Read against the carrier, a deep heuristic is a lookup whose key had to be constructed through several links, and the basis is "one lookup plus a few key-construction operations" — the basis claim in its most specific form, with a prediction for distillation: fidelity saturates at a small m because deep layers apply the same linking step repeatedly. The counter-picture (deep layers as *wide* sums of shallow heuristics; the bag-of-heuristics and compositionality-gap results) is what a static once-per-token architecture can afford; chains are what the generated part needs. Rung 5's second curve (iterating past the teacher's depth) is where the two pictures separate.

**Decisions proposed (pending the researcher's answers to [programs/DESIGN.md](../experiments/programs/DESIGN.md) §11).**
- **D119 (proposed).** Re-sequence: rung 4 = *programs* (step 1: a mixture of worlds with a cue, recombination and count-extrapolation held out; RT as the labelled core at 10 seeds, EX / EI16 as unlabelled references, planted factorized and table routers); rung 5 = the distillation test (risk 1) with the chains-against-bags hypothesis; rung 6 = label sources. The case for the swap: step 1 tests the star's central claim directly on the carrier we have, with a held-out design we can state now, and rung 5 is uninformative if it fails; the case against is that a hand-written prior can only fail in the recombination / extrapolation split, so those cells must be sharp (the design's P2 and P3).
- Rung 3's derive-step leftovers (I34, I35) drop in priority under the PFN frame: they concern the walk.
- The design draft is revision 1 ([programs/DESIGN.md](../experiments/programs/DESIGN.md)); it goes to an independent design review after the researcher's three choices (held-out structure, cue encoding, demonstrations now or next).

## Entry 75 (6 Nov 2026): the architectural family named; the scaling claim stated; Lean fixed as the real-model target; the rewrite carrier designed

**The family.** The researcher asked whether the proposal is a type of MoE with dynamic halting and experts reused per layer that are not just MLPs. At the block-diagram level it is: Sparse Universal Transformer (Tan et al. 2023), MoEUT (Csordás et al. 2024), Mixture-of-Recursions (Bae et al. 2025), behind them Universal Transformers with ACT, and for the training-signal result Routing Networks (Rosenbaum 2018) and Modular Networks (Kirsch et al. 2018, EM over module assignments to avoid collapse — a direct precedent for F35/F40). Added to [literature.md](literature.md) with where the boundary sits: not the architecture; what the router is (amortized posterior over programs on an explicit state), how it is trained (posterior / search against gating), and what it does (F30, F37, F45–F49, rungs 4–5). The fact owed an explanation: MoEUT routes depth with shared experts and is parameter-matched-even with dense, not better.

**The scaling claim, stated precisely** (the researcher's): a dense transformer's parameters are linear in depth and a token needing k > L steps is unreachable at any width; a library of m primitives is constant in k, *provided* the compositions at depth k use the primitives of depth 8 — the basis claim quantified as m(L) saturating (rung 5's distillation instrument). The win is parameters and generalization to unseen depth, not FLOPs (serial steps). MoEUT's flat result is consistent with it on two readings: average perplexity is dominated by lookup tokens and the gain lives in the depth tail (so evaluate stratified by required depth, never the mean); and gating-trained routers may learn bags, not chains.

**The domain and the opponent.** Math, to minimize lookup (the researcher). The opponent there is dense-with-chain-of-thought, which already iterates with parameters constant in depth; the library's claim against it is compute per step (one expert against a forward pass plus decoding), no serialization of intermediate state, and a legible, searchable program. **Lean** is the real-model target: tactic = primitive, proof state = computational state, kernel = exact executor, proof length = depth, ambient theory = world, premise selection = the lookup part, structurally separated; labels exist (Mathlib traces); verifier-guided search exists (expert-iteration provers). First real model on synthetic symbolic math with derivation traces; Lean as the real-data step; natural-language math as a benchmark only.

**Distillation** is kept with a changed role: a calibration arm in the toy (recover a planted m) and the first measurement on a real teacher (m(L)), not the test of the basis claim on synthetic data, where the direct comparison is cleaner.

**The rewrite carrier** ([rewrite/DESIGN.md](../experiments/rewrite/DESIGN.md), revision 1): equational rewriting — terms, seven directed rules as primitives, worlds = axiom sets stated in context, BFS oracle, exact verifier, exact posteriors by DP on the reachable graph, ties. Rung 4 on it (symbolic): recombination across held-out axiom sets (P2 ≥ 6/10), term-size extrapolation as rules-against-tables (P3 ≤ 3/10), depth closure (check), labels against search, ties, one arm with worlds inferred from example proofs. Rung 5 on the same data (outline): the neural basis with the depth variable moved *inside* the step (lookahead needed to choose the next tactic), dense-L against a scratchpad against a library trained by labels, search and gating; stratified endpoints; distillation calibration. The pre-scale toy the researcher asked for.

**Decisions proposed (pending the researcher's §12 choices).**
- **D120 (proposed).** Lean is the real-model target; the toy ladder is re-based on the rewrite carrier; [programs/DESIGN.md](../experiments/programs/DESIGN.md) revision 1 is superseded (kept as record); the route carrier's results stand and it retires.
- **D121 (proposed).** Sequence: rung 4 (rewrite, symbolic) → rung 5 (rewrite, neural basis and internal depth, with distillation calibration) → first real model on synthetic symbolic math → Lean.

**Addendum (6 Nov, 10:41–10:53): the depth corrected, and an unknown held.** The researcher corrected the carrier's §1: proof length is the *external* loop (the kernel's, or a scratchpad's); the thesis's depth is per decision — at each next token the model branches over candidates, rolls each forward in the latent stream with its own routing and halting, and selects, to beat chain-of-thought at matched exploration. The design's strata become per-decision difficulty (B\*, D\*), the library arm gets the propose → rollout → value → select structure, the CoT opponent is matched in exploration budget, and the Lean endpoint is kernel calls and tokens per proof. The decision whether a rollout is latent only, executor-stepped, or split was explored (cost, fidelity and drift, training with the kernel available only at training time, generalization to new worlds as prior-only against prior-plus-dynamics, elasticity, legibility) and **held open at the researcher's request**: rung 4 does not depend on it; rung 5 measures it (the latent-against-exact lookahead curve) instead of assuming it. **D122 (proposed).** The three rollout forms are arms of rung 5, not a design choice; rung 4's results are reported under both readings.

## Entry 76 (6 Nov 2026): the rewrite carrier's design reviewed; revision 3

The researcher fixed revision 2's choices at the defaults; the design went to an independent review that **implemented the carrier in miniature** ([rewrite/review/review1_design.md](../experiments/rewrite/review/review1_design.md), with nine tables of numbers). Found sound: the carrier is small (≤ 6k reachable nodes per world at size 5), exact posteriors are feasible there, garden paths are real and deep under tree-edit distance (greedy succeeds in 0–0.4 of layer-sampled instances at k ≥ 5; lookahead-3 recovers 0.57–0.93 at k = 4), ties are real (8–156 shortest proofs at k = 4–6), cross-axiom arbitration exists in the triples and full set (10–52 % of first decisions). Found broken, all for reasons unrelated to the hypotheses: **(B1)** the random-walk generator accepts ≤ 3 % of walks at k ≥ 5 and nothing at k ≥ 7 in most worlds, so the stream would be 71–83 % one-step instances and the length strata empty; **(B2)** the world almost never binds — in 96–100 % of naive instances the full-world shortest proof already lies inside W, so recombination reduces to a validity mask and P8's null is undefined; **(B3)** a hand-written greedy-TED + mask policy already scores 0.925–0.985 on the held-out worlds, above P2's bar (L25 again); **(B4)** the categorical (rule, flat-position) head and absolute positions make size extrapolation fail by construction (rows never trained), so P3's "table signature" would have been an artefact; **(B5)** D\* as defined was tie-dominated and ill-defined past the remaining distance; **(B6)** no in-distribution ceiling on the held-out worlds, and the "planted router" was the oracle.

**Revision 3** ([rewrite/DESIGN.md](../experiments/rewrite/DESIGN.md) §14): layer sampling with stated stratum weights; **the binding stratum (a missing axiom lengthens the proof — the Lean situation) as Q1's core**, with a derived cue-blind ceiling; a symbolic lookahead ladder as the reference on every stratum and P2 / P4 preregistered against it; a pointer head with tree-structural positions as the router, the absolute design as the identical-architecture control, so Q2 becomes a contrast with both outcomes informative; GP1 / D\*_sep / D\*_any with (σ, steps) ordering; RT-all as the ceiling; held-out pairs moved to the binding-rich {A, I}, {C, I}; I made invertible; protocol from curves with stratum counts, masked and unmasked readings, failure categories; rung 5's outline tightened (expansions as the unit, terminal evaluation per form, the CoT opponent's signal and selection rule stated). 41 runs; the dataset build is the cost.

**Decision pending:** the researcher's go-ahead to implement (tests first, derive step, pilot, code review, grid). D120–D122 stand as proposed.

## Entry 77 (6 Nov 2026): rung 4 implemented on the rewrite carrier; dataset built; independent code review 2 fixed five blocking items; pilot ready

**Implementation** (`src/topology_design/rewrite/`, `experiments/rewrite/`; commits 03fdb10 → 19e34d8). The carrier exactly as [DESIGN rev 3](../experiments/rewrite/DESIGN.md) §1: terms, the eight directed rules and fifteen worlds, the untyped executor and verifier, BFS, shortest DAG and path counts, the binding test, trace classes (Foata), TED/SED, GP1 / D\*_sep / D\*_any under the (σ, steps) order, the symbolic ladder, and the two exact posteriors (RT-post over shortest proofs as state sequences; PF's flat-prior forward–backward, checked against brute-force route enumeration). The generator does layer sampling in two phases (BFS, binding, GP1 before rejection; D\*, ladder, posteriors after), with rejection rates fixed per world from a 600-candidate warm-up and a 5 % acceptance floor. The router is the structural encoder (tree features, a relative bias over segment pair × path relation, a pointer head with HALT from SEP) and the parameter-matched absolute/categorical control. Training covers all eight arms; evaluation is committed rollouts masked and unmasked with failure categories and the per-instance ladder outcomes stored beside the router's, so every symbolic rung is read on the router's own denominator. 30 tests, including a planted oracle router run through the harness and the analysis tested on fixtures cut from real run records (L33). The choices §1–§6 left open are fixed in DESIGN §12a; the ones worth knowing: eval at sizes 6–7 is at k ≤ 4 (the radius-6 balls of size-7 terms reach 16–47k nodes in the triples and the full world); the acceptance floor keeps one-step instances in every stream; held-out training files (RT-all's stream) at 5,000 per world.

**The dataset** (`data/rewrite/`, outside git; ≈ 45 min per 10k-instance world on two cores). Realized training stream, 8 worlds × 10,000: binding 0.01 / 0.08 / 0.75 / 0.13 in {A} / {C} / {I} / {D} and 0.27–0.34 in the pairs; GP1 0.00 in {A}, {C} (no garden path exists there), 0.15–0.36 elsewhere; k ≥ 3 is 0.19–0.86 of a stream; distinct (t, s) 4.8–7.7k of 10k per world; the ladder on the training streams of the pairs: greedy 0.38–0.57, lookahead-2 0.47–0.70, lookahead-3 0.68–0.89. Eval sets (600 per world) populate the binding stratum at 0.26–0.53 in the multi-axiom worlds. The {A, C} k = 7–8 stratum exists and defeats every rung of the ladder within T (0/0/0 in the smoke build). Eval sets at sizes 6–7, the held-out training files and the PF pass were still building when this entry was written.

**Code review 2** ([review/review2_code.md](../experiments/rewrite/review/review2_code.md); the reviewer re-derived the lookahead values, D\* and GP1 by brute force (1,791 / 1,791 and 116 / 116 agreements), timed every arm, measured the encoding cache's memory, reconstructed the Kaggle bundle and ran the test gate inside it, and trained a router for 2,000 steps to probe the extrapolation inputs). Verdict "fix first", five blocking items, all cheap and all fixed (DESIGN §15): the bundle omitted the analysis module and fixtures the test gate loads, so the grid would never have started (B1 — the second bundle failure after Entry 72, L35); the encoding cache would have exhausted memory on the RT-examples shard (B2); RT-all would have silently trained on the eight training worlds and P2's ceiling been read against RT-canon (B3); **untrained embedding rows at sizes 6–7 would have decided P3 in both directions** — the structural router's depth/size/path rows beyond the training range (≈ 0.1 of size-6–7 success in the probe) and the control's learned positions past the training length (100 % of size-6–7 inputs) — review 1 had moved the artefact out of the head, review 2 found it in the inputs (B4, L34); the automatic P4 reading passed at zero success because the d = 1 and d = 2 rungs tie at zero on GP1 cells, and P2's two conditions were counted in separate seeds (B5). Should-fix items taken: the P3 ratio matched on (world, k, binding) cells; unmasked columns beside the masked pass rule, now stated in §6; P6 read as the total-variation distance to the RT-post target; PF targets on every training instance (the 2,500-instance PF stream would have confounded the signal with data size); lr swept for every arm in the pilot; P4 preregistered on cells where the rungs separate, with Q3's per-cell reach reported; the curve subset at 100 per world because the evaluation cadence costs as much as training; embedding init scales. Budget (reviewer's CPU timings extrapolated): RT arms ≈ 15–18 min per run on a T4 at 12k steps, RT-examples ≈ 25, EI16 ≈ 90–120 (its beam is Python-bound); the 41-run grid ≈ 18–20 GPU-hours, inside one session on two T4s with `--per-gpu 2` and little margin — the pilot's `seconds` decide the main grid's steps.

**Pilot** ([PILOT.md](../experiments/rewrite/PILOT.md)): 24 runs (every arm × lr ∈ {3e-4, 1e-3, 3e-3}), seed 100, 3,000 steps, curves every 500; it sets lr and steps per arm from the training-world curves and reads nothing else (L30). Launch pending the finished dataset.

**Decisions.** D123 (proposed, 6 Nov): the implementation choices of DESIGN §12a and the review-2 changes of §15 are part of the preregistration; the main grid's steps are set by the pilot's throughput, not its results.

## Entry 77a (6 Nov 2026): the rung-4 dataset finished and derived; P4's preregistered cells came out unreadable and were amended before any run; the bundle checked (L35); dataset handed over, pilot launchable

Addendum to Entry 77, which was written while the eval sets at sizes 6–7, the held-out training files and the PF pass were still building. Nothing in Entry 77 is rewritten.

**Eval at sizes 6–7** (`eval7_*`, 600 per world, cap 7, radius 4, k ≤ 4; 4–929 s per world). Binding 0.03 / 0.11 / 0.16 / 0.01 in {A} / {C} / {I} / {D}, 0.37–0.56 in the pairs, 0.41–0.53 in the triples except {A, C, I} (0.00); GP1 is thin at these sizes (0.01–0.21; P4 is read on eval5, so nothing the design counts on is lost); the ladder in the pairs: greedy 0.53–0.85, lookahead-2 0.72–0.93, lookahead-3 0.91–0.99 — sizes 6–7 at k ≤ 4 are easier for the symbolic rungs than sizes ≤ 5 at k ≤ 6, as the radius predicts. The eval binding target (0.50) was infeasible at the 5 % acceptance floor in {A, C}, {A, I}, {A, C, I}, {A, C, D} and {A, I, D} (realized 0.37–0.42 where the stratum exists); the files are used as built, with the realized rates in `meta.json` and derive §1.

**Why {A, C, I}'s binding stratum vanishes at sizes 6–7** (eval5 0.26 against eval7 0.00). In eval5 its binding instances sit on tiny terms — 95 / 176 at size 1, 39 / 126 at size 2, ≤ 10 % at sizes 3–5: the duplication-against-identity growth race (0 → (0+0)+(0+0) is two Db steps in the full world and three I steps inside {A, C, I}), which the size cap at 6–7 removes. A fact of the stratum, not a bug: binding in that world measures "D would be shorter", and D is only shorter on terms with room to grow. The full world's binding is 0.00 everywhere by construction (W is the whole rule set, d_W = d_full), so P2's second row and P5's held-out stratum are the pairs and the triples (ledger, derived constraints).

**Held-out training files** (RT-all's stream only; 5,000 per world; train targets): {A, I} binding 0.37 / GP1 0.29, {C, I} 0.42 / 0.26, {A, C, I} 0.23 / 0.23 (infeasible at the floor), {A, C, D} 0.28 / 0.42, {A, I, D} 0.26 / 0.54, {C, I, D} 0.29 / 0.30, {A, C, I, D} 0.00 / 0.29; 158 s ({A, I}) to 2,138 s ({A, C, I}) per world. My estimate of this build in the previous message to the researcher (≈ 4.5 h) was about three times too high, extrapolated from one contended measurement.

**The PF pass** (`--add-pf`, review 2 S4): PF targets for every training instance of the eight training worlds, 2,500 → 10,000 per world, 2–152 s per world — not the 30–40 min I had estimated. `write_jsonl` was made atomic first (tmp + `os.replace`, 61c3a34): the pass rewrites 10k-record files in place and a kill mid-write would have corrupted one silently. The training files with PF targets are 1.4 MB ({C}) to 87 MB ({I, D}); a process holding all of them is ≈ 4.4 GB (measured), which the trainer avoids for every arm but PF (B2) and which Kaggle's 29 GB holds for the grid's three PF seeds together. The dataset is 254 MB in 46 files.

**The derive step** (`experiments/rewrite/derive/derived.md`, `derived.json`; 48 min on two cores). The first run hung after 20 minutes: the parent held the PF targets, the forked workers duplicated them under copy-on-write, one worker was OOM-killed at the 8 GB cgroup limit and `Pool.map` waited forever at load average 0 — found by looking at the load, fixed by cutting each PF target to its `p_success` at load (the only field the step reads; 0711677). The readings (DESIGN §16): cue-blind ceiling 0.39–0.42 in the pairs, 0.41–0.66 in the triples, 0.69–0.89 in the single-axiom worlds, 1.00 in the full world; recombination 0.000–0.003 of held-out binding instances use a decision never taken in a training world — every held-out detour is recombination; Q2's unaddressable share at sizes 6–7 0.05–0.30 per world, first decisions covered by a training (pattern, rule) key 0.87–1.00 and by a (pattern, position, rule) key 0.47–0.71; RT-post target entropy 0.06–0.55 nats with 1–35 trace classes per instance; RT-examples' ceiling 0.86–0.99 in the pairs, 0.77–0.98 on the multi-axiom binding stratum; the uniform policy's success (PF's prior) 0.18 at k = 1 falling to 0.0007 at k = 6; oracle 150 / 150. The random tie-break moves the greedy rung by up to ±0.3 on individual cells (mean +0.03) and the d = 3 rung hardly at all.

**P4's cells were unreadable as preregistered, and were amended before any run.** On the built eval sets none of the eleven preregistered cells (GP1 instances of the training worlds at k ≥ 4, by (world, k)) separates: on GP1 instances the greedy rung fails by construction and the lookahead-2 rung sits within 0.08 of it (pooled 0.03 / 0.07 / 0.30) — review 2 S7's worry, realized. §6 would have reported P4 "not readable", before a single run, on the rung's Q3 core — the situation L27 exists to prevent and the derive step exists to catch. The same separation rule applied to every (world, binding, GP1, k) cell of the training worlds at k ≥ 4 with n ≥ 10 selects 6 of 41 cells, 170 instances per seed, pooled rungs 0.29 / 0.45 / 0.72; P4 is read on them with its statement, strictness, count and falsifiers unchanged, the GP1-only reading printed beside it, and Q3's per-cell reach on every (k, binding, GP1) cell. What changes: the measured lookahead is no longer pinned to one identified first decision nor to a strict garden path — on the 170 instances greedy fails on 121, 43 at the first decision through lexicographic σ-ties and 78 later, and a strict garden path exists somewhere on the canonical proof in 71 (my first draft said "a later decision in 0.9 of the instances", which was 1 − GP1 share, not a measurement; the reviewer ran the policy and corrected it). The selection reads only the symbolic ladder stored with the instances. `analyze.py` computes both cell sets from the run records themselves, tested on records cut from the fixture with planted ladder outcomes (a router at 0.7 between rungs 0.5 and 0.9 passes 1/1; one at the d = 3 rung does not). **The amendment went to an independent review before being adopted** ([review3_p4_amendment.md](../experiments/rewrite/review/review3_p4_amendment.md): "sound with changes", none blocking) — it recomputed the cells and the ladder from scratch (0 mismatches), found no leak, and attached the caveats now in DESIGN §16: the lower separation is tie-break-dependent (under the random tie-break five of the six cells lose it; the d = 2 / d = 3 edges the pass rule uses are robust under both tie-breaks and under coarser partitions), the cells' gaps carry a winner's curse, the power is adequate for a true rate in [0.48, 0.70] and the near-edge case is named "undecided", the fixed instances leave instance-sampling error to a printed bootstrap, and two unscored alternative sets are printed (upper-separation-only, n 1,245, 0.22 / 0.27 / 0.56; garden-path-anywhere, n 138, 0.06 / 0.17 / 0.49). All of its changes were taken the same day; `analyze.py` now also checks its recomputed cell set against the one `derived.json` fixed and prints a capitalized mismatch.

**Bundle check (L35).** The Kaggle bundle was reconstructed from `launch.py`'s include lists after the review-2 fixes; inside it, run as the kernel runs it (cwd = the bundle's code directory, `PYTHONPATH=code/src`), `pytest -q -x tests/test_rewrite.py` passed (30 tests, 149 s), and a four-step CPU run of `run_rewrite.py` on the real data loaded all 31 eval files, trained RT-canon (612,359 parameters, stream 8 × 10,000) and wrote a complete result: the curve (step, loss, time, per-cell eval5 / eval7 / evalk78), the final per-set `cells` (239 on eval5) and 9,000 per-instance records on eval5 with the ladder outcomes beside the router's. The launcher's 15-training-file check covers RT-all's stream.

**Process.** `PILOT.md` had gone stale against `grids/pilot.json` (16 runs, 4,000 steps, curves every 250 against the grid's 24 / 3,000 / 500) — corrected (6bdb089) before anyone launched from it. §14's review table had been split by the insertion of §15 — repaired. The ledger's Direction section had not absorbed Entries 74–77 — a dated update added (0711677). The dataset (254 MB tarball with `SHA256SUMS`) was sent to the researcher as a file; the desktop link to the Mac dropped every time the 254 MB transfer was attempted over it. *8 Nov:* the researcher's first launch attempt found no `data/rewrite/` (the tarball had not been downloaded) and the launcher's message blamed the flag rather than the missing directory (fixed). With the link up, the tarball went over in seven 40 MB pieces, was reassembled (hash equal to the original), extracted and verified on the Mac — 48 / 48 checksums, 15 training and 31 eval files — and the pilot was launched from [PILOT.md](../experiments/rewrite/PILOT.md). **The pilot's test gate failed on Kaggle** (`1 failed, 27 passed`, grid not run): `test_analysis_on_real_fixtures_is_consistent` found no `*.json.gz` under `tests/fixtures/rewrite/` — the fixtures are tracked and were in the tar, so the cause is Kaggle's dataset pipeline auto-extracting `code.tar.gz` and gunzipping its nested `.gz` members (this was the first bundle to carry any: the eight fixtures and the 46 data files; the data would have failed next). The local bundle check (L35) had passed because it unpacked the tar itself. `run_on_kaggle.py` now reports how the bundle was unpacked, re-gzips any fixture, data or prior member that arrived gunzipped, and logs the counts; checked end to end against a simulated auto-extracted input (49 members restored, 31 tests passed, the smoke grid ran). L37. *Relaunch, 22:36:* the gate failed again and the new log line showed the real layout — Kaggle turns a nested `x.jsonl.gz` into a **directory** `x.jsonl/` holding the decompressed file, so the first fix re-gzipped the files inside those directories (`data/rewrite/eval7_D.jsonl/eval7_D.jsonl.gz`) and found only 38 of 54 members. I had simulated the layout I assumed rather than the one Kaggle produces; the one line of diagnostics I added is what made the second failure informative. Three changes, each checked: the launcher now ships the archive as `code.bundle`, a name Kaggle's pipeline does not treat as an archive, so the kernel unpacks it itself and the members are never touched; it writes the list of `.gz` members into `job.json`, and the kernel restores both the directory-wrapped and the bare layouts and then checks every listed member, printing the missing ones with a listing of the roots and stopping before the gate if any is absent; four simulated layouts (our own archive, directory-wrapped, bare, `.gz`-named directory) each pass the gate and run the smoke grid, and a bundle with one member removed stops with `BUNDLE INCOMPLETE`. The launcher's dry run builds a 255 MB `code.bundle` with 54 members and a matching manifest.

**Decisions.** D123 stands. **D124 (proposed, 6 Nov):** P4's cells are the §16 set (every (world, binding, GP1, k) cell of the training worlds at k ≥ 4 with n ≥ 10 whose rungs separate), fixed from the dataset before the pilot; the §6 GP1-only reading is reported, not scored. **D125 (proposed):** the full world carries no binding stratum; P2's second row and P5 are read on the pairs and the triples; {A, C, I}'s binding row is reported as its own small cell.

## Entry 78 (9 Nov 2026): the rung-4 pilot ran and was read — lr per arm, 12,000 steps, the main grid needs four shards and a session budget

**Provenance.** Kaggle kernel `topology-fleet-rewrite-pilot`, bundle 0a99f17 (third launch; Entry 77a has the two failed gates), 2 × T4, `archive extracted by this script; .gz members expected 54, present after restore 54`, test gate 31 passed, 24 runs (every arm × lr ∈ {3e-4, 1e-3, 3e-3}, seed 100, 3,000 steps, curves every 500), both shards rc 0, session ≈ 2.3 h. Raw records `results/kaggle_rewrite-pilot/` on the researcher's machine (26 MB, outside git); the records without their per-instance lists are in [results_data/pilot/](../experiments/rewrite/results_data/pilot/) and the reading is [pilot_report.md](../experiments/rewrite/results_data/pilot_report.md), produced by `experiments/rewrite/pilot_read.py` (the lr rule tested on fixture records with planted outcomes).

**What the pilot reads, and only that** (PILOT.md; L30): each arm's own training-world rows of eval5.

1. **lr per arm** — highest training-world masked success at 3,000 steps on the full 600 per world: RT-canon 1e-3 (0.938 against 0.902 / 0.888), RT-post **3e-4** (0.919 against 0.916 at 1e-3 — a margin inside the noise, s.e. ≈ 0.006; the rule is applied as written), RT-all 1e-3 (0.905 on its fifteen worlds; 1e-3 also on the eight), RT-absolute 1e-3 (0.756), EI16 1e-3 (0.899 against 0.888), PF 1e-3 (0.921), RT-examples 1e-3 (0.871 against 0.867 — also inside the noise), RT-shuffled 1e-3 (0.934). 3e-3 is the worst lr for every arm. `make_grids.py` LR set; `grids/main.json` regenerated.
2. **Steps** — every arm's training-world curve is still rising at 3,000 at its chosen lr (gain over the last 1,000 steps above two binomial standard errors, or the loss still falling by more than 5 % per 500 steps: 8 of 8 arms; the last curve point recomputed on the same first-100-per-world subset as the others, since the stored one is on the full 600) → **12,000 steps** (PILOT.md item 2).
3. **Throughput** — per training step 53–87 ms for the teacher-forced arms, 158 ms for RT-examples (its context is three times longer), **678 ms for EI16** (its beam search); a curve evaluation 8–13 s (23 s with examples), the full one about six times that. Projected to 12,000 steps with curves every 250: 20–28 min per teacher-forced run, 53 min RT-examples, **148 min EI16**; the 41 runs are **27 process-hours, 12.3 of them EI16's five seeds**. Two shards would need 13.6 h — over Kaggle's 12 h session, which saves nothing when it is hit. So: four shards (`--per-gpu 2`; the work is CPU-bound Python, and the four vCPUs are two hyperthreaded cores, so the speed-up is somewhere between 1.3 and 2 and the session between 7 and 10.5 h), and **a session budget** in the runner — a shard does not start a run it cannot finish inside 10.5 h at the slowest pace it has seen, the kernel then ends normally with what it has, its summary says `INCOMPLETE: relaunch … to resume`, and the launcher's existing resume uploads the finished results. EI16's beam finds a route for 0.75–0.80 of instances in the first 500 steps and 0.99 by step 2,000 (it imitates ≈ 130 rows per step against the teacher-forced arms' 256; steps, not rows, are the matched unit, §12a.7).
4. **Harness sanity** — no cell with unmasked above masked, no NaN loss, every failure category populated in the unmasked reading, 9,000 / 9,000 / 600 per-instance records and 7 curve points in all 24 runs, ladder outcomes stored. The analysis code runs end to end on the real records, and its §16 cell set recomputed from them equals the derive step's (6 cells) — the drift check added by review 3, exercised for the first time.

**One within-arm observation in the sanity table that bears on the design, not on a prediction.** Masked minus unmasked success on an arm's own training worlds is 0.00–0.04 for seven arms and **+0.18 for RT-shuffled** (0.93 masked, 0.75 unmasked, at every lr): a router trained without a usable cue takes out-of-world actions, and the masked reading removes them for it — under the masked reading the action set carries much of what the cue would. P8's reference (the cue-blind ceiling) and its scored reading have to be checked against that before the main grid; that check went to the pass-rule audit (Entry 79), which rests on the dataset alone and was told not to open the pilot.

**Not read:** held-out worlds, the binding stratum, sizes 6–7, k = 7–8, any arm against another. The pilot's records are not part of the reported seeds.

**Changes before the main grid** (tested; 33 tests): `run_rewrite.py` — per-arm costs measured in the pilot (EI16 6.9, RT-examples 2.5, PF 1.3, RT-post 1.1 against RT-canon's 1; the placeholders had EI16 and RT-examples both at 4), the session budget, the three PF runs kept in one shard (a shard process holds ≈ 3 GB, a PF one ≈ 5.5 GB; three side by side would crowd a 29 GB session), one torch thread per shard at four shards; the four shards come out at 6.6–6.9 h each with an EI16 run first in each. `run_on_kaggle.py` — the shard pipelines run under `pipefail` and a non-zero status is printed: a simulation in this container (8 GB) had a shard killed for memory and the kernel reported `rcs=[0, 0]`, because the status it read was `sed`'s (L38); the summary now also prints results against the grid's size. The launcher-to-kernel path was run locally on the real bundle (254.7 MB, 54 members): gate 32 passed and 1 skipped (the pilot-reading test, whose script is not in the bundle), a two-run smoke grid complete, and the budget path leaves both runs and says so.

**Decisions.** **D126 (proposed, 9 Nov):** main grid at the lrs above, 12,000 steps, four shards, 10.5 h session budget with resume. The launch waits for the pass-rule audit's handling (Entry 79).

## Entry 79 (9 Nov 2026): the pass-rule audit — six of the seven remaining rules could not carry their claims; amended before the main grid (DESIGN §17)

**Why an audit.** P4's preregistered cells had turned out unreadable at the derive step (Entry 77a) although review 2 had flagged the risk. On 8 Nov the researcher asked whether the design needed another review after that; my answer was a narrow one — every other pass rule walked against the derive step's numbers by someone who had not written them, before the main grid. I then spent the pilot's wait on the Kaggle failures instead of setting it up, so it ran on 9 Nov, after the pilot. It was given the design, the analysis code and the dataset, and told not to open the pilot.

**What it found** ([audit4_pass_rules.md](../experiments/rewrite/review/audit4_pass_rules.md), verbatim; its scripts beside it). Verdict: do not launch on the current preregistration.
- **P2 (the core) could pass without composition.** 0.66 of the held-out pairs' binding stratum (AI 0.85, CI 0.47) and 0.52 of the triples' has a proof within the horizon inside a training world contained in the held-out one. A policy that plans only inside the eight training worlds and lets the mask choose scores 0.839 on the pairs' stratum — both pair conditions met, nothing composed. RT-all ≥ 0.9 sat 0.21 above the lookahead-3 rung, with no statistic over its three seeds; below 0.9 the rule could neither pass nor be falsified.
- **P8's reference is not a ceiling.** A world-blind planner beats the "cue-blind ceiling" by more than 0.05 in 14 of 14 worlds, masked (the action set gives the world away) and unmasked (the fewest-axioms world in which the goal is reachable is usually the right one).
- **P3's "≥ 0.9" was passed by a router that lost 15 %** — sizes 6–7 are easier on matched cells (the rungs' own ratio 1.06–1.13) — and its second clause counted as "must fail" 265 instances that have an addressable route within the horizon (318 truly must fail).
- **P7's "within 0.1 of the ceiling"** is a perfect planner's bar (0.954 where lookahead-3 scores 0.686). **P5 and P6** had no stated statistic or tolerance. **No verdict was computed in code** for any rule but P4, and only P4 had a planted pass and a planted fail.

**Checked before acting on it.** The P2 recount with the library's own search (608 / 209, 1,065 / 509, per world as reported); the training-world reference re-implemented in the library and equal to the auditor's to the instance (0.839 / 0.531, 0.602 / 0.212); the ladder's matched ratios and the must-fail set recomputed in a new derive section (§9: 1.126 / 1.106 / 1.057; 583 → 318); P7's weighted ceiling. The cue-blind policies' rates are the auditor's and are quoted as such.

**The amendment** (DESIGN §17; §6 stays as the record of what was preregistered). No run changes: the records carry both readings per instance.
- **P2** is scored on the binding instances that need the unseen combination (pairs n = 209: AI 47, CI 162; triples n = 509), with the pass bar unchanged (≥ greedy + 0.2 and ≥ 0.7 in ≥ 6/10), the triples' clause and the falsifier measured against a non-composing floor (the larger of the greedy rung and the training-world reference: 0.531 / 0.212), RT-all's seed-median gating only the falsifier, and the unmasked reading naming what a masked pass means (composes under the cue, or composes decisions only when the executor supplies the action set).
- **P3** clause 1 is read against 0.9 × the nearest rung's own ratio; clause 2 on the 318 must-fail instances and the rest, unmasked, with counts over the five seeds.
- **P5, P6** get statistics and counts; **P7** is read against ceiling × RT-canon's own rate, unmasked.
- **P8:** the ceiling clause is withdrawn (X40); the second clause and the falsifier are kept, with the 0.05 the rule already used, on each reading separately.
- Each change is marked tighter or looser in §17's table. One is looser in a way that favours a pass: P2 no longer requires RT-all ≥ 0.9.

**What I had seen of the pilot when writing §17, and the one place it matters.** Stated in §17: the training-world readings, the kernel log's all-world averages, the sanity table's masked-minus-unmasked gaps, and one header line read by accident. The sanity table (RT-shuffled 0.93 masked against 0.75 unmasked on its training worlds; RT-canon 0.94 / 0.93) makes it likely that P8's falsifier fires on the preregistered masked reading and that the new unmasked clause holds. Moving P8 to the unmasked reading would therefore have turned a probable failure into a probable pass. It was not moved: the masked falsifier stays the scored verdict of the preregistered reading and will be reported as P8 falsified under the mask if it fires; the unmasked clause — which the auditor, blind to the pilot, asked for on the grounds that the reference assumed no mask — is an addition with its interpretation fixed in advance.

**Implemented and tested.** `proxy.trainworld_policy` / `trainworld_rollout` / `goal_distances`; `derive.py` §9 (flags per instance in `derived.json`: `p2_comb`, `eval7_mustfail`; `--only-9` merges the section into the existing output in 22 s); `analyze.verdicts()` computes every rule's verdict and `render_verdicts()` prints them with their numbers, the §6 tables still printed unscored and P1 with its unmasked column; planted pass-and-fail tests for P2 (seven outcomes, including the non-combination instances not counting and the two pair conditions being joint within a seed), P3 (both clauses, and a router the bare 0.9 passes and the normalised rule does not), P5, P6, P7, P8 (every cell of the reading matrix). **Every run now saves its router** (`train.save_checkpoint`; the file alone reproduces the run's final evaluation instance for instance — tested), so a later stratum or an enriched eval set is scored without retraining (L40). 39 tests; the launcher-to-kernel path re-run on the real bundle (gate 38 passed, 1 skipped; two smoke runs with their checkpoints, 2.5 MB each); the amended analysis runs end to end on full-size records with every verdict computed (its output on the pilot was not read).

**Review 5** (independent, of §17 and its implementation, pilot-blind): running at this commit; its outcome and handling are appended to this entry before the launch command is given.

**Review 5's outcome and handling (same day, before the launch).** [review5_amended_rules.md](../experiments/rewrite/review/review5_amended_rules.md): "sound with changes", two of them blocking; `verdicts()` otherwise implemented §17 as written, and its own engine reproduced every count I had cited (0 mismatches on all 1,673 held-out binding instances; the reference deterministic under reversed orders).
- **Blocking 1 — an error of mine since the implementation choices (§12a.5), carried through the derive step and into §17:** "the categorical head's rows 9–12 are never labelled at size ≤ 5". No action is valid at prefix positions 7–8 of a term of ≤ 5 leaves, nor an A rule at 5–6. Counted on the eight training files (17 s): 52 labelled slots, none at positions 7–12. By them, 986 of the training worlds' 4,800 eval7 instances have no route within T (not 318), and 2,007 no shortest proof the head can address (not 583; §16's "0.05–0.30 per world, 0.00 in {A}" should have read 0.15–0.63, 0.39 in {A}). The reviewer planted an RT-absolute failing exactly its artefact: the rule as I had written it called that "a table beyond the artefact". `derive.py` §9c now classifies every instance by the labelled slots (2,793 a shortest proof / 1,021 only a detour / 986 none — the reviewer's counts to the instance) and P3's second clause is read on those groups. L41.
- **Blocking 2 — what a masked P2 pass can claim.** The falsifier's floor (0.531 on the pairs' stratum) is one non-composing policy among several, 0.35–0.75 by variant; the same policy with the greedy move as its fallback — single-world plans glued by greedy moves — scores **0.751 / 0.497** and is itself above P2's pass bar. So a masked pass shows that decisions recombine under the executor's mask, not that the router plans over the unseen combination. The glued reference is now in the library, stored per instance and printed; §17 and the report say what a masked pass does and does not claim; the unmasked reading and the 52 / 256 instances the glued reference does not solve are printed as the rows that bear on the difference (unscored: 52 is too few to carry a rule).
- **Thresholds:** P3's normaliser is interpolated between the rungs and capped at 1 / success (it had jumped between rungs and asked an oracle-strength router for more than §6 did); P7's first clause is called a band that can hardly fail, with RT-shuffled printed beside it — I did not add a scored comparison there, having seen both arms' unmasked rates in the pilot.
- **Accounting:** the reviewer found §17's tighter / looser column incomplete in my favour in three places — withdrawing P8's ceiling clause is a loosening; P8's unmasked clause and its 0.05 are not blind (the sanity table puts the unmasked gap ≥ 0.14 above the masked one); and I had seen RT-all's pilot average (0.905 over its fifteen worlds) when I dropped "RT-all ≥ 0.9" from P2's pass, the only change that can turn a §6 non-pass into a pass. All three are now stated in §17, and a pass that §6 would not have counted is labelled so in the verdict line.
- **Tests:** 42 of 60 mutants of my first planted tests survived (every count boundary, most thresholds, median → max, three swapped readings). The tests now pin each at its boundary; the mutation check brought up to the amended code (`review/review5_scripts/mutate.py`, 76 mutants: every count, threshold, statistic and reading of `verdicts()`) kills 76 of 76. 39 tests in the suite.
- Recorded in DESIGN §17 (revised in place) and §17a (what changed from the first version, commit 16a4f7e). The derive step was then re-run whole from the amended code (58 min on two contended cores): every key of `derived.json` equal to the committed one, so sections 1–8 are unchanged and §9 is the same whether merged or regenerated; the committed files are that run's. The launcher-to-kernel path was run once more on the final bundle (gate 38 passed, 1 skipped; two smoke runs with checkpoints; the analysis reads the kernel's records with the corrected groups: 986 / 1,021 / 2,793).

**What the audit and the review cost and bought.** Two pilot-blind agents, about 600k tokens and three quarters of an hour of wall time each, against a 27-GPU-hour grid whose core prediction would have been scored on a stratum two-thirds of which needs no composition, with a null read against a ceiling that is not one, an absolute control whose artefact was mis-counted, and no computed verdict. The order was wrong: this belonged at the derive step, before the pilot (L39), and the amendments now carry a statement of what the pilot had shown me.

**Decisions.** **D127 (proposed, 9 Nov):** the main grid is scored on §17; §6's tables are printed unscored. **D128 (proposed):** P2's stratum is the held-out binding instances that need the unseen combination; an enriched set scored from the saved routers may be reported beside it, not in its place. D126 (lrs, steps, shards, budget) stands.

## Entry 80 (10 Nov 2026): the rung-4 main grid ran and was scored — recombination not shown, size extrapolation failed, the prior's reach exceeded the ladder

**Premise.** The main grid of Entry 79 (41 runs, 12,000 steps, [grids/main.json](../experiments/rewrite/grids/main.json)), scored on DESIGN §16 / §17 / §17a with the verdict code and tests fixed before the launch (D127).

**How it ran.** Two Kaggle sessions on two T4s, four shards. Session 1 (bundle `20ebab0`, launched 9 Nov 15:42 PST) produced 29 runs and stopped at the 10.5 h session budget, as the guard is meant to: EI16 took 272–281 min a run against the 148 projected from the pilot, and the grid 44.8 process-hours against 27 (the cause of the factor was not established). The fetch of session 1 broke partway (three checkpoints and a zero-byte file); `launch.py` got a retrying fetch that clears empty files and says when a grid is incomplete (`32c9f53`), the researcher re-fetched and relaunched. Session 2 (bundle `32c9f53`, 10 Nov 08:56; the bundles differ only in `launch.py`): "resumed with 29 prior results", gate 38 passed / 1 skipped, 83 of 83 archive members present, the other 12 runs (RT-all s0, RT-canon s1–s9, RT-shuffled s0–s1), `results=41 of 41`, `rcs=[0, 0, 0, 0]`. Session 2's fetch overwrote session 1's kernel log and shard logs on the researcher's machine; session 1's result files and checkpoints are intact. 41 records and 41 saved routers; the records were copied into the session container and analysed there (`analyze.py`, exit 0; "the §16 cells recomputed from the run records equal the derive step's").

**Verdicts under §17** ([results_data/main_report.md](../experiments/rewrite/results_data/main_report.md), last section; write-up [RESULTS.md](../experiments/rewrite/RESULTS.md)).
- **P2 (core, recombination): not met, not falsified.** On the held-out binding instances that need the unseen combination — pairs n = 209, floor 0.53, lookahead-3 0.73, glued reference 0.75 — RT-canon's seed-median is 0.47 [0.28, 0.67] masked; both pair conditions in 0/10 seeds; above the floor in 4/10, one seed more than the falsifier's ≤ 3/10. Triples n = 509: 0.38 [0.27, 0.56], above the floor (0.21) in 10/10, below the glued reference (0.50). Unmasked: 0.37 and 0.31; above the pairs' floor in 1/10. RT-all on the same instances 0.94 [0.93, 0.95] and 0.85.
- **P3 (core, size extrapolation): falsified on both clauses.** Structural: matched ratio 0.33–0.58 against 0.9 × ρ (ρ 1.01–1.04), 0/10. Absolute: 0.00 on the must-fail group in 5/5 as derived, and 0.00–0.01 on the group with a shortest proof on labelled slots, 0/5.
- **P4 (core, lookahead absorbed): falsified toward "three steps".** On the six §16 cells (n = 170; rungs 0.29 / 0.45 / 0.72) RT-canon 0.94 [0.88, 0.96]; ≥ the d = 3 rung in 10/10, strictly between d = 2 and d = 3 in 0/10.
- **P5 holds** (EI16 0.68 [0.42, 0.77] against RT-canon 0.63 on the held-out binding stratum). **P6 does not hold as written** (RT-post's distance to its target 0.12–0.14 against the uniform reference's 0.10, 0/10; RT-canon canonical-first ≥ 0.9 in 9/10). **P7 holds** (0.89 / 0.90 / 0.92 against 0.89 ± 0.1; held-out 0.44 / 0.38 / 0.45 against 0.55). **P8: masked falsified** (0.94 against 0.91 / 0.90), **unmasked holds** (0.93 against 0.58 / 0.58). P1 (reported): {A, C} at k = 7–8, RT-canon 0.53 [0.32, 0.62] where the lookahead-3 rung scores 0.02.

**What I computed after reading the verdicts (post hoc, unscored; scripts and outputs in `results_data/`).**
- *Every arm on P2's strata and their complements* (`comb_by_arm.py`): on the held-out binding instances provable inside a contained training world RT-canon scores 0.93 (pairs, n = 399) and 0.75 (triples, n = 556) against 0.47 and 0.38 on those that need the combination. §6's stratum as first preregistered reads 0.77, pair conditions in 7/10 seeds, triples' clause 10/10, RT-all 0.97 and 0.90: the original rule would have passed or come within a hair of it. RT-shuffled on the needs-the-combination pairs 0.60 [0.59, 0.61] masked (two seeds) and 0.15 unmasked; RT-post 0.64 / 0.59 (at lr 3e-4, so not matched to RT-canon).
- *Sizes 6–7 broken down* (`main_extra.py`): RT-canon is below the greedy rung at every k (k = 1: 0.65 against 1.00), 0.24–0.29 of its episodes end in a wrong halt, and success falls with nesting depth past the feature clamp (k = 1: 0.85 / 0.81 at depth 3 / 4, 0.59 at 5, 0.43 at 6). RT-all the same. The structural features were clamped to the size-≤ 5 range after review 2 (L34) so that no untrained row is read; the clamp makes the nodes that larger terms add share features. P3's falsifier reading ("the structural router also learned a table") is therefore not separated from an encoding that cannot tell the new nodes apart, and the absolute control's collapse (0.03 at k = 1) is confounded with its clamped positions in the same way. The verdict stands; its preregistered interpretation does not follow. This is a design failure of mine on the input side of the same prediction a second time.
- *P6's rule* (`p6_check.py`): the uniform-over-shortest-actions reference equals the target on 1,003 of the 2,213 instances, so it scores zero there and cannot be beaten. On the other 1,210 RT-post is closer to its target than uniform in 10/10 seeds (0.150–0.170 against 0.186; deviations from uniform correlated 0.51–0.59 with the target's). I wrote that rule in §17, planted-tested it with exact outcomes, and two blind reviews passed it; the defect showed only when a real, noisy policy was scored. The verdict stays "does not hold"; the split is a post hoc reading.
- *Curves:* RT-canon's held-out curve peaks before the end in 10/10 seeds and ends 0.01–0.10 below its maximum; training-world curves are flat or rising at the end.

**Argument.** (i) Q1, the rung's core: zero-shot recombination of stated axioms across worlds was not shown at the preregistered level in any seed, on instances a fifteen-world router solves at 0.94, and the median seed is not distinguishable from a policy that composes nothing. It was also not cleanly refuted: 4/10 seeds are above the floor, the stratum is 209 instances in two worlds, and in {A, C, D} the router sits at the lookahead-3 rung with greedy at 0.03. (ii) Under the mask the stated axioms do nothing measurable — on training worlds (P8) or held-out ones (RT-shuffled no lower than RT-canon, two seeds). The cue matters only when the router must name valid actions itself. For the thesis's Lean picture, where the kernel supplies the valid tactics' effects but not which are worth trying, which reading is the relevant one is the open rollout question (I36), now with numbers on both sides. (iii) Q3: inside the trained term sizes the no-search prior is far past three steps of lookahead and still at 0.53 at proof length 7–8; the ladder was too short to measure where it ends. (iv) Q2 was not answered: the instrument conflated rule-against-table with the encoding's aliasing.

**Not yet done.** Independent results review (review 6) is running on the records, the report and RESULTS.md; its outcome and the ledger's findings go in the next entry. The two questions left with the researcher on 9 Nov: whether P2's pass needs RT-all ≥ 0.9 (moot for this grid: nothing passed; RT-all's triples median is 0.85) and whether to score an enriched needs-the-combination set from the saved routers (D128 allows it beside the verdict; it could separate "at the floor" from "a little above", not produce a pass).

**Lessons recorded:** L42 (a reference that equals the target on part of the stratum), L43 (clamping an extrapolated feature aliases it; check one-step instances first), L44 (cost projection and per-session fetch directories).

## Entry 81 (10 Nov 2026): independent review 6 of the rung-4 results — numbers and verdicts stand; one false statement of mine and eight readings corrected; the eval sets overlap the training stream

**Premise.** Entry 80's readings were the author's. A reviewer who had not designed, run or analysed the grid was given the 41 raw records, the report, the companions and RESULTS.md, and asked to recompute with its own code and attack the readings ([review6_results.md](../experiments/rewrite/review/review6_results.md); nine scripts beside it, none importing `analyze.py`; about 430k tokens and 23 minutes).

**Its verdict: sound with changes.** Every number behind the verdicts and the three companions reproduced (differences: rounding, one bootstrap bound); every scored verdict is what DESIGN §17 / §17a / §16 yields; the "needs the combination" flag recomputed from scratch with 0 mismatches on 1,673 instances; eval7 checked against the executor and eleven record invariants with 0 violations. No scored verdict changes.

**Blocking — a false statement, and behind it a defect of the dataset.** RESULTS §4 said "the instances are new pairs over terms of the trained sizes". The build never excluded training instances from the eval sets: 1,716 of the 4,800 training-world eval5 instances (0.357) are verbatim training pairs (0.28 / 0.22 / 0.32 at k = 4 / 5 / 6; 20 of the 170 in P4's cells), and about 30 % of held-out eval instances are rows RT-canon was trained on inside a contained training world. I had not checked it and neither had any of five earlier reviews. P4 survives on the 150 never-trained pairs (0.93 [0.87, 0.95] against 0.73, 10/10), P8 survives (masked gaps 0.040 / 0.041), P1 has no overlap; the pooled k = 5–6 rates are raised by recall (k = 6: 0.73 on new pairs, 0.99 on repeated ones); and RT-canon scores 0.89–0.97 on the held-out rows it had trained on in a contained world against 0.45–0.71 on the rest, so Entry 80's point that held-out generalization is mostly instances needing no unseen combination is stronger than I wrote it.

**Should-fix, each accepted.**
- *RT-shuffled under-reported.* I had written "not below RT-canon … inside its range" from the one stratum where the difference is smallest. Masked, the token-blind router is ahead over all held-out instances (0.78 / 0.80 against 0.68) and above every RT-canon seed on three strata; unmasked the two are level on held-out worlds and the token-blind one leads in the full world in 10/10 seeds. Reading the true cue at an unseen pattern costs success relative to ignoring it; part of P2's shortfall is the cue as an unseen input. Entry 80's argument (ii) — "the cue matters only when the router must name valid actions itself" — holds on training worlds and on the needs-the-combination strata, not on held-out worlds as a whole.
- *The late held-out decline is not in the records.* Entry 80's "peaks before the end in 10/10 seeds and ends 0.01–0.10 below its maximum" used a statistic that is negative for any flat noisy curve, across points on different instance sets. In window means the held-out curves are still rising (0.683, 0.692, 0.696). Withdrawn.
- *P2's 4/10 is a coin flip.* The fourth seed above the floor is above by 2 instances of 209; instance resampling returns the falsifier in 50 % of draws. Entry 80's "it was also not cleanly refuted: 4/10 seeds are above the floor" stands as a description of the verdict and is not evidence of recombination.
- *The {A, C, D} argument* (router at the lookahead-3 rung with greedy at 0.03) shows nothing about composing: the glued reference scores 0.43 there against the router's 0.37. In {A, I} the router is below its own floor in 10/10 seeds. Withdrawn; per-world references printed.
- *"No arm reaches 0.7 on the pairs"* is true of medians; four of ten RT-post seeds and one of five EI16 seeds do, and RT-post is above the floor in 8/10. The RT-post comparison is confounded by its data as well as its lr (1.1–2.5 times as many distinct rows; the pilot chose its lr on 0.919 against 0.916 at one seed).
- *Sizes 6–7.* Entry 80's "the absolute control's collapse is confounded with its clamped positions in the same way" is wrong about the code: that router has fixed sinusoidal positions and no clamp. Its confound is total — every eval7 input is longer than any it trained on — so clause 2 could not have held and the rule, which I wrote and review 5 passed, was mis-specified. A third confound was unnamed: training at size cap 5 against eval at cap 7, so growth at the largest size was never a label and 52 % of one-step eval7 instances need it. And "the measurement cannot say why" under-reported the records: with no node past depth 4 and no growth, one-step instances are solved at 0.95 [0.92, 0.97] in every seed; with the rewritten node at depth ≥ 4, 0.25 with a wrong halt in 0.72; growth or a deep rewrite accounts for 89 % of one-step failures. Post hoc, not a clean pass (the multi-step clean subset is still below greedy), and no episode was replayed.
- *P6.* The defect I found in the rule is real, and RT-post also misses the auditor's 0.05 bar, and PF — never trained on the target — is closer to it than RT-post on both halves. My own check script had printed PF's rows and I did not report them.
- *P4's wording.* "Does more than three steps of lookahead's work" → "beats the proxy-guided lookahead-3 policy": the router's success is unrelated to the ladder's instance by instance.

**Checked before acting on it** (`results_data/review6_checks.py`, my own code): the overlap (1,716 of 4,800; per k as reported), P4 on the 150 never-trained pairs (0.93 [0.87, 0.95] against 0.73, 10/10), RT-shuffled against RT-canon on the held-out strata (as reported to ± 0.01), the one-step split at sizes 6–7 (0.95 / 0.79 / 0.77 / 0.35), the absolute router's encoding and the caps in the data files (read). Taken as the reviewer's without recount: the instance resampling of P2, the per-world references, the window means, the held-out rows trained in contained worlds, RT-post's counts under P2's rule.

**Handled.** RESULTS.md revised (its §8 lists the nine changes; the earlier version is commit `b91c12b`); the curve table in `main_extra.md` carries the caveat; the ledger has the findings (F51–F55), the failed predictions (X42–X45, and P2 as not met), the overlap as a derived constraint, and L45–L47. Not done: P4's verdict in `analyze.verdicts` (it is read from the P4 table's counts; stated in RESULTS); a replay of failing size-6–7 episodes through a saved router; deduplicated eval sets.

**Argument, revised from Entry 80.** (i) Q1: not shown, and "not falsified" only by a margin the instances cannot carry; the instrument could not separate failing to compose from the cue being an unseen input; most apparent held-out generalization needs no composition and a third of it is recall. This is the rung's core and the thesis's step (1) leans on it; the honest state is "no support, weak refutation, low power". (ii) The cue: under the mask it does nothing on training worlds and costs on held-out ones (two seeds). (iii) Q3: the labelled prior's reach inside the trained sizes is past the ladder, on never-trained pairs too. (iv) Q2: not asked cleanly; the records point at the clamp boundary and the cap, and at one-step rules transferring where neither applies.

**What this review cost and bought.** 23 minutes against a write-up that carried a false statement about its own eval sets, a manufactured trend, an argument from the one world and the one stratum that suited it, and a wrong sentence about the code. Six of the nine corrections were to things I had inferred instead of counted (L45–L47, and L41 again).

**Decisions.** **D129 (proposed, 10 Nov):** rung-4 results are quoted from the revised RESULTS.md; training-world rates on this dataset are quoted on never-trained pairs beside the pooled ones. **D130 (open, the researcher's):** whether to rescore the 41 saved routers on an enriched needs-the-combination set and deduplicated eval sets before rung 5's design — no training; can move P2 off its coin flip, cannot produce a pass — and what the recombination result changes in rung 5, whose outline assumed a labelled prior that composes across worlds.

## Entry 82 (10 Nov 2026): stepping back after rung 4 — the researcher's critique of the rung, the routing-signal and route-labelling discussion, and a pushback literature review

**Premise.** After Entries 80–81 the researcher asked what the program is trying to do, what rung 4 was meant to prove and what its results mean, then criticized the rung and opened the route-labelling question. A conceptual discussion; nothing was run except a literature search.

**The researcher's critique of rung 4, which I accept.** (i) The rung was poorly designed for its core question: a PFN interpolates where it has support, the domain is proofs, and eight training worlds give no support for reading "does not recombine" — insufficient support confounds the reading. (ii) What signal does the router route on, and is it informative? My answers: the state and goal were informative and were used (F53); the axiom tokens were names, not descriptions, with eight patterns (F52); the valid-action set says what is legal, not what is useful; and what a router needs in a new world — the consequence of its actions there — is in none of its inputs and comes only from looking ahead. My one disagreement, kept: full support over combinations is impossible in mathematics, so off-support generalization cannot be designed away; coverage has to be the measured variable.

**A post hoc reading recorded as a hypothesis, not a finding.** In distribution the no-search prior beats proxy-guided lookahead-3 (0.94 against 0.72); on the pairs that need an unseen combination lookahead-3 with a crude heuristic beats the prior (0.73 against 0.47). The prior wins in support and search wins off it; rung 4 asked the prior alone to do what the thesis assigns to search.

**The researcher's ideas on route labelling, and where the discussion left each.**
- *Primitives as shifts in a latent space with one decoding region per token; label routes by composing shifts into the target region.* Well-posed as search with the model's own primitives as executor and the output layer as goal test. Constraints found in discussion: a constant shift commutes and collapses depth, so shifts must depend on the state; attention couples tokens' routes; the labels are relative to primitives still being learned; greedy descent on a distance fails where detours are needed. The researcher's two objections stand: we already have negatives for learning routes and primitives together (F22–F23, F32 — I had cited rungs 2–3, where primitives were handed over), and a token's set of *intermediate* latents is not a contiguous volume (only the final region under a linear readout is convex, and it lumps unrelated reasons together). Geometry gives a test for arrival, not a compass.
- *Designed rather than learned primitives, complete under composition.* Universality is cheap (gate sets, compiled transformer programs, proof systems); the cost is route length, and designed primitives need a designed state. For Lean it is the natural reading and sidesteps risk 2, at the price of distinctiveness.
- *A latent of equal hypercubes, one per token.* The researcher's own objection is right: if a cube is a token the state is only the answer and computation collapses to a lookup. The repair is to decode a few digits of the cell index and keep the rest as registers, with primitives acting on coordinates; intermediate labels are then an execution trace, which exists for mathematics and not for language.
- Both labelling routes point at the same untested item: a dense teacher's layer-by-layer path as per-step targets, i.e. risk 1 (a trained deep model as a router over m primitives, as a curve in m).

**The pushback review** ([literature.md](literature.md), new section; abstracts or pages read for the entries marked there). What it does to the thesis:
- *The depth claim is a cost claim.* A transformer with a scratchpad has unbounded serial depth (Merrill & Sabharwal); dense transformers shortcut T sequential steps in O(log T) or O(1) layers in distribution (Liu et al.) and amortize search very far (Ruoss et al.) — F53 is an instance. The motivation survives off-support: compositional depth fails out of distribution (Dziri et al.; Wang et al., who suggest cross-layer knowledge sharing).
- *Reuse has support; the router does not.* A single 2-layer block recursed with deep supervision (TRM, after HRM) beats far larger models on puzzle reasoning with no library and no router. In trained LLMs, replacing middle layers by one repeated layer is the worst intervention (Sun et al.), so layers are not one primitive; whether they are m is unmeasured.
- *Learned routing has a poor record.* The joint-learning pathologies are catalogued (Rosenbaum et al. 2019); learned modular systems fall short of their potential on known-modular data (Mittal et al.); a fixed hash matches learned routing in MoE (Roller et al.).
- *Rung 4's P2 was predicted.* Below a task-diversity threshold a transformer behaves as the Bayesian estimator over the finite training tasks and cannot solve unseen ones (Raventós et al.); eight worlds is far below. The researcher's critique is that paper's finding.
- *Not new:* amortized posteriors over programs with a growing library (DreamCoder), designed primitives with trace supervision (NPI), policy plus search over Lean (AlphaProof).

**Argument.** The thesis is three claims, and each has a cheaper standing alternative: depth — a scratchpad; reuse — one looped block; routing — a fixed assignment. What would be ours is the conjunction: a library with a posterior- or search-trained router on an explicit state beating all three, off-support, with coverage measured. Rungs 0–4 studied the router with primitives handed over and never ran any of the three comparisons. That is the honest position of the program after four rungs.

**Decisions.** None taken. **D131 (open, the researcher's):** which comparison leads the next design — where a dense model runs out (depth against data), the single looped block as the opponent at matched parameters, or the m-primitive fidelity of a trained dense model (risk 1). D130 (rescoring rung 4's routers) is, on today's discussion, sharpening a question the researcher considers badly posed; left open, not recommended.

**Rejected alternatives.** Treating rung 4's P2 as evidence against recombination; a standalone rerun of size extrapolation; continuing to rung 5 as outlined on 6 Nov without the looped-block and scratchpad opponents.

**Expected falsifiers of the thesis, as now stated.** A dense model with a scratchpad matching the library at equal compute off-support; a single recursed block matching it at equal parameters; a hashed or gating-trained router matching the search-trained one.

## Entry 83 (11 Nov 2026): the researcher moves the program from discrete primitives to looped encoders that learn soft programs; what the loop literature says; the contribution is not yet identified

**Premise.** A second day of conceptual discussion after Entry 82; nothing was run. The researcher worked from "a layer is a time step of a latent state" to a change of direction. Recorded here because it changes what the program is.

**The line of thought, in order.**
1. *Layers as time steps; many paths from input to output.* What I argued: cross-entropy pins only where a trajectory ends, so whatever chooses the path is not in the loss. A network is a rule for moving (a vector field), dense = a different rule per step, loop = one rule. Two lengths trade off — the path's (compute) and the rule's (parameters) — and generalization follows the second. Gradient descent deforms a path and cannot jump to another; search jumps. Supervising the path removes the problem where a path is known (flow matching; chain of thought); for latent reasoning nobody wrote the path down.
2. *What a looped block does* ([literature.md](literature.md), 11 Nov section): one rule, driven by a re-injected input. A gradient field in the in-context regression case (weights hold the prior, context the problem, state the iterate); an algorithmic state-transition for parity and addition. Loops work because of a symmetry in time, a short rule, compute separated from parameters, closure under self-application, and stable fixed points.
3. *The researcher's pivot.* Away from discrete primitives "to match the field, in that an expert will absorb whatever information and distribute accordingly"; a hierarchy of thinking with lookup cheapest; hierarchy should not be forced (the bitter lesson), and equal expert usage is itself a forced structure. My qualifications: the bitter lesson's two halves are learning and search; abstraction is forced by diversity outrunning parameters, not by parameters; load balancing conflates keeping experts alive, hardware convenience and an unjustified equal-mass belief (my inference, not a finding).
4. *What scale adds to a loop.* Ouro: looping raises manipulation, not storage (≈ 2 bits per parameter either way). Experts in shared layers recover storage at constant compute (MoEUT, ModernALBERT, Tying the Loop at 7B). Not found: whether experts inside a loop take on transformations as well as facts.
5. *Facts against transformations in Lean.* The facts are a library that exists as text; fresh lemma names supplied in context make keyed memorization impossible, so what the weights learn is transformation. Disentangle in the data, not in the weights.
6. *Tokens.* Autoregression conflates memory, commitment and thinking; tokens serve the first two. In Lean the kernel supplies memory (the proof state) and commitment, so a latent need only hold one decision's thinking and long proofs live outside the model.
7. *The researcher's architecture sketch:* a looped encoder writing to register tokens, an autoregressive decoder reading them. Every part exists (Perceiver, fast-slow latent recurrence, AbbIE, TRM, AlphaProof's encoder–decoder); the combination was not found as a named architecture in a brief search.

**Corrections of mine during the discussion.** I told the researcher that none of the experts-in-loop papers tested reasoning or composition; SUT reports compositional-generalization gains (our own literature notes said so). Corrected the same day.

**Where the program stands.** The original motivation was primitives for learning programs, with a router as an amortized posterior over them, trained by posterior or search targets. The researcher has replaced the discrete primitives with a looped encoder that learns soft programs as latent transformations. Consequences I stated to the researcher: (i) the architecture is occupied ground — looped encoders, experts in loops up to 7B, looped LMs at 7.7T tokens; (ii) rungs 0–4 studied routers over handed-over discrete primitives, so their relevance shrinks (what transfers: loops extrapolate in depth and learned halting is fragile, F22–F24, as Ouro also found; amortization inside the trained range, F53; coverage, F51–F52, as INT found in 2021); (iii) no contribution is established. Candidates are measurements, not systems: whether a looped encoder's soft program has program structure (reusable latent moves across problems — the original thesis as a hypothesis about learned dynamics); what experts add when nothing can be memorized; where extra compute should go off-support (loops, search, scratchpad); the latent path against a proof system's intermediate states. The claim "latent transformations are best represented in an encoder rather than at the per-token decoded level" is a design choice we argued toward, not a result; the 2026 decoder-step-recurrence papers go the other way.

**Decisions.** **D132 (the researcher's, 11 Nov, as stated in discussion):** the program's object is latent transformation in looped transformers, in a Lean-like domain; discrete designed primitives are set aside. **D131 is superseded** by this. **Open:** the contribution, and a proper novelty search before any design.

**Expected falsifier of the candidate claim** (a looped encoder trained on proofs learns a small set of reusable latent moves, and that reuse predicts generalization to unseen lemma combinations and longer proofs): the per-iteration displacements do not fall into reusable types, or they do and reuse does not predict generalization.

## Entry 84 (11 Nov 2026): a deeper literature check on probing latent trajectories — the measurements exist, the niche narrows to proofs, reuse and experts

**Premise.** After Entry 83 the researcher asked for a deeper check on latent-trajectory probing and whether "which encoded states decode to the correct solution, and how the encoder gets there" can be defined by measurements. Ten searches, eleven pages read (most truncated by the reader); notes in [literature.md](literature.md), "Probing latent trajectories".

**Result.** Yes, with an existing toolkit: per-iteration decoding and loss; the iteration's residual as convergence and halting signal; path independence under changed starting states (Anil et al. 2022 — correlated with generalization to harder instances and usable per instance as a label-free check); basin mapping and spurious attractors (the HRM and TRM analyses of 2026); probes with causal interventions (the Sokoban RNN work is the standard); per-step data attribution. Entry 83's candidate "the latent path against known structure" is therefore not new as a kind of study: it has been done for Sudoku, mazes, ARC, Dyck, parity, graph reachability and Sokoban.

**What the published analyses found, which we should expect to meet.** In the recursive reasoners trained with a loss at every step, the loop mostly does not reason stepwise: most accuracy arrives at the first recursion (TRM on ARC: 38.25 % → 40.50 % over four steps), loss is flat and then drops (HRM), trajectories fall into the nearest fixed point, failures are spurious attractors, and restarts or noise are the fix. Stepwise or parallel-frontier computation is reported where the task is sequential (Sokoban plans, reachability, parity). So "what path does the loop take" has two known answers, and which one a proof task gives — and how the training signal decides it — is the open part.

**What was not found** (targeted searches, not a review): these measurements on a theorem prover's latent states; reuse of latent moves across problems; what experts do to trajectories. Lean's specific advantage: the kernel checks the decoding of any state and a goal has many valid next steps, so the set of correct states can be mapped exactly as a union of regions.

**Consequence for Entry 83's candidates.** The candidate claim (reusable latent moves that predict generalization) stands as unanswered; "the latent path against a proof system's intermediate states" is narrowed to "which of the two regimes, and what decides it"; the caution that decodable is not used now has a number behind it (latent-step ablations flipping ~10 % of answers in Coconut-style models).

## Entry 85 (12 Nov 2026): exploration in loops, parallel streams, and what is left of the original thesis when the model is given room

**Premise.** Continued discussion (11 Nov afternoon, 12 Nov); nothing run. The researcher's line: looped encoders learn attractors and get caught in them without exploring (the slime-mold contrast); could parallel branches or streams hold alternatives; and, on 12 Nov, "less hand engineering, more giving space to the model — what exists, and what can we contribute along the lines of the original thesis".

**Where the discussion of exploration ended.**
- The failure in the recursive reasoners is early commitment, not attractors: one deterministic trajectory, a contractive map, a time-invariant rule that cannot explore first and commit later unless the state carries a clock or an uncertainty, a loss on the answer at every step, and no internal evaluator. A loop is an amortized solver — rung 4's result once more.
- The slime-mold pattern (explore in parallel, reinforce by two-ended evidence, commit late) is what the Sokoban network and the continuous-thought models were found to do; both were trained with delayed or final-only supervision. Hypothesis, from a handful of papers: when supervision arrives decides the regime more than the architecture does.
- Branching belongs in the state, not the weights: several register groups with their own identities through one shared loop. Gradient descent will differentiate distinct streams (identical streams are a saddle) but fills free space with division of labour, not rival hypotheses, unless the objective pays for alternatives (delayed reward, credit for any stream the kernel accepts, a later selection stage). Collapse has two sources — same rule and input, and attention's averaging — and re-injected stream identities address both. Tension recorded: path independence (one endpoint from any start) is what generalizes in Anil et al.; many endpoints are legitimate only where many answers are valid, which proofs allow and the kernel can check.
- A Lean tactic, for reference in later design: a named instruction with arguments that transforms a proof state; the arguments (which lemma, which term, which intermediate claim) are where ideas live; branches are worth having only if they differ in strategy or key idea, not in surface form.

**The 12 Nov literature addition** ([literature.md](literature.md), "Structure that emerges when a loop is given room"). Looped transformers reach systematic compositional generalization where vanilla ones fail, extrapolate in depth with more loops and then overthink (Kohli et al. 2026); final-answer-only training yields identifiable stepwise mechanisms with learned progress signals, and errors compound past the horizon (Liang et al., Sep 2026); roles separate from a single asymmetry (Shen et al. 2026); compositional intermediate representations depend on the curriculum (Lee et al. 2025); dense models already carry composable task vectors (Todd et al. 2024). The lever in every case was a training condition, not a module.

**What this does to the candidates of Entries 83–84.** "A looped model composes and generalizes systematically" and "its mechanism can be read" are done, this year, on supplied programs with one kind of step. What is not found: the loop *choosing* among kinds of operation — the original thesis's router — and whether that choice lives in the state as a distribution over next steps that sharpens across iterations. That is the original claim (the router is an amortized posterior over programs) turned into a measurement on an unengineered loop, and rung 4's carrier already supplies its ground truth (the exact posterior over shortest-proof first steps, the verifier, coverage control). Stated to the researcher as the candidate contribution, with the cautions that novelty is unverified in a field producing a relevant paper every few weeks, and that small-scale findings did not improve with scale in Kohli et al.

**Decisions.** None. D132 stands; the contribution remains open.

## Template for future entries

Date, premise, evidence (with run or test links), argument, decisions, rejected alternatives, expected falsifiers.
