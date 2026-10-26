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

## Template for future entries

Date, premise, evidence (with run or test links), argument, decisions, rejected alternatives, expected falsifiers.
