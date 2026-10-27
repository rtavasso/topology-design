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

## Template for future entries

Date, premise, evidence (with run or test links), argument, decisions, rejected alternatives, expected falsifiers.
