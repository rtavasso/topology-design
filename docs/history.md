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

## Template for future entries

Date, premise, evidence (with run or test links), argument, decisions, rejected alternatives, expected falsifiers.
