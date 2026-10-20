# Experimental protocol: from controlled graph changes to interpretable learning dynamics

**Active revision (19 October 2026):** [critical review](14_critical_review_and_revised_scope.md), [matched linear study](15_matched_linear_topology_study.md) and [decision gates](17_decision_gates_and_novelty.md) take precedence over any older broad architecture-search plan. Functional learning and actual one-example transfer come **before** full circuit extraction.

This document is a study design, not a command to launch a giant sweep. The coding agent should implement capabilities incrementally and write actual results back to versioned experiment reports.

## 1. Pre-register the hypothesis for each graph mutation

Required proposal fields:
- **Question / hypothesis:** one mechanism-sensitive prediction.
- **Topology contrast:** exact graph node/edge lists, hashes and minimal edit difference.
- **Matched operations:** trainable params, learned operator templates, evaluation count, activations, output head, loss, optimizer, examples, seeds.
- **Resource budget:** training steps, tokens, wall time, fixed addition count and inference FLOPs.
- **Observable:** *first* function evolution and actual one-example transfer; *later* causal utility of specific observed changes. Path depth alone is not a mechanism.
- **Type of proposed advantage:** representational capacity, optimization among similarly expressive models, or mechanism formation. Explicitly assess mismatched expressive competence; alternative explanations include initialization scale, path length, output gauge, graph symmetry, data support, representational pressure, and intervention artifacts.
- **Falsification:** outcome under which the hypothesis fails.
- **Controls:** negative and positive controls, and whether expected circuit was planted.
- **Measurement risks:** what instrument would fail to detect.

Do not define a hypothesis after observing a favorable run without labeling it exploratory.

## 2. Revised study phases and acceptance gates

**Gate 0 — Exact algebra and engineering:** implement graph expressions, shape/parameter/resource auditor, symbolic/finite-difference/autograd oracles and a two-logit CE trainer. Establish the matched linear pair's initial-function, parameter-Jacobian, empirical NTK and first-SGD-update equality at zero; verify its $BAx$ mixed derivative and CE Hessian difference. The existing scalar residual and polynomial fixtures remain. **Pass:** mathematical identities and reproducibility tests pass; do not attribute function changes to an unchecked graph.

**Gate 1 — Linear function/mode learning:** run the **parallel versus composed** matrix graph pair from [Study 1](15_matched_linear_topology_study.md) on controlled linear conditional-token teacher data. Same learned operators and fixed readout, common CE, data/order and paired initialization. Track centered-logit functions, teacher modes and unsupervised modes, actual $T_t(i,j)$, first-order prediction errors and effect sensitivity to scales and optimizer. **Pass:** an interpretable, reproducible topology-dependent optimization difference (or a documented null identifying a limit) across independent seeds; circuit discovery is **not** a prerequisite.

**Gate 2 — Nonlinear acquisition with competence checks:** select small feed-forward DAG rewires where both models demonstrably fit the chosen compositional task at fixed capacity. Full finite-input function atlas, unexpected interaction discovery and example transfer are primary. If one target is unrepresentable by one topology, reclassify the result as **expressivity**, not a circuit-learning result. **Pass:** repeatable differences in *what function is acquired* and transfer beyond any obvious approximation gap.

**Gate 3 — Targeted causal explanation:** only after a functional difference survives Gate 2, calibrate interpretation instruments on planted/unknown-to-detector mechanisms, redundant paths and baselines. Use inference-time ablation/patching, fidelity and joint interventions; use training-time replay interventions to distinguish acquisition from expression. **Pass:** at least a bounded causal claim with reported intervention sensitivity, not necessarily a unique global circuit.

**Gate 4 — Scale/data regime map:** vary model width, feature demand, sparsity/co-activation, composition complexity and data support independently and jointly. Confirm where the earlier effect persists, reverses or disappears. Held-out program families and optimizer controls test transfer. **Pass:** credible boundary conditions, not necessarily an unconditional improvement.

**Gate 5 — Search/attention only after prediction:** test whether validated graph descriptors forecast functional/circuit measurements on unseen graphs and tasks. Only then consider random/evolutionary/Bayesian search or small-attention extension. Never let fast early CE substitute for the actual dependent variable.

Decision to continue/revise/stop each hypothesis is defined in [decision gates](17_decision_gates_and_novelty.md). A negative result with sound instrument controls is scientifically valid.

## 3. Training run specification

Use softmax CE with exact vocab and logits readout consistent across graphs. For binary labels, preserve 2-logit convention. Dataset generator and splits are determined before training.

Control:
- cloned initialization state of each named learned operator in paired graphs;
- identical sample order and batching; if dropout exists, synchronize random streams or disable initially;
- optimizer config, weight decay, precision and LR schedule;
- fixed training tokens/steps, not 'train until convergence' with different budgets;
- default no auxiliary training loss or learned architecture gate;
- checkpoint steps dense around known transitions and log-spaced for long runs;
- explicit dataset and checkpoint IDs to reconstruct later probes.

Run SGD initially to validate one-example gradient predictions and then AdamW as an optimizer robustness check. A separate study may vary regularization because grokking can depend on it.

## 4. Probe design

- **Canonical function probes:** exhaustive small input domain or stable stratified sample. Same across checkpoints and graphs.
- **Seen exception probes:** identical train contexts/keys; measure forgetting and retention.
- **Unseen rule probes:** new latent combinations; test generalization.
- **Shortcut intervention probes:** same latent target, altered shortcut distribution; isolates shortcut dependence.
- **Related/unrelated/counterfactual probes:** support single-example transfer analysis.
- **Discovery-held-out probes:** do not use to select candidate circuits; measure external validity of detected behavior.
- **Mechanistic counterfactual pairs:** alter a single latent feature or internal activation under realistic and unrealistic baselines, recording both.

Ensure sufficient coverage for each subgroup and report sampled support. A confidence interval around a biased probe set is not causal validity.

## 5. Time-resolved observation: the mandatory core

At each checkpoint, prioritize:
1. Store centered-logit function and probabilities on fixed probes (same graph/output gauge and probe distribution); retain train/test CE as a sanity check.
2. Decompose learned function changes: for linear Study 1, projected teacher-mode and unsupervised logit-map/SVD evolution; for small nonlinear domains, complete Boolean Fourier or appropriate categorical interactions without assuming the expected circuit.
3. On selected checkpoints, clone state and perform controlled one-example SGD updates. Record **actual finite-step loss/logit changes** on a balanced relation-labeled probe grid, compare with first-order gradient estimates and audit generalization/retention.
4. Track optimization explanations (initial Jacobians, time-varying tangent kernel, mixed parameter curvature, representation spectra) as needed to discriminate hypotheses.
5. Look for unexpected repeatable behaviors; reserve task generator latent labels for audit rather than passing them to an all-knowing probe.

**Only after a reproducible functional change is detected** should the study perform broader causal graph analyses:
6. Calibrate interventions on known and unknown-to-detector mechanisms.
7. Discover candidate subgraphs for those actual behaviors, with multiple baselines and fidelity/error coverage.
8. Test training-time acquisition interventions and joint redundancy effects when the inference graph alone is ambiguous.

Causal-depth utility profiles, SAE feature decompositions and automated full-network circuit discovery are **not** prerequisites to initial linear CE training. Details: [functional measurement](16_functional_learning_and_transfer_measurement.md) and [mechanistic measurement](06_mechanistic_measurement.md).

## 6. Interventions and controls

For each candidate subgraph consider zero, mean, resampled and causal-corruption/clean-patch baselines where applicable; benchmark off-distribution effects. For redundant circuits examine multi-edge removal, not just isolated edges. Compare a sham intervention of equal numerical footprint.

Differentiate:
- removal of edge contribution at inference;
- freezing parameters during training;
- stopping gradients along a path while preserving forward activation;
- deleting an edge from the beginning;
- training a new subgraph from scratch.

These answer different causal questions. A re-trained subnetwork is *not* the same as a faithful retained circuit in the original model.

## 7. Statistics and robustness

Use paired random seeds where possible and report the distribution of within-seed differences; include multiple dataset seeds and program instances. Define primary endpoints before comparing candidate graphs. Report confidence intervals/bootstrap appropriately at the *run* or task-instance level, not treating thousands of correlated probe examples as independent replications.

Where a large family of graphs is inspected, address multiple comparisons, selection bias and confirmation on new seeds and held-out tasks. Report effect size, not only p-values or best-case runs.

An observed topology effect must be interrogated via:
- raw initialization vs calibrated activation/Jacobian scale;
- operator labeling/permutation;
- optimizer choice;
- data-generation seeds;
- probe distribution;
- baseline sensitivity;
- width-feature-load coupling;
- ablation redundancy;
- checkpoint selection.

## 8. Research gating before automated topology search

An architecture search objective is legitimate only after one or more structural descriptors have **out-of-sample predictive value** for function/transfer/circuit measurements. Required evidence:
- exact/paired graph identities checked; expressive function-class differences documented or controlled;
- stable functional phenomenon across independent seeds and task instances, not just validation CE;
- initial first-order effects separated from higher-order parameterization and regularization where possible;
- source-agnostic functional discovery tested on unknown planted interactions before declaring a discovered circuit;
- intervention-baseline, redundancy and feature-load biases examined for mechanistic outcomes;
- results transferred to an independently generated task family and more than one model scale;
- random/local search baselines and multiple-comparison correction for any guided search.

Do not launch a search to compensate for a null experiment. Make the hypothesis more discriminating or stop it.

See [decision gates](17_decision_gates_and_novelty.md).

## 9. Experiment-report template

**Research question / graph mutation / predicted effect**  
**Formal topology, shapes, parameter/FLOP audit**  
**Task, data, splits and causal variables**  
**Loss, optimizer, initialization, seeds, training duration**  
**Pre-registered endpoints and instrument calibration**  
**Plots: CE only as context; function/mode evolution and exact one-example transfer FIRST; causal mechanisms and representation/depth descriptors only where those have been validated**  
**Sensitivity and alternate explanations**  
**What was learned; what was not identified**  
**Falsification results and next clean experiment**.

A null result with verified instrumentation is more valuable than an impressive but uninterpretable training curve.
