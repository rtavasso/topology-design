# Experimental protocol: from causal question to an interpretable result

This document is a study design, not a command to launch a giant sweep. The coding agent should implement capabilities incrementally and write actual results back to versioned experiment reports.

## 1. Pre-register the hypothesis for each graph mutation

Required proposal fields:
- **Question / hypothesis:** one mechanism-sensitive prediction.
- **Topology contrast:** exact graph node/edge lists, hashes and minimal edit difference.
- **Matched operations:** trainable params, learned operator templates, evaluation count, activations, output head, loss, optimizer, examples, seeds.
- **Resource budget:** training steps, tokens, wall time, fixed addition count and inference FLOPs.
- **Observable:** functional interaction, causal-depth utility, transfer/interference, or circuit emergence; say how computed.
- **Alternative explanations:** initialization scale, path length, output gauge, graph symmetry, data support, representational pressure, intervention artifacts.
- **Falsification:** outcome under which the hypothesis fails.
- **Controls:** negative and positive controls, and whether expected circuit was planted.
- **Measurement risks:** what instrument would fail to detect.

Do not define a hypothesis after observing a favorable run without labeling it exploratory.

## 2. Study phases and acceptance gates

### Phase 0 — infrastructure, exact oracles, no architecture search

Deliver a deterministic declarative graph runtime; topology generator; parameter/compute matching audit; 2-logit CE trainer; manifest and checkpoint/probe persistence; exact scalar/linear polynomial calculator; finite-difference gradient/Hessian tests. Tests must reproduce docs/03 and docs/04 examples. Must not claim circuit insights yet.

**Pass:** all mathematical fixture tests pass; same seed and config give identical CPU outcomes within stated floating-point tolerance; graph hash and manifests identify reproducible runs.

### Phase 1 — measurement validity with planted mechanisms

Train or construct tiny networks with known short/long computational routes, redundant circuits, superposed features and one neutral route. Build functional atlas and inference-time interventions. Confirm discovered graph mechanisms are faithful under multiple baselines and avoid reporting nonexistent circuits. Include a known mechanism that was *not* among detector templates.

**Pass:** calibration report quantifies mechanism discovery sensitivity and specificity, intervention baseline sensitivity, and bias from fixed circuit candidates.

### Phase 2 — matched graph pair longitudinal experiment

Choose a single one-edge difference within the narrow topology grammar. Train paired seeds on rule+shortcut+exception task under softmax CE. Snapshot regularly *before* shortcut saturation, around saturation, during possible circuit formation and late training. Run mechanistic reports offline.

**Pass:** complete longitudinal functional atlas and causal-depth profiles across at least two probe distributions, with actual train-time gradient dynamics and uncertainty. A negative result is valid if fully measured.

### Phase 3 — small grammar enumeration and controlled search

Enumerate small $L$ where tractable (e.g. 4–6 operators gives 24–720 labeled source-parent graphs). Compute structure without training everything. Compare meaningful representatives and one-edge neighbors; use random sampling as a baseline if training many graphs. Verify graph equivalence and module-permutation controls.

**Pass:** a measured relationship or documented nonrelationship between a graph descriptor and discovered mechanisms, across paired runs.

### Phase 4 — scale / feature-demand regime map

Vary capacity alone, feature demand alone, both together, sparsity, program complexity and dataset size. Replicate selected topology effects and probe measurement invariance.

**Pass:** report boundaries of the effect and plausible confounders; no universal scaling claim from one setting.

### Phase 5 — attention and LLM relevance

Port only validated instrumentation and graph grammar concepts to small transformers; select tasks whose circuits can still be discovered/causally measured. Treat attention weight routing as learned and distinguish it from fixed skip topology. LLM-scale training is a later destination, not acceptance for earlier science.

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

## 5. Time-resolved analysis

At each checkpoint:
1. Store output probabilities and centered logits over probes.
2. Compute function decomposition where mathematically defined and record unanticipated interactions.
3. Compute lightweight representation spectra/effective dimensionality.
4. Evaluate targeted example transfers with cloned optimizer state.
5. Run full causal-depth subgraph search only on selected checkpoints; record search budget, exact vs approximate and intervention rule.
6. Cluster functional fingerprints of candidate circuits across checkpoints.
7. Compare changes in discovered circuits to learning dynamics and outcome distributions.
8. Estimate training-time path dependence through carefully chosen replay/freeze/gradient intervention clones.

Do not call a decodable feature causal without perturbation evidence. Avoid analyzing only final checkpoint; a useful circuit may have formed and later been replaced.

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

## 8. Decision gates before topology search

No search until the measurement suite:
- rediscovers planted mechanisms without template leakage;
- returns a 'cannot identify' conclusion when ambiguity/redundancy prevents unique circuits;
- passes invariance/representation basis tests where exact symmetry applies;
- distinguishes memorization from rule transfer at the *behavioral* level;
- shows a stable relation between path interventions and actual output changes;
- is computationally tractable with realistic sampling/approximation budgets.

When search begins, rank by multiple mechanistic outcomes and do not optimize early CE as proxy until validated against long-term mechanism outcomes.

## 9. Experiment-report template

**Research question / graph mutation / predicted effect**  
**Formal topology, shapes, parameter/FLOP audit**  
**Task, data, splits and causal variables**  
**Loss, optimizer, initialization, seeds, training duration**  
**Pre-registered endpoints and instrument calibration**  
**Plots: CE only as context; functional atlas; causal-depth reproduction and utility; one-example transfer; representational spectra; discovered circuit fingerprints over time**  
**Sensitivity and alternate explanations**  
**What was learned; what was not identified**  
**Falsification results and next clean experiment**.

A null result with verified instrumentation is more valuable than an impressive but uninterpretable training curve.
