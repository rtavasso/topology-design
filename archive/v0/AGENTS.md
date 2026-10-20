# Coding agent operating contract — topology-design

**Read in this order before modifying code:** [README](README.md), [critical review](docs/14_critical_review_and_revised_scope.md), [decision gates](docs/17_decision_gates_and_novelty.md), [matched linear study](docs/15_matched_linear_topology_study.md), [functional learning metrics](docs/16_functional_learning_and_transfer_measurement.md), [intellectual history](docs/01_intellectual_history.md), [research charter](docs/02_research_charter.md).

This repository exists to investigate **how parameter-free feed-forward routing changes learning dynamics, functional reuse and interference, and the acquisition/expression of computations**. It is not a generic NAS benchmark or an attempt to build a complete mechanistic-interpretability package immediately.

## Scientific priority and guardrails

1. **Primary studies train with one vocabulary-wide softmax cross-entropy.** Binary toy tasks use two logits. CE is an optimizer objective and sanity check, not the primary scientific outcome.
2. **Pure graph comparison:** use identical named learned operations, shapes, activation templates, parameter counts and learned-operation evaluations. No new learned gates, extra losses, predefined token groups, recurrence or extra inference passes as candidate designs. Diagnostic controls outside this family must be visibly labeled.
3. **One graph change at a time**, with explicit proof of matching and output/gradient behavior where available. Track fixed additions, FLOPs, critical path, memory and activation/Jacobian scales.
4. **Separate claims:** (a) changed expressive function class; (b) changed optimization path within similar expressive class; (c) changed emergence and causal utility of computations. A benchmark win proves none of these alone.
5. **Memorization and generalization are outcome distributions**, not automatically separate neural circuits. Do not assume short=memorization, deep=abstraction or that increasing path length improves generalization.
6. **Mechanistic discovery is inherently partial.** An SAE feature is not canonical, an ablation score depends on baseline, and an inference-time lesion does not prove a path was involved in acquisition.
7. **Analyze function before a candidate mechanism.** Preserve source-agnostic discovery of unexpected changes, with held-out probes and late audit against task-generator truth.
8. **Scaling confounders matter.** Width, feature demand, sparsity, task complexity, number of distinct training examples and measurement resolution vary independently.
9. **No big framework before a falsifiable effect.** Stop/pivot decisions in [decision gates](docs/17_decision_gates_and_novelty.md) are binding on proposed next work.

## Active milestone order

### M0 — math and executable controlled graph pair

Implement the explicit parallel/composed pair from [Study 1](docs/15_matched_linear_topology_study.md):
$$h_P=x+Ax+Bx,\qquad h_C=x+Ax+B(x+Ax).$$
Use a **fixed identical readout** for softmax CE and exact $A=B=0$ for the analytic initial control. Verify:
- same reachable linear input-output maps and trainable matrix sizes;
- same forward outputs, parameter-to-logit Jacobians and empirical NTK at exact zero;
- same first SGD gradient and update;
- exact output difference $RBAx$ and first-step discrepancy $R\Delta B\Delta A x$;
- different mixed second derivatives and potentially different CE Hessians;
- same learned multiply count and fixed merge count; record critical path and memory.

Add all mathematical fixtures from [validation tests](docs/12_validation_tests.md) incrementally. No result about circuit formation should be claimed from the linear pair.

### M1 — first actual training result: function and example-transfer time series

Small teacher-generated **linear conditional token tasks**, identical data/order and seed-paired runs. Track centered-logit/probability functions, teacher-mode and unsupervised learned-mode trajectories, $T_t(i,j)$ exact cloned single-example SGD effects and first-order approximation error. Replicate across data/model seeds; test output scale, small nonzero initialization, LR, weight decay and ultimately optimizer choice.

**Deliverable:** a bounded report on whether higher-order topology-specific parameter interactions alter function/mode learning beyond the identical initial tangent geometry. Include failures, counterexamples and uncertainties. Do not build auto-circuit discovery to do this.

### M2 — nonlinear functional acquisition and reuse

Only after M1 is documented, compare small nonlinear graphs on tasks **both can demonstrably fit at comparable capacity**. Use a finite/full-domain functional atlas and one-example transfer on unseen cases and shortcuts. If one graph cannot express the task at fixed width, report it as **expressivity**, not a training-circuit advantage. Reuse graph generator/probes; do not immediately build a large NAS framework.

### M3 — targeted mechanistic validation

For *observed* unanticipated functional changes, evaluate actual graph-edge interventions, behavior-specific necessity/sufficiency, redundancy, patch-baseline sensitivity, and when justified training-time gradient-route interventions. Only now should a limited automated circuit search be implemented; [causal-depth scores](docs/06_mechanistic_measurement.md) are secondary and operation-granularity-dependent.

### M4 — scale and transfer, later search

Vary width at fixed latent feature demand and vary width with demand together; control sparsity, correlations, data support, composition complexity and optimizer. Use held-out program families and new seeds. Architecture search and attention follow only if earlier measurements predict out-of-sample effects.

## Workflow for every experimental change

1. Write a concise hypothesis, graph difference, controlled variables and **what result would falsify the explanation**.
2. Determine whether the effect could follow from expressive capacity, initialization/parameterization, target-data statistics, or genuine mechanism formation. Label the intended class.
3. Build the smallest analytical or planted fixture; write tests *before* expensive training.
4. Log exact run provenance: graph hashes, code commit, data-generator seed/split hashes, RNG state, operator template/initial weights, model and optimizer configs, checkpoint/probe manifests and compute.
5. Report **actual finite-step updates** and functional outcomes; use tangent kernels and gradient projections as explanations, not replacements.
6. Use paired runs and independent task seeds. Report effect sizes, uncertainties and negative/null regimes; avoid example-level pseudo-replication.
7. Apply [research decision gates](docs/17_decision_gates_and_novelty.md) before broadening model/task size or adding instrumentation.
8. Amend the [decision log](docs/10_decision_log_and_hypotheses.md), including the prior idea being superseded and why.

## Do not silently substitute

- A faster CE curve for better learned computation.
- A different representable function class for a purely optimization-driven effect.
- Equality of parameter count for equality of critical-path compute or effective nonlinear depth.
- Equality of initial NTK for equality of Hessian or later learning.
- A high-order parameter product or input Fourier interaction for a semantic long-path circuit.
- Short-path/long-path labels for a unique parameter decomposition in a shared DAG.
- A trained replacement subnetwork for an inference-time intervention on the original fixed trained graph.
- SAE feature counts or neuron selectivity for true circuit count.
- A probe for a desired feature for discovery of unknown features.
- A finite-step AdamW update for simple SGD gradient dot-product predictions.
- An inference-time lesion for evidence that a route was needed **during learning**.

## First deliverable contract

The coding agent's first plan should be **M0 + the minimal M1 measurement pipeline**, with explicitly deferred auto-circuit search, SAE pipelines, Bayesian NAS and large transformers. The output is tested code, fully specified config and reproducible tiny results, plus a report that distinguishes algebraically certain predictions from observed learning effects.

**Important:** historical docs preserve earlier proposed orders and doubts; [critical review](docs/14_critical_review_and_revised_scope.md), this AGENTS.md, and [decision gates](docs/17_decision_gates_and_novelty.md) describe the **current** priority. Do not erase intellectual history to harmonize it retroactively.
