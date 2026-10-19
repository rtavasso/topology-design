# Instructions for coding agents

Read [README](README.md), [researcher's original questions](docs/00_questions_in_original_words.md), [intellectual history](docs/01_intellectual_history.md), [charter](docs/02_research_charter.md), [measurement framework](docs/06_mechanistic_measurement.md), and [first experiments](docs/13_first_experiment_scenarios.md) **before proposing an implementation**. This repository captures an evolving *research program*, not a performance-driven NAS challenge.

## Mission and epistemic discipline

Build a reproducible environment for answering **how parameter-free information-routing topologies affect the development, expression, interference, memorization and generalization of learned computations**. Do not replace this with “find the best validation accuracy.” We want mathematical explanations and causal evidence that link a specific edge or routing choice to a specific learning dynamic and discovered computation.

Maintain a distinction between (a) exact deductions, (b) hypotheses, (c) method assumptions, (d) observed results. Never describe synthetic sketches as measured results. Preserve null results and false leads.

## Non-negotiable primary experiment constraints

- **Softmax cross-entropy** on tokens, including binary tasks represented by two logits. No MSE as the main training loss. MSE can appear in exact teaching examples and instrumentation calibration, explicitly identified as such.
- **Feed-forward acyclic computation**; no recurrent inference/extra inference iterations in the primary architecture family.
- **No added trainable gates, auxiliary parameters, or teacher networks** as architecture innovations. Diagnostic interpretability methods can train separate instruments only when their limitations and costs are explicit.
- **No a priori semantic token groupings or path-specific supervised tasks.** One combined vocabulary distribution is the base condition.
- Match parameter count, tensor shapes, learned modules and evaluations, activation functions, optimizer, data order, seeds, and budgets in controlled comparisons. Report arithmetic add counts and depth even if parameter-free edges increase FLOPs.
- Treat initialization scale, residual scaling, LayerNorm/BatchNorm placement, parameter symmetry, weight decay, and output readout as first-class confounders.
- **No assumed short=memorization / long=generalization conclusion**. Any such relationship is to be *measured*; length of path, order of parameter interactions, nonlinear compositional depth, causal importance and computational semantics are different notions.
- No unverifiable claim that an extracted subgraph is *the* unique circuit. Report interventions and reconstruction fidelity.
- Hold out whole task-generating families or parameter regimes; do not search and test on the same family then call it an architectural law.

## Developer workflow

1. Specify hypothesis, estimand, topology change, matched control, and anticipated falsification in a short experiment proposal.
2. Build a tiny exact oracle or calibration case *before* optimizing performance.
3. Implement deterministic graph compilation with graph validation and shape checks.
4. Save full experiment provenance (config, git commit, environment, seed, data-generation seed, graph hash, matching controls).
5. Log both model outputs and selected internal states at checkpoints. Store a stable probe-input set; analyses should be recomputable offline.
6. Run light online metrics and defer expensive Jacobian, exhaustive functional atlas and causal-subgraph analysis to post hoc jobs.
7. Report uncertainty across multiple paired seeds, effect sizes, and task/scale interactions, not only best runs.
8. Record failure modes and surprising outcomes in docs/10_decision_log_and_hypotheses.md (or a timestamped successor). Update any mathematical definitions if implementation exposes ambiguity.
9. Before generalizing a finding, attempt basis-robustness, intervention-baseline sensitivity, negative controls, and held-out task-family replication.

## Initial build order (not a demand to build everything immediately)

**Milestone 0: scientific infrastructure** 
- Define schema for a DAG with named learned operations and parameter-free routing; deterministic compile/execute; graph isomorphism/dedup where tractable.
- Exact linear-symbolic or numerical oracle for small graphs; finite-difference/autograd checks; matched-configuration auditor.
- A two-logit CE toy trainer, deterministic data generator, checkpoint/probe API, reproducible run manifest.
- Tests from docs/12_validation_tests.md.

**Milestone 1: observation before search**
- Output-function atlas on small enumerated input domains; Fourier coefficients for balanced Boolean domains and canonicalized logit contrasts.
- Single-example counterfactual training/update analysis, cross-example influence, Jacobian access diagnostics.
- Small-subgraph intervention runner with selectable zero/mean/resampled baselines and fidelity reporting; record whether a subgraph is a subset of the trained graph and never retrain it for the main causal-depth metric.

**Milestone 2: controlled mechanism study**
- Synthetic rule+shortcut+arbitrary-exception dataset with controlled causal variables; train/holdout/shifted distributions and explicit data provenance.
- Time-resolved curves of rule dependence, exception retention, interference, causal-depth utility and representation geometry.
- Paired topology mutations; enough repetitions to observe noise. No Bayesian optimization until instrument reliability and scale controls have been demonstrated.

## Never silently substitute

- A linear probe for causal use.
- A high-order input interaction for a long computational path.
- Individual-neuron selectivity for absence of superposition.
- An SAE dictionary count for number of real features.
- Validation accuracy for mechanistic explanation.
- Gradients of independent branches for independently supervised losses.
- A decrease in loss on an example for that token becoming argmax.
- Parameter count alone for capacity matching.
- A trained replacement subnetwork for a pure inference-time intervention.
- Path ablation as proof of absence when redundant paths can compensate.

## How to report findings

Each experimental report should contain: question; topology diagrams/formal edge lists; controlled variables; instrument calibration; task distribution and splits; pre-registered metrics; time-resolved plots; effect sizes and uncertainty; causal-intervention findings; checks against alternative explanations; failures; next discriminating experiment. Include machine-readable raw output and the full executable config.

## Relationship to LLMs

LLMs are the **destination**, not the initial scale. Prefer simpler settings permitting exhaustive input enumeration, analytic checks and exact edge interventions. Scale one factor at a time: widths, latent feature demand, sparsity/correlation, required composition, data support, then attention/token context. Claims should state the domain where they hold.

## Agent kickoff instruction

Start by reading all documents and writing a **short implementation plan with explicit out-of-scope items**. Implement Milestone 0 only first. At every stage use the smallest program that can falsify a scientific claim. Don't scaffold a giant general-purpose mech-interp or NAS library until a validated measurement requires it.
