# Topology Design: Mechanistic Research on Information Routing and Learning

**Status:** research charter and agent handoff (19 October 2026). **No experiments have been run in this repository.** Claims about topology advantages are hypotheses unless marked as mathematical deductions or cited literature.

## North star

> Can parameter-free, acyclic computational routing preserve useful shallow behavior, selectively destroy inadequate representations, and construct deeper reusable computations—while ensuring errors unresolved by simpler computations remain accessible to the operations capable of resolving them?

The larger goal is **understanding how topology shapes memorization, generalization, circuit formation, and interference**, eventually enabling more intentional allocation of parameters and computation in language models. We are **not** searching for a magic topology, equating low CE with understanding, or assuming longer paths generalize and shorter paths memorize.

This project began with a first-principles question about residual connections and evolved into a mechanistic experimental program. **Retain that development; the rejected ideas explain our controls.**

## Read this first (for coding agents)

1. [Original research questions](docs/00_questions_in_original_words.md): the researcher's wording, retained to preserve intention and pivots.
2. [AGENTS.md](AGENTS.md): working contract, implementation order, and guardrails.
3. [Research charter](docs/02_research_charter.md): problem definition, constraints, and what constitutes a useful result.
4. [Intellectual history](docs/01_intellectual_history.md): what led here, why priorities changed, and paths not to revive uncritically.
5. [Mathematical foundations](docs/03_first_principles_math.md) and [gradient dynamics](docs/04_gradient_dynamics.md): exact examples, derived claims, and limitations.
6. [Topology grammar](docs/05_topology_space.md): controlled family of graphs and confounders.
7. [Mechanistic measurement](docs/06_mechanistic_measurement.md): discovering computations without pre-assuming a circuit.
8. [Task and scale design](docs/07_datasets_and_scale.md), [protocol](docs/08_experimental_protocol.md), [software blueprint](docs/09_software_blueprint.md), [test plan](docs/12_validation_tests.md), and [first discriminating experiments](docs/13_first_experiment_scenarios.md).
9. [Decisions and hypotheses](docs/10_decision_log_and_hypotheses.md); [annotated literature](docs/11_literature.md).

## The scientific object

We vary **parameter-free edges and fixed routing** in a feed-forward DAG whose trainable transformations, count of trainable parameters, activation types, objective (softmax cross-entropy), and training regimen are controlled. Analyze small linear graphs exactly; add nonlinearities and known generative tasks; then scale only when mechanisms are measurable. Additional learned gates, auxiliary supervised objectives, recurrence, or externally specified token groups are *not* part of the primary comparison. They can be secondary diagnostic controls, clearly labeled.

Three foundational observables:

- **Residual-error accessibility:** what portion of a CE error signal is outside shallow first-order function changes but accessible to deeper ones?
- **One-example transfer/interference:** which other contexts improve or degrade after learning one example, and why?
- **Circuit development and utility:** what computations emerge over training, how do we find them without knowing their form, and what is their causal importance for seen exceptions versus unseen structured cases?

Evaluate these across **task complexity × model capacity × feature demand/sparsity × data support × training time**. Benchmark CE and accuracy are *context*, not primary research conclusions.

## A central warning

1. **Path length is not semantic complexity.** A high-order matrix product can still be a linear function; one nonlinear module can implement a complex rule.
2. **Ablation is not automatically a circuit measurement.** Redundancy, off-distribution replacements, and shared nonlinearities make naive lesion effects misleading.
3. **Neurons and SAE features are not canonical units.** Superposition, basis choices, and analysis resolution confound apparent circuit counts.
4. **Topology effects depend on parameterization and scale.** Identical parameter count is necessary but insufficient; track signal scales, fixed arithmetic, learned-operation counts, and representational pressure.
5. **A single CE loss over additive logits is coupled.** A path can fit examples before other paths build useful features. Artificial independence, forced disagreement, or hand-specified token groups are not the desired solution.
6. **No claim of empirical success yet.** All proposed metrics need validation on known mechanisms, negative controls, and representation transformations before use in topology search.

## Proposed program

- **Phase 0:** build a reproducible graph executor, exact small-graph oracle, instrumentation and unit tests. Establish that purportedly matched graph pairs are actually matched.
- **Phase 1:** linear and tiny nonlinear graph studies; demonstrate which graph differences are mathematical and which are optimization-dependent. Calibrate measurement reliability using planted mechanisms.
- **Phase 2:** synthetic token-prediction tasks mixing reusable latent rules, shortcuts, and arbitrary exceptions; capture training histories and basis-robust functional changes.
- **Phase 3:** causal-depth and training-time intervention studies; reveal which routes actually acquire and express memorization/generalization behavior.
- **Phase 4:** scale sweeps, controlled edge mutations, then small exhaustive/random/evolutionary searches; only use Bayesian surrogates after validating that predictors relate to mechanisms, not merely fast CE descent.
- **Phase 5:** attention and small transformers, then progressively more language-like data; eventually test whether resulting principles transfer to LLM-scale architectures.

## How to use the documentation

Documents deliberately separate:
- **[MATH]** exact derivations or results under stated assumptions;
- **[LITERATURE]** externally established findings, with cited source;
- **[HYPOTHESIS]** untested interpretation or prediction;
- **[DESIGN]** proposed experimental choice;
- **[PIVOT]** approach discussed and set aside, with rationale;
- **[OPEN]** unresolved scientific or engineering question.

A coding agent should **not** silently promote hypotheses to facts, or rewrite the scientific question to match convenient pre-existing benchmarks. Before implementing a major system, first implement the smallest falsifying test. Update the decision log with the result and its limitations.

See [AGENTS.md](AGENTS.md) for explicit deliverables and acceptance criteria.

## Repository content

This initial commit establishes **documentation only**. Code, data, experiment results, checkpoints, and validated conclusions are future work. No reproducible experiment or improvement claim should be inferred from the presence of a proposed protocol.
