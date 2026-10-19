# Software blueprint for research agents (minimal, test-driven)

This is a design sketch for the first codebase, not a request to implement all parts immediately. Prefer a small Python/PyTorch core plus offline analysis modules, typed configs, tests and immutable experiment artifacts. Avoid a generalized plugin framework or complex dashboard before scientific validity is demonstrated.

## 1. Architectural boundaries

Separate five responsibilities:
- **Graph:** declare/validate/compile fixed-operator DAGs, enumerate/hash and audit them.
- **Data:** generate deterministic synthetic programs, shortcuts, exceptions and probe partitions; store private generator truth for *post hoc* auditing.
- **Train:** softmax CE loop with paired seeds and checkpoints. Ordinary training does not include causal intervention or extra auxiliary loss.
- **Observe:** offline function/representation/gradient measurements on checkpoint+probe artifacts.
- **Intervene:** reproducible edge/activation/gradient interventions and subgraph search, with baseline and budget metadata.

This separation prevents instrumentation choices from accidentally becoming part of the learned model.

## 2. Suggested package structure (adapt only with a reason)

- src/topology_design/graphs/ — schemas, validators, DAG executors, parent-selection family, path enumerator, static cost/parameter auditor.
- src/topology_design/models/ — fixed learned operator templates, output heads, initialization modes, optimizer factories.
- src/topology_design/data/ — rule generators, exceptions, shortcuts, deterministic splits and finite-domain enumerators.
- src/topology_design/training/ — runner, snapshots, paired-run orchestrator, checkpoint metadata.
- src/topology_design/analysis/function/ — canonical logits, Boolean Walsh transform, empirical interaction atlas, functional similarity.
- src/topology_design/analysis/gradients/ — autograd Jacobian-vector probes, single-example paired update analysis, accessibility and conditioning estimates.
- src/topology_design/analysis/causal/ — edge patches, subgraph generation, intervention baselines, reproducibility/utility profiles.
- src/topology_design/analysis/representations/ — covariance spectra, CKA/subspace metrics and later controlled SAE experiments.
- src/topology_design/reports/ — validated result serialization and plots/tables with uncertainty.
- configs/, experiments/, tests/, notebooks/ (optional), artifacts/ (ignored large local outputs), docs/ (scientific documents).

Do **not** generate an empty class for every imagined future feature. Build modules when a milestone needs them.

## 3. Minimal API contracts

**GraphSpec**: version, ordered learned operator specs, node IDs/kinds, named ports/edges, fixed ops, output/readout spec, dtype/dimension, annotations. Must support stable hash, serialization, topological validation, stats, and deterministic compilation.

**MatchedPairAudit**: graph hashes, editable edge diff, learned parameter total, per-operation shapes and weight seeds, learned evals, activation type counts, fixed operation counts, total estimate of FLOPs/activation storage, initialization signal stats. Fail loudly if an asserted primary matched pair diverges on required properties.

**TaskSpec**: generator version, latent program ID and parameters, shortcut reliability, exception ID/label sampling policy, train/test/probe seeds, split sizes, complete data checksum, ground-truth blinding flag.

**RunSpec**: graph/task/optimizer specs, steps, batch size, LR, weight decay, seeds (model/data/order/probes), framework version/commit, deterministic settings, logging and checkpoint cadence.

**ModelObservation**: checkpoint ID, probe set ID, centered logits/probs/labels, selected layer activations, gradient summaries, graph metadata. For large runs, store compressed arrays in columnar/binary format with manifest.

**InterventionSpec**: graph/checkpoint/probe IDs, masked nodes/edges, replacement type (zero/mean/resample/counterfactual), seed and reference distribution, instrumentation version, whether gradients or only activations are changed.

**MechanismReport**: behavior contrast ID, discovered candidate subgraph(s), search coverage/budget, reproduction fidelity, CE utility by distribution, necessity/sufficiency/joint interactions, uncertainty and unexplained fraction.

## 4. Reproducibility artifacts

Use a content-addressed experiment directory or stable run ID from a canonical JSON config + git revision + graph hash + dataset hash. A run record should include:
- full resolved configs, environment, installed versions, hardware;
- RNG seeds and states where reproducibility requires;
- dataset-generation code version and split hashes;
- initial weights and optimizer state;
- checkpoints and a fixed manifest of probe inputs;
- output predictions and selected internal activations;
- metric computation version and raw intermediate measurements;
- explicit claims with accompanying plots/error bars.

If a random generator or framework kernel is nondeterministic, record it and avoid claiming bit-for-bit reproducibility. A later analysis version should be able to reproduce a published metric from stored checkpoint+probe data.

## 5. Common run pipeline

1. Instantiate TaskSpec and all fixed split/probe sets.
2. Instantiate a graph pair from GraphSpec; run matching audit.
3. Draw one initialization state for named learned modules and clone identically across topologies. Resolve topology-dependent calibrated initialization separately.
4. Train paired models with same example order, objective and budget; capture checkpoints.
5. Run light per-checkpoint CE/accuracy/calibration and function-probe logits.
6. Run offline atlas and representation metrics.
7. At chosen checkpoints, run optional per-example cloned-gradient-update analyses.
8. Use functional changes (not expected mechanism templates) to generate behavioral contrasts.
9. Search and validate causal subgraphs, recording intervention baseline, exact/approximate search status and fidelity.
10. Aggregate paired-run effects and write a research report; keep raw data and nulls.

## 6. Suggested configuration (illustrative fields)

Graph:
- family: parent_residual_v1
- operator_count: 4
- width: 32
- operator_template: 2layer_relu_mlp
- parents: [0,1,0,2]
- readout: shared_vocab_linear
- vocabulary_size: 2

Task:
- family: rule_shortcut_exception_v1
- generator_seed: 0
- program_family: boolean_composition
- shortcut_train_probability: 0.9
- shortcut_shift_probability: 0.5
- exception_fraction: 0.05
- latent_feature_count: 6
- latent_activity_probability: 0.5

Train:
- loss: softmax_ce
- optimizer: sgd
- batch_size: 64
- steps: 20000
- weight_decay: 0.0
- checkpoint_schedule: log_and_transition_dense
- model_init_seed: 17
- data_order_seed: 19

Analysis:
- canonical_logits: center_vocabulary
- probe_distribution: balanced_factorial
- functional_atlas: walsh_exact_if_enumerable
- intervention_baselines: [zero, conditional_mean, resample]
- causal_subgraph_search: exact_if_small
- feature_analysis: spectral_first
- blind_mechanism_discovery: true

These are **example settings, not recommended hyperparameters**. Never assume $20k$ steps, width 32 or 5% exceptions suffice; calibrate.

## 7. Computational scaling

For tiny graphs, exact all-input/all-subgraph evaluation is preferred and gives clean validation. When moving to larger models:
- cache/stream activation outputs on a fixed probe set;
- use autograd JVP/VJP and randomized spectral estimators;
- subsample intervention candidates using discovery-stage behavior contrasts;
- report how estimates change with probe/search budget;
- retain a tiny exactly enumerable problem as continuous reference.

Exhaustive subgraphs scale as $2^E$ (worse for multi-edge interventions); do not attempt exhaustive search on nontrivial models. Train-time gradient interventions are expensive: use cloned checkpoint microstudies instead of instrumenting every regular run.

## 8. CLI / entry points (eventual, minimal)

- validate-graph: graph shape, acyclicity, graph hash, path descriptor and matching audit.
- generate-task: deterministic split and probe generation with checksum.
- train: one GraphSpec/TaskSpec/RunSpec.
- train-paired: matched topology mutation, cloned initialization and common order.
- analyze-functions: output atlas and unanticipated structure candidate list.
- analyze-causality: directed interventions and small-subgraph search on selected checkpoint+probe distribution.
- report: markdown/plots/machine-readable summary, with source run references.

These names are illustrative; a simple script is acceptable at Phase 0.

## 9. What not to build yet

- Full LLM training harness, CUDA custom kernels, automated hyperparameter tuning or cloud orchestrator.
- A large architecture search controller or Bayesian predictor.
- An SAE training farm as the default circuit extractor.
- Model-side learned gating, recurrence, grouped token heads or multiple training losses.
- A polished visualization app before metric reliability and causality are established.

The agent should deliver running, tested scientific primitives rather than scaffolding every future phase.

## 10. Repository governance

- Keep docs/01 intellectual history immutable in spirit: add dated amendments rather than erasing why an idea was abandoned.
- Make controlled scientific changes through short proposals with falsification conditions.
- Every result references immutable graph hash, dataset checksum and code commit.
- New metrics require calibration fixtures, sensitivity tests and documented failure conditions.
- Separate 'foundational math fixture passed' from 'training experiment run' and 'scientific hypothesis supported.'
