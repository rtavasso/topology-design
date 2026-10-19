# Software blueprint for research agents (minimal, test-driven)

**19 October 2026 implementation revision:** first deliver a tiny exact parallel/composed matrix-graph study with softmax CE, time-resolved learned functions and one-example update-transfer. Do **not** build a general causal subgraph searcher before this yields useful observations. The previous architecture inventory remains a future modular target, not an initial scaffold. See [AGENTS.md](../AGENTS.md), [study](15_matched_linear_topology_study.md) and [gates](17_decision_gates_and_novelty.md).

This is a design sketch for the first codebase, not a request to implement all parts immediately. Prefer a small Python/PyTorch core plus offline analysis modules, typed configs, tests and immutable experiment artifacts. Avoid a generalized plugin framework or complex dashboard before scientific validity is demonstrated.

## 1. Architectural boundaries

Separate five responsibilities:
- **Graph:** declare/validate/compile fixed-operator DAGs, enumerate/hash and audit them.
- **Data:** start with deterministic **linear conditional-token teachers and fixed probe distributions**, then extend to synthetic programs, shortcuts, exceptions and latent-truth auditing.
- **Train:** softmax CE loop with paired seeds and checkpoints. Ordinary training does not include causal intervention or extra auxiliary loss.
- **Observe:** start with exact centered-logit function/mode histories and per-example cloned-update transfer; add representation/gradient/influence instruments selectively.
- **Intervene (later):** targeted edge/activation/gradient interventions and subgraph search only for a discovered, replicable function-level phenomenon, with baseline and budget metadata.

This separation prevents instrumentation choices from accidentally becoming part of the learned model.

## 2. Suggested package structure (adapt only with a reason)

- src/topology_design/graphs/ — first a strict parallel/composed matrix-graph executor and matching auditor; later DAG schemas, parent-selection family, path enumerator and resource auditor.
- src/topology_design/models/ — fixed learned operator templates, output heads, initialization modes, optimizer factories.
- src/topology_design/data/ — first linear softmax teacher with known modes, deterministic splits/probes; later rule generators, exceptions, shortcuts and finite-domain enumerators.
- src/topology_design/training/ — runner, snapshots, paired-run orchestrator, checkpoint metadata.
- src/topology_design/analysis/function/ — first centered logits, linear logit-map and teacher-mode/unsupervised mode histories; later Boolean Walsh transform, categorical interaction atlas and functional similarity.
- src/topology_design/analysis/gradients/ — analytic Jacobian/Hessian checks and **actual cloned single-example update-transfer matrix** first; later randomized Jacobian projections, accessibility and conditioning estimates.
- src/topology_design/analysis/causal/ — **deferred until a function-level phenomenon is established**: edge patches, retained-subgraph analysis, intervention baselines, reproduction and utility profiles.
- src/topology_design/analysis/representations/ — covariance spectra, CKA/subspace metrics and later controlled SAE experiments.
- src/topology_design/reports/ — validated result serialization and plots/tables with uncertainty.
- configs/, experiments/, tests/, notebooks/ (optional), artifacts/ (ignored large local outputs), docs/ (scientific documents).

Do **not** generate an empty class for every imagined future feature. Build modules when a milestone needs them.

## 3. Minimal API contracts

**GraphSpec**: begin with a strictly validated parallel_matrix_v1/composed_matrix_v1 family with identical dimensions/matrices and fixed output readout; later generalize to versioned ordered learned operator specs, node IDs/kinds, named ports/edges and fixed ops. Must support stable hash, serialization, topological validation, stats, and deterministic compilation.

**MatchedPairAudit**: graph hashes, editable edge diff, learned parameter total, per-operation shapes and weight seeds, learned evals, activation type counts, fixed operation counts, total estimate of FLOPs/activation storage, initialization signal stats. Fail loudly if an asserted primary matched pair diverges on required properties.

**TaskSpec**: begin with linear_softmax_teacher_v1: $M_*$, readout $R$, temperature, input covariance, class probabilities, train/test/probe seeds and dataset hash. Later extend to latent program IDs, shortcut reliability, exception IDs/labels and blinding of generator truth.

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

## 5. Initial minimal pipeline (no mechanism extractor)

1. Generate fixed linear softmax teacher dataset, train/test samples, and complete/stratified probe set with recorded seeds.
2. Compile the parallel and composed matrix graph with **identical matrix parameter tensors and fixed readout**.
3. Assert same initial outputs, Jacobians, NTK and first SGD gradient at $A=B=0$. Verify second-order $BAx$ divergence and the CE Hessian distinction using tiny numerical fixtures.
4. Train with softmax CE, identical data order and optimizer states; checkpoint on a dense enough schedule to observe changes.
5. Save centered logits/probabilities on fixed probes, learned linear logit maps and teacher-aligned plus unsupervised mode spectra.
6. For selected checkpoints, clone model/optimizer state and compute actual single-example SGD effects on a small balanced grid of probes; check the $\eta g_j^\top g_i$ limit.
7. Aggregate paired seeds, target-mode orientations and initialization/scale/optimizer sensitivity; report differences and counterexamples.
8. Apply [Gate 1](17_decision_gates_and_novelty.md). Only then build nonlinear rules, more general DAG enumeration or targeted causal analysis.

## 6. Suggested first configuration (illustrative, not a calibrated recipe)

- graph.family: linear_residual_pair_v1
- graph.dim: 8
- graph.candidates: parallel, composed
- graph.learned_ops: two untied $8\times8$ matrices A, B, evaluated once each
- graph.init: exact_zero_control
- head.type: fixed_common_linear_readout
- head.vocab: 2
- task.family: linear_softmax_teacher_v1
- task.teacher_matrix: matrix with recorded singular/eigen modes
- task.input_covariance: controlled mean-zero Gaussian or discrete finite-mode inputs
- task.temperature: controlled scalar
- task.data_seed / train_size / probe_seed: explicit
- train.loss: softmax_ce
- train.optimizer: sgd (later AdamW replication)
- train.steps / batch_size / learning_rate / weight_decay: explicit, tuned in pilot not asserted here
- analysis: centered logit functions, teacher/unsupervised mode histories, exact cloned update-transfer
- analysis.baselines: duplicate graph, frozen $B=0$, small nonzero init, controlled mode orientations.

A width of eight or two tokens is illustrative; capacity and readout rank must be audited for the chosen teacher. No implementation of a nonlinear operator, generic graph compiler, SAE training or full circuit search is needed to verify the **first empirical scientific question**.


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
- analyze-causality (**later**): directed interventions and small-subgraph search on selected checkpoint+probe distribution.
- report: markdown/plots/machine-readable summary, with source run references.

These names are illustrative; a simple script is acceptable at Phase 0.

## 9. What not to build yet

- Full LLM training harness, CUDA custom kernels, automated hyperparameter tuning or cloud orchestrator.
- A large architecture search controller or Bayesian predictor.
- An SAE training farm as the default circuit extractor.
- Model-side learned gating, recurrence, grouped token heads or multiple training losses.
- A polished visualization app before metric reliability and causality are established.

The agent should deliver running, tested scientific primitives rather than scaffolding every future phase. **A robust linear-function/transfer experiment with explicit negative regimes beats a large unvalidated graph-interpreter framework.**

## 10. Repository governance

- Keep docs/01 intellectual history immutable in spirit: add dated amendments rather than erasing why an idea was abandoned.
- Make controlled scientific changes through short proposals with falsification conditions.
- Every result references immutable graph hash, dataset checksum and code commit.
- New metrics require calibration fixtures, sensitivity tests and documented failure conditions.
- Separate 'foundational math fixture passed' from 'training experiment run' and 'scientific hypothesis supported.'
