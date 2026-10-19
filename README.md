# Topology Design
### A mechanistic science of how information routing shapes learning

**Status:** research documentation and coding-agent handoff, revised **19 October 2026** following a constructive adversarial review. **No experiments have been run in this repository.** The mathematical fixtures are deductions under specified assumptions; the learning and architecture hypotheses are unverified.

## The research question

> **How does parameter-free feed-forward connectivity shape the acquisition, reuse, interference, and causal organization of learned functions—and which effects persist across task and model scale?**

The long-term motivation remains the development of computational topologies that make it easy to **preserve useful information, replace inadequate representations, and construct new reusable computations**, while allowing unresolved errors to train the operations that can resolve them. We are not claiming that longer paths generalize, shorter paths memorize, or residual connections solve learning efficiency.

This is **not a neural architecture search benchmark**. Softmax cross-entropy is the training loss, but the scientific measurements are *how the model's function changes*, *how a single update transfers to other examples*, and, later, *which learned computations causally implement and acquire those behaviors*.

## Critical methodological revision

The original plan emphasized general circuit discovery and a causal path-depth utility score. Our subsequent critique identified three severe confounders:

1. **Topology rewires can change the function class, not just optimization.** A parallel graph may be unable to express a composition that a sequential graph can; different outcomes then need not reveal a new learning mechanism.
2. **Mechanistic circuit extraction is not canonical.** It is sensitive to intervention choices, path granularity, redundancy, feature representation and analysis resolution. An inference-time lesion need not identify a route important during training.
3. **The existing literature already explains substantial pathway competition and feature specialization.** Novelty requires a controlled, predictive link from a particular fixed graph rewire to a learning dynamic and eventually to reusable learned computation.

We now begin with **a matched deep-linear graph pair** having the same representable linear function class, initial input-output function and parameter-to-logit Jacobian (hence initial empirical tangent kernel), but different second-order parameter interactions. Only then move into nonlinear feature acquisition and targeted mechanistic interventions.

[Read the complete critical review](docs/14_critical_review_and_revised_scope.md) and [decision gates](docs/17_decision_gates_and_novelty.md).

## Active first experiment: matched linear routing

Use two matrices $A,B\in\mathbb R^{d\times d}$ with identical learned-parameter budgets and evaluations:

$$
h_P=(I+A+B)x,\qquad
h_C=(I+B)(I+A)x=(I+A+B+BA)x.
$$

Both maps represent every linear transformation of $x$ and (at $A=B=0$) compute $x$ with identical first-order parameter Jacobians. The composition term $BAx$ creates different higher-order parameter dependencies, with no new learned parameters, recurrence or gates. Train both with the **same fixed readout and vocabulary-wide softmax CE**, and compare exact functional trajectories, mode acquisition and one-example transfer.

Do **not** claim that $BA$ is a semantic circuit or that the models have identical CE Hessians. The latter **differ** because of the mixed parameter derivative. Study 1 is a precisely controlled optimization investigation, not proof of nonlinear generalization.

[Full mathematical specification and tests](docs/15_matched_linear_topology_study.md).

## Measurement priorities

**First: observe changes in learned functions.** On an enumerated or fixed probe input set, track centered logits, probabilities, functional modes and unexpected input interactions. Do not assume a circuit template.

**Second: observe what one example teaches.** At checkpoints, clone model/optimizer state and take one update on example $i$. Measure actual changes to other examples $j$, including beneficial transfer, destructive interference and useful reuse. Relate this to gradient geometry but do not replace finite-step observations with first-order approximations.

**Third: explain selected functional changes causally.** Once we have a real, replicable phenomenon, use subgraph intervention, redundancy tests and training-time replay to find which operations express and acquire it. Report ambiguity and intervention sensitivity; path-depth KL/utility is **secondary**.

**Fourth: vary scale and data demand.** Width, latent feature count, feature sparsity/co-occurrence, task complexity, data support and optimizer are independent axes. Only then consider model-guided topology search, attention or LLM-scale applications.

[Functional measurement specification](docs/16_functional_learning_and_transfer_measurement.md) | [Mechanistic instruments and their limits](docs/06_mechanistic_measurement.md).

## Document map: a full handoff for coding agents

**Orientation, decisions and history**
- [AGENTS.md](AGENTS.md) — implementation contract, sequencing, out-of-scope and acceptance.
- [Original questions](docs/00_questions_in_original_words.md) — the motivating intuitions preserved in the researcher's words.
- [Intellectual history](docs/01_intellectual_history.md) — pivots from reverse residuals and gates through CE, frustration and mechanistic discovery; preserved rather than retroactively rewritten.
- [Research charter](docs/02_research_charter.md) — problem, definitions and evidence standards.
- [Critical review / revised scope](docs/14_critical_review_and_revised_scope.md) — objections to the initial program and the active narrowing.
- [Decision log and hypotheses](docs/10_decision_log_and_hypotheses.md) — what was rejected, what remains open and why.
- [Novelty and decision gates](docs/17_decision_gates_and_novelty.md) — continuation, pivot and stop conditions.

**Theory and measurement**
- [First-principles mathematics](docs/03_first_principles_math.md) — exact residual and path-polynomial examples.
- [Gradient and CE dynamics](docs/04_gradient_dynamics.md) — per-example transfer, pathway coupling, productive frustration and the limitations of initial gradients.
- [Matched linear-graph study](docs/15_matched_linear_topology_study.md) — current flagship causal comparison.
- [Functional learning and transfer](docs/16_functional_learning_and_transfer_measurement.md) — primary quantitative observation stack.
- [Mechanistic measurement](docs/06_mechanistic_measurement.md) — eventual circuit discovery, interventions and scale-dependent limitations.

**Execution, controls and scholarship**
- [Topology space](docs/05_topology_space.md) — initial pair, then controlled parent-residual DAG family and generalization.
- [Data, memorization/generalization and scale](docs/07_datasets_and_scale.md).
- [Experimental protocol](docs/08_experimental_protocol.md).
- [First experiment scenarios](docs/13_first_experiment_scenarios.md) — active sequence and historical experiments moved to later studies.
- [Software blueprint](docs/09_software_blueprint.md).
- [Validation tests](docs/12_validation_tests.md).
- [Annotated literature](docs/11_literature.md) — includes prior-art boundaries and methods for tracing training-time mechanism origins.

## Active study order

| Study | Question | Outcome |
|---|---|---|
| 0 — exact controls | Do the graphs match what the mathematics requires? | symbolic, forward, Jacobian, Hessian and CE-update fixtures |
| 1 — matched linear pair | Does fixed routing change learned modes beyond identical initial tangent dynamics? | function/transfer trajectories, confidence bounds |
| 2 — nonlinear acquisition | Does topology alter emergence of new functions when both can learn the target? | unanticipated functional changes, held-out transfer |
| 3 — causal mechanisms | Which operations acquired and express the discovered changes? | necessity/sufficiency, redundancy, intervention fidelity |
| 4 — scale and task regimes | When does the effect persist or disappear? | boundary/phase map |
| 5 — design principles/search | Do validated graph properties predict unseen architecture outcomes? | held-out topology/task and attention replication |

Passing one stage is *not* permission to claim the next; [gate criteria](docs/17_decision_gates_and_novelty.md) govern escalation.

## Non-negotiable scientific habits

- Same trained operators, parameter count, activation functions and softmax CE for the primary graph comparison; no added learned gate, auxiliary branch objective, token partition or recurrent inference.
- Record fixed arithmetic, critical path, activation memory, initialization scaling and actual compute—not only parameter count.
- Label findings as **mathematical**, **expressivity**, **optimization**, **functional transfer** or **causal mechanism**.
- Preserve conflicting results, unidentifiability and failures to generalize; never optimize early CE as a proxy for meaningful circuit formation without validation.
- Do not infer that a high-order input interaction requires a long path, or that long paths serve generalization.
- Validate the mechanism-discovery instrument against planted and held-out examples before making circuit-level claims.

**Repository state:** documentation only. No code, model weights, datasets, instrument calibration results, or architecture improvements have been produced here yet.
