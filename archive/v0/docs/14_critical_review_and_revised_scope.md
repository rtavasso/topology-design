# Critical review and revised scientific scope (19 October 2026)

**Status: active methodological correction to the original research plan.** This is a constructive critique, not a withdrawal of the north-star question. Read [the intellectual history](01_intellectual_history.md) for the earlier reasoning; this document specifies the objections that now control experiment priority.

## Revised question

> **How does fixed parameter-free computational connectivity change the acquisition, reuse, interaction, and interference of learned functions, and under what capacity, data, and optimization regimes do these effects persist?**

The *motivation* remains: design routes capable of preserving useful information, replacing inadequate representations, and building new computations without cutting off useful unresolved error. The *scientific question* must not presuppose that long routes learn reusable structure, short routes memorize, or any single arrangement solves all three objectives.

**CE is the required training objective, not the scientific outcome.** The primary outcomes are time-resolved learned functions, consequences of example-specific updates, and eventually the causal implementation and formation of those computations. Do not equate benchmark improvements with a mechanism.

## Seven objections and how they change the plan

### 1. An overly broad north star is easy to rationalize after the result

"Preserve, destroy, construct" can explain almost any topology outcome after the fact. Turn it into **pre-registered, falsifiable, narrower hypotheses** with specific pairs of graphs, fixed operator templates, data distributions, predicted function-level effects, alternative explanations and stopping decisions.

**Required change:** classify each proposed result as representational, optimization, or mechanism-formation evidence. A failure to distinguish these categories invalidates the causal interpretation. Do not call an apparent advantage a 'new topology learning principle' if it arises solely from higher expressive capacity.

### 2. Varying edges can also change expressive capacity and nonlinear depth

For learned nonlinear operators $F_1,F_2$:
$$f_{\rm parallel}(x)=x+F_1(x)+F_2(x),$$
$$f_{\rm composition}(x)=x+F_1(x)+F_2(x+F_1(x)).$$
They have equal learned operator count and parameter budget but **different function classes at finite capacity**, potential nonlinear compositional depth, input distributions, gradient interactions and critical path. If the sequential network learns a compositional task that the parallel one cannot represent, the result is an **expressivity separation**, not necessarily a special learning-dynamics finding.

**Required change:** first use an analytically tractable graph pair with the same overall linear function class, same initial function and same first-order parameter-to-logit Jacobian. Then in nonlinear studies either demonstrate the target is realizable by both or report expressivity explicitly as a factor. Since checking equality of nonlinear function classes is generally infeasible, use constructive witnesses/teacher fits, positive controls, approximation curves and careful qualification rather than claiming full equivalence.

### 3. Relevant work already studies feature competition and pathway specialization

The project's novelty cannot be "pathways compete" or "deep networks form features." Priority antecedents include:
- Saxe, McClelland & Ganguli (2014): analytically solvable learning dynamics in deep linear networks.
- Shi, Shea-Brown & Buice (NeurIPS 2022): specialization between multiple learning pathways under a common objective.
- Qiu, Kuang & Goel (ICML 2024): competition and phase effects between core and spurious Boolean features.
- Xu & Ziyin (ICLR 2025): distinct finite-width feature learning regimes in linear models.
- Chen, Luo & Pan (ICML 2026): training-data attribution and intervention on emergence of interpretable units.

See [literature map](11_literature.md). **Replication/extension is valuable, but label it as such.** Novelty candidate: a controlled causal explanation of how *specific fixed graph rewires* affect emergence and utility of acquired functional modes across scales and feature loads.

### 4. Causal-depth scores are not canonical or always reliable

Minimum retained-subgraph depth depends on:
- where we define a learned operator boundary (a 2-layer MLP could be one or two ops);
- allowable subgraphs and intervention replacement;
- redundant pathways and compensation;
- off-distribution activations introduced by lesions;
- subgraph-search budget, correlated concepts and probe distribution.

It can also misdiagnose a route that was *important during acquisition* but becomes dispensable *during expression*. Circuit faithfulness studies warn of drastic sensitivity to ablation methodology.

**Required change:** demote causal-depth $D_k,U_k$ to exploratory/secondary metrics, never a unique circuit ontology or primary evidence of "long-path learning." Primary is a detected functional change, followed by causal investigation of how the trained graph implements it, with multiple baselines. Training-time gradient/edge interventions are a separate question from inference-time lesion.

### 5. "Memorization circuit" and "generalization circuit" may not be distinct objects

A reusable algorithm can operate on arbitrarily memorized keys. Exception storage and a shared transform may overlap deeply. Superposition can encode instances or statistically reusable features in the same subspace. Behavioral memorization and generalization are distributional properties, not guaranteed separable neural modules.

**Required change:** study **information reuse and update locality**—how many behaviors rely on a functional component and which behaviors are disturbed by its update. Treat path allocation as a measured outcome; do not label a short subnet 'memory' or a deep subnet 'abstraction' by construction.

### 6. We were proposing two research projects before validating either

An automated graph-wide mechanism discoverer and a topology-learning theory are each difficult in their own right. Building a full DAG/NAS/mech-interp framework before observing a meaningful graph-dependent learning phenomenon is a major failure mode.

**Required change:** staged instrumentation:
1. Exact algebra + functional trajectories.
2. One-example update-transfer matrix and generalization tests.
3. Focused causal validation *only for observed novel functional changes*.
4. Broad source-agnostic discovery once pilot effects and instrument reliability justify it.
5. Scale tests and only later architecture search.

Reliable partial measurement is enough to advance from (1) to (2). Unrestricted automated circuit discovery is **not a prerequisite** for the first topology experiment.

### 7. Fixed topology may not dominate LLM learning efficiency

LLMs already contain learned, input-dependent information routing (attention), data-dependent representation formation and in-context memory. The human-versus-LLM comparison also involves prior knowledge, active sensing, grounding, objectives and memory systems.

**Required change:** treat parameter-free static routing as a clean causal variable for early science, **not** an article of faith that it will be the engineering solution. Human sample efficiency motivates study but cannot serve as evidence for a topology hypothesis. Later expand constraints if results establish limits of static wiring.

## Three claims that must be separately labeled in every experiment

| Claim type | Scientific question | Legitimate evidence |
|---|---|---|
| **Representational effect** | What functions can the graph express efficiently at fixed width/params? | mathematical function-class analysis, constructive witnesses, approximation bounds |
| **Optimization effect** | Given comparable representable targets, how does the parameterization change the trajectory? | matched initial state/Jacobians where possible, identical CE/data/optimizer, gradient/transfer measurements |
| **Mechanism-formation effect** | Does the topology cause different reusable, causally validated computations to arise? | new functional mode in trajectory, held-out utility, causal acquisition/expression studies, robustness |

Effects can coexist. A claim about the third must address the first two as confounds.

## Updated measurement priority

**Primary first:**
- time series of the *entire learned function* on a finite or explicitly sampled probe domain, using centered logits and probabilities;
- change in functional modes (not automatically change in circuits);
- actual $T_t(i,j)$ effect of one example's update on many other examples, including related, unrelated and adversarial cases;
- repeatability of those effects across paired initialization and task seeds.

**Intermediate explanatory:**
- first-order tangent geometry and higher-order parameter interactions;
- task-distribution eigendirections/singular modes and their acquisition times;
- generalization and instance-retention partitions where relevant.

**Later mechanistic:**
- inference-time necessity/sufficiency/redundancy, subgraph fidelity, training-time acquisition interventions, basis-aware representation analysis;
- $D_k,U_k$ as explicitly intervention-dependent **secondary descriptors**, not primary metrics.

## Stronger first contrast: equivalent-linear-function models with different higher-order interaction

See [Matched linear topology study](15_matched_linear_topology_study.md).

$$z_A=(I+A+B)x,\qquad z_B=(I+B)(I+A)x=(I+A+B+BA)x.$$

At $A=B=0$, they have the same function and parameter Jacobian, hence same initial tangent kernel under the same fixed readout. Both can express **every linear map** of the input. Nonetheless the $BAx$ term changes second-order parameter derivatives and later CE learning dynamics. This isolates a **higher-order parameterization/routing effect**, not expressive superiority at the function-class level.

Important: we deliberately do not claim they have equal full Hessians, parameter-distance geometries, implicit regularization, or learning trajectories after the first step. Those differences are what we're investigating. And at linear scale we measure **mode learning and specialization**, not semantic nonlinear circuit formation.

## Recommended research sequence

| Study | Core question | Must precede |
|---|---|---|
| 0. Algebra and exact controls | Are we truly changing only fixed routing, and do analytic derivatives match? | all training claims |
| 1. Matched linear-graph CE dynamics | Does a higher-order parameter interaction create predictable differences beyond initial NTK? | nonlinear claims |
| 2. Nonlinear function acquisition | Does a routing change alter emergence of functional interactions *when both can fit the rule*? | broad interpretability |
| 3. Targeted causal mechanism validation | Which graph operations acquire and express a discovered functional change? | mechanism claims |
| 4. Width, task demand, sparsity and data regimes | Is an established effect stable/predictable beyond one toy case? | search/general design claims |
| 5. Search and attention extension | Can a validated relationship guide architecture choices beyond the discovery family? | claims about generalizable topology design |

Detailed falsification and stop criteria: [Decision gates](17_decision_gates_and_novelty.md).

## An explicit possibility: null results are scientifically productive

If graph differences vanish under an appropriate initialization control, are entirely explained by non-matched expressivity, or fail to transfer beyond one synthetic family, **document that limitation and stop expanding the associated instrumentation**. The goal is a grounded design principle *or a clear boundary showing when such a principle cannot hold*.

## Source notes

These objections are informed by [Shi et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html), [Qiu et al. 2024](https://proceedings.mlr.press/v235/qiu24e.html), [Saxe et al. 2014](https://arxiv.org/abs/1312.6120), [Chen et al. 2026](https://proceedings.mlr.press/v306/chen26cv.html), [Miller et al. 2024](https://arxiv.org/abs/2407.08734), [Anani et al. 2026](https://proceedings.mlr.press/v306/anani26a.html), [Hadad et al. 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html), and [Dinh et al. 2017](https://arxiv.org/abs/1703.04933). These papers do not establish our proposed graph-pair effect; they inform the standards and prior-art boundary.
