# First discriminating experiments — revised active sequence (19 October 2026)

**This document replaces the initial priority order, not its scientific motivations.** Previously proposed shortcut/frustration, arbitrary exception, causal-depth, and scaling experiments remain below at later stages. The initial priority is now a topology effect that can be separated from **expressive function class and initial first-order learning geometry**.

See [critical review](14_critical_review_and_revised_scope.md), [matched pair](15_matched_linear_topology_study.md), [measurement](16_functional_learning_and_transfer_measurement.md), and [decision gates](17_decision_gates_and_novelty.md). **No data or results exist yet.**

## S0 — Exact mathematics and measurement oracles (before training)

**Question:** Can we produce graph rewires whose effects on numerical forward computation, gradients and Hessians are exactly predicted?

Fixtures:
1. Identity-origin of scalar residual vs nonresidual; ReLU half-line and finite expressivity; bottleneck rank and spectral bound.
2. Scalar polynomial DAGs $(1+bc+abc)x$ vs $(1+c+abc)x$, at exact zero and small nonzero starts.
3. **New flagship pair**: $h_P=x+Ax+Bx$ and $h_C=x+Ax+B(x+Ax)$.
4. Softmax CE gradient and Hessian decomposition, and first SGD update equality for the flagship pair at $A=B=0$.
5. Centered-logit gauge invariance and finite-step $T(i,j)$ compared with $\eta g_i^\top g_j$.
6. A duplicate-graph null and an intervention that sets $B=0$, yielding identical predictions.

**Expected exact result:** the matched pair has same function, parameter Jacobian, initial empirical tangent kernel and initial SGD update, but different mixed second derivatives; after the same first update their logits differ by $R(\Delta B)(\Delta A)x$. These are mathematical *unit tests*, not evidence for beneficial learning.

**Gate:** all numerical implementations match the derivations before proceeding.

## S1 — Matched linear topology under softmax CE (first empirical experiment)

**Question:** At fixed linear input-output function class and identical initial first-order geometry, does a graph rewire change how learned input-output modes develop?

**Architecture:** two matrices $A,B\in\mathbb R^{d\times d}$, a fixed common readout $R$ and the parallel/compositional expressions in [matched study](15_matched_linear_topology_study.md). Each learned matrix applies once; no extra gate, recurrence, normalization trick or branch-specific loss.

**Data:** controlled linear conditional token teacher $p_*(y|x)=\mathrm{softmax}(RM_*x/\tau)$; vary teacher mode spectra, feature covariance, class balance and finite-sample budget. Use two-logit softmax for the smallest demonstration and later $V>2$ with explicitly controlled readout rank.

**Measurements (primary):**
- the learned conditional probability function and centered-logit map on fixed probes, not just train CE;
- acquisition trajectories of projected teacher modes and unsupervised output modes;
- actual cloned one-example update transfer $T_t(i,j)$ with related/unrelated probes, benchmarked against first-order estimate;
- per-example interference/retention across a continued training stream;
- difference in full CE Hessian mixed terms and higher-order $BA$ contribution as a mechanistic *parameterization* explanation.

**Controls:** exact-zero init; small matched nonzero init; LR sweep; batch/seed pairing; fixed param/head/addition/learned-op count; output norm and activation scales; optimizer (SGD then AdamW); target mode orientation; identical-graph null; explicit case where composed routing is *not* better.

**Interpretation:** a repeatable effect is an **optimization / learned-linear-mode** finding. It is *not* yet a nonlinear circuit-generalization discovery. Failure to find robust differences after controls is valid and should trigger [Gate 1](17_decision_gates_and_novelty.md).

## S1R — Replication of multiple-pathway learning results

**Question:** Do our function and transfer instruments recover the pathway specialization phenomena already reported for deep linear networks under a shared objective?

Use a compatible miniature version of the setting in [Shi, Shea-Brown & Buice (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html); state deviations from their assumptions and loss (our primary training remains softmax CE; exact reproduction under the paper's original objective is a **separate diagnostic**).

**Purpose:** literature-grounded calibration and novelty boundary, not new results. Saxe (2014) provides an additional analytic comparator. Do not use this as an excuse to expand the project before S1 measures are reliable.

## S2 — Nonlinear feature acquisition with expressive-competence controls

**Question:** When two DAGs can demonstrably represent the same specified target task, does a fixed rewire change which input-output functions they *learn* and how updates generalize?

**Architecture:** parent-selected family $h_i=h_{i-1}+F_i(h_{k_i})$ with identical, single-evaluation 2-layer ReLU blocks; start with sequential, raw-input and **one-edge** variants. Note that these have different function classes in general. For each chosen task, establish target realizability for *both* topologies (constructive weights, favorable full-data fit, or strong controlled approximation evidence). If this fails, classify the result as expressivity, not learning dynamics.

**Data:** small complete Boolean domain with multiple latent rules of various interaction orders. Train on a subset; hold out structured combinations; analyze complete centered-logit Walsh spectra and cross-example update transfer. A discovered coefficient is a **functional interaction**, not automatically a graph route or new circuit.

**Measurements:** emergence/persistence of unexpected functional components, loss and transfer on held-out rule cases, matched-CE comparison, gradients and learned modes, optional representation spectra. A circuit investigator should not be handed the generator's latent rule labels until after discovery auditing.

**Falsifiers:** only one network can represent the target; apparent differences are explained by output temperature/scale; effects disappear across seeds or upon equalizing effective approximation competence.

## S3 — Productive unresolved error, shortcuts and arbitrary exceptions

**Question:** Does routing systematically change whether useful unresolved CE error produces reusable functions, instance-specific storage, or harmful interference?

**Toy calibration:** $y=x_1x_2$, $P(x_2=1)=.9$; the short margin $ax_1$ is stationary at $a=\log9$, while a *provided* compositional term $d x_1x_2$ has $\partial_dL=-.18$. This only demonstrates error availability. Do not feed the term directly in the main learning experiment: have the nonlinear modules construct it.

**Primary task:** program-generated token rule, correlated shortcut, persistent random-label exceptions on some training keys, with explicit seen exceptions, unseen rule combinations and shifted shortcut distributions.

**Measurements:** function atlas and transfer for (i) original train examples, (ii) unseen rule contexts, (iii) shortcut-contradicting contexts, and (iv) other stored exceptions. Separate **acquisition**, **expression**, **retention** and **transfer**. No pre-assigned memorization layer or group-specific CE.

**Pushback to test:** a memorized fact can be processed by a general algorithm; memorization/generalization are not guaranteed to have disjoint circuits. Measure dependence and update locality rather than count 'memory neurons'.

## S4 — Targeted mechanism tests for *observed* functional changes

**Prerequisite:** S2 or S3 yields a reproducible novel function/transfer pattern that matters on held-out probes. Do **not** build full automatic circuit discovery to make S1 possible.

Build planted calibration cases first:
- a known short route and a known nonlinear composition;
- two redundant routes (single-edge lesion small, joint lesion large);
- large-activation but causally inert node;
- representation basis changes at exactly valid linear interfaces;
- known superposed sparse features and an unknown-to-detector functional interaction.

For selected observed changes:
- perform inference-time edge/node ablations with multiple replacement baselines;
- test necessity, sufficiency and redundancy;
- match mechanism fingerprints through checkpoints;
- try controlled training-time gradient stop/edge rewires with cloned optimizer state to distinguish **acquisition** from expression.

**Important:** retained-subgraph causal-depth $D_k,U_k$ remains a *secondary operational profile*, not the primary discovered object. A positive causal finding must survive reasonable intervention changes; report ambiguity and unexplained behavior.

## S5 — Capacity, feature-demand, sparsity and task-complexity regime map

**Question:** Is a graph effect intrinsic to fixed routing or conditional on representational crowding / data support?

Independently vary:
- width with task latent features fixed;
- latent feature demand with width fixed;
- both together at approximately constant designed feature load;
- feature sparsity/co-occurrence and importance;
- nonlinear composition order at fixed feature count;
- distinct-example count, shortcut contradiction frequency and training duration.

Record function/transfer metrics as primary and causal measurements for selected regimes. Do not equate width with number of atomic features or declare an SAE count canonical.

**Success:** a reproducible boundary/phase map showing where graph effects occur, disappear or reverse. Two widths do not constitute a scaling law.

## S6 — Held-out program and small-attention transfer

Only after a stable topology-to-learning-mechanism relation emerges, test it on a new data-generating family: e.g. modular addition, symbolic grammars, small transformers/attention. This addresses whether the result is a broad design principle or a special property of one operator/task.

Keep attention's **learned content-dependent routing** conceptually separate from the fixed edges being studied.

## Study reporting (mandatory)

1. Precise question, candidate graph edge edit, comparator and classification as expressivity/optimization/mechanism.
2. Algebraic proof or approximation competence control and topology/compute audit.
3. Pre-registered predictions, falsifiers, raw initialization/scale and optimizer settings.
4. Function trajectories and individual-example update transfer, including counterexamples and negative values.
5. Generalization to new inputs **on clearly specified distributions**, distinct from arbitrary-exception recall.
6. If mechanism claims: intervention fidelity, redundancy, alternate baseline and training-vs-inference distinction.
7. Paired seeds, task seeds, confidence bounds, compute and checkpoint history.
8. Whether [decision gate](17_decision_gates_and_novelty.md) passed, failed or was inconclusive, with next discriminating action.

## Relationship to the original sequence

Original E0 scalar polynomial fixtures remain as S0. Original E1 raw-vs-sequential residual study is now **S2**, after S1 controls expressive class. Original E2 frustration and E3 rule+exception studies are now **S3**. Original E4 causal calibration is now **S4**, not a precondition for measuring S1. Original E5/E6 become **S5/S6**. Those pivots are deliberate and recorded in [history](01_intellectual_history.md).

**Do not optimize an architecture search objective until these studies establish which mechanistic measurements are reliable and predictive.**
