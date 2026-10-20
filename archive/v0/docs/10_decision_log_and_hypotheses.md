# Decision record, rejected approaches, live hypotheses, and unresolved tensions

**19 October 2026 note:** A constructive adversarial review changed the active *sequence*, not the core motivation. [Critical review](14_critical_review_and_revised_scope.md), [matched linear study](15_matched_linear_topology_study.md), and [decision gates](17_decision_gates_and_novelty.md) supersede older suggestions to begin with generalized circuit discovery or broad topology enumeration. Retain previous decisions as history.

This record complements the chronological intellectual history. It is written so a new agent can recover **why** each choice was made, not merely follow a frozen checklist.

## Non-negotiable orienting statements

1. **The research target is computational mechanisms and learning dynamics, not CE minimization.** CE is the stipulated training loss.
2. **Longer computational paths are not presumed to generalize; shorter paths are not presumed to memorize.** Discover the actual allocation.
3. **Preservation / destruction / construction remain the organizing behaviors.** We want topologies where useful changes are accessible without gratuitous disruption, not merely lots of skip edges.
4. **Productive friction is allowed.** Isolating branches to remove all competition may destroy useful cooperative learning. We want unresolved learning pressure to remain accessible to computations capable of resolving it.
5. **No presumed circuit templates as primary discovery.** Avoid only measuring an expected parity/Fourier/two-layer circuit; build a detector that can be surprised.
6. **Scaling has hidden covariates.** Feature count, activity sparsity, correlations, data support, effective capacity and measurement resolution all change alongside width.
7. **The first scientific output can be a validated negative result or an accurate instrument**, not an 'improved architecture'.

## Research decisions

### D01 — Exclude recurrent feedback from primary topology comparison
- **Reason:** extra inference iterations, changing computational budget and temporal credit assignment would confound fixed routing advantages.
- **Retain:** top-down routing in an acyclic multi-stage graph and iterative feedback as future comparative research.
- **Revisit when:** feed-forward mechanisms are measured and we can properly compute-match recurrence.

### D02 — Do not add trainable gates
- **Reason:** learnable gate weights change expressive operator set and parameter budget; our first question is numerical effects from routing alone.
- **Retain:** gate networks as separately labeled nonprimary control for conditional overwriting.
- **Caveat:** parameter-free fixed masks/additions can also restrict function class; not automatically fair without audits.

### D03 — Avoid a priori token partitions / separately supervised heads
- **Reason:** factorized CE can remove learning-signal competition but allocates predictive authority by hand and prevents flexible revision of earlier decisions.
- **Retain:** exact conditional-softmax example in dynamics docs for understanding loss coupling; external auxiliary supervision as diagnostic control.
- **Revisit when:** after evidence of an emergent, data-driven distinction that a *fixed prior* division of labor preserves useful function.

### D04 — Productive frustration instead of maximizing gradient magnitude
- **Reason:** a large gradient can update wrong/shared features; low aggregate gradient can coexist with slow useful feature formation. Disagreement is neither necessary nor sufficient for useful learning.
- **Working form:** residual CE signal beyond shallow first-order direction, accessible to deeper directions.
- **Uncertainty:** the short/long decomposition is not unique in shared DAGs, and accessibility may not predict true circuit development.

### D05 — Source-agnostic discovery before circuit probes
- **Reason:** probing known circuits blinds researchers to unexpected algorithms, distributed representations and novel mechanisms.
- **Approach:** functional atlas → unanticipated behavior contrasts → causal mapping → longitudinal tracing → audit against generator truth.
- **Caveat:** there is no fully assumption-free discovery; input and measurement basis, probe distribution and interventions create selection bias.

### D06 — Do not equate neuron/SAE interpretability with mechanism
- **Reason:** features can be distributed, superposed, split/absorbed or representation-basis dependent. Width changes representational demand and resolution.
- **Approach:** prioritize basis-robust functional/causal measures; use SAEs as optional instruments only after calibration.

### D07 — Delay architecture search
- **Reason:** random/Bayesian/NAS search could reward rapid CE descent, short-path convenience or instrumentation bias rather than actual circuit formation.
- **Approach:** start with exact graph grammar and one-edge mutations; characterize mechanisms and confounds. Search only after a validated mechanistic objective exists.

### D08 — No straight shot to LLM scale
- **Reason:** tiny controlled graphs support exhaustive domains, exact path expansion, true mechanism interventions and phase-map experiments; large LLMs make key quantities unidentifiable/expensive.
- **Destination:** small attention models then language modeling once causal topology principles replicate.

### D09 — Preserve negative and ambiguous outcomes
- **Reason:** ambiguous circuit identity, no causal significance, or scale reversals identify the boundary of the proposed science. Re-running until a positive figure appears defeats our goal.

### D10 — Separate expressivity, optimization and mechanism formation (19 Oct 2026)
- **Critique:** changing fixed connections can increase nonlinear composition and hence finite-width expressive capacity; one graph's better generalization need not imply different gradient dynamics.
- **Decision:** first compare graphs with the *same representable linear map class* and exactly matched initial function and parameter-to-logit Jacobian. For nonlinear tasks establish selected-target competence in both graphs or explicitly call the difference representational.
- **Falsification:** topology advantages disappear once function-class and scale differences are controlled.
- **Retain:** raw-input versus sequential nonlinear graphs as a later *expressivity-plus-learning* study, not the initial causal test.

### D11 — Initial network pair: same NTK, different second-order parameter interaction
- **Decision:** parallel $x+Ax+Bx$ versus composed $x+Ax+B(x+Ax)$, with common fixed readout and ordinary CE.
- **Why:** same trainable matrices, learned operation and fixed addition counts, function class, initialization output and Jacobian at $A=B=0$; difference $BAx$ changes higher-order derivatives and subsequent updates.
- **Important limit:** this is not equal full CE curvature, identical implicit bias or a measure of nonlinear circuit formation.
- **Next:** analytic fixtures → time-resolved linear mode learning and per-example transfer → nonlinear tests.
- **Details:** [matched study](15_matched_linear_topology_study.md).

### D12 — Primary metric: learned function and actual single-example transfer
- **Critique:** path-depth KL/utility cannot unambiguously measure circuit development; intervention baselines and operator granularity make it noncanonical, while acquisition and expression can use different routes.
- **Decision:** first observe the full centered-logit function and its unexpected temporal changes, then exact cloned one-example update effects $T(i,j)$, with related/contradictory/unrelated probes. Causal-depth and feature-level analyses are secondary, later.
- **Control:** perform actual finite-step effects; dot-products are only small-SGD-step approximations.

### D13 — Targeted causal interpretability after a demonstrated function-level effect
- **Critique:** building a source-agnostic mechanistic framework is itself an ambitious research program that could consume the project before any topology effect is known.
- **Decision:** validate partial measurements early; once function/transfer differences replicate, investigate *that behavior* using baselined patching, redundancy checks, and separate training-time interventions. Don't make solving all mech interp a prerequisite.

### D14 — Treat reuse / interference as more foundational than 'memory circuits'
- **Critique:** arbitrary facts and general rules can share computations, and features/instances can be represented in superposition.
- **Decision:** label memorization and generalization via specified distributions; measure shared functional dependencies and update locality. Never declare short=memorization or deep=generalization based on wiring alone.

### D15 — Explicit gates and stopping conditions
- **Critique:** the broad motivational question could explain any result after the fact.
- **Decision:** each hypothesis requires a null/negative control, plausible alternative explanation, reproducible functional signal, and continue/pivot/stop criterion. Search/large-scale work only after predictive out-of-sample topology–mechanism links.
- **Details:** [decision gates](17_decision_gates_and_novelty.md), [experiment sequence](13_first_experiment_scenarios.md).


## Live hypotheses, explicitly unproven

**H01: Local trainability via short paths.** Parameter-free routing alters low-order parameter dependence and Jacobian conditioning, potentially enabling faster updates without more learnable capacity. Prediction: source-parent mutations change first-order functional accessibility near matched initialization. Exact linear examples support possibility; nonlinear generalization untested.

**H02: Early fitting can starve compositional mechanisms.** A short predictive route can saturate the shared softmax CE gradient before more reusable long computations develop. Prediction: shortcut-favorable routing produces earlier train CE collapse but delayed or suppressed causal appearance of rule circuits in some regimes. Counter: residual routing may also facilitate deep learning; the effect can reverse.

**H03: Productive residual error catalyzes circuits.** If shallow update directions saturate yet deep operations have accessible novel directions, the network can develop a new reusable mechanism. Prediction: regularized residual-access score plus observed gradient trajectory predicts causal circuit formation beyond raw gradient magnitude. Counter: access can be memorization or spurious parameter reachability.

**H04: Topology affects representation packing and interference.** Different routing changes which inputs/features share parameters and compete in hidden states. Prediction: at matched width and task load, topology changes causal-overlap/interference and the stability of instance-specific versus shared behaviors. Counter: differences vanish with basis-robust analysis or are entirely explained by feature demand and initialization.

**H05: Acquisition paths differ from expression paths.** A route may be important during learning and redundant afterward, or essential to express a computation it did not create. Prediction: training-time edge-gradient intervention and inference-time lesion can rank different components. Counter: intervention changes optimization too broadly to isolate a cause.

**H06: Route utility depends on scale / task regime.** Width at fixed feature demand versus width with proportional feature growth can show different topology effects. Prediction: at least one claimed topological advantage interacts with measured representational pressure. Counter: no robust interaction after matching data/support and capacity.

**H07: Memorization and generalization need not occupy disjoint subgraphs.** Shared feature extractors can process both arbitrary stored facts and compositional rules. Prediction: discovered mechanisms show shared components, with separation possibly in stored instance-specific states or readout. Counter: some task families do show strong disjointness; characterize when.

**H08: Topological mechanism signatures may transfer across sizes.** Certain graph descriptors might predict causal-depth utility or circuit-emergence times on unseen widths/task families. Prediction: held-out graph and task validation beats null/random. Counter: no stable predictor; graph effect dominated by operator capacity and data geometry.

**H09 (NEW): second-order routing may change linear CE mode learning beyond a matched initial tangent kernel.** Parallel and composed $A/B$ networks have identical initial function/parameter Jacobian but different $BAx$ interaction. Prediction: in some mode/data regimes a reproducible difference in full function and one-example transfer emerges at finite updates. A universal positive effect is **not** predicted. Clear boundary: no effect after controls or only trivial initialization/implicit-regularization explanation.

**H10 (NEW): graph-induced nonlinear function acquisition is distinguishable from expressivity if targets are accessible in both graphs.** Prediction: for selected same-task-realizable target, one-edge routing mutation changes when a new functional dependency emerges and how it transfers; if one graph cannot fit, classify as expressive capacity instead.

**H11 (NEW): causal path depth is not a stable independent scientific target without fixed granularity and intervention protocol.** Prediction: $D_k,U_k$ vary under reasonable ablation baselines, while some broader functional transfer effects remain robust. If they do not vary, that's a positive robustness finding—not a requirement.

**H12 (NEW): reuse/interference patterns may predict downstream scaling better than 'memory neuron' counts.** Prediction: topology effects on actual related/unrelated example-update transfer persist or change systematically with feature demand and width, even as neuron-level interpretations shift. Requires independent data/seed/basis controls.


## What about humans learning from few examples?

This was an important motivation, **not a clean one-to-one experimental comparison**. Humans bring rich priors, multimodal grounding, active exploration, different memory systems, evolved inductive biases, and prior learning from many examples. LLMs also perform rapid in-context adaptation that isn't an SGD update. Treat the broad human/LLM sample-efficiency gap as a reason to ask about reusable representations and interference, not as evidence that skip topology is the main limiting factor.

## Misconceptions to actively test/reject

- 'Residuals make the loss globally convex.' False in general; explicit negative-eigenvalue counterexample.
- 'The identity route adds a learned degree of freedom.' Not in $y=(1+w)x$ versus $y=wx$.
- 'ReLU residuals trivially subtract.' Needs signed projection/appropriate path.
- 'A single example that reduces CE must make target top-1.' False.
- 'If train CE approaches zero, no learning can continue.' False at finite logits; small gradients may still support gradual changes and optimizer dynamics.
- 'Disagreement should be maximized to induce useful frustration.' False; conflict can be noise/instability.
- 'Short path means memorization; long path means generalization.' Unsupported.
- 'Fourier interaction order equals graph path depth.' False.
- 'Ablation importance equals unique circuit discovery.' False under redundancy and off-distribution replacements.
- 'Neuron or SAE feature count rises with model width in an interpretable way.' No, representation and instrument scale together.
- 'A black-box performance proxy is a mechanistic topology metric.' Not until validated.

## Open choices to resolve through science, not taste

- Graph family: parent-residual first, then arbitrary DAG rewires; conditions for expanding.
- Operator class: bias-free tiny MLP vs 2-layer ReLU blocks vs normalized blocks; pre/post normalization needs dedicated controlled comparisons.
- Definition of long-path **learning** when learned operators appear on many paths; causal gradient tags vs module ancestry vs intervention necessity.
- Which subgraph masking baselines are least misleading for given architectures and data.
- How to compare causal depth at matched graph complexity and intervention budgets.
- How to detect mechanisms in a source-agnostic way beyond Boolean Fourier domains.
- How to quantify representational pressure without assuming atomic true features.
- Minimum replication needed to claim a topology effect across seeds and task families.
- How to move from MLP DAGs to attention without treating learned attention as merely a skip.
- Whether deliberately modular parameter allocation improves sample efficiency or simply reduces beneficial sharing.

## Future decision amendments

For any material change append: date; premise; evidence; mathematical/conceptual argument; discarded alternatives; expected falsifiers; scope; links to run manifests; agent responsible. Keep rejected options legible rather than deleting them.
