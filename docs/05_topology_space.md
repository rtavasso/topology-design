# Topology grammar, controls, and candidate-generation strategy

Goal: a **scientifically auditable** family of parameter-free, single-pass DAGs that can be exhaustively enumerated at small scale and later searched. The architecture is the object of intervention; changing learned operators, logits head, objective or training length is not an edge-only result.

## A. Narrow primary family (recommended first)

Fix $L$ learned transformations $F_1,\ldots,F_L$ mapping $\mathbb R^d\to\mathbb R^d$. Each $F_i$ contains the *same internal number/type of learned layers and activations* for every topology; no parameter sharing unless explicitly part of the fixed design.

Define:
$$
h_0=x,\qquad
h_i=h_{i-1}+F_i(h_{k_i}),\quad 0\le k_i<i,\quad i=1,\ldots,L.
$$

The discrete topology is the parent-selection vector $(k_1,\ldots,k_L)$. Every operation $F_i$ executes exactly once and there is one accumulation addition per step. At stage $i$ there are $i$ choices, giving $L!$ labeled configurations prior to functional-equivalence and symmetry pruning.

Properties:
- All computations are DAGs; no learned gating or recurrence.
- Every operation has equal output shape, parameter count and output-accumulation status.
- Every candidate has an identity backbone from input to output, fixed count of additions and number of learned operation evaluations.
- Each learned operation receives a preceding accumulated representation $h_{k_i}$, changing whether it processes raw inputs, earlier partial updates, or the latest state.
- There is **not** a path-unique decomposition of the trained nonlinear output: modules can interact nonlinearly through shared states.
- Labels matter for paired seeds: same $F_i$ should begin with the same weights across candidates. Also report results when learned-weight labels are permuted to rule out fortunate placement.

Example $L=4$:
- **Sequential residual:** $(0,1,2,3)$, each module receives latest state.
- **Parallel/raw-input:** $(0,0,0,0)$, every learned module receives original $x$; backbone sums independent feature updates. This is a useful negative control for learned inter-module composition.
- **Mixed:** $(0,1,0,2)$: $F_3$ receives raw $x$, $F_4$ receives $h_2$.

The parent-selected source is **not** itself a separate learned gate; source edge is fixed within a run.

### What this grammar cannot answer

It always adds learned outputs onto the same backbone, so it does not explore all possibilities for replacing, deleting, masking or selectively isolating previously accumulated features. It also forbids arbitrary skip fan-in or a choice of whether a learned op's output is delivered to multiple future locations. Do not oversell it as exhaustive search over all DAGs.

## B. Expanded grammar (after narrow-family validation)

Fix a multiset of learned operator instances and input/output tensor interfaces. Allow parameter-free:
- identity edge rerouting;
- additive merges with fixed coefficients;
- fan-out of activations;
- optionally fixed permutations/projections **only in a separate experimental family**, since they change available basis and possibly rank;
- multiple fixed readouts only if every graph has identical parameter budget and supervised output semantics.

Require:
1. topological ordering and acyclic validation;
2. no unused learned operator without explicit classification as degenerate control;
3. every operator evaluated once;
4. full accounting of additions, parameter-free transforms, critical path length, activation memory and FLOPs;
5. either matched counts of fixed merges or a factorial comparison separating merge count from wiring;
6. same trainable operation shapes, activation type/count, output dimension and loss;
7. graph equivalence/hashing and isomorphism-aware dedup where feasible.

Do not blur 'same trainable parameter count' and 'same computation budget.' Both are independently reported.

## C. A declarative graph intermediate representation

Represent a DAG in a stable, inspectable and hashable data structure:
- ordered **nodes** with unique IDs and kind (input, fixed_add, learned_op, output; later fixed_routing);
- **ports** and ordered typed edges; if addition is commutative, canonicalize its unordered inputs for topology hashes;
- **learned_op template IDs** and initialization/seed mappings;
- **tensor shapes**, device/dtype, output head, and activation placement;
- graph metadata with version and approved experimental family;
- deterministic graph serialization and SHA fingerprint, independent of run-time node object identity.

Run a compile-time validator that rejects cycles, missing required nodes, uninitialized ports, tensor incompatibility, accidentally repeated execution of a learned module, disconnected outputs and hidden parameter changes.

Return both computational graph and a **path inventory** at small scale:
- learned-operation-count histogram of directed input-output paths;
- number of routes through each learned op;
- whether a path bypasses particular nonlinearity or bottleneck;
- reachability and topological min-cut properties;
- source-parent nesting, fanout, shared dependencies;
- operator-wise gradient paths *under a stated linearization*.

Path count is structural; it does not measure numerical path contribution, because weights, nonlinearities and cancellations determine the latter.

## D. First-principles graph descriptors

For linear scalar networks, compute symbolic weight polynomials and their coefficient/degree support exactly. For small vector linear networks, use ordered noncommutative products of matrices. For nonlinear graphs use local Jacobians, Hessian-vector products and activation-conditioned effective graphs.

Candidate descriptors:
- minimum learned-parameter interaction degree of each output direction;
- number of trainable first-order directions near initialization;
- singular value/rank/conditioning of input-output Jacobian;
- singular value/rank/conditioning of parameter-to-logit Jacobian;
- path degree distribution, fan-in, bottleneck bypasses;
- percentage of graph influenced by a learned module, measured through actual sensitivities;
- analytic expressivity witnesses (functions each graph can/cannot represent at fixed capacity).

If a descriptor's value is sensitive to arbitrary parameter scaling/basis, report the convention and calibration.

## E. Null hypotheses and paired graph mutations

Prefer **one-edge changes**:
- change only $k_i$ for a specified $i$;
- preserve all module identifiers and initialization seeds;
- run counterbalanced module-weight permutations;
- for strict added-edge grammar, swap one edge while holding edge count fixed;
- run controls for matched parameter count, output functions at initialization, gradient norm and forward Jacobian where achievable.

Every mutation's pre-registration includes:
1. expected change in path inventory or local gradients;
2. prediction for time-resolved learned mechanisms;
3. predicted influence on memorized exceptions vs rule generalization;
4. potential confounders/falsification;
5. one minimal negative control.

An edge that doesn't change symbolic function or allowed nonlinear interaction is not automatically useless; it may alter optimization geometry. But it needs a corresponding explanation.

## F. Initialization is an explicit experimental factor

At least:
- **raw matched initial weights:** same operator weight tensors and seeds across topology candidates. This isolates edges but not resulting signal scales;
- **calibrated signal statistics:** adjust a specified *fixed* scalar or initialization protocol to match activation norms and perhaps Jacobian spectra, tracking whether parameter/fixed-computation constraints changed;
- **multiple initialization regimes:** small/random/or exactly zero only when mathematically appropriate. Exact zero can create symmetry or inactive product-gradient paths and is a deliberately extreme diagnostic, not necessarily a fair training baseline.

Keep a manifest of exactly what was matched; don't claim all gradient singular values can always be matched without changing other features.

## G. Search strategy: explanation before automation

**Step 1:** enumerate all small narrow-family configurations, compute exact invariants, prune only provably equivalent graphs, select representative contrasts and mutants.

**Step 2:** compare multiple seeds with complete checkpoint histories, collecting interpretable dynamics. Do not early-stop based solely on early CE.

**Step 3:** check whether an analytic descriptor predicts circuit formation/utility, not only final loss, across task families and widths.

**Step 4:** random and local evolutionary search with mechanism-aware objectives and uncertainty. Keep Pareto tradeoffs; audit graph equivalence and training budget.

**Step 5:** consider Bayesian surrogate/model-guided search only when pretraining proxies have shown out-of-sample prediction of the actual scientific measurements. Evaluate surrogate generalization to unseen graphs and task families.

A DARTS-like supernet/continuous edge-weight relaxation is not a clean primary experiment: it introduces learnable architecture weights, potentially changes gradient flow during search, and can be biased toward easily optimized identity operations. It can be a future engineering strategy after mechanisms are established.

## H. Special caution about labeling 'short' and 'long'

In a residual DAG, a single learned operator or parameter may lie on many effective paths of different lengths. You cannot in general assign a unique short-path parameter group. The narrow grammar permits classifying modules by **source ancestry**, but even that can overlap.

Use path-intervention analyses to estimate output contributions/necessity; where decomposition is ambiguous, report bounds or a multi-attribution set rather than inventing a unique assignment.
