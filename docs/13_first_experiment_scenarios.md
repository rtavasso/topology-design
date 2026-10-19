# First discriminating experiments: precise, minimal scenarios

This document is an execution order for a coding agent *after* Phase 0 mathematical and graph tests. These are examples of falsifying experiments, not empirically confirmed benefits. They deliberately start with small networks where behavior can be measured exactly.

## E0 — Scalar topology algebra and gradient access

**Question:** Can an edge-only rewire change the order at which learned parameters influence predictions without changing the attainable scalar linear functions?

Construct graphs A/B from docs/03:
- A: $u=x+ax;\ v=bu;\ w=cv;\ y=x+w=(1+bc+abc)x$.
- B: $u=ax;\ v=bu;\ w=x+v;\ y=x+cw=(1+c+abc)x$.

**Analytic predictions:** at $a=b=c=0$, output identity for both; first-order parameter Jacobian vanishes in A and contains $x$ through $c$ in B. B has $O(|\epsilon|)$ local parameter displacement for output correction while A's simple two-weight route has $O(\sqrt{|\epsilon|})$.

**Tests:** symbolic expansion, finite differences, autograd; train both with a two-logit CE head on a deterministic binary token function (include exact zero and small random starts as separate conditions). Do **not** assert CE training convergence is mathematically guaranteed or generalizable from this scalar example.

**Outputs:** weights/gradient norms over time, function slopes, optimizer dependence, latency and parameter matching. **Falsification of broad conjecture:** if nonlinear practical effects disappear under calibrated random init, local zero behavior is not a strong general topology predictor.

## E1 — Strict narrow-family topology vs cheap vs compositional solutions

**Question:** Does choosing earlier versus later inputs to the same learned operations change whether compositional input interactions emerge?

Use $h_i=h_{i-1}+F_i(h_{k_i})$, $L=4$ with identical $F_i$ (e.g. tiny 2-layer ReLU width $d$), and at least:
- sequential $(0,1,2,3)$;
- all raw-input $(0,0,0,0)$;
- mixed $(0,1,0,2)$;
- one-edge neighbor of sequential.

**Data:** balanced small Boolean $x$, rules sampled from an undisclosed mixture of low- and higher-order Boolean interactions. Use two-logit CE. Enumerate full input domain for output atlas. If an operator $F_i$ can independently realize the full task at chosen width, acknowledge that as alternative explanation; do not equate all-raw topology with 'cannot solve XOR' without checking module capacity.

**Observation:** tracked Fourier interaction components (all subsets), representation spectra, training CE, graph perturbation effects, source-agnostic behavior cluster, causal-depth profile under several baselines.

**Controls:** equal learned operations, exact params, module seeds and updates, matched fixed merges, calibrated signal, training steps, graph label permutations. Blind the mechanism detector to generator interaction order.

**Success:** reproducible difference in **emerging functional interactions and causally implicated paths** linked to one edge, not merely mean loss. **Failure:** one graph performs better but the interpreter cannot identify a mechanism → improve instrumentation before expanding search.

## E2 — Productive unresolved error and circuit acquisition

**Question:** When an easy shortcut reaches its CE optimum, can routing determine whether a new reusable rule emerges, and whether remaining errors can train it?

Start with the distribution in docs/04: $y=x_1x_2$ and $P(x_2=1)=.9$. Short margin $ax_1$ has optimum $a=\log9$, while a supplied term $d x_1x_2$ has negative loss gradient at $d=0$. Confirm mathematically (fixture only).

Then replace the *supplied interaction* with a nonlinear trainable DAG that must learn an equivalent compositional rule from $(x_1,x_2)$, without exposing $x_1x_2$ as an input.

**Measurements:** time when shortcut reaches plateau; residual gradient alignment with shallow/deep directions; actual development of previously absent input interaction; later ability to predict contradictory $x_2=-1$ cases and balanced OOD cases; causal route utility. Contrast routing parent vectors and independent random seeds.

**Controls:** ensure training distribution has enough contradictory examples to identify the true rule; vary contradiction rate (0.1, 0.01, etc. as *sweep examples*, not calibrated defaults), initialization and optimizer. An absent rule with zero contradictory evidence is not circuit failure but informational nonidentifiability.

**What this can rule out:** 'strong shallow CE feature necessarily kills long learning' is false if rule circuits reliably develop. Conversely long gradients can exist without helpful feature creation; separate access from acquisition.

## E3 — Rule+exception memorization and distinct evaluation populations

**Question:** Does a topology create different dependencies for arbitrary memorization and reusable rule behavior?

Generator from docs/07: latent symbolic rule over small finite domain, persistent random labels on selected training keys, plus optional shortcut that changes correlation at test. **No manually allocated memory path**, no exception-aware training loss.

At every checkpoint:
- measure exact recall of seen exception keys (retention);
- rule correctness on unseen combinations;
- shortcut-broken CE;
- actual one-example update influence on old exceptions and unseen rule examples;
- functional atlas changes and causal route utility for each population.

**Discovery procedure:** generate behavior contrasts from the observed function and influence matrix *without passing exception masks/program truth* to the circuit search. Subsequently reveal the generator truth to audit the discovered mechanisms.

**Controls:** rule-only (no exceptions), exception-only (random lookup), shortcut-only, correlated and independent keys, fixed params/compute. Beware a unique-key embedding table dominating the model budget.

**Success:** identify whether topology changes causal sharing/overlap or disruption of learned computations, including cases where both behaviors use shared circuits. **Failure:** if extractor presupposes memorization=short path, invalidate method.

## E4 — Redundancy and measurement calibration

**Question:** Can the instrument detect mechanisms that single-component ablation misses?

Construct a model with **two independent trained or analytically planted equivalent routes** producing the same needed output. On chosen probe domain, removing one should have small utility loss but removing both substantial loss. Include a neutral high-activation operation and a negative-utility operation on an OOD distribution.

Run:
- exact all-subgraph enumeration with zero baseline;
- alternative mean/resampled baselines;
- sufficiency and necessity separately;
- compare discovered graph to known planted computation;
- test $D_k,U_k$ monotonicity under nested exact subgraph classes.

**Success:** flags redundant circuit multiplicity, neutral route and sensitivity to baseline; reports ambiguity as needed. If fails, do not make mechanistic claims on E1–E3.

## E5 — Width × feature load × sparsity regimes

**Question:** Is a topology effect intrinsic to wiring or an artifact of crowded representations?

For selected E3 graphs, compare:
- width $d$ increasing with fixed generator factors;
- factor count increasing with fixed width;
- both increasing together;
- activation sparsity/co-occurrence independently varied;
- rule compositional depth varied at fixed factor count.

Use multiple seeds and enough training duration to observe delayed effects. Collect representation effective dimension, learned functional interactions, circuit sharing/redundancy, exception retention and unseen-rule utility. Avoid making 'monosemantic neurons' the axis; topology may change feature geometry without changing count.

**Success:** a phase map and boundary cases, even if effects reverse. Correlate outcomes with design load and measured covariance spectra but never assume a universal ratio that explains everything.

## E6 — Held-out program generalization of design principles

Only after a metric reliably predicts E1–E5 dynamics:
- choose routing comparisons from earlier studies;
- apply to held-out program family (e.g. modular addition or symbolic grammar);
- keep learned operator budgets fixed;
- compare whether predicted causal mechanism signature recurs, not only accuracy.

If a property of topology ceases to predict learning in a novel family, report the new boundary. Do not introduce Bayesian search until predictions have out-of-sample validity.

## Minimum report structure for each scenario

- **Pre-registered hypothesis** including causal graph mutation and opposite outcome.
- **Proof of matched controls** plus explicit differences in fixed arithmetic.
- **Loss/train trajectory** for context.
- **Blind functional discovery outputs**, including unanticipated patterns.
- **Causal intervention/utility analyses** with uncertainty, baseline sensitivity and unexplained fraction.
- **Behavioral partitions:** seen exceptions, unseen latent rule, shortcut-conflicting, broad probe.
- **Training-time acquisition probes** when feasible.
- **Multiple seed/task/width sensitivity**, explicit negative and contradictory cases.
- **Conclusion bounded by measured regime**, next discriminating test and links to raw artifacts.

## Precise work order for an agent

Implement E0 mathematical fixtures and its graph interpreter first, then E4 measurement-calibration fixtures **before** trusting E1–E3 circuit outputs. Next run E1 small graph pair with complete-function probes, E2 and E3 to introduce nontrivial learning/memorization, finally E5–E6 for scaling and transfer.

This order prioritizes *instrument validity and falsifiability* over a performance leaderboard.
