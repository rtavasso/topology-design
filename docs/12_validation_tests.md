# Validation cases, falsification tests, and scientific acceptance checks

A mechanistic experiment is only as good as its instrumentation. Every failure below is a reason **not** to launch a topology search yet. Many tests are exact and inexpensive.

## 1. Graph integrity and matching

**G1. Acyclic validation:** reject self-loops, cycles, missing ports and forward references not permitted by the selected family.

**G2. Operator conservation:** count exactly the same learned parameter tensors, shapes, operator executions and nonlinear activations for each asserted matched pair. Fail comparisons that silently duplicate learned ops.

**G3. Resource accounting:** verify fixed merge count and operator eval count; report FLOPs and activation memory across candidates. Different graph edges may change runtime even when learned evals equal.

**G4. Parent grammar count:** for $L=1,\ldots,6$, enumerate $L!$ labeled parent vectors $(k_1,\ldots,k_L)$ with $0\le k_i<i$. Distinguish labeled enumeration from function-isomorphic/deduplicated graph count.

**G5. Graph serialization:** roundtrip preserves behavior and structural hash; reordering commutative addition edges does not spuriously produce different topology hashes if canonicalization claims that property.

**G6. Paired state:** identical named learned modules have bit-identical initialization in paired raw-seed experiment; identical data order; initial forward outputs may differ *because topology differs*, which is expected.

**G7. Nondegenerate paths:** detect unreachable/disconnected learned operators and graph regions where the input-output function cannot depend on an operation. Report instead of silently treating as ordinary candidate.

## 2. Exact mathematical fixtures

**M1. Scalar residual:** verify plain $wx$ and $(1+w)x$ share possible slopes but identity at different $w$.

**M2. ReLU finite expressivity:** verify $x+2\mathrm{ReLU}(-x)=|x|$ numerically over symmetric values. Check that a single bias-free ReLU plain branch cannot represent absolute value on both half-lines (analytic assertion, not solely a finite optimization failure).

**M3. Linear rank bottleneck:** with random full-rank $W_1,W_2$ of inner width $m<d$, plain Jacobian has rank $\le m$; residual Jacobian can have rank $d$. Use numerical rank threshold documented.

**M4. Residual injectivity local bound:** random $J_F$ with spectral norm $<1$ implies $\sigma_{\min}(I+J_F)\ge1-\|J_F\|_2$ within tolerance. Do not test 'global invertibility' from a single Jacobian.

**M5. Scalar DAG polynomials:** numeric DAG results match $(1+bc+abc)x$ and $(1+c+abc)x$ at arbitrary weights. Their parameter gradients at zero are $(0,0,0)$ vs $(0,0,x)$.

**M6. Local displacement:** optimal norm for $bc=\epsilon$ at $a=0$ is $\sqrt{2|\epsilon|}$; the one-weight correction $c=\epsilon$ has norm $|\epsilon|$. Confirm for multiple $\epsilon$ (signs).

**M7. Hessian counterexample:** for $L=\frac12(1+a)^2(1+b)^2$, at origin Hessian [[1,2],[2,1]], eigenvalues 3 and -1.

**M8. One-example projection:** SGD on $f=x+Wx$ from $W=0$ with target zero, unit $x$, $\eta=1$ gives $W'=-xx^\top$, annihilates direction $x$ and preserves every vector orthogonal to $x$.

**M9. Softmax gauge:** adding same constant to each logit leaves CE and probabilities unchanged; centered logit atlas unchanged.

**M10. Two-logit CE:** gradient w.r.t. logits equals $p-e_y$; binary margin implementation agrees to tolerance.

**M11. Shared CE coupling:** for additive binary margin $m=a+b$, cross-Hessian $\sigma(m)(1-\sigma(m))$. Numeric check for multiple finite margins.

**M12. Shortcut-frustration toy:** with $q=.9$, $a=\log9,d=0$, analytic $\partial_aL=0$, $\partial_dL=-.18$ under specified distribution. Verify via numerical expectation and autograd; note finite Monte Carlo has sampling noise.

**M13. Product path starvation:** set $d=bc$ with $b=c=0$, gradients vanish even with nonzero $\partial_dL$. Move one factor nonzero, gradient becomes available.

**M14. Differentiable CE Hessian:** compare autograd Hessian to $J^\top CJ+\sum_j r_j H z_j$ for a tiny nonlinear model.

**M15. Conditional-softmax identity:** diagnostic only. Fixed partition $z_{gi}=s_g+t_{gi}-\mathrm{LSE}(t_g)$ yields factorized $P(g,i)$, CE separated; this is NOT a candidate architecture. Mean-centered approximation differs off uniform logits.

## 3. Functional discovery and generalization tests

**F1. Full-domain Walsh inversion:** exhaustive Boolean function recovered to numerical precision from coefficients; Parseval/orthogonality hold under uniform input distribution.

**F2. Unknown interaction:** plant a random high-order interaction not listed in hand-written detector features. Functional atlas finds it with no explicit target-order probe.

**F3. Softmax gauge:** functional decomposition on centered logits does not report common logit shifts as learning.

**F4. Data distribution sensitivity:** correlated versus uniform Boolean probes yield documented differences; detector clearly labels measure and basis.

**F5. Shortcut contrast:** train/held-out conditional shortcut strength predictable by generator; shifted probe holds latent rule fixed.

**F6. Exception retention:** repeated exact training key retrieves its arbitrary label; unseen random exception labels are NOT treated as generalization target.

**F7. Dataset leakage:** no accidental shared keys/program instances between disjoint partitions when declared; generator truth not supplied to source-agnostic analysis stage.

**F8. One-example transfer:** for tiny model and small learning rate, actual $\Delta L_j$ matches $-\eta g_j^\top g_i$ to first order; deviation grows with step size. Test multiple related and unrelated inputs.

## 4. Mechanistic extraction and causal depth

**C1. Planted short path:** retained depth-limited graph reproduces target and predicts zero/near-zero KL where mathematically appropriate.

**C2. Planted nonlinear composition:** the exact subgraph needed is present; depth-limited search must not find false shallow equivalent on a sufficiently discriminating probe domain.

**C3. Redundant routes:** removing either route alone has small effect, joint removal large effect. Mechanism reports recognize redundancy; no 'neither important' inference.

**C4. Neutral large activation:** a high-activation unused component has negligible causal utility. Attribution by magnitude alone fails this test.

**C5. Negative-utility component:** removing a misleading path *improves* target CE on shifted distribution; report negative $U_k$ or negative lesion cost rather than clipping to zero.

**C6. Nested subgraph monotonicity:** with $\mathcal C_k\subseteq\mathcal C_{k+1}$ and exact minimization, $D_k$ and $U_k$ nonincreasing. A noisy approximate optimizer can violate this; report as search error, not natural nonmonotonicity.

**C7. Baseline sensitivity:** zero, mean, sampled and counterfactual patch changes causal estimates in known example; implementation records method and outcome.

**C8. Redundant circuit search:** if multiple equally faithful subgraphs exist, system reports multiplicity/ambiguity rather than 'the unique circuit'.

**C9. Fidelity gap:** if a discovered subgraph cannot faithfully reproduce behavior on held-out probes, report unexplained behavior fraction rather than claiming full mechanism.

**C10. Computational complexity:** exhaustive search agrees with heuristic on tiny graph; run-time/search budget and graph coverage are logged on larger graphs.

**C11. Gradient-through-edge vs output ablation:** prove by a toy example that the effect of stopping gradient during training may differ from ablation at inference; label both correctly.

## 5. Representation and instrument robustness

**R1. Orthogonal basis symmetry:** for an exact compatible representation rotation with compensating downstream map, predictions unchanged and basis-robust geometric metrics unchanged within tolerance.

**R2. General invertible basis:** document which metrics are *not* invariant. Never assume CKA invariance to arbitrary invertible transformations.

**R3. Superposition fixture:** plant sparse high-dimensional features in fewer observed dimensions; recover interference evidence while not claiming exact atomic neuron count.

**R4. Width-demand sweep:** widening a model while increasing number of features can *increase* packing pressure; measurement should not equate width with monosemanticity.

**R5. SAE sensitivity:** optional SAEs with different dictionaries/regularization can split/merge apparent features; pipeline reports reconstruction and cross-resolution stability rather than a canonical count.

**R6. Effect of probe set:** discovered mechanisms on discovery probes must have held-out fidelity and intervention validation.

## 6. Reproducibility and statistics

**S1. Same-run replay:** same exact config/data/code/seeds replicates deterministic CPU outputs; log nondeterminism if GPU differs.

**S2. Paired controls:** same module initialization and data order across graph mutations; included module-label permutation sanity check.

**S3. Run-level uncertainty:** confidence intervals bootstrap independent seeds/task instances, not correlated probe rows.

**S4. Search bias:** mechanism found on a used task family must replicate on a held-out program family before a general principle is declared.

**S5. Negative-result preservation:** failed circuit search, instrument ambiguity and failed topology hypotheses produce saved report and raw metrics.

**S6. Null baseline:** identical graphs under different IDs should have no systematic difference except stochastic/numerical noise.

## 7. First milestone done when...

- Graph family works, $L!$ enumeration and graph auditing reproducible.
- All applicable mathematical fixture tests above pass.
- A two-token softmax CE toy can train from a stored config and produce immutable checkpoints/probes.
- An exact small-domain output atlas detects an unknown planted interaction.
- One inference-time subgraph intervention demonstrates the expected causal effect and records its baseline.
- The repo can output a run manifest and basic report that another developer reproduces.

Until those gates pass, do not claim a discovered rule about architectural topology. Add tests in small coherent increments as the corresponding features are implemented.
