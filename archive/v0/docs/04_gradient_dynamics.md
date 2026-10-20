# Gradient dynamics, softmax CE, productive frustration, and circuit acquisition

This is the analytical bridge between **forward topology** and **learning outcomes**. A circuit's participation in inference and its acquisition through gradient updates are separate questions.

## 1. Cross-entropy gradient in function and parameter space

For logits $z_\theta(x)\in\mathbb R^V$, $p=\mathrm{softmax}(z)$ and target $y$:

$$r(x)=\nabla_zL=p-e_y.$$

If $J_\theta(x)=\partial z_\theta(x)/\partial\theta$ is $V\times P$, then

$$g_x=\nabla_\theta L_x=J_\theta(x)^\top r(x).$$

For a parameter update $\delta\theta$, a first-order logit change at test input $u$ is:

$$\Delta z(u)\approx J_\theta(u)\,\delta\theta.$$

For one SGD example $x$:

$$\Delta z(u)\approx-\eta J_\theta(u)J_\theta(x)^\top r(x).$$

This is the finite-dimensional **matrix-valued empirical tangent kernel** ($K_\theta(u,x)=J(u)J(x)^\top$). At large steps or as representations evolve, this approximation becomes inaccurate.

For another labeled example $u$, the loss change is:

$$\Delta L_u\approx-\eta\,g_u^\top g_x.$$

Positive $g_u^\top g_x$ implies first-order helpful transfer; negative implies interference. Measure the *actual* updates as well, under SGD and AdamW (whose preconditioning changes this simple relation).

An instantaneous single-use affine weight matrix gradient is the outer product $\delta_\mathrm{out}h_\mathrm{in}^\top$ (rank $\le1$); a tied/reused matrix accumulates a sum over positions. Topology changes both the activation $h_\mathrm{in}$ and backpropagated error $\delta_\mathrm{out}$, hence which low-rank circuit modifications a sample can induce.

## 2. Softmax shared-error coupling of routes

If $z=z_S(\theta_S)+z_L(\theta_L)$ with **disjoint parameters**, both routes receive the same logit residual:

$$\nabla_{\theta_S}L=J_S^\top r,\qquad
\nabla_{\theta_L}L=J_L^\top r.$$

When $p_y\to1$, $\|r\|_2\le\sqrt2(1-p_y)$; bounded $J_L$ implies $\|\nabla_{\theta_L}L\|\to0$ regardless of whether long-path mechanisms are desirable for generalization.

The cross-block loss Hessian is

$$H_{SL}=J_S^\top C(p)J_L,\qquad
C(p)=\mathrm{diag}(p)-pp^\top,$$

under the disjoint-parameter, additive-logit assumption. For binary $m=a+b$ the cross derivative is $\sigma(m)(1-\sigma(m))>0$ at finite $m$. Shared parameters or shared nonlinear computation add further terms/coupling.

In a binary target-positive case, let $m=a+bc$, $L=\log(1+e^{-m})$, $q=\sigma(-m)$. Then $\partial_a L=-q$, $\partial_bL=-cq$, $\partial_cL=-bq$. If $a=7,bc=-2$, target probability is $\sigma(5)\approx.9933$ despite the long component opposing the target. The long route sees small gradients.

**The short route can satisfy the combined objective before the long one learns a reusable rule.** But this is not automatically the case for all data distributions: contradictory data can leave residual gradients.

## 3. A priori conditional token grouping was a mathematical counterexample, not our design

For tokens indexed by group $g$ and within-group index $i$,

$$P(g,i)=P_S(g)P_L(i|g)$$

gives $-\log P(g,i)=-\log P_S(g)-\log P_L(i|g)$. With $z_{gi}=s_g+t_{gi}-\log\sum_j e^{t_{gj}}$, one global softmax exactly implements the factorization. It removes a type of cross-path loss coupling, but **imposes preselected token groups, allocates predictive responsibility by fiat, and adds fixed nonlinear normalization**.

A cheap linear approximation is $z_{gi}=s_g+t_{gi}-\bar t_g-\log n_g$, with $\bar t_g=\frac1{n_g}\sum_j t_{gj}$. Mean-centering eliminates some first-order group-mass dependence near uniform within-group predictions; residual higher-order coupling remains. Jensen's inequality gives the group mass factor $R_g=n_g^{-1}\sum_i\exp(t_{gi}-\bar t_g)\ge1$. This demonstrates why simple centering cannot achieve exact factorization.

**[PIVOT]** Do not use hand-chosen token groups as the flagship architecture: independent branches are not the target; allowing deep computation to revise shallow decisions is essential. Keep this as a clearly labeled secondary control for studying coupling.

## 4. Productive frustration, carefully operationalized

Working idea: learning pressure that the current shallow function cannot address, but deeper directions can, should remain usefully accessible and drive circuit formation.

On $N$ examples stack all CE logit residuals $R\in\mathbb R^{NV}$ and relevant Jacobians $J_S,J_L$ (fixed checkpoint, specified parameter groups). A local shallow stationary condition is:

$$J_S^\top R\approx0,$$

while deeper trainable directions remain:

$$J_L^\top R\ne0.$$

These depend on the choice of examples/weights/parameter decomposition and do **not** uniquely diagnose cognition or semantic 'frustration'. Gradient opposition and predictive disagreement are different from unresolved-error accessibility; neither should be maximized as an objective.

For accessibility beyond the shallow span define
$$R_\perp=(I-P_S)R,\quad \tilde J_L=(I-P_S)J_L,\quad
A_L=\frac{\|P_{\mathrm{col}(\tilde J_L)}R_\perp\|^2}{\|R_\perp\|^2},$$
when denominator $>0$. This is an unregularized local projection diagnostic. It can misleadingly approach 1 in heavily overparameterized models. Prefer also a **bounded-step or ridge-regularized** improvement measurement, e.g.

$$\delta_L^*=\arg\min_{\|\delta\|\le\epsilon}
\left[R^\top J_L\delta+\frac12\delta^\top J_L^\top C J_L\delta\right]$$

with constraints excluding components explainable by $J_S$ if the scientific question demands novel directions. Compare predicted and actual finite-step improvements. $C$ is block-diagonal softmax curvature per example; sample weighting and stabilizing ridge must be reported.

**Key caveat:** paths in a general residual DAG share parameters and activations, so there is no canonical partition into short- and long-*path* parameter Jacobians. Diagnostic solutions:
- in the primary controlled family, classify **learned operations** by graph ancestry/participation, acknowledging overlap;
- use edge interventions or differentiable edge tags during analysis, *not* learned path-specific gates in the trained model;
- report ambiguity when attributions overlap;
- never imply independent parameters imply disjoint circuits.

## 5. Exact toy for unresolved error and long-path learning

Let $x_1$ be balanced in $\{\pm1\}$, $x_2=+1$ with probability .9 and otherwise -1, true label $y=x_1x_2$. Consider binary target-sign margin $m=ax_1+d(x_1x_2)$.

Expected binary CE:
$$L(a,d)=.9\,\log(1+e^{-(a+d)})+.1\,\log(1+e^{a-d}).$$

At $d=0$, short-only optimum $a=\log9\approx2.197$:

$$\partial_a L= -.9(.1)+.1(.9)=0,\qquad
\partial_dL=-.9(.1)-.1(.9)=-.18.$$

The shallow shortcut has reached its current optimum while adding the correct compositional feature still reduces loss. This is the motivating instance of *productive unresolved error*. If $d=bc$ and $b=c=0$, then $\partial_bL=c\,\partial_dL=0$ and $\partial_cL=b\,\partial_dL=0$—the loss asks for a feature but the parameterized route cannot learn it at first order. Move away from exact zero and gradients are proportional to other factors.

Do not claim this proves networks spontaneously *discover* the interaction. We supplied $x_1x_2$ analytically. Real circuit formation must be tested with modules that can learn the feature, without providing it directly.

## 6. Alternative topologies with same weights can alter gradient access

Two learned scalars $a,b$ and two fixed additions:

**Nested residual**: $u=x+ax$; $y=u+bu=(1+a+b+ab)x$.

**Increment processing**: $u=ax$; $y=x+u+bu=(1+a+ab)x$.

At zero weights, output identity. $\partial y/\partial b$ is $x$ for nested but $0$ for increment processing. The first offers immediate trainability of the second operation; the second requires earlier learning but makes later computation dependent on an increment. (Depending on how additions are counted, explicitly count both merges; these examples are pedagogical and must be instantiated identically in the matching auditor.)

This comparison does **not** establish which will build more compositional representations. Short-route accessibility may fit shortcut features faster and potentially starve deeper feature learning, or may catalyze it. Train and measure.

## 7. Curvature and the tension between easy fit and new features

For CE loss with logits $z_\theta$, Hessian for one example is
$$H_\theta L=J^\top C(p)J+\sum_{j=1}^V (p_j-e_{y,j})H_\theta z_j.$$

The first term is positive semidefinite; the second can be indefinite. With logits affine in weights, the second vanishes and loss is convex in parameters, but the feature map is fixed. Higher-order interactions among parameters can make optimization harder **and** allow representations to change during learning.

Tracking the empirical tangent kernel $K_t$ gives a proxy for changing first-order functional geometry:
$$\Delta J_\theta z(x)\approx H_\theta z(x)\,\Delta\theta.$$

Kernel drift alone is not proof of a useful new circuit; it must correspond to new generalizable function and causal intervention effects.

## 8. Temporal definitions to measure, not assume

- $T_{\rm shortcut}$: time to learn an easy predictive rule;
- $T_{\rm composition}$: first validated emergence of a novel compositional computation;
- $T_{\rm generalization}$: time to a defined held-out rule-transfer threshold;
- $T_{\rm cleanup}$: when memorizing components are reduced/replaced (not necessarily occurring).

Grokking and lazy-to-rich transition research motivate different orders among these timescales. It is possible for useful circuit structure to develop even after train accuracy saturates; 'small gradients' does not imply 'exactly zero learning'. Measure changes longitudinally.

## 9. A training example's target probability vs argmax

If identical context $x$, same target $y$, and CE decreases, $p(y|x)$ necessarily increases. Yet argmax can remain another token or even move away from the target as other probabilities change. A later update can undo improvement. A paraphrased or shifted context may not benefit. Distinguish:
- acquisition on the trained context;
- expression in output logits;
- retention during subsequent optimization;
- transfer to related contexts.

This distinction is central to studying "hidden" learning versus failure to generalize.

## 10. What to implement for these dynamics

On a small stable probe set, compute logits/probabilities, loss gradients, per-example cross-loss deltas after cloned single-example updates, Jacobian spectra/estimated rank, groupwise projected-gradient residuals, and actual finite-step versus first-order predictions. Intersect these with *causally identified* mechanisms from the measurement framework. Avoid relying on high-dimensional full Hessians for the first implementation. Use Hessian-vector and Jacobian-vector products when scale requires.

No result from this document by itself identifies a specific memorization or generalization circuit.


## 11. Revised research target: measured functional updates before inferred circuits (19 Oct 2026)

The original discussion correctly emphasized CE competition, product-path trainability and useful unresolved error. But a constructive review identified a risk: **better compositional task performance may simply follow from greater expressive capacity**, and **path-depth lesions are not a canonical measure of circuit acquisition**.

We now start with the matched linear graph pair $h_P=(I+A+B)x$ and $h_C=(I+B)(I+A)x$. These have equal linear function classes and equal initial parameter-to-logit Jacobians at $A=B=0$, but the latter has a mixed $BAx$ term. Under a common fixed readout and softmax CE, their initial CE gradient and empirical NTK are the same; their full Hessians need not be. An identical first SGD update yields a *second-order* difference in logits. This lets us ask exactly what **higher-order numerical dynamics of parameter interactions** add beyond initial gradient smoothness.

The primary empirical outcome is no longer raw error accessibility alone. It is:
$$T_t(i,j;\eta)=L_j(\theta_t)-L_j(\theta_t-\eta\nabla L_i(\theta_t)),$$
the **actual** one-example loss improvement or interference on context $j$, tracked over training and combined with complete functions/unsupervised learned-mode trajectories. For small SGD steps:
$$T_t(i,j;\eta)=\eta\,g_i^\top g_j+O(\eta^2).$$
At finite step $T(i,j)$ need not be symmetric; with AdamW the true state-dependent update replaces $-\eta g_i$.

The useful question is not whether gradients are large, or whether short and long logits disagree. It is **how information acquired from an example becomes reusable, how other behaviors are overwritten, and which computational dependencies are formed**. The short/long framing is not a unique decomposition when routes share parameters. A causal circuit analysis becomes appropriate *only once* a real function-level phenomenon has been established and replicated.

See [critical review](14_critical_review_and_revised_scope.md), [matched study](15_matched_linear_topology_study.md), and [measurement priorities](16_functional_learning_and_transfer_measurement.md). Prior CE grouping derivations remain historical demonstrations of coupling, **not** an approved token-partition architecture.
