# Task families, memorization/generalization contrasts, and scale

**Active ordering (19 October 2026):** the initial empirical dataset is **linear conditional-token teacher data for a matched graph pair**, not parity, exceptions or grokking. Those remain follow-on nonlinear/generalization settings after we establish what fixed routing changes when expressivity and initial tangent geometry are controlled. See [matched linear study](15_matched_linear_topology_study.md) and [critical review](14_critical_review_and_revised_scope.md).

The task suite is a **scientific instrument**, not a benchmark leaderboard. It must reveal mechanism formation without telling the interpreter where the mechanism is. Training uses softmax CE over token labels. The generator may know latent ground truth for auditing, while the discovery algorithm must initially operate blind to it.

## 0. First study: linearly generated token distributions with known modes

Use $x\in\mathbb R^d$ from a controlled mean-zero distribution with known covariance and input subspaces. Let a fixed teacher matrix $M_*$ define:
$p_*(y|x)=\operatorname{softmax}(R M_*x/\tau+c_*),$
where $R\in\mathbb R^{V\times d}$ is the same **fixed full-row-rank readout** for both experimental graphs where dimensions permit. Student logits are $R h_\theta(x)+c$ for either parallel $h_P=(I+A+B)x$ or composed $h_C=(I+B)(I+A)x$.

Student function class must be identical; any representable target logit map with teacher/readout mismatch should be explicitly accounted for (a common-logit shift does not affect softmax). Sample a finite train dataset once and reuse exactly; also hold a fixed integration/probe set to estimate expected CE, conditional probabilities and learned linear logit modes.

**Vary:** singular strengths/orientations of $M_*$, alignment with covariance directions, class balance, finite sample size, label stochasticity, initial $A/B$ scales, optimizer and feature dimension. Keep training objective **softmax CE** in every primary trial.

**Discover first:** evolution of the centered student logit map and its singular/Fourier-like linear modes, from data-only probes as well as teacher-aligned decomposition. Observe one-example counterfactual transfer $T_t(i,j)$, not only teacher-mode fitting or aggregate CE.

**Why this precedes nonlinear work:** graph pair has identical attainable linear functions and identical parameter-to-logit Jacobians at exact zero, but a second-order $BAx$ interaction. It isolates a real parameterized-topology question without assuming a long nonlinear rule can form. No claims about circuits encoding XOR or memorization mechanisms are justified at this stage.

**Controls:** identical graph rerun, $B=0$ limiting equivalence, label/class balance changes, modes not observable through readout $R$, zero versus small nonzero init, matched step count versus matched achieved CE, and examples where composed routing is worse.

## A. Why simply training on parity isn't enough

A parity classifier is a good mathematical calibration: Boolean Fourier coefficients reveal interaction order exactly over a balanced finite domain. But a model can learn parity in unexpected ways; a graph with no substantial generalization might still memorize a small domain; and discovering the expected parity term is not the same as source-agnostic circuit discovery.

After the linear study, nonlinear mechanism research should use **at least three families**:
1. Simple known interaction / graph circuits for instrument calibration.
2. Rule + shortcut + arbitrary exception tasks for distinguishing reusable structure from instance-specific retention.
3. A held-out family of compositions (modular arithmetic / small latent programs / controlled symbolic grammar) to test method and topology generality.

No topology should be tuned solely to one family and then claimed to be universal.

## B. Calibration family: balanced Boolean mappings

Inputs $x\in\{\pm1\}^n$ with controlled independences. Token labels determined by e.g. a selected parity, low-order Boolean function, or shallow decision tree. For small $n$, enumerate the full input domain (never just training samples).

Control:
- interaction order $k$;
- number of relevant and irrelevant input factors;
- training example subset size;
- feature imbalance/correlation;
- model width and learned-operation depth;
- raw-versus-calibrated initialization.

Observations: output logit contrast Fourier coefficients across *all* subsets; emergence of interactions; causal graph interventions; exact verification that a learned path is needed. Some operations can compute complex rules in one stage; do not assume Fourier order measures graph path length.

## C. Follow-on nonlinear family: structured rules, shortcuts and memorized exceptions

Generate a deterministic latent program $y_{\rm rule}=f_\omega(u_1,\dots,u_r)$ outputting one of $V$ tokens. Include an input identity key $k$ and optionally nuisance factors $n$.

For a randomly sampled fixed set of training keys $\mathcal E$, assign persistent exception labels $y_{\rm exc}(k)$ independent of $f_\omega$ (reject coincidental equal labels when explicitly wanting an exception). Define:

$$y(u,k)=\begin{cases}
y_{\rm exc}(k), & k\in\mathcal E,\\
f_\omega(u), & \mathrm{otherwise}.
\end{cases}$$

This is a *deterministic* mapping if the same $(u,k)$ is shown again. Include training repetitions where useful. Random exceptions require instance/key storage; no statistical regularity should let an unseen arbitrary exception be predicted.

Add a shortcut input $s$, correlated with $y$ in training (or with $y_{\rm rule}$, specify which). Change its reliability in shifted evaluation. Example $P(s=y)=q$ with $q=.9$ on train, $q=.5$ or reversal at test. Keep the true label definition unchanged. Ensure shortcut randomness does not accidentally produce contradictory labels for identical full input without intending label noise.

### Dataset partitions (each must be explicit)

- **Train seen-rule:** nonexception training keys/tuples.
- **Train seen-exception:** arbitrary fixed exceptions; include consistent repeated contexts.
- **IID validation:** held-out observations under the same shortcut/data process (include seen/unseen key treatment explicitly).
- **Rule-compositional test:** unseen combinations of latent values and new nonexception keys; stratified by interaction/order.
- **Shortcut-broken test:** controlled $q=.5$, $q<.5$ or shortcut removal, holding the underlying rule constant.
- **Exception retention test:** exact held-out repeats of keys that were trained as exceptions, evaluated after later training, separate from unseen exceptions.
- **New exception test:** never claim random unseen exceptions should generalize; use them only to test whether an update can acquire a novel fact quickly while preserving old ones.

### Avoid artificial success

If an exception indicator is supplied explicitly, a trivial router can separate cases; this may be a useful control but weak evidence about emergent partitioning. In a harder version, supply only the key and allow the model to learn which cases are exceptions. Use balanced and unbalanced keys. Likewise, do not feed an explicit feature that already computes a purported compositional target in the main circuit-formation task; that belongs in a planted-mechanism control.

Tune key encoding, embedding size and latent-factor frequency; otherwise per-key memorization capacity and nominal network width become confounded. Every topology sees the same inputs/readout.

## D. Program family and unknown mechanisms

Generate a family of small compositional symbolic programs, for example lookup → transform → combine → compare → output token, with varied operator type and nesting. Prefer examples whose complete finite input domain remains enumerable initially.

Do not tell the mechanism-discovery pipeline the program structure, expected intermediate values, or required circuit length. The generator stores these privately for **post-discovery auditing**.

Split by program instances or operator combinations, not just random examples, to test whether measurements and topology trends persist outside the search distribution.

A stronger generator can include two competing valid representations of the same target function, different shortcut features, and task families that genuinely favor short computations. Architecture principles must have boundary conditions rather than an unconditional claim 'longer is better'.

## E. Why modular addition and tiny attention models belong later

Modular addition has an established interpretable Fourier-based mechanism and documented training transitions; ideal for validation of a discovery pipeline. It is not an unbiased target for discovering a mechanism we do not already know. After MLP graph instruments pass tests, a minimal attention-only or small transformer may reveal attention-specific route sharing and induction-like computations.

Attention introduces a new class of routing: **data-dependent weights from learned queries/keys**, not merely fixed skip edges. Treat this as an extension in which internal learned routing is fixed across compared models, while externally parameter-free connectivity varies. Document head count, context length and attention FLOPs.

## F. Scale and the hidden representational-pressure axis

Vary separately:
1. **Model width / parameter budget** at fixed latent features and task distribution.
2. **Latent feature demand** at fixed width (how many features, relative importance, correlations).
3. **Joint width and feature demand** to approximately hold feature load constant.
4. **Feature sparsity** and concurrent coactivation at fixed feature count.
5. **Composition complexity** ($k$, program depth, interaction type) at fixed feature count.
6. **Data support** (#distinct examples; repeats; contradiction frequency).
7. **Training steps and optimizer** (SGD first, AdamW replication; weight decay as controlled factor).
8. **Input/token length** later, once attention exists.

Define a *design* load $\rho_{\rm design}$ such as expected concurrently active latent features divided by representation width, **not** a claim that this is intrinsic superposition density. Record raw latent count, activation probabilities, covariance/importance distribution, and width. The measured effective dimension $d_{\rm eff}$ is not a true feature count.

Two model sizes are insufficient to infer scaling. At each regime use enough seeds to estimate success probability and heavy-tailed transitions, especially when grokking delays vary.

## G. Study capacity and data thresholds as phase maps

Define operational outcomes under a fixed training budget:
- $P^*(G,C)$: minimum capacity at which a specified generalizing mechanism or behavioral threshold is reached with prespecified success probability;
- $N^*(G,C)$: minimum unique examples needed, for capacity and task setting $C$;
- $T_{\rm circuit}(G,C)$: checkpoint/time at which a causally supported mechanism crosses a prespecified utility threshold;
- $T_{\rm rule}$, $T_{\rm shortcut}$, $T_{\rm exception}$: separately report development of these behaviors, rather than assuming ordering.

These are estimated quantities with uncertainty, not analytic scale laws. A topology may help under low capacity, hurt under high representational pressure, or vice versa. Report interaction terms.

Test at least a small Cartesian subset of $(\text{width},\text{feature count},\text{sparsity},\text{composition complexity},\text{dataset size})$. A fully factorial sweep may be expensive; use staged designs, holding nuisance variables explicit.

## H. Train/test loss is necessary but not the mechanism

Always log CE, accuracy/calibration and shortcut-broken outcomes; they detect invalid training or data leakage. But a lower CE alone does not show memorization or generalization circuits are separate, or that one topology caused circuit formation.

The primary comparisons are longitudinal **functional atlases** and **actual example-to-example transfer / interference**. Causal behavior-specific route utility follows where a reproducible functional change justifies targeted interventions; depth-based utility is a secondary, baseline-dependent descriptor. Use held-out task families and blinded discovery to avoid hindsight analysis.


## I. Expressive-competence control for nonlinear comparisons

Raw-input-fed residual blocks and sequential residual blocks do **not** generally represent the same finite-width nonlinear functions. If one graph generalizes to a compositional rule and the other cannot even fit the full input domain, we have demonstrated an **expressivity difference**, not isolated circuit-formation efficiency.

For each nonlinear target chosen in a learning-dynamics study:
1. Construct example weights that solve/approximate the task in both graphs if possible, or run a pre-declared strong-capacity/long-training approximation check without using test results to tune only one candidate.
2. Report residual approximation gap / failure to fit for each at matched learned capacity.
3. Evaluate function acquisition at equal optimizer steps and at comparable **achieved train CE** where feasible. The first is a compute comparison, the second distinguishes delayed learning from distinct fitted functions.
4. If competence cannot be established, relabel the result as an expressivity study with separate conclusions.

No empirical fitting test proves whole function-class equivalence. It merely bounds competence for a selected task, which is enough for a carefully qualified comparison.

## J. Training histories and source-agnostic audits

Choose tasks that have both predictable structure and arbitrary instance-specific exceptions, but **do not tell the circuit detector** the known program or which exception memorization is supposed to occupy which graph region. Use the generator truth only to score discovery and behavior transfer after observing the learned function.

Remember: a general circuit may act on memorized keys; a network can store instances and reusable features using overlapping or superposed directions. Behavioral generalization and memorization do not guarantee separately identifiable subgraphs.

See [measurement hierarchy](16_functional_learning_and_transfer_measurement.md), [protocol](08_experimental_protocol.md) and [decision gates](17_decision_gates_and_novelty.md).
