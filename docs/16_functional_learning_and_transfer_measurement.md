# Measurement first: functional learning trajectories and one-example transfer

**Status:** active primary measurement protocol for Studies 1–2. This document takes precedence over any earlier suggestion that causal-depth scores or fully automated circuit discovery are needed before the first topology experiment.

## 0. What exactly are we trying to observe?

We need to detect how a fixed graph changes:
1. **Acquisition**: new input-output dependencies and modes appearing over training.
2. **Reuse**: whether a parameter update changes multiple appropriately related behaviors.
3. **Interference/retention**: whether learning a new case destroys an old behavior.
4. **Function novelty**: which patterns appear that weren't selected by the task designer or a mechanistic probe.
5. **Implementation** (later): causal routes that express/acquire a detected behavior.

None of these is simply test accuracy, total CE, neuron selectivity, path count, empirical NTK drift or SAE feature count.

A network has no unique factorization into 'memorization' and 'generalization' mechanisms. Those terms describe behavior on specified distributions, while information sharing and update locality are closer to candidate internal variables.

## 1. Define observable function space, not a neuron ontology

For checkpoint $\theta_t$, probe set $X_Q=(x_1,\dots,x_N)$, and $V$ logits $z_{\theta_t}(x)$, remove the softmax gauge:
$$\tilde z_\theta(x)=z_\theta(x)-\frac1V\mathbf1\mathbf1^\top z_\theta(x).$$
Form a matrix $F_t\in\mathbb R^{N\times V}$ by stacking these centered logits. Store also $P_t$ of probabilities and CE against a specified target distribution.

For any two times:
$$\Delta F_{t_1\to t_2}=F_{t_2}-F_{t_1}.$$

Function comparisons can use:
- logit contrast changes and Frobenius energy with an explicit scaling convention;
- changes in probability and KL divergence (sensitive to calibration/saturation);
- SVD of $F_t$, $\Delta F$, and their alignment to input or teacher modes (for linear studies);
- stable unsupervised clustering of rows/changes across input cases;
- full Walsh–Fourier / categorical interaction atlas for enumerable finite domains (for nonlinear studies).

**Do not confuse a new functional mode with a new neural circuit.** A simple rescaling or temperature change can increase logit Fourier energies with no new algorithm. Normalize spectral shape, track calibration and test causal effects before interpreting it.

A full Boolean Fourier expansion needs uniform independent Boolean probe measure. For correlated/biased data, report the actual probe measure and use a suitable generalized decomposition rather than silently reusing uniform-domain orthogonality.

## 2. One-example causal training update: actual transfer matrix

For examples $(x_i,y_i)$ and $(x_j,y_j)$ at checkpoint $\theta_t$, define the **counterfactual single-example SGD effect**:
$$
T_t(i,j;\eta)=L_j(\theta_t)-L_j(\theta_t-\eta \nabla_\theta L_i(\theta_t)).
$$
Positive = the update on $i$ improves loss on $j$; negative = interference. Evaluate on cloned model weights (and for optimizer variants, cloned optimizer states). This is a *causal intervention on optimization*, not a correlational explanation from a gradient dot product.

First-order SGD estimate:
$$T_t(i,j;\eta)=\eta\, g_j^\top g_i+O(\eta^2),\qquad g_i=\nabla_\theta L_i.$$

Estimate $\frac{T_t(i,j;\eta)}{\eta}$ for a sequence of small $\eta$ to test first-order accuracy and identify higher-order update effects. For AdamW, use the actual state-dependent update $\Delta\theta_i$; $\eta g_i^\top g_j$ is not the correct universal approximation. Define a corresponding **logit-transfer tensor** $\Delta z_{\theta_t}(x_j\mid i)$ for finer outcome detail.

Properties:
- $T(i,j)$ is **not generally symmetric** at finite step size, even when $g_i^\top g_j$ is symmetric, due to curvature and different update directions.
- $T$ is defined relative to fixed labeled examples and the training protocol; reweighting losses or batch order changes results.
- A positive transfer does not automatically mean the same abstract rule was acquired; correlated shortcuts can give positive transfer.
- Large transfer magnitude can be bad if it changes unrelated contexts. The useful pattern is relation-dependent, and that relation must be evaluated independently.

## 3. How to measure reuse without prescribing neuron circuits

Generate an **evaluation relation matrix** $S(i,j)$ from controlled task latents, but withhold this from unsupervised transfer clustering. Categories might include:
- pairs sharing the same underlying compositional factor with different surface details;
- pairs sharing a shortcut but conflicting on the true rule;
- identical remembered exception key vs unrelated exception key;
- unrelated generative factors or target classes.

Analysis order:
1. Discover clusters/low-rank components of $T_t$ or logit-transfer tensors *without latent labels*.
2. Validate on held-out probes and checkpoint seeds.
3. Reveal latent relationships to audit specificity, precision/recall, and overlooked cases.
4. Test whether a topology mutation systematically changes **alignment between beneficial transfer and independent latent-rule relations**.
5. Contrast effects at matched number of updates **and** matched achieved train loss to distinguish slower optimization from learning a different functional solution.

Useful summaries (none canonical):
- conditional mean of $T(i,j)$ by latent relation;
- fraction of unrelated pairs harmed beyond a threshold, with confidence intervals;
- operator and Frobenius spectra of the transfer matrix, normalized for number of probes and step size;
- novelty of transfer subspaces over training;
- overlap of update-affected populations for different examples;
- acquisition time of a generalizable mode vs time needed to memorize an exception;
- actual loss rebound on earlier examples after streams of updates.

Use task-instance/seed-level uncertainty; do not bootstrap millions of correlated matrix entries as though they were independent experiments.

## 4. Link transfer to topology analytically

For logits $z_\theta(x)$, softmax residual $r=p-e_y$ and Jacobian $J_\theta(x)$:
$$g_i=J(x_i)^\top r_i.$$

Initial parameter-Jacobian structure predicts first-order transfer; higher-order learning changes both weights and those Jacobians. Compare:
$$K_t(x_i,x_j)=J_t(x_i)J_t(x_j)^\top,$$
and the change $\Delta K_t$ across checkpoints.

For the matched linear graph pair in [Study 1](15_matched_linear_topology_study.md), $K_0$ is exactly matched but mixed parameter second derivatives differ. This makes **deviation from common first-order predictions** an explanatory object, not a failure of control.

If two networks share $K_0$, it does **not** follow that they will have the same *finite-step* update effects or generalization after one update. The $O(\eta^2)$ terms and later geometry may differ.

Output representation/singular-mode analyses in the first study should be done both with teacher alignment and without (unsupervised learned-mode discovery), so the analysis isn't fully template-driven.

## 5. Functional novelty is not mechanistic identification

The measurement stack has an escalation protocol:

**Level A — measured changed function.** A new reproducible input-output dependency is visible on a distribution, and can't be explained by a common logit offset or trivial gain alone.

**Level B — measured learning transfer.** Updates on selected examples systematically induce the behavior in related unseen cases, with enough trials and counterexamples to show it's not merely a shortcut artifact.

**Level C — causal implementation.** Targeted intervention on a trained graph route selectively changes the discovered behavior, with held-out-probe fidelity and baseline sensitivity.

**Level D — acquisition mechanism.** Controlled training-time intervention changes when/if the behavior develops, beyond effects on the same route's inference-time expression.

A study can legitimately stop at Level A or B. Don't build an unrestricted circuit discoverer as an artificial blocker.

## 6. What about causal depth $D_k$ and $U_k$?

The earlier [measurement document](06_mechanistic_measurement.md) defines minimum retained-subgraph KL and loss utility at depth $k$. Retain those as **secondary, intervention-dependent descriptive profiles**, only after:
- fixing an explicit learned-operation granularity;
- checking full-model/retained-subgraph fidelity;
- testing multiple replacement baselines;
- checking redundant pathways and the inability of lesions to identify past acquisition;
- verifying exact small-graph searches against heuristic estimates.

Path depth is not equivalent to input-interaction order, mode reuse or memorization/generalization.

## 7. A concrete collection strategy for small versus larger models

**Tiny linear graphs:** symbolic form and full parameter Hessian feasible; analytically generated mode distributions; every checkpoint's projected linear logit map; a modest $N\times N$ exact transfer matrix.

**Small nonlinear graphs:** enumerate small input domains; Walsh–Fourier or ANOVA basis; transfer on a selected balanced probe grid; focused subgraph intervention only for functional changes worth explaining.

**Larger graphs:** probe subsampling, randomized SVD/JVP/VJP, stratified example pairs, checkpoint selection from coarse functional atlas; report estimator variance and sensitivity. Never imply exhaustive discovery.

## 8. Controls and threats to validity

- **Probe selection:** fixed, discovery, and held-out probes; task families held out entirely when extrapolating.
- **Gauge and scale:** common-logit shifts, temperature and saturation can distort raw changes.
- **Optimizer state:** use cloned state, and do not mix SGD first-order formula with Adam's update.
- **Representational capacity:** see [datasets and scale](07_datasets_and_scale.md). Comparable performance doesn't prove same mechanism; wildly different approximation floors preclude clean learning dynamics comparison.
- **Finite data:** sample noise may create false input relations or spurious transfer.
- **Symmetries:** coordinate/feature remappings can change neuron labels with no functional difference.
- **Counterfactual shift:** patching/edge ablations may create off-manifold states; treat claims as conditional on intervention.
- **Mode selection bias:** scanning a huge space of candidate spectral changes can overfit; validate discoveries on new probes/seeds.

## 9. What to deliver first

A small, reproducible artifact with:
- one graph pair, both trained with full softmax CE;
- function snapshots and centered-logit spectral change;
- actual exact single-example update-transfer matrix at selected checkpoints;
- first-order approximation and error;
- structured probe relations **revealed only for audit**;
- paired seeds with CI across runs;
- a short report with both unexpected learned function changes and null observations.

Do not implement an SAE farm, graph-wide circuit trawler or Bayesian NAS controller for this deliverable.
