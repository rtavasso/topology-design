# Flagship Study 1: same function class and initial tangent geometry, different higher-order routing

**Active proposed first training study.** Phase 0 should validate the exact mathematics first. The question is not whether a "long path learns generalization"; it is whether a parameter-free rewire changes feature/mode learning dynamics **after first-order effects have been matched**.

## 1. Architectures and strict match

Let $x\in\mathbb R^d$ and $A,B\in\mathbb R^{d\times d}$ be independent trainable matrices. Define:
- **Parallel P:** $h_P=x+Ax+Bx=(I+A+B)x$.
- **Composed C:** $u=x+Ax,\ h_C=u+Bu=(I+A+B+BA)x$.

Each uses the same two $d\times d$ learned matrices, each evaluated **once**, and two parameter-free vector additions. P evaluates $Ax$ and $Bx$ from the same input; C feeds the first residual state to $B$. Both are acyclic and single-pass.

For token logits, use a **fixed common readout** $R\in\mathbb R^{V\times d}$ (full row rank if $V\le d$) and fixed equal bias $c$:
$$z_P(x)=Rh_P(x)+c,\qquad z_C(x)=Rh_C(x)+c.$$
Use standard vocabulary-wide softmax CE for both. A shared *trainable* readout is a separate controlled factor; a fixed readout isolates the $A/B$ parameterization in the first study.

Match dimensions, data, CE, optimizer, seeds, initialization, learned multiply count, additions, output head, and nominal FLOPs. Differences in activation memory and dependency critical path must be recorded.

## 2. Expressive competence: exactly identical linear function classes

Each input-output map is linear in $x$. P can realize any $M\in\mathbb R^{d\times d}$ by $A=M-I,B=0$; C can realize the same $M$ with the same choice. Consequently both sets of $h(x)$ equal all linear maps $Mx$, and the achievable logit maps $RMx+c$ match.

**This does not imply** equal parameter efficiency, norm regularization, optimization minima, or computational dependency. Both share expressive *sets*, not parameterization geometries. They cannot represent nonlinear XOR of raw independent inputs (without nonlinear feature engineering), so Study 1 must not be described as proof of nonlinear circuit discovery.

## 3. Initial function, parameter Jacobian, and tangent kernel match

Initialize $A_0=B_0=0$. Then:
$$z_{P,0}(x)=z_{C,0}(x)=Rx+c.$$

For matrix perturbations $\Delta A,\Delta B$:
$$Dz_P|_0[\Delta A,\Delta B]=R(\Delta A+\Delta B)x,$$
$$Dz_C|_0[\Delta A,\Delta B]=R(\Delta A+\Delta B)x.$$

Thus **full parameter-to-logit Jacobians are identical** at initialization:
$$J_{P,0}(x)=J_{C,0}(x).$$

With the same parameter-coordinate metric this gives the same empirical logit NTK:
$$K_{P,0}(x,x')=K_{C,0}(x,x').$$

Also, CE logits and residuals coincide, so initial CE parameter gradients coincide for *every* example, batch and data distribution. **The very first SGD step is identical in the two models** when starting from zero and using the same optimizer state and samples.

However the models' full CE Hessians do **not** generally match, because C has nonzero mixed second derivatives from $BAx$, whereas P has none in the $A/B$ logit map. This is precisely the higher-order effect left unconstrained.

## 4. Second-order functional divergence

For arbitrary matrices:
$$z_C-z_P=RBAx.$$

In the linear expansion about $A=B=0$, this term is bilinear in trainable parameters. For two sets of first-step parameter increments $\Delta A,\Delta B$, the difference after that first update is exactly:
$$z_{C,1}(x)-z_{P,1}(x)=R(\Delta B)(\Delta A)x,$$
assuming the *same first-step increments* in both networks (true for matched zero init and identical SGD state). For SGD, both increments scale as $O(\eta)$ for a fixed batch, so output divergence starts at $O(\eta^2)$.

For $V=2,d=1$ choose logits $(m,0)$ with fixed readout $R=[1;0]$, bias 0 and $x=1$:
$$m_P=1+a+b,\quad m_C=(1+a)(1+b)=1+a+b+ab.$$
At $a=b=0$, target-$1$ CE has derivative $r=\sigma(1)-1$, and SGD makes
$$a_1=b_1=-\eta r.$$
Then
$$m_{C,1}-m_{P,1}=\eta^2r^2>0.$$
For target 0, $r=\sigma(1)>0$ and the same positive second-order term *opposes* decreasing the incorrect logit margin. **There is no universal beneficial sign**. Use this asymmetry as a test fixture.

Mixed second derivatives:
$$D^2_{BA}z_C|_0[\Delta B,\Delta A]=R\Delta B\Delta A x,$$
where $D^2_{BA}z_P|_0=0$. The full Hessian of CE differs through its residual-weighted logit second-derivative term:
$$H_\theta L = J^\top C(p) J + \sum_j (p_j-e_{y,j})\,H_\theta z_j.$$
The positive semidefinite $J^\top CJ$ contribution matches initially; the second term need not. Do not call this 'identical loss landscape at initialization'.

## 5. Is exact zero a fair initialization?

Zero initialization of all $A/B$ matrices is intentional here because it creates **the exact matched-first-order control**. These are linear residual branches, so zero does not suppress all gradients as it can in a product-only architecture.

But exact zero creates symmetry and can be an atypical training regime. To test relevance:
1. analytic zero-init controls;
2. small nonzero matched weight init with $\|A\|,\|B\|$ swept and expectation of initial Jacobian mismatch explicitly quantified;
3. an alternative matched-*function* construction if feasible without changing the parameter count/graph;
4. matched activation/readout scales, optimizer seeds and data order;
5. weight decay and SGD vs AdamW as labeled controlled factors.

Do **not** assume that zero-init effects generalize. If differences vanish at realistic scales or reverse under optimizer choice, that is a main result.

## 6. Train distributions: linear modes, softmax CE

Construct $x$ from controlled mean/covariance (e.g. independent Gaussian or orthogonal categorical one-hot). Generate teacher conditional token distribution:
$$p_*(y|x)=\mathrm{softmax}(R M_* x/\tau+c_*),$$
where $M_*$ is a fixed matrix with known singular-mode strengths and singular directions. Sample fixed labels for finite datasets and optionally compute population CE from teacher probabilities over an enumerated/large fixed integration set.

The two architectures share teacher, examples, distribution and readout. Choose dimensions and $R$ so target logit functions are representable by both. For comparing mode acquisition, track the **observable projected logit map**, e.g. $R(I+A+B)$ or $R(I+B)(I+A)$, and its projection onto identifiable teacher modes. Beware that softmax is invariant to a common-logit shift and that rank-deficient $R$ hides unobservable modes.

**Negative controls:**
- identical topology twice, paired seeds, expected no systematic difference;
- $B$ frozen zero (identical functions for all $A$);
- a constant dummy bilinear term set to zero through intervention;
- data where the bilinear term has little effect by symmetry;
- vary target mode strengths, covariance and rank;
- compare matched training *steps* and matched achieved CE (both answer different questions).

## 7. Primary measurements (function and training transfer)

At each checkpoint:
1. **Complete or sampled function trajectory:** centered logits/probabilities on an invariant probe set, change from initialization and between checkpoints; teacher mode fits and output-logit singular spectra.
2. **Single-example update-transfer matrix:** exact counterfactual $T_t(i,j)=L_j(\theta_t)-L_j(\theta_t-\eta g_i)$ for finite $\eta$; $g_i=\nabla_\theta L_i$. For small-step SGD compare with $\eta g_i^\top g_j$ and report approximation error.
3. **Mode acquisition time:** time at which each pre-registered teacher mode is expressed to a specified tolerance *plus* an unsupervised spectrum of learned modes to avoid only seeing expected ones.
4. **Interference/reuse:** which input/output subsets improve or degrade jointly after an update; held-out examples from same teacher vs different modes.
5. **Optimization geometry:** $J_t$, cross-mode gradient alignment, the evolution of the empirical tangent kernel, Hessian-vector estimates and output divergence from analytic $BAx$ prediction.
6. **Robustness:** seeds, target mode spectra, data size, regularization, optimizer and width.

Teacher modes are explanatory ground truth; test some predictions and function discoveries *without using the teacher modes to select them*, then audit against the generator.

## 8. Falsifiable predictions and interpretations

**H1 [mathematically certain]:** at exact zero, functions/Jacobians/NTKs/initial SGD updates match, while the second-order $BAx$ effect is present. Failure signals a graph/gradient implementation bug.

**H2 [empirical]:** in some teacher/data regimes the compositional graph's higher-order term systematically changes the speed and localization of learned modes, beyond identical initial tangent dynamics.

**H3 [empirical]:** the sign/benefit of that change depends on task statistics and target-label distribution; counterexamples should exist. A universal improvement claim is *not* predicted.

**H4 [empirical, longer term]:** certain measured interactions in this exactly controlled setting predict how nonlinear graph rewires encourage or suppress later feature acquisition. This is not established by H1–H3.

**Falsifiers/limits:** no consistent effect after paired seeds; effects entirely predicted by simple reparameterization/optimizer anisotropy; no transfer to nonlinear tasks; sensitivity dominated by init/scale. These outcomes delimit what topology alone explains.

## 9. Study reports must NOT claim

- 'Circuit formation' merely because $BA\neq0$.
- 'Generalization advantage' merely because training CE drops faster.
- 'Parameter-independent topology effect' when initialization/regularization controls are missing.
- 'Identical gradients at all checkpoints' (they match only at zero under the setup).
- 'Identical convexity/curvature' (they are not).
- 'Nonlinear expressive advantage' (both models are linear in $x$).

## 10. Relation to prior art

[Saxe et al.](https://arxiv.org/abs/1312.6120) analytically study deep-linear parameter dynamics; [Shi, Shea-Brown & Buice](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html) study competing pathways and specialization; [Xu & Ziyin](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9941833e8327910ef25daeb9005e4748-Abstract-Conference.html) analyze finite-width feature-learning regimes. **Reproduce their relevant nulls/baselines and situate any new graph-specific finding precisely.** The paired graph construction above is a proposed controlled experimental object, not an established novel theorem.

## Deliverables for the first coding milestone

- Exact graph expressions, shapes and matching audit.
- Analytic and autograd checks for Jacobians, same first update, $BAx$ divergence and CE Hessian differences.
- Small 2-logit CE trainer with fixed common readout; teacher generator with controllable modes.
- Logged function trajectories and a small one-example transfer matrix, with actual finite-step and first-order estimates.
- Matched-seed replication and at least one counterexample where C is not better.
- Report that distinguishes **pure mathematical identity**, **observed optimization difference**, and **unproven generalization/transfer mechanism**.
