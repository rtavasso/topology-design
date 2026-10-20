# First-principles mathematics of information-routing topology

Purpose: retain exact numerical examples that motivated the project, separate representational capacity from ease of learning, and give testable toy oracles. Assume stated architectures exactly; **do not extrapolate these conclusions universally to transformers**.

## I. Topology as a parameter-dependent function

Given learned scalar-multiplication nodes, fixed identity edges, additive merges, and no nonlinearities, an acyclic computation has:

$$
y(x;\theta)=x\sum_{p\in\mathcal P}\prod_{i\in p}w_i
$$

where paths with no learned multiplier contribute 1. A DAG determines which monomials exist. This path expansion assumes single scalar input/output and **only linear, additive operations**; it does not apply unmodified to nonlinear activation, normalization, attention or tied parameters, where paths interact.

Important distinct quantities:
- **Graph path length:** count of edges, may include parameter-free edges.
- **Learned-operation path length / polynomial degree:** count of learned multipliers on a monomial.
- **Nonlinear compositional depth:** count of sequential nonlinear operations actually causally required.
- **Expressive function class:** set of functions achievable by some weights.
- **Initial tangent directions:** Jacobian of the function with respect to parameters at initialization.
- **Reachability/conditioning:** how far in weight space or along optimization it is to desired behavior.
No one quantity can substitute for the others.

### Linear residual shifts parameter origin

Without skip: $y=wx$. With skip: $y=x+wx=(1+w)x$.

| Desired function | plain $w$ | residual $w$ |
|---|---:|---:|
| identity $y=x$ | 1 | 0 |
| attenuate $y=.8x$ | .8 | -.2 |
| erase $y=0$ | 0 | -1 |
| negate $y=-x$ | -1 | -2 |

Same scalar linear function class; different initialization/weight-decay preference. Identity at zero means useful default, not more degrees of freedom. Also, at corresponding solutions, sensitivity to perturbation $\delta w$ is the same scalar $x\,\delta w$. 'Farther from zero' is not intrinsically 'less stable after learning'.

### ReLU counterexamples

For $x>0$, $y=x+\mathrm{ReLU}(wx)$ equals $x$ for every $w<0$. This is a broad identity regime but $\partial y/\partial w=0$ on such examples. It cannot subtract a positive correction for positive $x$ unless a signed output projection or other structure is available.

Using **two trainable scalars** $v,w$ and one ReLU:
- plain $f_P(x)=v\mathrm{ReLU}(wx)$;
- residual $f_R(x)=x+v\mathrm{ReLU}(wx)$.

The residual network computes $|x|$ with $w=-1,v=2$, but the plain network with a single bias-free ReLU cannot compute $|x|$ on all real $x$ (one half-line is identically zero). Conversely the plain network computes $f=0$ by $v=0$; the residual network cannot compute the zero map on both half-lines because its residual branch is zero on one half-line. **Finite-width expressivity can change in either direction**, not merely optimization convenience. Repeating with additional ReLUs may eliminate the difference.

### Linear information bottleneck

For $x\in\mathbb R^d$ and $m<d$:

$$ f_P(x)=W_2\phi(W_1x), \quad J_{f_P}=W_2D_\phi W_1,\quad \mathrm{rank}(J_{f_P})\le m.$$

A dimension-matched residual version is $f_R(x)=x+W_2\phi(W_1x)$ with Jacobian $I+W_2D_\phi W_1$, which can have full rank. A bypass changes the **cut structure** of information flow without adding learnable parameters. This does not imply its gradient training will favor generalization.

### Near-identity residuals and erasure

If $f(x)=x+F(x)$ and at some input $\|J_F(x)\|_2<1$, then
$$\sigma_{\min}(J_f(x))\ge1-\|J_F(x)\|_2>0.$$
So the block cannot locally collapse an input direction at that point. If $F$ is globally Lipschitz with constant strictly below 1 on the entire domain, $x+F(x)$ is injective. These conclusions are conditional on the norm bound; deeper compositions and strong $F$ can erase. Linear $F(x)=-x$ erases exactly.

## II. Matched scalar DAGs with different parameter degrees

Use 3 trainable scalar multiplications $a,b,c$ and 2 fixed additions.

**A**: $u=x+ax$; $v=b u$; $w=cv$; $y=x+w$.

$$y_A=(1+bc+abc)x.$$

**B**: $u=ax$; $v=bu$; $w=x+v$; $y=x+cw$.

$$y_B=(1+c+abc)x.$$

At $a=b=c=0$, both output identity. Derivatives at the origin:

$$\nabla_{a,b,c}y_A=(0,0,0),\quad
\nabla_{a,b,c}y_B=(0,0,x).$$

For a target correction $y=(1+\varepsilon)x$, the A graph can realize it with $a=0,bc=\varepsilon$; minimum two-weight Euclidean displacement is $\sqrt{2|\varepsilon|}$ (assuming real signed weights). B achieves it by $c=\varepsilon,a=b=0$, displacement $|\varepsilon|$. For sufficiently small nonzero $\varepsilon$, the latter is smaller.

This example is about **local accessibility**, not absolute superiority. Both achieve arbitrary scalar slopes; initializing at exactly zero makes A first-order stationary for all targets, though higher-order directions exist. Vary seeds away from zero and compare across scales.

## III. Single-example learning can erase selectively

Take $f_W(z)=z+Wz$, $W_0=0$, and squared-error loss $\ell(W;x)=\frac12\|x+Wx\|^2$ for a single unit vector $x$ with target zero. **Squared error here is a mathematical teaching case, not our training protocol**.

$$\nabla_W\ell|_0=xx^\top,\quad W_1=-xx^\top$$

after one SGD step with $\eta=1$. Therefore:

$$f_{W_1}(z)=(I-xx^\top)z.$$

The learned operation destroys the projection onto $x$ and preserves the orthogonal complement for *all* $z$. An ordinary linear layer initialized at $I$ would do the same. This illustrates that selective erasure is possible without gates; the challenge is whether useful feature directions are present and accessible from the learned representation.

## IV. Local gradient, curvature and global convexity are different

Two linear residual blocks:

$$f(a,b;x)=(1+a)(1+b)x.$$

For $x=1$, target 0 and squared error $L=\frac12(1+a)^2(1+b)^2$, Hessian at $a=b=0$:

$$H=\begin{pmatrix}1&2\\2&1\end{pmatrix},\quad \mathrm{eig}(H)=\{3,-1\}.$$

Not even locally convex at that point. Residual connections can improve conditioning and avoid severe gradient decorrelation in some regimes, but not guarantee global convexity. Hessian eigenvalues also change under parameter reparameterization, so naive cross-topology 'flatness' comparisons require care.

For general vector-output models under squared error:

$$H_L=J_\theta^\top J_\theta+\sum_j (f_j-y_j)H_\theta f_j.$$

For cross entropy, see the next document: the curvature has a positive semidefinite Gauss-Newton-like term *plus* a model second-derivative term. If the logit function is affine in parameters, the softmax CE objective is convex in those parameters; with fixed features this does not allow formation of new feature detectors.

## V. Long-path functional depth is not automatically deeper abstraction

Even $W_3W_2W_1x$ is a linear map in $x$. A high learned-path degree need not induce nonlinear interaction in *input variables*. Conversely a learned nonlinear operation can encode XOR or other complex functions in one stage if its internal architecture permits it.

Therefore use an **exact linear expansion as an instrument calibration** and only use nonlinear graphs and controlled functions to study circuit formation. The function space might be identical between architectures while the optimization trajectories differ.

## VI. Exact learning objectives and sampling scope

**Primary objective** for vocabulary $\mathcal V$ is

$$p_\theta(j|x)=\frac{\exp(z_{\theta,j}(x))}{\sum_{k\in\mathcal V}\exp(z_{\theta,k}(x))},\qquad
L=-\log p_\theta(y|x).$$

A binary problem should be implemented as a 2-logit softmax (or equivalent logistic margin) with comparable conventions. CE is not a metric of circuit utility. If $L(x,y)$ decreases on an *identical input/model checkpoint*, $p(y|x)$ increases, but $y$ need not become the top-ranked token; later training may reverse the change, and other contexts may not benefit.

When analyzing logits, be aware of **softmax gauge freedom**: adding the same scalar to every vocabulary logit leaves predictions unchanged. Functional atlases should use canonical logit contrasts (e.g. centered logits) or probabilities to avoid measuring irrelevant common shifts.

## VII. Math oracles required in CI

1. Evaluate each linear graph numerically and compare to symbolic polynomial on random inputs/weights.
2. Verify $J_{\theta=0}$ for graphs A/B and minimum displacement asymptotics.
3. Verify ReLU absolute-value and zero-map constructive examples, including domain/activation conditions.
4. Verify bottleneck Jacobian ranks numerically for generic random matrices.
5. Verify Hessian eigenvalues $3,-1$ by autograd and finite differences.
6. Verify one-step projection result and invariance of orthogonal inputs.
7. Verify logit-shift invariance of CE/functional metrics.

These examples are scientific fixtures; preserve them even after developing a larger framework.


## VIII. New initial study: same linear function class and tangent geometry, different routing curvature (19 Oct 2026)

The early scalar examples show that edges can *change initial first-order learnability*. Our revised first training study removes that explanatory shortcut: it pairs two graphs whose initial first-order geometry is **identical**, to ask whether higher-order parameter interactions alone have observable learning consequences.

Let $A,B\in\mathbb R^{d\times d}$ be two learned matrices, $x\in\mathbb R^d$:

$$h_P=(I+A+B)x,\qquad h_C=(I+B)(I+A)x=(I+A+B+BA)x.$$

Both graphs use the same learned matrices once and two fixed additions, and can represent *every* linear map $Mx$ with $A=M-I,B=0$. Under common fixed readout $R$ and bias $c$, set logits $z=Rh+c$ and optimize the **same softmax CE**.

At $A=B=0$:
$$h_P=h_C=x;\quad
Dz_P[\Delta A,\Delta B]=Dz_C[\Delta A,\Delta B]=R(\Delta A+\Delta B)x.$$

Thus initial logits, parameter Jacobians, CE gradients and empirical tangent kernels match. But:

$$z_C-z_P=RBAx,$$
$$D^2_{BA}z_C[\Delta B,\Delta A]=R\Delta B\Delta A x,\qquad D^2_{BA}z_P=0.$$

Therefore the **full CE Hessians generally differ** through $\sum_j(p_j-e_{y,j})H_\theta z_j$, even at identical initial logits/NTK. Same first SGD update from zero means post-step divergence $R(\Delta B)(\Delta A)x=O(\eta^2)$ for fixed first gradients.

For a scalar toy ($x=1$, logits $(h,0)$):
$$h_P=1+a+b,\quad h_C=1+a+b+ab.$$
With target token 1, CE margin derivative at zero is $r=\sigma(1)-1<0$. SGD first step $a_1=b_1=-\eta r$ yields $h_C-h_P=\eta^2r^2>0$, further supporting target 1. With target token 0, $r=\sigma(1)>0$ and the same *positive* difference **opposes the desired reduction** of the incorrect logit margin. So the second-order effect is not universally good.

**This is an optimization/parameterization experiment, not circuit discovery.** Function classes match *only for linear input-output maps*; these graphs cannot make XOR from raw independent inputs. Exact zero init is a deliberate analytic control that may behave differently from standard random init. Normalization, initialization scales, weight decay and optimizer choice remain confounds to study.

Full protocol: [matched study](15_matched_linear_topology_study.md), [functional measurement](16_functional_learning_and_transfer_measurement.md), [validation tests](12_validation_tests.md). The original Section II scalar first-order-accessibility example remains a distinct diagnostic and should not be discarded.
