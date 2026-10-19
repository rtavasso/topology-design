# Intellectual history: how and why the research question changed

This document preserves the *reasoning trajectory*, including objections and ideas abandoned or deferred. It is deliberately not rewritten as though the present position had been obvious from the beginning. Coding agents should use this history to avoid accidentally rebuilding a rejected direction.

## Origin: reverse residual connections

The conversation began by asking whether information could flow from deeper to earlier layers via reverse residual/feedback connections. Related mechanisms include top-down feature fusion, recurrent inference, and predictive coding.

**Why we moved away:** recurrent feedback adds inference iterations and a new compute budget; the clean causal question is instead how **feed-forward DAG topology** changes numerical function and learning behavior. Top-down connections are not forbidden forever; a DAG can include a later *computed* state in a subsequent top-down *stage*. But actual cycles/repeated passes are excluded from primary comparisons to avoid a compute confound.

## Pivot 1: neurons as gates vs residual branches as optional updates

Initial intuition: a residual neuron can pass through a value or add a positive/negative correction, whereas a plain nonlinear neuron has to actively learn identity. Therefore routing appears to change the effective degrees of freedom and stability.

Crucial refinement:
- Compare $y=wx$ with $y=x+wx=(1+w)x$. Both represent the same scalar linear functions, but $y=x$ requires $w=1$ versus $w=0$. Residual topology shifts the **origin in parameter/function space**, often favorable under near-zero initialization and weight decay.
- This is not an extra learned degree of freedom. It alters accessibility and regularization preference, not scalar function-class capacity.
- In $y=x+\mathrm{ReLU}(wx)$ for positive $x$, all $w<0$ yield identity; *but* the ReLU branch is then dead on those examples and receives zero gradient. This illustrates a preservation-versus-adaptability tension.
- One unweighted ReLU branch cannot subtract a positive ReLU response; a signed output weight or suitable branch is needed. Avoid repeating the mistaken claim that any negative preactivation creates a negative correction.

A foundational distinction emerged: **expressibility, accessibility from initialization, and robustness once represented are not interchangeable**.

## Pivot 2: what numerical behaviors are *hard* for residual topology?

Under $y=x+F(x)$, preserving input is cheap when $F$ is small. Erasing input requires $F(x)=-x$; strongly contracting/replacing directions may require coordinated updates. For $\|J_F\|_2<1$, $I+J_F$ has positive smallest singular value and is locally invertible, thus cannot collapse a local input direction. This is conditional, not a universal statement that residuals cannot forget.

Counterargument: a linear residual map with $F(x)=Wx$ can erase by $W=-I$. Distance from initialization is not numerical perturbation fragility. A plain linear layer at $W=I$ behaves similarly. Neither topology dominates all tasks.

**Abandoned simplistic framing:** short paths only preserve; long paths only construct; residuals are always good at retention and bad at forgetting. These are candidate tendencies under stated parameterizations, not intrinsic semantics.

## Pivot 3: topology as a polynomial in trainable weights

For a linear DAG whose nodes are learned scalings and fixed additions, each input-output path contributes a product of its weights. The output is a sum of path monomials. Routing can change the lowest nonzero degree of parameter interaction.

An example with identical three learned scalar multipliers and two additions:
- $f_A(x)=(1+bc+abc)x$
- $f_B(x)=(1+c+abc)x$.

At zero weights, both compute identity, but the first has no first-order trainable direction while the second has one. The ability to make a tiny functional change differs despite equal representable scalar linear maps. In the first model, correcting $1\to1+\varepsilon$ near zero may require $O(\sqrt{|\varepsilon|})$ joint displacement; in the second only $O(|\varepsilon|)$ via $c$. Precise parameter norms, signs and initialization conditions matter.

**Resulting direction:** study distributions of **trainable path degrees**, not just raw edge counts or overall depth.

## Pivot 4: fixed graph topology, not learned gates or recurrence

A learned gate $h'=g(h)\odot h+F(h)$ appears to solve conditional retention/forgetting, but adds parameters and changes the operator class. Recurrent feedback increases compute. These confound interpretation.

**Primary experimental choice:** same learned transformations, neuron counts, nonlinearities, loss and optimizer; vary only parameter-free feed-forward edges/addition routing. Count extra addition and memory costs explicitly. Learnable gates and recurrence remain labeled nonprimary comparison baselines, not search candidates.

## Pivot 5: smooth loss vs useful learning from one example

Residuals can improve gradient correlation and effective conditioning; this does *not* make the loss globally convex. Even two scalar residual linear blocks can have negative Hessian eigenvalues for squared error at some settings.

Then we asked what one sample actually does to other samples. For SGD, the parameter gradient $\nabla_\theta L_i$ changes a second loss $L_j$ locally by $-\eta\nabla L_j^\top\nabla L_i$. The logit parameter Jacobian and (finite) neural tangent kernel describe first-order function transfer. This distinguishes fitting one example, retaining earlier examples and learning a rule that transfers.

**Critical correction:** high gradient magnitude and smooth geometry alone do not guarantee good generalization, useful credit assignment, or reusable circuit formation. Fast fitting via an accessible shallow solution may *remove incentives* to construct deeper features.

## Pivot 6: softmax cross entropy and conflicting learning pathways

The training objective was fixed to **softmax CE**, even for toy token prediction. If logits are $z=z_S+z_L$, both branches see $r=p-e_y$ and gradients $J_S^\top r,J_L^\top r$. A short path that pushes target probability close to 1 reduces residual for both, even if the long path individually contains poor evidence. This is loss coupling, not necessarily gradient vanishing through layers.

A short path may fit the training context while the long path lacks a reusable mechanism. But a full CE loss approaching zero means there is little remaining first-order training signal (with bounded Jacobian). One cannot demand guaranteed continued branch learning without altering representational constraints, training signal or objective.

We explored an **exact factorized softmax** using fixed token groups with coarse and within-group decisions: $P(g,i)=P_S(g)P_L(i|g)$. Its CE separates. It can be rendered as one global softmax over $z_{gi}=s_g+t_{gi}-\log\sum_j e^{t_{gj}}$. Mean-centering within-group logits is a cheaper approximate orthogonalization near uniform conditional distributions. Both are mathematically useful examples of loss coupling and its removal.

**Why we rejected this as the primary solution:** choosing token groups a priori imposes the division of labor and restricts the ability of deep representations to revise shallow decisions; extra fixed nonlinear normalization differs from edge-only topology; branch independence is not our scientific goal. Keep as a *diagnostic control*, not the main candidate family. Do not construct a grouped vocabulary benchmark as the flagship project.

## Pivot 7: productive frustration, not forced independence

The user's concept of a machine analogue of *frustration* redirected the discussion. Useful friction is not disagreement for its own sake; it is unresolved training signal that a shallow computation cannot address but a deeper computation could.

For a dataset-wide stack of CE residuals $R$ and Jacobians $J_S,J_L$, a local situation $J_S^\top R\approx0$ but $J_L^\top R\ne0$ represents error outside the shallow trainable directions but accessible to deeper changes. This is a proposed operational condition, not an assertion that $R$ lives in a unique representation subspace. With shared parameters and finite optimization budgets, group definitions and projections are ambiguous.

A shortcut-versus-compositional toy: true binary label $y=x_1x_2$ with $x_2=1$ 90% of the time. A shortcut margin $ax_1$ locally optimizes at $a=\log9$. A deep-capable margin $dx_1x_2$ still has a nonzero derivative driving $d$ up. However, if $d=bc$ and $b=c=0$, its factor gradients vanish: useful unresolved error exists but cannot update the desired computational route.

**Important qualification:** frustration is only useful if it drives the creation of *generalizable* new computation, not oscillation, memorization, or cancellation. The goal is not maximal persistent loss or antagonism. Productive frustration should be temporary and resolved by learned capability.

## Pivot 8: from fastest CE to memorization/generalization mechanisms

The first proposed program emphasized residual-error accessibility, one-example transfer/interference, and circuit formation, with synthetic parity/shortcut and modular addition tasks. Literature on gradient starvation, grokking, lazy-to-rich training, and residual path-length suggests topology can alter both easy fitting and later representation development.

Then the user corrected the target: **CE is the training loss, not the research objective**. We want to discover how topology governs where memorizing versus generalizing computations live and how they interact, potentially yielding future principles for allocating parameters or training.

Classic validation accuracy is weak evidence. We need detailed *measurement of mechanisms within the model* and explanations linked to topology.

## Pivot 9: mechanism discovery should not presume the expected circuit

A researcher who only probes for a two-layer circuit can miss an entirely different learned algorithm. Input-output function structure must be discovered before it is given a mechanistic explanation.

The proposed pipeline therefore became:
1. Characterize unanticipated changes in the function over a controlled probe distribution (exhaustively when possible; Fourier / categorical functional decompositions are examples).
2. Inspect internal representations at several resolutions without declaring any one coordinate or SAE dictionary canonical.
3. Search for causally important subgraphs without assuming the expected feature's location.
4. Test mechanism hypotheses by interventions, including redundancy and cross-subgraph interactions.
5. Track how functional signatures, mechanisms, and learning dependencies evolve over training.

A depth-limited causal-utility profile was proposed, comparing the trained full model to untrained retained subgraphs with only $k$ learned operations per route. This is a *baseline-sensitive operational measure*, not a unique mathematical decomposition of all nonlinear networks.

### Why memorization/generalization isn't automatically a short/long split

The same general circuit can process both an arbitrary stored fact and a reusable rule. Information storage, retrieval and transformation may reside in different parts of a shared graph. We **measure** which behavior depends on what and leave open the possibility there is no clean allocation.

## Pivot 10: scale and superposition

A neuron can be polysemantic because of superposition, distributed bases, feature overlap, sparsity and feature-demand-to-capacity ratio. Larger models may learn *more* features as well as having more dimensions, so comparing “monosemantic neuron count” across width is invalid.

SAE features are also noncanonical; dictionary size and measurement resolution can alter apparent circuit counts. Characterize coordinate-free/basis-robust functional and subspace properties and validate the measurement instruments on synthetic known mechanisms.

We must vary:
- capacity at fixed latent task demand;
- capacity and latent feature demand together at approximately fixed representational pressure;
- composition depth at fixed number of features;
- data support, shortcut reliability and feature sparsity separately.

Do not infer scale laws from two widths or mistake width for a hidden feature-demand variable.

## Pivot 11: topology search *after* mechanistic instrumentation

An initial suggestion was exhaustive/random/local-evolutionary/Bayesian graph search. The final decision is to **delay optimization**.

First validate what the instruments identify, how graph mutations change accessibility, when computations appear, whether inferred causal subgraphs actually matter, and how those effects change with scale. A surrogate predicting early CE could optimize the wrong property. Random search is an important future baseline, not the initial objective.

The earliest valuable discovery is **a reproducible mapping from a specific graph property to a measurable learning mechanism**, not a candidate with a slightly higher score.

## What is still open

- Do topology-specific functional mechanisms exist that are stable across seeds, scales and task families?
- Can we causally separate **acquisition**, **expression**, **retention**, and **transfer** of information?
- Is a learned-computation-depth profile stable under legitimate path interventions and alternate baselines?
- Can low-order path accessibility support high-order feature formation, or does it crowd it out?
- How much of memorization/generalization difference is represented by shared modules versus specialized pathways?
- Can source-agnostic mechanism discovery reveal something other than the circuit our probes were designed to detect?
- Do controlled edge rewires produce predictable changes in circuit formation after matching signal propagation?
- Are proposed 'topology benefits' simply initialization/regularization artifacts?
- What genuinely transfers from tiny DAGs to attention and LLMs?

**Never reduce this trajectory to “train many ResNets and choose the best generalization score.”**
