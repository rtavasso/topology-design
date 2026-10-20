# Phase 0 — Exact dynamics of linear residual topologies

**Status:** derivations below are proved here or cited; every numbered result has a test in `tests/` (see the table at the end). Results marked *conjecture* or *open* are not proved and are work items, not facts.

**Why this document exists.** The v0 plan ([archive](../archive/v0/docs/15_matched_linear_topology_study.md)) proposed to *measure*, by training, whether the matched pair
$$h_P=(I+A+B)x,\qquad h_C=(I+B)(I+A)x$$
learns differently "beyond its identical initial tangent kernel." Most of that question has a closed-form answer from deep-linear-network theory. We derive it once, test it, and spend empirical effort only on what the theory leaves open (§8).

---

## 0. Setup and notation

- Input $x\in\mathbb R^d$, fixed readout $R\in\mathbb R^{V\times d}$, fixed bias $c$, logits $z=RW(\theta)x+c$, softmax cross-entropy $L$.
- Every graph considered here is **linear in $x$**, so the loss depends on the parameters $\theta$ only through the end-to-end matrix $W(\theta)$.
- Write $G(W):=\partial L/\partial W = R^\top\,\mathbb E\big[(p-e_y)\,x^\top\big]$ (dataset or batch average). For one example, $G=R^\top r x^\top$ with $r=p-e_y$: **rank one**.
- Probe-level transfer: the change of logits at $x_j$ caused by an update.

**Linear parent-selection family** (the v0 "narrow grammar" with linear blocks):
$$h_0=x,\qquad h_i=h_{i-1}+W_i\,h_{k_i},\quad 0\le k_i<i,\qquad W(\theta)x=h_L.$$
Special cases: **parallel** $k=(0,\dots,0)$ gives $W=I+\sum_iW_i$; **sequential** $k=(0,1,\dots,L-1)$ gives $W=\prod_{i=L}^{1}(I+W_i)$. The v0 matched pair is $L=2$.

---

## 1. Proposition 1 — every topology in the family is first-order identical at zero

At $\theta=0$, every $h_i=x$. Perturbing one $W_i$ by $\Delta$ changes $h_i$ by $\Delta x$, and every later state passes that change through to the output with coefficient $I+O(\theta)$. Hence for **every** parent vector $k$:
$$W(\theta)=I+\sum_{i=1}^L W_i+O(\|\theta\|^2).$$
So all $L!$ graphs share, at $\theta=0$: the function, the parameter-to-logit Jacobian, the empirical NTK, and the first gradient step.

**Relation to prior work.** This is the parent-selection phrasing of a known message. Residual blocks have the same conservation laws as the same blocks without the skip (Marcotte, Peyré & Gribonval, ICML 2025), so topology cannot act through local invariants. Cite that as corroboration. Our contribution is the family-wide statement and its use as an experimental control.

**Consequences.**
1. The v0 "carefully matched pair" is not a special construction. Matched first-order geometry holds for the whole family. Topology enters **only through the $O(\|\theta\|^2)$ terms.**
2. If branch weights stay at scale $\epsilon$, topology-dependent effects are suppressed by a factor $\sim\epsilon$. The effect of topology is controlled by **how far training must move the branches** (initial scale, branch multiplier, and how far the target is from identity). This *richness* axis, not width, is the primary experimental variable (see [SPEC §3](SPEC.md)).

## 2. Proposition 2 — the parallel graph is a single matrix

For $W=I+\sum_iW_i$, $\partial L/\partial W_i=G$ for every $i$ at all times. Therefore:
- under SGD, GD or gradient flow, **$W_i-W_j$ is conserved**;
- under Adam/AdamW started from identical optimizer states, the moment estimates are identical, so the Adam *step* is identical and $W_i-W_j$ is again conserved (weight decay aside);
- from equal initialization, the model is exactly one matrix $W$ trained with learning rate $L\eta$ (SGD).

Weight decay $\lambda$ on each $W_i$ shrinks $W_i-W_j$ to zero. With $W_i=(W-I)/L$ it acts as the penalty $\tfrac{\lambda}{2L}\|W-I\|_F^2$: decay is **toward identity**, at $1/L$ the nominal strength.

**Relation to prior work.** This is an exact linear instance of known results. Permutation symmetry drives identical branches toward homogeneous collapse (Ziyin, ICML 2024), and parallel or multi-branch networks convexify (Wang, Ergen & Pilanci, ICLR 2023; Zhang, Shao & Salakhutdinov, AISTATS 2019). Present it as a lemma, not a discovery.

**Implication.** "Same learned-parameter budget" between parallel and composed graphs is nominal. The parallel graph has $d^2$ effective parameters and is a **convex** problem (linear softmax regression).

## 3. Proposition 3 — the sequential graph has a closed-form preconditioned flow

Let $F_i=I+W_i$, so $W=F_L\cdots F_1$.

**Balancedness (Arora, Cohen & Hazan 2018; Du, Hu & Lee 2018).** Under gradient flow, $F_{i+1}^\top F_{i+1}-F_iF_i^\top$ is conserved. (Proof for $L=2$: $\dot F_1=-\eta F_2^\top G$ and $\dot F_2=-\eta GF_1^\top$, so $\tfrac{d}{dt}F_1F_1^\top=-\eta(F_2^\top GF_1^\top+F_1G^\top F_2)=\tfrac{d}{dt}F_2^\top F_2$.) At $W_i=0$ every $F_i=I$, so the network starts balanced and **stays balanced**.

**End-to-end flow.** For a balanced chain (Arora et al. 2018, Thm. 1):
$$\dot W=-\eta\sum_{j=1}^{L}(WW^\top)^{\frac{j-1}{L}}\,G\,(W^\top W)^{\frac{L-j}{L}}.$$
For $L=2$, $\dot W=-\eta\big[(WW^\top)^{1/2}G+G(W^\top W)^{1/2}\big]$. The parallel graph instead has $\dot W=-L\eta\,G$.

**Mode-wise reading.** Write $W=USV^\top$. The component of the update along $u_av_b^\top$ is the parallel update multiplied by
$$\gamma_{ab}=\frac1L\sum_{j=1}^{L}s_a^{2(j-1)/L}\,s_b^{2(L-j)/L}\qquad\big(L=2:\ \gamma_{ab}=\tfrac{s_a+s_b}{2}\big).$$
At $W=I$, $\gamma\equiv1$, consistent with Proposition 1. As learning proceeds, comparing the two graphs **instantaneously at equal $W$** (their trajectories differ, so this is not a whole-trajectory comparison):
- **Amplified directions ($s>1$) are updated faster** — rich get richer.
- **Directions being shrunk ($s\to0$) are updated more slowly.**

**Corollary 3a (no erasure in finite time; orientation is locked).** On the diagonal, $\dot s_a=-L\eta\,s_a^{2-2/L}\,(u_a^\top Gv_a)$ (Arora et al. 2018). With bounded $G$ (CE gradients are bounded for bounded data), $|\dot s_a|\le C\,s_a^{2-2/L}$ with exponent $\ge1$, so no singular value reaches 0 in finite time. Hence under gradient flow from $W=I$ the sequential chain's $W$ stays invertible and $\det W>0$ for all time. **The chain can never fully erase a direction or reverse orientation.** Logit maps that seem to need either can still be reached through the softmax gauge and $\mathrm{null}(R)$ when $V<d$. The parallel graph has no such constraint.

**Relation to prior work.** This is anticipated by [Bartlett, Helmbold & Long (ICML 2018)](https://arxiv.org/abs/1802.06093): under squared loss, gradient descent with identity initialization fails to converge for symmetric targets with a negative eigenvalue, even with regularization toward identity. What may be new here:
- the softmax-CE setting with a fixed readout, where the gauge and $\mathrm{null}(R)$ provide an escape;
- the resulting division of labor in which shortcuts carry orientation (Prop. 7).

This is the first concrete *preserve/destroy* statement the theory gives. In this family, composition makes amplification fast and makes deletion slow and never complete. It is a parameterization effect, not a circuit. It also gives a candidate mechanism for Prop. 7: directions the chain can't easily shrink or flip are left to a shortcut.

## 4. Proposition 4 — per-example transfer is a function of the current function

For the sequential chain, the empirical NTK (logit-space, matrix-valued) is
$$K(x,x')=\sum_{j=1}^{L} R\,P_jP_j^\top R^\top\;\big(x^\top Q_j^\top Q_j\,x'\big),\qquad P_j=F_L\cdots F_{j+1},\ Q_j=F_{j-1}\cdots F_1.$$
Under balancedness $P_jP_j^\top=(WW^\top)^{(L-j)/L}$ and $Q_j^\top Q_j=(W^\top W)^{(j-1)/L}$.

($L=2$ proof: using $F_1F_1^\top=F_2^\top F_2$, $WW^\top=F_2(F_1F_1^\top)F_2^\top=(F_2F_2^\top)^2$, so $F_2F_2^\top=(WW^\top)^{1/2}$ by uniqueness of the PSD square root. Likewise $W^\top W=F_1^\top(F_2^\top F_2)F_1=(F_1^\top F_1)^2$. General $L$ uses the same telescoping; see the balanced-chain lemma in Arora et al. 2018.) So for $L=2$
$$K(x,x')=R(WW^\top)^{1/2}R^\top\,(x^\top x')+RR^\top\,\big(x^\top(W^\top W)^{1/2}x'\big),$$
while the parallel graph has the constant kernel $K_P=L\,RR^\top(x^\top x')$.

**Implications.**
- The parallel graph is a fixed kernel machine for its whole run, i.e. permanently lazy.
- The sequential graph's kernel reweights **input similarity** by $(W^\top W)^{1/2}$ and **output coupling** by $(WW^\top)^{1/2}$.
- Under gradient flow from zero, first-order one-example transfer $T(i,j)$ is **determined by the learned function $W_t$**. It is not an independent observable. The v0 plan's two "primary measurements" (function trajectory and transfer matrix) were largely the same measurement. What *is* independent is the deviation caused by finite steps, Adam, weight decay, or imbalanced initialization (§8).

## 5. Proposition 5 — the first single-example step only rescales the step size

Start at $\theta=0$ and take one SGD step on one example. Every $W_i$ receives the same update $-\eta G$ with $G=R^\top rx^\top$. Since $G^2=\kappa G$ with
$$\kappa=x^\top R^\top r=r^\top Rx=\mathbb E_{v\sim p}[(Rx)_v]-(Rx)_y,$$
every polynomial in copies of $G$ is a scalar multiple of $G$. Hence, **for every parent vector**, $W_1-I=-\eta\,\phi_k(\eta\kappa)\,G$ for a topology-dependent scalar function $\phi_k$ (parallel: $\phi\equiv L$). The logit change at **every** probe $x_j$ is the parallel change times one scalar. For the sequential chain:
$$\frac{\Delta z_{\rm seq}(x_j)}{\Delta z_{\rm par}(x_j)}=\frac{1-(1-\eta\kappa)^L}{L\,\eta\kappa}\qquad\big(L=2:\ 1-\tfrac{\eta\kappa}{2}\big).$$

**Reading.** $\kappa$ compares the target's *bias-free* logit $(Rx)_y$ with its $p$-weighted mean. $\kappa<0$ means the example is already favored. For $0<\eta|\kappa|$ small (precisely, in the step-size range where the ratio stays in $(0,2)$; for $L=2$ that is $\eta\kappa<2$):
- composition takes a **larger** step on examples the model already leans toward ($\kappa<0$);
- it takes a smaller step on examples it gets wrong ($\kappa>0$).

For large $\eta|\kappa|$ the ratio can become negative or overshoot. The much-discussed $BAx$ term adds no new direction for a single example. The only genuinely new first-step structure comes from **minibatches** ($G$ of rank > 1): the cross-example term $\eta^2RG^2x_j$ for $L=2$, plus higher powers of $G$ for larger $L$.

The v0 sign counterexample (target 0 vs 1 in the scalar toy) is the $d=V-1=1$ special case of this formula.

## 6. Proposition 6 — endpoints (what can and cannot differ at convergence)

- **No complete or quasi-complete separation, full-rank inputs, $R$ full row rank.** The minimizer exists. The loss is convex in $M=RW$ and has a unique minimizer up to the softmax gauge $M\mapsto M+\mathbf 1u^\top$. The parallel graph converges to it (convex problem).
  - **Landscape.** For the sequential graph, $\ell(W)=\mathrm{CE}(RW+c)$ is convex and differentiable in $W$. The identity offset is only a translation of parameters. So Laurent & von Brecht (2018; convex losses) applies directly: every local minimum of the factored problem is global. (Kawaguchi 2016 covers squared loss only.)
  - **Remaining gap.** Whether gradient flow from $W=I$ actually *converges* to a global minimum rather than lingering near a saddle. Corollary 3a guarantees invertibility along the path, which removes the usual rank-deficient saddles, but we have not proved convergence.
  - *Status: landscape proved via citation; convergence tested numerically on small problems.* If it holds, **all differences in this regime are transient**.
- **Separable data.** Both diverge in norm. Parallel GD on $W$ is GD on $M=RW$ preconditioned by $RR^\top$, so its directional limit is a max-margin solution in a norm set by $R$ (cf. Soudry et al. 2018). The sequential directional limit is a deep-linear implicit-bias question (Ji & Telgarsky 2019) complicated by fixed $R$. **Open: do not assume the limits coincide.** This is one of the few places where the linear pair could show a lasting difference.

## 7. Proposition 7 (conjecture) — mixed graphs are a linear pathway race

Mixed parent vectors combine chains and shortcuts. For $k=(0,1,0,2)$:
$$h_1=F_1x,\quad h_2=F_2h_1,\quad h_3=h_2+W_3x,\quad h_4=h_3+W_4h_2\ \Rightarrow\ W=(I+W_4)F_2F_1+W_3.$$
This is a depth-3 chain in parallel with a depth-1 shortcut, both receiving the shared signal $G$.

**Exact reduction (proved; tested).** Write $P=(I+W_4)(I+W_2)(I+W_1)$. Then $W=P+W_3$, and $\partial L/\partial P=\partial L/\partial W_3=G(P+W_3)$. The factors $I+W_1$, $I+W_2$, $I+W_4$ appear only in $P$, so they form a balanced chain from zero. By Prop. 3 the dynamics close on the pair:
$$\dot P=-\eta\sum_{j=1}^{3}(PP^\top)^{\frac{j-1}{3}}\,G\,(P^\top P)^{\frac{3-j}{3}},\qquad \dot W_3=-\eta\,G,\qquad G=\nabla L(P+W_3).$$
So the shortcut gets the un-preconditioned gradient, while the chain gets a preconditioner set by **$P$'s** singular values (not $W$'s). By Corollary 3a, $\det P>0$ throughout, so any orientation reversal or erasure of the end-to-end map must come from $W_3$. The same reduction applies to any graph whose learned operators split into balanced chains plus shortcuts.

**Conjecture (testable).** Directions that must be amplified are increasingly carried by the chain. Directions that must be erased or reversed are carried disproportionately by the shortcut, because the chain's preconditioner shrinks as the relevant singular values fall. This is the linear analogue of "short paths destroy, long paths construct."

**Preliminary numerics** (exploratory, [`experiments/phase0/prop7_pathway_probe.py`](../experiments/phase0/prop7_pathway_probe.py); teacher with three amplified and three shrunk modes; 4 data seeds). All four matrices receive the same gradient at zero, so the first-order baseline share for the shortcut is 1/4.
- **Subspace level supports the conjecture.** The shortcut carries 0.24 of the change in the amplified subspace and 0.40–0.42 in the shrunk subspace.
- **Per mode it is non-monotone and seed-stable:** 0.16, 0.25, 0.39 for the amplified modes $s^*=4, 2.5, 1.5$; 0.48, −0.2, 0.44 for the shrunk modes $s^*=0.6, 0.3, 0.1$. In one shrunk mode the shortcut *opposes* the change.

So the simple story is at best a coarse average, and the exceptions are unexplained. Do not cite this as a result until it is derived.

Status: the reduction above is proved. The *conjecture* about which subspace each path carries is not; it requires analysing the coupled $(P,W_3)$ ODE. A plausible source of the per-mode exceptions: $P$ and $W$ have different singular bases, so modes mix, and $\det P>0$ forces $W_3$ to handle any orientation-relevant component. For parent vectors that do not split into chains plus shortcuts, use Marcotte, Gribonval & Peyré (2023) to find all conservation laws. We have not found a published treatment of implicit bias for general linear residual DAGs; if none exists, this is a candidate theory contribution.

## 8. What the theory does **not** cover (the only empirical questions left in the linear setting)

1. **Finite step size.** Balancedness drifts by $O(\eta^2)$ per step. Measure the drift and how much it moves the trajectory away from the closed form.
2. **Adam/AdamW.** Coordinate-wise normalization breaks balancedness. The parallel single-matrix reduction (Prop. 2) still holds.
3. **Weight decay on branches.** Decaying $W_i$ toward zero means decaying $F_i$ toward $I$. Under gradient flow with learning rate $\eta$ this adds $\eta\lambda\,(F_2+F_2^\top-F_1-F_1^\top)-2\eta\lambda(F_2^\top F_2-F_1F_1^\top)$ to the balancedness derivative (finite-difference tested), so **weight decay breaks balancedness** (tested). This differs from the standard deep-linear analysis, which decays factors toward zero.
4. **Small random (imbalanced) initialization.** This is the regime Kunin et al. (2024) and Dominé et al. (2025) show controls lazy vs. rich learning. The initial NTKs then differ at $O(\epsilon)$.
5. **Minibatch cross-example terms** (Prop. 5's $\eta^2RG^2x_j$).
6. **Separable-data implicit bias** (Prop. 6).
7. **General parent vectors** (Prop. 7).

Items 4, 6 and 7 are scientifically interesting. Items 1–3 and 5 are robustness checks.

## 9. Test map

| Result | Test |
|---|---|
| Prop. 1 first-order universality (all parent vectors, $L\le4$) | `test_linear_family.py::test_first_order_universal` |
| Prop. 2 parallel conservation (SGD, Adam) | `::test_parallel_conservation_sgd`, `::test_parallel_conservation_adam` |
| Prop. 3 balancedness drift $\to0$ as $\eta\to0$; end-to-end flow | `::test_balancedness_drift_scales_with_eta`, `::test_end_to_end_flow_matches_arora` |
| Prop. 3 mode-wise factor $\gamma_{ab}$ | `::test_modewise_preconditioner` |
| Prop. 4 closed-form NTK ($L=2,3$) | `::test_closed_form_ntk` |
| Prop. 5 first single-example step = scalar rescale (all parent vectors; chain ratio formula) | `::test_first_step_is_scalar_rescale`, `::test_first_step_chain_ratio` |
| v0 fixtures: $RBAx$ divergence, Hessian differs only in residual term | `::test_bax_divergence`, `::test_hessian_difference_is_residual_term` |
| Corollary 3a $\det W>0$ along chain training | `::test_chain_stays_orientation_preserving` |
| Prop. 6 non-separable endpoint equivalence | `::test_nonseparable_endpoints_coincide` |
| Prop. 7 exact reduction to $(P,W_3)$ flow | `::test_mixed_graph_reduces_to_chain_plus_shortcut` |
| §8.3 weight-decay balancedness derivative; breaks balancedness | `::test_weight_decay_balancedness_derivative`, `::test_weight_decay_breaks_balancedness` |
| v0 scalar fixtures (polynomials, Hessian {3,−1}, ReLU $\lvert x\rvert$, bottleneck rank, projection, gauge, shortcut toy, product starvation, $\lVert r\rVert$ bound) | `test_scalar_fixtures.py` |

## References

Arora, Cohen & Hazan (ICML 2018), *On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization*. Du, Hu & Lee (NeurIPS 2018), *Algorithmic Regularization in Learning Deep Homogeneous Models*. Kawaguchi (NeurIPS 2016). Laurent & von Brecht (ICML 2018). Soudry et al. (JMLR 2018). Ji & Telgarsky (ICLR 2019). Kunin et al. (NeurIPS 2024). Dominé et al. (ICLR 2025). Marcotte, Gribonval & Peyré (NeurIPS 2023). Full annotated list: [literature.md](literature.md).
