# Training signal and stability: mixture of experts, looped block, loop of learned blocks

**15 Nov 2026. Derivation with a toy; no model trained.** Written for the researcher's question (history Entry 98): formalise why a mixture of experts trains, why a looped transformer trains, and why choosing one learned block per loop pass for the whole state does not train easily. Every numbered statement is checked numerically in [`tests/test_routing_signal.py`](../tests/test_routing_signal.py) under the same label. The toy runs are in [`experiments/theory/routing_toy.md`](../experiments/theory/routing_toy.md) (EXPLORATORY, ten seeds, one step size).

Status of each statement: **identity** (elementary, holds in general), **toy** (exact inside the stated toy), **computed** (numerical, no closed-form proof here), **condition** (stated, not proved).

## 0. The result in one table

| | mixture of experts (per token) | looped block | loop of learned blocks, one chosen per pass |
|---|---|---|---|
| objective | $L(h+\sum_k g_k f_k)$: the mixture is *inside* the loss | $L(F^{n}(h_0))$: no mixture | $\mathbb{E}_{z\sim\pi}\,\ell(z)$ or $-\log\mathbb{E}_{z\sim\pi}\,s(z)$: the mixture is *outside*, over $K^n$ routes |
| what the router's signal is | exact first-order comparison of the experts that ran, on the same input (M2) | — | a comparison of counterfactual returns (H1); it must be enumerated, sampled or searched |
| cost of an exact signal | the experts that ran, added over sites | one pass | $K^n$ routes |
| gain of the router–block feedback loop at the unspecialised point | $\lvert r\rvert/K$, the current residual (M4) | — | $p/K$ for expected loss, $1/K$ for the likelihood, $0$ under a fixed uniform prior (H4, H6) |
| what ends the loop | the residual reaching zero: total amplification $\cosh\lvert r_0\rvert$ (M4) | — | commitment: a deterministic router (H5, H6) |
| does the loss care which assignment | no: any partition fits | — | yes: a program is one of $n!$ among $K^n$, and wrong ones are dead ends (H5) |
| what "unstable" means | nothing harmful: bounded symmetry breaking | step size: sensitivity grows with the loop count (L2, L3) | a plateau of length $\propto 1/p$, then lock-in to whichever asymmetry came first |

In all three cases exact gradient descent decreases its objective. None of them is unstable in the sense of a loss that rises. The differences are in the *gain* of the loop between router and blocks, in what *stops* it, and in whether the place it stops matters.

## 1. Mixture of experts

One token with state $h$. Experts $f_k(h;\theta_k)$, router logits $s(h;\phi)$, a selected set $S$ (top-$k$), gates $g=\mathrm{softmax}(s_S)$:

$$h' = h + \sum_{k\in S} g_k\, f_k(h), \qquad \delta := \partial L/\partial h'.$$

**M1 (identity).** $\;\partial L/\partial\theta_k = g_k\,(\partial f_k/\partial\theta_k)^{\top}\delta$ for $k\in S$, and zero otherwise. An expert learns in proportion to its gate.

**M2 (identity).** For $j\in S$,

$$\frac{\partial L}{\partial s_j} = g_j\Big(\langle\delta,f_j\rangle-\sum_{k\in S} g_k\langle\delta,f_k\rangle\Big),$$

and zero for $j\notin S$. With $a_j:=-\langle\delta,f_j\rangle$ (how much expert $j$'s output would lower the loss, to first order), gradient descent on the logits is $\dot s_j = g_j(a_j-\bar a)$: the replicator equation. The comparison is *exact*, uses only quantities backpropagation already has, and has no sampling noise.

**M3 (identity).** The gated layer is one differentiable function of $(\theta,\phi)$ away from changes of the selected set, so along gradient flow $\dot L=-\lVert\nabla L\rVert^2\le 0$.

**M4 (toy): the feedback loop is driven by the residual and dies with it.** Scalar toy: $\hat y=\sum_k g_k w_k$, $L=\tfrac12(\hat y-y)^2$, $r=\hat y-y$. Gradient flow is $\dot w_k=-g_k r$, $\dot s_j=-g_j r\,(w_j-\hat y)$, and

$$\dot r=-r\Big[\sum_k g_k^2+\sum_j g_j^2 (w_j-\hat y)^2\Big].$$

Around the symmetric point ($g_k=1/K$), write $\sigma_j=s_j-\bar s$, $\omega_j=w_j-\bar w$. To first order $\dot\sigma=-(r/K)\,\omega$, $\dot\omega=-(r/K)\,\sigma$, $\dot r=-r/K$. The asymmetry grows at rate $\lvert r\rvert/K$ — the loop has positive gain — but $r(t)=r_0e^{-t/K}$, so $\int_0^\infty \lvert r\rvert/K\,dt=\lvert r_0\rvert$ and

$$\sigma(\infty)=\sigma(0)\cosh\lvert r_0\rvert \quad (\omega(0)=0),$$

for every $K$. Checked for $K\in\{2,4\}$, $\lvert r_0\rvert\in\{1,3\}$ to 1 %. *Reading:* rich-get-richer is real, its strength is the error still to be fitted, and it stops when the token is fitted. With several tokens the same holds token by token: specialisation proceeds where, and for as long as, there is residual.

**M5 (identity for quadratic $L$): what "soft" costs.** With Hessian $H$ and $\bar f=\sum_k g_k f_k$,

$$\sum_k g_k\,L(h+f_k)\;-\;L\big(h+\bar f\big)=\tfrac12\sum_k g_k\,(f_k-\bar f)^{\top}H\,(f_k-\bar f).$$

A mixture of experts optimises the second term on the left (mixture inside), not the expected loss of a hard choice (mixture outside). They agree to first order in the expert outputs and differ by curvature times the spread of the outputs. For a small residual update to one token the difference is small; for a whole-state step repeated over $n$ passes it is not, and the blended state is one no single route visits.

## 2. Looped block

$h_{t+1}=F(h_t,x;\theta)$ for $t=0,\dots,n-1$, loss $L(h_n)$. Let $J_t=\partial F/\partial h$ and $G_t=\partial F/\partial\theta$ at pass $t$, and $\lambda_n=\partial L/\partial h_n$, $\lambda_t=J_t^{\top}\lambda_{t+1}$.

**L1 (identity).** $\;\nabla_\theta L=\sum_{t=0}^{n-1}G_t^{\top}\lambda_{t+1}$. The signal is exact and dense; there is no choice, so no comparison to estimate.

**L2 (identity).** $\;\dfrac{\partial h_n}{\partial\theta}=\sum_{t=0}^{n-1}\Big(\prod_{u=t+1}^{n-1}J_u\Big)G_t$, hence $\big\lVert\partial h_n/\partial\theta\big\rVert\le\sum_t\rho^{\,n-1-t}\lVert G_t\rVert$ when $\lVert J_u\rVert\le\rho$. The Gauss–Newton curvature of $L$ in $\theta$ scales as the square of this, and gradient descent needs a step below $2/\lambda_{\max}$. So the constraint tightens with the loop count unless the block is contractive: bounded for $\rho<1$, polynomial at $\rho=1$, exponential for $\rho>1$.

**L3 (identity, scalar case).** For $h_{t+1}=a\,h_t+x$: $\partial h_n/\partial a=\sum_t a^{\,n-1-t}h_t$. At $a=1$, $h_0=0$ this is $x\,n(n-1)/2$: sensitivity grows as $n^2$ with no expansion at all, because the same parameter acts at every pass.

*Reading.* The looped block's difficulty is conditioning. One parameter vector receives the sum of per-pass gradients, which may conflict; the total still descends (it is one objective), but anything the objective does not contain — behaviour at loop counts not trained, say — can be lost while it does (F23).

## 3. Loop of learned blocks, one chosen per pass

$h_{t+1}=h_t+f_{z_t}(h_t;\theta_{z_t})$ with $z_t\in\{1,\dots,K\}$ drawn from a router $\pi(\cdot\mid h_t;\phi)$. A route is $z=(z_0,\dots,z_{n-1})$; it succeeds with probability $s(z;\theta)$; $p:=\mathbb{E}_{z\sim\pi}\,s(z)$ and the expected loss is $1-p$.

**H1 (identity).** For the logit of block $j$ at pass $t$ (state suppressed),

$$\frac{\partial p}{\partial s_{t,j}}=\pi_{t,j}\,\big(Q_{t,j}-p\big),\qquad Q_{t,j}:=\mathbb{E}[\,s(z)\mid z_t=j\,],$$

and $\nabla_{\theta_k}p=\mathbb{E}_z\big[\nabla_{\theta_k}s(z)\big]$, which is zero for a block no sampled route uses. The form is M2's, with one change that carries everything: $Q_{t,j}$ is a *counterfactual return* — what the end result would be had block $j$ been applied at pass $t$ — and one forward pass does not contain it. It takes $K^n$ routes to compute, or sampling, or search.

**H2 (identity): reward and likelihood differ by exactly the factor $p$.**

$$\nabla(-\log p)=-\frac{\nabla p}{p},\qquad\text{and at any stationary point of }p:\quad \nabla^2(-\log p)=\frac{\nabla^2(1-p)}{p}.$$

Also $\nabla\log p=\mathbb{E}_{z\sim\mathrm{post}}\big[\nabla\log\pi(z)+\nabla\log s(z)\big]$ with $\mathrm{post}(z)\propto\pi(z)s(z)$, the router's *own* posterior. Expected reward and route-marginal likelihood point the same way; the likelihood divides by the current success probability.

**H3 (identity): what sampling costs.** One correct route among $K^n$, a uniform router, reward $R=\mathbb{1}[z=z^*]$, the score-function estimate of $\partial p/\partial s_{t,z^*_t}$ from one sampled route:

| estimator | signal-to-noise per sample |
|---|---|
| reward, no baseline | $K^{-n/2}\big/\sqrt{1-K^{-n}}$ |
| loss $1-R$, no baseline | $\approx\sqrt{K-1}\;K^{-n}$ |

So of order $K^{n}$ samples with a sensible baseline — one must see a success — and $K^{2n}/(K-1)$ without. The same number governs importance-weighted posterior samples: the effective sample size of $M$ own-policy samples is about $Mp$ (F43). A mixture of experts never pays this, because M2 needs no sampling.

**The conjunctive toy** (for H4–H6). $n$ passes, $K$ blocks. Block $k$ has competences $q_k=\mathrm{softmax}(\theta_k)$ over the $n$ operations; they sum to one, so one block cannot do every job, and — because the block's parameters are the same at every pass — a block used at two passes must split itself between two jobs. The router is one softmax per pass. $s(z)=\prod_t q_{z_t,t}$, so $p=\prod_t\sum_k\pi_{t,k}\,q_{k,t}$. Unspecialised start: $p=n^{-n}$. A program (each pass a different, fully specialised block): $p=1$.

**H4 (computed): the plateau.** The unspecialised point is a stationary saddle. Its escape rate — the most negative Hessian eigenvalue — is

$$-\frac1K\ \text{ for } -\log p,\qquad -\frac pK=-\frac{n^{-n}}{K}\ \text{ for } 1-p,$$

for $(n,K)\in\{(2,2),(3,3),(4,4),(2,4),(3,6)\}$ (and by H2 the ratio is $p$ always). Under expected loss the gain of the loop is the probability that *all the other passes* already succeed, so the plateau lengthens as $n^{n}$. Under the likelihood the gain does not depend on the route length.

**H5 (toy): a reused block is a dead end.** Take $n=K=3$: passes 1 and 2 use block A, which has become half-good at both jobs; pass 3 uses block B; block C is idle. Then $p=\tfrac14$; the router's gradient pushes *away* from C at passes 1 and 2 (C's competence $\tfrac13$ is below A's $\tfrac12$); C's own gradient is proportional to the probability it is chosen, which is near zero, and is the same toward both jobs. Exact gradient ascent on $p$ stays below 0.26 for 3,000 steps. *This is the chicken-and-egg in one line: an untrained specialist loses to a trained generalist, and only being chosen would train it.*

**H6 (toy): the loop that causes lock-in is the one that creates the program.**
- Under a *fixed uniform prior* over routes the marginal success is $\prod_t\frac1K\sum_k q_{k,t}$, which is the same for every doubly stochastic competence matrix — uniform blocks included. The objective is flat in the direction of specialisation: gain zero.
- Under the router's *own* posterior every deterministic router is a fixed point ($\mathrm{post}=\pi$ when $\pi$ is a vertex), whatever the blocks.

So with blocks that start identical there is nothing for a posterior under a fixed prior to prefer; specialisation exists only through the self-reinforcing loop between router and blocks, and that loop has no notion of a *correct* place to stop.

**H7 (condition, not proved): moving targets.** $Q_{t,j}$ depends on the blocks, and with tied blocks one step on $\theta_k$ moves the targets at every pass that uses $k$. The standard sufficient condition for such coupled updates is a separation of rates (router fast relative to blocks, or the reverse). Consistent with F24; not derived here.

## 4. What the toy did, against what I predicted

Ten seeds, $K=n$, step size 0.5, 16 sampled routes per step; "learned" means $p>0.9$ ([routing_toy.md](../experiments/theory/routing_toy.md)).

| signal | $n=2$ | $n=3$ | $n=4$ | when it fails |
|---|---|---|---|---|
| exact, expected loss | 109 steps | 540 | 4,507 | — |
| exact, likelihood (own posterior) | 55 | 99 | 145 | — |
| exact posterior under a fixed uniform prior | 0/10 | 0/10 | 0/10 | stays at $p\approx n^{-n}$, router near uniform |
| sampled reward, no baseline | 109 | 431 | 7/10 (9,290) | $p=0.25$, router deterministic |
| sampled reward, batch baseline | 114 | 572 | 4,664 | — |
| sampled own posterior, graded weights | 54 | 98 | 140 | — |
| sampled reward, binary verdicts | 99 | 460 | 3,211 | — |
| sampled own posterior, binary verdicts | 48 | 99 | 7/10 (271) | $p=0.25$, router deterministic |
| search (best route under current blocks) | 38 | 7/10 (79) | 4/10 (119) | $p=0.25$, router deterministic |

**More starts for the exact signals (added the same day, after an interactive drawing of the toy showed a start from which both exact signals stalled).** Over 200 starts the exact likelihood ended in the reused-block state ($p=\tfrac14$) in 11 at $n=4$ and 1 at $n=3$; exact expected loss in 1 of 40 at $n=4$ and 0 of 100 at $n=3$. So the dashes in the last column mean "none in ten seeds", not "never": the dead end of H5 is reachable by exact gradients from a few per cent of starts, more often for the higher-gain signal.

Predictions were written in the script before each batch.
- **Held.** Likelihood-type signals learn in a number of steps that grows about linearly in $K$ (55, 99, 145); exact expected loss grows about as $1/p_0$ (17–27 steps per unit of $1/p_0$).
- **Failed: the fixed prior.** I predicted the exact posterior under a fixed prior would learn at every $n$. It never did, for the reason in H6, which I derived after seeing it.
- **Failed: sampled signals at $n=4$.** I predicted reward-only would fail from $n=3$ and the sampled own posterior would struggle at $n=4$. With graded rewards both learned; with a baseline, reward-only never lost a run. What lost runs were the *fast* signals — search (6/10 lost at $n=4$), own posterior from binary verdicts (3/10), reward without a baseline (3/10) — and every loss was the same state: a block reused, $p=\tfrac14$, a deterministic router (H5).
- **Partly held: search.** It locks into block-reusing routes as predicted, in fewer runs than predicted (I had 0.5, 0.78, 0.91 lost; observed 0, 0.3, 0.6).

## 5. What the formalisation says

1. **The three differ in where the mixture sits.** Inside the loss, the router's comparison is exact, local and first order, and it is valid because each expert's contribution is a small residual step (M2, M5). Outside the loss, the comparison is between whole routes.
2. **Gain.** The router–block loop has gain proportional to the residual in a mixture of experts (self-limiting), to $p$ under expected loss in a loop of blocks (a plateau exponential in the route length), to a constant under the likelihood, and to zero under a fixed uniform prior.
3. **Stopping.** A mixture of experts stops when the data are fitted, wherever that leaves the assignment, and the loss does not care. A loop of blocks stops at commitment, and most committed states are not programs.
4. **A trade the toy makes visible.** Signals with high gain escape the plateau quickly and commit to the first asymmetry; signals with low gain are slow and commit less often to a wrong one. Among the signals that work from binary verdicts, none was both fast and safe at $n=4$: reward with a baseline was safe and slow (3,211 steps), the own posterior fast (271) and lost 3 of 10; search was fastest and lost 6 of 10. The signals that were fast and lost no run in ten needed either all $K^n$ routes or the graded success probability of each sampled route — and over 200 starts even the exact likelihood lost about one in twenty.
5. **Idle blocks are the mechanism of the dead end** (H5): a block that is not chosen is not trained, so it cannot become worth choosing.

## 6. Streams recombined by a gated sum inside the loop

The researcher's variant (15 Nov, history Entry 100): $K$ streams inside the looped block, each a learned map of the shared state, recombined by a gate and added to the residual stream; the same streams and gate at every pass:

$$h_{t+1}=h_t+\sum_{k=1}^{K} g_k(h_t)\,f_k(h_t;\theta_k),\qquad g=\mathrm{softmax}\big(s(h_t;\phi)/\tau\big),\qquad t=0,\dots,n-1 .$$

This puts the mixture *inside* again, at every pass. Four things change relative to §3, and one does not.

**S1 (identity): the signal becomes exact and cheap.** With $\lambda_{t+1}=\partial L/\partial h_{t+1}$,

$$\frac{\partial L}{\partial s_{t,j}}=g_{t,j}\Big(\langle\lambda_{t+1},f_j(h_t)\rangle-\sum_k g_{t,k}\langle\lambda_{t+1},f_k(h_t)\rangle\Big),\qquad \nabla_{\theta_k}L=\sum_t g_{t,k}\,\big(\partial f_k/\partial\theta_k\big)_t^{\top}\lambda_{t+1}.$$

It is M2 at every pass, with the loss gradient replaced by the signal backpropagated through the remaining passes. No sampling, no counterfactual return: $nK$ stream evaluations give the whole gradient, where the hard choice needed $K^n$ routes. H3 no longer applies.

**S2 (identity): for linear streams and gates that do not depend on the state, the gated sum *is* the average over routes.** If $f_k(h)=A_kh$ and the gates are one vector per pass,

$$h_n=\prod_t\Big(I+\sum_k g_{t,k}A_k\Big)h_0=\mathbb{E}_{z\sim g}\Big[\prod_t (I+A_{z_t})\,h_0\Big].$$

The $K^n$ terms collapse into $n$ sums of $K$. The conjunctive toy is of this kind: its $p=\prod_t\sum_k\pi_{t,k}q_{k,t}$ is already a gated sum. So everything §3–§4 found for the *exact* signals is a statement about this topology: with a likelihood loss, escape rate $1/K$ and no plateau; with a loss linear in success, escape rate $p/K$.

**S3 (computed): otherwise it is not, and the difference has a definite order.** For smooth streams of size $\epsilon$ (and independent gates) the blended state differs from the route average at order $\epsilon^3$ — the first two orders cancel; the gap doubles eightfold when $\epsilon$ doubles (checked: ratio 7–9). The *loss* differs at order $\epsilon^2$ by M5. So the gate's signal is the exact gradient of the blended computation, and a first-order stand-in for "what if stream $j$ were applied alone". For small residual steps the two agree; for steps large enough to count as a primitive they do not, and the gap compounds over passes. The gate learns which *blend* is good.

**S4 (identity): the gate adds a term to the loop's Jacobian.** With $F=[f_1,\dots,f_K]$ and $C=\mathrm{diag}(g)-gg^{\top}$,

$$J_t=I+\sum_k g_k\,\frac{\partial f_k}{\partial h}+\frac1\tau\,F\,C\,\frac{\partial s}{\partial h},\qquad \lVert C\rVert_2\le\tfrac12,$$

and only the spread of the streams' outputs around their gated mean enters ($FC=F_cC$). By L2 the loop's conditioning worsens with this term: in proportion to how *different* the streams' outputs are, how sharp the gate is ($1/\tau$), and how fast the logits move with the state. It vanishes when the gate saturates ($C\to0$) and is largest where the gate is undecided. Specialisation and a sharp gate are what make a program readable, and they are what this term penalises.

**What does not change: the stopping rule and the dead end.**
- The loop between gate and streams is M4's: its gain is the residual, so it stops when the training loss is fitted. If a blend of unspecialised streams fits the data, nothing sharpens the gate and no program appears. Specialisation needs pressure: streams too small to each do everything, or data the blend cannot fit.
- H5 holds unchanged, because the toy is this topology: a stream whose gate is near zero is barely trained, and exact gradients ended in the reused-stream state from about one start in twenty (§4).

**A gate floor does not cure it at small values (toy; prediction failed).** Giving every stream at least $\varepsilon/K$ of the gate, $g=(1-\varepsilon)\,\mathrm{softmax}(s)+\varepsilon/K$, I predicted $\varepsilon=0.05$ would leave at most 2 of 200 starts outside a program at four passes. Observed ([routing_floor.md](../experiments/theory/routing_floor.md)): 15 of 200 with no floor, 11 at $\varepsilon=0.01$, 8 at $0.05$, 0 at $0.2$. My reasoning was that the idle stream would be trained up to parity with the stream doing two jobs. It is not: the floor trains the idle stream a little on *every* pass's job, which makes it a weaker generalist, not a specialist. Only a floor large enough to change the dynamics as a whole removed the dead end, and a floor that large is a blend.

**Summary of the change.**

| | hard choice per pass (§3) | gated sum of streams per pass (§6) |
|---|---|---|
| objective | average of the loss over $K^n$ routes | loss of the blended computation |
| cost of the exact gradient | $K^n$ routes, or sampling, or search | $nK$ stream evaluations |
| gate's signal | counterfactual return | exact, first order, along the blended path |
| equals the route average | by definition | only for linear streams and state-independent gates; else off by $O(\epsilon^3)$ in the state, $O(\epsilon^2)$ in the loss |
| plateau | $\propto 1/p$ under expected loss | none under a likelihood loss (rate $1/K$ in the toy) |
| what ends the loop | commitment | the residual reaching zero — possibly with the gate still soft |
| reused-block dead end | yes | yes, at a few per cent of starts in the toy |
| loop conditioning | as L2 | L2 plus a gate term $\propto$ spread of streams $\times$ sharpness |
| a program to read | yes, by construction | only if the gate ends near a vertex |

## 7. Limits

The toy's router sees only the pass index; success factorises over passes; blocks are competence vectors, not functions; $K=n$; ten seeds and one step size. M5 is exact only for quadratic loss, and the toy cannot show the gap compounding over passes because its success is multiplicative. The toy does not reproduce F40 (reward-only signals collapsing whatever the baseline): here a baseline was enough, so F40 involves something the toy lacks, most likely a state-dependent router. H7 is a condition, not a result. In §6, S2 needs linear streams and gates that do not read the state, and S3 is a computed order, not a bound with constants; what a state-dependent gate does to the correspondence with routes is not analysed. Nothing here is a statement about any trained network.
