# Phase 5 design — fast and slow paths: time constants and routing of the learning signal

> **Status (22 Oct 2026): stages B–D on hold after the pilot.**
> - The preregistered learning-rate rule selects an unstable regime.
> - The headline measure (composition at the end of training) saturates.
> - The scale multipliers are ill-conditioned.
>
> See [PILOT.md](PILOT.md) and [docs/LEDGER.md](../../docs/LEDGER.md) (X9, H1–H3, I9). The design below is unchanged, for the record.

**Status:** design, revision 3 (21 Oct 2026). Nothing has run.
- Revision 1 was reviewed independently by two agents, one scientific and one technical with small numerical checks.
- Revision 2 was re-reviewed by a third agent, with a real-network check.
- All findings and how each is handled are in the *Change log* at the end.
- Each study is written at preregistration level, and is frozen as a preregistration when its gate is reached.

**Claim types** (AGENTS rule 5):
- 5A is *measurement*.
- 5B–5E are *mechanism* claims about how the learning signal allocates function across paths.
- Studies that change the optimizer per branch (5B, 5C) or add a loss (5D) are **labeled diagnostics outside the primary parameter-free family** (AGENTS rule 1). Their structural factor, a chain versus a parallel group receiving the treatment, stays parameter-free.

"Low frequency" means the **temporal** frequency of the sequence of gradients over training steps. It is unrelated to the Fourier frequencies of modular addition.

## 1. Why this phase

In Studies 1–4, topology could act only through the gradient. Every graph saw all the information and was first-order identical at initialization. What we measured was topology as a preconditioner for a first-order learner.

**Observations (Studies 2 and 4):**
1. Longer chains have lower stable step sizes.
2. Memorization and generalization **co-locate**. The path that wins carries both. A memorizing block pushes unseen inputs toward wrong answers (gen < 0), then the same block carries the rule.
3. No path cancels another's memorization.
4. The learning rate decides which path wins.
5. After the training set is fit, generalization proceeds on the weight-decay clock: delay × lr is nearly constant within a graph.

**The researcher's idea.** On the residual stream, each new example should be both imprinted (memorized) and mined for what it shares with other examples (generalized), without the two desires masking each other. Memorization is fast and generalization slow, so give them different time constants in different branches: the long branch hears the slow part of the learning signal, the short branch the fast part. Optionally, consolidate what the fast branch learns into the slow one.

**Two forms of masking, both measurable:**
- **Signal masking:** once one path fits an example, the shared error p − e_y for that example vanishes, so other paths stop hearing about it. This is gradient starvation (Pezeshki et al. 2021).
- **Output masking:** a memorizing component lowers the true-label logit on unseen inputs, opposing the generalizing component on the shared readout.

## 2. What the optimizer already does: derived and checked constraints

These facts determine which manipulations can work at all. Numerical checks are in `DESIGN_checks.md`; the reviewers' scripts are cited there.

1. **Linear time filters cannot select.** A linear time-invariant filter $H$ of the gradient multiplies the integrated displacement of *every* non-oscillating component by the same $H(0)$. A memorization burst and a generalization drift keep their ratio. Grokfast-style low-pass emphasis ($g+\lambda\,\mathrm{EMA}(g)$, $H(0)=1+\lambda$) changes when and how smoothly things arrive, not which component wins.
   - A pure high-pass filter ($H(0)=0$) removes the net displacement of all components alike.
   - This is an open-loop statement. Through the shared error, *which branch* fits first can still change. Section 2 is not used to forecast routing outcomes.
2. **AdamW largely undoes pre-optimizer gradient scaling.** Per-coordinate normalization cancels any stable per-coordinate gain.
   - In the reviewers' AdamW simulations, λ = 2 low-pass emphasis raised a burst's displacement by about 9% and a drift's by about 0.3%, not 3×.
   - On the real network (pk012000, one seed, shadow optimizer), pre-Adam routing changed the long branch mostly in *step size*: its update norm rose 2–3× with cosine 0.91–0.96 to the unrouted update.
   - **Two-sided routing reversed the other branch:** $3g-2\ell$ points along $-g$ when the EMA outgrows the gradient. Cosine with the unrouted update was −0.43 to −0.86.
   - **So routing must act on the preconditioned update, be norm-matched, and be one-sided.**
3. **Decoupled weight decay is already a time filter.**
   - Each step multiplies weights by $(1-\mathrm{lr}\cdot\mathrm{wd})$, so any displacement not continually re-supported by the gradient decays with time constant $\tau=1/(\mathrm{lr}\cdot\mathrm{wd})$.
   - In the AdamW toy with wd = 1, a burst's displacement decayed to 0.001 while a drift's stayed at 1.0.
   - This explains observation 5, and it means **per-branch (lr, wd) is the most direct way to give branches different time constants.** Per branch, lr sets the speed and τ = 1/(lr·wd) the memory.
   - The equilibrium weight norm scales as 1/wd and is reported.
4. **The agreement split behaves differently in the two branches.**
   - With 8 disjoint groups of about 174 examples, per-example idiosyncrasy averages out within groups. A 7-of-8 sign-agreement mask passes by chance 7% of the time.
   - Measured densities (asym plateau, one seed):
     - **S:** 4–15%, near chance.
     - **L:** about 30%, ranging 9–59%.
   - The consensus part aligned more with memorization than with generalization (cos 0.26 vs 0.07).
   - So whether agreement isolates the generalizing signal here is an open empirical question for 5A, not an assumption.
5. **Each branch's weight scale is set by its decay.**
   - At AdamW's fixed point the update balances decay ($\hat u \approx \mathrm{wd}\cdot\theta$), so a branch's weight scale goes roughly as 1/wd (cf. Xie & Li 2024).
   - A two-layer ReLU block is degree-2 homogeneous, so its logit scale goes roughly as 1/wd².
   - Any per-branch decay manipulation therefore changes **how much** of the output a branch carries, independently of **what kind** of function it carries.
   - Role measures must be **scale-invariant** (§3).
6. **EMA of the preconditioned update is a second, slower momentum.**
   - At α = 0.98 it stays within 6–11° of the update itself during the plateau, nearly the identity. At α = 0.998 the angle is 28–38°.
   - Its angle spikes (up to 74°) mostly at instability events, so any effect of time routing may be mediated by stability.

## 3. Shared setup

**Task:** modular addition, $p=59$, train fraction 0.4, full-batch AdamW (β = 0.9, 0.98), weight decay 1.0 unless a study varies it. The rule is shallow (stated limitation).

**Graphs.** Six blocks, identical shapes (415,872 parameters), and bit-identical initial weights for a given seed. Verified by the technical reviewer.
- **asym = `pk012000`:** long branch **L = blocks {1, 2, 3}** (chain, depths 1, 2, 3) and short branch **S = blocks {4, 5, 6}** (each reads $h_0$).
  - Branches are defined explicitly. Block 1 belongs to L even though its depth is 1.
  - The branches interact only through the shared embedding, readout and loss: L never reads S's writes, and S reads only $h_0$.
- **sym = `pk000000`:** groups **A = {1, 2, 3}** and **B = {4, 5, 6}**, all reading $h_0$. It differs from asym only in the parent edges of blocks 2 and 3.
- Shared parameters (embeddings, readout) always use the base optimizer settings.

**Shared learning rate.** Each study uses one base lr for every arm. Per-arm lr tuning (AGENTS rule 4) is replaced by a shared lr, because Study 4 showed per-arm tuning confounds which path wins. Primary base lr: 2e-3.
- **Pilot,** shared by 5A and 5B: asym and sym at lr ∈ {1e-3, 2e-3, 4e-3}, 3 seeds, up to 40,000 steps.
- The base lr is the largest lr at which both graphs grok in at least 2 of 3 seeds within 40,000 steps. In a one-seed check, asym had not grokked by step 5,000 at 2e-3.
- The pilot also gives between-seed standard deviations, used for power (below).

**Budget per run:** up to 40,000 steps, continuing 5,000 steps after grokking.
- Evals every 10 steps below step 200, every 25 below 1,000, then every 100.
- Memorization latency (5E) is read every step from the training logits.

**Two measurement tiers** (AGENTS rule 6; a claim must agree across both):
1. **Function-space support, exact and additive** (`grokking.support`, leak-free):
   - gen = mean test-label centered logit;
   - mem = mean train-label centered logit minus gen;
   - computed per branch.
   - With two non-interacting branches, direct and total-effect attribution coincide exactly (checked to 1e-7), so this is one tier, not two.
2. **Nonlinear tier:** test accuracy, train accuracy and test loss of each branch alone (the other branch ablated) and of the full network.

**Derived quantities:**
- **Composition (primary role measure, scale-invariant):** $c_X(t)=\text{gen}_X/(|\text{gen}_X|+|\text{mem}_X|)$, in [−1, 1].
  - Both gen and mem are linear in branch X's logits, so rescaling X's output cancels exactly.
  - +1 means purely generalizing, −1 means purely anti-generalizing; values near 0 mean memorization-dominated.
  - **Role contrast:** $R=c_L-c_S$, the difference in what *kind* of function each branch carries, measured at a single time.
- **Shares** $s_X(\text{gen}) = \text{gen}_X/(\text{gen}_L+\text{gen}_S)$, clipped to [−1, 2], and likewise for mem.
  - The Study-4-style **role score** $D_X = s_X(\text{gen})|_{t_\text{end}} - s_X(\text{mem})|_{t_\text{mem}}$ is secondary. It is scale-dependent (§2.5), so it is never the sole basis of a claim.
- **Single-time co-location** $K_X(t)=|s_X(\text{gen})(t) - s_X(\text{mem})(t)|$.
- **Fixed measurement times:**
  - $t_\text{end}$ is the mean over the last 2,000 steps of a **fixed horizon**, the same for every arm in a study. Runs do not stop early, so composition is defined whether or not a run grokked.
  - $t_\text{mem}$ is the peak of total mem.
  - Grokking delay is reported, censored at the horizon, with censored runs as ties at the top.
- **Output masking:** the step-weighted integral of the negative part of $\text{gen}_X$ over steps 0–10,000, and the first step after which $\text{gen}_X$ stays positive.
- **Signal masking** (counterfactual): $M(t) = 1-\lVert g_L^{\,\text{S present}}\rVert / \lVert g_L^{\,\text{S ablated}}\rVert$, the drop in L's training gradient caused by S's contribution to the logits. Also its projection on ∇gen.
- **Stability:**
  - post-fit collapses (train accuracy below 0.9 after first fit) and post-fit minimum train accuracy;
  - per-branch weight norms and output RMS;
  - path efficiency $\lVert\sum_t u_t\rVert/\sum_t\lVert u_t\rVert$ of each branch's updates.

**Leakage guard.** Every measurement that uses test-set gradients calls `torch.autograd.grad`, never `.backward()`. A test asserts that `p.grad` is unchanged by measurement. In the current trainer, measurement happens between `backward()` and `opt.step()`.

**Statistics.** Seeds are the unit of replication, paired across arms (bit-identical initial weights and data).
- Paired differences are tested with **exact sign-flip permutation tests**; 1,024 flips at n = 10, or a random 100,000 above that.
- 95% CIs are BCa bootstrap.
- Holm correction within each study.
- **Effect-size floor:** a role claim needs |ΔR| ≥ 0.10 (composition contrast), in addition to significance.
- **Seeds:**
  - **20 per arm** for the arms in a study's ★ primary contrast;
  - **10 per arm** otherwise.
  - Fixed now; the pilot has too few seeds to estimate a paired SD.
  - At σ_Δ = 0.15, n = 20 gives power of about 0.8 for ΔR = 0.10 two-sided with an exact test.
- **Replication rule** for ★ headline claims: a stage-1 result that is supported must replicate on **at least 15 fresh seeds**, with p < 0.05 on the new seeds alone and in the same direction. Totals may exceed 29.
- **Families:** each study names its primary ★ contrast and its Holm family below. Exploratory results are never Holm-corrected or claimed.
- **Stability sensitivity analysis:**
  - collapses, post-fit minimum and path efficiency are reported per arm;
  - every role claim is re-tested excluding **whole seed pairs** in which either run collapsed, and with the paired difference in collapse count as a covariate in a rank regression.
  - It is called a *sensitivity analysis*, not mediation, because these variables are post-treatment. A claim that does not survive is labeled stability-dependent.

## 4. Study 5A — where is the generalizing signal, and is there masking? (measurement only)

**Runs.** No intervention.
- asym and sym at the base lr, plus `pk0000`, `pk0120` and `pk0123` from Study 4.
- 10 seeds (500–509).

**Measures** at every eval, per branch (per block for the Study-4 graphs):

*a. Where gen-directed motion comes from.*
- The share of $\lVert\nabla_\theta\text{gen}\rVert^2$ lying in branch parameters versus shared parameters (embeddings, readout).
- Projection shares onto $\widehat{\nabla\text{gen}}$ of three per-step motions: the gradient step, the decoupled-decay step $-\mathrm{lr}\,\mathrm{wd}\,\theta$, and the realized AdamW update.

*b. Splits of the gradient.*
- Time split $(\ell, g-\ell)$ at α ∈ {0.98, 0.998}.
- Agreement split:
  - groups resampled at random every eval;
  - m ∈ {4, 8, 16};
  - primary m = 8, τ = 0.75.
- **Separation statistic** for each split part G (its only denominator is ‖g‖, so it cannot flip sign through a near-zero projection):
$$\tilde\Sigma = \frac{\langle -G_\perp,\ \widehat{\nabla\text{gen}} - \widehat{\nabla\text{mem}_\perp}\rangle}{\lVert g\rVert}, \qquad G_\perp = G - \frac{\langle G,g\rangle}{\lVert g\rVert^2}\,g$$
  - $\nabla\text{mem}_\perp$ is the memorization gradient with its ∇gen component removed.
  - $\tilde\Sigma$ measures whether the part's *deviation from the full gradient* leans toward generalization or memorization.
  - It is 0 when G ∝ g (e.g., ℓ ≈ g), and 0 in expectation for random masks.
  - It is reported minus a **matched null**:
    - for the agreement split, a random coordinate mask of equal density, redrawn each eval, plus a top-|g| magnitude mask of equal density;
    - for the time split, a **matched-angle random rotation**: g rotated toward a random direction orthogonal to it, by the measured angle between ℓ and g, norm-matched.
- **On the preconditioned update.** The same statistic is computed with $\hat u$, a shadow AdamW update on the unrouted trajectory, in place of g, and the û-splits used by 5C in place of G. This is what 5C's gate uses.

*c. Masking.* $M(t)$ (signal masking) and output masking, in asym and sym.

**Phases:** fitting ($t<t_\text{fit}$), plateau ($t_\text{fit}\le t<t_\text{grok}$), clean-up.

**Hypotheses.** These are empirical, not derived. Plateau phase, per-seed means pooled over graphs.

- **H5A-1 (agreement separates):** null-corrected $\tilde\Sigma_\text{agree}$ > 0, on the gradient.
  - A one-seed check suggests it may be negative in the asym plateau; any outcome is informative.
- **H5A-2 (time does not separate memorization from generalization):** null-corrected $\tilde\Sigma_\text{time}$ at α = 0.998, on the gradient, judged by two one-sided tests.
  - The band δ is half the between-seed SD of the *null* statistic in the same runs, so it is calibrated without looking at the effect.
  - **Equivalent** if the 90% CI lies within ±δ; **different** if the 95% CI excludes 0; **inconclusive** otherwise.
- **H5A-3 (signal masking exists):** in asym, $M(t)$ averaged over the plateau is > 0. S's presence reduces L's gradient.
- **H5A-4 (decay drives the plateau):** in the plateau, the decay step's projection share onto $\widehat{\nabla\text{gen}}$ exceeds the gradient step's. *Basis:* observation 5 and §2.3.

**Exploratory:**
- the high-frequency power share in windows *preceding* collapses versus matched collapse-free windows (the collapse windows themselves are excluded);
- results at m = 4 and m = 16, and at α = 0.98.

**Holm family (5A):** H5A-1, H5A-2 (its "different" branch), H5A-3, H5A-4. No ★ contrast; 5A is measurement.

**Gate for 5C**, on the **preconditioned-update** version of $\tilde\Sigma$, the split that 5C actually routes. A split runs as confirmatory in 5C only if its null-corrected $\tilde\Sigma_{\hat u}$ has a 95% CI above 0 in the plateau, pooled over asym's L. Otherwise it runs as one exploratory pair (G→L and G→S), and its other 5C arms are dropped.

**Compute:** about 50 runs. Group gradients use `torch.func.vmap`, and the agreement split costs about 2× per eval step only. Roughly 1–1.5 hours.

## 5. Study 5B — different time constants per branch (the direct test)

**Question.** If one branch is fast with short memory and the other slow with long memory, does it end up carrying a different *kind* of function? Does removing the shared-error masking change L's role?

**Prediction (complementary learning systems, CLS).** The branch with the longer memory accumulates what examples share, so it is generalization-rich. The branch with the shorter memory imprints and forgets, so it is memorization-rich.
- No competing *derived* prediction is offered. Revision 2's circuit-efficiency prediction, that stronger decay hosts generalization, was not derivable: a margin or efficiency argument puts both kinds of function preferentially in the cheaper, low-decay branch, the same direction as CLS.
- A **pure scale account** (§2.5) predicts *no change in composition*, only in shares. The composition measure distinguishes CLS from it.

**Arms.** Base lr $\eta_0$ and base wd = 1, split geometrically around the base so that no branch leaves the range Study 4 explored. Shared parameters stay at base settings.

| arm | L (or A in sym) | S (or B in sym) | what differs | seeds |
|---|---|---|---|---|
| base | η₀, wd 1 | η₀, wd 1 | — | 20 |
| memory 4 | η₀, wd 0.5 | η₀, wd 2 | L remembers 4× longer than S (τ ratio 4) | 20 |
| memory ¼ | η₀, wd 2 | η₀, wd 0.5 | L remembers 4× shorter | 20 |
| memory 4-scaled | as memory 4, plus fixed output multipliers α_L, α_S | | the scale change of §2.5 removed | 20 |
| memory ¼-scaled | as memory ¼, plus fixed output multipliers | | | 20 |
| speed 4 | 2η₀, wd 0.5 | η₀/2, wd 2 | L steps 4× faster at equal τ | 10 |
| speed ¼ | η₀/2, wd 2 | 2η₀, wd 0.5 | L steps 4× slower at equal τ | 10 |
| unmasked-L (asym only) | trained on CE($z_0+z_L$) only, i.e., S's logits detached from L's loss | full loss | removes S's signal masking of L at equal capacity (labeled diagnostic) | 10 |

- **Output multipliers for the scaled arms:** set from the pilot (3 seeds of each unscaled memory arm). α_X is the ratio of branch X's output RMS at step 20,000 in base to that in the unscaled arm; it is fixed before stage 1. Multipliers use the existing per-block `alphas`.
- **sym** runs base, memory 4, memory ¼ and both speed arms at 10 seeds.
- **Horizon:** 50,000 steps, no early stop, for every arm.
  - Decay rates set the generalization clock. Study 4's 6–19 lr·wd·steps means up to about 19,000 steps for the branch with lr·wd = 1e-3.
- **Total:** asym 8 arms (5 × 20 + 3 × 10 = 130 runs); sym 5 arms (50 runs); pilot 18 + 6 = 24 runs.

**Hypotheses** (composition at $t_\text{end}$; asym unless stated):
- **★ H5B-1 (memory length sets the role; CLS):** $R(\text{memory 4}) - R(\text{memory ¼}) > 0$ by at least 0.10. The branch remembering longer is relatively generalization-rich.
  - It must hold with the same sign in the **scaled** pair. Otherwise the result is labeled scale-dependent and not supported.
  - *Falsifier:* ΔR ≤ 0, or it holds only without scale matching.
- **H5B-2 (speed sets the role):** $R(\text{speed ¼}) - R(\text{speed 4})$, two-sided.
  - CLS predicts > 0 (the slower branch is the generalizer).
  - Study 4's lr-migration predicts the faster branch takes the function. That concerns shares, not composition, so it makes no composition prediction.
- **H5B-3 (removing signal masking changes L's role):** $c_L(\text{unmasked-L}) - c_L(\text{base}) > 0$, and L's generalization rises earlier: the first step at which $\text{gen}_L > 0$ persistently.
  - *Basis:* H5A-3. If S's fit starves L of signal, freeing L's error should let it build generalizing function sooner.

**Holm family (5B):** ★H5B-1, H5B-2, H5B-3.

**Exploratory:**
- sym versus asym for the same time-constant arms (the topology × time-constant difference-in-differences). The rule here needs no composition, so a null cannot mean "wiring irrelevant".
- shares and $D_X$;
- grokking delay;
- per-branch norms.

**Sensitivity:** speed arms change each branch's effective step, and hence its stability (Study 2); the stability sensitivity analysis of §3 applies to every claim.

**Compute:** about 200 runs × 50,000 steps at about 30 ms/step on 10 workers, roughly **8 hours**. It splits naturally: pilot plus asym ★ arms first (about 4.5 hours), then the rest.

## 6. Study 5C — routing the preconditioned update (the frequency/agreement idea)

**Question.** With the optimizer's own scaling neutralized, does steering the *direction* of a branch's update toward the slow, or agreeing, part of the signal make that branch the generalizer?

**Routing**, defined to respect §2.2:
- Per branch, keep AdamW moments on the raw branch gradient.
- Split the **preconditioned update** $\hat u=\hat m/(\sqrt{\hat v}+\epsilon)$ into a G-part and a remainder:
  - *time:* $\mathrm{EMA}_\alpha(\hat u)$ and the rest, with α = 0.998 (fixed, see below);
  - *agreement:* the coordinate mask from group gradients, applied to $\hat u$.
- **One-sided:** the receiving branch's update is replaced by its G-part, rescaled to the norm of its own unrouted update (*direction-only*, hard routing).
  - The norm is taken over the branch's flattened parameters.
  - If ‖G-part‖ < 1e-12 (an empty mask), the unrouted update is used for that step.
  - The other branch is untouched. This drops the "short branch hears the fast part" half of the idea, deliberately, because two-sided routing reversed that branch (§2.2).
- Decay is unchanged.
- **Time split:**
  - α = **0.998**, fixed; at 0.98 the EMA of û is nearly û (§2.6).
  - The EMA starts at zero, with bias correction.
  - It is, in effect, a second, slower momentum plus norm matching, and is named as such in reporting.
- **Manipulation check:** the median angle between the routed and the shadow unrouted update of the receiving branch over the plateau must be **at least 20°**. This is the only threshold; under hard routing, a "G-share" is 1 by definition. Arms failing it are reported and excluded from the hypotheses.
- **Per-coordinate amplification:** norm matching amplifies kept coordinates by about 1/√density under the agreement mask (about 1.3× in L, about 2.4× in S). It is reported per arm and covered by the matched-density nulls.

**Arms.** At the base lr; 10 seeds or more per power.
- **asym:**
  - none;
  - G→L and G→S for each split passing the 5A gate;
  - nulls, **for both L and S:**
    - agreement split: random mask at matched density, and top-|û| mask at matched density;
    - time split: matched-angle random rotation, i.e., û rotated toward a random orthogonal direction by the angle the time-routed arm achieved, norm-matched.
- **sym:** none, and G→A for each passing split.
- **Positive control:** uniform Grokfast (λ = 2, α = 0.98) on all parameters of asym. Does it accelerate grokking here, as published?

**Hypotheses:**
- **H5C-1 (routing creates a division of labor):** $R(\text{G→L}) - R(\text{none}) \ge 0.10$ and > 0 by test. Composition contrast; shares are reported.
- **★ H5C-2 (signal-specific):** $R(\text{G→L})$ exceeds R under every matched null for L. This is an intersection-union test.
- **H5C-3 (within-graph topology comparison):** $[c_L(\text{G→L})-c_L(\text{none})] - [c_S(\text{G→S})-c_S(\text{none})]$, each against its own branch's matched nulls.
  - Two-sided. Same graph, same stability profile.
  - Positive means the long branch makes better use of the routed signal than the short one.
  - EXPLORATORY on this task, for the reason given for the sym-vs-asym comparison in §5.

**Holm family (5C):** H5C-1 and ★H5C-2. H5C-3 is exploratory. Seeds: 20 for the ★ arms (none, G→L, L-nulls), 10 otherwise.

**Compute:** up to about 150 runs at the 5B horizon; agreement arms cost about 2× per step. Up to about 5 hours, much less if the gate drops a split.

## 7. Study 5D — consolidation from the fast branch into the slow branch (diagnostic)

**Change from revision 1.** Weight-space coupling is dropped. The hidden units of independently initialized blocks are not aligned, so coupling their weights is shrinkage toward unrelated features, not consolidation. Consolidation is done in function space.

**Arms.** asym, at the base lr and at the 5B memory arm with the larger R if ★H5B-1 was supported, else base settings; 10 seeds.
- **C0:** no consolidation.
- **C1:** the long branch receives an extra loss $\kappa\lVert z_L(x)-\mathrm{sg}(\bar z_S(x))\rVert^2$ on training inputs.
  - $\bar z_S$ is the per-example EMA (β = 0.99) of the short branch's direct centered logits; sg is stop-gradient.
  - κ ∈ {0.01, 0.1}.
- **C-null:** as C1 at κ = 0.1, but each example's target is the EMA logit vector of a *different* training example (a fixed random permutation). This preserves the target statistics and destroys example correspondence.

**Hypotheses:**
- **H5D-1 (consolidation transfers memorization more than generalization):** for C1 versus C0, $\Delta c_L < 0$. L's composition shifts toward memorization.
  - Scale-invariant and defined whatever the signs, unlike revision 2's ratio.
  - *Basis:* the consolidation target is the fast branch's fit to training labels.
  - *Falsifier:* ratio ≤ 1. Consolidation then preferentially transfers what generalizes, which is the researcher's hoped-for outcome.
- **H5D-2:** C1 differs from C-null. Consolidation's effect is example-specific, not a regularizer.

**Holm family (5D):** H5D-1, H5D-2.

**Compute:** about 40 runs at the 5B horizon. Roughly 1.5 hours.

## 8. Study 5E — new examples arriving

**Protocol.**
- Start with 348 training pairs (10% of the grid).
- Add 35 new pairs every 500 steps for 30 increments, reaching 1,398 pairs (40.2%) at step 15,000; then train 15,000 more steps.
- Order is fixed per seed. Test set: the complement of the final training set.
- Agreement groups, where used, are redrawn at random each step from the current training set.

**Arms:** asym base; asym at the 5B arm with the largest R (if ★H5B-1 is supported); asym with the best 5C routing (if ★H5C-2 is supported); sym counterparts of the included treatments. 10 seeds.

**Measures:**
- **Memorization latency:** steps until at least 90% of an increment's new pairs are correct, read every step from the training logits.
- **Generalization progress:** the area under the test-gen curve over steps 0–15,000. Not the sum of window gains, which telescopes to an endpoint.
- **Interference:** the rise in mean training loss on previously seen pairs in the 500 steps after each increment. Loss is used because accuracy sits at its floor.
- **Per-branch attribution** of new pairs' mem and of test gen.

**Probes** at steps 2,000, 8,000, 15,000 and 25,000, on 50 random unseen pairs. Each pair is left out of the test set while it is probed.
- *First-order (exact, additive over branches):*
  - imprinting $\langle\nabla_{\theta_X} z_y(x),\,-\nabla_{\theta_X}\ell_x\rangle$;
  - transfer $\langle\nabla_{\theta_X}\text{gen},\,-\nabla_{\theta_X}\ell_x\rangle$;
  - per branch X.
- *Protocol-matched:* add the pair to the training set, run 50 AdamW steps with the run's optimizer state on a copy, then record the pair's own logit change and the test gen change, per branch.

**Hypotheses:**
- **H5E-1:** a treatment that passed 5B or 5C increases the area under the gen curve relative to asym base, by at least 10%.
- **H5E-2 (no memorization slowdown):** median latency at most 1.25× base. Non-inferiority on the upper CI of the paired ratio.
- **H5E-3 (path specialization in what one example teaches):** at step 15,000, $[\text{L's transfer share} - \text{L's imprinting share}]$ is larger in the treatment than in base. Difference-in-differences.

**Holm family (5E):** H5E-1, H5E-2, H5E-3.

**Compute:** about 40 runs × 30,000 steps. Roughly 1.5 hours.

## 9. Order and gates

**Order:**
1. Pilot (base lr, power).
2. **5A** and **5B** together; 5B does not depend on 5A.
3. **5C**, with arms gated by 5A.
4. **5D**, whose setting comes from 5B.
5. **5E**, whose arms come from 5B and 5C.

**Fixed now:**
- every hypothesis, statistic, threshold and gate above;
- seeds from the pilot by the power rule;
- the base lr by the pilot rule.

**Reading the phase as a whole:**
- If neither time constants (5B) nor direction routing (5C) moves the composition contrast R by 0.10, co-location is robust to the learning signal in this setting. That supports the scaffold reading: generalization grows in the memorized structure.
- The program would then move to a task that requires composition. There, the 5B sym-vs-asym and H5C-3 topology comparisons can become confirmatory.

## 10. Prior art and novelty boundary

- **Grokfast** (Lee et al. 2024): whole-network low-pass gradient amplification accelerates grokking on modular arithmetic. Here it is the positive control in 5C. We do not claim the filter.
- **Gradient agreement and coherence:**
  - Coherent Gradients (Chatterjee 2020);
  - weak versus strong gradient directions (Zielinski, Chatterjee et al. 2020);
  - AND-mask (Parascandolo et al. 2021);
  - Gradient Agreement Filtering (Chaubard et al. 2024).
  
  Agreement filtering at whole-network level reduces memorization. We do not claim the filter.
- **Masking through the shared error:** gradient starvation (Pezeshki et al. 2021).
- **Memorization channels:** example-tied dropout (Maini et al. 2023) routes memorization into designated neurons. It is the closest existing "memorization channel", but uses a learned per-example mechanism; ours uses wiring plus learning-signal routing.
- **Fast and slow learners:**
  - complementary learning systems (McClelland, McNaughton & O'Reilly 1995);
  - fast weights (Hinton & Plaut 1987; Ba et al. 2016);
  - CLS-ER (Arani et al. 2022);
  - Mean Teacher;
  - Lookahead.
- **Grokking mechanisms:** circuit efficiency (Varma et al. 2023), progress measures and clean-up (Nanda et al. 2023), the Slingshot mechanism (Thilak et al. 2022).

**What would be new:**
- per-branch time constants and routing *crossed with parameter-free topology* (chain versus parallel receiving branch);
- exact function-space accounting of where memorization and generalization reside;
- the CLS versus circuit-efficiency discrimination via the sign of the per-branch decay effect;
- per-branch accounting of what one example teaches.

## 11. Known limits and deviations from AGENTS.md, stated before any data

- **The shallow-rule task** prevents confirmatory topology claims (the 5B sym-vs-asym comparison, H5C-3); a task where short paths fail is needed.
- **Optimizer changes and an added loss** make these diagnostics (rule 1). Each is labeled as such.
- **Shared lr instead of per-arm tuning** (rule 4), justified by Study 4's lr confound.
- **Richness sweep deferred** (rule 4). No topology claim from this phase is made without it.
- **Repo priority:** AGENTS.md lists Phase 0 as current. This phase is the researcher's direction, logged in docs/history.md.
- **Role assignment:** designating L as the slow branch is itself an a priori assignment. The swap arms (speed and memory ratios ¼ vs 4; G→S) test whether roles follow the treatment or the wiring.
- **Graph family:** accumulated states prevent two independent chains, so the topology control is chain versus parallel.
- **Batch regime:** full batch only.
- **One optimizer.** A plain-SGD replication of the 5B base and memory arms is planned as the first follow-up.
- **Compute.** The phase totals roughly 18–20 hours of MacBook time at stage-1 seed counts, plus replications. It is planned as several overnight runs, in the order of §9.

## Change log (before any run)

**Revision 3 (21 Oct 2026),** after a second-round review of revision 2, which included a real-network check (pk012000, seed 500).

**Blocking findings and their fixes:**

| Finding | Fix |
|---|---|
| Σ′ blew up in L: its denominator ⟨−g,∇gen⟩ changed sign at 11 of 28 plateau evals | Replaced by $\tilde\Sigma$, whose only denominator is ‖g‖; matched nulls kept (§4) |
| H5B-2's sign could not tell CLS from circuit efficiency, and decay changes branch weight scale (≈1/wd) and logit scale (≈1/wd²) | Headline is now the scale-invariant composition contrast R = c_L − c_S. Scale-matched memory arms must agree in sign. The circuit-efficiency prediction is dropped as underived; a pure scale account predicts no change in composition (§2.5, §5) |
| Time routing at α = 0.98 was nearly the identity (6–11°); the manipulation check was ill-posed; the gate was measured on the wrong object; the time split had no null | α fixed at 0.998. The check is the angle alone. The 5C gate uses the û-split. Matched-angle random-rotation nulls for both splits and both branches (§4, §6) |

**Important findings and their fixes:**
- **The agreement split is not near chance in L.** Density there is about 30%, not 4–15% (corrected §2.4). Matched-density nulls now cover S too, and per-coordinate amplification is reported.
- **Power.** Seeds are now fixed: 20 for ★ arms, 10 otherwise. The replication uses at least 15 fresh seeds, and totals may exceed 29. Holm families and ★ contrasts are named per study; the "omnibus" is dropped.
- **Censoring.** 5B runs to a fixed 50,000-step horizon without early stopping, with composition at fixed times.
- **The isolation arms confounded capacity.** Replaced by an unmasked-L arm (L trained on CE(z₀+z_L)) at equal capacity.
- **The speed arms were under-specified.** Each branch's lr and wd is now stated per arm, split geometrically around the base.
- **5C sym arms had no baseline.** Added sym-none.

**Minor fixes:**
- EMA(û) is named as a second momentum; path efficiency is added.
- Norm-match scope and the empty-mask fallback are specified.
- The shuffled-EMA null (memory-heavy) is replaced by the rotation null.
- The mediator analysis is renamed a sensitivity analysis and excludes whole pairs.
- 5D's ratio is replaced by Δc_L.
- Compute is re-estimated.

**Revision 2 (21 Oct 2026),** after two independent reviews of revision 1.

**Blocking findings and their fixes:**

| Finding | Fix |
|---|---|
| Pre-Adam λ routing is mostly a step-size change; it does not reach the update | Routing moved to the preconditioned update, made direction-only (norm-matched), one-sided and hard (§6) |
| Two-sided routing reverses the other branch | One-sided routing only |
| A constant-k step control cannot match a time-varying norm ratio | Dropped; the norm match makes it unnecessary |
| §2's "no linear filter can exclude" claim was misstated | Restated as "cannot select"; §2 is no longer used to forecast outcomes |
| The Σ statistic was biased (non-zero for random splits) and redundant with gen/mem | Replaced by the share-of-projection Σ′, with ∇mem orthogonalized to ∇gen and matched nulls (§4) |

**Important findings and their fixes:**
- **The idea was not tested directly.** Added 5B, per-branch time constants as the direct test, with CLS and circuit-efficiency predictions that the data can tell apart. Added the counterfactual masking measure (H5A-3) and isolation arms (H5B-3).
- **Branch-level total-effect attribution equals direct attribution here.** A nonlinear tier was added instead (§3).
- **Leakage risk from ∇gen.** Test gradients now use autograd.grad, plus a test (§3).
- **The agreement split may be near chance.** It is now a 5A question with matched nulls and a gate (§2.4, §4).
- **Weight decay as the plateau driver.** Decay is a measured component (H5A-4), and the share of ∇gen lying in shared parameters is reported.
- **Weight coupling of unaligned units.** Dropped; function-space consolidation with a permutation null instead (§7).
- **The interaction test is infeasible on a shallow-rule task.** H5B-4 and H5C-3 are now exploratory.
- **No effect sizes or power.** Added an effect floor of 0.10, a power rule and a fresh-seed replication rule (§3).
- **Percentile bootstrap at n = 10.** Exact sign-flip tests instead.
- **Stability as a mediator.** Mediator analysis added.
- **Missing nulls and controls.** Random and magnitude masks, and a positive control, added (§6).
- **Prior art.** Novelty boundary added (§10).
- **5E measures were ill-posed.** Latency is read every step; progress is an area under the curve; interference uses loss; probes are first-order and protocol-matched; the arithmetic is corrected.
- **path_measure cost at L = 6.** Branch-level ablations only (3 forward passes).
- **"Low frequency" was ambiguous.** Defined as temporal frequency.
