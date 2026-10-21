# Phase 5 design — fast and slow paths: time constants and routing of the learning signal

**Status:** design, revision 2 (21 Oct 2026). Nothing has run.
- Revision 1 was reviewed independently by two agents, one scientific and one technical with small numerical checks. Their findings and how each is handled are in the *Change log* at the end.
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
4. **The agreement split is weak at full batch.**
   - With 8 disjoint groups of about 174 examples, per-example idiosyncrasy averages out within groups. A 7-of-8 sign-agreement mask passes by chance 7% of the time.
   - In the asym plateau, measured mask densities were 4–15%, near chance. The consensus part aligned more with memorization than with generalization (cos 0.26 vs 0.07; one seed).
   - So whether "agreement" isolates the generalizing signal here is an open empirical question for 5A, not an assumption.

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
- Share $s_X(\text{gen}) = \text{gen}_X/(\text{gen}_L+\text{gen}_S)$, clipped to [−1, 2], and likewise for mem.
- **Role score** $D_X = s_X(\text{gen})|_{t_\text{end}} - s_X(\text{mem})|_{t_\text{mem}}$. $t_\text{end}$ is the mean of the last 5 evals; $t_\text{mem}$ is the peak of total mem.
- **Single-time co-location** $K_X(t)=|s_X(\text{gen})(t) - s_X(\text{mem})(t)|$, at $t_\text{grok}$ and at $t_\text{end}$. It avoids comparing two different times.
- **Output masking:** the step-weighted integral of the negative part of $\text{gen}_X$ over steps 0–10,000, and the first step after which $\text{gen}_X$ stays positive.
- **Signal masking** (counterfactual): $M(t) = 1-\lVert g_L^{\,\text{S present}}\rVert / \lVert g_L^{\,\text{S ablated}}\rVert$, the drop in L's training gradient caused by S's contribution to the logits. Also its projection on ∇gen.
- **Stability:** post-fit collapses (train accuracy below 0.9 after first fit), post-fit minimum train accuracy, and per-branch weight norms.

**Leakage guard.** Every measurement that uses test-set gradients calls `torch.autograd.grad`, never `.backward()`. A test asserts that `p.grad` is unchanged by measurement. In the current trainer, measurement happens between `backward()` and `opt.step()`.

**Statistics.** Seeds are the unit of replication, paired across arms (bit-identical initial weights and data).
- Paired differences are tested with **exact sign-flip permutation tests**; 1,024 flips at n = 10, or a random 100,000 above that.
- 95% CIs are BCa bootstrap.
- Holm correction within each study.
- **Effect-size floor:** a role or allocation claim needs |ΔD| ≥ 0.10 or |Δs| ≥ 0.10, in addition to significance.
- **Seeds:** $n=\max\big(10,\lceil(2.49\,\hat\sigma_\Delta/0.10)^2\rceil\big)$, capped at 29. Here $\hat\sigma_\Delta$ is the pilot's paired SD of the primary contrast, giving power 0.8 one-sided for ΔD = 0.10.
- **Replication rule** for headline claims (marked ★): a result supported at stage 1 must replicate on fresh seeds, making 29 in total, with one-sided p < 0.05 on the new seeds alone. This replaces "escalate", which was optional stopping.
- **Mediators:** collapses and post-fit minimum train accuracy are reported per arm. Every role claim is re-tested with collapsing runs excluded, and with collapse count as a covariate in a paired rank regression. A claim that does not survive is labeled stability-mediated.

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
- For each split part G, the **share of projection** $\pi_G(t)=\langle -G,\hat t\rangle/\langle -g,\hat t\rangle$ for targets $t = \nabla\text{gen}$ and $t=\nabla\text{mem}_\perp$, the memorization gradient with its ∇gen component removed.
- **Separation:** $\Sigma' = \pi_G(\nabla\text{gen})-\pi_G(\nabla\text{mem}_\perp)$, minus the same statistic for a **matched null**.
  - The null for the agreement split is a random coordinate mask of equal density, redrawn each eval, plus a top-|g| magnitude mask of equal density.
  - The null for the time split is a time-shuffled EMA, built from the same gradients in random order within the window.
  - $\Sigma'$ is about 0 for random splits, and about 0 when ℓ ≈ g (checked analytically by the technical reviewer).
- Also on realized updates: each part's first moment divided by one shared $\sqrt{\hat v}$ from a shadow AdamW.

*c. Masking.* $M(t)$ (signal masking) and output masking, in asym and sym.

**Phases:** fitting ($t<t_\text{fit}$), plateau ($t_\text{fit}\le t<t_\text{grok}$), clean-up.

**Hypotheses.** These are empirical, not derived. Plateau phase, per-seed means pooled over graphs.

- **H5A-1 (agreement separates):** null-corrected $\Sigma'_\text{agree}$ > 0.
  - The reviewer's one-seed check suggests it may be negative in the asym plateau; any outcome is informative.
- **H5A-2 (time does not separate memorization from generalization):** null-corrected $\Sigma'_\text{time}$ at α = 0.98, judged by two one-sided tests with band δ = 0.05:
  - **equivalent** if the 90% CI lies within ±δ;
  - **different** if the 95% CI excludes 0;
  - **inconclusive** otherwise.
- **H5A-3 (signal masking exists):** in asym, $M(t)$ averaged over the plateau is > 0. S's presence reduces L's gradient.
- **H5A-4 (decay drives the plateau):** in the plateau, the decay step's projection share onto $\widehat{\nabla\text{gen}}$ exceeds the gradient step's. *Basis:* observation 5 and §2.3.

**Exploratory:**
- the high-frequency power share in windows *preceding* collapses versus matched collapse-free windows (the collapse windows themselves are excluded);
- results at m = 4 and m = 16, and at α = 0.998.

**Gate for 5C.** A routing split runs as confirmatory in 5C only if its null-corrected $\Sigma'$ has a 95% CI above 0. Otherwise it runs as one exploratory pair (G→L and G→S), and its other 5C arms are dropped.

**Compute:** about 50 runs. Group gradients use `torch.func.vmap`, and the agreement split costs about 2× per eval step only. Roughly 1–1.5 hours.

## 5. Study 5B — different time constants per branch (the direct test)

**Question.** If one branch is fast with short memory and the other slow with long memory, does a division of labor emerge, and without masking? Does it matter whether the slow branch is a chain?

**Two competing predictions, both preregistered:**
- **Complementary learning systems (CLS):** the slow, long-memory branch accumulates the shared structure (generalization), and the fast, short-memory branch imprints and then forgets (memorization).
- **Circuit efficiency** (Varma et al. 2023; Nanda et al. 2023): weight decay selects the efficient, generalizing circuit. The branch under *stronger* decay therefore hosts generalization, and the branch under weak decay keeps memorization.

The two disagree on the direction of the decay effect, so the experiment can discriminate between them.

**Arms.** At the base lr (shared parameters at base settings); 10 seeds, or more per the power rule.

| arm | branch settings (first branch : second branch) | holds fixed |
|---|---|---|
| base | equal | — |
| speed ¼, speed 4 | lr ratio ¼ or 4 (L or A relative to the other) | the time constant τ of each branch (so wd scales with 1/lr) |
| memory ¼, memory 4 | wd ratio 4 or ¼ (so τ ratio ¼ or 4) | lr |
| L-alone, S-alone | the other branch's multiplier set to 0 for all of training (capacity halves; isolation arms) | — |

- In asym, "first branch" is L. In sym, it is A, and sym runs the same arms except the isolation arms.
- Total: asym 7 arms, sym 5 arms, i.e., 12 arms, about 120 runs.
- Notation: "speed 4" means the first branch steps 4× faster at equal memory. "memory 4" means the first branch remembers 4× longer at equal speed.

**Hypotheses** (asym unless stated):
- **H5B-1 (time constants create a division of labor):** at least one of the four time-constant arms changes $D_L$ by at least 0.10 relative to base (an omnibus over 4 arms, then Holm). The **direction** is preregistered as the competing pair above:
  - CLS predicts $D_L$ rises in *speed ¼* and *memory 4*;
  - circuit efficiency predicts $D_L$ rises in *memory ¼*.
- **H5B-2 (direction of the memory effect)★:** sign of $D_L(\text{memory 4}) - D_L(\text{memory ¼})$. Positive supports CLS; negative supports circuit efficiency.
  - Tested two-sided. Its sign is the discriminating result, so it is the headline and must replicate.
- **H5B-3 (masking):**
  - L-alone generalizes faster than L does with S present: L-alone's grokking delay is shorter than base's, and its gen output-masking integral is smaller.
  - The S-alone versus base comparison is reported symmetrically.
  - *Basis:* H5A-3. If S masks L, removing S speeds L.
  - Capacity halves in the isolation arms; this is stated, and the nonlinear tier reports fit.
- **H5B-4 (topology × time constants, EXPLORATORY on this task):** for the arm with the largest |ΔD_L| in asym, compare ΔD in sym (difference-in-differences).
  - The rule here needs no composition, so a null cannot mean "wiring irrelevant". This becomes confirmatory only on a task that defeats short paths.

**Mediation.** Speed arms change each branch's effective step, and so its stability; Study 2 found long chains unstable at larger steps. Every claim must survive the mediator analysis in §3.

**Compute:** about 120 runs × ~20,000 steps at 6 blocks. Roughly 2–2.5 hours.

## 6. Study 5C — routing the preconditioned update (the frequency/agreement idea)

**Question.** With the optimizer's own scaling neutralized, does steering the *direction* of a branch's update toward the slow, or agreeing, part of the signal make that branch the generalizer?

**Routing**, defined to respect §2.2:
- Per branch, keep AdamW moments on the raw branch gradient.
- Split the **preconditioned update** $\hat u=\hat m/(\sqrt{\hat v}+\epsilon)$ into a G-part and a remainder:
  - *time:* $\mathrm{EMA}_\alpha(\hat u)$ and the rest, with α from 5A;
  - *agreement:* the coordinate mask from group gradients, applied to $\hat u$.
- **One-sided:** the receiving branch's update is replaced by its G-part, rescaled to the norm of its own unrouted update (*direction-only*, hard routing). The other branch is untouched.
- Decay is unchanged.
- **Manipulation check (pass thresholds, preregistered):** a median angle of at least 20° between the routed and shadow unrouted update of the receiving branch over the plateau, *and* a rise of at least 0.15 in the G-part's share of update power. Arms failing either are reported and excluded from the hypotheses.

**Arms.** At the base lr; 10 seeds or more per power.
- asym: none; G→L and G→S for each split passing the 5A gate; random-mask → L at matched density (null); top-|g| mask → L at matched density (null, agreement split only).
- sym: G→A for each passing split.
- **Positive control:** uniform Grokfast (λ = 2, α = 0.98) on all parameters of asym. Does it accelerate grokking here, as published?

**Hypotheses:**
- **H5C-1 (routing creates a division of labor):** $D_L(\text{G→L}) - D_L(\text{none}) \ge 0.10$ and > 0 by test.
- **H5C-2 (signal-specific)★:** $D_L(\text{G→L})$ exceeds $D_L$ under both null masks. This is an intersection-union test.
- **H5C-3 (within-graph topology comparison):** $[D_L(\text{G→L})-D_L(\text{none})] - [D_S(\text{G→S})-D_S(\text{none})]$.
  - Two-sided. Same graph, same stability profile.
  - Positive means the long branch makes better use of the routed signal than the short one.
  - EXPLORATORY on this task, for the reason in H5B-4.

**Compute:** up to about 100 runs; agreement arms cost about 2× per step. Up to about 2.5 hours, less if the gate drops arms.

## 7. Study 5D — consolidation from the fast branch into the slow branch (diagnostic)

**Change from revision 1.** Weight-space coupling is dropped. The hidden units of independently initialized blocks are not aligned, so coupling their weights is shrinkage toward unrelated features, not consolidation. Consolidation is done in function space.

**Arms.** asym, at the base lr and at the 5B time-constant setting that maximized $D_L$ if H5B-1 was supported, else base settings; 10 seeds.
- **C0:** no consolidation.
- **C1:** the long branch receives an extra loss $\kappa\lVert z_L(x)-\mathrm{sg}(\bar z_S(x))\rVert^2$ on training inputs.
  - $\bar z_S$ is the per-example EMA (β = 0.99) of the short branch's direct centered logits; sg is stop-gradient.
  - κ ∈ {0.01, 0.1}.
- **C-null:** as C1 at κ = 0.1, but each example's target is the EMA logit vector of a *different* training example (a fixed random permutation). This preserves the target statistics and destroys example correspondence.

**Hypotheses:**
- **H5D-1 (consolidation transfers memorization more than generalization):** for C1 versus C0, the change in $\text{mem}_L$ at $t_\text{mem}$ divided by the change in $\text{gen}_L$ at $t_\text{end}$ is > 1. This is a paired ratio test on log scale.
  - *Basis:* the consolidation target is the fast branch's fit to training labels.
  - *Falsifier:* ratio ≤ 1. Consolidation then preferentially transfers what generalizes, which is the researcher's hoped-for outcome.
- **H5D-2:** C1 differs from C-null. Consolidation's effect is example-specific, not a regularizer.

**Compute:** about 40 runs. Roughly 40 minutes.

## 8. Study 5E — new examples arriving

**Protocol.**
- Start with 348 training pairs (10% of the grid).
- Add 35 new pairs every 500 steps for 30 increments, reaching 1,398 pairs (40.2%) at step 15,000; then train 15,000 more steps.
- Order is fixed per seed. Test set: the complement of the final training set.
- Agreement groups, where used, are redrawn at random each step from the current training set.

**Arms:** asym base; asym at the best 5B time-constant cell (by $D_L$, if H5B-1 is supported); asym with the best 5C routing (if H5C-2 is supported); sym counterparts of the included treatments. 10 seeds.

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
- If neither time constants (5B) nor direction routing (5C) moves $D_L$ by 0.10, co-location is robust to the learning signal in this setting. That supports the scaffold reading: generalization grows in the memorized structure.
- The program would then move to a task that requires composition. There, the H5B-4 and H5C-3 topology comparisons can become confirmatory.

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

- **The shallow-rule task** prevents confirmatory topology claims (H5B-4, H5C-3); a task where short paths fail is needed.
- **Optimizer changes and an added loss** make these diagnostics (rule 1). Each is labeled as such.
- **Shared lr instead of per-arm tuning** (rule 4), justified by Study 4's lr confound.
- **Richness sweep deferred** (rule 4). No topology claim from this phase is made without it.
- **Repo priority:** AGENTS.md lists Phase 0 as current. This phase is the researcher's direction, logged in docs/history.md.
- **Role assignment:** designating L as the slow branch is itself an a priori assignment. The swap arms (speed and memory ratios ¼ vs 4; G→S) test whether roles follow the treatment or the wiring.
- **Graph family:** accumulated states prevent two independent chains, so the topology control is chain versus parallel.
- **Batch regime:** full batch only.
- **One optimizer.** A plain-SGD replication of the 5B base and memory arms is planned as the first follow-up.

## Change log (before any run)

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
