# Rung 2 design review 3 (independent, adversarial): revision 3, "the router's training signal"

**Reviewed:** `experiments/routefind/DESIGN.md` revision 3 at `f34b73d`, against the ledger's "Current orientation" (risks 1–7, L27), F28–F34, L25–L27, `route/DESIGN.md` and `route/RESULTS.md`, review 1 of this rung. Code read: `route/estimators.py` (`posterior_counts`, `pf_loss`, `tb_loss`, `ac_loss`), `route/train_route.py` (`estimator_loss`, `program_stats`, the schedule), `route/router.py` (value head), `derive/signals.py`, `derive/search.py`, `make_grids.py`, `analyze.py`.

**Checks run** (CPU, 2 threads, each under 2 minutes; scripts in `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/rev3/`; nothing in the repo was changed):
- `signals_small.py`: `derive/signals.py` at 120 episodes per cell, to spot-check §5.
- `pooled.py`: PF's fixed point **pooled by the features the router can read with one lookup**: answer/key/dead, own pair's flag, "a pair whose value is my key is flagged" (value-slot read), c, h. For a router that fits a fixed target by cross-entropy, the class-normalized posterior counts are the fixed point. The script applies that policy, committed and sampled, to fresh training episodes and to the pp4/6/8 strata, at 60 and 250 episodes. It also computes the same pooled target under the rung-1 readout at a flat policy.
- `exgrad.py`, `exal.py`: the **step-0 descent direction** on per-state logits at a uniform policy, summed per feature class, for EX (rung-1 readout), PF, and EX with PF's answer-only likelihood ("EX-AL").
- `tbfloor.py`: the share of the TB reward on floor routes by k, and how PF's E/L split depends on the prior's step cost.
- The AC16 value head at init (3 seeds, real router).

## Verdict: **do not run as written; revise (revision 4).**

The sampled-signal core is well chosen: IW16, TB16, RL16, AC16 and EI16 on δ = 1 are genuinely uncertain and tied to risk 3. But five problems need fixing before the pilot:
- the arm sold as the causal test (PF, P2) is derivable, and is confounded at step 0;
- the success criterion can be met by an arm that does exact enumeration;
- AC16's critic is not the zero critic the design assumes;
- TB16 is not "PF, sampled";
- P3 and P4, the predictions for risks 3 and 4, are not yet written, so no reviewer has seen them.

Every fix is cheap: no new machinery except two small estimator variants.

---

## Blocking

### B1. Every PF cell is derivable (L27). PF is a check, not the core.

PF minimizes cross-entropy to a *fixed* target. Its fixed point is the target pooled over whatever the router can tell apart. That pooled target can be computed, and I computed it (`pooled.py`):

| cell | the pooled target's argmax at the decisive classes | committed route of the fixed point (P-words in3 / in4 / in5 / in8; all-H) | pp4 / 6 / 8 (committed) | **sampled** shortest-correct (P-words in3 / in4 / in5 / in8) |
|---|---|---|---|---|
| δ1, 16 ans | own-flagged (0,0): SET 0.60 / HOP 0.40; pred-flagged (0,0): SET 0.83; pred-flagged c ≥ 1: PREV 0.63–0.68; unflagged c ≥ 1: HOP 0.63–0.65 | **1.00 / 1.00 / 1.00 / 1.00; 1.00 — program E** | 0.02 / 0.00 / 0.00 | **0.41 / 0.37 / 0.14 / 0.06** |
| δ0, 16 ans | own-flagged (0,0): SET 0.83; own-flagged c ≥ 1: PREV 0.64–0.69; unflagged: HOP | **1.00 everywhere — program L** | **1.00 / 1.00 / 1.00** | 0.37 / 0.29 / 0.16 / 0.10 |
| δ1, 4 ans | same argmaxes (SET 0.58 at own-flagged; PREV 0.51 at pred-flagged c = 1) | **1.00 everywhere — E** | 0.05 / 0.04 / 0.02 | 0.32 / 0.17 / 0.14 / 0.06 |
| δ0, 4 ans | same argmaxes | **1.00 everywhere — L** | 1.00 | 0.32 / 0.15 / 0.14 / 0.06 |

What follows:
- **P2, both halves, is decided in advance.**
  - On δ = 0 the fixed target's argmax is L at every class, with large margins (0.83 / 0.17, 0.64 / 0.27). Cross-entropy to a fixed target is convex in the per-class logits, so there is no basin in the class basis to trap it.
  - "PF learns δ = 0" therefore says only that the router can fit a soft label whose argmax is the oracle. That is supervised learning, not a test of F29.
- **P5's PF line ("lands in E") is derived.** It is also partly a prior artefact. The 0.60 / 0.40 split comes from the flat prior's SET padding. With a prior step cost of 1.5 nats per move it shrinks to 0.54 / 0.46 (`tbfloor.py`).
- **P6's PF line is refuted by the derivation.** The dose leaves every argmax and the committed program unchanged. Only the margins move, for example 0.63 → 0.51 at the PREV class.
- **The δ = 1 PP failure for PF is derived.** SET is never the argmax at c ≥ 1 in any training class. So the first deviation will not be EX's `SET>PREV`; see S3.
- **§5's sentence is wrong.** "Whether a router that pools the target across episodes by state features does better on long chains is not derivable" (§5, line 99): it is derivable, and the answer is yes, at 1.00 committed at k = 8, where the per-episode argmax gives 0.11.

**Fix.**
- Put the pooled-target derivation into `derive/signals.py` and its table into §5.
- Demote PF to a 2-seed check per cell. What the check tests: the router fits the class target, the committed route matches the table above, and the sampled reading matches the derived 0.06–0.41.
- Use the freed runs for B2's arm.

### B2. P2 is not a causal test of F29's mechanism. At step 0, PF *is* EX with a different likelihood.

The step-0 descent on the logits, summed per class at a uniform policy (δ = 0, per 100 episodes; `exgrad.py`, `exal.py`):

| class | EX (rung-1 readout) | **EX-AL** (EX, answer-only likelihood) | PF |
|---|---|---|---|
| own-flagged (0,0): SET | +1.41 | **+17.26** | +17.34 |
| own-flagged (1,1): PREV | +0.93 | **+9.13** | +9.28 |
| own-flagged (1,0): PREV | +1.18 | **+13.27** | +13.48 |
| unflagged (0,0): SET | +0.74 | **+54.73** | +54.11 |
| unflagged (0,0): HOP | +189 | +359 | +367 |

What this shows:
1. **The design's "whose posterior" has zero effect at step 0.** EX-AL and PF agree to within 1 %; EX's policy is the flat prior then. The whole step-0 difference between EX and PF is the likelihood:
   - the rung-1 readout gives "halt or die on a key" 1/16, almost a decoy's expected value;
   - that dilutes the label-carrying signal at flagged states by about 12×;
   - and it removes the SET-padding push at unflagged states (+0.7 vs +54).
2. **EX's own step-0 direction is correct in every class under δ = 0.** SET is up at own-flagged (0,0), PREV up at own-flagged c ≥ 1, and HOP up and PREV down at unflagged c ≥ 1. F29's stray rule (PREV at an unflagged (1,0)) does not appear in the step-0 target in this basis.
   - So the basin is made by the dynamics: parameter sharing of the c and h directions across classes whose gradients differ in size by about 100:1, and the first updates. It is not made by the target's content.
   - A fixed target removes the dynamics and the likelihood dilution together, so a PF escape cannot be attributed to either.
3. **The failure reading is wrong.** "PF fails δ = 0 ⇒ the stray rule is an attractor of the task itself; remedies must change the state or the library" (Reading the outcomes, line 160). A fixed-target fit has no attractor. A PF failure would mean the router cannot represent the class target, or a bug.

**Fix (this is the uncertain, mechanism-relevant cell P2 was meant to be).**
- Add **EX-AL**: the exact current-policy marginal with the answer-only likelihood (readout 0 on keys and ⊥, a log floor). Run δ = 0 at 3 seeds and δ = 1 at 2.
- Run it on CPU first (L25). If EX-AL learns δ = 0 at batch 16 in 1,500 steps, F29's basin is the readout's key mass, and that is the finding.
- Restate P2 as two contrasts:
  - EX vs EX-AL on δ = 0: does the likelihood make the basin?
  - EX-AL vs PF: does self-consistency make it, given the likelihood?
- Predict EX-AL's direction from item 2 above. Its flagged-state signal starts 12× larger, but it also pushes stray SET up at every unflagged state (+55), the ingredient F29 names. So the outcome is genuinely open.
- Replace the two P2 outcome rows accordingly.

### B3. The success criterion lets rung 2 succeed through exact enumeration, and its "deterministic" clause is undefined.

- **Exact enumeration counts as success.** Success is "a signal that never computes the router's own exact posterior" (§1, line 27). PF qualifies, yet it runs forward–backward over all 390 states. Risk 3 is "a router training signal that generalizes **without exact inference**". By B1, PF meets the committed criterion by construction, so as written rung 2's success is derivable.
- **"Deterministic" is undefined.**
  - The checks list (line 151) defines it as "sampled accuracy on P-words ≥ 0.95". In that sense PF fails it by construction: sampled shortest-correct 0.06–0.41, sampled answer accuracy 0.35–0.66.
  - TB16 fails it too, if it converges. Its optimum is the same soft posterior (B5).
  - `analyze.learned` uses the committed `min_acc_P` only.

**Fix.**
- Success = a signal that does not enumerate the state space (IW16, TB16, RL16, AC16, EI16, and PF-IS16 if added; see B5) reaches committed `min_acc_P` ≥ 0.95 on in4 and in8 within the stated budget.
- Report determinism (sampled shortest-correct) as a separate endpoint, with PF's derived level written in.
- Say in §1 that the "generalizes" clause of risk 3 is mostly guaranteed by the Markov router (F30). The one generalization outcome a signal can change is the precedence at the unseen conjunction (S3).

### B4. AC16's critic is not zero-initialized; at init it is a random, structured advantage on SET.

- **What the design says.** §5 (line 113): "AC16 starts from a zero critic, so its first updates are RL16's".
- **What the code does.** `router.py:68–69` builds `vhead = SLinear(R, d, 1)` with the default uniform(±1/√d) init on a LayerNormed input. At init (3 seeds, real router):
  - V has standard deviation 0.32–0.41 over key states, with means −0.04 / −0.14 / −0.88;
  - **V(s, c, h) − V(s, 0, 0) has standard deviation 0.23–0.33.**
- **Why it matters.** The c and h directions are shared across symbols. So the TD(0) advantage of SET at step 0, −λ + V(s, 1, 1) − V(s, 0, 0), is a seed-dependent push of ±0.3 on SET at *every* state. That is 15× λ and a third of the full reward range.
  - This is exactly the "stray-SET mass at init" ingredient of F29 and L26. AC16 seeds could fall into "SET everywhere" or "never SET" for reasons unrelated to bootstrapping.

**Fix.**
- Zero-init `vhead.weight` and `vhead.bias`.
- Add a test that V ≡ 0 at init, and that AC16's first update equals the TD(0) update with V = 0.
- Write down what the first 100 steps teach (L26). With V = 0, TD(0) credits only the final action and charges −λ to every move. Credit reaches SET only after the critic has learned V at the landing state, so AC16 is expected to start *slower* than RL16 (see S1).

### B5. TB16 is not "the fixed-prior posterior, sampled", and the readings built on that are confounded.

- **What `tb_loss` computes.** With log Z estimated as the on-policy sample mean (VarGrad, `estimators.py:290–298`), its gradient is Σ_k (d_k − d̄) ∇log π(r_k) with d = log π − log R. That is REINFORCE on the reward log R(r) − log π(r) with a mean baseline. In expectation it is the gradient of the **reverse** KL(π ‖ p₀(·|y)) (Richter et al. 2020). PF descends the **forward** KL.
- **How it differs from PF beyond "sampled":**
  1. **Projection.** The pooled Markov class without a step index or the label does not contain the per-episode posterior. Forward KL averages (B1's table); reverse KL seeks modes. They can land on different programs and different determinism.
  2. **Entropy regularization.** The −log π term is an entropy bonus, and log π₀ adds 1.39 nats per action. The entropy bonus counters exactly the failure IW16 showed on CPU (identical samples, zero leave-one-out advantage, "SET everywhere").
  3. **Reward shaping.** The floor sets dying or halting on a key at −13.8 and a wrong answer at −6.6. That is the opposite of ELL's stop-over-gamble asymmetry (review 1 B3). For the *target* the floor is harmless: under 0.5 % of reward mass at every k (`tbfloor.py`).
- **Readings this breaks:**
  - "TB16 ≈ PF: the fixed-prior target survives sampling" (line 163) does not follow.
  - "IW16 collapses while TB16 or PF learns ⇒ the current-policy posterior is the fragile part" (line 165) would be explained by the entropy term just as well.

**Fix (pick one).**
- (a) Relabel TB16 as "reverse KL / maximum-entropy RL toward the fixed posterior", and rewrite those two rows.
- (b), recommended: add **PF-IS16**, the sampled forward-KL estimator.
  - Draw K = 16 on-policy samples and take self-normalized weights w̃_k ∝ π₀(r_k) e^{−λ s_k} L_y(r_k) / π(r_k); the gradient is Σ_k w̃_k ∇log π(r_k) (reweighted wake-sleep).
  - With the same likelihood, it differs from IW16 *only* by the factor π₀/π in the weights. That makes IW16 vs PF-IS16 the clean sampled "whose posterior" contrast, as EX-AL vs PF is the exact one.
  - About 20 lines of code, plus an enumeration test at K = 2 like the existing ones.

---

## Should fix

### S1. AC16 vs RL16 (P4): the framing is wrong, and the contrast has two confounds besides bootstrapping.

- **Framing.** §1 (line 23): "setup steps have zero immediate value, so a value signal can credit them only by bootstrapping". This is false. Monte-Carlo returns credit SET with the full downstream return; REINFORCE is unbiased for setup steps.
  - What TD(0) changes is variance and bias.
  - At a one-hop setup distance with 16 RLOO samples per query, the Monte-Carlo variance is already low. The likely result is AC16 ≈ RL16, or AC16 slower (B4's last point). The design's "AC learns where RL does not" is the least likely outcome and has no mechanism behind it.
- **Confound (i): the auxiliary loss.** The value head shares the trunk, so the critic loss shapes the policy's features. 0.5 Σ_t (V − target)² per route is of the same order as the actor loss and shares one gradient-clip budget per run.
- **Confound (ii): the baseline.** RL16's baseline is the per-query leave-one-out mean of 16 returns, which is label-informed. AC16's is a label-free V(x).
- **No δ = 0 partner.** AC16 has 3 seeds on δ = 0 and RL16 none (§6), so an AC16 δ = 0 result cannot be attributed to the value function.

**Fix.**
- Add **AC16-MC** (`gae = 1`: the same head, trunk and loss; Monte-Carlo returns with V as a baseline) on δ = 1 and δ = 0. AC16 vs AC16-MC is the bootstrapping contrast; AC16-MC vs RL16 isolates the baseline and the auxiliary loss.
- Add RL16 on δ = 0 (3 seeds).
- Log actor and critic gradient norms separately.
- State in advance what makes P4 uninterpretable: RL16 at ceiling (criterion within about 1.5× EX) or at floor (stuck all-H or halting) in the pilot. In either case P4 moves to δ ≥ 2 (§11) rather than being read.

### S2. P6's dose is confounded, and its predicted directions contradict the design's own §5.

- **What 16 → 4 answers also changes:**
  - the readout constants: non-answer 1/16 → **1/4**, wrong answer ε/15 → ε/3 (`exec.readout`; EX, IW16, RL16 and AC16 all see these);
  - the label's entropy, halved;
  - the strength of the halt-and-guess baseline.
- **What "spurious" means for this router.** Decoy-specific spurious routes are not *programs* for a Markov router: they differ per episode and pool out (B1: PF's class argmaxes do not move). The one consistent spurious program is **all-H**, whose coincidence rate on P-words goes from 1/16 to 1/4. That is what the dose really manipulates for reward and imitation signals.
- **Direction.**
  - "PF and TB16 degrade most" is refuted for PF by derivation (B1).
  - "EI16 ≥ IW16" runs against §5's own numbers. At the stray policy the beam keeps the oracle on 0.77 of P-words, while 16 samples contain it on 1.00. Min et al.'s regime (consistent spurious programs) is not this one.
- **Refutability.** With 3 seeds per arm, "EI16 ≥ IW16 in learned seeds" is satisfied by a tie (3/3 vs 3/3, or 0/3 vs 0/3), the likeliest outcome. So the prediction cannot be refuted.

**Fix.**
- Dose spurious density at n_ans = 16 with the readout fixed: a decoy-answer-sharing parameter ρ, the probability that the decoy chain ends on the query's answer, at ρ ∈ {1/16, 1/4}.
- Re-derive §5 at that dose.
- Predict directions from the derived numbers (all-H strengthened for RL, AC and EI; averaging signals unchanged at the argmax).
- Use a strict inequality with a stated margin, or report it as exploratory.

### S3. The PP precedence is not a check for non-EX signals. It is the one generalization outcome a signal can change.

- **Why it is open.** F30 says the precedence at the unseen conjunction is "whichever logit the training dynamics made larger". The dynamics differ by signal, and sampled signals visit "flagged with a count" states through stray SETs and train them. So "every learned run on δ = 1 fails PP … E fails by precedence (F30)" (line 149) is not derived for IW16, TB16, RL16, AC16 or EI16.
- **PF's case is derived but differs from EX's.** The conjunction's parent class (own-flagged, c = 1, h = 0) has argmax HOP (0.54), not PREV. So PF fails PP, but its first deviation will not be `SET>PREV`.

**Fix.**
- Move `cf_pending_set` and pp4/6/8 per signal from "checks" to a measured endpoint.
- Predict a direction where one can be derived, as for PF above. This is where risk 3's "generalizes" actually depends on the training signal.

### S4. The determinism check is wrong for PF and TB16.

"Learned runs are deterministic: sampled accuracy on P-words ≥ 0.95" (line 151) fails by construction for PF (0.35–0.66 sampled answer accuracy, B1). It fails for TB16 at its optimum too.

**Fix.** Restate the check per arm, using derived levels for the posterior-matching arms (B3).

### S5. The pilot selects on the main grid's own seed 0.

- `make_grids.py:62` runs the pilot at seed 0. `per_run_init` derives init and sampling from the seed, so `rfmain_*_s0` at the selected lr re-runs the pilot run chosen *because* it did best.
- With count criteria of ≥ 3/5, that seed is 20 % of the evidence, selected on the outcome.
- "Ties → 1e-3" picks an arbitrary lr for arms that fail at every lr; that shows a failure at one lr only.
- The δ = 0 controls run at 1e-3 only (review 1 N6 still open).

**Fix.**
- Use disjoint pilot seeds (for example 100+).
- Break ties among failing lrs by the final `crit_obj` or held-out loss.
- Run the δ = 0 pilot controls at two lrs, or scope δ = 0 claims to "at the δ = 1-selected lr".

### S6. The schedule makes the effective budget about 3,000 steps, and the 4× criterion does not fit.

- **The schedule.** The cosine runs to `steps = 4000` (`train_route.py:408`). The lr factor is 0.15 at step 3,000 and 0.04 at 3,500.
- **The criterion.** §6 says "EX reaches criterion by 600–1,200 steps, so the 4× success criterion fits inside 4,000", but 4 × 1,200 = 4,800.
- **The reference.** `analyze.py` takes EX's reference as the median over this grid *and* rung 1's runs, which ran a 6k cosine schedule.
- So "does not find the program within 4× EX" is partly a schedule artefact for slow sampled signals.

**Fix.**
- Use a constant lr after warm-up for every rung-2 arm, or a 6k cosine with the 4× criterion read inside it.
- Take EX's reference from this grid's schedule only.
- Say whether "4× EX" is paired by seed or uses the median.

### S7. The count thresholds leave gaps, and P5's EI line cannot tell a bias from a coin flip.

- **P1:** ≥ 3/5 vs ≤ 1/5 leaves 2/5 unmapped.
- **P2 on δ = 1:** the falsifier has no count.
- **P5's EI line.**
  - EI's E/L choice at init is tie-breaking by the init logits. Equal-length E and L routes differ only by tiny logit differences, and §5 shows beam MAP = E on 0.10–0.71 of P-words *per seed*.
  - Under a fair coin, "majority L in ≥ 3/5" has probability 0.5, and the falsifier "E in ≥ 4/5" has 0.19.

**Fix.**
- Close the gaps.
- Extend `derive/search.py` to report, per seed, the init beam's MAP = E vs MAP = L share on δ = 1 P-words, and predict EI's program per seed from it. That turns P5's EI line into a sharp, seed-level prediction.

### S8. "Reading the outcomes" is incomplete.

Missing rows, each needing its own reading:
- PF or TB16 committed-correct but sampled-stochastic (the derived case).
- **TB16 learns δ = 1 and fails δ = 0.** This is the informative TB row: self-reference through on-policy sampling traps even a fixed target.
- AC16 slower than or worse than RL16 (B4, S1).
- AC16 and RL16 both fail. Today "AC16 ≈ RL16" merges both-succeed and both-fail, which mean different things.
- RL16 or AC16 stuck in an all-H or halting basin.
- IW16 at 2/5.
- EI16 in E, or split.
- Signals *improving* at the dose: denser label hits at init, §5's 0.13–0.52 vs 0.10–0.32.
- The EX-AL rows (B2).
- Success only through EI16 (hard EM with a state-merging beam: say whether that counts as "without exact inference").

### S9. `set_late` codes E as "everything that is not L", and averages over failed routes.

- **The code.** `program_stats` (`train_route.py:156–179`) scores any SET not at a P-leaving symbol as E. A failing "SET everywhere" run reads as E (set_late ≈ 1/k).
- **Averaging.** It averages over all committed routes that take a SET, failed ones included, and reports only the mean. A run that mixes programs across queries (for example, L where the value-slot read is unambiguous) shows an uninterpretable 0.5.

**Fix.**
- Classify each SET three ways: at the oracle-SET state's position before the P-leaving state (E), at the P-leaving state (L), elsewhere (other).
- Compute only on `min_ok` routes.
- Report the per-query distribution.
- Apply the P5 falsifier for δ = 0 on that basis.

### S10. P3 and P4 are unwritten.

- **What is missing.** The predictions for risks 3 and 4 (lines 143–144) are to be "set after §9, before the pilot". This review therefore cannot check the rung's core.
- **Inputs from this review:**
  - TB16's expected gradient is entropy-regularized REINFORCE on log R (B5), with a strong penalty for dying, so it should not fall into ELL's halting basin.
  - Its fixed point is stochastic (B3).
  - AC16 with a zero critic starts slower than RL16 (B4, S1).

**Fix.** Write P3 and P4 with directions, counts and falsifiers. Have them re-reviewed briefly (design only) before the pilot.

---

## Minor

- **M1. The §2 expressivity table is right.** I checked each entry against the router: one cross-attention layer, no attention among pairs, and the prev slot cannot reach the successor's flag under δ = 0. One caveat: under δ = 1, **9.1 %** of P-leaving states have an ambiguous value-slot read, because a decoy from another chain points at the same key (400 episodes). The extra pointer is never flagged in training, so L stays readable at a lower margin (about 0.5 vs 1). E reads its own pair without ambiguity. Note this as a possible bias toward E in P5.
- **M2. P5's EX line is already observed** (rung 1: RM's committed route = the oracle = E, 10/10). Its PF line is derived (B1). Label both as checks. The core content of P5 is IW16 and EI16; TB16, RL16 and AC16 are exploratory.
- **M3. IW16 on δ = 0 is not derived to fail** ("sees no oracle route", line 64). TB16, RL16 and AC16 see the same 16 samples, and learning needs label-reaching routes with SET/PREV fragments, not the whole oracle route (10–35 % hit the label at init). Call it "expected to fail by analogy with EX (same expected gradient)". Two seeds are fine.
- **M4. §5's numbers reproduce** (`signals_small.py`, 120 episodes).
  - Posterior mass E / L / other / wrong at k = 3: 0.229 / 0.229 / 0.538 / 0.003 (design: 0.23 / 0.23 / 0.53 / 0.00).
  - At k = 5: 0.066 / 0.066 / 0.85 / 0.017 (design: 0.07 / 0.07 / 0.84 / 0.02).
  - At k = 8: 0.004 / 0.004 / 0.91 / 0.085 (design: 0.00 / 0.00 / 0.86 / 0.13; sampling noise).
  - At 4 answers, k = 3: 0.17 / 0.17 / 0.65 / 0.01 (design: 0.18 / 0.18 / 0.64 / 0.01).
  - The target at E's SET state: 0.59 / 0.41 on δ = 1 (design 0.60 / 0.40); 0.88 on δ = 0, falling to 0.77 at k = 8 (design 0.89 → 0.78).
  - Note: on δ = 0 the per-episode argmax policy's *oracle-route* accuracy is 0.00 while its shortest-correct accuracy is 0.92. The per-episode posterior ignores expressivity and prefers E (padding favours the earlier SET). Pooling by router-readable features removes this (B1). Worth one sentence in §5.
- **M5. The TB floor (10⁻⁶) is harmless for the target** (under 0.5 % of reward mass at every k, at both doses). It still sets the on-policy penalty for dying (−13.8 vs −6.6 for a wrong answer): a reward-shaping constant. State it.
- **M6. PF's loss scale differs from EX's.** PF sums posterior counts over every step (about k·log 4 per query) and EX takes log 16 per query, so clipping binds differently. Clip binding is recorded; read speed comparisons with that in mind.
- **M7. For the ledger (not a design change).** In the one-lookup class basis, EX's step-0 descent is correct at every class under δ = 0 (B2, item 2). F29's text, "learned in the first 50 steps from the mass that strays into SET at init", should be qualified: the stray rule is not in the step-0 target. It must come from the parametrized dynamics, such as the shared c and h directions and a 100:1 ratio of gradient sizes across classes. EX-AL is the cheapest test.

## Which core cells are genuinely uncertain (L27)

| cell | status | action |
|---|---|---|
| EX (all cells) | observed in rung 1, or the reference | 2–3 seeds as reference (as now) |
| **IW16 δ1** | uncertain (CPU collapse at batch 16; ≈ EX in expectation) | core |
| PF (all cells) | **derived** (B1) | check, 2 seeds |
| **EX-AL δ0 / δ1** (new) | uncertain; separates likelihood from self-consistency | core (B2) |
| **TB16 δ1, δ0** | uncertain; δ0 tests self-reference through sampling | core, relabelled (B5) |
| **PF-IS16** (new, recommended) | uncertain; the clean sampled "whose posterior" partner to IW16 | core (B5b) |
| **RL16 δ1** (+ δ0, new) | uncertain | core |
| **AC16 / AC16-MC** | uncertain once zero-init (B4, S1) | core |
| EI16 δ1 | success uncertain; program predictable per seed from the init beam (S7) | core, with a per-seed prediction |
| EI16 δ0 | observed on CPU | check |
| dose | uncertain but confounded (S2) | redesign as ρ at fixed readout |

**Budget.**
- PF demoted from 5 + 3 + 3 to 2 + 2 + 2 seeds: −5 runs.
- Additions: EX-AL 5, PF-IS16 8, AC16-MC 8, RL16 δ0 3: +24 runs.
- Net about +19 runs, ≈ +1.9 T4-hours at 0.09 s per step. If the budget is fixed, EX-AL and the zero-init critic come first, then PF-IS16, then AC16-MC.
