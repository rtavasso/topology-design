# Rung 2 review 4 (pre-pilot re-review, independent, adversarial): revision 4

**Reviewed:** `experiments/routefind/DESIGN.md` revision 4 at `6028e8e`: §§1–9, §15, and P2–P6 in particular. Read against review 3 (`review3_design.md`, `review3_code.md`). Code: `git diff f34b73d HEAD -- src tests experiments/routefind`, covering:
- `estimators.py`: `canonical_routes`, `arrival_halt_loss`, `pfis_loss`, `ac_loss(actor_scale)`;
- `train_route.py`: `estimator_loss` for exal / pfis / tb / ac, `program_stats`, `critic_warmup`, `evaluate`;
- `router.py`: the value head;
- `data.py`: `ans_pool`;
- `make_grids.py`, `analyze.py`, and both grids.

**Checks run.** All on CPU with 2 threads, each under 3 minutes. Scripts are in the session scratchpad under `review4/`. No repo file was changed except this one.
- `canon.py`: `canonical_routes` and `arrival_halt_loss` on six hand-built routes.
- `pfis_consist.py`: PF-IS16's estimator (canonical routes plus the halt term) at large K (10 × 4,000 samples) against exact PF's gradient, using random per-state logits on two episodes.
- `halt_leak.py`: gradient norms per estimator at init for the real router (3 seeds, batch 16). Also, the direction in which the local halt loss moves HALT at key states.
- `halt_dyn.py`: 50 training steps (batch 16, lr 1e-3, the real warm-up schedule) of PF-IS16 with the halt term, PF-IS16 without it (monkeypatched), and RL16. It tracks π(HALT | key), π(HALT | answer) and clipping.
- `tbfloor.py`: TB16's gradient norm at init against the reward floor, and the floor routes' share of the residual.
- The revision-4 unit tests (12) and the trainer smoke tests for exal / pfis / tb / ac / pf (5) all pass.
- Both grids, regenerated in memory from `make_grids.py`, are identical to the committed JSON.

## Verdict: **launch the pilot.** Fix B1–B3 before the main grid. B3 is a two-line logging change, so make it before the pilot.

The code changes made after review 3 are correct:
- The canonical route without a HALT term is the right objective, not just a choice that happened to work on CPU (issue S5c).
- PF-IS16's gradient on key states converges to exact PF's.
- The critic is zero and detached, and the policy init is paired with RL16's.
- The grids match §6.

None of the problems below affects lr selection. They are in the preregistration and the analysis:
- P3's falsifier fires automatically when both arms fail, and the CPU runs make that the likeliest outcome.
- P4's ACMC16 falsifier is at noise level with 3 vs 5 seeds.
- The determinism endpoint that §4, §5.2 and §7 rely on is not recorded, and since no checkpoints are kept it cannot be recovered after the runs.

---

## What was verified (passes)

1. **`canonical_routes` on hand cases** (`canon.py`; T = 6, an answer at symbol 51, dead at 64):

   | route | canonical actions | moves | arrival |
   |---|---|---|---|
   | HOP, HOP(→answer), HALT | HOP, HOP | 2 | answer |
   | HOP, HOP(→answer), HOP, SET (no-ops), HALT | HOP, HOP (identical to the row above) | 2 | answer |
   | HOP, SET, HALT on a key | HOP, SET, **HALT kept** | 2 | none (−1) |
   | six HOPs, arriving at the answer exactly at T | all six | 6 | answer (state at T) |
   | six HOPs, ending at T on a key | all six (forced halt, no HALT term) | 6 | none |
   | PREV(→dead), no-op HOPs, HALT | PREV | 1 | dead |

   - `arrival_halt_loss` equals the hand mean of −log π(HALT | arrival) over the four arrived routes, to 10⁻⁷.
   - `route_logprob` of each canonical route is the prefix log-probability.
   - π₀ is finite on every canonical route, because no action at a terminal state survives the cut.
   - The conventions agree with `posterior_counts`: π₀(HALT | key) = 1/4 is kept for a halt on a key, and there is no HALT term at T.
2. **The no-HALT canonical route is the correct objective.** The canonical route's probability under π is the marginal over everything the router does after arriving. That marginal is π(prefix), because the continuations sum to 1. The first version scored π(prefix)·π(HALT | arrival), which is not the probability of any event. p₀ already halts on arrival with probability 1, so p₀ is unchanged by the cut.
   - TB16 is therefore exactly the VarGrad reverse KL between π's push-forward onto canonical routes and p₀, as §3 names it.
   - PF-IS16 is the canonical forward KL, plus a halt term that is weighted differently from PF's (S5).
3. **PF-IS16 converges to PF on key states** (`pfis_consist.py`). Comparing the key-state gradient of the large-K PF-IS16 estimator with exact `pf_loss`:
   - cos 0.9993 with relative error 0.036 (k = 2, T = 5);
   - cos 0.9957 with relative error 0.093 (k = 3, T = 6), where ESS/K is only 0.009.

   On terminal states the canonical part gives zero gradient, and the halt term supplies it (norm 0.72–0.84 against PF's 0.51–0.65).
4. **`pfis_loss`** gives zero weight to routes ending on a key or dead. Queries with no finite weight give no gradient and are excluded from the normalizer.
5. **The value head and critic warm-up.**
   - V = 0 at init, and the head reads the trunk through a stop-gradient.
   - The policy parameters equal RL16's at a seed (test).
   - With `actor_scale = 0` the policy receives zero gradient. Adam with a zero gradient moves nothing; decoupled weight decay over 200 ramped steps is about 4·10⁻⁴ relative, which is negligible.
6. **`program_stats`.** Early excludes late, and classification uses only shortest-correct routes. This matches §2.
7. **Grids** (`make_grids.py`, regenerated in memory, identical to the JSON).
   - The pilot has 26 runs, all at seed 100: 7 candidates × {3e-4, 1e-3, 3e-3} on δ = 1, plus EX-AL, PF-IS16, TB16, RL16 and AC16 on δ = 0 at 1e-3.
   - The main grid has 91 runs at seeds 0–4, 5,000 steps, `per_run_init` on, and `ans_pool` ∈ {0, 4}. Per-cell counts equal the §6 table.
   - The AC arms: ACMC16 has gae 1.0, AC16 0.9 and AC16T0 0.0, all with `critic_warmup` 200 and `value_head` on. EI16B has `warm_steps` 1000.
   - Names are unique. The size is 11.4 + 3.2 T4-hours, as §6 states.
8. **The success clock** in `analyze.success` is learned plus a confirmed criterion ≤ 3 × the median EX criterion on the same cell, from this grid only, as in §1.

## Review 3 items: resolved, partly resolved, or open

| item | design | code | status |
|---|---|---|---|
| B1 PF derivable | PF is a check; §5.2 table | `derive/pooled.py` | resolved. But the CPU result contradicts the derived E program, and the reading sends that outcome to "investigate" (S2). |
| B2 EX-AL | P2 restated | `exal` with the answer-only readout and a 10⁻³⁰ floor | resolved |
| B3 success | candidates, committed route | `success()` | resolved. The one-sentence F30 note in §1 is missing (M7), and `analyze` prints SUCCESS for the references too (M4). |
| B4 zero critic | §3 | zeroed and built after the head (test) | resolved |
| B5 TB ≠ PF | relabeled; PF-IS16 added | `pfis_loss` | resolved |
| S1 P4 | ACMC16 (δ = 1 only), RL16 on δ = 0, stop-gradient, uninterpretability clause | EV and \|TD\| logged | mostly resolved. There is no ACMC16 δ = 0 cell and no separate actor/critic gradient norms (M6, M10). |
| S2 dose | answer pool with the readout fixed | `ans_pool` (test) | resolved as a design. P6's directions now contradict P2's mechanism (S3). |
| S3 PP endpoint | an endpoint | `analyze` uses oracle-route accuracy on pp8 | **papered over** (S4) |
| S4 / code 6 determinism | separate endpoint: "sampled shortest-correct" | `samp_acc_P` is sampled **answer** accuracy | **not implemented** (B3) |
| S5 pilot seed | seed 100, tie rule | grid | seed resolved. The third bullet (δ = 0 at a second lr, or scoped claims) is **not handled** (S7). |
| S6 schedule | 5k steps, 3× clock | yes | acceptable. The cosine still reaches 15 % of peak by step 4,000. |
| S7 counts | P1 / P2 closed | — | P1 and P2 are fine. P3, P4 and P5's EI line have new gaps (B1, B2, S1). |
| S8 missing rows | §15 says "added" | — | **partly.** No row for "success only through EI16" or for "TB16 learns δ = 1, fails δ = 0" (S6). |
| S9 set_late | positive E / L / other | yes (test) | resolved |
| code 1 TB no-ops | canonical routes | yes | resolved and verified |
| code 3 / 5 | §5.5 and the §3 caveat | diagnostics | resolved. The text still says TD(0) where AC16 is now TD(0.9) (M1). |
| code 7 label boundary | E ≤ 0.25, L ≥ 0.75, mix | yes | resolved. P5's δ = 0 falsifier omits "mix" (S1). |

---

## Issues

### B1. BLOCKING (main grid): P3's falsifier is satisfied when both arms fail, and its other clauses leave gaps.

**What P3 says.** PF-IS16 succeeds in ≥ 2 more of 5 seeds than TB16, or, at equal counts, is earlier in ≥ 4 of 5 pairs. The falsifier: "TB16 succeeds in at least as many seeds as PF-IS16 and reaches criterion no later."

**What goes wrong.**
- **Both fail.** At 0/5 vs 0/5 the falsifier holds vacuously: "at least as many", and "no later" because neither reaches criterion. §9 reports every candidate except EI16 in all-H at 7.5 % of the budget, so both-fail is a likely outcome. P3 would then be recorded as *refuted*, with the reading "reverse KL is at least as good", which the data do not show.
  - Under a null where both arms have the same per-seed success rate p, the falsifier's count condition holds with probability 1.00 at p = 0, and 0.62–0.66 at p ∈ {0.2, 0.5, 0.8}.
- **A one-seed difference** (for example 3 vs 2) is neither the prediction nor the falsifier, and is not called inconclusive.
- **"≥ 4 of 5 pairs"** needs both arms to learn on at least 4 seeds, which is impossible at 3/5 each.
- **The mechanism sentence is misleading.** "Its gradient clips on ≈ 95 % of steps" does not separate the arms. PF-IS16 clipped on 50/50 of my first steps (`halt_dyn.py`), and its init gradient norm is 23–28 against TB16's 68–93 and RL16's 2–3.

**Fix.**
- Add a floor clause: if both arms have ≤ 1/5, P3 is **uninterpretable (both fail)** and reported as such, not refuted.
- Map a difference of exactly one seed to **inconclusive**.
- Read timing only over seeds where both arms succeed, needing ≥ 3 such pairs and earlier in all but at most one of them. Otherwise timing is not read.
- Replace the clipping sentence with what does differ: the forward-KL weights vs the reverse-KL residual and its floor (M2).

### B2. BLOCKING (main grid): P4's falsifier is at noise level for ACMC16, and its prediction and falsifier are not complements.

**Noise level.** "The learned fraction exceeds RL16's by more than 0.25" with ACMC16 at 3 seeds and RL16 at 5. Under the null (equal p), ACMC16 exceeds RL16 by more than 0.25 by chance with probability:
- 0.22 at p = 0.2;
- 0.30 at p = 0.5;
- 0.16 at p = 0.8.

Two-sided, the gap exceeds 0.25 with probability 0.38–0.60. For AC16 vs RL16 (5 vs 5), it exceeds 0.25 two-sided with probability 0.22–0.34.

**Not complements.**
- The prediction is two-sided: "within 0.25 … within 25 %". The falsifier is one-sided: AC "exceeds".
- **AC16 or ACMC16 doing clearly worse than RL16** is not mapped, yet it is a likely result. With explained variance around 0.05 (§9), ACMC16 is roughly REINFORCE with a near-zero baseline, against RL16's label-informed RLOO baseline.
- "Paired criterion steps within 25 %" and "≥ 4 of 5 pairs" again need both arms to learn on almost every seed.

**Fix.**
- For 5 vs 5, require a difference of at least 2 seeds.
- Either raise ACMC16 to 5 seeds on δ = 1 (+2 runs, about 0.25 T4-hours) or label ACMC16 vs RL16 exploratory.
- Add the outcome "AC16 / ACMC16 ≥ 2 seeds worse than RL16" with its own reading: the label-blind value is a worse baseline than RLOO.
- Read timing only over pairs where both learn (n ≥ 3).
- Keep the floor clause, and note that it is decided from 1 pilot seed per lr. If the main grid's RL16 seeds differ from the pilot's, use the main grid's RL16.

### B3. BLOCKING (main grid; fix before the pilot because it is free): "sampled shortest-correct accuracy" is never recorded, and the runs keep no checkpoints.

**Where the design relies on it.**
- §4: "Determinism: sampled shortest-correct accuracy on P-words".
- §5.2: PF's derived sampled shortest-correct, 0.42 / 0.26 / 0.29 / 0.00.
- §7 check: "PF … sampled shortest-correct ≤ 0.5 at in4".

**What the code records.** `evaluate` (`train_route.py:226–229, 282`) records only:
- `samp_acc` = the sampled route reaches the label, at any length;
- `samp_route_acc` = the oracle's route.

`analyze.py` reads determinism from `samp_acc_P`. A sampled route that reaches the label with redundant SETs counts as deterministic. Nothing is saved from which the right metric could be recomputed after the runs.

**Fix.** In the sampling loop add `samp_min.append(((s_s == fb["q_ans"]) & (st_s == osteps)).float())` and emit `samp_min_acc_{P,noP}`. Use it in `analyze` for the determinism row and the PF check. Training is unchanged, so do it before the pilot.

### S1. SHOULD-FIX: P5's EX-AL line contradicts §5.1 and §9, and the EI16 line's denominators do not work.

**The EX-AL line.**
- §5.1 shows that EX-AL's step-0 target equals PF's to 1 %.
- §9 shows exact PF and PF-IS16 heading for **L** on δ = 1.
- P5 predicts EX-AL lands in **E** "because its target averages toward the current policy's E-favouring posterior". Nothing in §5 derives an E-favouring current-policy posterior for EX-AL. EX's E program in rung 1 came with the rung-1 readout. On δ = 1 EX-AL also has only 2 seeds.
- **Fix:** move EX-AL to "open" beside PF and PF-IS16, or derive the claim. Keep EX (observed) and IW16 (rung-1 readout, like EX).

**The EI16 line.** "≥ 4/5 learned seeds" with the falsifier "≤ 2/4" cannot be applied when 2 or 3 seeds learn. A per-seed init beam `set_late` near 0.5 is also a coin flip.
- **Fix:** with n learned seeds and n ≥ 3, the prediction holds if all but at most one match their init beam program, and fails if at most half match. Exclude seeds whose init beam `set_late` lies in [0.4, 0.6]. If n < 3, the line is not read.

**The δ = 0 falsifier.** It lists "E or other". Add **mix**: any E SET on δ = 0 is equally unexpected.

### S2. SHOULD-FIX: the PF check's CPU-predicted outcome sends the whole rung to "investigate before reading anything else".

**The conflict.**
- §5.2 derives PF's δ = 1 fixed point as program E. §7's check requires "committed oracle program".
- §9: exact PF on δ = 1 is at `set_late` 0.78–1.0 (L) on shortest-correct routes with `min_acc` 0.84.
- The reading row "PF fails to reach its derived fixed point → investigate before reading anything else" then fires on the outcome the CPU run points to.

**Why E is fragile.** The derivation's E rests on a soft class margin, SET 0.60 / HOP 0.40 at E's state. That margin shrinks to 0.54 / 0.46 under another prior step cost, and the L class's target is sharper (SET 0.83). So it is plausible that a parametrized router fitting the sharper class first stays in L.

**Fix.** Split the row before the main runs:
- **PF does not learn (committed < 0.95):** investigate; possibly a bug.
- **PF learns, but in L on δ = 1:** the one-lookup pooling derivation is wrong about *which* program the function class reaches, and is recorded as a failed derivation. The other cells are read normally. PF's PP and determinism predictions, which depend on E, are withdrawn.

### S3. SHOULD-FIX: P6 puts IW16 among the "averaging signals" that should not degrade, which contradicts P2's own mechanism.

- **The §5.2 argument does not cover IW16.** "The targets' argmaxes do not move" is about the **fixed-prior** posterior. IW16 averages under the **router's own** posterior, which P2 holds responsible for reinforcing early stray mass.
- **The dose raises exactly that mass.** It raises the all-H program's coincidence on P-words from 1/16 to 1/4. All-H is the one consistent spurious program, and every candidate sits in it on CPU. IW16 is the arm most exposed to it.
- **"Fewer learned seeds" is ill-posed.** The dose has 3 seeds against 5 at pool 0, so it must be a fraction with a margin.
- **No floor clause.** If RL16 / AC16 fail at pool 0, which the CPU runs make likely, "degrade" cannot be measured.

**Fix.**
- Put IW16 in the "degrade" group or mark it open.
- State degradation as a learned fraction lower by more than 0.25 at 3 vs 5 seeds, or a paired criterion later by ≥ 25 % over pairs where both learn.
- Add "not read if the arm learns ≤ 1/5 at pool 0".

### S4. SHOULD-FIX: the PP endpoint reads oracle-route accuracy, so any δ = 1 run that learned L fails PP by construction.

- **The code.** `analyze.py`, "learned and passes pp8", uses `route_acc` on pp8: equal to the oracle's actions, and the oracle is E on δ = 1.
- **Who it hits.** On δ = 1, PF, PF-IS16 and EI16 have all gone L on CPU.
- **The design's own wording.** PP is "an endpoint, not a check" because it is the one generalization outcome a signal can change (review 3, S3). Mixing it with program identity breaks that.

**Fix.** Use `min_acc` on pp4 / pp8 (recorded per stratum; analysis only), and report `route_acc_pp` beside it. §5.2's "pp4 / 6 / 8" for PF should say which metric it uses.

### S5. SHOULD-FIX: the local halt term is necessary and principled, but it is a supervised signal the reward arms lack, and early on it is as large as PF-IS16's whole gradient.

**It is necessary.** Without the term, PF-IS16 drives π(HALT | answer) from 0.22 to **0.002** within 50 steps.
- HALT at a key never carries weight, the push-down on HALT reaches the answer states through the shared parameters, and canonical routes put no gradient on answer states.
- `committed` counts no-op moves at an answer as steps (`exec.py:154–156`), so `min_ok` (steps = oracle steps) could then never pass.
- The term is also PF's own target at arrival states (PF's counts include HALT on arrival). The term itself is fair.

**But it differs from what RL16 / IW16 / AC16 get.**
- **Size.** At init its gradient norm is 21–24, against 23–28 for PF-IS16 in total (term included), 68–93 for TB16, and 2–3 for RL16. Under per-run clipping at 1, it takes a large share of PF-IS16's early steps.
- **Leakage.** Its descent direction raises HALT at key states (cos +0.63 to +0.67 with ∇ mean log π(HALT | key)).
  - In 50 steps PF-IS16 still lowered π(HALT | key) from 0.21 to 0.06, so it does not cause a halting collapse.
  - Over the same steps, π(HALT | answer) rose from 0.22 to 0.47.
  - RL16, from the step cost alone, kept π(HALT | answer) at 0.23 while lowering π(HALT | key) to 0.03.
- **So the success metric can differ between the two families on halting alone.** The reward arms must learn halting from λ = 0.02 per no-op. TB16 / PF-IS16 get it supervised.
- **Weighting.** PF-IS16's term is an on-policy mean over every arrival, dead included. PF's exact decomposition weights it by the posterior. PF never trains HALT at dead, and has no HALT count at an arrival exactly at T.

**Fix.**
- (a) In `analyze`, flag a run whose answer accuracy on P-words is ≥ 0.95 while `min_acc_P` is < 0.95 and `p_halt_d0` is < 0.5 (already recorded) as a **halting failure**. A P1 / P3 / P4 difference must not rest on it.
- (b) Optionally weight PF-IS16's halt term by w̃, the exact forward-KL decomposition. Otherwise state the deviation in §3.
- (c) In §3, replace "TB16 could shrink it freely" with the push-forward argument (verified item 2). The no-HALT form is the correct objective, not a choice tuned on CPU, and saying so removes a researcher-degree-of-freedom concern.

### S6. SHOULD-FIX: the likeliest success outcome has no reading row.

- **Success only through EI16.** On CPU, EI16 is the only candidate that leaves all-H. Review 3 S8 asked for a row on success only through hard EM with a state-merging beam, saying whether that counts as "without exact inference". §15 says the S8 rows were added, but this one is missing.
- **"TB16 learns δ = 1 and fails δ = 0"** (self-reference through on-policy sampling) is also missing.

**Fix.** Add both rows. For EI16, write what success through it means for risk 3: search plus imitation works and the amortized signals do not. Say what that implies for the next rung.

### S7. SHOULD-FIX: δ = 0 claims rest on one lr, and review 3 S5's third bullet is not handled.

- **Which lrs.** P2(a) reads EX-AL at a fixed 1e-3. P2(b) reads PF-IS16 at the lr selected on δ = 1. The δ = 0 pilot controls run only at 1e-3.
- **Why it matters.** "EX-AL stalls" is the causal premise of the reading "self-consistency builds the basin". A stall specific to one lr is a weak premise.

**Fix.** Either add EX-AL δ = 0 at 3e-3 to the pilot (+1 run, about 0.1 T4-hours), or scope P2 and its reading row to "at lr 1e-3", and say so in §15.

### S8. SHOULD-FIX: ESS/K reads collapse as success.

- **The flaw.** Identical samples give ESS/K = 1. The 0.79 in §9 was measured while PF-IS16 sat in all-H, where most of the 16 samples are the same route.
- **What init looks like.** Only 13–24 % of samples reach any answer at init (`tbfloor.py`, 2 seeds, mixed training words), so ESS/K at init is at most about 0.2.
- **Why it matters.** The diagnostic is meant to separate "the importance weights are fine" from "the samples collapsed" (IW16's CPU failure).

**Fix.**
- Log the number of distinct canonical routes per query, or the fraction of queries whose K samples are all one route, beside ESS/K.
- Report both on P-words only.

---

### Minor

- **M1. Stale text after the TD(0) → TD(0.9) change.**
  - The reading row "AC16 slower than ACMC16 or RL16: … TD(0)'s early halting push" (DESIGN line 288) should name TD(0.9). TD(0)'s halting is the AC16T0 check.
  - §3's caveat says "TD(0) bootstraps". The bias now applies at weight 1 − 0.9ⁿ.
  - The `ac_loss` docstring still claims "RL16 vs AC16 isolates bootstrapping". The baselines also differ (B2).
- **M2. P3 has one more confound.** TB16 differs from PF-IS16 also by the reward floor and by near-permanent clipping. TB16's init gradient norm is 74–91 at floor 10⁻⁶, 41 at 10⁻³ and 30–33 at 10⁻². The floor routes are 76–87 % of samples and carry 24–34 % of the squared residual. So the floor is not the main cause of TB's large gradient, but it does shape the update. Scope P3's reading to "TB with the 10⁻⁶ floor", or add a 2-seed floor-10⁻² check.
- **M3. EX-AL vs PF differ in one more term.** EX-AL charges λ·E_π[steps] on-policy, while PF puts λ inside the posterior. The effect is small (λ = 0.02), but it is a second difference besides "whose posterior". Note it in P2's reading.
- **M4. `analyze` prints SUCCESS counts for EX / EX-AL / PF.** Mark them as references. The dose's success clock uses EXd1p4, a median of 2 seeds; state this in §1 or §7.
- **M5. The main grid encodes lr 1e-3 for every arm until the pilot rule is applied.** The rule does not say which lr AC16T0, SI16, ELL16 and EI16B take. Add the mapping, for example AC16T0 → AC16's lr and EI16B → EI16's, and a regeneration step or assert before launch.
- **M6. ACMC16 has no δ = 0 cell** (review 3 S1 asked for one). That is acceptable because P4 is δ = 1 only; say so in §15.
- **M7. Review 3 B3's sentence is missing from §1.** It should say that the "generalizes" part of risk 3 is mostly guaranteed by the Markov router (F30), and that PP precedence is the outcome a signal can change.
- **M8. The critic warm-up counts against the AC arms' clock.** Its 200 steps count toward the AC arms' success clock and budget (4 %). State it in §3.
- **M9. A cross-link for reading P4.** §5.5's TD(0) trap exists because the rung-1 readout pays 1/16 for halting at a key. That is the same dilution P2 tests through EX vs EX-AL. All three reward arms carry it, so P4's arms share a confound P2 could expose. This is worth one sentence in the P4 reading. No new arm is needed.
- **M10. Separate actor and critic gradient norms** (review 3 S1) are not logged. This is optional.

## Are P2–P6 genuinely uncertain (L27), and do they follow from §5 and §9?

| prediction | uncertain? | consistent with §5 / §9? | falsifiable at its seeds? |
|---|---|---|---|
| P2(a) EX-AL δ = 0 ≤ 1/3 | yes. The 1-seed CPU stall is at 7.5 % of the budget, where PF was also still all-H until about 1,000 steps. | yes | yes (3 seeds, ≤ 1/3 vs ≥ 2/3). Scope the lr (S7). |
| P2(b) PF-IS16 δ = 0 ≥ 2/3 | yes. It has no δ = 0 CPU run, and §5.4 gives the oracle in 0.00–0.03 of samples at init or the stray policy on δ = 0. Say that the evidence is PF exact plus the δ = 1 run. | yes | yes, conditional on PF learning. Define "PF learns" (≥ 1/2). |
| P3 PF-IS16 > TB16 | yes | yes, but the clipping sentence is wrong | **no** until B1 is fixed |
| P4 AC ≈ RL | yes, though likely uninterpretable at the floor | yes. Add the "worse" outcome. | **no** for ACMC16 until B2 is fixed |
| P5 EX-AL in E | no basis for it | **contradicts** §5.1 / §9 | fix S1 |
| P5 EI16 per seed | yes | yes | not at n < 4 learned seeds (S1) |
| P6 IW16 does not degrade | yes | **contradicts** P2's mechanism | exploratory; fix S3 |

## Remaining reasons not to launch the pilot

None. The pilot's purpose is lr selection, and its rule is well defined even if every lr fails: ties go to the last-1,000-step mean and then to 1e-3. Making B3 first costs nothing and means the pilot records carry the determinism metric. B1, B2 and S1–S7 are preregistration and analysis fixes. They must be in DESIGN.md before the main grid launches.
