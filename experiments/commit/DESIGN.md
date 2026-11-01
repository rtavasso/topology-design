# Rung 3: commitment (revision 2)

**Status:** design, 1 Nov 2026. Revision 2 applies design review 1 ([review/review1_design.md](review/review1_design.md); handling in §13) to revision 1 (`7310e91`). For a code review once implemented, and a short re-review of the open rows after the derive step (§5.3). History Entry 70; ledger risks 4 and 8, H12–H14, I30, I33.

**Direction.** After rung 2 (F35–F41), two questions decide how the router of later rungs should be trained, and both concern a router that has already committed to a program. **Core A (risk 8):** when the data later require a different decision at a state the router has already sharpened, does it move, and does that depend on how its targets are made? This is the live disagreement of Entry 70 §2: the researcher's position that varied data break ties, against the lock-in mechanism of F36. **Core B (risk 4's prerequisite):** can a reward signal be kept exploring long enough to learn, so that contextual value becomes readable (H14)? **Secondary A′:** the easier kind of plasticity, learning decisions at states the committed router never visited. **Checks C (H13):** on the saved routers, whether disagreement among sampled routes is an uncertainty signal, against the exact marginal.

**Scope (AGENTS hard rule 1).** The rung-1 router, executor and data; the rung-2 signals. Nothing neural. Every claim labeled.

---

## 1. The questions

**QA (overturning a committed decision).** A router committed to one program is trained further on a distribution where, at a state it visits on every relevant word, the decision it has sharpened is now wrong and one other decision is right. For each way of making its targets, does it move, and how fast, as a function of how often the data relabel that state?

| reading | prediction |
|---|---|
| *ties do not matter* (researcher): the loss relabels the decision when disambiguating data arrive; the procedure's first choice is a transient | every signal moves, at a rate set by the fraction of relabeling data, not by the signal |
| *plasticity is a property of the target* (Claude): a committed learner moves only if its targets can see the new decision | movement orders by how the target is made (exact fixed prior, search under the router's own policy, importance-sampled fixed prior, the router's own posterior), and is predicted per seed by the target's coverage of the new route at the committed policy |
| *exactness, not priors*: only an enumerating target moves | only PF and the teacher-forced ceiling move |

**QA′ (filling in a new decision).** The same committed routers see words that need decisions at states they never visited; the old program stays right on the old words. Does every signal fill the new decisions in? F30 says the data determine what was undetermined; whether every *signal* delivers that is the question, and the derived fixed point of the exact fixed-prior target says it may not (§5.1).

**QB (exploration before value).** Rung 2's reward arms collapsed into a deterministic policy within 200–400 steps (F40). Two fixes differ in kind: an entropy term keeps the policy soft; a beam supplies routes the policy would not sample. Which, if either, lets a reward signal learn the deferred program; and once one does, does bootstrapped value beat Monte-Carlo value (rung 2's unread P4)?

**QC (disagreement as uncertainty).** For a stochastic router, does disagreement among sampled routes predict which queries its committed route gets wrong? In the toy this is a Monte Carlo estimate of the exact marginal, so it is a check with a known ceiling.

## 2. Setting

Unchanged from rungs 1–2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4; [routefind/DESIGN.md](../routefind/DESIGN.md) §§2–3): 48 keys, 16 answers, decoys, the flag deferred (δ = 1) or local (δ = 0), the product state (symbol, pending c ≤ 2, set-here), HOP / PREV / SET / HALT, the Markov router with one cross-attention layer, ε = 0.02, λ = 0.02, T = 14, the strata, the held-out criterion set.

**The committed starting points.** Under δ = 1 the shortest programs E (SET at the predecessor, carry) and L (SET at the P-leaving state) are tied on the training distribution. Rung 2's phase-1 programs are deterministic: EI16 → L (10/10), EX → E (3/3). Rung 2 saved no weights, so phase 1 retrains them with checkpoints, under the **commitment gate** of §3.

**Shift A: the decision the data relabel.** Under δ = 0 the flag sits on the P-leaving pair, E is inexpressible, and E's rule is exactly F29's stray rule: the E router SETs at the flagged state (which is now the P-leaving state, L's correct SET) and then HOPs, where L needs PREV. So one in-support decision changes, HOP → PREV at the class (own flag = 1, c = 1, set-here = 1), and every P-word relabels it. The L router of δ = 1 reads the value slot to decide its SET; under δ = 0 that read never fires, so the L router never SETs and must learn SET at the own-flag state and then PREV: two decisions, one of them (PREV at set-here) the same relabel. The **dose** is `p_P`, the probability that an admissible step is P: at p_P = 0.3 (the standard) 16 % of queries are P-words; at 0.15, 10 %; at 0.08, 6 % (measured on the generator). This is a clean dose: the task stays well posed at every p_P.

**Shift B (secondary): the decisions the data add.** PP words (runs of two or three consecutive P-steps, today only an evaluation stratum) need decisions at states no committed router visits. Review 1 showed that both programs can complete them at the oracle's length: a PP run admits at least five shortest route types (E-E, E-L, L-L, L-double, E-double), and a one-lookup policy that stays L on ordinary words and adds three off-support decisions reaches 0.90 on PP words. So **PP words do not break the E/L tie**; revision 1's claim that they favour E was wrong. What they do is add decisions, plus a small push on ordinary words: flagged pairs inside PP runs create false pointers that corrupt L's value-slot read on 2–4 % of ordinary P-words. The dose is `pp_frac`, applied to chains of length ≥ 4 only: pp_frac = 1.0 makes 20 % of queries PP words, 0.25 makes 4.6 % (measured). Endpoints separate the ordinary-word program (on clean strata) from PP completion by route type.

**Program R** (F38) is counted as its own category throughout.

## 3. Design

All training at a **constant** learning rate of 1e-3 (the rung-2 pilot's choice for every signal) after a warm-up, so that a slow mover is not recorded as a failure by a decaying schedule (L31). Phase 1 warms up 500 steps; phase 2 warms up 100 steps with fresh optimizer moments, and every phase-2 clock starts at the end of its warm-up.

**Phase 1 (commitment).** δ = 1, p_P = 0.3, 2,500 steps × batch 64, per-run init, seeds 0–6; the full router state (parameters and the frozen bases' seed) is saved at the end. Arms: **EI16** (→ L) and **EX** (→ E) as the starting points, 7 seeds each so that 5 pass the gate; **PF** (3) and **PF-IS16** (5) for checks C.

**Commitment gate** (preregistered; applied at the checkpoint before anything else): shortest-correct committed accuracy ≥ 0.95 at in4 and in8; program label E (`set_late` ≤ 0.25) for EX and L (≥ 0.75) for EI16 on P-words at in8; sampled shortest-correct ≥ 0.95; not R. Seeds that fail are excluded and the next seed takes their place; the exclusions are reported. At each passing checkpoint the derive step records the **logit margins** at the classes phase 2 must flip (shift A: HOP − PREV at (own = 1, c = 1, h = 1); shift B: the three PP classes of review 1 B1), so that reachability under the phase-2 budget is stated before running (L24: Σ lr ≈ 3 units per parameter over 3,000 steps).

**Phase 2, shift A (core A).** Training distribution δ = 0, 3,000 steps. The treatment is the phase-2 signal, applied to the same saved router:

| phase-2 signal | how its targets can see the new PREV at the committed policy |
|---|---|
| **RT** (teacher forcing on the δ = 0 oracle route) | always: the full-information ceiling; the one arm known to move if anything can |
| **PF** (exact fixed-prior posterior) | by recursion, whatever the policy does; its pooled fixed point under δ = 0 is L (rung 2 §5.2) |
| **EX** (own posterior, exact) | by recursion over all routes under its own policy; sees the correct route with whatever posterior mass the stray policy leaves it |
| **EI16** (ε-mixed beam, hard EM), at ε = 0.1 and 0.3 | only if the route the beam *keeps* is the oracle's. On the CPU E checkpoint under δ = 0 the beam keeps a shortest-correct route on 93–98 % of P-words at both ε, but it is **E's route** (SET at the unflagged predecessor), which is correct per episode and inexpressible as a rule; the oracle's L route is a completion on 0/54 words because both candidates have exactly one ε-floored action and end at the same product state, and the merge keeps the more probable prefix, decided by π(SET \| predecessor) ≈ 2·10⁻⁶ against π(PREV \| s, 1, 1) ≈ 10⁻⁷ (code review 1, item 4). So from the E start, ε is not a coverage knob and EI16's target is a route the router cannot learn; from the L start the same tie may go the other way (the L router's rule "set-here ∧ pending ⇒ PREV" is in place). Both are measured at the checkpoints (§5.3). |
| **PF-IS16** (fixed prior, self-normalized IS from 16 own samples) | only if a sample takes the PREV |
| **IW16** (own posterior, 16 samples) | only if a sample takes the PREV, and the target on the rest prefers the old route |

Cells:
- from the **E routers** at p_P = 0.3: RT 2, PF 5, EX 5, EI16(0.1) 5, EI16(0.3) 3, PF-IS16 5, IW16 5;
- from the E routers at p_P = 0.15 and 0.08: PF, EX, EI16(0.1), PF-IS16, IW16 × 3 each;
- from the **L routers** at p_P = 0.3: RT 2, PF, EX, EI16(0.1), PF-IS16, IW16 × 3;
- **stability controls**: EX and EI16 routers continued on the unchanged δ = 1 stream under their own signal, 2 each (nothing should move);
- **matched from-scratch references** on δ = 0 at constant lr: EX, EI16, PF, PF-IS16 × 3 (rung 2's δ = 0 cells were on a cosine schedule; these separate "cannot move" from "cannot learn δ = 0 at this lr").

**Phase 2, shift B (secondary A′).** Training distribution δ = 1 with pp_frac = 1.0 (20 % of queries PP), 3,000 steps, from the **L routers**: RT 2, PF 3, EX 3, EI16(0.1) 3, PF-IS16 3; and an **f = 0 control** for every signal from the same routers (2 each), because the fixed prior prefers E on ordinary words with no PP words at all (§5.1) and every signal's drift on unchanged data is the baseline the shift is measured against. Evaluation on both the mixture strata and **clean strata** (pp_frac = 0), since false pointers alone drop an unchanged L router to ≈ 0.95 on ordinary P-words in a PP-polluted context.

**Core B (exploration for the reward signals), δ = 1 from scratch, 5,000 steps.** *Revised after the CPU sanity run (§9): the forced-breadth reward arms are checks; the forced-breadth posterior arm is the core.*
- **Forced-breadth reward arms (checks, 2 seeds each).** The K routes per query are the ε-mixed beam's **completions** (ε = 0.1, W = 16, as EI16) (every kept state completed by HALT at every step, and the forced halt at T; `beam_search` already scores them and keeps the best; it has to return them all), weighted by self-normalized importance weights w(r) = π(r) / Σ_{r′ ∈ B} π(r′) over the beam support, with a per-query baseline over the support. **RL16-B:** ∇J = Σ_r w(r) (R(r) − b) ∇log π(r), R = p(y | r) − λ·moves (rung 2's reward). **ACMC16-B / AC16-B:** the same support and weights with the rung-2 value head as baseline (λ = 1) or bootstrapping (TD λ = 0.9), 200-step critic warm-up. 5 seeds each on δ = 1; RL16-B and AC16-B also 3 on δ = 0. On CPU (§9) RL16-B's support weights put all their mass on the policy's own route from step 200 (effective sample size 0.004 of the valid completions) and it sat in all-H for 1,500 steps: the beam supplies breadth, the self-normalized weights under π throw it away as soon as π is peaked. That is the "forced breadth is not exploration" branch of PB2, observed before the grid, so these arms run at 2 seeds as checks (L29), and **P4 is not read in this rung.** The lesson is conceptual: a reward signal that keeps exploring has to weight the routes it did not sample by something other than its own probability of them, and every such weighting (uniform, likelihood-tempered, fixed-prior) is a posterior target. The value question needs a vehicle inside a posterior target (per-step credit by a bootstrapped value instead of whole-route weights), which is not designed here.
- **RL16+H(β)**, β ∈ {0.01, 0.03, 0.1}, 2 seeds each, as a **check**: §5.2 derives its stationary coverage and expects it to fail, for a reason different from revision 1's.
- **PFB16** (the fixed-prior posterior restricted to the beam's completions, soft EM over the beam), the core-B arm: the forced-breadth signal that does not reweight by π. Relabeled per review 1 S7: biased relative to PF (no importance correction; the beam's merge rule keeps one path per state, so the prior's padding cannot act and its program is near-derivable to be L, like EI16's). 5 seeds on δ = 1, 3 on δ = 0, 3 at rung 2's dose (answer pool 4) against a **matched constant-lr PF-IS16 dose reference** (3 seeds). It answers QB in the only form the CPU run left open: whether a soft target over a searched support learns where sampling (PF-IS16) and the beam's MAP (EI16) each have a known weakness (F41's dose fragility; EI16's L bias).

**Checks C (no training).** On every saved phase-1 PF and PF-IS16 router (the stochastic ones; EX and EI16 are deterministic and have no disagreement), on the held-out strata, per query: the committed route's correctness; the fraction of 16 sampled routes whose answer agrees with the committed answer (the statistic, fixed here: PF's sampled answer accuracy is 0.37, so raw disagreement is near its maximum everywhere and uninformative); the exact marginal's probability of the committed answer (the enumerating ceiling); the router's policy entropy at each state of the committed route by the oracle's decision there. AUROC is computed **per stratum**, only on strata where committed error is between 10 % and 90 % (len16, len32, alt, deepP8 for these routers), with the marginal's AUROC on the same queries; pooled AUROC is not reported, because it would be a stratum detector.

## 4. Endpoints

- **Movement** (shift A): shortest-correct committed accuracy on P-words at in4 / in8 under δ = 0, every 100 steps; time to ≥ 0.95 from the end of warm-up; the PREV logit at the relabeled class over time (the margin's trajectory); program label and R.
- **Filling in** (shift B): pp8 shortest-correct by route type (E-E, E-L, L-L, L-double, E-double; a classifier on the committed route); the ordinary-word program on clean strata (`set_late` on isolated-P words); in4 / in8 on clean and on mixture strata.
- **What the signal sees at the checkpoint** (§5.3, per seed): for the beam at ε ∈ {0.1, 0.3}, three numbers on P-words — the kept route is the oracle's; the oracle's route is among all completions; the kept route is shortest-correct at all — and the kept route's program; 16-sample coverage of the oracle route; EX's exact posterior mass on it; the log-prob margins at the two relabeled classes ((own flag, c = 1, set-here) HOP − PREV, and (own flag, c = 0, h = 0) HOP − SET, the latter for the L start, which never reaches the first), on the states the δ = 0 oracle route visits.
- **During phase 2** the criterion curve records, every 100 steps, shortest-correct accuracy on the criterion set's P-words (`min_acc_P`, with `n_P`) and both margins, so movement and the margin's trajectory are read on P-words, not on the all-word proxy (code review 1, item 5).
- Core B: criterion step, distinct routes per query, policy entropy, the beam's hit rate and the weights' effective sample size; the value head's explained variance.
- Checks C: per-stratum AUROC of the agreement statistic and of the marginal; entropy by decision type (reported; derived ratio ≤ 1.7×, §5.4).

## 5. Derived before measuring

**5.1 Fixed points of the exact fixed-prior target** (`routefind/derive/pooled.py`; [derive/pooled_pp_120.txt](derive/pooled_pp_120.txt), 120 episodes; review 1's `pp_pooled.txt` at 300). Under δ = 0 the pooled target is L with large margins (rung 2). Under δ = 1 it is E on ordinary words at every pp_frac, including 0: the predecessor class is SET 0.59–0.61 / HOP 0.40 whatever the PP fraction, because the preference comes from the flat prior's SET padding, not from the data. So **PF drifts from L toward E with no shift at all**; its shift-B cell is a check against its f = 0 control, not a test of the data's pull. Under the PP mixtures the pooled target **fails pp8 (0.01–0.02 at pp_frac 0.05 and 0.25, 0.08–0.14 at 1.0)**: the decisions a PP completion needs at the second P-leaving state pool into classes dominated by ordinary-word HOPs. The pooled model is blind to the router's generalization along the count direction (rung 1's probe (a): PREV on any carried count), so whether the actual PF router fills PP in is open, and that is QA′'s non-derivable part.

**5.2 Why the entropy fix should fail, and the beam fix is open.** Revision 1 derived coverage at the uniform init; review 1 pointed out that an entropy arm does not sit there. It learns all-H in a few hundred steps and sits at an all-H policy with residual mass set by β. At residual mass 0.15–0.6 on the other actions, 16 samples contain the oracle route on 1.6–4.4 % of P-words (≈ 1–3 oracle samples per step), so the setup step is occasionally rewarded; but the maximum-entropy stationary policy suppresses PREV at carried-count states by e^{−0.9/β}, because PREV loses the chain on every all-H word, and the all-H words are 84 % of the stream. So RL16+H is expected to fail at every β, and it is a check. The beam-supported arms see label-reaching routes on P-words from the start (the beam reached the label on 0.68–1.00 of k = 3 P-words at the stray policy, rung 2 §5.4), and their self-normalized weights concentrate on the routes the policy already prefers, which is the same proposal bias PF-IS16 showed (F37). Whether that bias leaves enough weight on the SET routes to learn is not derivable: core B's open question.

**5.3 Coverage and margins at the committed routers** (after phase 1, before phase 2 is preregistered; `gate.py`). Per passing checkpoint: the beam's kept route is the oracle's δ = 0 route on P-words at ε ∈ {0.1, 0.3}, the oracle route among all completions, the kept route shortest-correct and its program; the sampler's coverage at K = 16; EX's exact posterior mass on the oracle route; the two margins. The open rows of §7 are set from these, and the per-seed coverage → movement analysis of PA3 uses them. Two facts already in hand from the CPU checkpoint (one seed, 1,500 steps at batch 16, so indicative only): the beam under the E router keeps E's per-episode route and never the oracle's (the merge tie above), so for EI16-from-E the relevant coverage is 0.00 whatever ε; and the design review's estimate that the E start is easier than EI16B's basin was about the number of improbable actions, which the merge tie overrides. The EX and PF-IS16 rows stay open; the EI16-from-E row is now expected to be stuck unless the GPU checkpoints break the tie differently, and the EI16-from-L row is the open one.

**5.4 Checks C.** Given each query's exact marginal, the distribution of 16-sample agreement is binomial, so the agreement statistic's AUROC is a function of the marginal's and k; what is not derivable is whether the marginal's uncertainty is informative about committed-route error at all (H13's falsifier). The entropy-by-decision ratio is derivable from the pooled target: 0.59 nats at HOP states, 0.67 at SET, ≈ 0.9–1.0 at PREV, so ≤ 1.7×, not the 3× revision 1 predicted; it is reported.

## 6. Grid and budget

| stage | cells | runs | steps |
|---|---|---|---|
| 1 | phase 1: EI16 7, EX 7, PF 3, PF-IS16 5 | 22 | 2,500 |
| 1 | core B: PFB16 5 + 3 + 3, PF-IS16 dose ref 3; checks RL16-B, ACMC16-B, AC16-B × 2, RL16+H 3 β × 2 | 26 | 5,000 |
| (derive) | gate, margins, coverage, checks C on the saved routers | – | – |
| 2 | shift A from E at p_P 0.3: RT 2, PF 5, EX 5, EI16(0.1) 5, EI16(0.3) 3, PF-IS16 5, IW16 5 | 30 | 3,000 |
| 2 | shift A from E at p_P 0.15 and 0.08: 5 signals × 3 × 2 | 30 | 3,000 |
| 2 | shift A from L at 0.3: RT 2 + 5 signals × 3 | 17 | 3,000 |
| 2 | stability 4; matched from-scratch δ = 0 references 4 × 3 | 16 | 3,000 |
| 2 | shift B from L: RT 2, 4 signals × 3, f = 0 controls 4 × 2 | 22 | 3,000 |

≈ 163 runs, ≈ 530k run-steps, ≈ 13 T4-hours at rung 2's rate (the beam-completion arms cost ≈ 3× an exact step on CPU; the GPU ratio is measured in stage 1), ≈ 6.5 h wall on two T4s, in two jobs with the derive step between them. If this has to shrink, the p_P = 0.15 row and the L-start row go first; the E-start row at 0.3 with its references is the rung. Seeds are below AGENTS rule 4 and defensible only for count predictions on near-bimodal outcomes.

## 7. Predictions (counts; thresholds fixed here; rows marked *open* are set from §5.3 before stage 2 and re-reviewed)

| # | prediction | falsifier | type |
|---|---|---|---|
| **PA1** | **Shift A from E at p_P = 0.3.** RT moves in 2/2 (if not, the margins are unreachable and the shift is unreadable). PF moves in **≥ 4/5**: a fixed target with the right argmax and no basin. IW16 moves in **≤ 1/5**: its samples take the PREV almost never and its target on the rest is the old route. EX, EI16 and PF-IS16: **open**; set from §5.3 by a preregistered rule — a signal whose coverage of the correct route on P-words at the checkpoint is ≥ 0.3 is predicted to move in ≥ 3/5, one with coverage ≤ 0.05 in ≤ 1/5, and between those the row is reported, not predicted. | Every signal in ≥ 4/5 (ties do not matter; F36 was about stray labels from a bad start). Only RT and PF (exactness). IW16 ≥ 3/5. | mechanism (risk 8) |
| **PA2** | **Dose.** For every signal that moves at 0.3, the time to move grows as p_P falls, and PF still moves in ≥ 2/3 at 0.08; a signal that does not move at 0.3 does not move at any dose (then its failure is not about how much relabeling data there is). | A signal that moves at 0.08 but not at 0.3 (then "movement" was noise); PF ≤ 1/3 at 0.08. | mechanism |
| **PA3** | **Coverage predicts movement per seed.** Across every (signal, router, dose) run of shift A with an open or predicted row, the checkpoint's coverage of the correct route (for EI16: the kept route is the oracle's) predicts whether the run moves with AUROC ≥ 0.75 over ≥ 20 runs. The within-EI16 ε contrast is **reported, not predicted**: the CPU checkpoint shows the merge tie, not ε, decides what the beam keeps from the E start. | AUROC ≤ 0.6: plasticity is not about what the target sees. | mechanism |
| **PA4** | **From the L start**, the same ordering holds: PF ≥ 2/3, IW16 ≤ 1/3, the rest open; and the L start is harder than the E start for every signal (two decisions, the first with no in-support read). | The L start moving where the E start does not. | mechanism |
| **PB1** | **Entropy is not enough** (check): RL16+H learns in ≤ 1/2 at every β. | ≥ 2/2 at some β. | optimization (H14) |
| **PB2** | **Forced breadth with self-normalized weights is not exploration** (check, after §9): RL16-B, ACMC16-B and AC16-B learn in ≤ 1/2 each; their effective sample size over the support falls below 0.01 within 500 steps. P4 is not read. | ≥ 2/2 for any of them (then the CPU run was a seed accident and P4 is read on the three arms at 5 seeds in a follow-up). | optimization (H14) |
| **PB3** | **The forced-breadth posterior learns.** PFB16 learns δ = 1 in **≥ 4/5** and δ = 0 in **≥ 2/3**, in **L** (the beam's support carries EI16's bias); at the dose it reaches criterion in at least as many seeds as the matched PF-IS16 reference, and is stochastic like PF (sampled shortest-correct < 0.5). | ≤ 2/5 on δ = 1: a beam of 16 is too narrow a support for a soft target. Program E (the merge-rule derivation was wrong). Dose equal or worse than PF-IS16 (F41's fragility is in the fixed prior itself, not in the proposal). | optimization |
| **PA′1** | **Filling in** (shift B from L, pp_frac 1.0): RT fills PP in (2/2). The sampled and searched signals fill in only where their routes reach the PP completions: EI16 and PF-IS16 open from coverage; EX ≥ 2/3 (its exact posterior on PP words is on the completions). PF: **open** between the pooled derivation (fails pp8) and the count-ray generalization (fills in); whichever it is, reported against its f = 0 control. The ordinary-word program on clean strata stays L for every signal except PF, whose f = 0 control already drifts toward E (§5.1). | Every signal fills PP in (filling in is signal-independent, as F30's reading suggests). PF's clean-strata program unchanged at f = 0 (the pooled prior-drift derivation was wrong). | mechanism (F30) |
| **PC1** | **Disagreement is informative within a stratum.** On len16, len32, alt and deepP8, the agreement statistic's AUROC for committed-route error is ≥ 0.65 and within 0.1 of the exact marginal's on the same queries, for PF and PF-IS16. | AUROC ≤ 0.55 on every stratum: breadth is not an uncertainty estimator for this carrier. | by construction, partly (H13) |

**Checks (reported, not predicted).** The stability controls stay put. The from-scratch constant-lr δ = 0 references reproduce rung 2's pattern (EX stalls; EI16, PF, PF-IS16 learn). Entropy by decision type ≤ 1.7×. From-scratch EI16 on δ = 0 lands in L.

**Reading the outcomes.**

| outcome | reading |
|---|---|
| Movement orders by target type and coverage predicts it per seed (PA1–PA3 as predicted) | Plasticity after commitment is a property of how targets are made, and the predictor is what the target can see at the committed policy. The synthesis of Entry 70 (search supplies targets, a fixed prior breaks ties) is about plasticity. Later rungs train the router with targets whose coverage of the alternatives is engineered, never with its own samples alone. |
| Every signal moves at 0.3 and the dose sets only the speed | Commitment is not sticky; F36 was about stray labels from a bad start. The researcher's reading holds and the signal can be chosen on cost. |
| Only RT and PF move | Plasticity requires enumeration; every scalable signal locks in. Before any primitive-learning rung the program needs a signal that can move, and does not have one. |
| EX moves, IW16 does not | The own-posterior target is plastic when exact and not when sampled: sampling, not self-consistency, is the lock. |
| EI16 at ε = 0.3 moves and at 0.1 does not | Search is plastic exactly as far as it searches; coverage is the variable to engineer, and it is cheap. |
| RT does not move | The margins after 2,500 sharpened steps are unreachable at this lr and budget (L24); the rung is re-run with a longer phase 2 or a shorter phase 1 before anything is read. |
| PB2 holds (the reward arms collapse even over a searched support) | A reward signal that keeps exploring must weight unsampled routes by something other than its own probability of them, and every such weighting is a posterior target. Risk 4 is reposed: value as per-step credit inside a posterior target, not as a whole-route reward. |
| PB2 fails (a reward arm learns over the beam support) | The CPU collapse was a seed accident; the three arms are rerun at 5 seeds and P4 is read there. |
| PB3 holds | A soft target over a searched support is the scalable PFN-style signal: it has PF's plasticity (no own-policy weights) without PF-IS16's coverage problem. Candidate for rungs 4+. |
| PB3 fails on δ = 1 | The beam's support is too narrow for a soft target; the searched-target family needs EI16's commitment to work. |
| PA′1: PF fails pp8 at pp_frac 1.0 | The fixed prior's padding buries rare decisions; the count-ray generalization does not rescue it. A PFN-style target needs a prior that does not spread over padding. |
| PA′1: everything fills in | Filling in undetermined decisions is signal-independent; only overturning determined ones (shift A) separates the signals. |
| PC1 holds | Breadth measures the router's uncertainty about the answer where it has any; the controller of I30 is worth building on a carrier with real epistemic uncertainty. |

## 8. Gates (CPU sanity results in §9)

- **G1 tests:** checkpoint save / load round-trips (logits equal on a fixed batch across stack sizes; `evaluate` equal to the phase-1 record; the bases' seed travels with the state); `pp_frac` (done: f = 0 leaves the stream byte-identical; the realized fraction over all queries is asserted); the constant schedule and the 100-step phase-2 warm-up; the beam returning all completions, and PFB16's target equal to PF's when the beam is exhaustive on a tiny case; RL16-B's gradient on a hand-built support; the agreement statistic and per-stratum AUROC on hand-built cases; the PP route-type classifier on hand-built routes.
- **G2 derive:** §5.1 (done), §5.2 (review 1's numbers, to be reproduced in `derive/`), §5.3 (`gate.py`, done on the CPU checkpoint) and checks C between stages (the agreement statistic and per-stratum AUROC are not yet implemented; they are needed at the derive step, not for stage 1).
- **G3 pilot:** none; lr is 1e-3 throughout.
- **G4 reviews:** design review 1 (handled, §13); a code review before stage 1; the open rows fixed from §5.3 before stage 2, with a short re-review.
- **CPU sanity (L25):** one seed of EX phase 1 → PF and EI16 phase 2 under δ = 0, and one of RL16-B, 1,500 steps each, emitting the checkpoint's margins and coverage, before stage 1. Done; §9.

## 9. CPU sanity (batch 16, constant lr 1e-3, seed 0; 1,500 steps ≈ 15 % of a phase-1 run's episodes)

| run | outcome |
|---|---|
| EX phase 1, δ = 1 | criterion 0.99 at 1,500; program E; in8 P-words 0.92 and sampled 0.91, so it **fails the gate** at this budget (the gate is calibrated for 2,500 × 64); the checkpoint is 5.8 MB; the loader, `--ckpt-dir` and `init_from` worked end to end |
| `gate.py` on that checkpoint, shift A (δ = 0, 54 P-words) | beam keeps the oracle route 0.00 at ε 0.1 and 0.3; oracle among all completions 0.00; the kept route is shortest-correct on 0.78–0.81 and is program E (`set_late` 0.09–0.12); 16 samples 0.00; EX's own posterior on the oracle route 2·10⁻⁵; **margin HOP − PREV = +16.9 logits** [16.0, 17.7] on the 65 relabeled states, p(PREV) = 5·10⁻⁸; margin HOP − SET at (own flag, 0, 0) = −11.1 (the E router already SETs there, which is L's first step) |
| same, shift B (pp_frac 1.0, 55 PP words) | beam keeps a shortest-correct completion on 0.82; 16 samples 0.00 |
| EI16 from that checkpoint under δ = 0 | stuck: shortest-correct on P-words 0.00–0.02 throughout; the HOP − PREV margin drifts 16.9 → 8.5–12.8; the beam's E routes make it SET at unflagged predecessors inconsistently (teacher-forced SET 0.16–0.68) |
| PF from that checkpoint under δ = 0 | **moves**: the HOP − PREV margin falls from +16.9 to +1.4 within 100 steps and is negative by 500; shortest-correct on P-words 1.00 at step 500 (0.94–0.99 after), program L (`set_late` 1.0); pp8 0.93; stochastic as derived (sampled shortest-correct 0.12–0.29). The fixed target undoes the commitment in ≈ 400 steps at batch 16 where search from the same weights did not move in 1,500. Reachability is not the obstacle; what the target pushes on is. |
| RL16-B, δ = 1 from scratch | all-H from step 200; effective sample size over the support 0.004 (one route carries all the weight); beam hit 0.99 throughout; 3× an exact step's time on CPU |

What changed because of it: the reward arms became checks and PFB16 the core-B arm (§3, §6, PB2–PB3). The reachability question for shift A is answered in the affirmative at one seed: a 17-logit margin falls in 100 steps under a target that pushes on it. The one-seed contrast PF moves / EI16 stuck is the design's prediction PA1 in miniature and is not a reading (L30); the grid has five seeds per cell, the dose, the L start and the ceiling.

## 9a. What this rung does not answer

- Anything about neural primitives or the basis (risks 1, 2); rung 4 is the distillation test (D110).
- Branching at inference or a controller: checks C calibrate the picture, they are not the picture.
- Abstraction of compositions (risk 9).
- Plasticity under gradual shifts; one step per cell is the minimum.
- The E/L tie itself: review 1 showed no δ = 1 data break it. Which tied program a signal picks remains F37's accident; shift A overturns a decision rather than breaking a tie.

## 10. Implementation (`3715b02` and the code-review fixes; [review/review1_code.md](review/review1_code.md))

Done: `RouteConfig.pp_frac`; `run_stack` saves each run's state (`save_state`) and loads a per-run `init_from` (run_route writes `<out>/ckpt/<name>.pt` and resolves names against `--ckpt-dir`; the Kaggle launcher uploads a `--ckpt-dir` and brings checkpoints back); `schedule: "constant"` with a configurable warm-up; clean (pp_frac = 0) strata evaluated beside the mixture strata; program identity on isolated-P words; `pp_set_placement` keyed by run length with SETs of other P-steps attributed away; `beam_completions` (every completion of the ε-mixed beam), `pfb_loss`, `rlb_loss` / `support_weights`, `ac_loss(weights)` for AC16-B, `rl_loss(beta)` with `route_entropy`; the criterion curve's `min_acc_P`, `n_P` and the two margins; `gate.py` (the gate, coverage in its three forms, the kept route's program, both margins, EX's posterior mass); `make_grids.py` (stage 1: 58 runs; stage 2 from the gate report); `analyze.py`; `tests/test_commit.py` (11). Verified independently by the code review: the completion set, the PFB weights, the RL-B gradient, TD(λ) by hand, checkpoints, the grids.
To do before stage 2: checks C (agreement statistic, per-stratum AUROC against the marginal) as a derive script.

## 10a. Code review 1 handling

| item | handling |
|---|---|
| Bug 1 `gate.py` not uploaded (tests import it) | Added to the launcher's include list. |
| Bug 2 `analyze.py` import order | Fixed. |
| Bug 3 `pp_set_placement` over-counted SETs of other P-steps and PPP runs | Rewritten: the first PP run is located in the word, its P-steps matched to the oracle's PREVs by ordinal, SETs of other P-steps attributed away, the histogram keyed by run length; tests on the five route types and on a word with an isolated P before the run. |
| 4 the merge tie: EI16-from-E keeps the inexpressible E route | Recorded in §3, §5.3, PA3; the gate reports the three coverage forms and the kept route's program. |
| 5 no P-word movement field or margin on the curve | `min_acc_P`, `n_P` and both margins recorded every 100 steps; `move_step` uses `min_acc_P`. |
| 6 the margin was None for L routers | Margins are taken on the oracle route's states; the SET margin is added. |
| 7 fetch did not move checkpoints | Mirrored. |
| 8 critic diagnostics polluted by invalid slots | Masked by weight > 0. |
| 9 checks C not implemented | Deferred to the derive step (§8, §10). |
| Minor | ESS over valid completions; `cfg.init_from` stored as a file name; the launcher refuses a grid whose `init_from` names are not in `--ckpt-dir`; core B's ε = 0.1 stated in §3. |

## 11. Prior work

Loss of plasticity and continual learning (Dohare et al. 2024); policy collapse and entropy regularization (A3C, SAC); expert iteration and search-improved targets (Anthony et al. 2017; Silver et al. 2017); off-policy policy gradients with self-normalized weights; soft EM and posterior regularization; adaptive self-consistency (Wang et al. 2023; Aggarwal et al. 2023); ensembles and MC dropout as epistemic uncertainty (Lakshminarayanan et al. 2017; Gal & Ghahramani 2016); value of computation (Russell & Wefald 1991); DreamCoder (Ellis et al. 2021).

New here, to our reading: the same committed router continued under six target types with the data shift held fixed, which separates "the data relabeled the decision" from "the learner could move"; per-seed coverage at the committed policy as a preregistered predictor of plasticity, with a within-signal coverage knob; the beam-supported reward signal as the exploration fix that lets contextual value be read.

## 12. History of this design

- **Revision 1** (`7310e91`): shift B as core A ("PP words break the tie toward E"); PF as a core arm; core B gated on the entropy arm; Kendall τ over 6 pairs.
- **Revision 2**: shift A as core A with a p_P dose and a per-seed coverage predictor; shift B reframed as filling in, with f = 0 controls and clean strata; PF and RT as ceilings and checks; forced-breadth reward arms unconditional, P4 read there; the entropy arm a check; commitment gate, margins, matched references, constant lr with a short phase-2 warm-up; checks C per stratum with a fixed statistic.

## 13. Review 1 handling ([review1_design.md](review/review1_design.md))

| item | handling |
|---|---|
| B1 shift B does not break the tie; "moves to E" ≠ "passes pp8" | Accepted; §2 rewritten. Shift B is secondary and is about filling in, with pp8 by route type and the ordinary-word program on clean strata as separate endpoints. |
| B2 PF's movement is prior-driven and f-independent | Accepted (also found independently, §5.1). PF is a ceiling in shift A and a check against its f = 0 control in shift B; f = 0 controls for every signal. |
| B3 doses misstated 5× | Accepted; doses are stated as measured fractions of queries (§2), asserted in the generator's test; shift A's dose is p_P. |
| B4 core B lacks the forced-breadth reward arm; the gate hung on a derivable failure; §5.2 at the wrong policy | Accepted. RL16-B / ACMC16-B / AC16-B unconditional at 5 seeds, P4 read there; RL16+H a 2-seed check with review 1's stationary-coverage derivation in §5.2; PFB16 relabeled and given a matched constant-lr reference. |
| B5 no commitment gate, no ceiling | Accepted: the gate, the margin record, RT from every checkpoint under every shift, matched from-scratch references. |
| S1 shift A is the clean tie-breaker; p_P dose; EX / EI16 rows open | Accepted; it is core A, with a dose series and open rows set from §5.3 by a preregistered coverage rule. |
| S2 Kendall τ | Replaced by per-seed coverage → movement AUROC over ≥ 20 runs and the within-EI16 ε manipulation. |
| S3 PC1 | Per-stratum AUROC on strata with 10–90 % error, the marginal on the same queries, the entropy ratio reported with its derived value. |
| S4 contamination | Clean and mixture strata evaluated side by side; program identity on isolated-P words. |
| S5 phase-2 clock | 100-step warm-up; clocks start after it; Σ lr stated against recorded margins. |
| S6 phase-1 length | Margins recorded; one duration arm (EX at 1,250 steps, 2 seeds) added to phase 1 if budget allows; otherwise listed in §9. |
| S7 PFB16 | Relabeled (biased; program near-derivable L); exploratory. |
| S8 IW16 phase-1 unused; EX 3 not 5 | IW16 has no phase-1 arm; EX and EI16 have 7 for the gate; PF and PF-IS16 serve checks C. |
| S9 checkpoint tests | In G1. |
| S10 schedule-mismatched references | Matched constant-lr from-scratch references added. |
| M1–M6 | §2 states the two decisions EX-from-E must learn in shift B; H12's "25 % minority" to be corrected in the ledger; §10 updated; ε named as the coverage knob; the CPU sanity run emits margins and coverage; the agreement statistic is fixed in §3. |
