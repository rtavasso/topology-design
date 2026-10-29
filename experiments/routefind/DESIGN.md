# Rung 2: finding routes without the exact E-step (revision 2)

**Status:** design, 29 Oct 2026; revision 1 (`e886574`) reviewed independently for design ([review1_design.md](review/review1_design.md)) and code ([review1_code.md](review/review1_code.md)); every item handled (§13). Ready for the pilot. Rung 2 of the ladder as re-planned on 28 Oct: stay symbolic, on rung 1's carrier, and ask what every rung-1 success rode on — the exact forward recursion over all product states — which no realistic state space allows.

**Ledger:** rung 1 [route/RESULTS.md](../route/RESULTS.md) (F28–F34; F29: setup credit arrives through the state, and the local trigger's basin); D75 (soft routing as the first E-step, search with amortization as the later one); B5/B6 (Entry 48); L25, L26.

**Scope note (AGENTS hard rule 1).** The comparison is between *training procedures* for one fixed model (the rung-1 router) on one fixed task; nothing neural is added (researcher's direction, 28 Oct). Every claim is labeled.

---

## 1. The question

**Q5 (B5).** When the router cannot be trained by the exact route marginal, does it still learn the routing rule — including a setup step whose payoff comes one hop later — from the answer alone, by procedures that only ever see the routes they sample or search? Which procedures, at what cost, and can any of them leave the basin the exact marginal could not (F29)?

**Why now.** F28–F29 are statements about exact inference: the exact marginal learns the deferred program because the init phase's stray actions, averaged over *all* routes, teach a state-only rule that is the oracle's under the deferred trigger; under the local trigger the same averaging builds a basin it never leaves. Any real system replaces the exact E-step by sampling or search, and those see different evidence. Every sampled or searched route is still scored exactly here, and the exact-marginal router is the reference on the same seeds.

**Two correct programs (found by the CPU sanity run, §9).** The deferred task has two equally short correct programs: **A**, the oracle's (SET at the flagged predecessor, carry the count one hop, PREV), which the exact marginal learns (rung 1, 10/10); and **B**, "SET at the P-leaving state, PREV at once", which reads the predecessor's flag by a value-slot match and is allowed by the executor because PREV needs only a pending count (the oracle's "not the one set here" condition is a convention of the written rule, not of the executor). A hand-built B policy reaches every label in the oracle's number of moves. Under δ = 0 the two coincide. Success and learning are therefore judged **program-agnostically** — *shortest-correct-route accuracy*: the committed route reaches the label in the oracle's number of moves — and which program a run learned is reported (`set_on_flag`: the fraction of its SETs taken at a flagged state; A = 1, B = 0). Oracle-route accuracy (program A) and the counterfactual probes, which are defined on A's states, are reported beside it.

**Success, stated before running.** Rung 2 succeeds if a learner that never computes the marginal reaches a shortest correct program on the deferred trigger (shortest-correct-route accuracy ≥ 0.95 on P-words at in4 and in8, deterministic) within **4× the exact router's episodes**, at ≤ 16 sampled routes per query or a width-16 beam. It fails if no such learner does. The generalization profile is reported separately, not as a success condition (design review N6/N7). **Scope:** at init, 16 samples already reach the label on most short queries (k = 1: 0.97–1.00, k = 2: 0.52–0.93) — the bound is nearly tight on most of the Zipf mass, and every claim is stated within that regime (design review N1).

**Not asked here.** Neural primitives or states; a learned value or MCTS (B6); exploration bonuses or curricula; a stress dose of spurious-route density (fewer answers, heavier tails; design review N3 — recorded as the first follow-up, D104); D99, D100.

## 2. Task, model, executor (unchanged from rung 1)

Exactly rung 1 revision 2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4): 48 keys, 16 answers, ⊥; H/P words with w₁ = w_k = H, P-runs ≤ 1 in training, decoys at P-steps; the flag deferred (δ = 1) or local (δ = 0); product state (symbol, pending c ∈ {0,1,2}, set-here h), 390 states, actions HOP / PREV / SET / HALT; the Markov router; readout ε = 0.02; step cost λ = 0.02; T_train = 14; Zipf(1.5) lengths ≤ 8; the same strata, probes, held-out criterion set (D97), oracle routes. Code: `route/{data,exec,router,pool}.py` untouched; `route/estimators.py` new; `route/train_route.py` gains the estimator switch, per-run init and sampling seeds, the warm start, and held-out search statistics. The `exact` and `teacher` paths are byte-identical to rung 1 (code review 1, verified).

## 3. The training procedures (`route/estimators.py`)

All use the label only through p(y | route end) — the information the marginal uses — and all carry the step cost.

| arm | objective / procedure | gradient | per query |
|---|---|---|---|
| **EX** | −log q(y) + λE[steps], q the exact route mixture (rung 1's RM) | exact (forward recursion) | all 390 states |
| RT | teacher forcing on the oracle route (rung 1's RM-T) — the speed reference, run on δ = 0 (rung 1 has it on δ = 1) | supervised | — |
| **IW16** | −log (1/K) Σ_k p(y \| r_k) + λE[steps], r_k ~ π: the K-sample importance-weighted bound (→ the exact objective as K → ∞) | VIMCO leave-one-out control variates | K = 16 samples |
| ELL16 | −E_π[log p(y \| r)] + λE[steps]: the expected log-likelihood = the K = 1 bound (an ELBO with the prior as posterior), estimated with 16 samples | RLOO | K = 16 |
| **RL16** | −E_π[p(y \| r)] + λE[steps]: REINFORCE with reward p(y \| r); its expectation is q(y), so it maximizes Σ q, not Σ log q (Guu et al. 2017's "RL" side) | RLOO | K = 16 |
| SI16 | sample-then-imitate (STaR / ReST-EM): 16 samples, the most probable one that reaches the label kept (log π − λ·steps), teacher forcing on it | imitation | K = 16 |
| **EI16** | expert iteration (hard EM over a beam): width-16 beam over product states under π̃ = 0.9π + 0.1·uniform, routes reaching the same state merged (π is Markov), every kept state completed by HALT; among completions reaching the label the MAP route (log π̃ − λ·moves) over the pruned route space (no no-op moves, halting at the first arrival at an answer); teacher forcing on it | imitation | beam W = 16 |
| **EI16B** | EI16 started inside EX's basin: the first 1,000 steps by the exact marginal (δ = 0 only), then EI16 | — | — |

**Checked before measuring (G1, `tests/test_routefind.py`, 16 tests).** At K = 2, the expectation of each score-function surrogate's gradient over *all* route pairs (exact enumeration, T = 3) equals the exact gradient of its stated objective to float precision: IW → −E[log mean(p₁, p₂)] + λE[steps]; ELL → −E[log p] + λE[steps]; RL → −E[p] + λE[steps]. At large K, the averaged VIMCO gradient matches ∇ of the exact marginal (cosine ≥ 0.97) and the averaged RLOO gradient matches ∇E[log p] computed by the recursion with a log readout. The beam returns the oracle route under the oracle policy and the brute-force MAP route over the pruned space (the code reviewer: 40/40 further cases, both deferrals, T ≤ 6); keys in float64 (no quantization). Runs in a stack are independent, a run's init and sampling stream do not depend on its stack, and the warm start switches at the stated step.

**Why MAP selection, and why over the pruned space.** With 16 answers a wide search reaches the label by spurious routes through decoys on almost every query, and the shortest of them is often not the oracle's; MAP selection makes EI hard EM, the route most consistent with the current router. Over the *unpruned* space a random router's MAP correct route often loops at the answer to the forced halt (π(HOP | answer) > π(HALT | answer) at init; code review, finding 7), which would teach looping; the pruned space makes EI imitate HALT at the first arrival.

## 4. Endpoints

All rung-1 endpoints (committed / marginal / sampled accuracy by stratum, P-word splits, steps, p(HALT) by distance, the two counterfactual probes, first deviation, held-out criterion curves, SET / PREV flip times, head norms), plus:
1. **Episodes and rollouts to criterion** (held-out route accuracy ≥ 0.95 at every k ≤ 8, confirmed by the next 100-step point).
2. **What the procedure sees**, on the held-out set at init (before any update) and every 100 steps, over all words and over words with a P, by k: the label is reached by a sample / the beam ("hit"); some sample is the oracle route ("oracle_any"); the route imitation would use is the oracle's ("oracle_kept": EI's beam route, SI's best correct sample).
3. **The exact objective on the held-out set** (`crit_obj`) for every arm.
4. **Gradient-clip binding** per arm (fraction of steps with ‖g‖ > 1): the surrogates differ in scale and clipping is where scale survives Adam (design review N6).
5. **Per-seed covariate:** the init value of oracle_kept on k = 3 P-words, against each EI / SI seed's outcome.
6. **Shortest-correct-route accuracy** (by stratum, P-word split, and by k on the held-out curve — the criterion is computed on it) and **program identity** (`set_on_flag`); oracle-route accuracy beside them.

## 5. Derived before measuring (`derive/search.py`; 1,200 training queries per deferral; the router at init for seeds 0–4, as the grids initialize it)

Revision 1 derived these at an *exactly uniform* policy, where every prefix ties and the beam's answer is an artefact of sort order (both reviews, blocking); they are re-derived here at the real initialized routers.

**What each procedure sees on words with a P (k = 3 / 4 / 5 / 8):**

| | δ = 1, init (seeds 0–4, range) | δ = 1, stray | δ = 0, init (range) | δ = 0, stray |
|---|---|---|---|---|
| 16 samples contain the oracle route | 0.00 everywhere | **1.00 / 0.98 / 0.90 / 0.80** | 0.00–0.02 | **0.03 / 0.00 / 0.02 / 0.00** |
| SI: best correct sample is the oracle's | 0.00 everywhere | 1.00 / 0.85 / 0.85 / 0.73 | 0.00–0.02 | 0.03 / 0.00 / 0.02 / 0.00 |
| 16 samples reach the label | 0.10–0.32 / 0.02–0.22 / 0.05–0.15 / 0.00–0.07 | 1.00 / 0.98 / 0.90 / 0.83 | same as δ = 1 | 0.16 / 0.22 / 0.17 / 0.20 |
| EI16: beam MAP correct route is the oracle's | **0.10–0.71** / 0.04–0.40 / 0.00–0.17 / 0.00–0.03 | 1.00 / 0.87 / 0.95 / 0.90 | **0.26–0.87** / 0.05–0.76 / 0.00–0.66 / 0.00–0.30 | **0.00 / 0.00 / 0.00 / 0.00** |
| EI4: same at W = 4 | 0.00–0.32 / 0.00–0.16 / … | 1.00 / 0.87 / 0.93 / 0.90 | 0.00–0.35 / 0.00–0.33 / … | 0.00 |

On all-H words at init, 16 samples reach the label at k = 1 / 2 / 3 / 8 on 0.97–1.00 / 0.52–0.93 / 0.20–0.57 / 0.00–0.06, and the beam's MAP correct route is the oracle's on 1.00 / 1.00 / 0.70–0.90 / 0.00–0.06. The "stray" policy is the one both rung-1 routers reached within 50 steps (HALT at answers; set-here ⇒ HOP 0.99; carried count ⇒ PREV 0.99; flagged (0, 0) ⇒ SET 0.22; else HOP 0.99); it is a caricature, used only where its numbers are extreme (0.00 or ≥ 0.8).

**The expected gradient at init** (design review B3, tabular policy, per 1,000 queries; EX and ELL exact by the recursion, IW-16 averaged): at the flagged state under δ = 1, EX pushes SET +3.1 and IW-16 SET +2.1, while **ELL pushes SET −5.4 and HALT +5.7**. At unflagged states (the stray SETs) EX +1.3, IW +2.2, **ELL −62**. At a carried count EX/IW push HOP +22/+24, **ELL pushes HALT +10.8**. The reason: ELL's per-route reward is −0.02 for the right answer, −2.77 for halting on a non-answer and −6.62 for a wrong answer, so it prefers stopping to guessing. The marginal is indifferent in expectation, because a decoy guess and a halt both give ≈ 1/16.

**Readings that set the predictions.**
- **Sampled likelihood procedures (IW, RL, ELL).** None sees an oracle route on a P-word at init, under either deferral. They can learn only by first building the stray rule from averaged stray mass, as EX did. Under δ = 1 the rule then shows them the oracle (0.80–1.00); under δ = 0 it does not (≤ 0.03). IW's expected gradient matches EX's at init; ELL's points into a halt-at-the-flag basin on both deferrals.
- **Imitation procedures.** SI imitates the best correct sample. On P-words at init that sample is never the oracle's (0.00), so SI imitates spurious decoy routes and never averages. EI's beam merges states and keeps the most probable *correct* route; at init it is the oracle's on 10–87 % of k = 3 P-words, depending on the seed and not on the deferral. EI is then close to teacher forcing on short words, the known hard-EM result when spurious routes are rare where the Zipf mass sits. The CPU sanity run (§9) confirms this for δ = 0.
- **Inside the basin.** At the stray policy under δ = 0 the beam's MAP correct route is the oracle's on 0.00 of P-words, for every mixing ε ∈ {0, 0.1, 0.3, 0.5} (design review N5). A procedure started there sees only spurious routes and its own basin.

## 6. Arms and grid (`make_grids.py`; grids `rf_pilot.json`, `rf_main.json`)

| arm | δ = 1 seeds | δ = 0 seeds | role |
|---|---|---|---|
| EX | 3 | 3 | the exact-marginal reference (plus rung 1's 10 / 5 seeds for flip and criterion times) |
| RT | — | 2 | teacher forcing on δ = 0: EI's speed reference (rung 1 has RT on δ = 1) |
| **IW16** | 5 | 5 | the sampled exact objective (**the δ = 1 success test**) |
| ELL16 | 3 | 3 | the K = 1 bound / expected log-likelihood (the halt-at-flag prediction) |
| **RL16** | 5 | 5 | REINFORCE on p(y \| r): the RL side of RL-vs-MML |
| SI16 | 3 | 3 | sample-then-imitate: hard EM without search |
| EI16 | 5 | 5 | search + imitation from scratch |
| **EI16B** | — | 5 | search + imitation started inside EX's δ = 0 basin (**the headline that can fail**) |

55 runs, 6,000 steps × batch 64 (EX reaches criterion by 600–1,200 steps), every run with its own init and sampling seed. Cost ≈ 0.09 s per run-step (rung-1 pilot; the code review measured each estimator at or below the exact marginal's cost): ≈ 8.2 T4-hours, ≈ 4.1 h wall on two T4s.

**Seeds (AGENTS rule 4).** 5 / 3 / 2 seeds per cell is below the rule's ≥ 10. That is defensible only because every prediction here is a count of near-bimodal outcomes: rung 1's comparable cells came out 10/10, 5/5 and 0/5. Paired-speed comparisons at 5 seeds are reported as exploratory.

**Pilot (G3):**
- IW16, RL16 and EI16 × lr {3e-4, 1e-3, 3e-3} on δ = 1 (seed 0, 4,000 steps).
- The δ = 0 controls at lr 1e-3: IW16, RL16, EI16B; plus IW16 at 3e-4 (design review N6).
- 13 runs ≈ 1.3 T4-hours.
- Rule: per procedure, the lr with the highest held-out in-range route accuracy at 4k steps on δ = 1 (ties → 1e-3). Applied to both deferrals. ELL16 and SI16 take their family's lr (RL16's and EI16's). EX and RT keep rung 1's 1e-3.

## 7. Predictions (thresholds fixed here; per-seed counts)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **The sampled bound on the deferred program (δ = 1) — open.** IW16 reaches a shortest correct program in **≥ 3/5** seeds within 4× EX's episodes, and where it does, program A (it averages like EX). The expected gradient matches EX's at init (§5), but the one CPU seed collapsed by step 300 into "SET at every state, never halt" (§9): once 16 samples are identical, every leave-one-out advantage is zero and the step-cost term cannot pull back. Whether that collapse is typical or a single-lr accident is the question. | IW16 in ≤ 1/5 within budget, by the collapse (all-identical samples, SET at answers): the sampled exact objective does not find the program without exact averaging, and **rung 2's failure condition** holds for the bound. Then RL16's and EI16's δ = 1 results say whether *any* non-exact procedure finds it. | optimization |
| **P2** | **Which sampled objective (δ = 1).** RL16 reaches a shortest correct program in **≥ 3/5** seeds and is slower than IW16 where both learn (Guu et al.: MML beats RL under spurious programs; RL's gradient is q-weighted, so it vanishes on hard queries). **ELL16 learns in ≤ 1/3.** Revision 2 predicted it would fail by halting at the flag (§5's init gradient). The one CPU seed failed instead by settling on the all-H program (§9), so both modes are reported and neither is predicted: halting (`end_ans` on P-words ≤ 0.5, first deviation SET>HALT) and all-H (`end_ans` ≈ 1, P-word answers at the decoy level). | RL16 faster than IW16 in every paired seed where both learn; ELL16 learning in ≥ 2/3 (then the halt-at-uncertainty gradient of §5 is outweighed in training). | optimization / mechanism |
| **P3** | **Imitation from scratch: search vs samples, and which program.** EI16 reaches a shortest correct program on both deferrals (**≥ 4/5 on δ = 0, ≥ 3/5 on δ = 1**), within 2× RT's criterion step on δ = 0 — expected by construction as hard EM with a short-route bias (labeled as such). **On δ = 1 its learned seeds use program B in the majority** (`set_on_flag` ≤ 0.5), unlike EX (A, 10/10 in rung 1): imitation of beam routes on short words reinforces whichever shortest correct route the init favours, and B's "SET where the P-step leaves" is the same local decision as δ = 0's (one CPU seed, observed: §9). The per-seed covariate is reported (seeds with init oracle_kept ≤ 0.3 on k = 3 P-words reach criterion later or learn B). **SI16 learns neither (≤ 1/3 each):** it never sees an oracle route on a P-word at init (0.00 for every seed) and does not average, so it imitates spurious decoy routes on P-words. | SI16 learning δ = 1 in ≥ 2/3 (sample-then-imitate builds the routing rule without search, from all-H imitation alone); EI16 below 4/5 on either deferral (then the beam's init window is too narrow at some seeds — read against the covariate). | optimization |
| **P4** | **Leaving the basin — the headline that can fail.** **EI16B** (search + imitation started after 1,000 exact-marginal steps inside EX's δ = 0 basin) learns δ = 0 in **≤ 1/5** seeds: at the stray policy the beam's MAP correct route is never the oracle's (0.00 for ε ≤ 0.5), so imitation reinforces the basin. The sampled likelihood procedures on δ = 0 (IW16, RL16, ELL16) fail like EX (**≤ 1 of 13 seeds**; ≤ 0.03 of sample sets contain the oracle route at the stray policy). | EI16B in ≥ 2/5: search under the current policy, with ε-mixing and state merging, escapes a basin the exact marginal could not — the strongest argument for the thesis's search-based learner. A sampled procedure learning δ = 0 in ≥ 2 seeds: sampling noise escapes what the exact gradient could not. | optimization (the headline) |
| P5 | **Checks (not predictions; mostly by construction, design review N7).** Every run that reaches program A on δ = 1 shows F30's profile (B learners' profile is reported, not predicted: the probes are defined on A's states) (len16, len32 within 0.05 of in8; probe (a) ≥ 0.9; pp ≤ 0.05 with SET>PREV), unless an estimator's off-route visits change probe (b) (then say which); every run that reaches it on δ = 0 passes pp (rung 1's untested RM-d0 line); learned runs are deterministic (sampled accuracy on P-words ≥ 0.95, E[steps] within 0.2 of the oracle's, `crit_obj` within 0.02 of EX's in this grid). | — | check |

**Reading the outcomes.**

| outcome | reading |
|---|---|
| P1 holds; P4: EI16B fails | The routing rule survives the loss of exact inference wherever the averaging that builds it can be sampled. No procedure here leaves a basin whose setup and payoff share a state. F29 becomes a statement about likelihood-driven learning in general, and the remedy must be in the state (D103 (i)) or in exploration, not in inference. |
| P1 holds; EI16B learns | Search under the current policy escapes a likelihood basin. The thesis's search-based E-step (D75) earns its place on a measured case, and rung 3 can ask the neural questions with that procedure. |
| P1 fails by collapse; RL16 or EI16 learns δ = 1 | The sampled *bound* is the wrong objective under sampling at this K (zero-variance collapse), not sampled learning as such. Record which objective carries the setup credit, and connect to F31 (commit vs average). |
| Every non-exact procedure fails δ = 1 | "Learned from next-token loss" depends on exact inference: the program's central claim fails at the first step away from exactness. The next question is what minimal search or curriculum restores it, before anything else in the ladder. |
| EI16 learns δ = 0 but not δ = 1, or more slowly than EX on δ = 1 | Imitation of beam routes and likelihood averaging favour different triggers. Read with the per-seed covariate. |
| SI16 learns; EI16 does not | Search is not what matters; imitation of samples suffices (the STaR/ReST result). |
| EX learns A, EI learns B on δ = 1 (P3 as predicted) | Two exact, equally short programs; likelihood averaging finds the one that carries a bit (the stray rule is A's rule), imitation of searched routes finds the one that re-reads the context. Which program a learner ends in is a property of the procedure, not of the data — a direct measurement of the procedures' inductive bias, and a caution for every later rung that scores routes against one oracle. |
| ELL16 halts at the flag | The K = 1 bound's stop-over-guess preference (§5) is a learning-dynamics hazard of per-route log rewards with an uninformative readout. Relevant wherever RL with log-likelihood rewards meets an "I don't know" action. |
| ELL16 settles on the all-H program (as at the CPU seed) | The K = 1 bound suppresses the stray SET mass that built EX's rule (§5: −62 at unflagged states) without committing to halting; it never acquires SET at all. The averaging over K samples (IW) is what carries the setup credit, not the per-route reward. |

## 8. Gates

- **G1 tests (16, passing).** The tests listed in §3; plus the rung-1 tests (14), and the full suite.
- **G2 derive (done).** §5, from `derive/search.py` at the real initialized routers.
- **G3 pilot.** §6: the lr rule, the cost per procedure, and the δ = 0 controls run in the pilot.
- **G4 reviews.** Review 1 (design and code) is done and handled (§13). This revision goes to the pilot without a second review; the researcher asked for one review round.
- **CPU sanity (L25).** §9.

## 9. CPU sanity run (before revision 2; batch 16, lr 1e-3, 1,500 steps ≈ 6 % of one run's episodes, seed 0, revision-1 code)

- **EI16, δ = 0: learns.**
  - Held-out route accuracy reaches 0.97, against 0.81 (the all-H share) until step 600.
  - PREV at the set-here state is at 1.00 from step 100. This is the opposite of EX-d0's basin, where it stayed at 0.00–0.10.
  - SET flips between steps 700 and 1,100 (0.15 → 0.79), reaching 0.90 at 1,500.
  - The beam's kept route is the oracle's on 0.95–1.00 of training queries throughout.
  - At 1,500 steps, route accuracy on P-words is 0.94 at in4 and 0.81 at in8; len32 is 0.45 and pp4 / pp8 are 0.77 (local trigger: PP is in support), all still rising.
  - EI never builds the stray rule. This is why P3 labels EI16 from scratch "expected by construction", and why the headline moved to EI16B.
- **IW16, δ = 1: collapses.**
  - By step 300 the router SETs at every state, including answers (SET is a no-op there), until the forced halt: teacher-forced SET 1.00, HOP 0.00, HALT 0.00.
  - It partly recovers by 1,500 (HALT 0.72, HOP 0.31).
  - Held-out route accuracy is 0.17; answer accuracy is 1.00 at k ≤ 2 (answers absorb) and 0.12 at in8.
  - This is the basis of P1's "open" status.
- **EI16, δ = 1: learns a different program.**
  - Teacher-forced SET at the oracle's state stays at 0.00 for all 1,500 steps, and held-out oracle-route accuracy stays at the all-H share (0.81).
  - Yet answers on P-words are at 0.78 (in4), 0.64 (in8) and 0.60 (len16) and rising, with committed steps equal to the oracle's: the router SETs at the P-leaving state and PREVs at once (program B, now named in §1).
  - The beam's kept route was the oracle's on 0.84 of training queries throughout. That equals the all-H share, so on P-words it was B's route.
  - This is why success is now judged on shortest-correct routes (§1), and why P3 predicts B for EI on δ = 1.
- **ELL16 (then "RF16"), δ = 1: fails, but not by halting.**
  - SET and PREV never flip (teacher-forced 0.00 for all 1,500 steps); held-out accuracy plateaus at the all-H share (0.81).
  - The router settles on the all-H program: on P-words it HOPs into the decoy and ends on a (wrong) answer (`end_ans` 1.00, answer accuracy on P-words 0.03–0.11).
  - The halt-at-the-flag basin that §5's init gradient points to did not appear at this seed: HALT dips to 0.88 at step 500 and recovers.
  - P2's ELL line is therefore restated to "fails" with both failure modes measured (halting at the flag: `end_ans` on P-words ≤ 0.5; the all-H program: `end_ans` ≈ 1 with answer accuracy at the decoy level). Its mechanism clause is no longer a prediction.

## 10. Fairness (AGENTS rule 4)

- **Same everything but the procedure.** One model, one task, one data stream per seed; the arms differ only in the training procedure.
- **Learning rates.** Tuned per procedure in the pilot; best-vs-best reported, with curves.
- **Equal episode budget.** Every arm gets the same 6,000 × 64 episodes. Rollouts per query are reported; EX's 390-state recursion is not a rollout count, and at this size FLOPs are not the comparison of interest.
- **Clipping.** Its binding rate is reported per arm, because the surrogates' scales differ.
- **Seeds.** As stated in §6.
- **The label.** It enters every arm only through p(y | route end), and the imitation arms' selection uses the same quantity.

## 11. What rung 2 does not answer

- Neural primitives or states.
- A learned value or MCTS (B6).
- Exploration bonuses or curricula.
- Spurious-route density as a dose (D104, the first follow-up).
- Whether any of this scales with the state space: the reference recursion is kept here; only its use is withheld.

## 12. Prior work

- **Score-function estimators for discrete latents.** REINFORCE (Williams 1992); RLOO (Kool et al. 2019); VIMCO and multi-sample bounds (Mnih & Rezende 2016; Burda et al. 2016). Rainforth et al. 2018 show that tighter bounds can lower the gradient SNR, which bears on the ELL16 (K = 1) vs IW16 framing.
- **Learning latent programs from denotations, with spurious programs.**
  - Guu et al. 2017: RL (reward ∝ q(y)) vs MML, randomized beam search and the β-meritocratic update. These map onto RL16 vs IW16, with EI as β → ∞.
  - Liang et al. 2017 (Neural Symbolic Machines) and 2018 (MAPO); Agarwal et al. 2019 (MeRL); Abolafia et al. 2018 (priority-queue training).
  - Min et al. 2019: hard EM beats MML when spurious solutions abound. This predicts the EI vs IW ordering under D104's stress, not at this toy's density.
- **Answer-only supervision of a latent chain.** STaR (Zelikman et al. 2022) and ReST-EM (Singh et al. 2023): SI16's form. TRICE (Phan, Hoffman et al. 2023): MCMC-EM with control variates. Self-Imitation Learning (Oh et al. 2018).
- **Search with amortization.** Expert iteration (Anthony et al. 2017) and AlphaZero. Stochastic beams and sampling without replacement (Kool et al. 2019/2020), relevant once π sharpens and samples duplicate (P1's collapse).
- **Forward lattices over latent traces.** Neural Program Lattices (Li 2017).
- **What is new here, to our reading:**
  - the exact marginal is the reference on the same seeds;
  - a setup step whose credit is known (F28–F29) to flow through the state;
  - exact accounts of what each procedure sees;
  - a basin, created by exact averaging, that search is asked to leave (P4).

  EI learning from scratch (P3) is the expected hard-EM result, and is labeled so.

## 13. Review handling (revision 1 → 2)

| review item | handling |
|---|---|
| **design B1 / code 1** — §5's init numbers were tie-breaking artefacts | re-derived at the real initialized routers, seeds 0–4 (§5); the δ = 0 / δ = 1 "window" contrast is gone (seed-dependent, not δ-dependent); EI4 dropped from the predictions and the grid; the per-seed init covariate is logged (`search_init`) |
| **design B2** — EI16 on δ = 0 from scratch is already decided; the race does not happen | §9 filled in; P3 relabeled "expected by construction"; **EI16B** (started inside the basin) is the headline that can fail (P4); RT on δ = 0 added as EI's speed reference; EI16 δ = 0 cut to 5 seeds |
| **design B3** — RF is not the RL objective and points into a halt-at-the-flag basin | relabeled ELL16 (the K = 1 bound), with the halting prediction in P2 and §5's gradient table; **RL16** (REINFORCE on p(y \| r)) added as Guu et al.'s RL side |
| design N1 — IW ≈ EX at init; scope | stated in §1 (scope) and P1; IW4 cut |
| design N2 / N3 — not exhaustive; low spurious density where the mass is | stated (§5); the stress dose recorded as D104 (first follow-up), not run, for budget |
| design N4 — sample-then-imitate | **SI16** added |
| design N5 — ε does not decide P4 | stated (§5) |
| design N6 — lr tuned only on δ = 1; clipping | the δ = 0 pilot controls include IW16 at a second lr; clip binding recorded per arm |
| design N7 — P4/P5 by construction; success criterion's profile | P5 is a check; success is defined on the rule only |
| design N8 — missing outcome rows | §7 table |
| design N9 — seeds; EX reference has no crit_obj in rung 1 | seeds justified (§6); EX at 3 seeds per deferral |
| design F — prior work | §12 |
| code 2 — search stats not split by P-words | held-out search stats per run, all words and P-words, by k, at init and every 100 steps |
| code 3 — float32 keys with a 1e6 offset | float64 keys, no offset; `found = isfinite` |
| code 4 — Kaggle kernel / folder collision with rung 1 | grids renamed `rf_pilot.json`, `rf_main.json` |
| code 5 — tests could not tell IW from RF | exact K = 2 enumeration tests for IW, ELL and RL |
| code 6 — init and sampling depend on the stack | `per_run_init`: each run's parameter slice and sampling generator are seeded from its own seed; `stack_seeds` recorded |
| code 7 — "brute-force MAP" is over the pruned space | stated in the docstring, §3 and the test |
| code 8 — analyze gaps | confirmed criterion; rollouts to criterion; the full P4/P5 check table against this grid's EX; per-seed covariate; clipping |
| code 9 — no step-0 measurement | `search_init` before the first update |
| CPU sanity (after the reviews) — EI on δ = 1 learns program B | §1 "two correct programs"; program-agnostic success (shortest-correct-route accuracy) and program identity (`set_on_flag`) recorded and used by `analyze.py`; P1–P3 restated on it |
