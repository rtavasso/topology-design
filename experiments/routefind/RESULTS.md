# Rung 2 results: the router's training signal

**Provenance.** Grid `grids/rf_main.json` as committed at `705e345` (design revision 4, [DESIGN.md](DESIGN.md); the full preregistered 93-run grid, not the 72-run D108 cut, which was committed after the launch), lr 1e-3 everywhere (the pilot rule, [PILOT.md](PILOT.md) §2), 5,000 steps × batch 64, per-run init and sampling seeds 0–4, two T4s (Kaggle `topology-fleet-route-rf-main`), 5.7 h wall. Records in [results_data/main/](results_data/main/); the analysis, with rung 1's EX runs as a reference, in [results_data/main_report.md](results_data/main_report.md) (`analyze.py`). Tests on the GPU: 47 passed. No run failed.

**Verdict.** Rung 2 succeeds. Three signals that never enumerate the state space — the sampled current-policy posterior (IW16), the sampled fixed-prior posterior (PF-IS16) and search + imitation (EI16) — reach a shortest correct program on the deferred trigger in 4/5, 5/5 and 5/5 seeds, within 1.1–1.6× the exact marginal's time. The reward-only signals fail by exploration collapse, not credit assignment. On the local trigger, where exact EM stalls (F29), every current-policy signal stalls and every fixed-prior or search signal learns: the basin is a property of training on the router's own posterior. Which program a signal lands in, and whether it is deterministic, are set by the signal.

## 1. Outcomes

Learned = shortest-correct committed route on P-words ≥ 0.95 at in4 and in8 at 5k. Success = learned and the held-out criterion (every k ≤ 8 ≥ 0.95, confirmed) within 3× EX's median on the same cell. Program on P-words at in8: E (SET early, carry), L (SET late), mixed, or R (§3). Deterministic = sampled shortest-correct ≥ 0.95. PP = shortest-correct ≥ 0.95 at pp8.

**δ = 1 (the deferred trigger; EX's criterion 800, so the success clock is 2,400):**

| arm | n | learned | success | criterion (median [range]) | program | deterministic | PP |
|---|---|---|---|---|---|---|---|
| EX (reference) | 3 | 3 | 3 | 800 [700–900] | E | 3 | 0 |
| EX-AL (reference) | 2 | 2 | 2 | 800 | E | 2 | 0 |
| PF (reference, check) | 2 | 2 | 2 | 1,050 [1,000–1,100] | **E** | 0 (0.05–0.08) | 0 |
| **IW16** | 5 | **4** | **4** | 1,300 [1,100–1,500] | E; 1 × **R** | 4 (+ R deterministic) | 0 |
| **PF-IS16** | 5 | **5** | **5** | 1,200 [1,100–1,600] | L (3), mixed (2): `set_late` 0.59–0.78 | 0 (0.13–0.16) | 0 |
| TB16 | 5 | 0 | 0 | > 5,000; all five at 0.97–0.98 overall and 0.78–0.86 on in8 P-words, still rising at 5k | (L) | – | – |
| RL16 | 5 | 0 | 0 | – | all-H | – | – |
| ACMC16 | 5 | 0 | 0 | – | all-H | – | – |
| AC16 (TD 0.9) | 5 | 0 | 0 | – | all-H | – | – |
| **EI16** | 5 | **5** | **5** | 900 [800–1,100] | **L** | 5 | 0 |

**δ = 0 (the local trigger; EX has no criterion, so no clock):**

| arm | n | learned | criterion | program | deterministic | PP |
|---|---|---|---|---|---|---|
| EX (reference) | 2 (+ 5 in rung 1) | 0 | – | all-H | – | – |
| EX-AL (reference) | 3 | 0 | – | all-H | – | – |
| PF (check) | 2 | 2 | 1,050 | L | 0 (0.07) | 1 (0.84–0.86) |
| IW16 (check) | 2 | 0 | – | all-H | – | – |
| **PF-IS16** | 3 | **3** | 1,100 [1,100–1,300] | L | 0 (0.11–0.12) | **3** |
| **TB16** | 3 | **2** | 3,600 [3,400–3,600] | L | 0 (0.18) | **2** |
| RL16 | 3 | 0 | – | halting | – | – |
| AC16 | 3 | 0 | – | all-H | – | – |
| **EI16** | 2 | **2** | 800 | L | 2 | **2** |
| EI16B (check) | 2 | 0 | – | all-H | – | – |

**δ = 1, answer pool 4 (the dose; EX's criterion 950, clock 2,850):**

| arm | n | learned | success | criterion | vs pool 0 | program |
|---|---|---|---|---|---|---|
| EX | 2 | 2 | 2 | 950 [900–1,000] | +19 % | E |
| IW16 | 3 | 3 | 3 | 1,700 [1,500–2,000] | +31 % | E |
| PF-IS16 | 3 | 0 | 0 | > 5,000 (0.91–0.94 on in8 P-words; crossed 0.95 overall at 2,000 and stayed there) | from 1,200 to not at criterion | mixed (`set_late` 0.32–0.51) |
| TB16 | 3 | 0 | 0 | – | – | – |
| RL16, AC16 | 3, 3 | 0 | 0 | – | – | all-H |
| EI16 | 3 | 3 | 3 | 1,300 [1,200–1,400] | +44 % | L |

**Checks.** SI16 0/2, ELL16 0/2 (all-H), AC16T0 0/2 (halts at the start; `crit_obj` 2.77 = −log 1/16), EI16B 0/2: all as derived or observed before.

## 2. The predictions (DESIGN.md §7)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | IW16 succeeds in ≥ 3/5 on δ = 1, in E | 4/5, all E; the fifth is program R (correct, +1 step), not a collapse | **held** |
| P2(a) | EX-AL learns δ = 0 in ≤ 1/3 | 0/3, all-H like EX (the pilot seed's program R did not recur) | **held** |
| P2(b) | PF-IS16 learns δ = 0 in ≥ 2/3, in L | 3/3, L, passing PP | **held** |
| P3 | PF-IS16 ≥ 2 more seeds than TB16 on δ = 1 | 5/5 vs 0/5 | **held**, with the caveat that TB16 is slow, not dead: all five were at 0.97–0.98 overall and rising when the schedule ended, and TB16 did learn δ = 0 in 2/3 at 3,400–3,600 |
| P4 | no advantage from the learned value; uninterpretable if RL16 is at the floor | RL16, ACMC16 and AC16 all at the floor in every cell | **not read** (as preregistered) |
| P5 | EX, IW16 → E; δ = 0 → L; EI16 matches its init beam program; EX-AL, PF-IS16, PF open | EX 3/3 E, EX-AL 2/2 E, IW16 4/4 E; every learned δ = 0 run L (PF 2, PF-IS16 3, TB16 2, EI16 2); PF → **E** (the pooled fixed point, §5.2); PF-IS16 → L-leaning mixtures; **EI16 → L in 5/5 (+ 3/3 at the dose, 2/2 on δ = 0) although its init beam program was E on 4 of the 5 seeds** | E / L lines **held**; the EI16 line **refuted** |
| P6 | RL16, AC16, EI16 degrade at the dose by ≥ 25 % or ≥ 0.33 learned fraction; PF-IS16 does not; IW16 open | reward arms excluded (floor at pool 0); EI16 +44 % (degrades); IW16 +31 %; **PF-IS16 from 5/5 to 0/3** | EI16 clause **held**; PF-IS16 clause **refuted** |

**Endpoints reported without prediction.**
- **PP precedence.** No signal that learned δ = 1 passes PP: the E programs fail by precedence (first deviation SET>PREV: EX 744/750, EX-AL 500/500, PF 437/500, IW16 1,072/1,250), the L programs by the read (SET>HOP: EI16 1,250/1,250, PF-IS16 922/1,250). Every signal that learned δ = 0 passes (PF-IS16 3/3, TB16 2/2, EI16 2/2; PF at 0.84–0.86). The training signal changes the deviation, not the outcome: F30's bound stands.
- **Determinism.** The EM and imitation signals are deterministic (EX, EX-AL, IW16, EI16: sampled shortest-correct 1.00). The posterior-matching signals are not (PF 0.05–0.08, PF-IS16 0.11–0.16, TB16 0.18), with 0.8–1.8 extra steps per sampled route, though their committed route is shortest-correct. TB16 δ = 0 samples the right answer on 0.99 (PF-IS16 0.70–0.81, PF 0.37–0.38). As derived in §5.2.
- **Critic.** Explained variance 0.05–0.09 wherever the policy did not collapse; distinct sampled routes per query 1.00 for every reward arm from step 200–400 on.
- **Halting.** No learned run fails to halt on arrival.

## 3. What rung 2 shows

**A. Three non-enumerating signals learn the deferred program within budget (optimization; 5 seeds each).** IW16 (4/5, median 1,300 steps), PF-IS16 (5/5, 1,200) and EI16 (5/5, 900) against EX's 800. In rollouts, 5.5–8.0 M sampled or searched routes against EX's 51k exact recursions. Risk 3 (a router training signal that generalizes without exact inference) is retired for this task family.

**B. F29's basin belongs to training on the router's own posterior (mechanism, causal by intervention on the target; 2–3 seeds per cell plus rung 1's 5).** On δ = 0:
- every current-policy signal stalls in all-H: EX 0/2 (+ 0/5), EX-AL 0/3, IW16 0/2;
- every fixed-prior or search signal learns L: PF 2/2, PF-IS16 3/3, TB16 2/3, EI16 2/2.
- EX-AL and PF share the likelihood and their step-0 gradient (§5.1). The one thing that differs between them is whose posterior the target is. The likelihood on its own changes nothing: EX-AL matches EX on both triggers (δ = 1: 800 steps, E, same PP failure).
- The pilot's EX-AL seed, which found program R (§3D), did not recur at 3 seeds. One escape in four is a seed effect on top of a stall.

**C. Which program a signal lands in is a property of the signal (mechanism, inductive bias).**
- Current-policy posteriors, exact or sampled, land in E (EX 3/3, EX-AL 2/2, IW16 4/4); so does exact PF (2/2, `set_late` 0.05–0.10), as the pooled derivation said.
- Imitation of the beam's MAP route lands in L in 10/10 (δ = 1, dose, δ = 0), whatever program the init beam returned (E on 4 of 5 δ = 1 seeds). P5's "matches its init beam" is refuted; the mechanism of EI16's L bias is open. The merge rule is one candidate: E and L reach the same product state after their PREV, and the beam keeps the more probable prefix, so whichever the router starts to prefer is the only one it ever imitates again.
- PF-IS16 lands in L-leaning mixtures (`set_late` 0.59–0.78 at pool 0, 0.32–0.51 at the dose), not exact PF's E. The self-normalized weights with K = 16 and the router as proposal tilt the target toward the routes the router already samples: the sampled fixed-prior estimator carries a current-policy component that the exact one does not.

**D. A third program, R (mechanism; 2 runs: IW16 δ = 1 seed 0 and EX-AL δ = 0 pilot seed 100).** Correct and deterministic on every in-distribution stratum, exactly one step longer than the oracle on every word including all-H words (none8: 9 vs 8; len32: 40.0 vs 39.0), HOP accuracy 0.10–0.28 at the oracle's c = 0 states, SET 1.00, PREV 1.00, and the count-2 → PREV probe 0.99–1.00: SET once at the start, then the oracle's rule one count higher. The PREV decision lives at c = 2, a state the init phase never labeled; R is a correct program built around the stray-phase labels in the spare count capacity (c ≤ 2). Under the shortest-correct criterion it counts as not learned; under answer accuracy it is perfect. It fails PP like E (HOP>SET, SET>PREV).

**E. Posterior-matching signals are stochastic at their optimum; EM and imitation are deterministic (by construction, confirmed).** PF, PF-IS16 and TB16 match a soft posterior whose pooled argmax is the oracle but whose mass is spread over SET-padded routes (§5.2–5.3): committed route exact, sampled route 0.8–1.8 steps long and shortest-correct on 0.05–0.18. Their committed accuracy also decays on long words (len32 − in8: PF −0.36 to −0.76, PF-IS16 −0.10 to −0.38) as per-state errors compound. A router trained this way needs a committed (argmax) reading to be a program.

**F. Reward-only signals collapse before they learn (optimization; 5 seeds, three arms, three cells).** RL16, ACMC16 and AC16 become deterministic within 200–400 steps (16 identical samples per query), freeze in all-H or halting, and never move; a learned baseline or a bootstrapped value changes nothing because the advantage is identically zero once the samples agree. On-policy sampling with no entropy term is not a usable router signal here, and P4 (contextual value) cannot be read until exploration is fixed.

**G. The spurious-density dose slows every signal and hurts the fixed-prior target most (optimization; 2–3 seeds).** EX +19 %, IW16 +31 %, EI16 +44 %; PF-IS16 from 1,200 steps to not at criterion by 5k, in a mixed program. The dose quadruples the label-coincidence rate of every wrong route; the fixed-prior posterior spreads over them (§5.3: 0.99 of the mass at k = 8 is on non-oracle routes), the current-policy posterior does not once the policy has sharpened, and imitation of one MAP route is indifferent. P6's "averaging is robust" was wrong in its fixed-prior form; IW16 and EI16 are the robust signals, EI16 the fastest in every cell.

**H. TB16 is the slow signal.** Gradient clipped on 65–78 % of steps; reaches δ = 0 in 2/3 at 3,400–3,600 and plateaus at 0.97–0.98 on δ = 1 when the cosine schedule reaches 15 % at step 4,000. The matched 5k budget records it as a failure; the curves say slower by ≈ 3× than PF-IS16 (reviews 3 S6 and 4 named this confound). On δ = 0 its samples get the answer right on 0.99 while shortest-correct on 0.18: the reverse KL with its entropy term produces an answer-reliable but route-stochastic router.

## 4. Failed predictions and surprises

- **P5, EI16's program follows the init beam:** refuted 5/5 (plus 3/3 and 2/2). EI16 lands in L regardless.
- **P6, PF-IS16 does not degrade at the dose:** refuted; it degrades most.
- **PF-IS16's program:** expected (by the pooled fixed point) to be E; it is an L-leaning mixture. The sampled estimator is not the exact one.
- **Program R:** not anticipated by the two-program analysis (§2 of the design). Two of the four sampled-or-exact escapes on the "wrong" side of a prediction (IW16 seed 0, EX-AL pilot seed) were R.
- **The pilot's EX-AL reading:** the count-shifted program on the pilot seed suggested the likelihood mattered on δ = 0; at 3 seeds it does not. Pilot surprises are not readings (L30).

## 5. Reading-table rows that fired (DESIGN.md §7)

- "EX-AL stalls on δ = 0, PF and PF-IS16 learn": F29's basin is built by self-consistency. Risk 3 moves toward amortized-posterior training, with §3E's determinism cost.
- "TB16 and PF-IS16 differ": the KL direction decides speed (3×) and, on δ = 0, answer reliability of the samples.
- "RL16, ACMC16 and AC16 all fail": reward-only signals need an exploration term before contextual value can be read (risk 4 follow-up, not run here).
- "Signals split by program": later rungs must measure program identity.
- "Reward and imitation signals degrade at the dose" fired for EI16; the averaging-is-robust row did not fire as written.
- "PF learns in L on δ = 1" did not fire: PF learned E, the pooled fixed point.

## 6. What this does not show

- Anything about neural primitives or states (risks 1–2).
- Contextual value (P4): unreadable without exploration.
- EI16's L bias mechanism; R's frequency (2 of 20 learned runs across EX-AL and IW16; 0 of 10 for EI16 and PF-IS16).
- TB16 at a longer schedule.
- Generalization beyond F30's bound: no signal passes PP on δ = 1.

## 7. Claim types

A, F, G, H: optimization. B: mechanism (causal, by changing the target with the likelihood held fixed). C, D: mechanism (inductive bias; D's reading of the program is inferred from endpoints, not inspected routes). E: by construction (§5.2), confirmed.
