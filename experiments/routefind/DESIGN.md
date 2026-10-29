# Rung 2: the router's training signal (revision 3)

**Status:** design, 29 Oct 2026. Revision 3 re-scopes rung 2 after the direction discussion of Entry 65 (D105–D106). It keeps the implementation and the review handling of revisions 1–2 (`e886574`, `43726e7`; [review1_design.md](review/review1_design.md), [review1_code.md](review/review1_code.md); §14). For an independent design and code review before the pilot.

**Direction:** the ledger's "Current orientation" (Entry 65). The router is amortized inference plus a contextual value of each layer. Rung 2 retires **risk 3** (a router training signal that generalizes without exact inference) and opens **risk 4** (contextual value of computation). The design rule is L27: the core is the cells whose outcome cannot be derived; derivable cells are checks at minimal seeds.

**Scope note (AGENTS hard rule 1).** One fixed model (the rung-1 router), one task family; the arms differ only in the training signal. Nothing neural is added. Every claim is labeled.

---

## 1. The question

**Q5′.** Which training signal teaches a router *the contextual value of a layer* — which primitive to apply next, given the sequence and the token's current computational state — in a way that generalizes, when the exact route posterior under the router's own policy is not available?

The signals form a 2 × 2 and two singles (§3):
- **Whose posterior.** The posterior under the router's *current policy* (EM; rung 1's exact marginal) vs under a *fixed prior* over routes (amortized inference, the PFN signal).
- **How it is computed.** Exactly, or from sampled routes.
- **Reward,** Monte-Carlo vs bootstrapped through a learned value V(x).
- **Search + imitation.**

**Why these, from what rung 1 left.**
- F29: the exact current-policy marginal builds a self-reinforcing basin under the local trigger. A fixed-prior target cannot reinforce itself, so whether it escapes is the causal test of F29's mechanism.
- F28 / D74: setup steps have zero immediate value, so a value signal can credit them only by bootstrapping. Whether a learned value does so where Monte-Carlo returns don't is the first test of "contextual layer value".
- Two equally short programs exist (§2). Which one a signal lands in is a measurement of its inductive bias.
- Real routing has many wrong-but-right routes; the toy at 16 answers has few where the mass sits. The spurious-density dose is where the signals should separate rather than all succeeding.

**Success, stated before running.** Rung 2 succeeds if a signal that never computes the router's own exact posterior reaches a shortest correct program on the deferred trigger: shortest-correct-route accuracy ≥ 0.95 on P-words at in4 and in8, deterministic, within 4× EX's episodes. It fails if none does. The δ = 0 and spurious-density cells are read for *which* signals survive, not for success.

**Not asked here.** Neural primitives or states (risk 2, and risk 1 — the basis claim — is the next rung's candidate, D106); MCTS or a learned value used for search (B6); curricula; D99, D100.

## 2. Task, model, executor, and the two programs

Unchanged from rung 1 revision 2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4): 48 keys, 16 answers (4 in the dose), decoys at P-steps, the flag deferred (δ = 1) or local (δ = 0), the product state (symbol, pending c ∈ {0,1,2}, set-here h), actions HOP / PREV / SET / HALT, the Markov router (one cross-attention layer from the state token over the pair tokens, no attention among pairs), readout ε = 0.02, step cost λ = 0.02, T_train = 14, the strata and probes, held-out criterion set (D97).

**Two shortest correct programs per trigger.** PREV needs only a pending count, so a P-step can be prepared in two places:
- **E** ("early"): SET one state *before* the P-leaving state, carry the count across a hop, PREV.
- **L** ("late"): SET *at* the P-leaving state, PREV at once.

The oracle is E under δ = 1 and L under δ = 0. Hand-built policies for both reach every label in the oracle's number of moves. **Expressivity with one lookup per decision:**

| | needs to read | δ = 1 | δ = 0 |
|---|---|---|---|
| E | at the predecessor: is the P-step leaving my *successor*? | own pair's flag — one hop, **expressible** | the flag of the pair my value points to — two hops, **not expressible** |
| L | at the P-leaving state: is my own exit a P-step? | the flag of the pair whose value is my key — one value-slot match, **expressible** | own pair's flag — one hop, **expressible** |

So under δ = 1 a signal can land in E or L; under δ = 0 only L is reachable. Program identity is measured as `set_late` (the fraction of a route's SETs taken at the P-leaving state: L = 1, E = 0). Success is judged on shortest-correct-route accuracy; oracle-route accuracy is reported beside it.

## 3. The training signals (`route/estimators.py`, `route/train_route.py`)

| arm | whose posterior / signal | computed | gradient | per query |
|---|---|---|---|---|
| **EX** | current-policy posterior p_π(r \| y) (EM; rung 1's RM) | exact | E_{p_π(r\|y)}[∇log π(r)] = ∇log q | all states |
| **IW16** | current-policy posterior | 16 samples | VIMCO | K = 16 |
| **PF** | **fixed-prior posterior** p₀(r \| y) ∝ π₀(r) e^{−λ·steps} L_y(r): flat prior (uniform at keys, HALT at answers), answer likelihood (1 − ε on the label, ε/15 on another answer, 0 on a key) | exact (forward–backward) | E_{p₀(r\|y)}[∇log π(r)]: the same M-step with the posterior under the fixed prior | all states |
| **TB16** | fixed-prior posterior, as a GFlowNet target: reward R(r) = π₀(r) e^{−λ·steps} max(L_y, 10⁻⁶) | 16 on-policy samples | trajectory balance with log Z estimated per query as the sample mean (VarGrad) | K = 16 |
| **RL16** | reward p(y \| r) − λ·steps, Monte-Carlo returns | 16 samples | REINFORCE, RLOO baselines | K = 16 |
| **AC16** | the same reward, **bootstrapped through the router's learned contextual value V(x)** (a value head on the router's trunk; no label input) | 16 samples | actor-critic, TD(0) advantages, critic regressed on TD targets | K = 16 |
| **EI16** | search + imitation: beam MAP correct route (revision 2, hard EM) | beam W = 16 | teacher forcing on the found route | W = 16 |

Checks, at 2 seeds each (derived, or observed on CPU in revision 2):
- **EI16B:** EI16 started after 1,000 EX steps inside EX's δ = 0 basin; derived to fail.
- **SI16:** sample-then-imitate; derived to see no oracle route on P-words at init.
- **ELL16:** the K = 1 bound; observed failing.
- **IW16 on δ = 0:** derived to see no oracle route.

A matching-pursuit / one-step-value router is not run. Its failure is by construction: the readout after SET equals the readout after HOP to a key (1/16), so one-step value never prefers SET.

**Why the answer likelihood for PF / TB.** With the rung-1 readout (1/16 on any non-answer), the flat-prior posterior is dominated by "halt at once and guess" (1/4 × 1/16 per query beats every real computation). A route explains an answer only if it outputs one. Teacher forcing on the oracle (rung 1's RM-T) is the full-information limit of PF: the posterior given the generating process.

**Checked before measuring (G1; tests in `tests/test_routefind.py`).**
- **The fixed-prior posterior.** Forward–backward over (state, step) equals brute-force enumeration of every route for both deferrals: Z to 10⁻⁹ relative, action counts to 10⁻¹².
- **PF's gradient.** It equals E_{p₀(r|y)}[∇log π(r)] by enumeration.
- **Trajectory balance.** Its loss is zero exactly when the log-ratios agree.
- **The actor-critic.** On a hand-built route its TD(0) critic targets and Monte-Carlo advantages match hand computation.
- **The revision-2 tests.** Exact K = 2 unbiasedness of IW, ELL and RL by enumeration, and the beam against brute force, still pass.

## 4. Endpoints

As in revision 2 (§4 there): shortest-correct-route accuracy and program identity by stratum and on the held-out curve; oracle-route accuracy; committed, marginal and sampled readings; the counterfactual probes; first deviation; episodes and rollouts to criterion (on shortest-correct accuracy, confirmed); `crit_obj` (the exact current-policy objective on the held-out set); gradient-clip binding; what each sampled or searched signal sees, at init and every 100 steps, on P-words by k. New:
- AC16's value head is recorded through its effect on the policy (flip times, program).
- PF's and TB's targets are characterized in §5.

## 5. Derived before measuring

**What the fixed-prior posterior contains** (`derive/signals.py`, 1,800 training queries per cell; P-words by chain length k):

| k | posterior mass on E / L / other routes to the label / wrong answer (δ = 1, 16 answers) | same, 4 answers | on all-H words, mass on the oracle route (16 / 4 answers) |
|---|---|---|---|
| 3 | 0.23 / 0.23 / 0.53 / 0.00 | 0.18 / 0.18 / 0.64 / 0.01 | 0.40 / 0.36 |
| 5 | 0.07 / 0.07 / 0.84 / 0.02 | 0.04 / 0.04 / 0.88 / 0.04 | 0.16 / 0.10 |
| 8 | 0.00 / 0.00 / 0.86 / 0.13 | 0.00 / 0.00 / 0.89 / 0.11 | 0.02 / 0.01 |

What "other routes" contains:
- Mostly near-oracle routes with redundant SETs: the flat prior gives each extra SET 1/4. On a one-step all-H word the oracle holds 0.75 and SET-padded variants hold the remaining 0.25.
- On longer words, decoy routes that happen to end on the label.

The target at E's SET states is split: under δ = 1, P(SET) = 0.60, P(HOP) = 0.40 (stable across k), and 0.58 / 0.42 at 4 answers. Under δ = 0 it is 0.89 / 0.11, falling with k to 0.78 at k = 8.

A per-episode argmax of the target reproduces the shortest correct route on k = 3 / 5 / 8 P-words with frequency 0.94 / 0.75 / 0.11, and on all-H words at k = 1 / 4 / 8 with frequency 1.00 / 0.88 / 0.33. Whether a router that pools the target across episodes by state features does better on long chains is not derivable.

**What each sampled or searched signal sees** (`derive/search.py`, the real initialized routers, seeds 0–4; P-words, k = 3):

| | δ = 1, 16 ans. | δ = 1, 4 ans. | δ = 0, 16 ans. |
|---|---|---|---|
| 16 samples contain the oracle route, at init | 0.00 | 0.00–0.03 | 0.00–0.02 |
| at the stray policy | 1.00 | 1.00 | 0.03 |
| beam MAP correct route = oracle, at init | 0.10–0.71 | 0.32–0.71 | 0.26–0.87 |
| at the stray policy | 1.00 | **0.77** | 0.00 |
| 16 samples reach the label, at init | 0.10–0.32 | 0.13–0.52 | 0.10–0.35 |

The dose's main effect is on imitation and on the posterior targets. Spurious correct routes outscore the oracle in the beam on about a quarter of P-words at the stray policy (0.77 vs 1.00), and the posterior's oracle share falls (k = 3: 0.23 → 0.18).

**The expected gradient at init** (design review 1, B3): IW's equals EX's in every state category; the K = 1 bound's points toward halting. AC16 starts from a zero critic, so its first updates are RL16's. The two differ as soon as V(x) is non-zero.

## 6. Arms and grid (`make_grids.py`: `rf_pilot.json`, `rf_main.json`)

| cell | EX | IW16 | PF | TB16 | RL16 | AC16 | EI16 |
|---|---|---|---|---|---|---|---|
| δ = 1, 16 answers | 3 | 5 | 5 | 5 | 5 | 5 | 5 |
| δ = 0, 16 answers | 2 | — (check, 2) | 3 | 3 | — | 3 | 3 |
| δ = 1, 4 answers (the dose, D104) | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

Plus the checks: SI16, ELL16 (δ = 1), EI16B, IW16 (δ = 0), 2 seeds each.
- **Size:** 76 runs × 4,000 steps × batch 64.
- **Budget margin:** EX reaches criterion by 600–1,200 steps, so the 4× success criterion fits inside 4,000.
- **Seeding:** every run has its own init and sampling seed (`per_run_init`).
- **Cost:** ≈ 0.09 s per run-step, so ≈ 7.6 T4-hours, ≈ 3.8 h wall.

**Seeds.** 5 / 3 / 2 per cell is below AGENTS rule 4. It is defensible only for count predictions on near-bimodal outcomes, which every prediction here is. Paired speed comparisons are exploratory.

**Pilot (G3).**
- The sweep: IW16, PF, TB16, RL16, AC16, EI16 × lr {3e-4, 1e-3, 3e-3} on δ = 1 (seed 0, 4,000 steps).
- The controls: PF, TB16, AC16 on δ = 0 at lr 1e-3 (L26).
- Size: 21 runs ≈ 2.1 T4-hours.
- Rule: per signal, the lr with the highest held-out shortest-correct accuracy at 4k steps on δ = 1, ties → 1e-3. It applies to every cell of that signal. EX keeps 1e-3.

## 7. Predictions (core cells only; thresholds fixed here; per-seed counts)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **Sampled current-policy posterior, δ = 1 (open).** IW16 reaches a shortest correct program in **≥ 3/5** seeds within 4× EX, in program E (it averages like EX). One revision-2 CPU seed collapsed into "SET everywhere": identical samples give zero leave-one-out advantages. | ≤ 1/5: the sampled current-policy posterior does not find the program within budget. | optimization |
| **P2** | **The fixed-prior posterior (PFN signal), exact.** On δ = 1, PF reaches a shortest correct program in **≥ 3/5** seeds, in program **E** (the target's argmax at E's SET states is SET, 0.60, and both programs are expressible). On δ = 0 — where EX's self-consistent posterior stalled in 5/5 (F29) — PF learns in **≥ 2/3**, in program **L** (the only expressible one): its target cannot reinforce the router's own basin. | PF failing δ = 0 (≤ 1/3) while learning δ = 1: the basin is not EM's self-consistency (F29's mechanism refuted); the stray rule is reachable from a fixed target too. PF failing δ = 1: a soft target split 0.60 / 0.40 across two programs, plus SET-padding mass, does not train a deterministic router. | mechanism (F29, causal) |
| **P3** | **The fixed-prior posterior, sampled (TB16) — open.** Direction and confidence are set after the CPU sanity run (§9) and before the pilot. | — | optimization |
| **P4** | **Contextual value: bootstrapped vs Monte-Carlo (AC16 vs RL16) — open.** Set after §9, before the pilot. | — | mechanism (risk 4) |
| **P5** | **Which program each signal lands in (δ = 1).** EX, IW16 and PF land in **E** in the majority of learned seeds. EI16 lands in **L** in the majority (revision 2's CPU seed): imitation reinforces whichever shortest route the beam keeps, and L's one value-slot read is as available as E's. Every learned run on δ = 0 is L (expressivity, §2). | Any learned δ = 0 run with `set_late` ≤ 0.5 (an E program the router should not express: a bug, or an unanticipated read). EI16 in E in ≥ 4/5. | mechanism (inductive bias) |
| **P6** | **Spurious density (4 answers) — which signals survive.** Directional, following Min et al. 2019 (hard EM beats MML when spurious solutions abound) against Guu et al. 2017 (MML beats RL): **EI16 ≥ IW16** in learned seeds at 4 answers, and **PF and TB16 degrade most** relative to 16 answers (their targets spread over spurious routes, §5). | IW16 > EI16 at 4 answers (MML beats hard EM even here); PF unchanged by the dose. | optimization |

**Checks (by construction or already observed; reported, not predicted).**
- Every learned run on δ = 1 fails PP. E fails by precedence (F30). L fails because at a PP arrival no pair's value points to the state, so its value-slot read finds no flag; this is derived, not observed.
- Every learned run on δ = 0 passes PP: all of L's reads are in support.
- Learned runs are deterministic: sampled accuracy on P-words ≥ 0.95.
- EI16B, SI16, ELL16 and IW16-δ0 fail.
- EI16 learns δ = 0 (replication of revision 2's CPU seed).

**Reading the outcomes.**

| outcome | reading (which risk it moves) |
|---|---|
| PF learns δ = 0 where EX did not (P2) | F29's basin is a property of self-consistent EM, not of the task. Amortizing a fixed-prior posterior (the PFN form) is a training signal that cannot trap itself. Risk 3 moves toward PFN-style training for the router. |
| PF fails δ = 0 | The stray rule is an attractor of the task itself. Remedies must change the state or the library (D103 (i)), not the inference target. |
| AC16 learns setup steps where RL16 does not (P4) | Bootstrapped contextual value credits setup steps that Monte-Carlo returns can't, within budget. This is the first evidence that "contextual layer value" needs a learned value function (risk 4). |
| AC16 ≈ RL16 | At this horizon the value function adds nothing over returns. Risk 4's question moves to longer setup distances. |
| TB16 ≈ PF | The fixed-prior target survives sampling: a scalable PFN-style signal. |
| TB16 ≪ PF | On-policy sampling cannot find the target's support. Amortized inference needs exploration or search to supply targets. |
| IW16 collapses (P1 false) while TB16 or PF learns | The current-policy posterior is the fragile part under sampling, not sampled learning as such. |
| Signals split by program (P5) | Which program a learner gets is a property of its training signal. Later rungs must measure program identity, not only accuracy. |
| EI16 ≥ IW16 at 4 answers (P6) | Search + imitation is the robust signal when spurious routes abound. |
| IW16 > EI16 at 4 answers | Averaging is the robust signal, and imitation's MAP selection is the fragile one. |
| Every non-exact signal fails δ = 1 | "Learned from next-token loss" depends on exact inference. The program must fix this before scaling anything. |

## 8. Gates

- **G1 tests:** 21 rung-2 tests (§3), the 14 rung-1 tests and the full suite.
- **G2 derive:** §5 (done).
- **G3 pilot:** §6.
- **G4 reviews:** an independent design and code review of this revision, before the pilot.
- **CPU sanity (L25):** §9, before P3 and P4 are fixed.

## 9. CPU sanity runs (batch 16, lr 1e-3, 1,500 steps ≈ 6 % of one run's episodes, seed 0)

From revision 2, already recorded:
- EI16 learns δ = 0, in program L.
- IW16 on δ = 1 collapses into "SET everywhere".
- EI16 on δ = 1 lands in L.
- ELL16 on δ = 1 settles on the all-H program.

Revision 3: PF on δ = 1 and δ = 0, TB16 and AC16 on δ = 1, running when this was written; entered below before the reviews.

## 10. Fairness (AGENTS rule 4)

- One model, one task, one data stream per seed; only the training signal differs.
- lr tuned per signal in the pilot; best-vs-best with curves.
- The same episode budget everywhere. Rollouts per query are reported; PF's and EX's exact recursions are not rollouts.
- Clip binding is reported per arm.
- The label enters every arm only through the answer likelihood or readout at the route's end. AC16's value head never sees the label.

## 11. What rung 2 does not answer

- Neural primitives or states (risks 1 and 2).
- Search guided by a learned value (B6).
- Setup distances beyond one hop. Risk 4 at longer horizons is the follow-up if P4 is ambiguous.
- The basis claim, which is the next rung's candidate (D106).

## 12. Prior work

- **Score-function estimators.** REINFORCE (Williams 1992); RLOO (Kool et al. 2019); VIMCO and multi-sample bounds (Mnih & Rezende 2016; Burda et al. 2016; Rainforth et al. 2018).
- **Spurious programs.** Guu et al. 2017; Liang et al. 2017/2018 (NSM, MAPO); Min et al. 2019; MeRL; priority-queue training.
- **Amortized posterior / PFN.** Müller et al. 2022 (PFNs); TabPFN; DreamCoder's recognition model; RL-as-inference (Levine 2018), in which the fixed-prior posterior is the optimum of KL-regularized RL with a log-likelihood reward.
- **GFlowNets.** Bengio et al. 2021; trajectory balance (Malkin et al. 2022); VarGrad (Richter et al. 2020); sampling compositional objects in proportion to reward.
- **Value of computation.** Russell & Wefald 1991; Hay et al. 2012.
- **Actor-critic and TD.** Sutton 1988; GAE (Schulman et al. 2016).
- **Search + imitation.** Expert iteration (Anthony et al. 2017); STaR, ReST-EM.
- **Modular routing.** Rosenbaum et al. 2019; Mittal et al. 2022; Kirsch et al. 2018.

New here, to our reading:
- the exact current-policy and fixed-prior posteriors as references on the same seeds;
- a basin, created by self-consistent EM, against which fixed-prior targets, value bootstrapping and search are tested;
- program identity under an expressivity constraint that is known exactly;
- a spurious-density dose.

## 13. History of this design

- **Revision 1** (estimators; `e886574`).
- **Revision 2** (both reviews applied; `a80b932`–`43726e7`).
- **Revision 3** (re-scoped to training signals after Entry 65; adds PF, TB16, AC16 and the 4-answer dose; demotes EI16B, SI16, ELL16 and the sampled δ = 0 arms to checks; drops RT; 4,000 steps).

## 14. Review handling carried over from revision 2

All items of design review 1 and code review 1 remain handled as listed in revision 2's §13: real-init derivation; per-run seeds; P-word search statistics; float64 beam keys; collision-free grid names; exact K = 2 enumeration tests; confirmed criterion; the pruned-space MAP. Two items were updated by this revision:
- **Program identity** is now SET placement relative to the P-leaving state (E / L), not "SET at a flagged state". The latter conflated program and trigger.
- **Program B** of revision 2 is L under δ = 1.
