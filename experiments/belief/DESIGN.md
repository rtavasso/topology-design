# Belief updating in a looped prover on a hidden-world carrier — design

**Revision 2, 13 Nov 2026.** Nothing is built and no model has been run. The only numbers in this document are exact, model-free computations ([prelim_info.md](prelim_info.md), [power.md](power.md), scripts beside them). Not yet reviewed; an independent design review comes before any code is trusted (§8). This replaces the rung-5 outline of 6 Nov (history Entries 83–89 give the path here).

**Changes from revision 0** (after the researcher fixed the scored scope and a power analysis, history Entry 89): the scored questions are Q1 and Q2 and nothing else; the scored statistic is a projection coefficient instead of a vote count (§7); the coverage grid moves to where the two references differ (§6, §8). **Revision 2** (the researcher's decision, same day): the latent arm is kept at one coverage level, so that Q1 is answered for both ways of holding the evidence; the comparison between them stays unscored.

## 1. Questions

**Core (scored).**
- **Q1 — Updating.** After the kernel's verdicts on its own proposals, does the model's distribution over next steps move as the exact posterior does, *beyond* what masking the rejected actions gives?
- **Q2 — Coverage.** On worlds it never trained on, is the model's belief the *generalising* posterior (prior over all rule sets) or the *memorising* one (prior over its training worlds only), as a function of the number of training worlds?

**Secondary (reported, not scored; the researcher's decision of 13 Nov).**
- **Q3 — Belief from the problem alone.** Before any verdict, how close is the model to the exact posterior given only the state and the goal?
- **Q4 — Where the evidence is held.** A latent carried across rounds against verdicts appended as tokens: belief quality, and FLOPs.
- **Q5 — Thinking against asking.** What does one more loop iteration buy, in belief quality, against one more kernel call?

**Not asked here.** Reuse of latent moves ("soft programs"); size or depth extrapolation; experts; parallel streams; real Lean; comparison with published systems. Each was discussed on 10–12 Nov; each would add a variable.

**Why these.** The original thesis said the router is an amortized posterior over programs. Q1–Q3 are that claim, measured against a posterior that can be computed exactly. Q2 is rung 4's recombination question asked the way the literature says to ask it (INT 2021; Raventós et al. 2023): coverage is the variable, and two exact references make opposite predictions on held-out worlds.

## 2. What is already known, so that it is not re-asked

- With the world known and a deterministic kernel, a verdict carries 0 bits and the exact update is a mask (history Entry 87). So the world must be hidden for Q1 to mean anything.
- Feedback loops through a checker work at scale in token form (COPRA; Goedel-Prover-V2); search in a model's own context can be learned (Stream of Search; Thinker). These are system results; none compares a model's belief with an exact posterior.
- A coverage threshold exists in theorem proving (INT: 25 against ≥ 100 axiom combinations) and in in-context regression (Raventós et al.). Neither has a kernel in the loop.
- Not found in targeted searches (not a full review): a recurrent prover's belief compared with the exact posterior conditioned on kernel evidence.

## 3. The carrier

**Terms and rules.** As rung 4: terms over a, b, c, d, 0 with +; size cap 5 in training and evaluation alike (no size extrapolation anywhere); the eight directed rewrite rules Af, Ab, C, If, IbR, IbL, Df, Db; an action is (rule, node).

**Worlds (H8).** A world W is a non-empty set of the eight rules — 255 worlds. The model is never told W. The kernel accepts an action iff its rule is in W; an accepted action rewrites the state, a rejected one leaves it unchanged. Whether a rule's pattern matches at a node is visible in the term, so a rejection means "this rule is not in the world".

**Why eight rule-units.** Computed ([power.md](power.md)): with the four axioms as hidden units the problem itself nearly identifies the world (2.0 bits left; the exact update differs from a mask by TV 0.037 on average). With the eight rules, 6.2 bits are left and the mean difference is 0.094. Finer groupings on the same executor (ten or twelve units, splitting some rules by whether they apply at the root) raise it to 0.121 and 0.134 and add about one usable point to the coverage grid; they are the fallback if the pilot puts Q2's crossover beyond 32 training worlds, not the starting point. A new algebra is not needed for power.

**Generator** (parameters provisional; fixed at stage 0, §8).
1. u ~ the term sampler, sizes 2–5; k ~ {2, …, k_max}; s uniform on the full-world sphere of radius k around u. The problem (u, s) is drawn without reference to the world.
2. W: each rule present independently with probability ρ, conditioned on d_W(u, s) ≤ k + Δ.
3. Budget: at most k + Δ accepted steps; rejections are counted, not capped below 8.

Provisional values: ρ = 0.5, Δ = 2, k_max = 5.

**Evidence.** After any history, what the kernel has revealed is two sets of rules: known present, known absent.

**Separation of training and evaluation.** (u, s) pairs are disjoint between the two; a held-out set of 64 worlds is excluded from every training set. The build writes a counted overlap table (pairs, start terms, goals, worlds) into the dataset's meta file and a test asserts it (L45).

## 4. Exact quantities (code, no model)

- **Posterior over worlds.** P(W | u, s, evidence) ∝ ρ^|W| (1 − ρ)^(8 − |W|) · 1[d_W(u, s) ≤ k + Δ] · 1[W agrees with every verdict].
- **Step target and its Bayes predictor.** The label at a decision is an action optimal in the true world, sampled uniformly. The cross-entropy minimiser given (state, goal, evidence) is
  q(a) = Σ_W P(W | …) · 1[a optimal in W] / #optimal_W, over the worlds that can still finish within the budget.
  The posterior is never a training target; it is only the yardstick.
- **Mask-only update m̃.** The belief over worlds frozen at P(W | u, s), with the actions rejected at this state removed.
- **Memorising posterior q_train.** The same formula with the prior restricted to the training worlds.
- **Reference policies.** Oracle (knows W); q-greedy; mask-greedy; q_train-greedy.
- **Projection coefficients (the scored statistics).** For a model's distribution p at a decision,
  β = Σ ⟨p − m̃, q − m̃⟩ / Σ |q − m̃|² over all decisions after a verdict — the share of the exact update the model makes (0 for a mask, 1 for the posterior);
  γ = Σ ⟨p − q_train, q − q_train⟩ / Σ |q − q_train|² over all decisions in held-out worlds — 0 for the memorising posterior, 1 for the generalising one.
  Decisions where the references coincide add nothing to either sum, so no threshold is needed (L42 is met by construction).
- **Informative decisions (reported).** I_upd = decisions after a verdict with TV(q, m̃) > 0.05; I_cov likewise for q against q_train.

**Preliminary numbers** (H8, ρ = 0.5, k ∈ {2, 3, 4}, 300 instances; [prelim_info.md](prelim_info.md)).

| quantity | value |
|---|---|
| world entropy after seeing (u, s) | 6.2 bits; 82 of 255 worlds consistent |
| first-step entropy, world known / hidden | 0.95 / 1.57 bits |
| oracle: solved / kernel calls / rejections | 1.00 / 3.5 / 0.0 |
| q-greedy | 0.93 / 3.8 / 0.4 |
| mask-greedy | 0.90 / 4.0 / 0.7 |
| share of post-verdict decisions in I_upd | 33 % (mean TV 0.09) |
| held-out worlds, 8–32 training worlds: share of decisions in I_cov | 46–47 % (mean TV 0.10–0.14) |
| held-out worlds, 128 training worlds | 22 % (mean TV 0.06) |
| solve-rate gap, generalising against memorising | ≤ 0.04 at every coverage |
| cost | 0.15–0.3 s per instance, one core |

**What these say about the design.** The two references separate clearly at the level of *distributions* and barely at the level of *solve rate*. So belief measures are primary and behaviour is secondary. A wrong proposal costs one rejection here; that is why behaviour is forgiving.

## 5. Models — as plain as the questions allow

One shared transformer block applied n times with the input re-injected on every pass; a pointer output over (rule, node); no halting head (the episode ends at the goal). Width and depth of the block start from rung 4's and are set in the pilot.

| arm | what it receives about the past | question |
|---|---|---|
| **token** | every (rule, verdict) so far, appended to the input each round; no state carried | Q1, Q2 (and Q3, Q5 as reported) |
| **latent** | only the latest verdict; a few register tokens carried from the previous round, so everything earlier has to be remembered inside the model | Q1 at one coverage level (N = 64); its difference from the token arm is reported, not scored |
| **mask** | nothing; actions rejected at this state are masked at the output | the neural mask-only reference: its β must come out near 0 |

All arms share the looped encoder. The latent arm is the one that holds the history internally — the researcher's idea of 11–12 Nov — and is included at one coverage level so that the first grid says whether that version can update at all. Loop count n ∈ {1, 4, 8} on the token arm, three seeds each at one coverage, is the only other extra. No experts, no parallel streams.

## 6. Training

- **Supervised only.** One sampled optimal action in the true world per decision; loss at the last loop iteration (a loss at every iteration is what the literature associates with early commitment).
- **Proposals must be world-blind.** If the data's proposals were chosen by something that knows the world (an optimal teacher with random deviations, say), the choice of proposal would itself leak the world and q would no longer be the Bayes predictor. Histories are generated by proposers that see only what the model sees: the model's own samples (refreshed during training), mixed with a goal-directed proxy proposer and a uniform one.
- **Latent arm.** Gradients flow across rounds; the truncation length and the stability of this are pilot questions.
- **Coverage.** Training worlds N ∈ {8, 16, 32, 64}, drawn once, nested; the held-out worlds (a quarter of all worlds) are common to all. The grid sits where the generalising and memorising posteriors differ ([power.md](power.md)): with 8–32 training worlds they differ by TV 0.15–0.17 on average; with 64 or more by 0.04.

## 7. Measurements and predictions

All models are scored on the same decision points — the exact q-greedy trajectory, teacher-forced — and, separately, on their own trajectories.

| | measure | read on |
|---|---|---|
| M1 | β (§4), with TV(model, q) and TV(model, m̃) beside it | decisions after a verdict, training worlds |
| M2 | γ (§4), with TV(model, q) and TV(model, q_train) beside it | all decisions, held-out worlds |
| M3 | KL(q ‖ model) at the first decision | reported |
| M4 | solved, kernel calls, rejections, beside the reference policies | own rollouts, reported |
| M5 | β and M3 by loop iteration and by n | token arm, reported |

**Scored predictions.**
- **P1 (Q1).** Token arm, and separately latent arm, at the largest coverage: β ≥ 0.5 (lower end of a 95 % bootstrap interval over instances) in at least 4 of 5 seeds — the model makes at least half of the exact update. Falsified if β ≤ 0.25 (upper end) in at least 4 of 5. Between: "partial". The mask arm must give |β| < 0.15; if it does not, the instrument is broken and nothing is read from P1.
- **P2 (Q2).** γ at each coverage, per seed, bootstrap over held-out worlds. I expect γ near 0 at 8 training worlds and above 0.5 by 64; the result is the curve and where it crosses 0.5. γ > 0.5 at 8 would be generalisation from very few worlds; γ < 0.5 at 32 would be memorisation with an eighth of the space seen. At 64 the two references are close (mean TV 0.04), so a reading there is weak and is labelled so.

**Why a projection and not the vote count of revision 0.** Computed on stand-in models ([power.md](power.md)): at the noise rung 4's routers showed (TV 0.10–0.17 from their targets), a model making three quarters of the exact update gets a vote share of 0.65–0.73 — on the two-thirds line — while β reads 0.65–0.71; with no noise the vote share is a step function of the mixture. β tracks the mixture (0.00, 0.25, 0.45, 0.71, 0.98 at noise 0.10) with a sampling error of 0.004–0.008 at 2,000 instances.

**Reported, with my expectation stated.** Q3: between the unconditioned mixture and q. Q5: belief error falls with n; the exchange rate against kernel calls is the result.

**Thresholds** are fixed at stage 0, before any model exists, and the verdicts are computed by code with planted tests: exact q must give β = 1 and pass; exact m̃ must give β = 0; q_train must give γ = 0; stand-in models at noise 0.05–0.20 must be classed on the right side of each bound (L39, L42).

**What the power analysis cannot see.** Seed-to-seed spread (wide in rung 4; the pilot runs three seeds to measure it before the main grid's seed count is fixed). Structured model error: the stand-in's noise has no direction, a real model's may, and β reads only the component along the exact update — hence the TVs beside it. Whether models of this size learn the task at all.

## 8. Stages and gates

| stage | what | gate to pass |
|---|---|---|
| 0. Derive (CPU) | carrier module, exact-posterior code, tests; fix ρ, Δ, k_max, eval sizes, the bounds of P1 and P2 | world entropy ≥ 4 bits; sampling sd of β ≤ 0.02 and of γ ≤ 0.05 at the chosen eval sizes; reference and stand-in models classed correctly; overlap table clean |
| 1. Pilot (three seeds, short) | the arms learn at all; learning rate; the latent arm's training across rounds is stable; own-rollout against proposer shift; seed spread of β and γ; step count for the main grid | curves rising; cost projection from the last interval (L44) |
| 2. Main grid | ≈ 34 runs: token × N ∈ {8, 16, 32, 64} × 5 seeds; latent × 5 seeds at N = 64; mask × 3 seeds at N = 64; token with n ∈ {1, 8} × 3 seeds at N = 64 | — |

**Reviews.** Independent design review of this document before stage 0's code is trusted; code review before the pilot; a pilot-blind audit of the pass rules before the main grid; a results review after it. As in rung 4, each by a reader who did not write the thing reviewed.

## 9. Confounds, and what the design does about each

| confound | where it bit before | handled by |
|---|---|---|
| The mask does the work | rung 4 P8; Entry 87 | hidden world; m̃ as the zero of β; the mask arm |
| Proposals leak the world | (new) | world-blind proposers |
| Reference equals target on part of the set | rung 4 P6 | projection coefficients weight each decision by how far the references are apart |
| Too few worlds | rung 4 P2 | coverage as the variable; two exact references |
| Eval overlaps training | rung 4 (36 %) | disjoint pairs, held-out worlds, counted and asserted |
| One canonical target | rung 4 P6 | uniform over optimal actions |
| Encoding range | rung 4 P3 | no size extrapolation; same sizes everywhere |
| Early commitment from per-step loss | HRM, TRM analyses | loss at the last iteration |
| Behavioural effect too small to read | prelim: ≤ 0.04 | belief measures primary |
| A vote count that sits on its own threshold | revision 0 of this design | β and γ with intervals ([power.md](power.md)) |
| Curve statistics without a null | L46 | window means; no "final minus max" |

## 10. Compute

Exact quantities: 0.15–0.3 s per instance on one core; evaluation sets of several thousand instances are affordable. Models of roughly rung 4's size; the main grid is budgeted at no more than rung 4's 45 process-hours, to be checked against the pilot.

## 11. Decisions

**Taken by the researcher (13 Nov).** The scored questions are Q1 and Q2. The latent arm is included at one coverage level (five seeds).

**Defaults in force unless overruled.**
1. **Carrier.** H8 on the existing executor; the ten-unit grouping only if the pilot shows Q2's crossover beyond 32 training worlds.
2. **On-policy data.** Added only if the pilot shows a shift between the model's own histories and the proposers'.

## 12. Limits, stated in advance

Synthetic, rewrite-only proofs; the hidden fact is the presence of a rule, a crude stand-in for a library a model does not fully know; eight rules; small models. Solve-rate differences will be small, so the claims will be about beliefs. Nothing here speaks to real Lean, and none of it is a comparison with existing provers.
