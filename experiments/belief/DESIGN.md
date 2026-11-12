# Belief updating in a looped prover on a hidden-world carrier — design

**Revision 0, 12 Nov 2026. Draft for the researcher.** Nothing is built and no model has been run. The only numbers in this document are exact, model-free computations ([prelim_info.md](prelim_info.md), script beside it). Not yet reviewed; an independent design review comes before any code is trusted (§8). This replaces the rung-5 outline of 6 Nov (history Entries 83–88 give the path here).

## 1. Questions

**Core (scored).**
- **Q1 — Updating.** After the kernel's verdicts on its own proposals, does the model's distribution over next steps move as the exact posterior does, *beyond* what masking the rejected actions gives?
- **Q2 — Coverage.** On worlds it never trained on, is the model's belief the *generalising* posterior (prior over all rule sets) or the *memorising* one (prior over its training worlds only), as a function of the number of training worlds?

**Secondary (reported, not scored).**
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

**Why eight rule-units and not the four axioms (H4).** Computed: under H4 the problem itself nearly identifies the world (2.0 bits left; the exact update differs from a mask on 12 % of decisions, mean TV 0.04). Under H8, 6.2 bits are left and 33 % of decisions differ (mean TV 0.09).

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
- **Informative decisions.** I_upd = decisions after at least one verdict with TV(q, m̃) > τ. I_cov = decisions in held-out worlds with TV(q, q_train) > τ. τ = 0.05 provisionally. Every scored rule is read on these sets only (L42: a rule is not scored where its reference equals its target).

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
| **token** | every (rule, verdict) so far, appended to the input each round; no state carried | Q1, Q2, Q3, Q5 |
| **latent** | only the latest verdict; a few register tokens carried from the previous round | Q4 (and Q1, Q2) |
| **mask** | nothing; actions rejected at this state are masked at the output | the neural mask-only reference: it must *fail* Q1's rule |

Loop count n ∈ {1, 4, 8} on the token arm for Q5. No experts, no parallel streams, no designed registers beyond the carried ones.

## 6. Training

- **Supervised only.** One sampled optimal action in the true world per decision; loss at the last loop iteration (a loss at every iteration is what the literature associates with early commitment).
- **Proposals must be world-blind.** If the data's proposals were chosen by something that knows the world (an optimal teacher with random deviations, say), the choice of proposal would itself leak the world and q would no longer be the Bayes predictor. Histories are generated by proposers that see only what the model sees: the model's own samples (refreshed during training), mixed with a goal-directed proxy proposer and a uniform one.
- **Coverage.** Training worlds N ∈ {16, 64, 160}, drawn once from the prior, nested; the 64 held-out worlds are common to all.
- **Latent arm.** Gradients flow across rounds (truncation length set in the pilot).

## 7. Measurements and predictions

All models are scored on the same decision points — the exact q-greedy trajectory, teacher-forced — and, separately, on their own trajectories.

| | measure | read on |
|---|---|---|
| M1 | D_q = TV(model, q) against D_m = TV(model, m̃) | I_upd |
| M2 | D_full = TV(model, q) against D_train = TV(model, q_train) | I_cov, held-out worlds |
| M3 | KL(q ‖ model) at the first decision | all instances |
| M4 | solved, kernel calls, rejections, beside the four reference policies | own rollouts |
| M5 | M1 and M3 by loop iteration and by n | token arm |
| M6 | FLOPs per decision and per episode | all arms |

**Scored predictions.**
- **P1 (Q1).** Token arm: D_q < D_m on at least two thirds of I_upd in at least 4 of 5 seeds. Falsified if D_q ≥ D_m on a majority in at least 4 of 5. The mask arm must come out on the other side; if it does not, the rule is broken and nothing is read from it.
- **P2 (Q2).** I expect a crossover: closer to q_train at 16 training worlds, closer to q at 160. The result is where the crossover falls. Closer to q at 16 would be generalisation from very few worlds; closer to q_train at 160 would be no generalisation with most of the space covered, and against the thesis.

**Reported, with my expectation stated.** Q3: between the unconditioned mixture and q. Q4: belief parity between latent and token, and *little* FLOP difference — histories here are at most about eight verdicts, too short for the cost of a growing context to show; a carrier with longer histories would be needed to test the cost claim. Q5: belief error falls with n; the exchange rate is the result.

**Thresholds** (the two-thirds, the seed counts, τ) are fixed at stage 0 from the reference policies and a noise floor, before any model exists, and the verdicts are computed by code with planted tests: exact q must pass P1, exact m̃ must fail it, q_train must land on the memorising side of P2, and a *noisy* q (the target plus estimation error of realistic size) must still pass (L39, L42).

## 8. Stages and gates

| stage | what | gate to pass |
|---|---|---|
| 0. Derive (CPU) | carrier module, exact-posterior code, tests; fix ρ, Δ, k_max, τ, eval sizes | world entropy ≥ 4 bits; I_upd ≥ 25 % of post-verdict decisions; I_cov ≥ 30 % at N = 16; reference policies land on the intended side of every rule; overlap table clean |
| 1. Pilot (one seed, short) | each arm learns at all; learning rate; BPTT length; own-rollout against proposer shift; FLOP accounting; step count for the main grid | every arm's curve rising; cost projection from the last interval (L44) |
| 2. Main grid | ≈ 39 runs: {token, latent} × N ∈ {16, 64, 160} × 5 seeds; mask × 3 seeds at N = 160; token with n ∈ {1, 8} × 3 seeds at N = 160 | — |

**Reviews.** Independent design review of this document before stage 0's code is trusted; code review before the pilot; a pilot-blind audit of the pass rules before the main grid; a results review after it. As in rung 4, each by a reader who did not write the thing reviewed.

## 9. Confounds, and what the design does about each

| confound | where it bit before | handled by |
|---|---|---|
| The mask does the work | rung 4 P8; Entry 87 | hidden world; m̃ as a reference; the mask arm; rules read on I_upd only |
| Proposals leak the world | (new) | world-blind proposers |
| Reference equals target on part of the set | rung 4 P6 | informative sets I_upd, I_cov |
| Too few worlds | rung 4 P2 | coverage as the variable; two exact references |
| Eval overlaps training | rung 4 (36 %) | disjoint pairs, held-out worlds, counted and asserted |
| One canonical target | rung 4 P6 | uniform over optimal actions |
| Encoding range | rung 4 P3 | no size extrapolation; same sizes everywhere |
| Early commitment from per-step loss | HRM, TRM analyses | loss at the last iteration |
| Behavioural effect too small to read | prelim: ≤ 0.04 | belief measures primary |
| Curve statistics without a null | L46 | window means; no "final minus max" |

## 10. Compute

Exact quantities: 0.15–0.3 s per instance on one core; evaluation sets of several thousand instances are affordable. Models of roughly rung 4's size; the main grid is budgeted at no more than rung 4's 45 process-hours, to be checked against the pilot.

## 11. Decisions that are the researcher's

1. **Scored scope.** Q1 and Q2 only (my recommendation), or Q4 as well?
2. **Carrier.** H8 on the existing algebra — ready now, about 6 hidden bits, modest behavioural headroom — or a larger algebra with more hidden bits, which needs a new executor and makes exact posteriors costlier?
3. **On-policy data.** In the first grid, or only if the pilot shows a shift between the model's own histories and the proposers'?

## 12. Limits, stated in advance

Synthetic, rewrite-only proofs; the hidden fact is the presence of a rule, a crude stand-in for a library a model does not fully know; eight rules; small models. Solve-rate differences will be small, so the claims will be about beliefs. Nothing here speaks to real Lean, and none of it is a comparison with existing provers.
