# Rung 2: finding routes without the exact E-step (revision 1)

**Status:** design, 29 Oct 2026, for independent design and code review before the pilot. Rung 2 of the ladder as re-planned on 28 Oct (Entry 63's discussion): stay symbolic, on rung 1's carrier, and ask the question that every rung-1 success rode on — the exact forward recursion over all 390 product states — which no realistic state space allows.

**Ledger:** rung 1 [route/RESULTS.md](../route/RESULTS.md) (F28–F34); setup credit through the state F29; the state's role F31; the step cost F33; the thesis formalism D75 (latent-route mixture, soft routing as the first E-step, search with amortization as the later one); B5/B6 (Entry 48); L25, L26 (run the cheapest configuration and the controls before preregistering).

**Scope note (AGENTS hard rule 1).** The comparison is between *training procedures* for one fixed model (the rung-1 router) on one fixed task; nothing neural is added (researcher's direction, 28 Oct: neural carriers confound the mechanism). Every claim is labeled.

---

## 1. The question

**Q5 (B5).** When the router cannot be trained by the exact route marginal, does it still learn the routing rule — including a setup step whose payoff comes one hop later — from the answer alone, by estimators that only ever see the routes they sample or search? At what cost in data and rollouts, and with the same generalization profile and the same determinism?

**Why this question now.** Rung 1 showed (F28) that the exact marginal learns the deferred program at near-trace speed, and (F29) that it does so because the init phase's stray actions teach a state-only rule that happens to be the oracle's under the deferred trigger; under the local trigger the same phase builds a basin the exact marginal never left. Both facts are about inference. Any real system will have to replace the exact E-step by sampling or search, and those see different evidence: a sampler sees a route only when it draws it, a search sees whatever its beam keeps. Whether the thesis's "learned from next-token loss" survives that replacement is the most consequential open question rung 1 left, and it can be answered exactly here because every sampled or searched route is still scored by the executor, and the exact-marginal router is available as the reference on the same seeds.

**Success, stated before running.** Rung 2 succeeds if a learner that never computes the marginal reaches the exact router's rule on the deferred trigger (committed route accuracy ≥ 0.95 on P-words in range, the F30 generalization profile, deterministic routes) within **4× the exact router's episodes** at **≤ 16 rollouts per query**. It fails if no such learner does — then "learned from next-token loss" depends on exact inference that does not scale, the most important negative result the program could get. Separately, the local trigger asks whether **search** can do what the exact marginal could not.

**Not asked here.** Neural primitives or a neural state (deferred by the researcher's direction; D103's constraints stand for when they come); a learned value or MCTS (B6; the beam here is policy-guided only); scaling the state space (the exact recursion is kept as the reference, not as an option — the question is what each estimator *sees*, not how the recursion's cost grows); D99, D100.

## 2. Task, model, executor (unchanged from rung 1)

Exactly rung 1 revision 2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4): 48 keys, 16 answers, ⊥; chains with H/P words, w₁ = w_k = H, P-runs ≤ 1 in training, decoys at P-steps; the flag deferred by δ = 1 (thesis setting) or local (δ = 0); product state (symbol, pending c ∈ {0,1,2}, set-here h), 390 states, actions HOP / PREV / SET / HALT; the Markov router (one cross-attention layer, d = 344, frozen symbol bases, no step index, no history); readout ε = 0.02; step cost λ = 0.02; T_train = 14; Zipf(1.5) lengths ≤ 8; the same strata, probes, held-out criterion set (D97), oracle routes and planted router. Code: `route/{data,exec,router,pool}.py` untouched; `route/estimators.py` new; `route/train_route.py` gains the `estimator` switch.

## 3. The estimators (`route/estimators.py`)

All use the label only through p(y | route end) — the same information the marginal uses — and all carry the step cost.

| arm | objective | gradient | rollouts / query |
|---|---|---|---|
| **EX** | −log q(y) + λ E[steps], q the exact route mixture (rung 1's RM) | exact, through the forward recursion | — (all 390 states) |
| **IW-K** | −log (1/K) Σ_k p(y \| r_k) + λ E[steps], r_k ~ π: the K-sample importance-weighted bound, → the exact objective as K → ∞ | VIMCO (Mnih & Rezende 2016): Σ_k (L̂ − L̂₋ₖ) ∇log π(r_k), with a leave-one-out score term on steps | K ∈ {4, 16} |
| **RF-K** | −E_π[log p(y \| r)] + λ E[steps]: the expected log-likelihood, a Jensen lower bound of the marginal that rewards each route on its own (the "RL" objective) | REINFORCE with leave-one-out baselines (RLOO) | K = 16 |
| **EI-W** | expert iteration / hard EM: a width-W beam over product states under π̃ = 0.9 π + 0.1 · uniform; routes reaching the same state merged (π is Markov); every kept state completed by HALT; among completions reaching the label, the MAP route under π̃ and the step cost; the router trained by teacher forcing on it | imitation of self-found routes | W ∈ {4, 16} |

**Checked before measuring (tests, G1).** The averaged RLOO gradient matches the exact gradient of E_π[log p(y|r)] − λE[steps] computed by the forward recursion with the log readout (cosine 0.993 at K = 64); the averaged VIMCO gradient at K = 256 matches the exact marginal's gradient (cosine 0.999); the beam under the oracle policy returns the oracle route, and under any policy with W ≥ the reachable state count it returns the MAP correct route found by brute force.

**Why MAP and not "the shortest correct route" for EI (found while checking).** With 16 answers, a wide search reaches the label by spurious routes through decoys for almost every query (W = 64, uniform π: 100 % of queries at every k), and the shortest of them is usually not the oracle's. MAP selection makes EI hard EM over the beam: the route most consistent with the current router. Spurious routes remain a real feature of the problem — they are part of the marginal too — and EI's imitation of them is measured (`search_oracle`: the fraction of queries whose kept route is the oracle's).

## 4. Endpoints

All rung-1 endpoints (committed / marginal / sampled accuracy by stratum, route accuracy split by P-words, steps, p(HALT) by distance, the two counterfactual probes, first deviation, held-out criterion curves by k and by depth of P, SET / PREV flip times, head norms), plus:
1. **Episodes to criterion** (held-out route accuracy ≥ 0.95 at every k ≤ 8, confirmed) and **rollouts to criterion** (episodes × queries × rollouts per query).
2. **What the estimator sees**, every 100 steps on the training batches, by k: the fraction of queries on which a sampled route reaches the label / one of the sampled routes is the oracle's (IW, RF); on which the beam found a correct route / the kept route is the oracle's (EI).
3. **The exact objective on the held-out set** (`crit_obj`) for every arm: how far each estimator's router is from the exact optimum in the exact objective's own units.
4. The SET flip time relative to EX's (the setup-credit cost of each estimator).

## 5. Derived before measuring (`derive/search.py`; 2,400 training queries per deferral)

What each estimator sees at the two policies that matter: **uniform** (init) and **stray** (the policy both rung-1 routers reached by step 50: HALT at answers; set-here ⇒ HOP 0.99; carried count ⇒ PREV 0.99; flagged (0, 0) ⇒ SET 0.22; else HOP 0.99).

| | δ = 1, uniform | δ = 1, stray | δ = 0, uniform | δ = 0, stray |
|---|---|---|---|---|
| K = 16 samples contain the oracle route (P-words, k = 3 / 5 / 8) | 0.01 / 0.00 / 0.00 | **0.97 / 0.95 / 0.66** | 0.01 / 0.00 / 0.00 | **0.01 / 0.01 / 0.02** |
| K = 4 samples contain it | 0.01 / 0.00 / 0.00 | 0.58 / 0.47 / 0.42 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| W = 16 beam: MAP correct route = oracle (P-words, k = 3 / 4 / 5 / 8) | 0.03 / 0.25 / 0.05 / 0.02 | **0.99 / 0.89 / 0.95 / 0.86** | **0.94 / 0.43 / 0.09 / 0.00** | **0.00 / 0.00 / 0.00 / 0.00** |
| W = 4 beam: same | 0.03 / 0.02 / 0.00 / 0.00 | 0.99 / 0.89 / 0.93 / 0.88 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 |
| W = 16 beam: MAP = oracle on all-H words (k = 1 / 4 / 6 / 8) | 1.00 / 0.90 / 0.40 / 0.00 | 1.00 everywhere | as δ = 1 | 1.00 everywhere |
| K = 16 samples reach the label at all (k = 1 / 3 / 8) | 1.00 / 0.39 / 0.08 | 1.00 / 0.99 / 0.75 | 1.00 / 0.39 / 0.08 | 1.00 / 0.73 / 0.28 |

Readings that set the predictions:
- **δ = 1:** at init no estimator sees the oracle route on P-words; once the stray rule forms (in rung 1 it formed from the exact marginal's stray SET mass; whether sampled gradients form it too is part of the question) both samplers and the beam see it on most P-words. So IW and RF should follow EX's path, slower by the sampling noise; EI should too.
- **δ = 0:** at the stray policy no estimator sees the oracle route on P-words (≤ 2 %; the beam's MAP correct route is a spurious one, 0.00). The only window is at init: the width-16 beam's MAP correct route is the oracle's on 94 % of k = 3 P-words (43 % at k = 4), and imitation supervises SET and the following PREV at the flagged state *directly*, which no likelihood-weighted estimator does. So EI-16 can escape the local-trigger basin only by learning SET→PREV from short P-words before its own early phase builds the stray rule; EI-4 (0.00 at init) cannot.
- **Spurious routes:** at init the width-16 beam's MAP correct route on long all-H words is mostly not the oracle's (0.40 at k = 6, 0.00 at k = 8); EI imitates noise on long chains until the router sharpens. Zipf weight on k ≥ 6 is 9 %.

## 6. Arms and grid

| arm | δ = 1 seeds | δ = 0 seeds | role |
|---|---|---|---|
| EX | 2 | 2 | the exact-marginal reference (rung 1 has 10 / 5 more seeds on the same code path; reproducibility check) |
| IW-4 | 3 | 3 | the sampled exact objective at a small budget |
| **IW-16** | 5 | 5 | the sampled exact objective (**the headline for δ = 1**) |
| RF-16 | 5 | 5 | the expected-log-likelihood ("RL") objective at the same budget |
| EI-4 | 3 | 3 | search at a small width (the derived null for δ = 0) |
| **EI-16** | 10 | 10 | search + amortization (**the headline for δ = 0**) |

56 runs, 6,000 steps × batch 64 each (EX reaches criterion by 600–1,200 steps: a ≥ 5× margin for the 4× success criterion), stacked by arm. Cost: 0.09 s per run-step for the routed arm (rung-1 pilot), the estimators' extra work is per-query tensor ops (measured in the pilot): ≈ 8.4 T4-hours, ≈ 4.2 h wall on two T4s.

**Pilot (G3):** IW-16, RF-16, EI-16 × lr {3e-4, 1e-3, 3e-3} on δ = 1 (seed 0, 4,000 steps); IW-16, RF-16, EI-16 at lr 1e-3 on δ = 0 (the controls run in the pilot, L26). 12 runs ≈ 1.2 T4-hours. Rule: per estimator, the lr with the highest held-out in-range route accuracy at 4k steps on δ = 1 (ties → 1e-3); applied to both deferrals and both budgets of that estimator; EX keeps rung 1's lr 1e-3.

## 7. Predictions (thresholds fixed here; per-seed counts; run-level bootstrap where paired)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **Sampling suffices for the deferred program (δ = 1).** IW-16 reaches EX's rule (`route_acc_P` ≥ 0.95 on in4 and in8) in 5/5 seeds, with its SET flip and episodes to criterion within **2×** EX's (EX: SET 600, criterion ≤ 1,200 steps); IW-4 within 4× in 3/3. Reason (§5): once the stray rule forms, 16 samples contain the oracle route on 66–100 % of P-word queries, and VIMCO's weights are the exact posterior restricted to the samples. | IW-16 below 0.95 in ≥ 3/5 seeds at 6k steps, or > 4× slower: the sampled bound does not find the setup step within budget — the thesis's "from next-token loss" needs exact inference (rung 2's failure condition). | optimization |
| **P2** | **The RL objective too, more slowly.** RF-16 reaches it in ≥ 4/5 seeds within 4× EX, slower than IW-16 (paired P(RF slower) ≥ 0.8): RLOO credits each route by its own log-likelihood, so a P-word route that reaches the label by coincidence through a decoy earns the same reward as the oracle route's; VIMCO's weights favour routes that make the target likely across the batch. | RF-16 faster than IW-16 (P ≥ 0.8 the other way): the per-route objective is the better learning signal here; or RF fails while IW passes (then the bound's averaging is what carries the setup credit). | optimization |
| **P3** | **The local trigger: only search escapes, and only if it wins the race (δ = 0).** IW-16, IW-4, RF-16 fail like EX (≤ 1 of their 13 seeds reaches `route_acc_P` ≥ 0.95; §5: ≤ 2 % of sample sets contain the oracle route at the stray policy). EI-16 learns δ = 0 in **≥ 5/10 seeds** (moderate confidence; the outcome is a race between imitation of k = 3 P-words, oracle MAP 94 % at init, and the formation of the stray basin, oracle MAP 0 %); EI-4 in 0/3. | EI-16 in ≤ 1/10: search under the current policy closes the window before imitation lands (the beam's MAP selection follows the policy into the basin); then neither inference nor search finds a setup step whose payoff shares its state — F29 holds for every estimator here. A sampled estimator learning δ = 0 in ≥ 2 seeds: sampling noise escapes the basin that the exact gradient could not (a striking result; check the seeds). | optimization (the headline for δ = 0) |
| **P4** | **The generalization profile belongs to the state, not the estimator.** Every run that reaches the rule on δ = 1 shows F30's profile: route accuracy 1.00 at len16 and len32 (within 0.05 of in8), probe (a) ≥ 0.9, probe (b) ≤ 0.2, pp4–pp32 ≤ 0.05 with first deviation SET>PREV. Every run that reaches it on δ = 0 passes pp (≥ 0.9; under the local trigger every state on a PP route is in the training support — rung 1's untested RM-d0 prediction). | A learned δ = 1 run with probe (b) ≥ 0.8 (an estimator fixed the precedence; say which and how), or a learned δ = 0 run failing pp. | mechanism |
| **P5** | **Determinism and cost.** Every run that reaches the rule is deterministic at the end: sampled-route accuracy on P-words ≥ 0.95 and E[steps] within 0.2 of the oracle's (F31's endpoints; RF's per-route objective pushes hardest toward commitment, IW's least). The exact objective on the held-out set (`crit_obj`) of every learned run is within 0.02 nats of EX's. | A learned run with sampled accuracy < 0.9 (it solved the task in the committed reading while keeping a stochastic policy) or `crit_obj` > EX's + 0.05 (a different optimum). | mechanism |

**Reading the outcomes.**
- P1 ∧ P3 (EI-16 escapes δ = 0): the routing rule survives the loss of exact inference on the deferred trigger, and search + amortization does something the exact marginal does not — it supervises a setup step and its payoff directly from a found route. The thesis's search-based learner (D75's later E-step) earns its place on a measured case; rung 3 can then ask the neural questions with the estimator chosen.
- P1 ∧ ¬P3: sampling suffices where the exact marginal sufficed, and nothing here escapes a basin whose setup and payoff share a state. F29 becomes a statement about likelihood-driven learning in general, not about one estimator, and the remedy must be in the state (D103 (i)) or in exploration, not in inference.
- ¬P1: sampled inference does not find the setup step within 4× — the program's central "learned from next-token loss" claim depends on exact inference. Then the question becomes what minimal search or curriculum restores it, before anything else in the ladder.
- P2's direction reads which objective carries the setup credit when the E-step is sampled — the marginal's averaging (IW) or the per-route reward (RF) — and connects to F31 (commit vs average).

## 8. Gates

- **G1 tests:** the estimator gradients against the exact gradients (RLOO vs ∇E[log p] by the recursion with the log readout; VIMCO at large K vs ∇log q); the beam against the oracle policy and against brute force on small horizons; `replay` against `oracle`; trainer smoke for every estimator; the rung-1 tests still pass.
- **G2 derive:** §5 (done, `derive/search.py`).
- **G3 pilot:** §6; the lr rule; the cost per estimator; the controls (δ = 0) run in the pilot.
- **G4 reviews:** an independent design review and an independent code review before the pilot.
- **CPU sanity (L25):** EI-16 on δ = 0 and IW-16 / EI-16 / RF-16 on δ = 1 at batch 16 for 1,500 steps before the reviews; reported in §9 as observed, not predicted.

## 9. CPU sanity run (before the reviews)

*To be filled in from `scratchpad/rf_cpu` before the reviews are requested.*

## 10. Fairness (AGENTS rule 4)

- One model, one task, one data stream per seed; the arms differ only in the training procedure. Learning rate tuned per estimator in the pilot; best-vs-best reported with curves.
- Budget: the same 6,000 × 64 episodes per run; rollouts per query reported (EX evaluates all 390 states per episode, which is not a rollout count; the comparison of interest is episodes and rollouts, not FLOPs, at this size).
- Seeds: 10 for the δ = 0 headline arm (EI-16), 5 for IW-16 / RF-16, 3 for the budget arms, 2 for EX (with rung 1's 15 EX seeds on the same code path as the reference distribution); per-seed counts in every prediction.
- The label is used by every arm only through p(y | route end); EI's selection uses the same quantity.

## 11. What rung 2 does not answer

Neural primitives or states (rung 3 as re-planned); a learned value or MCTS (B6); exploration bonuses or curricula; the planted-gate block (D99); the precedence dose–response (D100); whether any of this scales with the state space (the reference recursion is kept; only its use is withheld).

## 12. Prior work

Score-function estimators for discrete latent structure: REINFORCE (Williams 1992), RLOO (Kool et al. 2019), VIMCO / multi-sample bounds (Mnih & Rezende 2016; Burda et al. 2016 for IWAE). Learning latent programs from denotations with spurious programs: Liang et al. 2017 (Neural Symbolic Machines, MAPO 2018 — memory-augmented policy optimization, exactly the spurious-route problem of §3), Guu et al. 2017 (from language to programs: RL vs MML, "randomized beam search" — the closest prior comparison of the marginal and the RL objective under spurious programs). Expert iteration (Anthony et al. 2017), AlphaZero, hard EM for weak supervision (Min et al. 2019). Neural Program Lattices (Li 2017). What is new here, to our reading: the comparison on a carrier where the exact marginal is available as a reference on the same seeds, with a setup step whose credit is known (F28–F29) to flow through the state, and with exact references for what each estimator sees.
