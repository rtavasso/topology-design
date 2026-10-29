# Rung 2: the router's training signal (revision 4)

**Status:** design, 29 Oct 2026. Revision 4 applies design review 3 and code review 3 ([review3_design.md](review/review3_design.md), [review3_code.md](review/review3_code.md); handling in §15) to revision 3 (`f34b73d`). Revision 3 re-scoped rung 2 after the direction discussion of Entry 65 (D105–D106). It keeps the implementation and the review handling of revisions 1–2 (`e886574`, `43726e7`; §14). Code: `77ca591` and later.

**Direction:** the ledger's "Current orientation" (Entry 65). The router is amortized inference plus a contextual value of each layer. Rung 2 retires **risk 3** (a router training signal that generalizes without exact inference) and opens **risk 4** (contextual value of computation). The design rule is L27: the core is the cells whose outcome cannot be derived; derivable cells are checks at minimal seeds.

**Scope note (AGENTS hard rule 1).** One fixed model (the rung-1 router), one task family; the arms differ only in the training signal. Nothing neural is added. Every claim is labeled.

---

## 1. The question

**Q5′.** Which training signal teaches a router *the contextual value of a layer* — which primitive to apply next, given the sequence and the token's current computational state — in a way that generalizes, when the exact route posterior is not available?

**Two kinds of arm.**
- **Exact references** enumerate the 390 product states: EX (rung 1's exact current-policy marginal), EX-AL (the same with the answer-only likelihood) and PF (the exact fixed-prior posterior). They are not candidates for risk 3. They locate F29's basin.
- **Candidates** see only the routes they sample or search: IW16, PF-IS16, TB16, RL16, ACMC16, AC16 and EI16.

**The contrasts** (§3 defines each signal):

| contrast | arms | what differs | why it is open |
|---|---|---|---|
| likelihood | EX vs EX-AL (δ = 0) | the readout of a halt at a key: 1/16 vs 0 | EX-AL's step-0 target equals PF's to 1 % (§5.1); whether it escapes F29's basin is a question about dynamics |
| whose posterior, exact | EX-AL vs PF (δ = 0) | the router's own policy vs a fixed prior as the prior over routes | PF's end state is derived (§5.2); whether a self-consistent posterior with the same likelihood gets there is not |
| whose posterior, sampled | IW16 vs PF-IS16 | importance weights p(y \| r) vs π₀(r) e^{−λ·steps} L_y(r) / π(r), and the likelihood | sampling, collapse modes |
| KL direction | PF-IS16 vs TB16 | forward vs reverse KL to the same fixed posterior | mode-covering vs mode-seeking; TB's entropy term counters IW16's collapse mode |
| contextual value | RL16 vs ACMC16 vs AC16 | route-level baselines vs a learned state value V(x) as baseline vs bootstrapping through V(x) (TD(λ = 0.9)) | setup steps have zero immediate value (F28, D74) |
| search + imitation | EI16 | hard EM on the beam's MAP correct route | which program the beam hands it |

**Why these, from what rung 1 left.**
- F29: the exact current-policy marginal stalls under the local trigger. Review 3 showed that EX's step-0 target already points the right way in every state class (§5.1), so the basin forms in the training dynamics. The likelihood contrast and the exact "whose posterior" contrast separate the two candidate causes:
  - the readout dilution (a halt at a key explains any label at 1/16);
  - self-consistency (the posterior is taken under the router's own policy, so early stray mass reinforces itself).
- F28 / D74: setup steps have zero immediate value. Monte-Carlo returns do credit them; what differs is variance and bias. RL16 → ACMC16 → AC16 separates a learned state baseline from bootstrapping.
- Two equally short programs exist (§2). Which one a signal lands in measures its inductive bias.
- Real routing has many wrong-but-right routes. The answer-pool dose (§6) raises the coincidence rate of every wrong route, with the readout held fixed.

**Success, stated before running (revised, review 3 B3).** Rung 2 succeeds if at least one **candidate** reaches, in a majority of its seeds on δ = 1:
- shortest-correct **committed** route accuracy ≥ 0.95 on P-words at in4 and in8;
- the held-out criterion within **3× EX's median criterion step** in this grid.

It fails if no candidate does. Determinism (sampled-route accuracy) is reported separately. A router that matches a soft posterior is stochastic by derivation: PF's fixed point samples shortest-correct routes on 0.00–0.42 of P-words (§5.2). The δ = 0 and dose cells are read for *which* signals survive.

**Not asked here.** Neural primitives or states (risk 2; risk 1, the basis claim, is the next rung's candidate, D106). MCTS or search guided by a learned value (B6). Curricula. D99, D100.

## 2. Task, model, executor, and the two programs

Unchanged from rung 1 revision 2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4):
- 48 keys, 16 answers, decoys at P-steps, the flag deferred (δ = 1) or local (δ = 0);
- the product state (symbol, pending c ∈ {0,1,2}, set-here h) and the actions HOP / PREV / SET / HALT;
- the Markov router: one cross-attention layer from the state token over the pair tokens, no attention among pairs;
- readout ε = 0.02, step cost λ = 0.02, T_train = 14;
- the strata and probes, and the held-out criterion set (D97).

**Two shortest correct programs per trigger.** PREV needs only a pending count, so a P-step can be prepared in two places:
- **E** ("early"): SET one state *before* the P-leaving state, carry the count across a hop, PREV.
- **L** ("late"): SET *at* the P-leaving state, PREV at once.

The oracle is E under δ = 1 and L under δ = 0. Hand-built policies for both reach every label in the oracle's number of moves (checked by code review 3).

**Expressivity with one lookup per decision:**

| | needs to read | δ = 1 | δ = 0 |
|---|---|---|---|
| E | at the predecessor: is the P-step leaving my *successor*? | own pair's flag (one hop): **expressible** | the flag of the pair my value points to (two hops): **not expressible** |
| L | at the P-leaving state: is my own exit a P-step? | the flag of the pair whose value is my key (one value-slot match): **expressible** | own pair's flag (one hop): **expressible** |

- Under δ = 1 a signal can land in E or L; under δ = 0 only L is reachable.
- Under δ = 1, 9.1 % of P-leaving states have an ambiguous value-slot read (two pairs point to them; review 3). This is a possible bias toward E.

**Program identity** (revised, review 3 S9). Each SET of a **shortest correct** committed route is classified positively:
- **late:** at a P-leaving symbol;
- **early:** the route's next non-SET action is a HOP onto a P-leaving symbol;
- **other:** neither.

A run's program on P-words at in8 is labeled L (`set_late` ≥ 0.75), E (≤ 0.25), mixed, or other (`set_other` > 0.25). Routes that are not shortest-correct are not classified.

## 3. The training signals (`route/estimators.py`, `route/train_route.py`)

Notation:
- π₀ is the flat prior: uniform at keys, HALT at answers and dead.
- L_y is the answer-only likelihood: 1 − ε on the label, ε/15 on another answer, 0 on a key or dead.
- The rung-1 readout instead gives 1/16 to a halt anywhere other than an answer.

| arm | signal | per query |
|---|---|---|
| **EX** | exact current-policy marginal, rung-1 readout: −log q(y) + λ E[steps]; gradient E_{p_π(r\|y)}[∇log π] (EM) | all states (reference) |
| **EX-AL** | the same with L_y in place of the readout | all states (reference) |
| **PF** | exact fixed-prior posterior p₀(r \| y) ∝ π₀(r) e^{−λ·steps} L_y(r), by forward–backward; loss −Σ counts · log π, gradient E_{p₀(r\|y)}[∇log π]: the PFN signal | all states (reference) |
| **IW16** | the K-sample bound on log q(y), rung-1 readout, VIMCO (current-policy posterior, sampled) | 16 samples |
| **PF-IS16** | PF's forward KL from on-policy samples: self-normalized importance weights w_k ∝ π₀(r_k) e^{−λ·steps_k} L_y(r_k) / π(r_k), loss −Σ w̃_k log π(r_k) | 16 samples |
| **TB16** | trajectory balance toward the same fixed posterior, log Z as the per-query sample mean (VarGrad). On-policy this is the **reverse** KL(π ‖ p₀(·\|y)): entropy-regularized REINFORCE on log R (Richter et al. 2020); not "PF sampled" | 16 samples |
| **RL16** | REINFORCE, reward p(y \| r) − λ·steps (rung-1 readout), RLOO baselines | 16 samples |
| **ACMC16** | actor-critic with the router's learned state value V(x) as baseline: Monte-Carlo returns (gae = 1), the same reward | 16 samples |
| **AC16** | the same, **bootstrapped**: TD(λ) advantages with λ = 0.9 (gae = 0.9) | 16 samples |
| **EI16** | beam MAP correct route (hard EM), teacher forcing on it | beam W = 16 |

Checks, at 2 seeds each (derived, or observed on CPU in revision 2):
- **EI16B:** EI16 started after 1,000 EX steps inside EX's δ = 0 basin; derived to fail.
- **SI16:** sample-then-imitate; derived to see no oracle route on P-words at init.
- **ELL16:** the K = 1 bound; observed failing.
- **AC16T0:** AC16 with pure TD(0) (gae = 0); observed to collapse into halting at the start (§5.5, §9).

**Canonical routes** (code review 3, bug 1). TB16 and PF-IS16 score each sampled route cut at its first arrival at an answer or dead state, which absorb.
- **Why the cut.** Without it, no-op moves at an answer score log π₀ = log 10⁻¹² and dominate TB's loss (41 % of routes at init).
- **The cut route ends there with no action**, as at the forced halt at T: π₀ halts at an answer or dead with probability 1. A first version kept a log π(HALT | arrival) term. TB16 could shrink it freely, which made routes into dead look under-weighted. The CPU run collapsed into PREV-to-dead (§9).
- **Why no HALT term is the correct objective, not a tuning choice** (review 4): under the router, the probability of the cut route is the probability of its prefix, because everything after the arrival sums to 1. So TB16 is still the reverse KL to the fixed posterior over routes as the executor reads them.
- **Halting at the arrival is taught by a separate local term,** −log π(HALT | arrival state). This is the prior knowledge π₀ encodes. RL16 and IW16 learn halting at answers from the step cost instead.
  - Without the term, PF-IS16 pushes π(HALT | answer) to 0.002 within 50 steps, and its committed routes could never count as shortest.
  - At init its gradient is about as large as PF-IS16's own (norm 21–24 vs 23–28) and pushes HALT up at keys too (review 4, S5). This is reported as a difference between the families (§7 endpoints).

**The critic warm-up.** The actor-critic arms train only the critic for their first 200 steps (policy evaluation before improvement); the actor switches on at step 200.

**The value head** (code review 3, bug 2; design review 3, B4):
- **Architecture:** a two-layer MLP on the router's trunk features, through a stop-gradient. The critic's loss trains only the head, so AC differs from RL16 only in the advantage.
- **Init:** its output layer is zero, so V = 0 at init. It is built after the policy head, so at a seed the policy parameters equal RL16's (test).
- **Inputs:** V(x) sees the pair tokens and the state, never the label or the query.
- **Caveat:** V is per episode and per state, blind to which query is being routed. 25 % of P-step decoys land on another query's oracle route, where TD(0) bootstraps a stray route's value from the owner's correct route (code review 3, item 5).

**Why the answer-only likelihood for PF, PF-IS16, TB16 and EX-AL.** With the rung-1 readout, the flat-prior posterior is dominated by "halt at once and guess": 1/4 × 1/16 per query beats every real computation. Teacher forcing on the oracle (rung 1's RM-T) is the full-information limit of PF: the posterior given the generating process.

**Checked before measuring (G1; tests in `tests/test_routefind.py`).**
- **The fixed-prior posterior.** Forward–backward equals brute-force enumeration of every route for both deferrals: Z to 10⁻⁹ relative, counts to 10⁻¹². Code review 3 confirmed it independently, to 3·10⁻¹⁵, with 4 answers as well.
- **PF's gradient.** It equals E_{p₀(r|y)}[∇log π] by enumeration.
- **TB16.**
  - The loss is zero exactly when the log-ratios agree.
  - A looped route and a cleanly halting route give the same canonical route and the same residual.
- **PF-IS16.** Its gradient is the self-normalized weights; the effective sample size is reported, and queries with no answer reached give no gradient.
- **The actor-critic.**
  - TD(0) critic targets and Monte-Carlo advantages match hand computation, with nothing bootstrapped past the end.
  - The value head is zero at init, takes no gradient into the trunk, and leaves the policy init unchanged.
- **Program classification.** Hand-built E, L and other routes are classified 0 / 1 / other; a route that is not shortest is not classified.
- **The answer pool.** It limits each episode to its distinct answers, and leaves the default data stream unchanged.
- **The revision-2 tests.** Exact K = 2 unbiasedness of IW, ELL and RL by enumeration, and the beam against brute force, still pass.

## 4. Endpoints

As in revision 2 (§4 there):
- shortest-correct-route accuracy and program identity, by stratum and on the held-out curve;
- oracle-route accuracy;
- committed, marginal and sampled readings;
- the counterfactual probes;
- the first deviation;
- episodes and rollouts to criterion (on shortest-correct accuracy, confirmed);
- `crit_obj` (the exact current-policy objective on the held-out set);
- gradient-clip binding;
- what each sampled or searched signal sees, at init and every 100 steps, on P-words by k.

New in revision 4:
- **Program identity, positively classified** on shortest-correct routes (§2).
- **PP precedence** (pp4 / pp8 route accuracy) for every learned run. This is an endpoint, not a check (review 3, S3): the precedence at the unseen conjunction is set by training dynamics (F30), which differ by signal.
- **Determinism:** sampled shortest-correct accuracy on P-words, reported apart from success.
- **The critic:** mean |TD error| and explained variance of the Monte-Carlo return.
- **PF-IS16:** effective sample size / K, and the fraction of queries whose samples reach any answer.
- **EI16's init beam program** per seed: the `set_late` of the beam's MAP correct routes at step 0.

## 5. Derived before measuring

### 5.1 The step-0 gradient (review 3, B2; reproduced)

The descent direction at a uniform policy, summed per one-lookup feature class (`exgrad.py`, `exal.py` in review 3's scratch; 100 episodes, δ = 0):

| class (own flag, c, h) | EX (rung-1 readout) | EX-AL | PF |
|---|---|---|---|
| unflagged, (0, 0) | HOP +189, SET **+0.7** | HOP +359, SET **+55** | HOP +367, SET **+54** |
| flagged, (0, 0) | SET +1.4 | SET +17.3 | SET +17.3 |
| flagged, (1, 0) | PREV +1.2 | PREV +13.3 | PREV +13.5 |

What this shows:
- **EX-AL and PF agree to within 1 % per class.** At step 0 the router's policy is the flat prior, so "whose posterior" contributes nothing yet.
- **The whole step-0 difference between EX and PF is the likelihood.** It gives about 12× more signal at flagged states, and a stray-SET push at unflagged states of +55 vs +0.7.
- **EX's own step-0 direction is correct in every class.** Pooled by class, its target's argmax is the oracle (`derive/pooled.py`, "EXflat": committed route 1.00 on every P-stratum).

So F29's basin is built by the dynamics, not present in the initial target. The two contrasts of §1 ask which ingredient of the dynamics builds it.

### 5.2 PF's fixed point (review 3, B1; `derive/pooled.py`, 150 episodes per cell)

A router that fits a fixed target by cross-entropy converges to the target pooled over the features it can distinguish. Pooled by the one-lookup classes (answer / key / dead, own flag, value-slot flag, c, h):

| cell | decisive class targets | committed route of the fixed point (P-words in3 / in4 / in5 / in8; all-H) | pp4 / 6 / 8 | sampled shortest-correct (P-words in3 / in4 / in5 / in8) |
|---|---|---|---|---|
| δ = 1 | own-flagged (0, 0): SET 0.60 / HOP 0.40; pred-flagged (0, 0): SET 0.83; pred-flagged c ≥ 1: PREV 0.63–0.68 | **1.00 everywhere: program E** | 0.03 / 0.00 / 0.00 | 0.42 / 0.26 / 0.29 / 0.00 |
| δ = 0 | own-flagged (0, 0): SET 0.83; own-flagged c ≥ 1: PREV 0.64–0.69 | **1.00 everywhere: program L** | **1.00 / 1.00 / 1.00** | 0.46 / 0.33 / 0.36 / 0.00 |
| δ = 1, answer pool 4 | the same argmaxes | **1.00: E** | 0.05 / 0.01 / 0.01 | 0.18 / 0.18 / 0.18 / 0.05 |
| δ = 0, answer pool 4 | the same argmaxes | **1.00: L** | 1.00 | 0.21 / 0.18 / 0.36 / 0.05 |

PF's end state is therefore derived: the oracle's program, PP failing under δ = 1 and passing under δ = 0, and stochastic. **PF is a 2-seed check per deferral.** What it tests is whether a fixed-target router reaches its derived fixed point within budget.

A per-episode argmax (no pooling) is worse on long chains: shortest-correct 0.94 / 0.75 / 0.11 at k = 3 / 5 / 8 (`derive/signals.py`). So the derivation holds only if the router pools across episodes.

### 5.3 What the fixed-prior posterior contains (`derive/signals.py`, 1,800 queries per cell; δ = 1, P-words by k)

| k | mass on E / L / other routes to the label / wrong answer | same, answer pool 4 | all-H words: mass on the oracle route (pool 0 / 4) |
|---|---|---|---|
| 3 | 0.23 / 0.23 / 0.53 / 0.00 | 0.17 / 0.17 / 0.65 / 0.00 | 0.40 / 0.36 |
| 5 | 0.07 / 0.07 / 0.84 / 0.02 | 0.04 / 0.04 / 0.91 / 0.01 | 0.16 / 0.13 |
| 8 | 0.00 / 0.00 / 0.86 / 0.13 | 0.00 / 0.00 / 0.99 / 0.01 | 0.02 / 0.00 |

- "Other" is mostly near-oracle routes with redundant SETs (the flat prior gives each extra SET 1/4), plus decoy routes that end on the label.
- The target at E's SET states splits SET 0.60 / HOP 0.40 under δ = 1, at both pools.
- With a prior step cost of 1.5 nats per move the split shrinks to 0.54 / 0.46 (review 3). Part of PF's preference for E is therefore a prior artefact.

### 5.4 What each sampled or searched signal sees (`derive/search.py`; the real initialized routers, seeds 0–4; P-words, k = 3)

| | δ = 1 | δ = 1, pool 4 | δ = 0 |
|---|---|---|---|
| 16 samples contain the oracle route, at init | 0.00 | 0.00–0.03 | 0.00–0.02 |
| at the stray policy | 1.00 | 1.00 | 0.03 |
| beam MAP correct route = oracle, at init | 0.10–0.71 | 0.10–0.65 | 0.26–0.87 |
| at the stray policy | 1.00 | **0.74** | 0.00 |
| 16 samples reach the label, at init | 0.10–0.32 | 0.10–0.61 | 0.10–0.35 |

The dose matters for imitation:
- Spurious correct routes outscore the oracle in the beam on a quarter of P-words at the stray policy.
- The all-H program's coincidence rate on P-words rises from 1/16 to 1/4. This is the one spurious *program* a Markov router can hold; decoy-specific routes pool out (§5.2).

The pooled argmaxes of the posterior targets do not move.

### 5.5 How the value arms start (code review 3, S3; the CPU runs of §9)

- **With V = 0, TD(0) gives every non-final action the advantage −λ,** whatever the outcome; only the final action sees the reward. A halt at a key earns +1/16, so TD(0) pushes toward halting from the first step (CPU: collapse within 100 steps).
- **A warmed critic does not remove the trap.** Under the init policy, the value of the start state is below the 1/16 that a halt at a key earns. So one-step bootstrapped improvement prefers halting there. Once every route halts at the start, no continuation is sampled again and V cannot learn better (CPU: collapse within 100 actor steps after a 200-step warm-up, critic explained variance 0.51 at the switch).
- **This myopic trap is a property of one-step bootstrapping against a poor policy, not of contextual value as such.** TD(λ = 0.9) mixes in enough of the Monte-Carlo return to avoid it (CPU). So AC16 uses λ = 0.9, and TD(0) is the check AC16T0.
- **ACMC16** (λ = 1) starts as REINFORCE with a zero baseline. Its first updates differ from RL16's in the baseline, not the return.

## 6. Arms and grid (`make_grids.py`: `rf_pilot.json`, `rf_main.json`)

| cell | EX | EX-AL | PF | IW16 | PF-IS16 | TB16 | RL16 | ACMC16 | AC16 | EI16 |
|---|---|---|---|---|---|---|---|---|---|---|
| δ = 1 | 3 | 2 | 2 (check) | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| δ = 0 | 2 | **3** | 2 (check) | 2 | 3 | 3 | 3 | — | 3 | 2 |
| δ = 1, answer pool 4 (the dose) | 2 | — | — | 3 | 3 | 3 | 3 | — | 3 | 3 |

Plus the checks: SI16, ELL16, AC16T0 (δ = 1), EI16B (δ = 0), 2 seeds each.

**The dose** (revised, review 3 S2). Each episode's chains end in one of 4 answers drawn per episode from all 16 (`ans_pool = 4`). The readout and the likelihood keep 16 answers, so their constants do not change. Only the coincidence rate of wrong routes rises: decoys and the all-H program land on the label with probability 1/4 instead of 1/16.

**Budget.**
- **Size:** 93 runs × **5,000 steps** × batch 64.
- **Schedule:** the cosine schedule is at 15 % of the peak lr by 80 % of training (review 3, S6), so the step count was raised from 4,000. Success counts the criterion within 3× EX's median in this grid, on the same schedule; rung-1 EX runs (6,000-step schedule) are a reference only.
- **Seeding:** every run has its own init and sampling seed (`per_run_init`).
- **Cost:** ≈ 11.6 T4-hours at the EX rate (0.09 s per run-step), ≈ 5.8 h wall. Sampled and exact-posterior arms cost more per step; the pilot measures it.

**Seeds.** 5 / 3 / 2 per cell is below AGENTS rule 4. It is defensible only for count predictions on near-bimodal outcomes. Paired speed comparisons are exploratory.

**Pilot (G3).**
- The sweep: the seven candidates × lr {3e-4, 1e-3, 3e-3} on δ = 1.
- The controls: EX-AL, PF-IS16, TB16, RL16 and AC16 on δ = 0 at lr 1e-3 (L26), plus EX-AL on δ = 0 at 3e-3, so that P2(a) does not rest on one lr (review 4, S7).
- Seed: **100**, not a main-grid seed (review 3, S5).
- Size: 27 runs ≈ 3.4 T4-hours at the EX rate.
- The main grid's lr values are placeholders (1e-3) until the pilot rule is applied; `make_grids.py`'s `LR` is set from the pilot before the main grid is generated.
- **Rule:** per signal, the lr with the highest held-out shortest-correct accuracy at 5k steps on δ = 1. It applies to every cell of that signal.
- **Ties** go to the lr with the higher mean committed shortest-correct accuracy over the last 1,000 steps, then to 1e-3. A signal that fails at every lr is reported as such.
- EX, EX-AL and PF keep 1e-3.

## 7. Predictions (core cells only; thresholds fixed here; per-seed counts)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **The sampled current-policy posterior (IW16), δ = 1.** Success in **≥ 3/5** seeds, in program E. One revision-2 CPU seed collapsed into "SET everywhere": identical samples give zero leave-one-out advantages. | ≤ 1/5. 2/5 is inconclusive and reported as such. | optimization |
| **P2** | **Where F29's basin comes from (δ = 0).** The CPU sanity runs (§9) point one way: at 7.5 % of a main run's episodes, EX and EX-AL both stall in all-H while PF has started learning L. Hence:<br>(a) **EX-AL learns in ≤ 1/3**: the answer-only likelihood alone does not prevent the basin.<br>(b) **PF-IS16 learns in ≥ 2/3**, in L: a sampled fixed target is not trapped either. | (a) EX-AL ≥ 2/3: the readout dilution was the cause; self-consistency with the right likelihood escapes.<br>(b) PF-IS16 ≤ 1/3 while PF learns: sampling cannot find the fixed target's support where the policy is stuck. | mechanism (F29, causal) |
| **P3** | **KL direction (PF-IS16 vs TB16), δ = 1.** PF-IS16 succeeds in **at least 2 more of 5 seeds** than TB16. If both succeed equally often, PF-IS16 reaches criterion earlier in every pair where both learn (≥ 2 such pairs). On CPU (§9), PF-IS16 shows the first sign of L at 1,300 steps, as exact PF did; TB16 shows none. Clipping does not separate them: both clip on most steps early (review 4). TB16's update is also shaped by its 10⁻⁶ reward floor (init gradient norm 74–91 vs 30–33 at 10⁻²). That is a confound of the TB arm, reported, not tuned. | TB16 succeeds in at least as many seeds as PF-IS16 (each ≥ 2/5), and no later where both learn. **Uninterpretable** if both succeed in ≤ 1/5. A one-seed gap is inconclusive. | optimization |
| **P4** | **Contextual value (RL16, ACMC16, AC16), δ = 1; 5 seeds each.** **No advantage from the learned value at this horizon:** AC16 and ACMC16 each learn within ±1 seed of RL16, and their criterion steps, in pairs where both learn, are within 25 % of RL16's. On CPU all three sit in all-H at 1,500 steps, and the query-blind critic explains ≈ 5 % of the return's variance (§9). A value that explains so little cannot credit a setup step better than the returns it is fit to. **Uninterpretable** if RL16 is at ceiling (criterion ≤ 1.5× EX) or at floor (all-H or halting at 5k in all pilot lrs). Then P4 moves to longer setup distances (§11) instead of being read. | **Better:** AC16 or ACMC16 learns in ≥ 2 more seeds than RL16, or reaches criterion ≥ 25 % earlier in every pair where both learn (≥ 3 pairs). That is evidence that a learned contextual value credits setup steps.<br>**Worse:** AC16 or ACMC16 learns in ≥ 2 fewer seeds than RL16. That also refutes "no difference"; see the reading table. | mechanism (risk 4) |
| **P5** | **Which program each signal lands in.**<br>δ = 1: EX and IW16 land in **E** in the majority of learned seeds (rung 1's EX learned E).<br>EX-AL, PF-IS16 and the PF check: **open**. EX-AL's step-0 target equals PF's (§5.1); PF's derived fixed point is E (§5.2), but on CPU exact PF and PF-IS16 head for L first (§9).<br>EI16: its learned program **matches its init beam program** (E if the init beam's `set_late` < 0.5) in every learned seed but at most one, when ≥ 3 seeds learn. Imitation reinforces whichever shortest route the beam hands it first; the init beam MAP is E on 0.10–0.71 of P-words, depending on the seed. With fewer than 3 learned seeds it is reported only.<br>δ = 0: every learned run is **L** (expressivity, §2). | Any learned δ = 0 run labeled E, mixed or other: a bug, or an unanticipated read. EI16 mismatching its init beam program in ≥ 2 learned seeds. EX or IW16 landing in L in the majority of its learned seeds (≥ 3 learned). | mechanism (inductive bias) |
| **P6** | **The dose (answer pool 4), δ = 1 — exploratory, strict margins; arms that fail at pool 0 are excluded.** The reward and imitation signals RL16, AC16 and EI16 see the all-H coincidence rise four-fold and the beam's oracle share fall to 0.74 (§5.4). They are predicted to degrade: a learned fraction lower by ≥ 0.33 than at pool 0 (3 dose seeds vs 5), or a criterion step later by ≥ 25 % in every seed pair where both learn. PF-IS16, whose fixed-target argmaxes do not move (§5.2), is predicted **not** to degrade by those margins. IW16 is open: its posterior is the router's own, which is what P2 tests. | RL16, AC16 and EI16 all inside both margins at the dose, or PF-IS16 degrading beyond them. | optimization |

**Endpoints reported without prediction.**
- PP precedence for every learned run, on shortest-correct accuracy at pp8 (any program; oracle-route accuracy would fail L runs by construction). Derived for PF only: it fails δ = 1 at the conjunction, where HOP wins and not PREV (§5.2), and passes δ = 0.
- Determinism: sampled shortest-correct accuracy (`samp_min_acc`).
- The critic's explained variance; distinct sampled routes per P-word query for every sampled arm (ESS / K alone hides collapse: identical samples give 1).
- Runs whose committed answer is right but which do not halt on arrival (p_halt at the answer < 0.5). These separate the families' halting signals: TB16 and PF-IS16 have a local halt loss, the reward arms only the step cost (§3).

**Checks (reported, not predicted).**
- PF reaches its derived fixed point (§5.2): committed oracle program, PP fails δ = 1 and passes δ = 0, sampled shortest-correct ≤ 0.5 at in4.
- EX δ = 0 stalls (rung 1's 5/5).
- EI16B, SI16 and ELL16 fail.
- AC16T0 collapses into halting at the start (§5.5).
- EI16 learns δ = 0 in L (replicating revision 2's CPU seed).

**A shared confound of the reward arms.** RL16, ACMC16 and AC16 are rewarded by the rung-1 readout, which pays 1/16 for halting at a key. That is the same dilution EX vs EX-AL tests, and it is what makes AC16T0's halting trap possible (§5.5). If P2(a) is refuted (EX-AL learns δ = 0), the reward arms' failures must be re-read with that in mind (review 4).

**Reading the outcomes.**

| outcome | reading (which risk it moves) |
|---|---|
| EX-AL stalls on δ = 0, PF and PF-IS16 learn (P2 as predicted) | F29's basin is built by self-consistency: the posterior under the router's own policy reinforces early stray mass. A fixed-prior (PFN-style) target, exact or sampled, does not trap itself. Risk 3 moves toward amortized-posterior training. |
| EX-AL learns δ = 0 | The basin came from the readout's 1/16 credit for halting at a key. Self-consistent EM with an answer-only likelihood is enough. The fix is in the likelihood, not the prior. |
| PF learns, PF-IS16 stalls on δ = 0 | The fixed target is right, but on-policy samples cannot reach its support from the stalled policy. Amortized inference needs exploration or search to supply targets. |
| PF does not learn (committed shortest-correct < 0.95 at in4 / in8) | The router cannot fit a fixed soft target within budget, or a bug. Investigate before reading the PF-based rows. |
| PF learns, in L on δ = 1 | The one-lookup pooling derivation got the program wrong (§5.2): the router's function class pools differently, or the transient seen on CPU does not resolve within budget. The other rows are read normally. |
| TB16 and PF-IS16 differ (P3) | The KL direction decides the program or the speed. A mode-seeking objective with an entropy bonus behaves unlike forward KL to the same posterior. |
| IW16 collapses (P1 false) while PF-IS16 learns δ = 1 | The current-policy weights are the fragile part under sampling, not sampled learning as such. |
| IW16 collapses and TB16 learns, but PF-IS16 does not | TB's entropy term rescues it; this does not show that the fixed posterior matters. |
| ACMC16 beats RL16 | A learned state baseline reduces variance enough to matter, with no bootstrapping. |
| AC16 beats ACMC16 | Bootstrapping through a learned contextual value credits setup steps faster than Monte-Carlo returns. This is the first evidence that "contextual layer value" needs a value function (risk 4). |
| AC16 slower than ACMC16 or RL16 | At this horizon, TD(0.9)'s bias from a weak, query-blind critic costs more than its variance saves. Risk 4's question moves to longer setup distances and a query-aware critic. |
| AC16 or ACMC16 clearly worse than RL16 (≥ 2 fewer seeds) | A critic explaining ≈ 5 % of the return's variance is a worse baseline than RL16's leave-one-out mean. "Contextual value" needs a critic that sees the query. |
| RL16, ACMC16 and AC16 all fail | Reward-only signals cannot find a setup step within budget at this density. Risk 4 needs a denser signal (a posterior target) for the router. |
| Signals split by program (P5) | Which program a learner gets is a property of its training signal. Later rungs must measure program identity, not only accuracy. |
| PP precedence differs between signals | The generalization profile at unseen conjunctions is set by the training signal, not only by the representation. |
| Reward and imitation signals degrade at the dose, averaging signals don't (P6) | Averaging is the robust signal when spurious routes abound. Min et al.'s hard-EM advantage needs consistent spurious programs, which a Markov router does not have here. |
| The dose leaves everything unchanged, or improves a signal | Spurious density at this level is not a bottleneck. A denser reward (more label-reaching routes) can help exploration. |
| Success only through EI16 | Search + imitation is the one non-enumerating signal that works here. Risk 3's answer is to supply the router's targets by search, not by sampled inference or reward. The rung-3 router needs a search-and-imitate loop and its cost. |
| TB16 learns δ = 1 but fails δ = 0 | The reverse KL and its entropy term find the program when the stray rule is not an attractor, but not against the basin. |
| No candidate succeeds on δ = 1 | "Learned from next-token loss" depends on exact inference. The program must fix this before scaling anything. |

## 8. Gates

- **G1 tests:** 32 rung-2 tests (§3), the 14 rung-1 tests and the full suite.
- **G2 derive:** §5 (done).
- **G3 pilot:** §6.
- **G4 reviews:**
  - review 3 (design and code), handled in §15;
  - a short re-review of P3, P4 and §9 before the pilot (review 3, S10).
- **CPU sanity (L25):** §9.

## 9. CPU sanity runs (batch 16, lr 1e-3, 1,500 steps ≈ 7.5 % of one main run's episodes, seed 0)

These are a sanity check and a direction for P2–P4, not data.

| run | outcome at 1,500 steps | revision |
|---|---|---|
| EI16 δ = 0 | learns, program L | 2 |
| IW16 δ = 1 | collapses into "SET everywhere" | 2 |
| EI16 δ = 1 | lands in L | 2 |
| ELL16 δ = 1 | all-H | 2 |
| PF δ = 1 | all-H until ≈ 900 steps, then moving toward **L** (set_late 0.78–1.0; teacher-forced PREV 0.29, SET 0.03; min_acc 0.84) | 3 |
| PF δ = 0 | all-H until ≈ 1,000 steps, then moving toward **L** (set_late 0.87; PREV / SET 0.13; min_acc 0.83) | 3 |
| EX δ = 0 | all-H throughout (SET ≤ 0.04) | 4 |
| EX-AL δ = 0 | all-H throughout (SET 0.00) | 4 |
| RL16 δ = 1 | all-H throughout (SET 0.00) | 4 |
| AC16, TD(0), zero critic | **halts at the start** within 100 steps (min_acc 0.00) | 4 |
| AC16, TD(0), 200-step critic warm-up | the same, within 100 actor steps (explained variance 0.51 at the switch) | 4 |
| ACMC16 (λ = 1), warm-up | brief halting, then all-H by 500 steps; explained variance ≈ 0.04 | 4 |
| AC16, TD(0.9), warm-up | all-H throughout; explained variance ≈ 0.05; clip binding 0.55 | 4 |
| TB16, first canonical form (HALT term at the arrival) | **collapses into PREV-to-dead** (§3) | 4 |
| TB16, corrected | all-H throughout; gradient clipped on ≈ 0.9–0.99 of steps | 4 |
| PF-IS16, first canonical form | all-H; ESS / K 0.79 | 4 |
| PF-IS16, corrected | all-H, then a first L SET at 1,300 steps (set_late 1.0, PREV 0.03); ESS / K 0.79 | 4 |

- **PF δ = 1 heads for L, not the E of its derived fixed point (§5.2).** Either the transient goes through L first, or the router's function class pools differently from the one-lookup classes. The PF check at 5k steps tells which.
- **The revision-3 TB16 and AC16 runs were stopped:** the canonical-route bug and the non-zero critic made them invalid.
- **Three implementation choices came from these runs**, each recorded in §3 and §5.5:
  - the canonical route ends without a HALT term;
  - the critic warm-up;
  - AC16's λ = 0.9 (TD(0) is the check AC16T0).
- **Every candidate except EI16 sits in all-H at this scale.** Rung 2 can fail. That outcome is in the reading table.

## 10. Fairness (AGENTS rule 4)

- One model, one task, one data stream per seed; only the training signal differs.
- lr is tuned per signal in the pilot; comparisons are best vs best, with curves.
- The same episode budget everywhere. Rollouts per query are reported; the exact recursions of EX, EX-AL and PF are not rollouts.
- Clip binding is reported per arm.
- The label enters every arm only through the readout or the likelihood at the route's end. The value head never sees the label or the query.
- At a seed, the AC arms' policy init equals RL16's (§3).

## 11. What rung 2 does not answer

- Neural primitives or states (risks 1 and 2).
- Search guided by a learned value (B6).
- Setup distances beyond one hop. Risk 4 at longer horizons is the follow-up if P4 is uninterpretable or AC16 ≈ ACMC16.
- The basis claim, which is the next rung's candidate (D106).

## 12. Prior work

- **Score-function estimators.** REINFORCE (Williams 1992); RLOO (Kool et al. 2019); VIMCO and multi-sample bounds (Mnih & Rezende 2016; Burda et al. 2016; Rainforth et al. 2018).
- **Spurious programs.** Guu et al. 2017; Liang et al. 2017/2018 (NSM, MAPO); Min et al. 2019; MeRL; priority-queue training.
- **Amortized posterior / PFN.** Müller et al. 2022 (PFNs); TabPFN; DreamCoder's recognition model; RL-as-inference (Levine 2018), in which the fixed-prior posterior is the optimum of KL-regularized RL with a log-likelihood reward; reweighted wake-sleep (Bornschein & Bengio 2015) for self-normalized importance-weighted forward KL.
- **GFlowNets.** Bengio et al. 2021; trajectory balance (Malkin et al. 2022); VarGrad and the reverse-KL equivalence (Richter et al. 2020; Malkin et al. 2023, "GFlowNets and variational inference").
- **Value of computation.** Russell & Wefald 1991; Hay et al. 2012.
- **Actor-critic and TD.** Sutton 1988; GAE (Schulman et al. 2016).
- **Search + imitation.** Expert iteration (Anthony et al. 2017); STaR, ReST-EM.
- **Modular routing.** Rosenbaum et al. 2019; Mittal et al. 2022; Kirsch et al. 2018.

New here, to our reading:
- exact current-policy and fixed-prior posteriors, and the likelihood split between them, as references on the same seeds;
- a basin, built by training dynamics under exact EM, against which fixed-prior targets, value bootstrapping and search are tested;
- program identity under an expressivity constraint that is known exactly;
- a spurious-density dose with the readout held fixed.

## 13. History of this design

- **Revision 1** (estimators; `e886574`).
- **Revision 2** (both reviews applied; `a80b932`–`43726e7`).
- **Revision 3** (re-scoped to training signals after Entry 65; `f34b73d`).
- **Revision 4** (review 3 applied).
  - **Added:** EX-AL, PF-IS16, ACMC16, RL16 on δ = 0; positive program classification; PP as an endpoint.
  - **Changed:** PF demoted to a check; TB16 relabeled; the dose replaced by an answer pool; success restricted to candidates; 5,000 steps; pilot seed 100.
  - **From the CPU runs:** canonical routes end without a HALT term (plus a local halt loss); a 200-step critic warm-up; AC16 uses TD(0.9), and TD(0) becomes the check AC16T0.

## 14. Review handling carried over from revision 2

All items of design review 1 and code review 1 remain handled as listed in revision 2's §13:
- real-init derivation;
- per-run seeds;
- P-word search statistics;
- float64 beam keys;
- collision-free grid names;
- exact K = 2 enumeration tests;
- confirmed criterion;
- the pruned-space MAP.

## 15. Review 3 handling

Design review 3 ([review3_design.md](review/review3_design.md)):

| item | handling |
|---|---|
| B1 PF derivable | Accepted. PF is a 2-seed check; the pooled table is in §5.2 (`derive/pooled.py`). P6's PF line is withdrawn. |
| B2 P2 not causal | Accepted; reproduced (§5.1). EX-AL added; P2 restated as EX vs EX-AL (likelihood) and EX-AL vs PF / PF-IS16 (self-consistency). The "task is the attractor" reading is withdrawn. |
| B3 success via enumeration | Accepted. Success is restricted to candidates, judged on the committed route; determinism is reported separately. |
| B4 critic not zero | Fixed: zero output layer, built after the policy head (test). |
| B5 TB16 ≠ PF sampled | Both fixes. TB16 is relabeled as reverse KL / entropy-regularized; PF-IS16 (forward KL, SNIS) is added. IW16 vs PF-IS16 also differs in the likelihood, which EX vs EX-AL measures exactly. |
| S1 P4 misframed | ACMC16 (gae = 1) added; RL16 added on δ = 0. The critic reads a stop-gradient trunk. TD error and explained variance are logged. Uninterpretability conditions are stated in P4. |
| S2 dose confounded | Replaced by a per-episode answer pool with the readout fixed. §5.3–5.4 re-derived. P6 directions are taken from them, with strict margins, as exploratory. |
| S3 PP open | PP is an endpoint for every learned run. Only PF's is derived. |
| S4 determinism by construction | Determinism is removed from success and reported, with PF's derived level. |
| S5 pilot seed | Pilot seed 100. The tie rule is extended; failures at every lr are reported. |
| S6 schedule | 5,000 steps. Success clock is 3× EX in this grid. |
| S7 counts | P1's 2/5 is inconclusive. EI16's program is predicted per seed from its init beam program. P2 has falsifier counts. |
| S8 missing rows | Added (§7). |
| S9 set_late | Positive E / L / other classification on shortest-correct routes. |
| S10 P3 / P4 | Set from the CPU runs (§7, §9); re-reviewed before the pilot ([review4.md](review/review4.md), handled below). |
| Minor | The 9.1 % ambiguous read is in §2. The ledger note on F29's "stray SET mass at init" is to be qualified (Entry 66). |

Code review 3 ([review3_code.md](review/review3_code.md)):

| item | handling |
|---|---|
| Bug 1 TB no-op moves | Canonical routes for TB16 and PF-IS16 (test). |
| Bug 2 critic init and RNG | Fixed (test: same policy parameters at a seed, V = 0). |
| S3 "starts as RL16" | Corrected (§5.5). |
| S4 critic trains the trunk | Stop-gradient (test). |
| S5 query-blind V | Caveat in §3. |
| S6 no success count | `analyze.py` counts SUCCESS, program labels, PP and determinism per arm. |
| Minor | The label boundary matches P5. Program identity is computed on shortest-correct routes only. The curve's "loss" is each signal's reported objective (TB: mean log-ratio; AC: negative return; PF-IS: ESS). Stale text fixed. |

Review 4 ([review4.md](review/review4.md); verdict: launch the pilot, fix three items before the main grid):

| item | handling |
|---|---|
| B1 P3 both-fail | A both-fail clause (uninterpretable); a one-seed gap is inconclusive; timing is compared only in pairs where both learn. |
| B2 P4 at noise | ACMC16 has 5 seeds; ±1-seed band and ≥ 2-seed falsifiers; a "worse" falsifier and reading row added. |
| B3 determinism not logged | `evaluate` records sampled shortest-correct accuracy (`samp_min_acc`, by P / no-P); analysis uses it (test). |
| S1 P5 | EX-AL is open; EI16's line is "all but at most one, when ≥ 3 learn"; mixed is added to the δ = 0 falsifier. |
| S2 PF check row | Split into "PF does not learn" (investigate) and "PF learns L" (read the rest normally). |
| S3 P6 | IW16 is open; margins are fractions (≥ 0.33) or timing in every pair; arms failing at pool 0 are excluded. |
| S4 PP on L runs | PP uses shortest-correct accuracy at pp8. |
| S5 halting | The correct justification for the no-HALT form is in §3; "answer right but no halt" is counted per arm. |
| S6 rows | "Success only through EI16" and "TB16 learns δ = 1 but fails δ = 0" added. |
| S7 one lr on δ = 0 | EX-AL δ = 0 at 3e-3 added to the pilot. |
| S8 ESS hides collapse | Distinct sampled routes per P-word query logged for every sampled arm (test). |
| Minor | AC16's reading row updated to TD(0.9); TB16's reward floor is named as a P3 confound; main-grid lr values are marked as placeholders; the readout confound of the reward arms is stated before the reading table. |
