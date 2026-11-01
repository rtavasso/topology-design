# Rung 3: commitment (revision 1)

**Status:** design, 1 Nov 2026. For an independent design review before implementation. Code does not exist yet; §10 lists what is needed. History Entry 70; ledger risks 4 and 8 (the 1 Nov update), H12–H14, I30, I33.

**Direction.** After rung 2 (F35–F41), two questions decide how the router of later rungs should be trained, and both are about what happens *after* a router has committed to a program. **Core A (risk 8):** when the data later start to care about a tie the procedure broke, does the committed router move, and does that depend on how its targets are made? This is the live disagreement of Entry 70 §2: the researcher's position that varied data break ties, against the mechanism of F36. **Core B (risk 4's prerequisite):** can a reward signal be kept exploring long enough to learn, so that contextual value becomes readable at all (H14)? **Checks C (H13):** on the saved routers, whether disagreement among sampled routes is an uncertainty signal, derivable against the exact marginal.

**Scope (AGENTS hard rule 1).** The rung-1 router, executor and data; the rung-2 signals. Nothing neural. Every claim labeled.

---

## 1. The questions

**QA (plasticity after commitment).** A router committed to one of two programs that are tied on its training distribution is trained further on a distribution where only one of them is right. For each way of making its targets, does it move, and how fast?

The readings and what each predicts (Entry 70):

| reading | prediction |
|---|---|
| *ties do not matter* (researcher): the loss breaks ties when disambiguating data arrive; the procedure's first choice is a transient | every signal moves to the required program, at a rate set by the fraction of disambiguating data, not by the signal |
| *plasticity is a property of the target* (Claude): a committed learner moves only if its targets can see the other program | movement orders by how the target is made: exact fixed prior > search under the router's own policy > importance-sampled fixed prior ≈ the router's own posterior; and search's plasticity is predicted by its route coverage at the committed policy |
| *exactness, not priors*: only an enumerating target can move | only PF moves anywhere |

The first outcome is a clean loss for the second reading and the cheapest possible place to learn it. The third is the worst outcome for the thesis: every scalable signal locks in.

**QB (exploration before value).** Rung 2's reward arms collapsed into a deterministic policy within 200–400 steps (F40). Two fixes differ in kind: an entropy term keeps the policy soft; a beam supplies routes the policy would not sample. Does either let a reward signal learn the deferred program? The derivation in §5.2 says the two should differ, which is why both are core.

**QC (disagreement as uncertainty).** For a stochastic router, does disagreement among k sampled routes predict which queries its committed route gets wrong? In the toy this is a Monte Carlo estimate of the exact marginal, so it is a check with a known ceiling; its value is to calibrate the picture of Entry 70 §4 before a carrier with real epistemic uncertainty.

## 2. Setting

Unchanged from rungs 1–2 ([route/DESIGN.md](../route/DESIGN.md) §§2–4; [routefind/DESIGN.md](../routefind/DESIGN.md) §§2–3): 48 keys, 16 answers, decoys, the flag deferred (δ = 1) or local (δ = 0), the product state (symbol, pending c ≤ 2, set-here), actions HOP / PREV / SET / HALT, the Markov router with one cross-attention layer, ε = 0.02, λ = 0.02, T = 14, the strata, the held-out criterion set.

**The tie and what breaks it.** Under δ = 1 the shortest correct programs E (SET at the predecessor, carry) and L (SET at the P-leaving state) are tied on the training distribution (words with isolated P-steps). Two changes to the data break the tie in opposite directions:
- **PP words** (two or more consecutive P-steps; today only an evaluation stratum, never trained on): L cannot do them under δ = 1, because at a PP arrival no pair's value points to the state and L's read finds nothing (routefind §2, confirmed: EI16's pp8 first deviation is SET>HOP in 1,250/1,250). E can, once its precedence at the conjunction "flagged ∧ pending" is set (F30; the E programs' pp8 failure is SET>PREV, a learnable decision). So PP words favour **E**.
- **The local trigger** (δ = 0): E is inexpressible (it would need a two-hop read), L is the only shortest program. And E's rule under δ = 0 is precisely F29's stray rule (set here ⇒ HOP, carried count ⇒ PREV). So δ = 0 favours **L**, from the worst possible starting point for an E router.

**The committed starting points** come from rung 2's phase-1 programs, which are deterministic: EI16 → L (10/10), EX → E (3/3), IW16 → E (4/4, one R); PF → E (2/2) and PF-IS16 → L-leaning mixtures (F37). Rung 2 saved no weights, so phase 1 retrains them with checkpoints.

**Program R** (F38) is a third program (SET once at the start, decisions one count higher) and is counted as its own category throughout.

## 3. Design

Two phases per run, both at a **constant** learning rate (1e-3, the rung-2 pilot's choice for every signal) after a 500-step warm-up, so that a slow mover is not recorded as a failure by a decaying schedule (L31).

**Phase 1 (commitment).** δ = 1, the rung-2 training distribution, 2,500 steps × batch 64, seeds 0–4, per-run init; the full router state is saved at the end. Arms: EI16 (→ L), EX (→ E), IW16 (→ E), PF (→ E), PF-IS16 (→ mixture), 5 seeds each. The saved routers are the starting points of phase 2 and the material of checks C.

**Phase 2, shift B (accumulation; the test of the researcher's argument as stated).** Training distribution = the phase-1 distribution with a fraction f of chains carrying a PP word (the `pp` generator of the strata, k ≥ 4), f ∈ {0.25, 0.05}. 3,000 steps. From the **L-committed EI16 routers**, the treatment is the phase-2 signal:

| phase-2 signal from the L router | how its targets can see E on PP words | seeds (f = 0.25 / 0.05) |
|---|---|---|
| PF (exact fixed-prior posterior) | by recursion, whatever the policy does | 5 / 3 |
| EI16 (beam under its own ε-mixed policy, hard EM) | only if the beam keeps routes that SET where the policy says HOP | 5 / 3 |
| PF-IS16 (fixed prior, self-normalized IS from its own samples) | only if one of 16 samples reaches the label on a PP word | 5 / 3 |
| IW16 (own posterior, sampled) | same coverage, and the target on ordinary words prefers L | 5 / 3 |
| EX (own posterior, exact) | sees E on PP words exactly; on ordinary words its target is L | 5 / 3 |

References: EX from the **E-committed EX routers** at both f (the precedence flip alone; the speed ceiling), 3 seeds each; from scratch on the f = 0.25 mixture, EX, EI16 and PF-IS16, 3 seeds each (what each signal does when it never committed; whether EI16 still lands in L when PP words are present from the start); EI16 continued on the unchanged distribution, 2 seeds (stability).

**Phase 2, shift A (replacement; F29's conflict from a committed start).** Training distribution = δ = 0 only, 3,000 steps, from the **E-committed EX routers**, signals EX, IW16, PF, PF-IS16, EI16, 3 seeds each. Reference: rung 2's from-scratch δ = 0 cells (EX 0/2 + 0/5, IW16 0/2, PF 2/2, PF-IS16 3/3, EI16 2/2) and EI16B (search started inside EX's δ = 0 basin, 0/2).

**Core B (exploration for the reward signals), δ = 1 from scratch, 5,000 steps, constant lr.**
- **RL16+H(β):** REINFORCE with RLOO and an entropy bonus β · H(π(· | x)) at the visited states, β ∈ {0.01, 0.03, 0.1}; 3 seeds each in stage 1.
- **PFB16:** the fixed-prior posterior **restricted to a beam**: the width-16 ε-mixed beam of EI16 supplies the completions, each weighted by π₀(r) e^{−λ·moves} L_y(r) and normalized over the beam; the router is trained by cross-entropy to that soft distribution (soft EM over the beam; PF with the beam as its support, instead of 16 on-policy samples). 5 seeds on δ = 1, 3 on δ = 0, 3 at rung 2's dose (answer pool 4).
- **Stage 2, conditional:** if RL16+H learns the deferred program in ≥ 2/3 seeds at some β, AC16+H(β*) and ACMC16+H(β*) run at 5 seeds, and P4 of rung 2 (bootstrapped vs Monte-Carlo value) is read there. If not, risk 4 stays blocked and the record says why.

**Checks C (no training).** On every saved phase-1 router, on the held-out strata, per query: the committed route's correctness; agreement among k ∈ {4, 16} sampled routes, on the answer and on the route; the exact marginal p(correct) (the enumerating ceiling); the router's policy entropy at each state of the committed route, by the oracle's decision there (HOP / SET / PREV / HALT). The readout's entropy at the committed end state is constant by construction (1 − ε on an answer), so the "spread of the prediction" baseline is known to carry nothing here.

## 4. Endpoints

- Shortest-correct committed accuracy on P-words at in4 / in8, and on **pp8** (the disambiguating stratum), on the held-out curve every 100 steps; time to pp8 ≥ 0.95.
- Program identity over phase 2: `set_late` and `set_other` on P-words every 100 steps (the E ↔ L trajectory), R as a category; the first deviation at pp8.
- Determinism (sampled shortest-correct), as in rung 2.
- **What the signal sees at the start of phase 2** (the derive step, §5.3): on PP words, the fraction of queries whose beam / 16 samples contain an E route (`oracle_kept`, `oracle_any` by k); under shift A, the fraction whose beam / samples contain an L route.
- Core B: distinct sampled routes per query (collapse), policy entropy, criterion step; for PFB16, the beam's hit rate and the target's entropy.
- Checks C: AUROC of each uncertainty signal for committed-route error, by stratum; policy entropy by decision type.

## 5. Derived before measuring

**5.1 PF's fixed points under the phase-2 mixtures** (`routefind/derive/pooled.py`, extended to the mixture): the pooled target under δ = 1 + PP at f = 0.25 and 0.05, and under δ = 0. Expected: E at both f (PP words add E-only mass to a target that was already E-favouring at pool 0), L under δ = 0 (as rung 2 derived). This makes PF's phase-2 cells checks of speed, not of direction, and it gives the direction the data are pulling toward in every cell.

**5.2 Why the two exploration fixes should differ.** Rung 2's `derive/search.py`: at a near-uniform policy, 16 samples contain the oracle route on 0.00–0.03 of k = 3 P-words. An entropy bonus keeps the policy near uniform exactly where it has not learned, which is where the setup step must first be rewarded; a reward gradient for SET at a flagged state needs a sampled route that SETs there *and* completes correctly, which the uniform policy almost never produces. So RL16+H is predicted to learn the all-H words (its collapse delayed) and still not the setup step. The beam, by contrast, reached the label on 0.68–1.00 of k = 3 P-words at the stray policy and kept the oracle on 0.74–1.00 (δ = 1), because it enumerates completions rather than sampling them. The beam-supported soft target (PFB16) therefore sees the setup step from the start, and it also has the fixed prior's property of not reinforcing the policy's own labels. This derivation is the reason core B is two arms and not one.

**5.3 Route coverage at the committed routers** (after phase 1, before phase 2 is preregistered): for each L router, the beam's and the sampler's coverage of E routes on PP words, under ε-mixing 0.1 and K = 16; for each E router under δ = 0, their coverage of L routes. These numbers turn the "open" rows of §7 into predictions, and they are the test of the second reading's claim that coverage predicts plasticity. Rung 2's EI16B is the one existing point: at EX's δ = 0 basin policy the beam's MAP correct route was the oracle's on 0.00, and EI16B did not move (0/2).

**5.4 Checks C are derivable in part.** Given each query's exact marginal p(correct), the distribution of k-sample agreement is binomial, so the AUROC of k-sample disagreement is a function of the marginal's AUROC and k. What is not derivable is whether the marginal's uncertainty is informative about *committed-route* error at all for a router whose stochasticity is SET padding (§5.2 of rung 2): that is H13's falsifier.

## 6. Grid and budget

| stage | cells | runs | steps |
|---|---|---|---|
| 1 | phase 1: EI16, EX, IW16, PF, PF-IS16 × 5 seeds | 25 | 2,500 |
| 1 | core B: RL16+H × 3 β × 3 seeds; PFB16 δ1 × 5, δ0 × 3, dose × 3 | 20 | 5,000 |
| (derive) | §5.3 on the 25 saved routers; checks C | – | – |
| 2 | shift B from L: 5 signals × (5 + 3) seeds | 40 | 3,000 |
| 2 | shift B references: EX from E × 2 f × 3; from scratch × 3 signals × 3; stability × 2 | 17 | 3,000 |
| 2 | shift A from E: 5 signals × 3 seeds | 15 | 3,000 |
| 2 | core B stage 2 (conditional): AC16+H, ACMC16+H × 5 | ≤ 10 | 5,000 |

≈ 127 runs, ≈ 380k run-steps, ≈ 9 T4-hours at rung 2's rate (0.085 s per run-step), ≈ 4.5 h wall on two T4s, in two jobs with the derive step between them. Seeds are below AGENTS rule 4 and defensible only for count predictions on near-bimodal outcomes; timing comparisons are exploratory.

## 7. Predictions (counts; thresholds fixed here; the "open" rows are set from §5.3 before phase 2 runs)

| # | prediction | falsifier | type |
|---|---|---|---|
| **PA1** | **Shift B, f = 0.25, from the L router.** PF moves to E and reaches pp8 ≥ 0.95 in **≥ 4/5**. IW16 stays L and fails pp8 in **≥ 4/5**. EX: **≥ 3/5** move (its exact target sees E on every PP word; its target on ordinary words is its own L, so it is slow). EI16 and PF-IS16: **open**, set from §5.3 — if the beam's E-coverage on PP words at the L router is ≥ 0.5, EI16 moves in ≥ 3/5; if ≤ 0.2, in ≤ 1/5; PF-IS16 likewise from the sampler's coverage, expected low. | Every signal in ≥ 4/5 (ties do not matter). Only PF (exactness). IW16 moving in ≥ 3/5 (sampled own posterior is plastic). | mechanism (risk 8) |
| **PA2** | **Shift B, f = 0.05.** PF still moves in **≥ 2/3**; every other signal **≤ 1/3**. The data fraction sets the speed for a target that sees E; for the others it sets nothing, because they do not see E. | PF ≤ 1/3 (then a 5 % minority is too little for anyone, and PA1's f = 0.25 is the only readable cell). Any other signal ≥ 2/3. | mechanism |
| **PA3** | **Shift A, δ = 0 from the E router.** EX and IW16 **≤ 1/3** learn L (lock-in: the E rule is the stray rule). EI16 **≤ 1/3** (EI16B replicated from a cleaner start). PF **≥ 2/3** (a fixed target has no basin). PF-IS16 **≤ 1/3** (coverage). | PF ≤ 1/3 (then even a fixed target cannot undo a commitment, and it is about the state, not the target). EX or EI16 ≥ 2/3 (a committed start is easier than F29's from-scratch stall). | mechanism (F29, F36) |
| **PA4** | **Coverage predicts plasticity.** Across the open rows, the signal–cell pairs ordered by their §5.3 coverage are ordered the same way by their count of movers (Kendall τ > 0 over ≥ 6 pairs). | τ ≤ 0: plasticity is not about what the target sees. | mechanism |
| **PB1** | **Entropy is not enough.** RL16+H learns the deferred program in **≤ 1/3** seeds at every β; its collapse is delayed (distinct routes > 1 for longer) but the setup step is never rewarded (§5.2). | ≥ 2/3 at some β; then stage 2 runs and P4 is read. | optimization (H14) |
| **PB2** | **The beam-supported soft target learns.** PFB16 learns δ = 1 in **≥ 4/5** and δ = 0 in **≥ 2/3**, in a program that is **open** (E by the fixed prior's pooled target, or L by the beam's support, as EI16); at the dose it degrades by less than PF-IS16 did (criterion reached in ≥ 2/3 within 5k). | ≤ 2/5 on δ = 1: a beam of 16 is too narrow a support for a soft target. Dose degradation equal to PF-IS16's: the fragility was the target, not the proposal. | optimization |
| **PC1** | **Disagreement is informative but padded.** On the stochastic routers (PF, PF-IS16), 16-sample disagreement on the *answer* predicts committed-route error on the held-out strata with AUROC ≥ 0.7, approaching the exact marginal's; disagreement on the *route* is near chance (SET padding, §5.2 of rung 2). Policy entropy concentrates at SET and PREV decisions over HOPs by ≥ 3×. | Answer disagreement at AUROC ≤ 0.6: breadth is not an uncertainty estimator here. Entropy flat across decision types. | by construction, partly (H13) |

**Checks (reported, not predicted).** EX from the E router learns the pp8 precedence at both f (the speed ceiling). Stability: EI16 on unchanged data stays L at pp8 ≈ 0. From-scratch EI16 on the f = 0.25 mixture: whether it still lands in L when PP words are present from the start (open, reported).

**Reading the outcomes.**

| outcome | reading |
|---|---|
| Movement orders by target type and tracks coverage (PA1–PA4 as predicted) | Plasticity after commitment is a property of how targets are made. The synthesis of Entry 70 (search supplies targets, a fixed prior breaks ties) is about plasticity, and it is real. Later rungs train the router with searched or fixed-prior targets and never with its own posterior. |
| Every signal moves at f = 0.25, most at 0.05 | Commitment is not sticky; F36 was about stray labels from a bad start. The researcher's reading holds; program identity is a transient and the choice of signal can be made on cost alone. |
| Only PF moves | Plasticity requires enumeration; every scalable signal locks in. The worst outcome: before any primitive-learning rung, the program needs a signal that can move, and does not have one. |
| EX moves but IW16 does not | The own-posterior target is plastic when exact and not when sampled: sampling, not self-consistency, is the lock. |
| EI16 moves where its coverage is high and not where it is low | Search is plastic exactly as far as it searches; coverage, not the prior, is the variable to engineer. |
| PA3: PF moves, nothing else | A fixed target undoes a commitment that search cannot: the AlphaZero structure (search under its own policy) is the one that locks in. |
| PB1 holds and PB2 holds | Exploration for a router is a search problem, not a temperature problem; the beam-supported soft target is the scalable form of the PFN signal and the candidate for rungs 4+. |
| PB1 fails | An entropy term suffices; stage 2 runs and risk 4 is read on the value contrast. |
| PB2 fails on the dose | F41's fragility is in the fixed prior itself, not in the sampling estimator. |
| PC1 holds | Breadth measures the router's own uncertainty about the answer; the controller of I30 is worth building on a carrier with real epistemic uncertainty. |

## 8. Gates

- **G1 tests:** checkpoint save / load round-trips; the PP-fraction generator (the fraction and the `pp` word shape; f = 0 reproduces the rung-2 stream byte for byte); the constant schedule; the entropy bonus's gradient on a hand-built policy; PFB16's target equals PF's restricted to the beam's completions on a tiny case, and equals PF when the beam is exhaustive; agreement and AUROC on hand-built cases.
- **G2 derive:** §5.1 before stage 1; §5.3 and checks C between stages.
- **G3 pilot:** none beyond stage 1; lr is 1e-3 throughout.
- **G4 reviews:** design review of this revision; a code review before stage 1; the open rows of §7 fixed from §5.3 before stage 2, with a short re-review.
- **CPU sanity (L25):** one seed of EI16 phase 1 → PF phase 2 at f = 0.25 and one of RL16+H(0.03), 1,500 steps each, before stage 1.

## 9. What this rung does not answer

- Anything about neural primitives or the basis (risks 1, 2); rung 4 is the distillation test (D110).
- Branching at inference, superposition, or a controller: checks C are the toy's calibration of the picture, not the picture.
- Abstraction of compositions (risk 9).
- Whether plasticity under *gradual* shifts (f rising over training) differs from a step at f; one step per cell is the minimum.

## 10. Implementation needed (for the code review; none of it exists)

- `run_stack`: save each run's full `state_dict` (and the frozen bases' seed) at the end; a per-run `init_from` config key that loads a saved router before training; a `schedule: "constant"` option.
- `RouteConfig.pp_frac`: in `generate`, with probability `pp_frac` a chain of length ≥ 4 draws its word from `forced_word(rng, k, "pp")`; `pp_frac = 0` must leave the stream unchanged.
- Estimators: `rl_loss` with an entropy bonus β at the sampled routes' states; `pfb_loss`: the beam's completions (all of them, not only the MAP) with their π₀ e^{−λ·moves} L_y weights normalized per query, and a cross-entropy to that distribution along the completions' routes. The beam already returns completions internally (`complete` in `beam_search`); it has to return them all.
- Analysis: program trajectory over phase 2; pp8 over time; coverage at the checkpoints (`derive/search.py` on saved routers); checks C (agreement, AUROC, entropy by decision).
- Grids: two jobs (`cm_stage1.json`, `cm_stage2.json`), the second generated after the derive step.

## 11. Prior work

Catastrophic forgetting and plasticity loss (continual-learning literature; "loss of plasticity" in deep RL, Dohare et al. 2024); policy collapse and entropy regularization (A3C, SAC); expert iteration and AlphaZero's search-improved targets (Anthony et al. 2017; Silver et al. 2017); soft EM and posterior regularization; adaptive self-consistency and agreement-based halting (Wang et al. 2023; Aggarwal et al. 2023); ensembles and MC dropout as epistemic uncertainty (Lakshminarayanan et al. 2017; Gal & Ghahramani 2016); value of computation (Russell & Wefald 1991); DreamCoder (Ellis et al. 2021).

New here, to our reading: the same committed router continued under five target types with the data shift held fixed, which separates "the data broke the tie" from "the learner could move"; coverage at the committed policy as a preregistered predictor of plasticity; the beam-supported fixed-prior posterior as a signal.
