# Rung 4 — programs: does an amortized router generalize across programs, given labels? (design, revision 1, 3 Nov 2026)

**Status: draft for discussion.** Three choices are left open for the researcher (§11) and the design goes to an independent review after them. Nothing here is implemented.

## 0. Where this comes from

The guiding star, restated by the researcher on 3 Nov (history Entry 74): the router is a **PFN over programs under a data prior** — given the primitives and the prior over their compositions, inference at test time is amortized routing, not a walk over routes. The research decomposition that follows: *(1)* assume the training corpus comes with the optimal program for every token, train the router on it, and ask whether it **generalizes across programs**; *(2)* only if it does, ask where the labels come from. This rung is step (1). It replaces the distillation test as rung 4 (that test becomes rung 5, with a sharper hypothesis: chains against bags, Entry 74); step (2) is rung 6.

What the ladder already says about step (1). Rung 1 ran a small version of it: teacher forcing on oracle routes generalized exactly from length ≤ 8 to 32 (F30), because the program is the **closure of per-state decisions** — a rule per computational state, and any new *sequence* of known decisions is reachable without having been seen. Rung 3 showed the other case: a program that needs a *decision never taken at some state* (the L routers' missing SET at the flag) does not come from labels at any reasonable budget (F49). So "generalize across programs" is not one question; it splits into (a) new sequences of known decisions — derivable, a check; (b) **new combinations of known decisions conditional on the world** — the amortized-inference question proper, not derivable; (c) new decisions — predicted to fail, a check; and (d) **rules against tables**: whether a decision that depends on a quantity (a count) is learned as a comparison that extrapolates or as a lookup that does not, which is the researcher's "everything the model learns is a heuristic" put to a test. The core of this rung is (b) and (d).

## 1. The questions

- **Q1 (core).** A router trained on a *mixture of worlds*, each with its own optimal program over the same four primitives and a cue in the context saying which world, is tested on worlds it never saw whose program is a **recombination** of decisions it did see, under different cues. Does it route correctly there? If the router has learned the decisions as separate functions of separate cue components, yes; if it has learned a table over (state, world), no.
- **Q2 (core).** One decision depends on a count the cue carries ("SET until the pending count equals m"). Tested at a count outside the training range: does the router extrapolate (it learned the comparison) or fail (it learned the table)?
- **Q3 (check).** Within training worlds, held-out words and lengths: F30's closure, replicated under the mixture.
- **Q4 (reference).** Do routes found *without* labels — the exact marginal (EX) and search + imitation (EI16), the signals rung 2 established — generalize across programs as well as labelled training? If yes, labels are not the bottleneck for composition and step (2)'s question changes shape.

What this rung does **not** test (§10): inferring the world from data rather than from a cue (the next PFN rung, with solved demonstrations in the episode); where labels come from (rung 6); chains against bags in a trained dense model (rung 5).

## 2. The world family

The carrier is rung 1's: 48 keys, 16 answers, pair tokens (key, value, prev, flag), chains with H- and P-steps, the four actions HOP / PREV / SET / HALT over the product state (s, c, h), the exact executor and the exact route marginal ([route/DESIGN.md](../route/DESIGN.md) §§2–4). Two things change.

**Two world parameters.** A world is (δ, m):
- **δ ∈ {0, 1}** — where the flag sits relative to the P-step (rung 1's deferral): on the P-step's own pair (local, program L-type: SET then PREV at once) or one state earlier (deferred, program E-type: SET, carry across one HOP, then PREV).
- **m ∈ {1, 2, 3} in training, 4 held out** — how far back a P-step goes: pair(s_i) is the pair **m places before** pair(s_{i−1}) in context order, so the route must SET m times at the flag and PREV m times after it. The pending count's range becomes c ∈ {0, …, 4} (NC = 5; 65 × 10 = 650 product states; the marginal's cost grows by 5/3).

The oracle rule per world, written for the executor: HALT if s is an answer (or ⊥); **SET** if pair(s) is flagged and c < m and (h = 1 or c = 0) — i.e. keep setting at the flag until the count reads m; **HOP** if δ = 1 and h = 1 (carry: the count was set here and applies to the next state's exit); **PREV** if c ≥ 1 (consume); else HOP. Under δ = 0 the first PREV fires at the flagged state itself (h = 1, c = m); under δ = 1 it fires at the next state. The oracle route of a word with n_P P-steps has k + m·n_P moves. The generator builds a P-step as a block of m + 1 pairs in reverse context order (the m − 1 inner pairs are pass-through pairs of other chains or fresh keys), and `check_episode` verifies every oracle route as in rung 1. **P-runs are limited to length 1 in training** for every world (as rung 1); the `pp` strata (runs of 2) are evaluated in m ≤ 2 worlds, where c ≤ 4 suffices.

**The world cue.** Two frozen directions in every pair token (and in the state token), orthogonal to the slots: **u_δ** carrying δ ∈ {0, 1} and **u_m** carrying m as a *scalar* (m · u_m), the same way the pending count c is already carried (c · u_c). The scalar encoding is deliberate: with c and m on scalar directions the decision "SET while c < m" is a linear comparison, so extrapolation to m = 4 is *possible* for the architecture (the planted router of §5 does it) and Q2 asks whether the trained router finds it. (Open choice §11.2: one-hot m would make Q2 derivable — a fail by closure — and is the weaker design.)

**The training mixture.** Episodes are drawn uniformly from the **four training worlds (δ, m) ∈ {(0, 1), (1, 1), (0, 2), (1, 3)}**, one world per episode, cue in every token. Held out: the two **recombination worlds (1, 2) and (0, 3)** — every decision each needs has been taken in training, under a different partner cue: "carry at the flag" was seen with m = 1 and m = 3 (δ = 1), "SET to 2" with δ = 0, "SET to 3" with δ = 1, "PREV ×2 / ×3" each in one world — and the **extrapolation worlds (0, 4), (1, 4)**, whose SET-to-4 and PREV-×4 visit the count c = 4 that no training route visits. (Open choice §11.1: the researcher's held-out structure; this is the recommendation.)

## 3. Arms

All on the mixture, constant lr after warm-up (L31), per-run init, one world cue per episode.

| arm | signal | role | seeds |
|---|---|---|---|
| **RT** | teacher forcing on the oracle route of the episode's world (the labels of step 1) | core: Q1, Q2, Q3 | 10 |
| **EX** | exact marginal likelihood of the answer (no route labels) | reference Q4 | 3 |
| **EI16** | ε-mixed beam (W 16, ε 0.1), hard EM (no route labels; rung 2's most robust signal) | reference Q4 | 3 |
| **RT-shuffled-cue** | RT with the cue permuted across episodes (the cue carries no information) | null: the cue is used | 2 |
| **RT-one-hot-m** | RT with m one-hot instead of scalar | Q2's encoding control, if §11.2 is decided for scalar | 3 |
| **RT-single-world** ×4 | RT on each training world alone | reference: what each world's program looks like unmixed; the margin and decision maps per world | 1 each |

Ten seeds for the core because the outcomes of Q1 and Q2 may be bimodal across seeds (F37: the learner's bias decides what the data do not), and a count over 10 is the first number that is not a coin.

## 4. Endpoints

Function-space, committed reading (the route that executes), per world × stratum: route accuracy (the committed route equals the oracle's action for action and halts with it), shortest-correct, answer accuracy, steps; the in-range strata `in1`–`in8`, `none`, the length strata to 32, `alt`, and `pp` in m ≤ 2 worlds — the rung-1 strata, per world.

**Decision maps.** For each world, at every product state the oracle visits on P-words: the committed action against the oracle's, aggregated by **decision type** — SET-at-flag (c < m), the m-th SET, carry (δ = 1, h = 1), PREV (c ≥ 1), HOP elsewhere, HALT. A held-out world's failure is then read as *which* decision broke, which separates a closure failure (an unvisited state) from a factorization failure (a visited state with a world-conditional decision), and in the extrapolation worlds separates "SETs to 3 and stops" (the table) from "SETs to 4" (the comparison).

**Generalization score.** For a held-out world: the fraction of in-range P-words (k ≤ 8) on which the committed route is the oracle's; a seed *generalizes* to that world at ≥ 0.95 (the rung-1/2/3 criterion) and *fails* at ≤ 0.5, with the band between reported.

## 5. Derived before measuring (`derive/`, before the pilot)

- **Closure (Q3 is a check).** Under the mixture, the states and decisions a held-out *word* in a training world visits are all visited in training; list them from the generator. Under the recombination worlds, list which (state, cue) pairs are new: by construction none of the *states* is new and every *decision* has been taken, only the (decision, partner-cue) pairs are new — written out as a table so that a failure is attributable.
- **Planted factorized router (L17).** Extend rung 1's planted router: SET on (flag − set-here) **gated by sign(m − c)**, PREV on (c − h·δ), carry on h·δ; verify it reproduces the oracle on every world including (0, 4) and (1, 4) at every k ≤ 32. This establishes that the architecture *can* express the factorized rule and the extrapolation, so Q1 and Q2 are optimization / inductive-bias questions, not expressivity.
- **Planted table router.** The same with the SET gate replaced by a lookup over m ∈ {1, 2, 3}: reproduces the training and recombination worlds and fails (0, 4), (1, 4) by SETting to 3 — the alternative outcome of Q2, planted, so the decision maps can be checked against both.
- **Counts.** P-words per world per 100 steps; the extrapolation worlds' route lengths (k + 4·n_P ≤ T).
- **Wandering mass at init under the mixture** (rung 1 §5's computation) for the EX reference.

## 6. Predictions (thresholds fixed here; counts over seeds; falsifiers)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1 (check)** | Within the four training worlds, RT reaches ≥ 0.95 route accuracy on every in-range stratum and on the length strata to 32 in ≥ 9/10 seeds (F30 under the mixture). | ≤ 7/10: the mixture itself breaks closure (then Q1 is not readable). | expressivity / optimization (derivable) |
| **P2 (core, Q1)** | RT generalizes to **both** recombination worlds in **≥ 6/10 seeds**, and the decision maps of the failures show the factorization break at the SET-count or carry decision, not at an unvisited state. | ≤ 3/10 seeds on either world: the router learned a (state, world) table; amortized routing does not compose across programs it has not seen together, and the labelled prior must cover the combinations. | optimization (inductive bias) |
| **P3 (core, Q2)** | RT extrapolates to m = 4 in **≤ 3/10 seeds** — it learned SET-to-m as a lookup over the training counts, and the decision maps show SETs stopping at 3 (the planted-table signature). | ≥ 7/10 seeds extrapolate: the router learned the comparison, i.e. a rule, not a heuristic, for a count it was never shown. | optimization |
| **P4 (reference, Q4)** | EX and EI16 generalize to the recombination worlds in no more seeds than RT does (labels do not hurt composition; they may not help it). Reported as counts; EI16 above RT by ≥ 2/3 would be the surprise. | — | reported |
| **P5 (null)** | RT-shuffled-cue is at ≤ 0.5 route accuracy on P-words in every world with a δ-dependent decision, and the single-world references are exact in their own world (the cue is doing the work in the mixture). | the shuffled-cue arm succeeds: the worlds are identifiable from the pair structure without the cue, and the cue is not what is being generalized. | check |
| **P6 (encoding control)** | RT-one-hot-m fails m = 4 in 3/3 (closure) and matches scalar RT on the recombination worlds. | one-hot generalizing to m = 4 (impossible by construction; would indicate a leak). | check |

**Reading the outcomes.**

| outcome | reading |
|---|---|
| P2 holds, P3 holds | The amortized router composes decisions across programs it never saw together, but learns counts as tables: programs are *rules over states and cues* with *heuristic constants*. Step (2) can build a labelled corpus that covers decisions, not their combinations, but must cover the range of every count. |
| P2 holds, P3 falsified | Rules all the way down on this carrier; the strongest form of the PFN claim; the labelled corpus need only cover the generators. |
| P2 falsified | No free composition across programs: the router memorizes (state, world); the prior must cover the combinations and the "PFN over programs" is a lookup over a product space — the labelled corpus would have to be exponential in the number of world parameters. Rung 6's question becomes how to make labels for combinations. |
| P4: EI16 ≥ RT | Search finds routes that compose as well as labelled ones; labels are not the bottleneck and step (2) is partly answered by rung 2's search. |
| P5 falsified | The worlds leak through the pair structure; redesign the cue before reading anything. |

## 7. Grid and budget

Pilot (G3): one seed per arm on the mixture, lr ∈ {3e-4, 1e-3} for RT, 6k steps, to set lr and steps; the pilot reads nothing (L30). Main: RT 10 + EX 3 + EI16 3 + shuffled 2 + one-hot 3 + single-world 4 = 25 runs at ≈ 6–10k steps, batch 64; at rung 1's measured cost with NS = 650 and T = 14 + 3m, ≈ 1.5× a rung-1 step: one Kaggle session.

## 8. Gates

- **G1 tests:** the m-step generator (blocks of m + 1 pairs; `check_episode` on every world); the oracle rule for every (δ, m) against a brute-force search over routes on tiny episodes; the NC = 5 executor's tables and the marginal's normalization; the planted factorized router exact on all six worlds plus (·, 4) at k ≤ 32; the planted table router failing (·, 4) in the predicted way; the cue directions orthogonal to the slots; decision-map aggregation on hand-built routes.
- **G2 derive:** §5 complete before the pilot.
- **G3 pilot:** lr and steps only.
- **G4 review:** an independent design review of this document (as rungs 1–3), then a code review before the main grid.

## 9. Fairness and measurement (AGENTS rules 4–6)

Same router, same parameter count in every arm; the cue is a frozen direction, not a learned embedding; lr from the pilot per arm; 10 seeds for the core and run-level counts; the identical-architecture nulls are the shuffled cue and the one-hot encoding; the planted routers are the expressivity references. Endpoints are function-space (committed routes); no representation comparison is made. Claims are labelled optimization unless a planted router or the closure table makes them expressivity.

## 10. What this rung does not answer

- **Inferring the program from data.** The cue hands the router the world; a PFN infers it from the episode. The next rung puts solved demonstration queries in the episode and removes the cue — the in-context identification question — and only then is the router a PFN in the full sense. This rung isolates the half that has to work first: that a router which *knows* the program can execute compositions of programs it has not seen.
- **Where labels come from** (rung 6: synthetic prior, search on real tokens, a teacher, an exogenous reasoner, read-off circuits; RESULTS of rung 3 §10 and Entry 74 for the candidates and what rungs 2–3 say about each).
- **Chains against bags** in a trained dense model (rung 5, the distillation test with its second curve).
- **Ties.** With the cue, every world has one shortest program; the posterior-over-programs training target (PF) and calibration over ties return in the demonstration rung, where the world is uncertain.

## 11. Open choices for the researcher

1. **The held-out structure** (§2): the recommendation is train {(0,1), (1,1), (0,2), (1,3)}, recombine {(1,2), (0,3)}, extrapolate {(0,4), (1,4)}. Alternatives: hold out only one recombination world and keep m ≤ 2 (cheaper, one bit of Q1); or add a third world parameter (e.g. which of two flag types marks a P-step) for a 2 × 3 × 2 family with four held-out cells (richer Q1, more generator work).
2. **The cue encoding** (§2): scalar m (recommended; makes Q2 a question) or one-hot (makes Q2 a closure check).
3. **Demonstrations now or next** (§10): this draft defers in-context world inference to the next rung; the alternative is one extra arm here with the cue removed and two solved queries per episode, which would need an encoder over demonstrations the current router does not have.

## 12. Prior work

PFNs (Müller et al. 2022) and in-context learning of algorithms; compositional generalization benchmarks (SCAN, COGS) and the systematicity literature; length and count extrapolation in transformers; DreamCoder's recognition model over a program prior (Ellis et al. 2021); rung 1's closure result (F30) and rung 3's representation reading of the L start (F49). New here, to our reading: a routed executor where "generalize across programs" is decomposed into closure, recombination under a cue, and count extrapolation, with planted routers for both outcomes of the extrapolation question.
