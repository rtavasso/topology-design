# Belief updating in a looped prover on a hidden-world carrier — design

> **Status, 13 Nov 2026 (evening): reviewed and found not sound as written** ([review/review2_design.md](review/review2_design.md); history Entry 93). The carrier, the exact mechanisms, the zero for deletion, the gate and the training claim of §7 hold. Four things block: the classes G and O do not mean "posterior" and "planner without belief" (§4, §8, §10 overclaim; the planner is the posterior with the prior taken to 1); at 4 training worlds some verdicts in the probes never occur in training; the class and cell rules are incomplete; the 32-world cell misses the stage-0 sampling-error gate on two of five draws. §2's "not found" is too wide (Park et al. 2024; Wurgaft et al. 2025; Wang et al. 2024). **Do not build from this revision.** Revision 4 waits for the researcher's decision on scope (D139).

**Revision 3, 13 Nov 2026.** Nothing is built and no model has been run. Every number here is an exact, model-free computation ([refs3.md](refs3.md), [power.md](power.md), [prelim_info.md](prelim_info.md); scripts beside them). Revision 2 was reviewed independently and found **not sound as written** ([review/review1_design.md](review/review1_design.md)); this revision answers that review and has **not yet been re-reviewed**. No stage-0 code is to be trusted before design review 2.

**What changed from revision 2.**
1. Question 1 is re-aimed (the researcher's decision, 13 Nov): from "does the model update beyond a mask, in distribution" — which a rule-identity lookup passed and which is the training objective's own optimum — to "what does the model do when the evidence leaves every world it trained on". It is scored on counterfactual rejection probes at a fixed state.
2. Question 2 is scored only where it has a defined answer: on decisions where some training world still fits the evidence.
3. The references are four **named mechanisms**, each exact, instead of two posteriors and a convention. The fourth, an **optimistic planner** that holds no belief, was found while computing this revision: under the statistics as first re-aimed it read as the generalising posterior on every count ([refs3.md](refs3.md), reading 1).
4. The mask arm is dropped: its own optimum was not the symbolic mask, and the new change statistic is zero for every deleting model by construction.
5. The conditions under which the training optimum is the Bayes predictor are stated (§7); they are the Decision-Pretrained Transformer's, and the design no longer calls them new.
6. World sets are drawn five times and runs are spread over the draws; goals, not (start, goal) pairs, are held out; the carrier and the grid are fixed here, model-free.

§14 lists where each of review 1's findings went.

## 1. Questions

A prover works in a world whose rules it is never told. The kernel's verdicts on its own proposals are its only evidence. It trained on N worlds.

**Scored.**
- **Q1 — outside the training worlds.** When the verdicts so far fit *none* of the worlds the model trained on, and one more rule is rejected, what does the model do with the rest of its answer: nothing beyond deleting that rule; reroute as the exact posterior over all worlds does; or reroute as a planner with no belief does? As a function of N.
- **Q2 — inside them.** While some training world still fits the verdicts, is the model's answer the memorising posterior (the optimum of its training objective), the generalising posterior, or the planner's? As a function of N.

**Reported, not scored.** Q3: the answer before any verdict. Q4: evidence held in a carried latent against evidence appended as tokens. Q5: what one more loop iteration buys against one more kernel call.

**Not asked here.** Reuse of latent moves; size or depth extrapolation; experts; parallel streams; real Lean; comparison with published systems.

**Why these.** The thesis said the router is an amortized posterior over programs. A posterior that is correct in distribution is what supervised training produces by construction (§2), so the claim with content is about what was learned *as a rule*: does a belief about worlds, learned from N of them, extend to combinations of rules never seen together? That is rung 4's recombination question, asked of beliefs, with coverage as the variable (INT 2021; Raventós et al. 2023).

## 2. What is already known

- With the world known, a verdict carries no information and the exact update is a mask (history Entry 87). Hence a hidden world.
- **A model trained by supervised learning to predict the optimal action, on histories whose actions do not depend on the hidden task, behaves as the Bayes posterior over tasks** — Theorem 1 of the Decision-Pretrained Transformer (Lee et al. 2023), under three conditions: the model fits its training distribution, the histories are "compliant", and there is support. Meta-trained recurrent agents match Bayes-optimal agents in behaviour and internal state (Mikulik et al. 2020). Small transformers match exact posteriors closely (entropy error of order 10⁻³–10⁻² bits) when a fresh hypothesis is drawn for every sequence (Agarwal, Dalal & Misra 2025). So an in-distribution match to the posterior is expected; here it is a gate, not a finding. (All three read in full; [literature.md](../../docs/literature.md), 13 Nov.)
- With a *finite* pool of tasks the optimum of the training objective is the posterior over that pool. In-context regression shows a threshold in the number of tasks above which a transformer behaves like the generalising solution instead (Raventós et al. 2023); INT reports generalisation in theorem proving with ≥ 100 axiom combinations and not with 25.
- Not found (five searches by reviewer 1, three papers read by me): a finite pool of hidden worlds with both posteriors exact, evidence the model elicits from a checker, evidence patterns outside every training world, and cheap non-Bayesian mechanisms scored beside the posterior.

## 3. The carrier (fixed; no pilot reading may change it)

**Terms and rules.** As rung 4: terms over a, b, c, d, 0 with +, at most five leaves (47,155 terms; the whole state graph has about 0.43 M edges and is built once); eight directed rewrite rules Af, Ab, C, If, IbR, IbL, Df, Db; an action is (rule, node). No size extrapolation anywhere.

**Worlds (H8).** A world is a non-empty set of the eight rules: 255 worlds. The model is never told the world. The kernel accepts an action iff its rule is in the world; an accepted action rewrites the state, a rejected one leaves it unchanged. The model's output ranges only over actions that are valid whatever the world (pattern matches, state changes, within the size cap; computed by the executor), so a rejection always means "this rule is not in the world".

**Generator.** Start u from the term sampler; k uniform on {2, 3, 4}; goal s uniform on the sphere of radius k around u with all eight rules; budget B = k + 2 accepted steps. Then the world: uniform over the worlds of the pool in use that can reach s from u within B (the prior is ρ = 0.5 per rule, which is uniform over worlds). A problem no world of the pool can solve is redrawn.

**An episode ends** at the goal; when the budget is used; when the true world can no longer reach the goal within what is left of the budget (nothing after that point is trained on or scored); or, in a model's own rollouts, after 16 kernel calls.

**Why H8, decided here.** With the four axioms as hidden units the problem nearly identifies the world. With the eight rules both scored statistics have signal at 4, 8, 16 and 32 training worlds in each of five draws (§5). Finer units (H10, H12) would, by the first power analysis ([power.md](power.md)), extend the readable range to about the same *fraction* of worlds (an eighth) at about four times the exact-computation cost, and they split rules by position, which is an artefact of the carrier. If the main grid returns "memorising" at every size, the larger carrier is the declared follow-up, as a new experiment.

**Separation.** Five independent draws of the world sets; each draw is 64 held-out worlds and nested training sets T₄ ⊂ T₈ ⊂ T₁₆ ⊂ T₃₂ from the other 191. A sixth draw is used only by the pilot. A fixed fifth of all terms are evaluation-only goals; since the goal is constant along an episode, no scored (state, goal) row occurs in training. The build writes counted overlap tables (goals, worlds, rows) and a test asserts them (L45).

## 4. Four mechanisms, each exact

At a decision the model's inputs are the state v, the goal s, the remaining budget b, and the verdicts so far (rules known present, rules known absent). Four ways of answering:

| | name | output |
|---|---|---|
| **M** | memorising posterior | Σ_W P_T(W \| inputs) · uniform over W's optimal actions, with the prior on the training worlds T. Defined when some training world fits the inputs — the decision is then **in support** |
| **G** | generalising posterior | the same with the prior on all 255 worlds |
| **O** | optimistic planner | no belief: uniform over the optimal actions of the single world made of every rule not yet rejected |
| **D** | deletion | the previous answer with the rejected action or rule removed and the rest renormalised. Every belief that is never revised is a case of D |

"Fits" means: contains every rule known present, none known absent, and can reach s from v within b. The posterior needs nothing else, so it is a function of the model's inputs.

**What training fixes and what it leaves open.** On training problems, M is the unique minimiser of the training loss (§7). Evaluation uses goals never trained on, so a model is at M there only if it learned the computation and not the problems; the gate (§5) checks that on training worlds. Out of support the training distribution has no examples at all, so the loss says nothing; what the model does there is its own generalisation. On held-out worlds 33–72 % of decisions are in support with 4 training worlds and 89–98 % with 32.

**Why O is on the list.** It is the cheapest thing that generalises: it needs no list of worlds and no probabilities, only shortest paths under "everything not yet refused". It differs from G in not hedging. A model could be right about every new world this way without holding a belief, and the two must be told apart for the thesis to have been tested.

## 5. Statistics: regression on the mechanisms

Every statistic is a set of least-squares coefficients of the model's output (or its change) on the mechanisms' outputs (or changes), pooled over decisions (or probes) of an evaluation set. They are linear in the model's output, so independent output noise adds variance and no bias.

**Levels — where the answer sits** (in-support decisions, held-out worlds):
 (p − M) ≈ c_G (G − M) + c_O (O − M).   M reads (0, 0); G (1, 0); O (0, 1).

**Changes — what a rejection does.** A *probe* at a decision is a rule r not yet tested, with a valid action there, that some fitting world lacks. The model is run twice at the same state and budget: on the evidence E, and on E plus "r rejected". Let RM_r delete r's actions and renormalise. The model's change beyond deletion is
 dp = RM_r[p(E + r−)] − RM_r[p(E)],
and dG, dO, dM are formed the same way from G, O, M. dp is exactly zero for D: a model that ignores the verdict, deletes the tried action, or deletes the rule. (RM_r is applied to both outputs; applied only to the earlier one, a model that ignores the verdict reads 0.09 instead of 0.)

- **Out of support** (no training world fits E + r−; held-out worlds):
 dp ≈ b_G dG + b_O dO + s_G g + s_O h,
 where g and h are the two switches of base a model sitting at M may make at the moment the evidence leaves support (M → G and M → O; zero when E was already out of support). G reads (1, 0); O (0, 1); "M in support, G outside" (1, 0) with s_G = 1; "M in support, O outside" (0, 1) with s_O = 1; D zero.
- **In support** (a training world fits E + r−; held-out worlds; reported):
 dp ≈ a_M dM + a_G dG + a_O dO.
- **Gate** (in-support probes, *training* worlds, held-out goals): U = a_M + a_G + a_O, the share of revision by any non-deleting mechanism. M, G and O each read 1; D reads 0.

**Probes as inputs.** Token arm: the pair (r, rejected) is appended to the verdict list. Latent arm: it is fed as the latest verdict, with the registers the model produced at this decision. A rejection changes neither state nor budget, so a probe is an input the model could meet one step later.

**What planted models read** (five draws; full tables in [refs3.md](refs3.md)).

| planted model | levels (c_G, c_O) | out-of-support changes (b_G, b_O) | class it must get (§8) |
|---|---|---|---|
| M | (0, 0) | "M in support, G outside": (0.87–1.02, −0.03–0.08); "…, O outside": (0.00–0.15, 0.84–1.00) | M; G or O outside |
| G | (1, 0) | (1, 0) | G; G |
| O | (0, 1) | (0, 1) | O; O |
| half G, half O | (0.5, 0.5) | (0.44–0.50, 0.50–0.54) | mixed |
| training prior, never revised | (0.00–0.25, −0.02–0.20) | 0 | M (mixed on some draws at 16–32); D |
| full prior, never revised | (0.85–0.99, 0.02–0.11) | 0 | G; D — "a prior without revision" |
| memoriser that falls back to the nearest training worlds | — | (−0.22–0.11, −0.05–0.25); within ±0.11, ±0.08 up to 16 worlds | D (mixed possible at 32) |
| posterior with marginals estimated from the training worlds | (0.52–1.06, −0.24–0.13) | (0.91–1.31, −0.21–0.10) | G — it is one (mixed on levels at 4 on some draws) |
| updater that goes half-way | — | (0.44–0.51, −0.01–0.04) | mixed |
| fitted 8 × 8 rule table | — | 0.12–0.22 up to 16 worlds, ≤ 0.41 at 32 (single-direction statistic) | D or mixed, never G |
| noisy stand-ins at 0 / 0.5 / 1 of the way M → G (noise TV 0.12) | 0.01–0.10 / 0.49–0.64 / 0.95–1.08 | −0.13–0.23 / 0.36–0.62 / 0.88–1.33 | as the noiseless model, within the cell's sampling error |

**Power** (H8, five draws). Signal per instance, levels: 0.16–0.40 at 4 training worlds, 0.05–0.16 at 32, 0.03–0.06 at 64. Out-of-support probes per instance: 8–11 at 4, 0.7–3.1 at 32, under 1 at 64. Sampling sd of each coefficient at 150 instances: levels ≤ 0.05; out-of-support ≤ 0.11 up to 16 training worlds and 0.07–0.25 at 32. Scaled to 4,000 instances per draw: ≤ 0.01, ≤ 0.02 and 0.01–0.05. **Both statistics are readable at 4, 8, 16 and 32 training worlds and at neither at 64**; the 32 cell is the thin one and stage 0 checks it (§9).

**What the statistics cannot see.** Updating on *accepted* steps: an accept changes the state, so there is no fixed-state counterfactual; accepted evidence enters only through levels. Mechanisms outside the span of the four: the residual is reported (share of Σ|dp|² and of Σ|p − M|² not explained), and a large residual is itself a finding.

## 6. Models

One shared transformer block applied n = 4 times with the input re-injected on each pass; logits over (rule, node), with the world-independent validity mask applied in the loss and in evaluation alike (validity is not a question here, and no mass can leak to invalid actions). No halting head. Width and depth start from rung 4's and are set in the pilot.

| arm | what it receives | runs |
|---|---|---|
| **token** | state, goal, remaining budget, and every (rule, verdict) so far; no state carried between rounds | N ∈ {4, 8, 16, 32} × 5 draws × 2 seeds = 40 |
| **latent** | state, goal, remaining budget, only the *latest* (rule, verdict), and four register tokens it wrote in the previous round | N = 16 × 5 draws × 2 seeds = 10 |
| token, loop count n ∈ {1, 8} | as token | N = 16 × 3 draws = 6 (reported) |

**Latent arm, specified.** Four registers of model width; a learned initial value; a "none" verdict token at the first decision; a round's registers are the block's output at the register positions on its last loop iteration; training runs each episode in order with gradients through all its rounds (no truncation; if the pilot shows this unstable, truncation at four rounds is the one permitted change and is reported as a deviation). It sits at N = 16 because both scored statistics are well powered there. Its reading conflates three things — can it revise, can it remember sixteen bits across rounds, was training across rounds stable — so a failure is placed with a linear probe of the registers for the known-present and known-absent sets (reported), and its difference from the token arm is reported, not scored.

**Seeds.** Ten runs per scored cell (five draws × two seeds) is the repository's minimum for a reported result (AGENTS.md, rule 4); the draws are the unit review 1 showed to matter. It is below the ≈ 29 asked for headline claims; results from this grid are "reported".

## 7. Training

- **Supervised only.** Every decision of an episode is a training row: inputs as in §6, target one action sampled uniformly from those optimal in the true world. Loss at the last loop iteration.
- **Fresh episodes throughout.** Problems are never repeated; the N training worlds are the only finite resource. All arms train for the same number of steps, fixed after the pilot; fewer worlds therefore means more steps per world, which is noted with the result, not controlled.
- **Conditions under which M is the minimiser of the loss in support** (Lee et al. 2023, Theorem 1 and Definition 6.1, specialised; review 1, finding 4):
  (a) rows where the true world can no longer finish are not generated;
  (b) the remaining budget is a model input;
  (c) no episode or row is selected on anything that depends on the world beyond (a) — in particular, unsuccessful episodes are kept;
  (d) the problem is drawn first and the world second, as in §3;
  (e) **compliance:** each proposal is conditionally independent of the world given the model's inputs.
- **Proposers.** Each proposal in a generated episode comes, with probability ½ each, from (i) uniform over valid actions of rules not yet rejected, (ii) uniform over the actions optimal if every rule not yet rejected were present. Both are functions of the model's inputs. Neither receives the world; a test asserts that no proposer is called with it (rung 4's proxy proposer took the world as an argument and is not used). The uniform part puts every probe input in the support of training. Histories from the model's own samples are added only if the pilot shows a shift between proposer histories and the scored trajectories.
- **Kept.** Every trained model (L40) and ten checkpoints at fixed steps.

## 8. Measurements, classes and predictions

**Evaluation sets** (exact mechanisms computed once per draw; all on evaluation-only goals).
- *Held-out worlds*: 4,000 instances per draw, decisions along G's greedy trajectory (ties broken by the first maximiser in a fixed action order), teacher-forced. Used at every N of the draw; which decisions are in support depends on N.
- *Training worlds*: 1,000 instances per (draw, N), along M's greedy trajectory. The gate.
- *Own rollouts*: 1,000 per run. Solved, kernel calls, rejections, beside the four mechanisms run as policies. Reported.

**Per run, by code, with 95 % bootstrap intervals** (over held-out worlds; over instances for the gate):

| reading | rule |
|---|---|
| gate | passed if the lower bound of U ≥ 0.5; otherwise the run is "not trained to revise" and is counted as such |
| levels class (Q2) | **M** if the upper bounds of c_G and c_O are both ≤ 0.25; **G** if the lower bound of c_G ≥ 0.5; **O** if the lower bound of c_O ≥ 0.5; otherwise **mixed** |
| out-of-support class (Q1) | **D** if the upper bounds of b_G and b_O are both ≤ 0.25; **G** if the lower bound of b_G ≥ 0.5; **O** if the lower bound of b_O ≥ 0.5; otherwise **mixed** |

**Per cell** (one N, ten runs): a class is the cell's verdict if at least 8 of the 10 runs have it and they come from at least 4 of the 5 draws. Otherwise the cell is "split" and the table of runs is the result. Read at the final step; the same readings at the ten checkpoints are reported as curves (a model may generalise early and memorise later; the verdict is not chosen from the curve).

**Predictions** (directional, with falsifiers; AGENTS.md rule 3).
- **P1 (Q1).** At 4 training worlds the out-of-support class is D. At 32 it is not D (G or O). Falsified at either end by the opposite verdict; a split cell is neither.
- **P2 (Q2).** At 4 training worlds the levels class is M. At 32 it is not M. Falsified likewise.
- **Between G and O I make no prediction.** I do not know whether a network that generalises here hedges. That split, and where between 4 and 32 each class changes, are the readings I expect to learn most from.
- Basis: the task-count thresholds of Raventós et al. and INT. A G or O verdict at 4 training worlds would first be treated as a leak and audited.

**How the two readings combine.**

| in support (levels) | out of support (changes) | what the model is |
|---|---|---|
| M | D | a lookup over its training worlds; nothing transfers |
| M | G | at the training optimum where it exists, the posterior beyond it: a belief over rules was learned beside the list of worlds |
| M | O | at the training optimum where it exists, optimistic replanning beyond it |
| G | G | the generalising posterior |
| O | O | a planner without belief: right about new worlds, without hedging |
| G | D | a prior that is not revised |

The in-support change coefficients (a_M, a_G, a_O) are printed beside each row as the check that a levels class comes with revision of the same kind.

**Also reported.** Residual shares (§5). Levels at out-of-support decisions on (G − O). First-decision levels (Q3). Revision 2's β with its reference lines (rule-identity lookup 0.53; accepts conditioned plus rule mask 0.70), for continuity. Share of held-out episodes whose whole evidence some training world explains (L45). Latent against token (Q4); loop counts (Q5).

**Planted tests, before any model.** The class code must return the right-hand column of §5's table for every planted model, at noise 0.05–0.20 for the stand-ins, on every draw and size (L39, L42, L48, L50).

## 9. Stages and reviews

| stage | what | gate |
|---|---|---|
| **Design review 2** | this document, by a reader who did not write it | — |
| 0. Derive (CPU) | carrier module on the prebuilt state graph; the four mechanisms; statistics and class code with planted tests; evaluation sets; overlap tables; compliance test; data throughput | every planted model classed as §5 says; sampling sd of each scored coefficient ≤ 0.04 in every (draw, N) cell at the chosen evaluation size — a cell that cannot meet it is declared unscored *before* any model exists; episode generation ≤ 10 ms |
| Code review | independent | — |
| 1. Pilot (the pilot draw only) | token at N = 16 (two seeds) and N = 4 (one), latent at N = 16 (two): do the arms pass the gate; learning rate, width, step count; stability of the latent arm; shift between proposer histories and scored trajectories; cost per run | curves by window means (L46); cost projected from the last interval (L44) |
| Pass-rule audit | pilot-blind, independent | — |
| 2. Main grid | 56 runs (§6) | — |
| Results review | independent | — |

**What the pilot may change:** optimisation settings, width, step count, the one latent-arm truncation, the addition of own-sample histories. **What it may not:** the carrier, the grid, the statistics, the classes, the predictions. No reading of a scored statistic is taken from the pilot (L30, L39).

## 10. Confounds, and what the design does about each

| confound | where it bit | handled by |
|---|---|---|
| The mask does the work | rung 4 P8; Entry 87 | hidden world; change statistics that are zero for every deleting model |
| A lookup passes the updating test | revision 2 (rule-identity lookup read 0.53) | counterfactual probes with the rule deleted from both outputs |
| The pass is the training optimum | revision 2, P1 | the in-distribution reading is a gate; the scored reading is out of support |
| An undefined reference on part of the set | revision 2, γ's fallback | Q2 on in-support decisions only; out of support scored by changes against defined mechanisms |
| A mechanism with no belief reads as the posterior | this revision (optimistic planner) | O is a named direction in every regression |
| The base moves when the evidence leaves support | this revision | two switch directions controlled for |
| A prior that is never revised reads as its posterior | review 1 | levels and changes read together |
| Proposals leak the world | Lee et al. 2023 (compliance) | proposers without access to the world, asserted by test |
| Selection on outcomes | review 1 | conditions (a)–(d) of §7 |
| One draw of the world sets | review 1 | five draws; verdict needs four of them |
| Scored rows overlap training | rung 4 (36 %); review 1 | evaluation-only goals; counted and asserted |
| Ties in the reference trajectory | review 1 (23 % of decisions) | one fixed tie rule |
| Mass on invalid actions | review 1 | validity mask in the model |
| Checkpoint choice | review 1; L46 | fixed final step; curves reported |
| Carrier chosen by a pilot reading | review 1; L30 | carrier and grid fixed here |
| Encoding range, canonical targets, early commitment | rung 4; HRM/TRM analyses | no size extrapolation; uniform over optimal actions; loss at the last iteration |

## 11. Compute and data

- **Exact side.** The state graph builds in 7 s. With it, an episode needs a bounded search in one world; the target is ≤ 10 ms per episode (to be measured at stage 0), so a million training episodes cost about three core-hours and can be streamed. Evaluation needs all 255 worlds per instance: about 1 s per held-out-world instance with today's script (most of it instances drawn and discarded), so five draws of 4,000 held-out-world instances and the training-world sets are of order ten core-hours, once.
- **Training.** Rung 4's main grid was 41 runs in 45 process-hours. This grid is 56 runs, ten of them episode-ordered (L44: double their estimate): about 75 process-hours as a planning figure, to be replaced by the pilot's projection. If that exceeds what the researcher wants to spend, the cuts in order are: the loop-count runs (6), then three draws instead of five (which breaks the ten-run minimum and would be said so).

## 12. Decisions

**Taken by the researcher.** Scored questions are Q1 and Q2 (D134). The latent arm is included at one coverage level (D136). Q1 is re-aimed at evidence outside the training worlds, scored by counterfactual probes (D137, 13 Nov).

**Taken in this revision, open to the researcher and to review 2.**
1. The optimistic planner as a third reference, so that each scored reading is one of four mechanism classes.
2. Grid of 4, 8, 16, 32 training worlds; five draws × two seeds per cell; latent arm at 16.
3. H8, fixed model-free; a larger carrier only as a declared follow-up.
4. No mask arm.

## 13. Limits, stated in advance

Synthetic, rewrite-only proofs; the hidden fact is the presence of a rule; eight rules, 255 worlds; small models. The readable range ends at 32 training worlds, an eighth of the worlds: a verdict of M at every size would say the threshold, if there is one, lies beyond the range, and would not show that the architecture cannot hold a belief over rules. The posterior and the planner are told apart only by hedging, on this carrier's geometry. Revision on accepted steps is not probed. The prior is uniform, so nothing is learned about whether a model picks up a prior's shape. Behavioural differences are small here (a wrong proposal costs one rejection), so the claims will be about answers, not solve rates. Nothing speaks to real Lean, and none of it is a comparison with existing provers.

## 14. Where review 1's findings went

| finding | disposition |
|---|---|
| 1. The mask arm's own optimum is not the mask | arm dropped; zero is by construction (§5) |
| 2. A rule-identity lookup passes P1; the pass is foregone in distribution | Q1 re-aimed; probes; in-distribution reading demoted to a gate (§1, §5, §8) |
| 3. γ's zero rests on a fallback; a non-updating model reads "generalising"; P2 had no rule | in-support decisions only; levels read with changes; rules over runs and draws with a falsifier and a split outcome (§5, §8) |
| 4. Unstated conditions for the Bayes-predictor claim | §7 (a)–(e) |
| 5. One draw of the world sets | five draws (§3, §8) |
| 6. Disjoint pairs do not separate rows | evaluation-only goals (§3) |
| 7. Ties in the scored trajectory | fixed rule (§8) |
| 8. Dead-world decisions scored | never generated or scored (§3) |
| 9. Text and code disagreed on the mask reference | that reference is gone; mechanisms are defined once, in §4, and the code's docstring repeats them |
| 10. A pilot reading decided the carrier | fixed here (§3, §9) |
| 11. No checkpoint rule; N confounded with steps per world | final step, curves reported; confound stated (§7, §8) |
| 12. Prior work | §2; literature.md |
| 13. Latent arm unspecified; its reading conflated | §6 |
| 14. Compute and data not budgeted | §11 |
| 15. Input and output underspecified; seeds; saving models | §3, §6, §7 |
| 16–22 (notes) | product-of-marginals learner classed with G (§5); the count of held-out episodes explained by a training world is printed (§8); uniform prior stated as a limit (§13) |
