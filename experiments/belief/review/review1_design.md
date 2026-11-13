# Review 1 — the belief experiment's design (DESIGN.md revision 2, 13 Nov 2026)

Reviewer: independent; did not write the design. Nothing was trained. All numbers below are from my own code (`review1_scripts/`, none of which imports `power.py` or `prelim_info.py`; only the repo's executor is reused). I read `power.py`, `power_coverage.py` and `prelim_info.py` for bugs. Carrier everywhere: H8, ρ = 0.5, Δ = 2, k ∈ {2, 3, 4}, size cap 5.

## Verdict: not sound as written

The carrier and the exact mathematics hold: the posterior over worlds and the Bayes-predictor formula are correct under conditions the design leaves partly unstated, and every headline number reproduces. The two scored rules do not hold up:

- **P1 voids itself.** The design's instrument check (mask arm |β| < 0.15) fails for a *perfectly trained* mask arm, whose Bayes-optimal β is 0.13–0.46 depending on its inputs and on the proposer mix. And the pass bar (β ≥ 0.5) is reached by a rule-identity lookup with no belief revision (β = 0.53).
- **P2 has no decision rule**, and γ's zero rests on an arbitrary fallback (absent from DESIGN.md) that carries 41–46 % of γ's denominator at 8–32 training worlds.

Both can be repaired on the same carrier, with no extra training runs: the changes are to the scored statistics, the evaluation probes and the written rules. Findings 1–4 are blocking; they should be settled before stage 0's code is written, because they change what stage 0 computes.

---

## Findings

### Blocking

**1. The mask arm is not a neural m̃; its Bayes optimum has β = 0.13–0.46, so the design's own rule voids P1.**

Sentences concerned: §5, *"mask | nothing; actions rejected at this state are masked at the output | the neural mask-only reference: its β must come out near 0"*; §7, *"The mask arm must give |β| < 0.15; if it does not, the instrument is broken and nothing is read from P1"*; §4, *"0 for a mask, 1 for the posterior"*.

m̃ is a belief frozen at P(W | u, s). The mask arm is a model that sees the *current* state. The current state is a function of the past: reaching v with b steps left means the rules on some path u → v were accepted, and under a goal-directed proposer it also says which preferred actions were probably rejected on the way. The arm's cross-entropy minimiser is the label distribution averaged over the histories that lead to its input, not m̃.

I computed that minimiser (`t5_maskarm.py`): per problem, Monte Carlo over the generative process (4,000 episodes, W ~ P(W | u, s), world-blind proposers, decisions dropped when the true world cannot finish), the arm's optimum under its output mask fitted per input by an MM iteration, then scored on the q-greedy trajectory like any model. 150 problems, 411 scored post-verdict decisions.

| proposer mix in training | input = (state, goal, remaining budget) | input = (state, goal) |
|---|---|---|
| all uniform | β = 0.345 (after an accept 0.43, after a rejection 0.10) | 0.125 |
| half q-sampling, half uniform | 0.456 (0.52 / 0.23) | 0.211 |
| all q-sampling | 0.464 (0.53 / 0.23) | 0.217 |

Check of the machinery: the same episodes keyed by (state, budget, known-present, known-absent) give β = 0.991–1.001 and mean TV to q of 0.008–0.017. Coverage: with q-sampling in the mix every scored decision's input was visited often enough to fit (409–411 of 411); under the all-uniform proposer 310 (budget-aware) and 346 of 411 were, carrying 65 % and 80 % of the signal energy, so that row is on a subset.

Five of the six cells exceed 0.15. A correctly trained mask arm would therefore declare the instrument broken. Two further consequences:

- β of the mask arm depends on the proposer mix (0.35 against 0.46), so it is not a property of the arm.
- A *token* arm that ignores its verdict tokens entirely would score the same 0.13–0.46. Part of what β credits as "the exact update beyond a mask" can be read off the state.

Caveat on my number: the fit is per problem, so the arm implicitly knows the start term. For the budget-aware column this is an upper bound on what a pooled model can infer; the mechanism (the state reveals accepted rules) does not depend on it — the exact posterior conditioned on accepted rules only, plus the action mask, already has β = 0.29 [0.26, 0.34] (`qP` in `t2_beta.py`).

**Change.** Remove the |β| < 0.15 void rule. Either (a) drop the mask arm (three runs saved) and rely on planted symbolic references, or (b) keep it and write its *computed* optimum into the design as its expected value. In both cases β's zero must stop being m̃ — see the statistic proposed under finding 2, which is exactly zero for any history-blind model.

**2. P1's bar is reached by a rule-identity lookup; the token arm's pass is close to foregone and would not show posterior updating.**

Sentences concerned: §7 P1, *"β ≥ 0.5 … the model makes at least half of the exact update"*; §1 Q1, *"beyond what masking the rejected actions gives"*; §9, *"The mask does the work | … | hidden world; m̃ as the zero of β"*.

In this carrier a rejection means the rule is absent, so every action of that rule is dead at every node and every later state. m̃ removes only *the actions rejected at this state*. The gap between the two is a table lookup on the verdict tokens. Scored as models on the q-greedy trajectory (600 instances, 1,624 post-verdict decisions, seed 0; seed 1 in brackets; `t2_beta.py`, `t6_misc.py`):

| "model" | what it does | β | share of the squared distance from m̃ to q that it closes |
|---|---|---|---|
| rule mask | frozen belief; all actions of rejected rules removed | **0.526**, bootstrap 95 % [0.43, 0.61] (0.530) | 0.52 |
| rule mask + boost | same, and actions of accepted rules multiplied by e^1 … e^2 | 0.61–0.67 (0.60–0.64) | 0.34 … −0.40 |
| accepts + rule mask | exact conditioning on accepted rules, rule mask for rejected ones | 0.702 [0.63, 0.77] (0.679) | 0.70 |
| rejections only | exact conditioning on rejected rules, accepted ones ignored | 0.874 (0.904) | 0.87 |
| accepts only + action mask | | 0.293 (0.268) | 0.38 |

Where β's signal sits: 89–92 % of Σ|q − m̃|² is on decisions with at least one earlier rejection, and 76–82 % on decisions after a state change — that is, mostly m̃ having *forgotten* a rejection made one state earlier, which no model given the verdict tokens will do.

What is left once the lookups are taken out:

- exact update against the rule mask: 48 % of the energy; TV > 0.05 on 29–30 % of post-verdict decisions; mean TV 0.062–0.064;
- exact update against accepts + rule mask: 30–32 % of the energy; TV > 0.05 on 11.5–12.6 %; mean TV 0.031–0.035. That is the level of H4 (12–13 %, 0.037), which §3 rejects as nearly degenerate.

So the answer to "is the exact update essentially a rule-level mask plus upweighting worlds with accepted rules" is: about half of it by β is the rule-level mask alone, about 70 % is the mask plus conditioning on accepted rules.

Why the token arm's pass is foregone (L27, L28): the token arm is trained toward a fixed target whose expectation *is* q. "The posterior is never a training target" (§4) is true per sample only. power.md's own table says a model at rung 4's error level (TV 0.10–0.17 from its target) reads β = 0.97–0.98. The only way to miss 0.5 is not to learn the task. And a pass cannot separate "updates as a posterior" from "fitted a supervised map of (state, goal, 16 evidence bits) in distribution" — in distribution these are the same thing.

A further instability: the rule mask's β sits on the bar, and its sampling sd is 0.025 at 2,000 instances, not the 0.004–0.008 of power.md (that figure is for unstructured noise around the line). A lookup model would flip between "pass" and "partial" by seed.

**Change.**
(a) Score Q1 on **counterfactual single-rejection probes at a fixed state**. For a decision with evidence E and a rule r not yet tested: Δq_r = q(· | E, r rejected) − RM_r[q(· | E)], where RM_r deletes all actions of r and renormalises; the model's Δp_r is formed the same way from *its own two outputs*; β_cf = Σ⟨Δp, Δq⟩ / Σ|Δq|². This is exactly 0 for any model whose response to a rejection is to delete actions from its own previous distribution (action-level or rule-level), for any history-blind model (the state is the same on both sides) and whatever the model's error in the no-evidence belief; it is 1 for the exact posterior. Signal (`t7_probes.py`, 300 instances): 13.9 probes per instance, mean TV 0.073, > 0.05 on 33 %, energy 0.416 per instance against 0.127 for the present β. Restricted to rejections that actually occur on the q-greedy trajectory the same statistic has less power (0.35 decisions per instance; sd ≈ 0.045 at 2,000 instances; a rule-mask-on-frozen-belief model reads 0.36–0.42 there, an own-mask model exactly 0; `t2_beta.py`). The probes need the uniform proposer in the training mix so that they are in distribution.
(b) Keep β against m̃ as a reported quantity, with the reference lines beside it: rule mask 0.53, accepts + rule mask 0.70, the state-only optimum from finding 1.
(c) Make the scored question one whose outcome is not fixed by the training target: read β_cf on **evidence patterns no training world agrees with** (held-out worlds; these are the decisions of finding 3 where no Bayes predictor of the training distribution exists). There "exact update, rule mask or mask" is an open question about extrapolation, which is the program's question. Training-world β_cf becomes a check.
(d) Run the planted negations through whatever rule replaces P1 before it is written down (L39, L48): rule mask, state-only optimum, accepts + rule mask.

**3. γ's zero is an arbitrary convention on the decisions that carry 41–46 % of its denominator; a non-updating model reads "generalising"; P2 has no rule.**

Sentences concerned: §4, *"Memorising posterior q_train. The same formula with the prior restricted to the training worlds"* and *"γ … 0 for the memorising posterior, 1 for the generalising one"*; §7 P2, *"the result is the curve and where it crosses 0.5"*.

(i) *The fallback.* In a held-out world the evidence eventually contradicts every training world, and "the same formula" then has an empty support. `power.py` (`Instance.trace`) and `prelim_info.py` (`train_scores`) fall back to the frozen training-world belief with the actions rejected at this state masked, then to q; when no training world is consistent with the problem at all, the reference is set to q. DESIGN.md does not mention any of this. Counts on the primary draw (800 held-out instances, re-implemented; `t4_coverage.py`):

| N | instances with no consistent training world (reference := q) | decisions: in support / fallback to mask / fallback to q / no consistent world | share of γ's denominator on fallback-to-mask decisions | instances where the evidence contradicts every training world |
|---|---|---|---|---|
| 8 | 27 % | 57 / 7 / 5 / 31 % | 41 % | 17 % |
| 16 | 4 % | 80 / 10 / 4 / 5 % | 46 % | 19 % |
| 32 | 0 % | 90 / 8 / 2 / 0 % | 46 % | 12 % |
| 64 | 0 % | 100 / 0 / 0 / 0 % (fallbacks below 0.5 %) | 9 % | below 0.5 % |

Over four other draws of the world sets the fallback share of the denominator is 30–44 % at N = 8, 23–47 % at 16, 10–33 % at 32 and 6–24 % at 64. On those decisions the model's input is outside its training support; no Bayes predictor of the training distribution exists, and "memorising" means whatever the convention says. Three conventions, scored as models against the author's reference:

| N | memoriser that falls back to the rule mask | to the nearest training worlds | to q |
|---|---|---|---|
| 8 | γ = 0.02 | 0.02 | **0.41** |
| 16 | 0.05 | 0.05 | **0.46** |
| 32 | 0.08 | 0.09 | **0.46** |
| 64 | 0.03 | 0.04 | 0.09 |

A model that is a pure memoriser wherever memorising is defined, and behaves sensibly once its training worlds are refuted, reads 0.41–0.46 — on the 0.5 line the design reads (0.23–0.47 across the other draws at N ≤ 16).

(ii) *γ does not measure updating.* A model with the full prior that never revises it (m̃) reads γ = 0.82 / 0.72 / 0.71 / 0.99 at N = 8 / 16 / 32 / 64 (0.71–1.00 across draws); with the rule mask 0.92–1.02. γ near 1 is therefore not "its belief is the generalising posterior" (§1 Q2); it is "its prior is not confined to the training worlds".

(iii) *No rule.* P2 states an expectation, no statistic over seeds, no falsifier, no name for the outcomes. AGENTS.md hard rule 3 and L39 (iii) both require them. The expectation ("above 0.5 by 64") is placed on the one grid point the design itself calls weak, which is the L36 pattern.

(iv) *The ruler changes with the dose* (L32). The share of decisions that are in support goes from 57 % to 100 % across the grid, and 27 % of instances at N = 8 contribute nothing by convention. The γ curve compares different effective decision sets.

**Change.** Score γ only on in-support decisions (some training world agrees with all the evidence), where q_train is the unique Bayes predictor of the training distribution. Report the count of such decisions per N beside it. Treat out-of-support decisions as a separate, named measurement against several references (q; rule mask on the training belief; nearest training worlds; mask) — this is also where finding 2 (c) puts Q1. Put β_cf on held-out worlds beside γ, since γ alone cannot tell a generalising posterior from a generalising prior. Write P2 as a rule: a statistic over seeds *and* world-set draws, a bar, a falsifier, an in-between outcome; compute the verdict in code with planted policies including the three memorisers and m̃ above.

**4. The Bayes-predictor claim depends on four things the design does not state.**

Sentence concerned: §4, *"The cross-entropy minimiser given (state, goal, evidence) is q(a) = Σ_W P(W | …) · 1[a optimal in W] / #optimal_W, over the worlds that can still finish within the budget."*

I checked the claim by brute force (`t3_sim.py`): six fixed problems, 300,000 episodes per proposer, one uniformly sampled optimal action as the label, empirical label frequencies tabulated by (state, budget, known-present, known-absent) and compared with q.

| proposer | world-blind? | mean TV(empirical, q) | same for labels drawn from q (null) | largest \|z\| over cells |
|---|---|---|---|---|
| uniform | yes | 0.0053 | 0.0049 | 4.2 (≈ 3,300 cells) |
| a sample from q | yes, history-dependent | 0.0026 | 0.0025 | 3.2 |
| q-greedy until the first rejection, then uniform over untested rules | yes, verdict-dependent | 0.0017 | 0.0018 | 2.9 |
| optimal in the true world with probability 0.5, else uniform | no | 0.034 | 0.004 | 135 |
| the sampled label itself (teacher forcing on the target) | no | 0.056 | 0.002 | 229 |
| a sample from q, **only episodes that reach the goal kept** | proposer yes, selection no | 0.015 (0.034–0.041 on two problems) | 0.003 | 102 |

So the claim holds — given the following, each of which has to be written into §3–§6:

- (a) **Decisions at which the true world can no longer finish are dropped from training.** The restriction "worlds that can still finish" is then legitimate, and it is the *only* reason it is: it is a selection on the world, and the formula absorbs it. The design does not say what label such a decision gets. Any other choice (a label toward the goal ignoring the budget, say) makes q wrong.
- (b) **The remaining budget is a model input** (or the start term and k). Given the live restriction, (state, goal, budget, known-present, known-absent) is sufficient — the condition d_W(u, s) ≤ k + Δ is implied. Without the budget it is not: q at the same state and evidence with one step less differs by TV > 0.05 on 26 % of decisions (mean 0.048; `t6_misc.py`). "Given (state, goal, evidence)" is not enough.
- (c) **No selection of episodes or decisions on anything that depends on the world beyond (a)**: not on success, not on a stratum such as rung 4's binding or GP1 rejection, not on the number of rejections. The last row of the table is the counterexample asked for: a world-blind proposer, and q is still not the Bayes predictor.
- (d) **Sampling order for a restricted world set**: the problem first, then the world from the prior restricted to the consistent training worlds. Drawing a training world uniformly and then a problem it can prove gives P(W | u, s) ∝ 1 / P(consistent | W), and q_train's weights are wrong.

On the other parts of the question: k being a function of (u, s) causes no problem; the budget, the fact that the episode continues and the states reached add nothing beyond (state, budget, known-present, known-absent), because termination is a function of the visible history.

World-blindness is sufficient only together with (c). It is not necessary in the form written (*"proposers that see only what the model sees"*): what is needed is that the proposal is conditionally independent of the world given what the model is given. A proposer may depend on earlier verdicts (third row), on the start term, on its own randomness. This is the *compliance* condition of Lee et al. 2023 (finding 12). One concrete risk: rung 4's proxy code (`proxy.py`: `lookahead_value`, `ladder_rollout`) takes the world as an argument; the "goal-directed proxy proposer" must be called with the full world, and a test should assert that no proposer receives W.

### Should-fix

**5. The world sets are drawn once; the coverage axis the design relies on is a property of that draw.**

Sentence concerned: §6, *"with 8–32 training worlds they differ by TV 0.15–0.17 on average; with 64 or more by 0.04"*; *"drawn once, nested"*.

Mean TV(q, q_train) over five draws of the held-out and nested training sets (763–800 held-out instances each; the first column is the author's draw):

| N | draw 7 | draw 8 | draw 9 | draw 10 | draw 11 |
|---|---|---|---|---|---|
| 8 | 0.159 | 0.195 | 0.149 | 0.176 | 0.160 |
| 16 | 0.168 | 0.154 | 0.122 | 0.155 | 0.123 |
| 32 | 0.148 | 0.064 | 0.079 | 0.130 | 0.078 |
| 64 | 0.042 | 0.056 | 0.054 | 0.063 | 0.047 |

In three of five draws the separation at 32 is half the quoted value; the informative range is then two grid points. The author's draw is also unusual at N = 8: 27 % of held-out instances have no consistent training world, against 2–9 % in the others (its eight worlds have sizes 1, 2, 2, 3, 4, 5, 5, 5, while the true world averages 5.1 rules). Model seeds do not sample this factor, so "where γ crosses 0.5" would be a statement about one draw.

DESIGN.md also quotes two different numbers for this quantity: §4's table (46–47 %, mean TV 0.10–0.14 at 8–32; from `prelim_info.py`, whose training sets are not nested and whose held-out set is the complement) and §6 (0.15–0.17; from `power_coverage.py`). At N = 64 the two sources give 0.08 and 0.04.

**Change.** Cross world-set draws with model seeds (at least three draws at N ≤ 32), or make the draw the unit of the bootstrap. Quote one source. Decide H8 against H10 now, from these model-free numbers, not from the pilot (finding 10).

**6. Disjoint (u, s) pairs do not make the scored rows disjoint from training.**

Sentence concerned: §3, *"(u, s) pairs are disjoint between the two"*; §9, *"Eval overlaps training | … | disjoint pairs, held-out worlds, counted and asserted"*.

A model is scored on rows (state, goal, budget, evidence) along a trajectory. An evaluation trajectory from u to s passes through states v; (v, s) can be a training pair or an intermediate row of a training episode with a different start. With 47,155 terms and balls of about 1,500 states this is likely to be common, and it is the level at which rung 4's recall was found (review 6: about 30 % of held-out instances were trained rows in a contained world). The overlap table lists pairs, start terms, goals and worlds, not rows.

**Change.** Hold out *goals*: s is constant along an episode, so every scored row is then new. Otherwise add (state, goal) and (state, goal, evidence) rows to the counted table and print every result on the never-trained rows beside the pooled one (L45).

**7. The scored trajectory is not defined: 23 % of q-greedy decisions have an exact tie for the argmax of q.**

Sentence concerned: §7, *"the exact q-greedy trajectory, teacher-forced"*.

519 of 2,224 decisions (seed 0). `prelim_info.py` and `power.py` break ties in different orders, and by raw float comparison. My recomputation of power.md's Q1 line differs from it only through this (1,624 decisions against 1,618; TV > 0.20 on 17.5 % against 19 %). **Change.** Fix the rule in the design (a stated order with a tolerance), or score on trajectories sampled from q with a fixed seed.

**8. Decisions at which the true world can no longer finish are scored.**

On the q-greedy trajectory 3.7 % of all decisions (28 of 747 on 200 instances) and 4.5 % of the post-verdict ones (73 of 1,624), carrying 5.4 % of β's energy, are at states from which the true world cannot reach the goal in the remaining budget. No training row exists with that world at that input; q there is conditioned on an event that is false in the episode. A Bayes learner will still output q, so the comparison is consistent, but "the exact posterior" is the wrong name for the reference at those decisions. **Change.** Report them separately, or end the scored trajectory there.

**9. The text and the code disagree on m̃.**

§4: *"The belief over worlds frozen at P(W | u, s), with the actions rejected at this state removed."* In both scripts the frozen belief is also restricted to the worlds that can still finish from the current state within the remaining budget, which revises it after accepted steps. All quoted numbers use the code's version. **Change.** State the code's definition, and write down for every reference its information set (L39 (ii)): m̃ knows the state and budget but not the accepted rules; the mask arm knows the state.

**10. A pilot reading of a scored quantity decides the carrier.**

§3, §11: the ten-unit grouping is used *"if the pilot puts Q2's crossover beyond 32 training worlds"*; on-policy data *"only if the pilot shows a shift"*. A three-seed, short pilot reading γ's crossover is a pilot surprise used as a reading (L30), and an amendment made after a pilot has to argue its independence from it (L39). Finding 5 already says the H8 axis is two to three points. **Change.** Decide at stage 0, by a written numerical rule on model-free quantities; give the pilot's shift check a threshold in advance.

**11. P2's reading has no checkpoint rule and confounds N with training time per world.**

Whether a model sits at the memorising or the generalising solution is known to move during training. The design reads γ once, at a step count set by the pilot; at fixed steps a model trained on 8 worlds has seen each world eight times as often as one trained on 64. **Change.** Record γ and β_cf at every checkpoint and read from window means (L46 is cited in §9, but no curve is planned for γ); state the checkpoint rule; report at matched steps and at matched training loss.

**12. Prior work the design does not cite, one piece of which is §4 and §6.**

Sentences concerned: §2, *"none compares a model's belief with an exact posterior"*, *"Not found in targeted searches … a recurrent prover's belief compared with the exact posterior conditioned on kernel evidence"*; §9, *"Proposals leak the world | (new)"*.

- [Lee et al. 2023, *Supervised Pretraining Can Learn In-Context Reinforcement Learning*](https://arxiv.org/abs/2306.14892) (Decision-Pretrained Transformer). A transformer trained by supervision to predict an optimal action in a hidden task from a query state and a set of past interactions; Theorem 1 shows it implements posterior sampling under the pretraining prior, *provided the in-context data are "compliant"* (each action conditionally independent of the task given the state and the earlier data; expert-biased data violate it). That is §4's Bayes-predictor argument and §6's world-blind requirement. It compares regret, not the action distribution against an exact posterior, and has no sweep over the number of tasks.
- [Agarwal, Dalal & Misra 2025, *The Bayesian Geometry of Transformer Attention*](https://arxiv.org/abs/2512.22471) ("Bayesian wind tunnels"). Small transformers compared with closed-form posteriors on a hypothesis-elimination task and on HMM filtering, to 10⁻³–10⁻⁴ bits, with MLP, LSTM and Mamba baselines. Passive sequences; no chosen queries, no accept/reject feedback, no coverage sweep, no prover.
- [Mikulik et al. 2020, *Meta-trained agents implement Bayes-optimal agents*](https://proceedings.neurips.cc/paper/2020/hash/d902c3ce47124c66ce615d5ad9ba304f-Abstract.html). Memory-based meta-learned agents compared with Bayes-optimal ones on prediction and bandit tasks, in behaviour and in computational structure.

With these, "a model's distribution compared with an exact posterior" and "a recurrent agent compared with the Bayes-optimal one under interactive evidence" both exist. The narrow claim — a prover, kernel verdicts on its own proposals, the next-step distribution against the exact conditional posterior, with coverage as a variable — was not falsified by my searches. Those were five queries and eight pages (listed under "What I could not check"); the wind-tunnel paper was read to 200,000 of 210,670 characters and the DPT paper in full through a summarising reader.

**Change.** Cite the three; replace "(new)" by the compliance condition; state the claim as the narrow one.

**13. The latent arm is not specified, and its P1 conflates three things.**

§5–§6: *"a few register tokens carried from the previous round"*, *"Gradients flow across rounds; the truncation length and the stability of this are pilot questions."* Missing: the number of registers and their initial value; whether a round's registers are the block's output at the last loop iteration; how training rows are formed (registers cannot be cached, so one training example is a whole episode run in order, on proposals that are not the arm's own unless it generated them); the truncation; what "the latest verdict" is at the first decision. A failed P1 for this arm would not separate "cannot update" from "cannot remember 16 bits across rounds" or "training across rounds was unstable". **Change.** Specify these; add a probe of the registers for the known-present and known-absent sets, so that a failure can be placed.

**14. Compute and data are not budgeted.**

One instance costs 0.16 s on this machine, of which 0.086 s is the full-world ball (`t6_misc.py`). A million training episodes is about 45 core-hours before any model step. *"The model's own samples (refreshed during training)"* needs labels at states the stored data do not contain, which means the ball and the per-world distances again for every refreshed episode; storing the distances is about 0.4 MB per instance for 255 worlds. The design gives no dataset size, no refresh schedule, and says nothing about whether labels are resampled. §10's "no more than rung 4's 45 process-hours" is for 34 runs and predates the latent arm's episode-at-a-time training (L44 asks for a 2× margin on policy-dependent arms). **Change.** State the dataset size, whether it is fixed or streamed, the refresh schedule and its cost; compute distances only for the training worlds when N is small.

**15. Model input and output are underspecified in ways that move β.**

- The output is *"a pointer output over (rule, node)"*. The references are distributions over the full-world-valid, state-changing, within-cap actions. If the model's distribution is not renormalised over that set, mass on invalid actions lowers β linearly: 10 / 20 / 30 % leaked gives 0.91 / 0.81 / 0.72 for an otherwise exact model.
- What the kernel does with a proposal whose pattern does not match, that exceeds the cap or that leaves the term unchanged is not said; if it is a rejection, *"a rejection means 'this rule is not in the world'"* (§3) is false for the model's own rollouts.
- *"rejections are counted, not capped below 8"* (§3) is not a rule.
- Seeds: 5 per cell, 3 for the mask and loop-count cells, with a 4-of-5 rule; AGENTS.md hard rule 4 asks for at least 10 paired seeds for reported results. Say why 5.
- Saving every trained model (L40) is not mentioned.

### Notes

**16. β is robust to the error structures I tried; this part checks out.** Mask-only models with wrong beliefs stay near 0: action temperature 0.5–2.0 gives −0.05 … 0.03; a wrong prior (ρ = 0.35, 0.75, or tilted by world size) −0.03 … 0.01; 30 % leaked to uniform 0.02; a hard argmax −0.11; a 30 % bias to the first supported action 0.12 (the largest, close to the 0.15 bound). Exact updaters stay near 1: temperature 0.5–3.0 gives 0.98–1.02; wrong prior 0.92–1.05; canonical bias 1.00; argmax 1.03; the only reductions are leaks (finding 15). No non-updating model reached 0.5 and no correct updater fell to 0.25. The weakness of β is where its zero is (findings 1–2), not its behaviour under noise.

**17. P1 on training worlds at N = 64: the choice of prior is harmless, but say which.** The model's Bayes predictor there is q_train(64), not q. On 845 training-world instances (2,355 post-verdict decisions) they differ by mean TV 0.044; q_train(64) scored against the full-prior references reads β = 1.000 and the mask under the 64-world prior 0.023.

**18. A learner with a product prior whose marginals are estimated from the training worlds reads as generalising.** γ = 0.83 / 0.92 / 0.94 / 0.95 on the primary draw at N = 8 / 16 / 32 / 64, and 0.69–1.02 over the other draws (0.62 at the lowest, in-support decisions only). The 0.5 reading is safe against it; reading γ < 1 as "partly memorising" is not.

**19. Leakage between held-out and training worlds is small by the strict count.** A training world identical to the true world on every rule that occurs in the problem's ball: 0–2 % of held-out instances at every N. By the loose count it is not: at N = 8 the evidence of the whole scored episode is explained by some training world in 56 % of held-out instances on the primary draw (77 % on draw 8). Print that share with the result; "on worlds it never trained on" then has its count (L45).

**20. Every single-rule verdict occurs in training at N = 8.** In five draws no rule is present in all or in none of the eight training worlds (the minimum is one world of eight). L34's input-side check passes for single tokens; pairs of verdicts are the out-of-support part (finding 3).

**21. ρ = 0.5 makes the prior uniform over the consistent worlds.** ρ^|W| (1 − ρ)^(8 − |W|) is constant. The carrier then cannot say whether a model learned the prior's shape; Q3 measures only the consistency constraint. The author's own row for ρ = 0.35 keeps 5.9 bits and 35 % informative decisions.

**22. The scripts.** I found no bug in `power.py`, `power_coverage.py` or `prelim_info.py` that changes a reported number. Beyond findings 3, 7 and 9: the stand-in noise is independent per decision, so the quoted sampling sd describes within-model noise only (the author says so in §7).

---

## Recomputed against reported

| quantity | reported | recomputed | |
|---|---|---|---|
| world entropy given (u, s) | 6.2 bits (6.22) | 6.218 | match |
| consistent worlds | 82 of 255 | 82.3 | match |
| true world size | 5.1 | 5.06 | match |
| first-step entropy, known / hidden | 0.95 / 1.57 | 0.955 / 1.573 | match |
| P(first proposal rejected) | 0.13 | 0.129 | match |
| oracle: solved / calls / rejections | 1.00 / 3.5 / 0.0 | 1.000 / 3.46 / 0.00 | match |
| q-greedy | 0.93 / 3.8 / 0.4 | 0.930 / 3.78 / 0.43 | match |
| mask-greedy | 0.90 / 4.0 / 0.7 | 0.897 / 4.03 / 0.71 | match |
| post-verdict decisions with TV(q, m̃) > 0.05 (300 instances) | 33 % of 832; mean TV 0.09 | 33.0 % of 833; 0.094 | match; one decision, tie-breaking |
| same, power.md (600 instances) | 1,618 decisions; 32 %; > 0.10 on 25 %; > 0.20 on 19 %; mean 0.090; 0.27 on the informative | 1,624; 32.3 %; 25.3 %; **17.5 %**; 0.0896; 0.273 | match except the > 0.20 share (tie-breaking, finding 7) |
| ball size | 1,497 | 1,497 | match |
| mean TV(q, q_train) at N = 8 / 16 / 32 / 64 / 128 | 0.160 / 0.167 / 0.148 / 0.042 / 0.041 | 0.159 / 0.168 / 0.148 / 0.042 / 0.041 | match |
| share with TV > 0.05, same N | 41 / 58 / 60 / 32 / 35 % | 41 / 59 / 60 / 32 / 35 % | match |
| β of exact q, of m̃ (by construction) | 1, 0 | 1.000, 0.000 | match |
| §4 table: I_cov 46–47 %, mean TV 0.10–0.14 at 8–32; 22 %, 0.06 at 128 | from `prelim_info.py` | not recomputed | inconsistent with §6 and power.md for the same quantity (finding 5) |
| solve-rate gap ≤ 0.04; H4, H6, H10, H12 rows; the stand-in β and vote tables | | not recomputed | |

Same seed and the same order of random draws as the author's scripts, so the instances are identical; distances by a different algorithm (a numpy world × state reachability recursion), cross-checked against a direct per-world breadth-first search on the rules: 0 mismatches in 240 (problem, world) pairs.

---

## Lessons check (LEDGER L27–L48)

| lesson | status | why |
|---|---|---|
| L27 high-information core | **violated** for P1 | The token arm's fixed point is q; power.md says a model at rung 4's error reads 0.97–0.98; a lookup reads 0.53. Q2 is uncertain, but see L36, L39. |
| L28 fixed-target arm's pooled fixed point | **violated** | The mask arm's fixed point was assumed to be m̃ and never computed; it is at β 0.13–0.46 (finding 1). |
| L29 new estimators on CPU first | partly honoured | β and γ were run on stand-ins; only on unstructured noise along the line. |
| L30 pilot surprise is not a reading | **violated** | The carrier switch and on-policy data hang on a three-seed pilot's reading of γ (finding 10). |
| L31 budget covers the slowest arm | unaddressed | No step budget for the latent arm; no matched-steps or matched-loss reading for Q2 (finding 11). |
| L32 the dose must not change the ruler | **violated** | γ's effective decision set changes with N (finding 3 iv). |
| L33 analysis code on real fixtures; classify from curves | unaddressed | Stage-0 item; no curve reading is planned for γ. |
| L34 input side of an extrapolation | honoured for single verdict tokens (note 20) | Unseen verdict *combinations* are the out-of-support inputs; uncounted in the design. |
| L35, L37, L38 bundle and job status | not applicable yet | No platform named. |
| L36 compute the cells; no "not readable" fallback | partly honoured | Cells were computed; P2's expectation still sits on the point the design calls weak, and the fallback carrier waits on the pilot. |
| L39 score the claim's negation; information sets; statistic, count, in-between name; verdict in code | **violated** in (i), (ii), (iii) | No negation was scored (rule mask passes P1; m̃ reads "generalising" on γ; a graceful memoriser sits at 0.41–0.46); information sets of m̃ and the mask arm not written; P2 has no rule. (iv) is planned. |
| L40 save every model | unaddressed | Not in the design. |
| L41 count, do not infer; every "never" and "always" gets a count | **violated** | "its β must come out near 0" (inferred from the arm's shape, false); "a rejection means the rule is not in the world" (true only if non-matching proposals cannot be made); the fallback's frequency was never counted. |
| L42 score off the instances where the reference equals the target; noisy planted pass | honoured for β | For γ the references are equal *by convention* on 27 % of instances at N = 8 and that count is not in the design. |
| L43 encoding at the edge of the range | honoured | No size extrapolation. |
| L44 cost from the last interval, 2× margin | partly honoured | Named at the pilot gate; the latent arm and the refresh are not budgeted (finding 14). |
| L45 counted overlap at every level the model could recall | partly honoured | Pairs, starts, goals, worlds are counted; rows along trajectories and effective world overlap are not (finding 6, note 19). |
| L46 a curve statistic needs its null | partly honoured | Cited; "where it crosses 0.5" over four points has no interval and no null. |
| L47 sentences about code written with the code open | **violated** twice | m̃'s definition (finding 9); q_train's fallback absent from the text (finding 3). |
| L48 stand-ins through the rule before it is written | partly honoured | Done along the line between the references; not for structured negations; P2 has no rule to run them through. |

---

## The inferential chain

**P1 passes (β ≥ 0.5).** Not ruled out: a rule-identity lookup (0.53–0.67); inference from the state with the verdict tokens unused (0.13–0.46, adding to the former); in-distribution supervised fitting of a target whose mean is q. A pass does not separate these from posterior updating.

**P1 fails (β ≤ 0.25).** Not ruled out: the task was not learned (shortest paths under a hidden world at this size); mass on invalid actions; for the latent arm, memory or training instability (finding 13). And before any of that the mask-arm check fires and nothing is read.

**P1 "partial".** An action mask plus conditioning on accepted rules reads 0.29; a rule mask reads 0.53 ± 0.05 at 2,000 instances. The band's edges fall on models that differ in a lookup.

**P2, γ low at N = 8.** Not ruled out: out-of-support inputs handled by a mask-like default (41 % of the denominator); 27 % of instances silent; one draw of eight small worlds.

**P2, γ high.** Not ruled out: the full prior with no updating (0.71–0.99); a memoriser that falls back sensibly (0.41–0.46 without generalising at all); a product prior with estimated marginals (0.83–0.95); an early checkpoint that has not yet memorised.

What would make the experiment informative is in findings 2 (c) and 3: the scored set is the evidence the training worlds cannot explain, the scored statistic is one that is zero for lookups and for history-blind inference, and the references on that set are several named policies, not one convention.

---

## What I could not check

- Anything about trained models: learnability, seed spread, whether a pooled mask arm reaches its per-problem optimum.
- The mask arm's optimum pooled across problems and under the real proposer mix (mine is per problem, two proposers).
- The H4, H6, H10, H12 rows, the stand-in tables, the solve-rate gap and the k ∈ {4, 5} rows.
- The share of evaluation rows that would repeat training rows (finding 6); it needs the dataset build.
- `experiments/rewrite/RESULTS.md` and `docs/literature.md` (10–12 Nov) were read in their summaries and tables, not line by line; history Entries 83–90, the lessons table and review 6's findings were read in full.
- Novelty: five searches (belief state / exact posterior / prover / verifier feedback; LLM belief updates and Bayes; meta-trained agents and Bayes-optimal agents; Wordle- or Mastermind-style hidden-code tasks against exact posteriors; in-context RL and posterior sampling), eight pages opened. Not a review. Also seen and not relevant enough to list above: arXiv 2603.17063, 2606.30440 (theory only), 2507.17951 (LLM coherence with Bayes' rule, no ground-truth posterior), 2607.17060 (recovering a model's implicit prior; no exact reference).

## Scripts (`review1_scripts/`; run with `PYTHONPATH=src` from the repo root, or from the directory itself with `PYTHONPATH=<repo>/src`)

| file | what it does |
|---|---|
| `core.py` | Instance generator (same random-draw order as the author's), ball, distances for all 255 worlds, mixtures; `brute_dw` cross-check |
| `reftrace.py` | Reference trajectories, reference distributions (q, m̃, rule mask, accepts-only, rejections-only), β |
| `t0_check.py` | Distances against a direct per-world search |
| `t1_headline.py` | Headline numbers of §4 |
| `t2_beta.py` | β under structured model error; lookup references; where the signal is; the step statistic |
| `t3_sim.py` | Brute-force simulation of the generative process under six proposers |
| `t4_coverage.py` | Q2: separations, fallback counts, γ of candidate learners, five world-set draws, P1's setting at N = 64 |
| `t5_maskarm.py` | The mask arm's Bayes optimum and its β |
| `t6_misc.py` | Bootstrap intervals, budget dependence, dead-world decisions, training-set contents, cost (needs `dec_0_600.pkl` written by `t2_beta.py 600 0`) |
| `t7_probes.py` | Power of the counterfactual single-rejection probes |
| `outputs/` | The outputs of the runs quoted above |
