# Review 2 — the belief experiment's design (DESIGN.md revision 3, 13 Nov 2026)

Reviewer: independent; did not write the design and did not write review 1. Nothing was trained. Every number below is from my own code (`review2_scripts/`), which imports the carrier library and review 1's instance generator (`review1_scripts/core.py`) and none of the author's analysis code. I read `refs3.py` for bugs. Carrier: H8, ρ = 0.5, Δ = 2, k ∈ {2, 3, 4}, size cap 5. "Five draws" means the author's draws 7–11 on the author's instance stream (150 held-out-world instances each), so that numbers compare one to one; draw 21 is a sixth draw of my own, sampled as §3 says.

## Verdict: not sound as written

The parts of revision 3 that answer review 1 hold. The four mechanisms are computed correctly, every planted number in refs3.md reproduces (one exception, S5), the change statistic is exactly zero for every deleting model, the gate cannot be passed by a model that does not revise, and the claim that M is the minimiser of the training loss is confirmed by brute-force simulation of the generator as specified. What does not hold is the reading of the classes and four places where the rules, as written, decide an outcome for a reason other than the question:

1. **The G and O classes do not separate "a posterior" from "a planner with no belief".** The exact posterior under a prior of 0.9 per rule is classed O on both readings in 20 of 20 cells; a planner that holds no weights over worlds is classed G out of support in 18 of 20; and on levels the class G is given to the uniform distribution and to a soft planner in 20 of 20. On levels the G–O axis is mostly output entropy.
2. **At 4 training worlds, 9–60 % of the out-of-support probes contain a verdict that never occurs in training** (all five draws), so one end of P1 is partly decided by untrained inputs (L34).
3. **The decision rules are incomplete where it matters**: "not D" and "not M" are not defined for a mixed verdict, and two models whose mechanism does not change with N (a nearest-training-world lookup; a training prior that is never revised) move from D or M at 4 training worlds to "mixed" at 32.
4. **The 32-world cell fails the design's own stage-0 gate on two of the five draws**, by the author's numbers and by mine; the design's answer is a "declared unscored" fallback, which makes P1's upper end unreadable (L36).

All four can be repaired on the same carrier and grid, with no new arm. Items 1 and 3 change what the class code and its planted tests compute, so they should be settled, and the revised §5 and §8 looked at once more, before stage 0 is written.

---

## Findings

### Blocking

**B1. Classes G and O are read as "posterior" and "planner without belief"; the statistics do not license that, and on levels they mostly measure how spread the answer is.**

Passages: §4, *"O | optimistic planner | no belief"*, *"A model could be right about every new world this way without holding a belief, and the two must be told apart for the thesis to have been tested"*; §8, *"G | G | the generalising posterior"*, *"O | O | a planner without belief"*; §10, *"A mechanism with no belief reads as the posterior | … | O is a named direction in every regression"*.

What I computed (`t1_heldout.py`, `t1_summary.py`; exact mechanisms, classes by §8 from a bootstrap over held-out worlds; 20 cells = 5 draws × N ∈ {4, 8, 16, 32}; gate U from `t4_gate.py`, three draws):

| mechanism | weights over worlds? | levels (c_G, c_O) | levels class | out of support (b_G, b_O) | class | gate U |
|---|---|---|---|---|---|---|
| uniform over valid actions of rules not rejected | none; ignores the goal | (1.20–1.75, −0.61–−0.38) | **G 20/20** | 0 | D | 0 |
| soft planner, exp(−excess length / τ), τ = 1 | none | (0.90–1.34, −0.24–−0.04) | **G 20/20** | (0.31–0.56, 0.12–0.33) | mixed 20/20 | 0.45–0.73 |
| version space: uniform over actions optimal in at least one fitting world | a set, no weights | as uniform (identity, below) | **G 20/20** | (0.91–1.40, −0.52–−0.18) | **G 19/20** | 0.72–0.96 |
| leave-one-out planner: average of O and of O with one more untested rule removed | none (≤ 9 fixed what-if worlds) | (0.39–0.47, 0.56–0.63) | O 20/20 | (0.52–0.79, 0.20–0.47) | **G 18/20** | 0.98–1.05 |
| planner in the "majority world" of the training set | none | (−0.15–1.08, −0.41–0.75) | G 3, O 4, mixed 13 | (−0.28–1.21, 0.13–1.01) | G 4, O 9, mixed 7 | — |
| uniform over shortest *paths* in the optimistic world (rung 4's target) | none | (−0.01–0.07, 0.94–0.98) | O 20/20 | (0.04–0.48, 0.70–1.06) | O 19/20 | — |
| exact posterior, prior 0.9 per rule | **yes** | (0.25–0.32, 0.70–0.74) | **O 20/20** | (0.31–0.37, 0.62–0.68) | **O 20/20** | ≈ 1.0 |
| exact posterior, prior 0.8 | yes | (0.47–0.56, 0.46–0.52) | mixed 17/20 | (0.53–0.60, 0.39–0.44) | G 18/20 | — |
| exact posterior, prior 0.2 | yes | (1.13–1.40, −0.39–−0.20) | G | (1.17–1.32, −0.28–−0.13) | G | — |
| G itself, over-confident (G³ renormalised) | yes | (0.42–0.81, 0.29–0.46) | **mixed 16/20** | (0.69–1.29, −0.14–0.40) | G 17/20 | 0.98–1.20 † |
| G itself, under-confident (G^0.5) | yes | (1.10–1.36, −0.29–−0.19) | G | (0.97–1.19, −0.25–−0.09) | G | — |

† U on in-support probes of held-out worlds; the other entries are on training worlds.

Three things follow.

- *O is not "no belief".* O is the ρ → 1 end of the product-prior posterior family: G(ρ) moves smoothly from (1.2, −0.2) at ρ = 0.2 through (1, 0) at 0.5 and (0.5, 0.5) at 0.8 to (0.1, 0.9) at 0.97, on both readings. The two classes separate *how much the answer hedges against untested rules*, which §13 says in one sentence (*"told apart only by hedging"*) and §4, §8 and §10 do not.
- *G is earned without a distribution over worlds.* Out of support, by a what-if planner and by a version space; on levels, by anything diffuse. The identity behind the uniform row: for any distribution uniform over a superset of G's support, its inner products with G, M and O are equal, so its coefficients on (G − M, O − M) are exactly those of the uniform distribution — about (1.5, −0.5) at every N.
- *Calibration moves a posterior between classes.* G-greedy's action is optimal for the optimistic planner at 98.5–99.6 % of scored decisions (`t3_inputs.py`), so G and O differ in the spread of mass, not in the preferred action. Sharpening the exact posterior makes it "mixed"; flattening a planner makes it "G". Out of support, a network's calibration is not constrained by training.

Why blocking: the classes are the experiment's scored output and the G-or-O split is the reading §8 says the author expects to learn most from; as written, "G | G" would be reported as "the generalising posterior" for a version-space learner, and "O | O" as "a planner without belief" for a Bayesian learner with a confident prior. It is also L51 applied once more: the cheap mechanisms above were not planted.

**Change.**
(a) State what each class measures: G = "hedged rerouting, as the uniform-prior posterior"; O = "unhedged rerouting". Drop "no belief" from §4, §8, §10. Report the best-fitting ρ of the family G(ρ) beside the two coefficients; it is one number with a plain meaning.
(b) Add a uniform direction to the levels regression. I tried it on two draws (`t8_levels_uniform.py`): uniform reads (0, 0, 1); a memoriser leaking 20 % to uniform reads (0.00, 0.00, 0.20) instead of (0.3, −0.1) "mixed"; the soft planner reads (0.19–0.40, 0.09–0.22, 0.51–0.65) instead of "G"; G, O, G(ρ) and the noisy stand-ins are unchanged. It does not, and nothing linear will, separate G(0.8–0.9) or the what-if planner from a G–O mixture.
(c) Give the G and O classes an upper bound and a residual condition (c_G ≤ about 1.25; explained share above a floor, with the noise correction of S5).
(d) Put the rows of the table above into the planted tests with the class each *should* get written beside it, before the class code is written (L39, L50, L51).
(e) Rewrite §8's table as "what each pair of readings is compatible with", listing these mechanisms.

**B2. At N = 4 the probes read verdicts that never occur in training (L34).**

Passages: §7, *"The uniform part puts every probe input in the support of training"*; P1, *"At 4 training worlds the out-of-support class is D."*

The uniform proposer makes every *proposal* possible. It cannot make a *verdict* possible that no training world produces: a rule present in all four training worlds is never rejected in training, and one present in none is never accepted. Counts on the out-of-support probes of the held-out sets (`outputs/t1_counts_tokens.txt`):

| draw | rules in every training world / in none (N = 4) | probes whose probed rule is never rejected in training | probes with at least one never-seen verdict in the evidence |
|---|---|---|---|
| 7 | Af / IbL | 17 % | 29 % |
| 8 | Af, IbL / — | 52 % | 60 % |
| 9 | Ab / — | 18 % | 27 % |
| 10 | Ab / — | 23 % | 30 % |
| 11 | — / Ab | 0 % | 9 % |
| 21 (mine) | — / — | 0 % | 0 % |

At N ≥ 8 the count is zero on all six draws. The chance that a random draw of four worlds has such a rule is 0.65 (0.06 for eight). Review 1's note 20 checked this at N = 8, the smallest size then; revision 3 added N = 4 and did not recount. On these probes a model's response to "(r, rejected)" is the response to an input it has no gradient for — an untrained embedding row if a (rule, verdict) pair is one token, an unseen combination if rule and verdict are embedded separately (§6 does not say which). "D at 4" is then partly a statement about initialisation. No evidence list at a probe is longer than training can produce (0 % on every draw and N), so length is not a problem.

**Change.** Draw the training sets so that at every N each rule is present in at least one training world and absent from at least one (rejection-sample the draw; say so), or score the affected probes separately and read P1 off the rest. Specify the verdict encoding in §6; a factored encoding (rule embedding + verdict embedding) is the one under which the remaining probes are clean. Add the table above to the stage-0 overlap tables (L41, L45).

**B3. The class and cell rules leave open the cases that change the verdicts.**

Passages: §8, *"At 32 it is not D (G or O). Falsified at either end by the opposite verdict; a split cell is neither"*; *"At 32 it is not M"*; *"otherwise the run is 'not trained to revise' and is counted as such"*; the class table.

- *"Not D", "not M" and "mixed".* "Mixed" is a per-run class and can be a cell verdict (8 of 10 runs). The text gives "not D (G or O)" for P1 and only "not M" for P2. It matters, because two models whose mechanism is the same at every N change class with N:

  | model | class at N = 4 (5 draws) | at N = 32 |
  |---|---|---|
  | memoriser that falls back to the nearest training worlds (a lookup), out of support | D D D D D | D, mixed × 4 (150 instances); mixed, bounds [−0.41, −0.08], [0.13, 0.37], at 1,000 instances on draw 8 |
  | training prior never revised, levels | M M M M M | M, mixed, M, M, mixed; mixed [0.15, 0.28] at 1,000 instances on draw 8 |
  | memoriser with 20 % of its mass leaked to uniform, levels | mixed × 5 | mixed × 5 |

  If "mixed" counts as "not D" or "not M", both predictions can be confirmed with no change of mechanism. If it does not, say what a mixed cell is for each prediction.
- *Runs that fail the gate.* Are they classed and counted toward the 8 of 10? A model that never learned to revise reads D out of support by construction; counted, it confirms "D at 4".
- *Cells declared unscored at stage 0* (B4): the cell rule assumes ten runs.
- *A run that satisfies both the G and the O clause* (both lower bounds ≥ 0.5) is G by the order of the table; I saw point estimates of (0.83, 0.58) and (0.68, 0.59), neither with both lower bounds above 0.5 at 150 instances, so the case is reachable and has no name. *Negative coefficients* satisfy the D and M clauses: a model that moves against the posterior, (−0.5, −0.5), is "D".
- *The bars sit on planted coordinates.* The half-way updater reads U = 0.50 exactly and passes or fails the gate by floating point; see also S5.

**Change.** Write the outcome table for P1 and P2 over all cell verdicts (D, G, O, mixed, split; M, G, O, mixed, split) and compute it in code with planted cells. Exclude gate-failed runs from class counts and give the cell rule its denominator. Make "both" and "negative" named outcomes. Move the gate bar off 0.5 or state the comparison's tolerance.

**B4. The 32-world out-of-support cell does not meet the stage-0 gate on two of five draws, and the gate does not say for which model.**

Passages: §5, *"Scaled to 4,000 instances per draw: ≤ 0.01, ≤ 0.02 and 0.01–0.05 … the 32 cell is the thin one and stage 0 checks it"*; §9, *"sampling sd of each scored coefficient ≤ 0.04 in every (draw, N) cell at the chosen evaluation size — a cell that cannot meet it is declared unscored"*.

refs3_heldout.md's own last column at N = 32 is 0.236 (draw 8) and 0.254 (draw 11) at 150 instances, which is 0.046 and 0.049 at 4,000. Mine, same instances, largest of four stand-ins at noise TV 0.12 (`t5_power.py A`): 0.053 and 0.048 on those draws, 0.017–0.025 on the other three, 0.063 on draw 21. Measured directly on 1,000 instances of draw 8, not scaled: 0.035–0.086 by stand-in, so 0.018–0.043 at 4,000; on a second sample 0.071–0.109, so 0.035–0.055. The sd depends on which model is planted and at what noise, and §9 names neither. With two draws unscored the cell has six runs and cannot reach eight; P1's upper end, the informative one, is then unreadable by construction. The design computed the cell, found it over the bar, and kept a fallback; L36 calls that a report, not a design.

**Change.** Decide now. Fix the five draws by seed in the design (the power numbers belong to draws 7–11; a sixth is thinner: 0.6 out-of-support probes per instance at N = 32 and 1.4 at N = 16). State the stand-in and noise level the gate is computed for. Then one of: about 8,000 held-out instances for the N = 32 reading; more probes per instance (probe along sampled as well as greedy trajectories); or a bar of 0.055 with the consequence for the 0.25 and 0.5 thresholds written down. See S4 for the cheaper gain in the number of worlds.

### Should-fix

**S1. Q2 is an in-distribution reading; its lower end is derivable and its upper end depends on settings the pilot chooses.**

Passages: §2, *"an in-distribution match to the posterior is expected; here it is a gate, not a finding"*; P2; §9, *"What the pilot may change: optimisation settings, width, step count"*.

An in-support decision is, by definition, an input some training world produces, so levels in support are read on the training distribution (new goals apart). The design's own argument makes M the fixed point there at every N, 32 included (I confirmed it, see "Verified"). So "M at 4" is the derivable cell (L27, L28): it tests fit and goal generalisation. "Not M at 32" is a bet that training stops short of its optimum at a simpler solution — the task-diversity effect — and the published form of that effect is transient: with a finite task pool, models generalise early and move to the memorising predictor with more training, the more slowly the more tasks (Park et al. 2024; Wurgaft et al. 2025; S2). Step count and width are therefore not neutral settings for Q2, the pilot sets both, and all sizes train for the same number of steps. The run cannot read a scored statistic in the pilot, but it can move P2 without reading one.

**Change.** Demote "M at 4" to a check. For P2 at 32, fix the training length by a rule on training loss stated here, read the class at each of the ten checkpoints and pre-register the *sequence* (L32, L33), and say which of "G early, M late" and "G throughout" confirms P2. State the width rule.

**S2. Prior work that pre-empts part of §2's "not found", and the statistic.**

Passage: §2, *"Not found …: a finite pool of hidden worlds with both posteriors exact, evidence the model elicits from a checker, evidence patterns outside every training world, and cheap non-Bayesian mechanisms scored beside the posterior."*

- [Park, Lubana, Pres & Tanaka, *Competition Dynamics Shape Algorithmic Phases of In-Context Learning* (arXiv 2412.01003)](https://arxiv.org/html/2412.01003v4). A finite set of N Markov chains (N from 2² to 2¹¹); four exactly computed algorithms — two posteriors with a discrete prior on the training chains ("retrieval") and two that use only the context ("inference"), each on unigram or bigram statistics; the model's output distribution **fitted as a linear combination of the algorithms' outputs** ("linear interpolation of algorithms"); in-distribution and unseen chains; a diversity threshold near 2⁶ and transient generalisation. This is §5's "regression on the mechanisms" and Q2's sweep, on passive sequences.
- [Wurgaft et al., *In-Context Learning Strategies Emerge Rationally* (arXiv 2506.17859)](https://arxiv.org/html/2506.17859v1). Memorising predictor (discrete prior on the D seen tasks) against generalising predictor (the true prior), both in closed form, in three settings, D from 2² to 2¹²; the model's position between them as a function of D and training steps; transience with a time scale superlinear in D.
- [Wang et al., *Understanding the Training and Generalization of Pretrained Transformer for Sequential Decision Making* (arXiv 2405.14219)](https://arxiv.org/html/2405.14219v2). Supervised pretraining on optimal actions; finite pools of 4, 16 and 100 environments; the transformer's action compared step by step with the Bayes-optimal decision under the pool; and the shift between pretraining histories and the model's own, repaired by mixing in model-generated histories — §7's deferred "own-sample histories".
- [Goddard et al., *When can in-context learning generalize out of task distribution?* (arXiv 2506.05574)](https://arxiv.org/abs/2506.05574): task diversity against out-of-distribution generalisation in regression. Abstract only.

What I did not find: a hard-evidence carrier in which "fits no training world" is a defined event (with soft likelihoods nothing is strictly out of support); counterfactual single-verdict probes at a fixed state; an optimism rival scored beside a posterior. Two searches found the first three papers. **Change.** Cite them; narrow §2's claim to those three items; credit the regression statistic.

**S3. The ruler still changes with the dose (L32).**

Between N = 4 and N = 32: the in-support share of decisions goes from 34–72 % to 89–98 %; out-of-support probes go from mostly "beyond" (draw 7: 7.7 of 9.5 per instance) to mostly "crossing" (2.2 of 3.1), where the two switch directions carry part of the fit; and the evidence gets longer — on draw 7 at N = 16, 73 % of the signal is on probes with one rejection, on draw 8 at N = 32, 96 % is on probes with two, which are 7 % of training rows (`t3_inputs.py`). The M threshold is in units of the levels signal, Σ|G − M|² per instance, which falls by a factor of 1.5–9 across the grid; that is why the never-revised training prior drifts from M to mixed (B3). P1 and P2 each compare two ends measured on different kinds of input. **Change.** Print every scored coefficient by probe type (crossing, beyond) and by number of verdicts, and require the class to hold in the stratum both ends share.

**S4. The quoted precision is within-model noise; the intervals a real model gets depend on a few held-out worlds.**

Passage: §8, *"95 % bootstrap intervals (over held-out worlds)"*; §5's sds.

The stand-in's noise is independent per output, so its bootstrap sd falls as 1/√instances. A real model's deviation varies by world, and that part does not. Effective number of held-out worlds carrying the out-of-support signal (Σe)²/Σe²: 15–23 at N = 4, 12–16 at 16 and 3–12 at 32 with 150 instances; 31, 23 and 12 with 1,000 instances on draw 8, where five worlds carry 56 % of the N = 32 signal (`t1_report.py --part=cond`). With a coefficient that varies across worlds with sd 0.3 around 0.7, on the real per-world Gram matrices, the nominal 95 % interval covers 90–94 % of the time — acceptable — but its half-width is 0.10–0.18, not 0.02–0.05, and the run is classed "mixed" 6–41 % of the time; around 0.1 it is "mixed" instead of D 20–52 % of the time (`t5_power.py C`). **Change.** Hold out every world outside T₃₂ (223 worlds, the same set for every N ≤ 32) instead of 64: at no training cost the effective worlds go from 31 / 23 / 12 to 76 / 52 / 21 (draw 8, 1,000 instances). Quote the between-world component in §5.

**S5. The planted "half G, half O" is not a model, and a real one is classed G; the residual has no noise floor.**

Passage: §5 table, *"half G, half O | (0.5, 0.5) | (0.44–0.50, 0.50–0.54) | mixed"*; §8, *"The class code must return the right-hand column … for every planted model"*; §5, *"a large residual is itself a finding"*.

`refs3.py` plants `0.5 * dq + 0.5 * do`, the average of the two *changes*, which is on the regression plane by construction. A model whose *output* is ½G + ½O has a different change, because RM_r renormalises: it reads (0.51–0.62, 0.44–0.48) out of support and is classed **G in 13 of 20 cells** (lower bound of b_G above 0.5). So §5's sentence *"linear in the model's output"* is true of levels and false of changes; I found no bias from it under independent noise up to TV 0.3 (`t6_misc.py`), so the consequence is confined to mixtures and to the next point. Residual share for exact mechanisms plus noise at TV 0.12: 0.29–0.69 out of support, 0.05–0.67 on levels; at TV 0.3, 0.7–0.9. A reported residual says nothing until the noise part is removed. **Change.** Plant mixtures as outputs; keep class bars off planted coordinates; estimate the residual from the cross-product of two seeds' (or two half-samples') deviations, or report it beside the same quantity for the noisy stand-in.

**S6. Class D means "no revision along G or O", not "nothing beyond deleting".**

Passage: §1, *"nothing beyond deleting that rule"*; §4, *"D | deletion"*.

Exactly zero holds for every deleting model (verified). The converse fails: a memoriser that goes to the uniform distribution when its worlds are refuted changes its answer by 3–16 times the posterior's change energy and is classed D in 7 cells of 20 and mixed in 13; a model that flattens with each rejection, 1.2–2.4 times, D in 16 of 20; one that sharpens, with b_O down to −0.28, D in 17 of 20. **Change.** Give D an energy bound (Σ|dp|² small against Σ|dG|², noise-corrected), and name the remainder "revises, along neither".

**S7. The probe is defined differently in the text, the held-out code and the gate code; RM_r is undefined where it matters most for an M-like model.**

Passage: §5, *"A probe at a decision is a rule r not yet tested, with a valid action there, that some fitting world lacks."*

"Some fitting world" is all 255 worlds in `heldout_part` and the training worlds in `gate_part`; on training worlds the two give 1.7 against 9.0 probes per instance at N = 4 (draw 7) and the same U for M, G and O. The code also requires that some fitting world *has* r (always true) and that G is not wholly on r (never bites: 0 of about 2,000 probes per draw). Undefined cases: O wholly on r at 4–12 % of out-of-support probes — harmless, the posterior's own change there is under 0.2 % of the signal; M wholly on r at 5–16 % of crossing probes — these are the probes at which an M-model's own proposal is refuted, and there RM_r of a network near M is its renormalised softmax tail. **Change.** One definition, in §5, with the pool named; a stated rule for probes where a reference or the model has more than, say, 0.95 on r (drop for all, by a reference-based criterion, and count them).

**S8. The tie rule is not fixed.**

23–28 % of scored decisions have a tie for G's maximum (review 1: 23 %). `refs3.py` breaks them by `max(sorted(...))` on raw floats: 0–3 ties per 150 instances (6–9 per 1,000) are between values equal to within 10⁻⁹ and not bitwise, and are decided by summation order — my counts differ from the author's by a decision or two per draw for this reason. The sort key also contains the target's index in the breadth-first enumeration of the ball. **Change.** Order by (rule, path); tie tolerance 10⁻⁹; both in §8.

**S9. The gate mixes two things, and N = 32 is never piloted.**

The gate is read on training worlds *with held-out goals*, so a failure is "does not revise" or "does not carry over to new goals"; §8 names it only the first. Read it on training goals as well and name the two outcomes. Cheap models at the bar: soft planner 0.45–0.73, version space 0.72–0.96, 70 % M + 30 % uniform 0.68–0.71; all revise, so this is not an error, but "trained to revise" is all a pass shows. The pilot runs N = 16 and N = 4; the cell at most risk for learnability, cost and power is N = 32 (L21, L26). The shift check (*"only if the pilot shows a shift"*) still has no definition or threshold, which review 1's finding 10 asked for.

**S10. Undefined or missing quantities.**

- Training episodes per run. §11 costs a million episodes at three core-hours; a run's consumption is steps × batch ÷ about 3 rows per episode (2.0–4.7 in my simulation) and is not stated; 56 runs of fresh episodes may be the largest item on a two-core machine.
- Verdict encoding (B2); whether the list is ordered and repeats accepted rules.
- The width rule and the step-count rule (S1).
- The seeds of the five draws (B4).

**S11. §8's table lists six of sixteen pairs.** Unlisted pairs that named mechanisms produce: (O, G) — the what-if planner; (G, mixed) — the soft planner; (mixed, G) — the sharpened posterior and G(0.8); (M, mixed) — the nearest-world lookup at 32. And "G | D, a prior that is not revised" is also what the uniform distribution and a posterior revised only on accepted steps read.

### Notes

**N1.** With two seeds per draw, eight runs span at least four draws; the clause *"from at least 4 of the 5 draws"* is implied by "8 of 10".

**N2. The cell rule does not always return "split".** Simulated with one coefficient per run, seed sd 0.10–0.15, draw sd 0.05–0.10, interval half-width 0.04 (`t5_power.py D`): a true value of 0.0 or ≥ 0.8 gets its verdict in 87–100 % of cells; 0.7 in 71–96 %; 0.6 in 25–40 %; 0.2 in 10 %, split otherwise. With seed sd 0.25, D at a true 0 is returned in 62 %. The rule is workable when models sit at a mechanism and returns split in between, as intended.

**N3. Latent arm.** Specified well enough to build. What it can show: at N = 16, on the same draws as the token arm, whether the classes survive when the evidence must be carried. What it cannot: anything about N; and a D or failed gate remains three-way (revision, memory, optimisation) unless the register probe is given a pass bar in advance. At a probe the arm sees the registers it wrote at this decision plus the probe verdict, which is the input of the next round after a rejection; no objection.

**N4. Descriptions of the cited papers.** DPT: Theorem 1, Assumption 1, Definition 6.1 and the support condition are as described; the proof takes the optimal policy deterministic (labels here are uniform over optimal actions — the specialised claim is verified below regardless); in the bandit experiments each pretraining dataset comes from its own task, so the appendix's sensitivity to the number of pretraining samples varies the number of tasks in effect (as far as I could read; Appendix B.3 itself was not returned to me). Agarwal et al.: the abstract says 10⁻³–10⁻⁴ bits; the body gives 3 × 10⁻³ (bijections) and 7.5 × 10⁻⁵ (HMM) for the main models and 0.007 and 0.049 for the small comparison models — §2's "10⁻³–10⁻²" matches neither set exactly. INT: 25, 100, 200 and 300 axiom combinations with a generalisation gap that shrinks; "not with 25" overstates a graded result.

**N5.** `refs3.py` samples held-out instances by drawing the world from the full prior and rejecting; §3 says problem first, then a world uniform over the pool. On draw 8 the two give 69 % against 67 % in support at N = 4 and 8.1 against 7.9 probes per instance. Immaterial; make the code follow §3.

**N6.** "Training prior, never revised" in `refs3.py` is restricted to worlds that can still finish from the current state within the budget, so it is revised by accepted steps and the budget — review 1's finding 9 again under a new name (L47).

**N7.** Proposer (ii) is the optimistic planner O. Compliance holds (it is a function of the inputs), and it keeps the scored trajectories in support: under the mix the per-step proposal probability of G-greedy's action has median 0.31 (10th percentile 0.19, minimum 0.02). Say in §7 that half the histories are O's.

**N8.** Under the proposer mix 59–60 % of training episodes end because the true world can no longer finish and 35–36 % reach the goal; 61–65 % of training rows carry no rejection, 26–28 % one, 7–9 % two.

**N9.** A posterior with marginals estimated from four training worlds reads U = 0.57–0.97 on in-support probes of held-out worlds; a legitimate model can sit near the gate bar at N = 4.

---

## Verified to be correct

- **Mechanisms and regressions.** G reads (1, 0) and O (0, 1) on levels and on out-of-support changes; M (0, 0) on levels and (1, 0, 0) on in-support changes; "M in support, G outside" (0.94–1.02, −0.03–0.03) with s_G = 1; "M in support, O outside" (0.00–0.15, 0.83–1.00); half-way updater (0.47–0.51, ≈ 0); on the five draws, and within 0.08 of these on the sixth.
- **Exactly zero for deletion.** A verdict-blind model, a frozen full or training prior with the rule mask, conditioning on accepted rules only with the rule mask, and the uniform distribution all read 0.00 to machine precision with zero energy. Deleting only the tried action also gives zero (RM_r removes the rule's other actions on both sides).
- **The training claim (§7).** Twelve problems, pools of 8 and 16 training worlds, 200,000 episodes each, the generator as specified (`t2_gen.py`): mean TV between the empirical label distribution and M is 0.0010–0.0055 against a null of 0.0009–0.0055 (labels drawn from M with the same counts); largest |z| 2.1–3.5 over 45–560 cells per problem; TV to G is 0.004–0.25. Continuing episodes after the true world is dead gives 0.010–0.205; keeping only solved episodes gives up to 0.183 with |z| up to 219. Ending the episode when the true world cannot finish is a selection on the world and is absorbed by "fits", as condition (a) says. That the posterior is a function of (state, goal, budget, known-present, known-absent): the solvability condition on (u, s, B) and survival at earlier decisions are implied by "contains the accepted rules and can finish from v within b"; checked by argument, the simulation being per problem.
- **The gate.** M, G and O each read U = 1.00; deletion 0.00; four non-belief revisions (to uniform, half-way to uniform, flatten, sharpen) read −0.46–0.08 and fail. No non-revising model passes.
- **Signal, counts, sampling sds, single-direction planner readings**: table below.
- **Noise.** No detectable bias in the change coefficients under independent output noise at TV 0.12, 0.2, 0.3.
- **Bootstrap coverage** 0.90–0.94 for a nominal 95 % (S4).
- **Scored trajectories are inside the support of the proposer mix** (N7); no probe's evidence list is longer than a training episode can produce.
- **refs3.py.** No bug that changes a reported number, apart from S5 (what is planted) and the mismatches in S7, N5, N6.

---

## Review 1's findings against revision 3

| # | finding | status in revision 3 |
|---|---|---|
| 1 | mask arm's optimum is not the mask | fixed (arm dropped; zero by construction, verified) |
| 2 | lookup passes P1; pass foregone in distribution | fixed for Q1 (lookups read 0; scored out of support). Q2 is still an in-distribution reading (S1) |
| 3 | γ's fallback; non-updating model reads "generalising"; no rule | fallback gone (in-support only). Levels alone still give G to a never-revised prior — handled by reading with changes — and to the uniform distribution (B1). P2 has a rule with gaps (B3). The ruler still changes with N (S3) |
| 4 | unstated conditions | fixed; (a)–(e) verified by simulation |
| 5 | one draw | fixed; the draw clause of the cell rule is redundant (N1); seeds of the draws not fixed (B4) |
| 6 | rows not disjoint | fixed in design (evaluation-only goals); to be counted at the build |
| 7 | ties | partly: an order is named, without tolerance, and it includes a ball index (S8) |
| 8 | dead-world decisions scored | fixed |
| 9 | text and code disagree on a reference | partly: that reference is gone; N6 is the same kind of mismatch, and S5, S7, N5 are new ones |
| 10 | pilot reading decides the carrier | fixed for the carrier; the shift check still has no threshold (S9); step count and width are a new route (S1) |
| 11 | no checkpoint rule; N confounded with steps per world | partly: final step and curves; confound stated, not controlled; transience not addressed (S1) |
| 12 | prior work | the three papers are cited correctly (N4); more exists (S2) |
| 13 | latent arm unspecified | fixed (N3) |
| 14 | compute and data | partly: episodes per run missing (S10) |
| 15 | input and output, seeds, saving | validity mask, seeds and saving fixed; verdict encoding still open (B2) |
| 16 | β robust | not applicable (β is reported for continuity) |
| 17 | which prior at N = 64 | not applicable (no 64) |
| 18 | product-of-marginals learner | classed G as stated; mixed on levels at N = 4 on two of five draws, as the design says |
| 19 | leakage counts | planned in §8 |
| 20 | every single-rule verdict occurs in training at N = 8 | **not carried to the new N = 4**, where it fails on five of five draws (B2) |
| 21 | uniform prior | stated as a limit; B1 shows it matters more than a limit: prior shape *is* the G–O axis |
| 22 | scripts | — |

---

## Recomputed against the author's

Five draws, 150 held-out-world instances each, the author's instance stream. Differences in the second decimal trace to tie-breaking (S8).

| quantity | refs3 / DESIGN | recomputed | |
|---|---|---|---|
| decisions in support, N = 4 / 32 | 33–72 % / 89–98 % | 34–72 % / 89–98 % | match |
| levels signal per instance, N = 4 / 32 / 64 | 0.16–0.40 / 0.05–0.16 / 0.03–0.06 | 0.162–0.393 / 0.045–0.173 / 0.026–0.063 | match |
| out-of-support probes per instance, N = 4 / 32 / 64 | 8–11 / 0.7–3.1 / < 1 | 7.7–10.5 / 0.7–3.1 / 0.2–0.8 | match |
| out-of-support signal per instance, N = 4 / 32 | 0.27–0.34 / 0.02–0.14 | 0.269–0.344 / 0.020–0.144 | match |
| correlation of (G − M, O − M) | 0.24–0.91 | 0.23–0.91 | match |
| G, O on levels and out of support | (1, 0), (0, 1) | (1.00, 0.00), (0.00, 1.00) | match |
| M in support, G outside | (0.87–1.02, −0.03–0.08) | (0.94–1.02, −0.03–0.03) | match |
| M in support, O outside | (0.00–0.15, 0.84–1.00) | (0.00–0.15, 0.83–1.00) | match |
| half G, half O, out of support | (0.44–0.50, 0.50–0.54), "mixed" | average of the two changes (draw 7): (0.49–0.50, 0.50–0.51); **mixture of outputs (five draws): (0.51–0.62, 0.44–0.48), G in 13/20** | **differs** (S5) |
| training prior never revised, levels | (0.00–0.25, −0.02–0.20) | (0.00–0.24, −0.02–0.21) | match |
| full prior never revised, levels | (0.85–0.99, 0.02–0.11) | (0.87–0.98, 0.03–0.11) | match |
| nearest-training-world memoriser, out of support | (−0.22–0.11, −0.05–0.25) | (−0.19–0.13, −0.06–0.25) | match |
| product-of-marginals posterior, levels / out of support | (0.52–1.06, −0.24–0.13) / (0.91–1.31, −0.21–0.10) | (0.52–1.05, −0.24–0.13) / (0.93–1.32, −0.22–0.09) | match |
| half-way updater, out of support | (0.44–0.51, −0.01–0.04) | (0.47–0.51, −0.01–0.02) | match |
| in-support changes: M, G, O, half-way | unit vectors; (0.5, 0, 0) | same | match |
| gate U: M, G, O / half-way / deletion | 1.00 / 0.50 / 0 | 1.00 / 0.50 / 0.00 | match |
| planner under the one-direction statistics: γ; β_cf | 0.9–1.3 on two draws, 0.1–0.8 on the third; 1.0–1.4 | 0.88–1.18 (draws 7, 8), 0.14–0.71 (draw 9); 1.04–1.42 | match |
| verdict-blind model, RM_r on the earlier output only | 0.09 | 0.115 (draw 7 held-out sample) | same order; different sample |
| sampling sd at 150 instances, levels | ≤ 0.05 | ≤ 0.057 | match |
| same, out of support, N ≤ 16 / N = 32 | ≤ 0.11 / 0.07–0.25 | ≤ 0.106 / 0.06–0.25 (half-way stand-in); up to 0.26 for the other stand-ins | match |
| scaled to 4,000, N = 32 | 0.01–0.05 | 0.017–0.053; 0.063 on a sixth draw; 0.018–0.055 measured from 1,000 | match; over the 0.04 gate on two draws (B4) |
| noisy stand-ins at 0 / 0.5 / 1, out of support | −0.13–0.23 / 0.36–0.62 / 0.88–1.33 | −0.18–0.12 / 0.25–0.83 / 0.93–1.26 (non-reviser / half-way / M-then-G, my noise draws) | match within sampling |
| ties on the scored trajectory | (review 1: 23 %) | 23–28 % | match |

Not recomputed: the fitted 8 × 8 rule table; the H10 and H12 remark in §3; refs3_gate.md's noisy stand-in columns.

---

## Lessons check (LEDGER L27–L51)

| lesson | status | why |
|---|---|---|
| L27 core from cells that cannot be derived | partly | P1 and P2 at 32 are open. P2 at 4 is the training optimum on training-distribution inputs (S1); P1 at 4 is partly decided by untrained inputs (B2) |
| L28 fixed-target arm's fixed point | honoured, with a consequence unsaid | The fixed point is M in support at every N (verified); P2 at 32 is therefore a prediction of non-convergence (S1) |
| L29 new estimators on CPU; degenerate states | partly | Run on stand-ins. The degenerate state — RM_r when the mass is on r — is in refs3.md reading 8 for O only; it is commoner for M (S7) |
| L30 pilot is not a reading | honoured in text | Step count and width still move Q2 (S1) |
| L31 budget covers the slowest arm | partly | Equal steps at all N, noted; N = 32 not piloted (S9) |
| L32 the dose must not change the ruler | **not met** | In-support share, probe type, evidence length and the M threshold all move with N (S3, B3) |
| L33 analysis code on real fixtures; classify from curves | planned | Final-step class with curves reported; for P2 the sequence is the object (S1) |
| L34 input side of an extrapolation | **not met at N = 4** | B2 |
| L35, L37, L38 | not applicable | no platform |
| L36 compute the cell; no "not readable" fallback | **not met for N = 32** | Computed, over the bar on two draws, fallback kept (B4) |
| L39 score the negation; information sets; in-between names; verdict in code | partly | (i) negations scored on the author's list only (B1); (ii) §4 gives the mechanisms' inputs; (iii) mixed and split exist, their use by P1 and P2 does not (B3); (iv) planned |
| L40 save every model | honoured | §7 |
| L41 count, do not infer | partly | "exactly zero" was counted. "Every probe input in the support of training" and "linear in the model's output" were not (B2, S5) |
| L42 reference equals target; noisy planted pass | honoured | noisy stand-ins; undefined-direction probes carry under 0.2 % of the signal |
| L43 encoding at the edge of the range | honoured | no size extrapolation |
| L44 cost from the last interval, 2× margin | partly | in §9 and §11; data per run missing (S10) |
| L45 counted overlap | planned | add the verdict table of B2 |
| L46 a curve statistic needs its null | partly | cited for the pilot; none stated for the ten-checkpoint class curves |
| L47 sentences about code with the code open | **not met** in four places | S5, S7, N5, N6 |
| L48 stand-ins through the rule | partly | along M → G and several off it; not at the bars, where planted models sit (B3, S5) |
| L49 a control arm is a model with inputs | not applicable | no control arm |
| L50 cheapest mechanisms that could pass | partly | O was found; B1 lists five more that get a class they should not, and two posteriors that do |
| L51 a cheap mechanism that reads high becomes a direction | partly | done for O; the uniform direction on levels is the same step (B1 b) |

---

## What I could not check

- Anything about trained models: learnability at 4–32 worlds, seed spread, calibration out of support, whether goal hold-out is passed.
- The 4,000-instance cells directly. Five-draw numbers are at 150 instances; the largest direct run is 1,000 instances on draw 8, twice.
- The gate with held-out goals, the overlap tables, the 10 ms episode target, the cost per run: nothing is built.
- Structured model error; my noise is independent and multiplicative, as the author's.
- power.md and prelim_info.md were not re-read; the rule-table part of refs3.py was read, not rerun.
- Literature, all through a summarising page reader, so I saw answers to my questions, not the pages. DPT: main text and appendices to the end, except Appendix B.3, which fell on a boundary and was not returned by either call. Agarwal et al.: 200,000 of 210,670 characters. Mikulik et al.: abstract only this time. Park et al.: the first 100,000 of 339,798 characters (main text; appendices unread). Wurgaft et al.: complete in two calls. Wang et al.: complete in two calls, with the finite-pool appendix read by a second query. Goddard et al., Kirsch et al. 2022, Raparthy et al. 2023: abstracts only. INT: the first 100,000 of 262,653 characters. Six searches; not a review.

## Scripts (`review2_scripts/`; from the repository root, `PYTHONPATH=src python3 experiments/belief/review/review2_scripts/<file> …`)

| file | what it does | as run |
|---|---|---|
| `r2lib.py` | Mechanisms at a decision, probes, RM_r, greedy walk with the tie rule, the per-world regression with bootstrap, §8's class rule | — |
| `t1_heldout.py` | Held-out worlds: levels, out-of-support and in-support changes for the design's planted models and about thirty others; counts for B2, S7, S8. Writes a pickle of per-world sums | `7 150 author` … `11 150 author`; `21 150 design`; `8 1000 design`; `8 1000 wide` |
| `t1_report.py` | Per-file tables: coefficients, bounds, classes, residuals, conditioning, counts | `outputs/heldout_*.pkl --part=all\|cond\|counts\|oos\|levels\|ins` |
| `t1_summary.py` | Across draws: coefficient ranges and the list of classes per model and N | `"outputs/heldout_author_d*_n150.pkl"` |
| `t2_gen.py` | Brute-force simulation of the generator against M, with two violations | `7 8 200000 6`; `9 16 200000 6` |
| `t3_inputs.py` | Training rows against scored probe inputs; G-greedy under the proposer mix | `7 16 150`; `8 32 150` |
| `t4_gate.py` | The gate on training worlds, two probe definitions, cheap and miscalibrated models | `7 120`; `8 120`; `9 120` |
| `t5_power.py` | A: sampling sds from the pickles; C: bootstrap coverage with between-world spread; D: the cell rule | `A "outputs/heldout_*_n150.pkl"`; `C <pickles>`; `D` |
| `t6_misc.py` | RM_r amplification, the one-sided-deletion reading, noise bias, mixture of outputs against mixture of changes | `7 150` |
| `t7_single_direction.py` | The planner under the one-direction statistics; readable-range numbers | `"outputs/heldout_author_d*_n150.pkl"` |
| `t8_levels_uniform.py` | Levels with a uniform direction added | pickles of draws 7 and 9 |
| `outputs/` | Text outputs of the runs above. The pickles (26 MB) are not copied into the repository; each is regenerated by its `t1_heldout.py` line in 2–4 minutes | |

A run of `t1_heldout.py` at 150 instances takes about 100 s and under 1 GB; nothing here needs more than two processes.
