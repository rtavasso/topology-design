# Rung 4 (the rewrite carrier): results

**Provenance.** Design: [DESIGN.md](DESIGN.md) revision 3 with §16 (P4's cells, amended at the derive step), §17 and §17a (the other pass rules, amended after the pass-rule audit and review 5; all before any main-grid run). Grid: [grids/main.json](grids/main.json), 41 runs of 12,000 steps (RT-canon 10, RT-post 10, RT-all 3, RT-absolute 5, EI16 5, PF 3, RT-examples 3, RT-shuffled 2; lr 3e-4 for RT-post, 1e-3 for the rest, from the pilot — [PILOT.md](PILOT.md)); 612,359 parameters (RT-absolute 612,841). Two Kaggle sessions on two T4s, four shards: bundle `20ebab0` (9 Nov, 29 runs inside the 10.5 h session budget) and bundle `32c9f53` (10 Nov, resumed with the 29, ran the other 12: RT-all s0, RT-canon s1–s9, RT-shuffled s0–s1); the bundles differ only in `launch.py`. Gate on the box in session 2: 38 passed, 1 skipped; `results=41 of 41`, `rcs=[0, 0, 0, 0]`. 44.8 process-hours against 27 projected (EI16 took 272–281 min a run against 148 projected). The first session's kernel log and shard logs were overwritten on the researcher's machine by the second session's fetch, so its gate line cannot be re-read; its 29 result files and checkpoints are intact. Records: `results/kaggle_rewrite-main/results/` (outside git; checksums in [results_data/main/RAW_SHA256SUMS](results_data/main/RAW_SHA256SUMS)), compact copies without the per-instance lists in [results_data/main/](results_data/main/), 41 saved routers on the researcher's machine. Analysis: [results_data/main_report.md](results_data/main_report.md) from `analyze.py` (the scored verdicts are its last section; P4's verdict is read from the counts in its P4 table — `analyze.verdicts` does not compute it, contrary to §17's "every verdict"); unscored companions written after the verdicts were read: [main_comb_by_arm.md](results_data/main_comb_by_arm.md), [main_extra.md](results_data/main_extra.md), [p6_check.txt](results_data/p6_check.txt).

**Independently reviewed** ([review/review6_results.md](review/review6_results.md), its nine scripts beside it, none importing `analyze.py`): every number behind the verdicts and the companions reproduced from the raw records; every scored verdict is what the written rule yields; one statement of mine about the data was false (the eval sets overlap the training stream) and eight readings were overclaimed, under-reported or wrong about the code. This file is the version revised after that review; §8 lists what changed, and [results_data/review6_checks.txt](results_data/review6_checks.txt) is my own recount of the review's main new numbers. No scored verdict changed.

Words used below. *Masked*: the argmax is taken over the valid actions and HALT (the executor supplies the action set). *Unmasked*: over all 105 slots; an invalid action ends the episode. *Binding*: the missing axiom lengthens the shortest proof (d_full < d_W). *Needs the combination*: a held-out binding instance with no proof within the horizon inside a training world contained in the held-out one (pairs n = 209, triples n = 509; DESIGN §17). Rates are seed-medians with [min, max] unless a count is given. "Post hoc" and "unscored" mark everything computed after the verdicts were read.

## 1. Summary

| # | question | verdict under §17 | the numbers | did the rule test its claim? |
|---|---|---|---|---|
| P2 (core) | recombination across held-out axiom sets | **not met, not falsified** — and the "not falsified" rests on two instances | pairs' needs-the-combination stratum: RT-canon 0.47 [0.28, 0.67] masked against a non-composing floor of 0.53, lookahead-3 0.73, RT-all 0.94; pair conditions in 0/10 seeds; above the floor in 4/10 (falsifier: ≤ 3/10) | yes, at low power (209 instances, two worlds) |
| P3 (core) | size extrapolation, structural against absolute positions | **falsified, both clauses** | structural: matched ratio 0.33–0.58 against 0.9 × ρ ≈ 0.92 (0/10); absolute: 0.00–0.01 even where a shortest proof lies on labelled slots (0/5) | **no** — clause 1 mixes three things; clause 2 could not have held (§3) |
| P4 (core) | how much lookahead the prior absorbs | **falsified** toward "three steps of search's work" | on the §16 cells (n = 170) RT-canon 0.94 [0.88, 0.96] against the lookahead-3 rung's 0.72, ≥ it in 10/10; on the 150 pairs never trained on, 0.93 against 0.73, 10/10 | yes, against a proxy-guided ladder that stops at d = 3 |
| P5 | search-found routes against labels | **holds**, with little power | held-out binding stratum (n = 1,673): EI16 0.68 [0.42, 0.77], RT-canon 0.63 | medians only; three of five EI16 seeds within 0.1 |
| P6 | the tie mixture | **does not hold** | RT-post's distance to its target 0.12–0.14 against 0.10 for the uniform reference (0/10); RT-canon canonical-first ≥ 0.9 in 9/10 | the rule had a defect; RT-post misses the auditor's alternative bar too (§5) |
| P7 | examples in place of stated axioms | **holds** | training binding 0.89 / 0.90 / 0.92 against 0.89 ± 0.1; held-out 0.44 / 0.38 / 0.45 against RT-canon's 0.55 | clause 1 can hardly fail (§17a) |
| P8 (null) | shuffled presence tokens | **masked: falsified; unmasked: holds** | training binding stratum: masked 0.94 against 0.91 / 0.90; unmasked 0.93 against 0.58 / 0.58 | yes; two shuffled seeds |

1. **Recombination under a stated cue was not shown, and it was not cleanly refuted either — by the thinnest margin.** On the held-out pair instances that cannot be proved inside any training world they contain, RT-canon (lr 1e-3) *scores* like a policy that does not compose (0.47 against 0.53; it does not *behave* like that policy — it solves 0.48 of the instances the reference solves and 0.45 of those it does not), while the same router trained on all fifteen worlds reaches 0.94. The fourth seed above the floor is above it by 2 instances of 209; resampling the instances returns the falsifier in half the draws (review 6). "4/10 above the floor" is not evidence that the router recombines. On the triples it is above its floor in 10/10 seeds (0.38 against 0.21) and, in 8/10, below the glued reference (0.50), which composes nothing beyond single-world plans joined by greedy moves. Unmasked: pairs 0.37, above the floor in 1/10.
2. **Reading the true cue at a pattern never seen in training costs success relative to ignoring it** (post hoc; two shuffled seeds). Masked, the token-blind RT-shuffled is ahead of RT-canon over all held-out instances (0.78 / 0.80 against 0.68 [0.63, 0.79]) and above every RT-canon seed in the full world (0.76 / 0.78 against 0.60), on the triples' non-binding instances and on the triples' binding instances that need no combination. Unmasked the two are level over all held-out instances (0.59 / 0.61 against 0.61) and in the full world, where no action is out of world, RT-shuffled leads by about 0.17 in 10/10 seeds. So part of RT-canon's held-out shortfall is the unseen token pattern as an input, not only a failure to compose decisions; "the tokens carry the world when the router must name valid actions itself" holds on training worlds (0.93 against 0.58) and on the needs-the-combination strata (0.37 against 0.15), not on held-out worlds as a whole.
3. **What looked like generalization to held-out worlds is mostly instances that need no unseen combination, and a good part of it is recall.** On the held-out binding instances provable inside a contained training world RT-canon scores 0.93 (pairs, n = 399) and 0.75 (triples, n = 556) against 0.47 and 0.38 on those that need the combination. About 30 % of held-out eval instances are verbatim (state, goal) rows RT-canon was trained on inside a contained training world, with only the presence tokens different; it scores 0.89–0.97 on those and 0.45–0.71 on the rest (review 6). §6's stratum as first preregistered (pairs' binding, n = 608) reads 0.77: its RT-canon conditions would have been met (7/10 seeds, triples 10/10), with its RT-all clause on the line (0.97; 0.90 [0.89, 0.92]).
4. **The eval sets overlap the training stream; I had written that they do not.** The dataset build never excluded training instances from the eval sets: 1,716 of the 4,800 training-world eval5 instances (36 %) are verbatim training (t, s) pairs (start terms are drawn uniformly over sizes 1–5, so small terms repeat). P4 holds on the pairs never trained on (above), P8's masked gaps are 0.040 / 0.041 there, P1's k = 7–8 instances have no overlap. The pooled training-world rates at k = 5–6 are raised by recall (k = 6: 0.73 [0.62, 0.83] on the 191 new pairs, 0.99 on the 88 repeated ones), and the training-world strata of P6–P8 are a quarter repeated instances.
5. **Inside the trained term sizes the no-search router beats the proxy-guided lookahead-3 policy by a wide margin**, on pairs it never trained on, and still solves half the instances at proof length 7–8 where that policy solves 0.02 (P1: 0.53 [0.32, 0.62]). Its success is unrelated to the ladder's (0.93 where lookahead-3 succeeds, 0.93 where it fails): it was trained on exact shortest-path labels, the ladder is limited by a tree-edit proxy, and the ladder stops at d = 3 — so this is a lower bound against one family of searchers, not a count of "steps of lookahead".
6. **Sizes 6–7: falsified as scored, but neither clause tested its claim, and the records locate the loss.** Three things are mixed in the eval7 set: (i) the structural features are clamped at depth 4 and subtree size 5 (my fix for L34), so nodes past depth 4 share features; (ii) training runs at size cap 5 and eval7 at cap 7, so a growth step at the largest size was never a label, and 52 % of one-step eval7 instances need one; (iii) for the absolute control, every eval7 term is longer than any it saw, so clause 2 could not have held whatever it learned — the rule was mis-specified. Post hoc, at k = 1: with no node past depth 4 and no growth (n = 199) RT-canon is at 0.95 [0.92, 0.97] in every seed on terms of 11–13 tokens it never saw; with growth 0.79; with a node past depth 4, 0.77; with both, 0.35 and a wrong halt in 0.56 of episodes (the router reads t as equal to s). That is evidence against "learned a table" for one-step rules, and it is not a clean pass: at k = 2–4 the in-range, non-growing instances are still below the greedy rung (0.86 / 0.76 / 0.66 against 0.97 / 0.86 / 0.72). No episode was replayed through a router with the clamp lifted, so the mechanism is inferred.
7. **Search-found routes did as well as labels at the median (P5); examples in place of stated axioms did worse on held-out worlds (P7).** EI16's held-out curves are the least stable in the grid.
8. **P6 does not hold, and the defect in its rule does not rescue it.** The uniform reference *is* the target on 1,003 of the 2,213 instances, so no learned policy can beat it there (I found this only after the verdict). On the other 1,210 RT-post is closer to its target than uniform in 10/10 seeds (0.150–0.170 against 0.186) — but so is PF, which was never trained on that target (0.140–0.150), and RT-post also misses the auditor's alternative bar of 0.05. RT-post's first decisions lean the right way; they are not at the derived mixture.
9. **Unscored, worth a matched test and nothing more:** RT-post is above RT-canon on every held-out stratum (needs-the-combination pairs 0.64 against 0.47 masked, 0.59 against 0.37 unmasked; P2's pair conditions in 4/10 seeds, above the pairs' floor in 8/10 masked and 7/10 unmasked; k = 7–8 0.72 against 0.53) and below it at sizes 6–7 (0.34 against 0.52). The arms differ in learning rate (3e-4 against 1e-3, chosen in the pilot on 0.919 against 0.916 at one seed) and in data (RT-post trains on 1.1–2.5 times as many distinct rows per world).
10. **The held-out rates are not converged.** In 1,000-step window means RT-canon's held-out success was still rising at the end (0.683, 0.692, 0.696), and within one run the held-out pairs' binding rate moves with sd 0.05 across the last nine checkpoints. The step-12,000 rates are a point on a slowly rising, noisy curve.

## 2. P2 — recombination (Q1)

Scored rule (§17): on the pairs' needs-the-combination stratum, RT-canon ≥ greedy + 0.2 and ≥ 0.7 in ≥ 6/10 seeds, and above the triples' floor in ≥ 6/10; falsified if above the pairs' floor in ≤ 3/10 while RT-all's median meets the pair conditions.

| stratum | n | greedy | lookahead-3 | floor (training-world reference) | glued reference | RT-canon masked | unmasked | RT-all masked |
|---|---|---|---|---|---|---|---|---|
| pairs, needs the combination | 209 | 0.40 | 0.73 | 0.53 | 0.75 | 0.47 [0.28, 0.67] | 0.37 [0.16, 0.56] | 0.94 [0.93, 0.95] |
| triples, needs the combination | 509 | 0.15 | 0.51 | 0.21 | 0.50 | 0.38 [0.27, 0.56] | 0.31 [0.17, 0.51] | 0.85 [0.84, 0.89] |
| pairs, beyond the glued reference | 52 | – | 0.56 | – | 0 | 0.45 [0.25, 0.60] | 0.37 [0.12, 0.48] | – |
| triples, beyond the glued reference | 256 | – | 0.36 | – | 0 | 0.30 [0.22, 0.46] | 0.24 [0.11, 0.43] | – |

Counts over RT-canon's ten seeds, masked: pair conditions 0/10; above the pairs' floor 4/10; above the triples' floor 10/10. Unmasked: 0/10, 1/10, 9/10. RT-all's median meets the pair conditions (0.94), so the falsifier's precondition held; RT-all's triples median is 0.85, so §6's ceiling clause (≥ 0.9 on both) is not met — moot, since nothing passed.

How fragile the count is (review 6). Seeds minus the floor, in instances of 209: +9, +7, −47, −20, −17, −42, +28, +2, −8, −52. Resampling the 209 instances with the floor recomputed gives the falsifier (≤ 3 seeds above) in 50 % of draws and ≥ 6 above in 3 %; the seed-mean difference from the floor is −0.067, instance-bootstrap 95 % [−0.138, +0.005].

Per world, needs the combination, masked (review 6's table; the references per world were missing from my first version):

| world | n | greedy | training-world reference | glued reference | lookahead-3 | RT-canon | seeds above the world's floor | seeds at or above glued |
|---|---|---|---|---|---|---|---|---|
| AI | 47 | 0.53 | 0.79 | 0.77 | 0.74 | 0.64 [0.19, 0.77] | 0/10 | 1/10 |
| CI | 162 | 0.36 | 0.46 | 0.75 | 0.72 | 0.48 [0.14, 0.66] | 5/10 | 0/10 |
| ACD | 218 | 0.03 | 0.20 | 0.43 | 0.36 | 0.37 [0.27, 0.50] | 10/10 | 1/10 |
| AID | 91 | 0.10 | 0.16 | 0.38 | 0.37 | 0.26 [0.03, 0.48] | 7/10 | 2/10 |
| CID | 195 | 0.32 | 0.25 | 0.61 | 0.73 | 0.49 [0.18, 0.77] | 8/10 | 2/10 |

What this does and does not say.
- No preregistered level was reached in any seed; the median seed is not distinguishable from a non-composing policy's rate on the pairs; the verdict "not falsified" would be "falsified" on half of resampled instance sets.
- No world shows recombination by this instrument: the glued reference, which composes nothing, is at or above RT-canon's median in every world ({A, C, D}: 0.43 against 0.37 — my first version offered that world as a sign of "something other than greedy moves", which is true and shows nothing about composing); in {A, I} RT-canon is below the world's own floor in 10/10 seeds.
- It is a statement about RT-canon at lr 1e-3. RT-post (summary item 9) meets the pair conditions in 4/10 seeds, and one of five EI16 seeds reaches 0.80; medians of every arm trained without the held-out worlds are below 0.7 (RT-post 0.64, EI16 0.65 [0.23, 0.80], RT-shuffled 0.60, RT-canon 0.47, RT-absolute 0.46, RT-examples 0.14, PF 0.13 — [main_comb_by_arm.md](results_data/main_comb_by_arm.md)).
- Part of the shortfall is the cue itself as an unseen input (summary item 2), which this design cannot separate from failing to compose decisions.
- The stratum is small (209, two worlds) and the rates are not converged (item 10). An enriched set scored from the saved routers (D128) would settle which side of the floor RT-canon is on; it cannot produce a pass.

## 3. P3 — sizes 6–7 (Q2)

Clause 1 (RT-canon, masked, training worlds): matched ratio R per seed 0.44 0.56 0.56 0.55 0.33 0.50 0.51 0.50 0.58 0.53 against 0.9 × ρ with ρ 1.01–1.04; 0/10 → falsified (≤ 3/10). Held-out worlds 0.30 [0.11, 0.40]; RT-all 0.47 on training worlds; RT-post 0.34.

Clause 2 (RT-absolute, unmasked, grouped by the 52 slots its head was trained on): must-fail group (n = 986) 0.00 in 5/5, as derived; detour-only group (n = 1,021) 0.00; the group with a shortest proof on labelled slots (n = 2,793) 0.00–0.01 → falsified. The clause conditions on the action slots and not on the input: the absolute router reads fixed sinusoidal positions (no clamp — my first version said "clamped positions", which is wrong about the code), every eval7 start term has 11 or 13 prefix tokens, and it never saw a term longer than 9. It could not have held.

Breakdown (post hoc; [main_extra.md](results_data/main_extra.md), `review6_scripts/r6_p3_probe.py`, [review6_checks.txt](results_data/review6_checks.txt)):

| RT-canon, training worlds | k = 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| sizes 6–7, masked | 0.65 | 0.59 | 0.48 | 0.40 |
| size ≤ 5, masked | 1.00 | 1.00 | 0.99 | 0.96 |
| greedy rung at sizes 6–7 | 1.00 | 0.96 | 0.85 | 0.69 |
| wrong halt, share of episodes | 0.29 | 0.24 | 0.24 | 0.24 |
| success at nesting depth 3 / 4 / 5 / 6 | 0.85 / 0.81 / 0.59 / 0.43 | 0.83 / 0.79 / 0.52 / 0.38 | 0.82 / 0.67 / 0.42 / 0.26 | 0.68 / 0.60 / 0.33 / 0.19 |

| k = 1, training worlds, masked | n | RT-canon | wrong halt (pooled) |
|---|---|---|---|
| no node of t or s past depth 4, no growth | 199 | 0.95 [0.92, 0.97] | 0.01 |
| no node past depth 4, growth | 199 | 0.79 [0.13, 0.94] | 0.29 |
| some node past depth 4, no growth | 287 | 0.77 [0.66, 0.81] | 0.18 |
| some node past depth 4, growth | 329 | 0.35 [0.06, 0.67] | 0.56 |

Review 6's finer cut: with the *rewritten* node at depth ≥ 4 (n = 251) 0.25 and a wrong halt in 0.72; 2,802 of the 2,944 one-step wrong halts are taken at step 0; growth or a rewrite at depth ≥ 4 accounts for 89 % of the one-step failures; the depth effect holds within (world, size, k) cells (−0.28 for depth ≥ 5, 10/10 seeds). The reviewer checked eval7 against the executor and eleven record invariants and found no evaluation bug.

What I can support: the structural router did not carry its multi-step competence to larger terms in any seed; the loss sits at the clamp boundary (path levels past 4 are not read, so two terms that differ only below that level look equal — the wrong halts) and at growth steps past the training cap; where neither applies, one-step rules transfer to never-seen larger terms at 0.95. What I cannot support: "learned a table" (the preregistered reading of the falsifier), or its opposite as a result — the clean subsets are post hoc, the multi-step clean subset is still below greedy, and the mechanism was not confirmed by replay. Q2 was not asked cleanly; that is my design failure on the input side of this prediction, the second after L34, plus a control whose clause was unpassable.

## 4. P4 and P1 — the prior against the lookahead ladder (Q3, Q4)

§16 cells (every (world, binding, GP1, k) cell at k ≥ 4 with n ≥ 10 whose rungs separate; six cells, n = 170; rungs 0.29 / 0.45 / 0.72; recomputed from the records and equal to the derive step's): RT-canon 0.94 [0.88, 0.96] masked, 0.92 unmasked; strictly between d = 2 and d = 3 in 0/10, ≥ d = 3 in 10/10 → falsified toward §6's "the amortized prior does three steps of search's work from data". Robustness (review 6): on the 150 instances that are not training pairs 0.93 [0.87, 0.95] against 0.73, 10/10; on the 76 whose goal was never a training goal 0.91 against 0.76, 10/10; dropping either of the two extreme cells leaves 10/10. Every other arm is above the d = 3 rung there too (RT-post 0.98, RT-all 0.95, EI16 0.94, PF 1.00, RT-examples 0.92, RT-shuffled 0.91 masked, every seed; RT-absolute 0.94 [0.68, 0.96], 4/5 — the 0.68 is seed 0, which did not converge).

By k (training worlds pooled, masked; these pooled rates include repeated training pairs): k = 4 binding GP1 0.94 against rungs 0.04 / 0.11 / 0.28; k = 5 0.86–0.92 against d = 3 at 0.18–0.61 (0.88 on the 312 new pairs); k = 6 0.73–0.91 against 0.19–0.73, the top of both ranges from a cell of 11 instances (0.73 on the 191 new pairs). P1 ({A, C}, k = 7–8, n = 600, no overlap with training): RT-canon 0.53 [0.32, 0.62], RT-post 0.72, EI16 0.76, PF 0.91 [0.73, 0.95], against 0.00 / 0.00 / 0.02 for the rungs.

Limits: the rungs are proxy-guided and stop at d = 3; the router's success is unrelated to theirs instance by instance; the terms are of the trained sizes and a third of the training-world eval instances are repeated training pairs (summary item 4); §3 shows the same routers below greedy on larger terms.

## 5. P5–P8

- **P5 holds, with little power.** Held-out binding stratum (n = 1,673), masked: RT-canon 0.63, EI16 0.68 (seeds 0.71, 0.58, 0.42, 0.68, 0.77; greedy 0.33); unmasked 0.55 / 0.49; on the 718 that need the combination 0.42 / 0.50. EI16 found routes for 97.6–98.5 % of the instances it tried. Its held-out curves drop 0.11–0.19 between consecutive checkpoints in three seeds.
- **P6 does not hold** (summary item 8). RT-canon: canonical first action 0.90–0.93 (0.89–0.92 on all 2,491 multi-class instances, still 9/10), mass on shortest-path first actions 0.94–0.96. RT-post: mass 0.96–0.97, canonical first 0.32–0.43. PF: mass 0.82–0.83.
- **P7 holds** (unmasked): clause 1 is a band that can hardly fail (§17a). Beside it, unscored: RT-shuffled, unmasked, scores 0.58 / 0.58 on the training binding stratum, so the two example proofs add about 0.3 over no cue there; on the held-out binding strata the two arms are level (pairs 0.52 against 0.54, triples and full 0.39 against 0.42), and on the needs-the-combination pairs RT-examples is the lower (0.14 against 0.60 masked, 0.08 against 0.15 unmasked).
- **P8:** masked gaps 0.03 / 0.04 (falsified: "under the mask the action set, not the presence tokens, carries the world"); unmasked gaps 0.36 / 0.35 (holds). On the 1,064 pairs of that stratum never trained on: 0.040 / 0.041 and 0.40 / 0.39. RT-shuffled unmasked fails by out-of-world actions (0.19 of training-world episodes, 0.26 of held-out).

## 6. Runs

All 41 runs are classified "learned" on their training worlds from the curve; RT-absolute s0 did not converge (0.73, loss 0.34 against 0.06–0.08 for its siblings) and the classifier still calls it learned; the others end at 0.94–1.00, on eval sets a third of which are repeated training pairs. Held-out curves: RT-absolute s3 flat at 0.34; PF flat or partial at 0.24 / 0.38 / 0.49; the rest learned or learned (partial). No run collapsed; no NaN losses. Failure categories on held-out worlds, masked: horizon throughout (RT-canon 0.31); unmasked RT-canon: horizon 0.23, no-match 0.10, out-of-world 0.04. Curves: [main_extra.md](results_data/main_extra.md) (a) — its "final − max" columns mix the 9,000-instance final point with 1,500-instance curve points and are negative for any flat noisy curve (expected −0.034 from binomial noise alone); read the window means in `review6_scripts/r6_curves.py` instead.

## 7. What was decided before and after seeing the data

Before any main-grid run: §16, §17, §17a, the verdict code and its planted tests. What I had seen of the pilot when writing §17 is stated there. After the verdicts: the split of held-out strata by "needs the combination" for all arms, the size 6–7 breakdowns, the curve tables, P6's split by whether the target is uniform, the overlap counts and everything attributed to review 6. Those are labelled post hoc or unscored wherever they appear; none changes a verdict.

## 8. What review 6 changed in this file

1. **False statement removed, overlap stated** (summary item 4, §4): "the instances are new pairs over terms of the trained sizes" → 36 % of training-world eval5 instances are verbatim training pairs; unseen-pair rates printed beside the pooled ones.
2. **RT-shuffled** (item 2): "not below RT-canon … inside its range" → ahead of it on held-out worlds under the mask, above every seed on three strata; "unmasked the tokens do the work" restricted to where it holds.
3. **The late held-out decline** (old item 8) withdrawn: the statistic was biased and on mixed denominators; the curves are still rising slowly (item 10).
4. **P2's 4/10** is no longer offered as evidence of recombination; the margins and the resampling are printed.
5. **The {A, C, D} argument** withdrawn; per-world references printed.
6. **"No arm reaches 0.7"** → true of medians; RT-post's and EI16's seeds and RT-post's counts under P2's rule printed; the data confound added to the lr confound.
7. **Sizes 6–7:** the absolute control has no clamp (my error about the code) and its clause was unpassable; the cap confound named; "cannot say why" replaced by the localisation and its limits.
8. **P6:** PF's row and the 0.05 bar added; "the rule had a defect" no longer suggests another outcome was close.
9. **P4:** "does more than three steps of lookahead's work" → "beats the proxy-guided lookahead-3 policy"; robustness rows added.
Notes taken: P4's verdict is not computed by `analyze.verdicts`; P5's power; P6's instance set; RT-absolute s0; small cells given with their n; "two thirds" is the pairs' share (the triples' is 0.52).

Not checked by anyone: an episode replayed through a saved router (the routers are on the researcher's machine); session 1's gate line; the stored training-world and glued references (used as stored by the reviewer; recomputed by me and review 5 before the grid).
