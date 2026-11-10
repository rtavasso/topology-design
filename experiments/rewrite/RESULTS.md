# Rung 4 (the rewrite carrier): results

**Provenance.** Design: [DESIGN.md](DESIGN.md) revision 3 with §16 (P4's cells, amended at the derive step), §17 and §17a (the other pass rules, amended after the pass-rule audit and review 5; all before any main-grid run). Grid: [grids/main.json](grids/main.json), 41 runs of 12,000 steps (RT-canon 10, RT-post 10, RT-all 3, RT-absolute 5, EI16 5, PF 3, RT-examples 3, RT-shuffled 2; lr 3e-4 for RT-post, 1e-3 for the rest, from the pilot — [PILOT.md](PILOT.md)); 612,359 parameters (RT-absolute 612,841). Two Kaggle sessions on two T4s, four shards: bundle `20ebab0` (9 Nov, 29 runs inside the 10.5 h session budget) and bundle `32c9f53` (10 Nov, resumed with the 29, ran the other 12: RT-all s0, RT-canon s1–s9, RT-shuffled s0–s1); the bundles differ only in `launch.py`. Gate on the box: 38 passed, 1 skipped; `results=41 of 41`, `rcs=[0, 0, 0, 0]`. 44.8 process-hours against 27 projected (EI16 took 272–281 min a run against 148 projected). The first session's kernel log and shard logs were overwritten on the researcher's machine by the second session's fetch; its 29 result files and checkpoints were not. Records: `results/kaggle_rewrite-main/results/` (outside git; checksums in [results_data/main/RAW_SHA256SUMS](results_data/main/RAW_SHA256SUMS)), compact copies without the per-instance lists in [results_data/main/](results_data/main/), 41 saved routers on the researcher's machine. Analysis: [results_data/main_report.md](results_data/main_report.md) from `analyze.py` (the scored verdicts are its last section); unscored companions written after the verdicts were read: [main_comb_by_arm.md](results_data/main_comb_by_arm.md), [main_extra.md](results_data/main_extra.md), [p6_check.txt](results_data/p6_check.txt). **Independent results review: pending** (review 6); until it is in, every reading below is the author's.

Words used below. *Masked*: the argmax is taken over the valid actions and HALT (the executor supplies the action set). *Unmasked*: over all 105 slots; an invalid action ends the episode. *Binding*: the missing axiom lengthens the shortest proof (d_full < d_W). *Needs the combination*: a held-out binding instance with no proof within the horizon inside a training world contained in the held-out one (pairs n = 209, triples n = 509; DESIGN §17). Rates are seed-medians with [min, max] unless a count is given.

## 1. Summary

| # | question | verdict under §17 | the numbers |
|---|---|---|---|
| P2 (core) | recombination across held-out axiom sets | **not met, not falsified** (one seed from the falsifier) | pairs' needs-the-combination stratum: RT-canon 0.47 [0.28, 0.67] masked against a non-composing floor of 0.53, lookahead-3 0.73, RT-all 0.94; pair conditions met in 0/10 seeds; above the floor in 4/10 (falsifier: ≤ 3/10) |
| P3 (core) | size extrapolation, structural against absolute positions | **falsified, both clauses** | structural: matched ratio 0.33–0.58 against 0.9 × ρ ≈ 0.92 (0/10); absolute: 0.00–0.01 even where a shortest proof lies on labelled slots (0/5) |
| P4 (core) | how much lookahead the prior absorbs | **falsified toward "three steps or more"** | on the §16 cells (n = 170) RT-canon 0.94 [0.88, 0.96] against the lookahead-3 rung's 0.72; ≥ the d = 3 rung in 10/10 |
| P5 | search-found routes against labels | **holds** | held-out binding stratum (n = 1,673): EI16 0.68, RT-canon 0.63 |
| P6 | the tie mixture | **does not hold as written**; the rule had a defect found afterwards | RT-post's distance to its target 0.12–0.14 against 0.10 for the uniform reference (0/10); RT-canon canonical-first ≥ 0.9 in 9/10 |
| P7 | examples in place of stated axioms | **holds** | training binding 0.89 / 0.90 / 0.92 against 0.89 ± 0.1; held-out 0.44 / 0.38 / 0.45 against RT-canon's 0.55 |
| P8 (null) | shuffled presence tokens | **masked: falsified; unmasked: holds** | training binding stratum: masked 0.94 against 0.91 / 0.90; unmasked 0.93 against 0.58 / 0.58 |

1. **Recombination under a stated cue was not shown.** On the held-out instances that cannot be proved inside any training world they contain, the router trained on eight worlds is at the level of a policy that does not compose (pairs: 0.47 against 0.53), while the same router trained on all fifteen worlds reaches 0.94 on those instances — so they are learnable, and the gap is the unseen combination. The falsifier was missed by one seed (4/10 above the floor against ≤ 3/10). On the triples the router is above its floor in 10/10 seeds (0.38 against 0.21) and below the glued reference (0.50), which composes nothing beyond single-world plans joined by greedy moves. Unmasked: pairs 0.37, above the floor in 1/10.
2. **What looked like generalization to held-out worlds is mostly instances that need no unseen combination.** On the held-out binding instances provable inside a contained training world RT-canon scores 0.93 (pairs, n = 399) and 0.75 (triples, n = 556); on those that need the combination 0.47 and 0.38 ([main_comb_by_arm.md](results_data/main_comb_by_arm.md), unscored). §6's stratum as preregistered (pairs' binding, n = 608) reads 0.77: RT-canon would have met §6's pair conditions in 7/10 seeds and its triples' clause in 10/10, with §6's RT-all clause on the line (0.97 on the pairs, 0.90 [0.89, 0.92] on the triples and the full set). So the rule as first preregistered would have passed, or come within a hair of passing, on instances two thirds of which need no unseen combination — the pass-rule audit's point, now measured.
3. **Under the mask the stated axioms are not what carries the world, on training or held-out worlds.** RT-shuffled (presence tokens made uninformative) is within 0.04 of RT-canon on the training worlds' binding stratum (P8's falsifier, as the pilot's sanity table had suggested it would be), and on the held-out needs-the-combination stratum it is not below RT-canon: pairs 0.60 [0.59, 0.61] against 0.47 [0.28, 0.67], triples 0.48 against 0.38 (two seeds; inside RT-canon's range). Unmasked the tokens do the work: 0.93 against 0.58 on training worlds, and RT-shuffled falls to 0.15 / 0.14 on the needs-the-combination strata.
4. **Neither position representation extrapolated to sizes 6–7, and the measurement cannot say why.** The structural router keeps half of its matched success (0.52), falls *below the greedy rung* at every k (k = 1: 0.65 against 1.00), ends a quarter of its episodes in a wrong halt, and its success falls with nesting depth past the clamp (k = 1: 0.81–0.85 at depth ≤ 4, 0.59 at 5, 0.43 at 6). RT-all fails the same way (0.47), so it is not a matter of worlds. The absolute control solves nothing (0.03 at k = 1, masked). Both routers read sizes 6–7 through features clamped to the size-≤ 5 range (the fix for L34), which aliases exactly the nodes the larger terms add; "learned a table" (the preregistered reading of the falsifier) and "the clamped encoding cannot tell the new nodes apart" are not separated by this grid ([main_extra.md](results_data/main_extra.md) (b), unscored).
5. **Inside the training range the no-search router does more than three steps of lookahead's work.** On the six separated cells it is at 0.94 where lookahead-3 is at 0.72; at k = 5–6 it scores 0.73–0.92 on cells where lookahead-3 scores 0.18–0.73; at k = 7–8 in {A, C} 0.53 [0.32, 0.62] where lookahead-3 scores 0.02 (P1, reported). The ladder stops at d = 3, so Q3's quantity is bounded below only. Read with item 4, this reach is for terms of the trained sizes.
6. **Search-found routes did as well as labels (P5), and examples in place of stated axioms did worse on held-out worlds (P7).** EI16's range over five seeds is wide (0.42–0.77). RT-examples on the needs-the-combination pairs: 0.14 masked, 0.08 unmasked.
7. **P6's rule was badly formed, which I found only after it returned "does not hold".** The uniform reference *is* the target on 1,003 of the 2,213 instances, so it scores zero there and no learned policy can beat it. On the other 1,210 RT-post is closer to its target than the uniform reference in 10/10 seeds (0.150–0.170 against 0.186), with the deviations from uniform correlated at 0.51–0.59 with the target's. That second reading is post hoc and is not a verdict; the verdict stays "does not hold".
8. **Unscored observations worth a test, not a reading:** RT-post is above RT-canon on every held-out stratum (needs-the-combination pairs 0.64 against 0.47 masked, 0.59 against 0.37 unmasked; k = 7–8 0.72 against 0.53) and below it at sizes 6–7 (0.34 against 0.52) — at a different learning rate (3e-4 against 1e-3), so the arms are not matched. RT-canon's held-out curve peaks before the end in 10/10 seeds and finishes 0.01–0.10 below its maximum while the training-world curve is flat or rising.

## 2. P2 — recombination (Q1)

Scored rule (§17): on the pairs' needs-the-combination stratum, RT-canon ≥ greedy + 0.2 and ≥ 0.7 in ≥ 6/10 seeds, and above the triples' floor in ≥ 6/10; falsified if above the pairs' floor in ≤ 3/10 while RT-all's median meets the pair conditions.

| stratum | n | greedy | lookahead-3 | floor (training-world reference) | glued reference | RT-canon masked | unmasked | RT-all masked |
|---|---|---|---|---|---|---|---|---|
| pairs, needs the combination | 209 | 0.40 | 0.73 | 0.53 | 0.75 | 0.47 [0.28, 0.67] | 0.37 [0.16, 0.56] | 0.94 [0.93, 0.95] |
| triples, needs the combination | 509 | 0.15 | 0.51 | 0.21 | 0.50 | 0.38 [0.27, 0.56] | 0.31 [0.17, 0.51] | 0.85 [0.84, 0.89] |
| pairs, beyond the glued reference | 52 | – | 0.56 | – | 0 | 0.45 [0.25, 0.60] | 0.37 [0.12, 0.48] | – |
| triples, beyond the glued reference | 256 | – | 0.36 | – | 0 | 0.30 [0.22, 0.46] | 0.24 [0.11, 0.43] | – |

Counts over RT-canon's ten seeds, masked: pair conditions 0/10; above the pairs' floor 4/10; above the triples' floor 10/10. Unmasked: 0/10, 1/10, 9/10. RT-all's median meets the pair conditions (0.94), so the falsifier's precondition held; RT-all's triples median is 0.85, so §6's ceiling clause (≥ 0.9 on both) is not met — moot here, since nothing passed.

Per world (needs the combination; masked / unmasked; greedy; lookahead-3): AI n = 47: 0.64 / 0.50; 0.53; 0.74. CI n = 162: 0.48 / 0.32; 0.36; 0.72. ACD n = 218: 0.37 / 0.34; 0.03; 0.36. AID n = 91: 0.26 / 0.24; 0.10; 0.37. CID n = 195: 0.49 / 0.39; 0.32; 0.73. ACI n = 5.

What this does and does not say.
- It does not say the router cannot recombine: 4/10 seeds are above the floor, the instance-bootstrap envelope over seeds on the pairs is [0.23, 0.73], and on {A, C, D} the router is at the lookahead-3 rung (0.37 against 0.36) with greedy at 0.03 — that world's instances are being solved by something other than greedy moves. It says the preregistered level was not reached in any seed and the median seed is not distinguishable from a non-composing policy on the pairs.
- The stratum is small (209) and two worlds. An enriched set scored from the saved routers (D128) would separate "at the floor" from "a little above it"; it cannot turn 0/10 into a pass.
- The masked reading cannot separate planning over the unseen combination from glued single-world plans (§17a); the routers are below the glued reference on both strata, so the question does not arise.
- All arms on these strata: [main_comb_by_arm.md](results_data/main_comb_by_arm.md). No arm trained without the held-out worlds reaches 0.7 on the pairs (RT-post 0.64, EI16 0.65 [0.23, 0.80], RT-shuffled 0.60, RT-canon 0.47, RT-absolute 0.46, RT-examples 0.14, PF 0.13).

## 3. P3 — sizes 6–7 (Q2)

Clause 1 (RT-canon, masked, training worlds): matched ratio R per seed 0.44 0.56 0.56 0.55 0.33 0.50 0.51 0.50 0.58 0.53 against 0.9 × ρ with ρ 1.01–1.04; 0/10 → falsified (≤ 3/10). Held-out worlds 0.30 [0.11, 0.40]; RT-all 0.47 on training worlds; RT-post 0.34.

Clause 2 (RT-absolute, unmasked, grouped by the 52 slots its head was trained on): must-fail group (n = 986) 0.00 in 5/5, as derived; detour-only group (n = 1,021) 0.00; the group with a shortest proof on labelled slots (n = 2,793) 0.00–0.01, R ≤ 0.01 → falsified ("failing more than the unaddressable share").

Breakdown (unscored, [main_extra.md](results_data/main_extra.md)):

| RT-canon, training worlds | k = 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| sizes 6–7, masked | 0.65 | 0.59 | 0.48 | 0.40 |
| size ≤ 5, masked | 1.00 | 1.00 | 0.99 | 0.96 |
| greedy rung at sizes 6–7 | 1.00 | 0.96 | 0.85 | 0.69 |
| wrong halt, share of episodes | 0.29 | 0.24 | 0.24 | 0.24 |
| success at nesting depth 3 / 4 / 5 / 6 | 0.85 / 0.81 / 0.59 / 0.43 | 0.83 / 0.79 / 0.52 / 0.38 | 0.82 / 0.67 / 0.42 / 0.26 | 0.68 / 0.60 / 0.33 / 0.19 |

The reading I can support: the structural router did not carry its size-≤ 5 competence to larger terms, in any seed, by a wide margin, and the failure is already present in one-step instances and in the halting decision — it is not a planning failure. The reading I cannot support: that it "learned a table". The structural features are clamped at depth 4 and subtree size 5 so that no untrained embedding row is read (L34); at sizes 6–7 that makes distinct nodes share features, and success falls along exactly that axis. Depth-≤ 4 terms of sizes 6–7 still lose 0.15–0.19 at k = 1, so depth aliasing is not the whole of it (subtree size is clamped too, and every term of size 6–7 has a root past the clamp). The absolute control's collapse is likewise confounded with its clamped positions. A fixed (non-learned) encoding, or training sizes that leave a margin inside the feature range, would be needed to ask Q2 at all; this grid did not ask it cleanly. That is a design failure on the input side, the second on this prediction after L34.

## 4. P4 and P1 — lookahead absorbed (Q3, Q4)

§16 cells (every (world, binding, GP1, k) cell at k ≥ 4 with n ≥ 10 whose rungs separate; six cells, n = 170; rungs 0.29 / 0.45 / 0.72; recomputed from the records and equal to the derive step's): RT-canon 0.94 [0.88, 0.96] masked, 0.92 unmasked; strictly between d = 2 and d = 3 in 0/10, ≥ d = 3 in 10/10 → falsified toward "the amortized prior does three steps of search's work from data". Every other arm is above the d = 3 rung there too (RT-post 0.98, RT-all 0.95, EI16 0.94, PF 1.00, RT-examples 0.92, RT-shuffled 0.91 masked, every seed; RT-absolute 0.94 [0.68, 0.96], 4/5). The sensitivity set (upper separation only, n = 1,245): 0.91 against 0.56.

By k (training worlds pooled, masked): k = 4 binding GP1 0.94 against rungs 0.04 / 0.11 / 0.28; k = 5 0.86–0.92 against d = 3 at 0.18–0.61; k = 6 0.73–0.91 against 0.19–0.73. P1 ({A, C}, k = 7–8, n = 600): RT-canon 0.53 [0.32, 0.62], RT-post 0.72, EI16 0.76, PF 0.91 [0.73, 0.95], against 0.00 / 0.00 / 0.02 for the rungs.

Limits: the ladder ends at d = 3 and at k ≥ 5 its top rung is far below the routers, so "how many steps" has no upper reading here; the instances are new pairs over terms of the trained sizes, and item 4 of the summary shows the same routers below greedy on larger terms.

## 5. P5–P8

- **P5 holds.** Held-out binding stratum (n = 1,673), masked: RT-canon 0.63, EI16 0.68 [0.42, 0.77] (greedy 0.33); unmasked 0.55 / 0.49; on the 718 that need the combination 0.42 / 0.50. EI16 found routes for 97.6–98.5 % of the instances it tried.
- **P6 does not hold as written** (summary item 7). RT-canon: canonical first action 0.90–0.93, mass on shortest-path first actions 0.94–0.96. RT-post: mass 0.96–0.97, canonical first 0.32–0.43. PF: mass 0.82–0.83.
- **P7 holds** (unmasked): clause 1 is a band that can hardly fail (§17a). Beside it, unscored: RT-shuffled, unmasked, scores 0.58 / 0.58 on the training binding stratum, so the two example proofs add about 0.3 over no cue there; on the held-out binding strata the two arms are level (pairs 0.52 against 0.54, triples and full 0.39 against 0.42), and on the needs-the-combination pairs RT-examples is the lower (0.14 against 0.60 masked, 0.08 against 0.15 unmasked).
- **P8:** masked gaps 0.03 / 0.04 (falsified: "under the mask the action set, not the presence tokens, carries the world"); unmasked gaps 0.36 / 0.35 (holds). RT-shuffled unmasked fails by out-of-world actions (0.19 of training-world episodes, 0.26 of held-out).

## 6. Runs

All 41 runs are classified "learned" on their training worlds from the curve (RT-absolute s0 to 0.73 only; the others end at 0.94–1.00). Held-out curves: RT-absolute s3 flat at 0.34; PF flat or partial at 0.24 / 0.38 / 0.49; the rest learned or learned (partial). Failure categories on held-out worlds, masked: horizon throughout (RT-canon 0.31); unmasked RT-canon: horizon 0.23, no-match 0.10, out-of-world 0.04. No NaN losses. Curves by step: [main_extra.md](results_data/main_extra.md) (a).

## 7. What was decided before and after seeing the data

Before any main-grid run: §16, §17, §17a, the verdict code and its planted tests. What I had seen of the pilot when writing §17 is stated there. After the verdicts: the split of held-out strata by "needs the combination" for all arms, the size 6–7 breakdown by k, category and depth, the curve table, and P6's split by whether the target is uniform. Those are labelled unscored wherever they appear above; none changes a verdict.
