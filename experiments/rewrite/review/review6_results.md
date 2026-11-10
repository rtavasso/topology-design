# Review 6 — the rung-4 results (RESULTS.md, main_report.md and the post hoc companions)

Independent review, 10 Nov 2026. I did not design, run or analyse the experiment. Everything below was recomputed from the 41 raw records in `results/kaggle_rewrite-main/results/` and from `derive/derived.json` with my own code (`review6_scripts/`, which does not import `analyze.py`); the model, executor and dataset code were read, and the dataset files were used for the integrity and overlap checks.

## Verdict: sound with changes

- **The numbers are right.** Every number behind the verdicts of RESULTS §1 and every main number of the three companions reproduces. I found no arithmetic mismatch; the only differences are rounding and one bootstrap bound that depends on the random draws.
- **The scored verdicts are what the written rules yield.** P2, P3, P5, P6, P7 and P8 follow DESIGN §17 / §17a on these numbers, and P4 follows §16. P4's verdict is not produced by `analyze.verdicts`, although §17 says every verdict is (finding 12).
- **The readings need changes.** One statement about the data is false and hides a train / eval overlap (finding 1, blocking). Several readings are overclaimed, under-reported or state something about the code that is not so (findings 2–9, should-fix). No scored verdict changes under any of them.

## Recomputed against reported

"Match" means equal at the precision printed in RESULTS.md. Rates are seed-median [min, max].

| quantity | reported | recomputed | status |
|---|---|---|---|
| Runs, steps, parameters | 41 runs, 12,000 steps, 612,359 (RT-absolute 612,841) | same; lr 3e-4 for RT-post, 1e-3 for the rest | match |
| Process-hours; EI16 minutes per run | 44.8; 272–281 | 44.8; 272–281 | match |
| Raw checksums | — | 41 of 41 files match `RAW_SHA256SUMS` | match |
| Instance metadata across runs | — | world, t, s, k, strata and ladder identical in all 41 records | — |
| P2 pairs stratum: n, greedy, lookahead-3, floor, glued | 209, 0.40, 0.73, 0.53, 0.75 | 209, 0.397, 0.727, 0.531 (111/209), 0.751 | match |
| P2 triples stratum: n, greedy, lookahead-3, floor, glued | 509, 0.15, 0.51, 0.21, 0.50 | 509, 0.153, 0.507, 0.212, 0.497 | match |
| "Needs the combination" flags | as derived | 1,673 held-out binding instances recomputed by BFS: 0 mismatches | match |
| RT-canon pairs, masked | 0.47 [0.28, 0.67] | 0.471 [0.282, 0.665]; per seed 120, 118, 64, 91, 94, 69, 139, 113, 103, 59 of 209 | match |
| RT-canon pairs, unmasked | 0.37 [0.16, 0.56] | 0.366 [0.163, 0.565] | match |
| RT-canon triples, masked and unmasked | 0.38 [0.27, 0.56]; 0.31 [0.17, 0.51] | 0.379 [0.269, 0.556]; 0.306 [0.165, 0.509] | match |
| Pair conditions; above pairs' floor; above triples' floor (masked) | 0/10; 4/10; 10/10 | 0/10; 4/10; 10/10 | match |
| Same counts, unmasked | 0/10; 1/10; 9/10 | 0/10; 1/10; 9/10 | match |
| RT-all pairs and triples, masked | 0.94 [0.93, 0.95]; 0.85 [0.84, 0.89] | 0.938 [0.928, 0.952]; 0.847 [0.839, 0.890] | match |
| Beyond the glued reference (n 52 / 256), masked | 0.45 [0.25, 0.60]; 0.30 [0.22, 0.46] | 0.452 [0.250, 0.596]; 0.303 [0.223, 0.465] | match |
| Instance-bootstrap envelope, pairs | [0.23, 0.73] | [0.22, 0.73] with my draws | random draws |
| Per world AI / CI / ACD / AID / CID, masked | 0.64 / 0.48 / 0.37 / 0.26 / 0.49 | same | match |
| §6 strata: pairs binding (n 608); joint 7/10; RT-all 0.97 and 0.90 [0.89, 0.92] | as stated | 0.775 [0.618, 0.867]; 7/10; 0.972; 0.899 [0.889, 0.921] | match |
| P3 clause 1: R per seed | 0.44 0.56 0.56 0.55 0.33 0.50 0.51 0.50 0.58 0.53 | 0.439 0.558 0.562 0.554 0.332 0.504 0.505 0.495 0.583 0.530 | match |
| P3 clause 1: ρ; count | 1.01–1.04; 0/10 | 1.013–1.040; 0/10 | match |
| P3 held-out; RT-all; RT-post | 0.30 [0.11, 0.40]; 0.47; 0.34 | 0.30 [0.11, 0.40]; 0.47; 0.34 | match |
| P3 clause 2 groups: n; unmasked success | 986 / 1,021 / 2,793; 0.00 / 0.00 / 0.00–0.01 | same n; 0 / 0–1 / 5–14 successes per seed; R 0.002–0.005; 0/5 | match |
| P4 §16 cells | six cells, n 170, rungs 0.29 / 0.45 / 0.72 | the same six cells from the per-instance ladder; 0.288 / 0.447 / 0.724 | match |
| P4 RT-canon masked; unmasked; counts | 0.94 [0.88, 0.96]; 0.92; 0/10 between, 10/10 ≥ d = 3 | 0.935 [0.882, 0.959]; 0.92; 0/10, 10/10 | match |
| P4 other arms; sensitivity set | as stated; n 1,245, 0.91 against 0.56 | same | match |
| P1 (n 600) | RT-canon 0.53 [0.32, 0.62]; RT-post 0.72; EI16 0.76; PF 0.91 [0.73, 0.95] | same | match |
| P5 | n 1,673; 0.63; 0.68 [0.42, 0.77]; greedy 0.33; unmasked 0.55 / 0.49; 718: 0.42 / 0.50 | 0.633; 0.679 [0.421, 0.766]; 0.325; 0.553 / 0.491; 0.419 / 0.501 | match |
| P6 RT-post distance; uniform reference; count | 0.12–0.14; 0.10; 0/10 | 0.120–0.135; 0.102; 0/10 | match |
| P6 RT-canon canonical first | 0.90–0.93; 9/10 | 0.897–0.928; 9/10 | match |
| P6 split: 1,003 uniform-target of 2,213; the other 1,210 | 0.150–0.170 against 0.186, 10/10; correlation 0.51–0.59 | 0.150–0.170 against 0.186, 10/10; 0.51–0.59 | match |
| P7 | 0.89 / 0.90 / 0.92 against 0.89; 0.44 / 0.38 / 0.45 against 0.55 | 0.893 / 0.898 / 0.925 against 0.891; 0.444 / 0.383 / 0.451 against 0.553 | match |
| P8 | masked 0.94 against 0.91 / 0.90; unmasked 0.93 against 0.58 / 0.58 | gaps 0.034 / 0.038; 0.356 / 0.353 | match |
| `main_comb_by_arm.md`, all arms, both strata and complements | — | every cell reproduces | match |
| `main_extra.md` (b): by size, by k, failure categories, nesting depth | — | every cell reproduces | match |
| `main_extra.md` (a): curve table, step of the maximum, final − max | — | reproduces as computed there; the 12,000 column and "final − max" compare two different instance sets (finding 3) | denominator differs |
| `p6_check.txt` | — | reproduces | match |
| Failure categories, held-out; RT-shuffled out-of-world 0.19 / 0.26 | as stated | same | match |

Screens for an evaluation bug, all clean:
- **Record invariants** (`r6_invariants.py`): 11 per-instance invariants over all 41 × 18,600 records and the cell aggregation of every run, 0 violations. Among them: an unmasked success implies a masked success, a success takes between k and T steps, and the stored cells equal the per-instance aggregation.
- **eval7 against the executor** (`r6_p3_probe.py` part 1): for the 4,800 training-world instances the cap is 7, my own BFS distance equals k, and at k = 1 the stored target set is exactly the set of valid actions that reach s. Every correct first action maps to a slot below HALT, and terms have 11 or 13 nodes against `MAX_NODES` 13.
- **Horizon and instances.** The ladder and the router are compared on the same instances with the same T = 2k + 2, and the ladder halts at the goal for free while the router must choose HALT. The comparison is fair; if anything it favours the ladder.

## Findings

### Blocking

**1. The eval sets overlap the training stream, and the write-up says they do not.**

Sentence (§4, Limits): *"the instances are new pairs over terms of the trained sizes"*.

The build does not exclude training instances from eval5 (`build_dataset.py` has no such step, and I found no check of it in the earlier reviews). From `r6_overlap.py`, which streams the eight training files:

| eval5, training worlds | n | (t, s) is a training instance | (t, s) is an RT-canon teacher-forced row | RT-canon masked on seen / unseen pairs (seed-mean; seed-median in the first and last rows) |
|---|---|---|---|---|
| all | 4,800 | 0.357 | 0.410 | 0.995 / 0.959 |
| k = 4 | 896 | 0.277 | 0.302 | 0.989 / 0.945 |
| k = 5 | 402 | 0.224 | 0.239 | 0.967 / 0.874 |
| k = 6 | 279 | 0.315 | 0.315 | 0.978 / 0.738 |
| the §16 cells | 170 | 20 instances | 25 instances | 1.00 / 0.930 |

- **Where it comes from.** Start terms are sampled uniformly over sizes 1–5, so small terms repeat: 90 % of eval instances with a one-leaf t are verbatim training instances, 70 % at two leaves, 55 % at three, 10 % at four, 0.6 % at five.
- **P4 survives.** On the 150 §16 instances that are not training instances RT-canon is 0.930 [0.873, 0.953] against a lookahead-3 rung of 0.73, at or above it in 10/10 seeds. On the 76 whose goal was never a training goal it is 0.914 [0.776, 0.934] against 0.76, again 10/10.
- **P8 survives.** On the 1,064 unseen pairs of the training binding stratum the masked gaps are 0.040 / 0.041, still under 0.05, and the unmasked gaps 0.40 / 0.39.
- **P1 is clean.** No k = 7–8 instance is a training instance or a teacher-forced row.
- **What must change.** The sentence is false. The per-k rates in §4 and summary item 5 are raised by recall at k = 5–6: at k = 6 RT-canon is at 0.74 on the 191 new pairs and 0.98 on the 88 repeated ones, and at k = 5 at 0.87 and 0.97. The training-world strata of P6, P7 and P8 are a quarter verbatim training instances (0.24 of the 1,408; 0.25 of P6's 2,213), and the "learned, 0.94–1.00" of §6 includes them.
- **It also bears on summary item 2.** Of the held-out eval5 instances, 30 % (pairs), 27 % (triples) and 32 % (full world) are verbatim (state, goal) rows that RT-canon was trained on inside a contained training world, with only the presence tokens different. RT-canon scores 0.97 / 0.89 / 0.92 on those and 0.71 / 0.57 / 0.45 on the rest. Of the held-out binding instances that need no unseen combination, 45 % (pairs) and 47 % (triples) are such rows. Item 2's point is stronger than written: much of what looked like held-out generalization is recall of trained rows under a new token pattern.
- **Fix.** Correct the sentence, state the overlap, and print the unseen-pair rates beside the pooled ones. All of it comes from the existing records; no run is needed.

### Should-fix

**2. Summary item 3 under-reports the shuffled-token arm: on held-out worlds the token-blind router is ahead of RT-canon, and "unmasked the tokens do the work" holds only on training worlds.**

Sentences (item 3): *"on the held-out needs-the-combination stratum it is not below RT-canon: pairs 0.60 [0.59, 0.61] against 0.47 [0.28, 0.67], triples 0.48 against 0.38 (two seeds; inside RT-canon's range)"* and *"Unmasked the tokens do the work: 0.93 against 0.58 on training worlds, and RT-shuffled falls to 0.15 / 0.14 on the needs-the-combination strata."*

Both sentences are true of the strata they name. The strata they leave out say more (`r6_misc.py`):

| eval5 stratum | RT-canon masked | RT-shuffled masked (2 seeds) | RT-canon seeds below both | RT-canon unmasked | RT-shuffled unmasked |
|---|---|---|---|---|---|
| held-out, all (n 4,200) | 0.68 [0.63, 0.79] | 0.79, 0.80 | 9/10 | 0.62 [0.57, 0.73] | 0.60, 0.61 |
| triples, non-binding (1,335) | 0.71 [0.65, 0.80] | 0.81, 0.81 | 10/10 | 0.67 [0.63, 0.77] | 0.69, 0.70 |
| triples, binding, no combination needed (556) | 0.75 [0.63, 0.88] | 0.92, 0.90 | 10/10 | 0.61 [0.57, 0.69] | 0.69, 0.67 |
| full world (600) | 0.60 [0.52, 0.67] | 0.76, 0.78 | 10/10 | 0.56 [0.49, 0.63] | 0.73, 0.74 |
| pairs, needs the combination (209) | 0.47 [0.28, 0.67] | 0.59, 0.61 | 9/10 | 0.37 [0.16, 0.56] | 0.13, 0.16 |

- **Masked.** RT-shuffled is above every RT-canon seed on three strata. With two seeds against ten the smallest attainable exact rank-sum probability is 1/66, and it is reached there. "Not below" and "inside RT-canon's range" describe only the strata where the difference is smallest.
- **Unmasked.** Over all held-out instances the two arms are level (0.62 against 0.60 / 0.61). In the full world, where no action is out of world, the token-blind arm leads by 0.17 in 10/10 seeds.
- **What it means for P2.** Reading the true tokens at a pattern never seen in training costs RT-canon success relative to ignoring them. Part of RT-canon's held-out shortfall is therefore an unseen-input effect of the cue, not only a failure to compose decisions. Item 1's "the gap is the unseen combination" should say so, and item 3 should print these rows. Two seeds is thin, and the write-up should say that too.

**3. Summary item 8's late held-out decline is not in the records.**

Sentence: *"RT-canon's held-out curve peaks before the end in 10/10 seeds and finishes 0.01–0.10 below its maximum while the training-world curve is flat or rising."*

- **The arithmetic reproduces, on mixed denominators.** Points up to step 11,750 are on the first 100 instances per world (1,500 instances); the point at 12,000 is on the full sets (9,000). `main_extra.py` takes "final − max" and the 12,000 column across the two. Recomputed on the same 1,500 instances (`r6_curves.py`) the drops are −0.03 −0.01 −0.06 −0.08 −0.04 −0.10 −0.02 −0.04 −0.03 −0.06, so the range stands.
- **The statistic is negative for any noisy flat curve.** A plateau of 25 points with only binomial noise at n = 700 gives an expected final − max of −0.034 and a peak before the end in 96 % of draws. The held-out curves are noisier than that (point-to-point sd 0.014–0.042).
- **Every other arm shows the same thing**, including RT-all's training-world curve (−0.01 in 3/3).
- **In window means the curve is still rising.** RT-canon's seed-mean held-out success by 1,000-step window from step 4,000: 0.645, 0.655, 0.672, 0.671, 0.680, 0.683, 0.692, 0.696. The mean over 10,000–12,000 exceeds the mean over 8,000–9,750 in 8/10 seeds (median +0.018).
- **Fix.** Replace the sentence with: held-out success was still rising slowly when the runs stopped, which is a limit on reading the 12,000-step rates as converged.

**4. P2's "4/10 above the floor" is a coin flip at the instance level, and the held-out rates wander within a run.**

Sentences (§1 table and §2): *"not met, not falsified (one seed from the falsifier)"*; *"It does not say the router cannot recombine: 4/10 seeds are above the floor"*.

- **Margins.** Seeds minus the floor, in instances of 209: +9, +7, −47, −20, −17, −42, +28, +2, −8, −52. The fourth seed above the floor (seed 7) is above by two instances; the binomial standard error of a seed is about seven.
- **Instance resampling.** Resampling the 209 instances, with the floor recomputed on each resample, gives three or fewer seeds above the floor — the falsifier — in 50 % of draws, and six or more in 3 % (`r6_posthoc.py`). The seed-mean difference from the floor is −0.067 with a 95 % instance-bootstrap interval of [−0.138, +0.005].
- **Checkpoint noise.** On the curve subset's held-out pairs binding cells (the same 103 instances at every checkpoint) one RT-canon run moves with sd 0.048 and range 0.16 across the last nine checkpoints, against a between-seed sd of 0.079 for the final rate. A seed's rate at step 12,000 is partly which checkpoint it is.
- **Fix.** "One seed from the falsifier" is accurate. "4/10 seeds are above the floor" should not be offered as evidence that the router can recombine: the verdict would be "falsified" on half of resampled instance sets.

**5. "At the level of a non-composing policy" is fair as a rate, but the per-world references are omitted and the {A, C, D} argument does not show what it is used for.**

Sentence (§2): *"on {A, C, D} the router is at the lookahead-3 rung (0.37 against 0.36) with greedy at 0.03 — that world's instances are being solved by something other than greedy moves."*

Per world, needs-the-combination stratum, masked (`r6_p2_flags.py`):

| world | n | greedy | training-world reference | glued reference | lookahead-3 | RT-canon | seeds above own floor | seeds at or above glued |
|---|---|---|---|---|---|---|---|---|
| AI | 47 | 0.53 | 0.79 | 0.77 | 0.74 | 0.64 [0.19, 0.77] | 0/10 | 1/10 |
| CI | 162 | 0.36 | 0.46 | 0.75 | 0.72 | 0.48 [0.14, 0.66] | 5/10 | 0/10 |
| ACD | 218 | 0.03 | 0.20 | 0.43 | 0.36 | 0.37 [0.27, 0.50] | 10/10 | 1/10 |
| AID | 91 | 0.10 | 0.16 | 0.38 | 0.37 | 0.26 [0.03, 0.48] | 7/10 | 2/10 |
| CID | 195 | 0.32 | 0.25 | 0.61 | 0.73 | 0.49 [0.18, 0.77] | 8/10 | 2/10 |

- **{A, C, D}.** The glued reference, which composes nothing, scores 0.43 there, above RT-canon's 0.37. "Something other than greedy moves" is true and is not evidence of recombination. Print the per-world references beside the per-world rates.
- **{A, I}.** RT-canon is below the world's own floor in every seed.
- **Rate against behaviour.** The summary's "at the level of a policy that does not compose" is right about the rate, 0.47 against 0.53. It is not a description of behaviour: RT-canon solves 0.48 of the instances the training-world reference solves and 0.45 of those it does not. RT-canon does not act like that reference; it scores like it.
- **"Below the glued reference".** *"the routers are below the glued reference on both strata"* holds for the medians. On the triples two of ten seeds are at or above it (0.499 and 0.556 against 0.497).

**6. "No arm … reaches 0.7 on the pairs" is true of medians only, and the other ten-seed arm is not at the floor.**

Sentence (§2): *"No arm trained without the held-out worlds reaches 0.7 on the pairs (RT-post 0.64, EI16 0.65 [0.23, 0.80], …)"*; item 1: *"the router trained on eight worlds is at the level of a policy that does not compose"*.

- **By seed.** Four of ten RT-post seeds (0.79, 0.73, 0.73, 0.71) and one of five EI16 seeds (0.80) are at or above 0.7.
- **P2's rule applied to RT-post**, unscored: pair conditions in 4/10 masked, above the pairs' floor in 8/10 masked and 7/10 unmasked, above the triples' floor in 10/10 on both readings.
- **Fix.** Item 1 describes RT-canon at lr 1e-3 and should say so in the sentence. The RT-post counts belong beside it, not only as a median in item 8. The confounds named there apply, and there are more of them than the learning rate: the pilot chose 3e-4 on 0.919 against 0.916 at one seed, and RT-post trains on 1.1–2.5 times as many distinct (state, goal) rows per world (for example 81,921 against 33,371 in {A, C}).

**7. Sizes 6–7: the records say more about why than the write-up allows, one named confound is wrong about the code, and one is missing.**

Sentences (item 4 and §3): *"Neither position representation extrapolated to sizes 6–7, and the measurement cannot say why"*; *"Both routers read sizes 6–7 through features clamped to the size-≤ 5 range"*; *"The absolute control's collapse is likewise confounded with its clamped positions"*; *"which aliases exactly the nodes the larger terms add"*.

(a) **The absolute control has no clamp.** `model.py` gives it fixed sinusoidal positions with no clamping (the `else` branch of `Router.encode`; DESIGN §12a.5 says the same). Its confound is different and total: every eval7 start term has 11 or 13 prefix tokens, and it never saw a term longer than 9. Clause 2's group "a shortest proof on labelled slots" conditions on the action slots, not on the input, so the clause could not hold for this router whatever it learned. The rule is mis-specified, not only the result confounded.

(b) **A third confound is not named: the cap.** Training runs at cap 5, so a term of the largest training size never has a growth action (IbR, IbL, Db) as a label. eval7 runs at cap 7, and 528 of the 1,014 one-step training-world instances (52 %) require adding a leaf. The size clamp reads 6 and 7 leaves as 5, exactly the size at which growth was never labelled, and it makes the root sizes of t and s read equal. This is separate from depth aliasing.

(c) **The failure is sharply localised** (`r6_p3_probe.py`; k = 1, training worlds, masked):

| k = 1 subset | n | RT-canon | wrong halt | RT-all | RT-post |
|---|---|---|---|---|---|
| no node of t or s deeper than 4, no growth | 199 | 0.95 [0.92, 0.97] | 0.01 | 0.97 | 0.95 |
| no node deeper than 4, growth | 199 | 0.79 [0.13, 0.94] | 0.20 | 0.55 | 0.45 |
| rewritten node at depth ≤ 3, no growth | 416 | 0.91 [0.88, 0.95] | 0.03 | 0.94 | 0.85 |
| rewritten node at depth ≤ 3, growth | 347 | 0.67 [0.13, 0.86] | 0.27 | 0.47 | 0.34 |
| rewritten node at depth ≥ 4 | 251 | 0.25 [0.03, 0.56] | 0.72 | 0.14 | 0.07 |

- By the depth of the rewritten node, seed-mean: 0.91 / 0.79 / 0.78 / 0.71 at depths 0–3, then 0.31 at depth 4 and 0.20 at depth 5.
- Of the wrong halts at k = 1, 2,802 of 2,944 are taken at step 0: the router reads t as equal to s.
- Growth or a rewrite at depth ≥ 4 accounts for 89 % of RT-canon's one-step failures.
- The depth effect holds within (world, size, k) cells: −0.28 for depth ≥ 5 against ≤ 4, negative in 10/10 seeds. The author's depth table is not a size or world artefact.

(d) **What this supports.** Where no feature but the root's subtree size is clamped and the step does not grow the term, the structural router carries one-step rules to never-seen terms of 11–13 tokens at 0.92 or better in every seed. That is evidence against "learned a table" as the reason for the falsifier, and it locates the loss at the clamp boundary and at the cap. The write-up's "cannot say why" under-reports its own records.

(e) **What it does not support.** A clean pass. The subsets are post hoc. At k = 2–4 the in-range-depth, non-growing instances (defined on the endpoints only, since intermediate states are not recorded) score 0.86 / 0.76 / 0.66 against 1.00 / 0.99 / 0.96 at size ≤ 5 and against a greedy rung of 0.97 / 0.86 / 0.72. A per-step loss of roughly 0.05–0.10 remains, and the router is still below greedy.

(f) **"Aliases exactly the nodes the larger terms add"** overstates: 398 of the 1,014 one-step instances have no node past depth 4, and only the root's size is clamped there.

**Fix.** Correct (a), name (b), and replace "cannot say why" with the table above, labelled post hoc. Add the wrong-halt spread to *"ends a quarter of its episodes in a wrong halt"*: it is 0.24–0.29 pooled and 0.12–0.52 by seed at k = 1.

**8. P6: the diagnosis is right, but "the rule had a defect" should not suggest another outcome was close.**

Sentences (item 7): *"P6's rule was badly formed … The uniform reference is the target on 1,003 of the 2,213 instances, so it scores zero there and no learned policy can beat it."*

- **Confirmed.** 1,003 of 2,213 targets are exactly uniform. With its 0.075–0.094 error on those, RT-post would need a distance of 0.12 or less on the other 1,210 to win overall; it has 0.150–0.170.
- **The alternative on the table fails too.** The auditor's bar of 0.05 (§17's P6 row) is missed by RT-post's 0.12–0.14.
- **The error on uniform-target instances is RT-post's own miss.** It is about half the uniform reference's error where the weights matter.
- **The post hoc statistic is not specific to RT-post.** PF, never trained on that target, is closer to it than RT-post on the 1,210 non-uniform instances (0.140–0.150 against 0.150–0.170) and on the uniform ones (0.056–0.065 against 0.075–0.094), after renormalising its 0.82 mass. "Closer than uniform, correlation 0.5–0.6" shows a first-decision distribution that leans the right way; it does not show RT-post "at its fixed point".
- **Labelling.** The reading is labelled post hoc and the verdict is kept, which is fair. Add the PF row and the 0.05 comparison.

**9. P4: the verdict stands, and its wording needs two qualifiers.**

Sentence (item 5): *"Inside the training range the no-search router does more than three steps of lookahead's work."*

- **Robustness.** With finding 1's duplicates removed the result holds (0.93 against 0.73). Dropping the cell that sits exactly on the 0.1 threshold (ID|1|1|4) gives 0.930 [0.887, 0.960] against 0.767, 10/10. Dropping ID|0|0|4, whose top rung is 1.00, gives 0.931 against 0.693, 10/10.
- **Qualifier 1: the unit.** RT-canon's success is unrelated to the ladder's on these instances: 0.932 where lookahead-3 succeeds and 0.928 where it fails. The router is not a deeper version of the ladder. It was trained on exact shortest-path labels, while the ladder's reach is limited by a tree-edit proxy. "More than three steps of lookahead's work" should read "beats the proxy-guided lookahead-3 policy"; DESIGN §1 already calls the rungs proxy-relative.
- **Qualifier 2: "three steps or more".** This is the author's label. §6's falsifier says "three steps of search's work".

### Notes

**10. P3 clause 1 as scored mixes three things.** The eval7 set combines position extrapolation, the clamp and growth past the training cap (finding 7). The "falsified" verdict is what the rule yields, and the write-up is right not to read it as "learned a table". A reader of the §1 table alone sees "falsified, both clauses" with no flag that neither clause tested its claim cleanly. Put the flag in the table row.

**11. P5 "holds" with little power.** EI16's seeds are 0.71, 0.58, 0.42, 0.68, 0.77 and three of five are within 0.1 of RT-canon's median; the means are 0.632 and 0.640. The rule is on medians and the write-up gives the range. EI16's held-out curves are the least stable in the grid: drops of 0.11–0.19 between consecutive checkpoints in three seeds, and a falling 2,000-step trend in 4/5.

**12. P4's verdict is not computed by `analyze.verdicts`.** §17 says *"The analysis computes every verdict (`analyze.verdicts`)"*, and RESULTS §1 lists P4 under "verdict under §17". The verdict table at the end of `main_report.md` has no P4 row; the reading comes from the counts in the P4 table. `separated_cells` also compares floats without the tolerance §17a.6 promises. Both tolerances give the same six cells on this data.

**13. P6's instance set in the code is narrower than the text.** The code takes multi-class instances with more than one shortest-path first action (2,213 of 2,491 multi-class training instances). On all 2,491, RT-canon's canonical-first share is 0.89–0.92, still 9/10. No effect on the verdict.

**14. RT-absolute seed 0 did not converge** (training worlds 0.73, loss 0.34 against 0.06–0.08; k = 7–8 at 0.01). §6 discloses it. It is the 0.68 at the low end of the RT-absolute range quoted in §4. The curve classifier calls it "learned".

**15. Small cells quoted without their n.** *"k = 6 0.73–0.91 against 0.19–0.73"* takes its 0.91 and its 0.73 rung from a cell of 11 instances. ACI's needs-the-combination row is 5 instances; the write-up gives that n and no rate, which is right.

**16. "Two thirds of which need no unseen combination"** (item 2) is the pairs' share (399/608). The triples' share is 0.52.

**17. Checks that came out as written.**
- Provenance: 41 results, return codes 0, 38 passed and 1 skipped in the second session's log.
- P7 and P8 readings, including RT-shuffled's failure by out-of-world actions.
- "Below the greedy rung at every k" at sizes 6–7: 10/10 seeds at every k.
- RT-all at sizes 6–7.
- The lr confound on the RT-post observation is stated.
- RT-all's median meets the pair conditions; §6's ceiling clause is not met on the triples (0.847; 0.899 on §6's own stratum).
- No NaN losses in the stored losses or the curve.
- No run collapsed, and on training worlds no run's mean over the last 1,250 steps exceeds its mean over the 1,250 before by 0.01 or more.

## What I could not check

- **No episode could be replayed.** The 41 saved routers are not in the repository. "Checked by hand against the executor" therefore means the instances, targets, slots and record invariants, not the model's forward pass. The wrong halts at depth ≥ 4 are consistent with the clamp, but I could not confirm the mechanism by feeding a failing instance to a router with the clamp lifted.
- **The first session.** The log of the first Kaggle session is gone, and the directory is not a git checkout here. I could not verify that bundles `20ebab0` and `32c9f53` differ only in `launch.py`, nor the gate result of the first session, which produced 29 of the 41 records, RT-canon seed 0 among them. That seed is not an outlier on any stratum.
- **The stored references.** The training-world and glued references were used as stored in `derived.json`; I recomputed the "needs the combination" flag but not the two reference rollouts. `abs_labelled_slots` and `eval7_abs_group` were also used as stored.
- **Curve-level P2.** The curve cells carry (world, binding, GP1, k) only, so the needs-the-combination stratum cannot be followed over training. Finding 4's checkpoint noise is measured on the pairs' whole binding stratum of the curve subset.
- **Intermediate states.** They are not recorded, so finding 7(e)'s multi-step subsets are defined on the endpoints.

## Scripts (`review6_scripts/`; none imports `analyze.py`)

Run from the repository root. `r6_common.py` builds a compact cache of the records on first use, about two minutes and 150 MB, at the path in `R6_CACHE` (default: the system temp directory).

| script | what it computes | command |
|---|---|---|
| `r6_common.py` | loader and cache; asserts the instance metadata is identical across the 41 runs | `python3 experiments/rewrite/review/review6_scripts/r6_common.py` |
| `r6_recompute.py` | every number behind P2–P8 and P1, and the verdict logic | `python3 …/r6_recompute.py` |
| `r6_posthoc.py` | the companions (all arms on P2's strata and complements; sizes 6–7 by size, k, category, depth; curve table) and P2's fragility | `python3 …/r6_posthoc.py` |
| `r6_p2_flags.py` | the "needs the combination" flag from scratch; per-world references | `PYTHONPATH=src python3 …/r6_p2_flags.py` |
| `r6_p3_probe.py` | eval7 integrity against the executor; depth within (world, size, k); depth of the rewritten node; growth; the three-way split | `PYTHONPATH=src python3 …/r6_p3_probe.py` |
| `r6_overlap.py` | train / eval overlap of instances and teacher-forced rows; rates on unseen pairs | `PYTHONPATH=src python3 …/r6_overlap.py` |
| `r6_curves.py` | curves on one denominator; window trends; the flat-plateau null; largest drops | `python3 …/r6_curves.py` |
| `r6_misc.py` | provenance sums; P2's rule on every arm; RT-shuffled against RT-canon on every stratum; P4 sensitivity; P6 post hoc; checkpoint noise | `python3 …/r6_misc.py` |
| `r6_invariants.py` | record invariants | `python3 …/r6_invariants.py` |
