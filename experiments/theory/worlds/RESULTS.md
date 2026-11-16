# Do streams recombine where a monolithic block does not? — results on the worlds carrier

**Status: EXPLORATORY.** Small synthetic carrier, twelve splits, three teachers, CPU. Design: `DESIGN.md` (revision 2; reviewed before any run, `review/review_design.md` and `review/review_design_rev2.md`). Tables: `stage0.md`, `stage2.md`; per-run numbers in the JSON files beside them. Every script and its predictions were committed before it ran. The results and this write-up were reviewed independently (`review/review_results.md`: "sound with corrections" — every number reproduced and no bug was found; the verdict on I37 was worded more strongly than the data allowed, and the reading below is the corrected one). History Entry 108.

## The question

Idea I37: does breaking a looped block into streams of learned primitives, recombined by a gate, let it handle combinations it was not trained on where one monolithic block does not? Its null: recombination comes with coverage, not architecture.

## The carrier in one paragraph

Six primitives act on a state; each has a trigger. A *world* is a set of two primitives, named in the input. On every one of four passes the block gets the state and the world. In `select` each pass applies the one primitive of the two whose trigger is larger at the current state; in `sum`, both. Training uses 7 or 9 of the 15 worlds (every primitive appears; the rest are held out). Exact references from the teacher place a held-out error on a scale: 1.0 not moving; about 0.8 ignoring the world; about 0.5 knowing the two primitives and blending them (`blend`); 0.2–0.3 choosing correctly only where trained comparisons force the choice (`trans`); the arm's own error on trained worlds.

Arms are a 2 × 2 with about 2,500 parameters each. **Read:** the world enters as one summed vector, or the block is run once per token and the branches are pooled by a softmax over one bid each. **Block:** one two-layer block, or eight streams with a gate.

## Result

`select`, 9 of 15 worlds in training, twelve splits. Errors are medians.

| Arm | Read | Block | Error on trained worlds | Error on held-out worlds | Where that sits |
|---|---|---|---|---|---|
| `mono` | summed | one block | 0.066 | 1.3 | worse than not moving, 12/12 |
| `mono`, width 120 (6.8× the parameters) | summed | one block | 0.023 | 1.0 | worse than ignoring the world, 12/12 |
| `mono`, best weight decay | summed | one block | 0.088 | 0.92 | worse than ignoring the world, 9/12 |
| `unit` (a gate on every unit) | summed | gated units | 0.015 | 0.63 | above the blend reference, 11/12 |
| `streams_self` | summed | streams | 0.0071 | 0.42 | around blending (0.88 of `blend`) |
| `streams_joint` | summed | streams | 0.0088 | 0.39 | around blending |
| `tok_mono` | per token | one block | 0.025 | 0.059 | 0.13 of `blend`, 0.28 of `trans` |
| `tok_streams` | per token | streams | 0.0081 | 0.041 | 0.08 of `blend`, 0.18 of `trans` |
| `oracle` (gate set from the truth) | — | streams | 2e-5 | 6e-6 | floor |

The four contrasts fixed in advance (ratio of held-out errors over the same splits; median, bootstrap 5–95 %, splits won):

| Contrast | Coverage 9 | Coverage 7 | The same ratio on trained worlds |
|---|---|---|---|
| C3 the read, with one block: `tok_mono` / `mono` | 0.05 [0.04, 0.06], 12/12 | 0.07 [0.05, 0.08], 12/12 | 0.37; 0.46 |
| C4 the read, with streams: `tok_streams` / `streams_self` | 0.10 [0.06, 0.14], 12/12 | 0.08 [0.06, 0.11], 12/12 | 1.10; 1.02 |
| C1 streams, under the per-token read: `tok_streams` / `tok_mono` | 0.65 [0.51, 0.86], 9/12 | 0.85 [0.59, 0.95], 9/12 | 0.33; 0.34 |
| C2 streams, under the summed read: `streams_self` / `mono` | 0.32 [0.30, 0.37], 12/12 | 0.55 [0.51, 0.62], 12/12 | 0.11; 0.15 |
| Same arm, another initialisation (`tok_mono`; `mono`) | 0.94; 1.03 | 1.05; 1.05 | 1.02; 1.03 |

## Reading, against the readings fixed before the runs

**"The read, not streams" is met** (C3 and C4 at most 1/5; C1 within [1/2, 2]) at both coverages. **"Gating, not grouping" is met** (`unit` within 2× of `streams_self`). **"Support for I37" — as the design defined it: streams against one block, under the same read — is not met** under either read. **The null is not met**: the read matters by a factor of ten to twenty.

1. **How the set of available primitives is read decides whether a block recombines.** Run once per token and pooled by bids, a single block with one set of weights is at 0.13 of the blend reference on worlds it never saw, and below a third of what the trained comparisons force. Read as one summed vector, a block of the same size is worse than not moving in 12/12 splits; at 6.8 times the parameters, where it fits trained worlds as well as the per-token block, it is still worse than ignoring the world in 12/12; weight decay (tried at the matched width only, and best at the top of its grid) does not change the regime. In a control where the world is irrelevant the summed-read block shows no gap (5e-6 on both), so the failure is not the unfamiliar vector as such — a weak control, since a block fitted to a world-independent target need not read the world at all.
2. **Under the per-token read, streams lower the held-out error by 15–35 %, short of the factor of two fixed in advance; whether that is fit or recombination is not resolved.** At equal learning rates the ratio is 0.65 and 0.80 (coverage 9) and 0.85 and 0.69 (coverage 7), streams ahead in 9–11 of 12 splits, outside the spread of two initialisations (0.94, 1.05) at coverage 9 and overlapping it at 7. Two things cut against reading it as better recombination: streams fit trained worlds three times better, and the *difference* between held-out and trained error is the same in the two arms (0.031 against 0.034). One thing cuts against reading it as fit alone: giving the single block more steps or 6.8 times the width improves its trained-world error (×0.76, ×0.64) and hardly moves its held-out error (×0.91, ×1.02; the width probe is the reviewer's, three splits).
3. **Under the summed read, streams are far better than the monolith and still do not recombine on `select`.** They fit trained worlds nine times better and are three times better on held-out worlds at coverage 9 (C2 = 0.32, 12/12), 1.8 times at coverage 7 (0.55; at most 1/2 in only 2/12) — at an error around the blend reference at coverage 9 (0.88 of it) and above it at coverage 7 (1.5 of it). That is an error level, not a mechanism: after one pass their update is nearer the true update than the blend in 11 of 12 splits. A gate on every unit, without grouping, is within 1.0–1.4 times of streams.
4. **`sum`** (coverage 9): both per-token arms are within 3× of their trained-world error in 12/12 splits — close to forced, since nothing is held out from a branch that sees one token, and summing branches is the task's own form. Under the summed read the monolith is above both of its references in 12/12 (median 1.06); streams reach 0.15 and are below half the smaller reference in 8 of 12 splits, which I had predicted they would not be. So on `sum`, separate streams under the summed read *do* partly recombine.
5. **Mechanism** (`select`, coverage 9, six splits, retrained). On one pass from fresh states the branch with the larger pooled weight is the primitive the teacher applies in 0.98–0.99 of cases on trained worlds; on held-out worlds, in 0.93–0.94 of the cases that trained comparisons do *not* force, for both per-token arms. (The 0.999 on forced cases is selection by margin and follows from a scalar bid once trained decisions are right; matched for margin the reviewer finds 0.984 on trained worlds and 0.947 on unforced held-out ones, and a learned bid difference linear in the true one, correlation 0.96.) So the per-token arms recover per-primitive bids of the teacher's form and those bids transfer to comparisons never trained. In `tok_streams` every token has its own stream (6/6 splits) and removing it raises the error on worlds containing that token at least 100 times more than on the others.

## What this says about I37

It depends on which half of I37 is meant, and the experiment separates the two.

- **"Parallel learned transformations, recombined by a learned gate"** — the shape of the computation, which is how the ledger words I37 and what Neural Interpreters is. On this carrier that shape is what recombines: one branch per available primitive, each producing an update and a bid, pooled by a competitive rule (C3 = 0.05, C4 = 0.10). On that reading the result supports I37.
- **"Each primitive in its own weights"** — streams as separate parameter sets, against one shared block. That was *not needed*: a single block conditioned on the token and run once per branch recombines; separate weights add 15–35 % at most. And separate weights *without* the per-item shape — streams chosen by a gate that reads a pooled summary of the context, the form Entry 104 arrived at — stayed at the level of blending on `select` (they partly recombined on `sum`).

In the design's own closing words, which the outcome matches: recombination comes from branching the computation per available primitive and pooling by a simple rule, whether or not the branches have separate weights.

**How much of this is by construction.** A good deal. Each branch of a per-token arm sees a state and one named primitive, both covered in training; on `sum` nothing at all is held out from a branch, and on `select` only the comparison of two bids is. Entry 107's first property — primitives not named by the input — is not met: the tokens name the candidates. What is not by construction is that the learned bids transfer to comparisons never trained (0.93–0.95 where the data do not force the answer), and that a gate over a pooled summary fails where a per-item gate works.

## Deviations from the design

Coverages 7 and 9 in stage 2 (the design named 9 and 11, with 7 secondary), changed on stage 0's evidence before stage 2. `sum` at coverage 9 only, without nulls or the wide and weight-decay monoliths. The wide monolith at `select`, coverage 9 only. The ceiling arm was run as a row of stage 2 rather than as its own stage (the review's advice). The extra learning rates for the per-token arms were chosen after C1 was known; rate 0.003, which also qualifies, was not run by me (the reviewer's probe on five splits gives 0.55 for `tok_streams` against 0.54 for `tok_mono`). Stage 3 on six splits.

## Predictions

Written after the design reviewer's one-seed probes, so not blind (said in `DESIGN.md`). Stage 0: R0a, R0b, R0e held; R0c partly (the per-token monolith was *full* in 5 of 6 splits at coverage 9, where I predicted at most 2); R0d partly (`sum`: full in 5/6 at coverage 7). Stage 2: R2a (a confirmation), R2b, R2d, R2f, R2g held; R2c partly (held at coverage 9; at coverage 7 streams beat the summed-read monolith by a factor of two in only 2 of 12 splits); R2e partly (the summed-read streams did recombine on `sum`). Stage 3's expectations held.

## Limits

- **The candidates are named.** Each token names an available primitive, so "one branch per token" is handed its items. Where the available primitives are not itemised in the input, something must produce the items, and nothing here tests that. (Rung 4's worlds did state their axioms as tokens.)
- **The `select` teacher has the shape of the per-token arms** — an available set, a bid per primitive, the larger bid wins. The carrier shows that an architecture of that shape finds and extends such structure where others do not; it cannot show the shape helps when the data are not like this.
- **A stream contains the teacher's primitive exactly** (V tanh(B h) of width 16), while the single block has to approximate six primitives through an added input. Every "ratio on trained worlds" reflects that, and it is part of why streams fit better.
- **Fifteen worlds, six primitives, three teachers, a six-dimensional state, four passes.** Twelve splits; AGENTS.md asks for more seeds and a full bootstrap for a headline claim.
- **Rates were chosen on three splits by trained-world error**, which cannot see held-out error; held-out error rises with the rate in both per-token arms, and C1 is reported at each rate pairing.
