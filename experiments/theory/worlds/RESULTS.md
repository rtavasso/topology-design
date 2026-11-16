# Do streams recombine where a monolithic block does not? — results on the worlds carrier

**Status: EXPLORATORY.** Small synthetic carrier, twelve splits, three teachers, CPU. Design: `DESIGN.md` (revision 2; reviewed before any run, `review/review_design.md` and `review/review_design_rev2.md`). Tables: `stage0.md`, `stage2.md`; per-run numbers in the JSON files beside them. Every script and its predictions were committed before it ran. History Entry 108.

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
| `mono`, width 120 (6.8× the parameters) | summed | one block | 0.023 | 1.0 | the same, 12/12 |
| `mono`, best weight decay | summed | one block | 0.088 | 0.92 | worse than ignoring the world, 9/12 |
| `unit` (a gate on every unit) | summed | gated units | 0.015 | 0.63 | worse than blending, 10/12 |
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

**"The read, not streams" is met** (C3 and C4 at most 1/5; C1 within [1/2, 2]) at both coverages. **"Support for I37" is not met under either read.**

1. **How the set of available primitives is read decides whether a block recombines.** Run once per token and pooled by bids, a single block with one set of weights is at 0.13 of the blend reference on worlds it never saw, and below a third of what the trained comparisons force — it has learned bids that compare correctly across primitives never compared in training. Read as one summed vector, the same block is worse than not moving, at any width or weight decay, although it fits trained worlds equally well at width 120. A control in which the world is irrelevant shows no gap at all, so this is not the unfamiliar vector as such; it is an unconstrained function of it.
2. **Streams do not supply recombination.** Under the summed read, streams fit trained worlds nine times better than the monolith and are three times better on held-out worlds — and still only at the level of blending the two primitives: they know which two are available and do not choose between them. Under the per-token read, streams fit trained worlds three times better than the single block and are 1.2–1.5 times better on held-out worlds; the gap between held-out and trained error is *larger* with streams (4.7× against 2.4× at coverage 9). The 0.65 is outside the spread of two initialisations (0.94) but it is not stable across learning rates whose fit is equally good: pairing the arms at other such rates gives 0.49, 0.80 and 1.08.
3. **Gating without grouping does no better than streams** under the summed read (`unit` is 1.0–1.4 times `streams_self` on held-out worlds); neither recombines.
4. **`sum`** (coverage 9): both per-token arms are within 3× of their trained-world error in 12/12 splits — this is close to forced, since summing per-token branches is the task's own form. Under the summed read the monolith is again worse than not moving; streams reach 0.15 (8 of 12 splits below half the smaller reference), which I had predicted they would not.
5. **Mechanism** (`select`, coverage 9, six splits, retrained). On one pass from fresh states, the branch with the larger pooled weight is the primitive the teacher applies in 0.98–0.99 of cases on trained worlds; on held-out worlds in 0.999 of the cases the trained comparisons force and 0.93–0.94 of the cases they do not, for both per-token arms. In `tok_streams` every token has its own stream (6/6 splits) and removing a token's stream raises the error on trained worlds containing that token at least 100 times more than on the others.

## What this says about I37

On this carrier the answer to "do streams let a loop recombine where a monolithic block does not" is **no, as posed** — and the experiment says what does. Recombination came from the *shape of the computation*: one branch per available primitive, each producing an update and a bid, pooled by a simple competitive rule. That shape is "parallel streams recombined by a gate". What it did not need is separate weights per stream: one shared block, conditioned on the token, run once per branch, recombined as well relative to its fit (this is the shared, code-conditioned interpreter of Neural Interpreters). And separate weights without that shape — streams chosen by a gate that reads a pooled summary of the context, which is the form Entry 104 arrived at — did not recombine.

## Predictions

Written after the design reviewer's one-seed probes, so not blind (said in `DESIGN.md`). Stage 0: R0a, R0b, R0e held; R0c partly (the per-token monolith was *full* in 5 of 6 splits at coverage 9, where I predicted at most 2); R0d partly (`sum`: full in 5/6 at coverage 7). Stage 2: R2a (a confirmation), R2b, R2c, R2d, R2f, R2g held; R2e partly (the summed-read streams did recombine on `sum`). Stage 3's expectations held.

## Limits

- **The candidates are named.** Each token names an available primitive, so "one branch per token" is handed its items. Where the available primitives are not itemised in the input, something must produce the items, and nothing here tests that. (Rung 4's worlds did state their axioms as tokens.)
- **The `select` teacher has the shape of the per-token arms** — an available set, a bid per primitive, the larger bid wins. The carrier shows that an architecture of that shape finds and extends such structure where others do not; it cannot show the shape helps when the data are not like this.
- **Fifteen worlds, six primitives, three teachers, a six-dimensional state, four passes.** Twelve splits; AGENTS.md asks for more seeds and a full bootstrap for a headline claim.
- **Rates were chosen on three splits by trained-world error**, which cannot see held-out error; held-out error is reported at every rate tried, and the one contrast that moved with the rate (C1) is reported as such.
- The single block under the per-token read fits trained worlds three times worse than streams (0.025 against 0.008); the comparison of their held-out errors is therefore read through the ratio on trained worlds, not on its own.
