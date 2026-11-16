# Do streams recombine where a monolithic block does not? — staged design on a "worlds" carrier

**Status: EXPLORATORY design, revision 2 (15 Nov 2026).** Revision 1 is in git at 808286e. It was reviewed before any run (`review/review_design.md`: "run after changes"); this revision makes those changes. What I had seen when writing it: 100–300-step smoke tests of every arm, and the reviewer's untuned one-seed probes at coverage 9 (table in the review). History Entries 106–108; idea I37 in the ledger. Code `worlds.py`, `stage0.py`; exact statements `tests/test_worlds.py`.

## 1. The question and the logic

I37: does breaking the block into streams of learned primitives, recombined by a gate, let a loop handle combinations it was not trained on where one monolithic block does not? Its null: recombination comes with coverage, not architecture.

Entry 105's comparison failed on four counts (Entry 106): the baseline never failed; programs were held out but nothing was held out from the block; the task named one primitive per pass; a shared controller did the recombining. Revision 1 of this design fixed those and, the review showed, walked into a fifth: its monolith read the world as one summed vector and failed on held-out worlds by producing an *unconstrained function of an unfamiliar vector* — worse than not moving, worse than ignoring the world — which is not the failure rung 4 showed (a model sitting *at* its non-composing reference), and a monolith that reads the world one token at a time recombined with no streams at all. So "streams against monolith" was confounded with **how the world is read**. Revision 2 separates the two.

## 2. The carrier

Six primitives f_p(h) = tanh(A_p h) on a six-dimensional state, each with a trigger direction w_p. A **world** is a set of two primitives, stated in the input as two tokens; fifteen worlds exist. A block is applied four times; on every pass it gets the state h and the world's two tokens (embeddings E, learned by each arm). No clock, no register, nothing outside the block.

- **select** (primary). Each pass applies *one* of the world's two primitives: the one whose trigger ⟨w_p, h⟩ is larger. Which one is not stated; it depends on the state and changes from pass to pass. Step 0.4.
- **sum** (secondary). Each pass applies *both*, summed. Step 0.2.

**Held-out worlds and the dial.** Each split is a random order of the fifteen worlds; the training set of coverage c is its first c worlds (7, 9, 11, 13, 15), so the dial is nested. The first seven already form a graph on the primitives that is 2-connected (connected, and still connected with any one primitive removed) and contains an odd cycle. Splits do not depend on the model's seed; twelve distinct splits, cycling through three teachers.

**What the training worlds determine** (test `test_what_training_worlds_determine…`). `sum`: every primitive is recoverable from the trained sums. `select`: less than revision 1 claimed. Training shows only the *sign* of each trained comparison. A held-out choice between a and b is forced by the data only where a chain of trained comparisons forces it at that state (a beats c, c beats b); that is 30–60 % of decisions. Beyond that, the right answer follows only under an assumption about the form of the bids (linear triggers). So there are three levels on held-out worlds, each with an exact reference computed from the teacher:

| Reference (select) | What it knows | Typical error |
|---|---|---|
| no change | nothing | 1.0 |
| `allmean` | ignores the world: applies the mean of all six primitives | 0.8 |
| `blend` | which two primitives; cannot choose | 0.5 |
| `trans` | chooses correctly wherever trained comparisons force it, blends elsewhere | 0.3 |
| the arm's own error on seen worlds | everything | floor |

For `sum` the references are `allmean` and `one` (applies one of the two).

**Regimes**, read off those references: *garbage* (at or above `allmean`: worse than ignoring the world); *does not choose* (around `blend`); *recombines* (at most half of `blend`); *beyond what the data force* (at most half of `trans`); *full* (within 3× of its own seen error). Rung 4's model was in the second.

**Controls for the unfamiliar-input confound** (F52): a teacher whose primitives are all identical, so the world is irrelevant — any gap there is the input alone; and a one-pass regression of the block's update on the six true primitive outputs on held-out worlds (R², and the share of weight on the world's own two primitives).

**Limits stated now.** Tokens map one-to-one to primitives: the candidates are named; what must be discovered is each primitive's function and trigger, from pairs only. The `select` teacher (an available set, per-primitive bids, larger bid wins) has the shape of the "bids" arms below; the carrier cannot show that streams help when the data are not of this shape. Fifteen worlds, three teachers.

## 3. Arms: read × block

All have about 2,500 trainable parameters (within 3 %; test).

| | One block | Streams |
|---|---|---|
| **Summed read** r = E[a] + E[b] | `mono`: two hidden layers of 40 on [h; r] | `streams_self`: 8 streams of width 16; gate logit of stream k = G_k·r + s_k(h) (each stream bids from the state; the world adds a bias). Also `streams_joint` (logits G r + MLP([r; h]), Entry 104's form) |
| **One branch per token**, pooled by a softmax over the branches' bids | `tok_mono`: the same two-layer block run on [h; E[a]] and on [h; E[b]]; each branch emits an update and a bid | `tok_streams`: each token picks streams (softmax of G·E[p]); a branch's update and bid are those of its streams |

`tok_mono` is one set of weights instantiated once per token — monolithic in parameters, branched in computation (the shared, code-conditioned interpreter of Neural Interpreters). `tok_streams` differs from it only in keeping each primitive in its own parameters. So the two columns ask I37's question twice, once under each read, and the two rows ask the question the review raised.

Further rows: `unit` (summed read; a soft gate on each of 60 units — gating without grouping); `mono` with weight decay (grid 1e-3, 1e-2, 1e-1; its **best held-out** error is reported, which is generous to it); `oracle` (streams with the gate set from the truth: the floor). Streams run densely; cost is Entry 104's subject.

## 4. Outcomes

Error = squared error of the change h_4 − h_0 relative to the teacher's change, on fresh inputs fixed per split (the same for every arm). Reported per run for seen and held-out worlds, per held-out world, and placed in the regimes of §2. The held-out/seen ratio is *not* an outcome (it grows with fit). Comparisons between arms are paired ratios of held-out error over the same splits: median, number of splits won, and a bootstrap interval over splits; an **identical-arm null** (`tok_mono` against itself with a different initialisation) gives the spread such ratios have by chance.

## 5. Tuning rules (L57)

- Each arm: learning rate from {0.003, 0.01, 0.03, 0.1}, cosine decay; extended if the best is at an edge. Summed-read streams arms also choose softmax or sigmoid scores.
- Selection by seen-world error on three other splits (100–102). Since that cannot see the outcome, held-out error is reported for every configuration, and **a verdict must hold at every rate whose seen error is within 2× of the best**.
- Steps: 4,000 and 8,000 are compared directly in stage 0; seen and held-out error are recorded every 1,000 steps.
- Fit: `mono`'s seen error must come within 2× of `tok_mono`'s at some width (40 or 120) and step count; if it cannot, `mono` is reported as under-fitting and its row is a reference, not a baseline.

## 6. Stages

**Stage 0 — the two monoliths, no streams.** (i) Rates at coverage 9. (ii) The dial: `select` at 7, 9, 11, 13, 15; `sum` at 7, 9, 15; six splits. (iii) Fit: `mono` at width 120; both at 8,000 steps (coverage 9, `select`). (iv) The irrelevant-world control.
Questions: which regime is each monolith in on held-out worlds, at each coverage? Is the summed read's failure the input alone (control) or an unconstrained function of it? Does the per-token monolith leave anything to fix?
**Stop rule:** if `tok_mono` is *full* on `select` at coverages 7 and 9 and `mono` is too, nothing can differ between arms: stop and report. Otherwise stage 2 runs. (If only `tok_mono` is full, the per-token column has no room; stage 2 still runs for the summed-read column and to confirm.)

**Stage 2 — the 2 × 2 and the further rows**, twelve splits, coverages 9 and 11 (7 as a secondary: eight streams there outnumber seven training worlds), both tasks, each arm tuned as in §5. The four contrasts (paired ratios of held-out error):
- C1 streams under the per-token read: `tok_streams` / `tok_mono`.
- C2 streams under the summed read: `streams_self` / `mono`.
- C3 the read, one block: `tok_mono` / `mono`.
- C4 the read, streams: `tok_streams` / `streams_self`.

Readings fixed in advance:
- **Support for I37** (under a given read): C1 or C2 at most 1/2 in at least 10 of 12 splits, outside the identical-arm null, with the streams arm at least *recombining* (≤ half of `blend`), and the weight-decay monolith not doing the same.
- **The read, not streams:** C3 and C4 at most 1/5, and C1 within [1/2, 2].
- **Gating, not grouping:** `unit` within 2× of `streams_self` where `streams_self` beats `mono`.
- **Null:** all four contrasts within [1/2, 2].

**Stage 3 — mechanism**, for any arm that recombines: agreement between the branch or stream with the largest weight and the true primitive on held-out worlds; removing a stream and checking which primitives suffer.

Independent review of results after stage 2 (or at a stop).

## 7. Predictions (committed before stage 0; I had seen the smoke tests and the reviewer's probes named above)

Stage 0. **R0a** `mono` on `select` at coverage 9 is in the *garbage* regime in at least 5 of 6 splits, at width 40 and 120. **R0b** In the irrelevant-world control both monoliths have held-out error within 1.5× of seen. **R0c** `tok_mono` on `select` at coverage 9 *recombines* in at least 5 of 6 splits and is below `trans` in at least 4, but is not *full* in at least 4. **R0d** `tok_mono` on `sum` is *full* at every coverage. **R0e** `tok_mono`'s held-out error on `select` falls from coverage 7 to 13.

Stage 2 (`select` unless stated). **R2a** C3 ≤ 1/5 in 12 of 12 splits, both tasks. **R2b** C1 within [1/2, 2]: with the per-token read, separate streams add nothing clear (confidence about 0.55). **R2c** C2 ≤ 1/2 in at least 10 of 12, but `streams_self` does not *recombine* (median above half of `blend`). **R2d** C4 ≤ 1/5. **R2e** `sum`: both per-token arms *full*; neither summed-read arm *recombines* (median above half of the smaller reference). **R2f** the weight-decay monolith's best is within 2× of `mono`. **R2g** `unit` within 2× of `streams_self`.

If R2a, R2b and R2d hold, the answer to I37 on this carrier is: recombination comes from branching the computation per available primitive and pooling by a simple rule, whether or not the branches have separate weights.

## 8. Budget

One run is 10–40 s on one core. Stage 0: about 170 runs. Stage 2: tuning about 250 short runs, main about 300. Two processes; two to three hours of wall time.
