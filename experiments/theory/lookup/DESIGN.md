# The lookup block against the other stream configurations and a monolithic block — design

**Status: DESIGN, revision 1, nothing run. EXPLORATORY.** Written 16 Nov 2026 at the researcher's request ("design the experiment to compare this version with the other moe stream configurations and monolithic"). To be reviewed by an agent that did not write it before any script is written. History Entry 113. The version under test is the one in `docs/weight_lookup.md` (Entries 110–112, idea I38).

## 1. What the experiment is for

To decide which stream configuration the program carries forward: the per-token streams arm that recombined in Entry 108, or the lookup block. Three things are measured for every arm: **recombination** (error on worlds never trained), **training** (error on trained worlds, failed runs, how blocks get allocated to primitives), and **compute per pass**.

## 2. What is already settled, and what that leaves to measure

The exact results (tests U1–U8, W1–W6) fix a good deal before anything is trained:

- Assembling one block from retrieved units computes the same function as summing the retrieved streams' outputs (U1). With stored keys shared inside blocks, the lookup *is* the per-token streams arm `tok_streams` (U4).
- Keys computed from the weights and bids computed from a block's own units change where numbers come from, not which functions can be expressed.

So on a carrier whose items are named by tokens, **the lookup's held-out error can at best equal `tok_streams`**. A design that promised more would be predicting an outcome by construction (lesson L58). What training can still decide:

| Open question | Why it is open |
|---|---|
| **A. Is naive gate-and-sum in weight space fragile here?** | The researcher's premise for wanting a lookup. Exact: the two mixings agree only at one-hot gates (W1), and under weight mixing every stream receives the same gradient direction (W2). Whether that breaks training on this carrier is not known. |
| **B. Does each new ingredient survive training?** | Entry 112 lists reasons it might not: addresses can no longer move ahead of content, soft retrieval has a merging feedback, a hard load can starve blocks (Entry 103). |
| **C. What does it cost?** | Arithmetic, given B: a hard load computes the loaded blocks only. |
| **D. Does the ranking hold as the bank grows?** | Starvation and merging are problems of many blocks; the cost gap grows with the bank. With six primitives all of this may look fine and say little. |

## 3. Carrier

The worlds carrier of Entry 108, unchanged (`experiments/theory/worlds/worlds.py`): P primitives f_p(h) = tanh(A_p h) on a six-dimensional state, a trigger w_p each; a world is two primitives named by tokens; four passes; the block sees the state and the world. Tasks `select` (apply the primitive whose trigger is larger) and `sum` (apply both). Some worlds are never trained; every primitive is. The failure this line is about is already reproduced there: the summed-read monolith is worse than not moving on held-out worlds in 12/12 splits. Exact references (`blend`, `trans`, own seen error) place every held-out error on a scale.

**One new dial: the number of primitives P ∈ {6, 12, 24}**, with training worlds fixed at 1.5 P (each primitive in three trained worlds on average — coverage 9 of 15 at P = 6). Trained worlds are then 60 %, 27 % and 13 % of all worlds. Blocks K = P + 2. The split sampler becomes constructive for P > 6 (a random cycle through all primitives plus random chords, checked by the existing `graph_ok`: 2-connected and not bipartite).

## 4. Arms

Every new arm is `tok_streams` with **one** change, plus the full combination. Parameters matched at about 2,500 for P = 6 (freed parameters go to block width).

| Arm | Read of the world | How blocks are addressed | Bids (state decides) | Load | Combination | New? |
|---|---|---|---|---|---|---|
| `mono` | summed | — | — | — | one two-layer block | Entry 108 |
| `tok_mono` | per token | — | from the block | — | one shared block, run per token | Entry 108 |
| `streams_self`, `streams_joint`, `unit` | summed | gate on the pooled read | separate net | soft | outputs | Entry 108 |
| **`tok_streams`** (T0) | per token | stored keys | separate net | soft | outputs | Entry 108 |
| `wmix` (T1) | per token | stored keys | separate net | soft | **weights: W̄ = Σ g_k W_k, then one block** | new |
| `wmix_sum` | summed | gate on the pooled read | separate net | soft | weights | new (the routing form of T1) |
| `selfbid` (T2) | per token | stored keys | **β_k · tanh(B_k h), the block's own units** | soft | outputs | new |
| `probe` (T3) | per token | **keys computed from the weights**: the block's outputs at C = 3 learned probe states, projected (U8) | separate net | soft | outputs | new |
| `probe_sg` (T3s) | as T3, gradient from key to weights cut | | | | | new |
| `hard` (T4) | per token | stored keys | separate net | **one block per token**, trained with Entry 104's schedule (soft until mean top probability > 0.8, then top-1) | outputs = assembled block (U1) | new |
| **`lookup`** (L) | per token | computed keys | own units | one block per token | assembled block | **the version** = T2 + T3 + T4 |
| `oracle` | — | set from the truth | — | — | — | floor |

Two fairness notes. On `sum`, `wmix` at block width 16 cannot represent the target at all: averaging two blocks' weights adds their functions only when the blocks occupy different hidden units (W4), which for six primitives of six units needs width 36. It is run at width 16 (matched parameters; expected to fail for that reason alone) and at width 36 (able to represent the target; about 1.4 times the parameters), and only the second bears on question A. On `select` the target gate is one-hot, where weight and output mixing agree, so `wmix` is a fair arm at width 16.

## 5. Measures

- **Held-out and seen error**, medians over splits, with the regime of Entry 108 (garbage / worse than blend / partial / recombines / beyond / full).
- **Failed runs:** seen error non-finite or more than 10 times the arm's median.
- **Allocation** (per-token arms): number of different blocks the P tokens load; one-to-one or not; blocks never loaded; for `probe`, the smallest distance between two primitives' keys.
- **Compute per pass:** multiply-adds for one state in a two-token world at inference, counting only blocks that are loaded when the load is hard. At P = 6 and matched parameters: `mono` 2,400; `tok_mono` 4,880; `tok_streams` about 2,280; `wmix` about 2,460 (mixing the matrices costs what computing the streams does when the gate depends on the state); `hard` about 570; `lookup` about 570. Keys computed from weights do not depend on the state and are computed once per forward call.
- **Hardening after training:** every soft per-token arm is also evaluated with a top-1 load, no retraining.

## 6. Contrasts and the readings fixed in advance

Ratios of errors over the same splits; median, bootstrap 5–95 %, splits won. A single change is **worse** if the median held-out ratio to `tok_streams` is at least 2 and it loses in at least 9 of 12 splits; **not worse** if the upper bootstrap bound is below 2; otherwise **unresolved**. The same for seen error. The noise floor is the same arm with another initialisation (0.94–1.05 in Entry 108).

| Reading | Condition | What follows |
|---|---|---|
| **Premise shown** (A) | `wmix` worse than `tok_streams` on `select`, or on `sum` at width 36 | naive weight mixing is fragile here; the lookup is the weight-space form that is not |
| **Premise not shown** | `wmix` not worse on both | on this carrier nothing is wrong with gate-and-sum in weight space; the case for the lookup rests on cost alone |
| **Ingredient survives** (B) | T2, T3 (or T3s), T4 each not worse on held-out and seen error, no more failed runs than `tok_streams`, allocation one-to-one as often | that ingredient is kept |
| **Ingredient costs accuracy** | worse on either error | it is dropped, or kept only in the form that was not worse (T3s for T3; hardening after training for T4) |
| **Adopt the lookup** (C) | L not worse than `tok_streams` on both errors at P = 6 and at the larger banks, at a third of its compute or less | the lookup becomes the stream configuration for what comes next |
| **Adopt in part** | L worse, but some ingredients survive | `tok_streams` plus the surviving ingredients |
| **Against the field** | L / `tok_mono`, L / `streams_self`, L / `mono` on held-out error, as in Entry 108 | reported; expected to repeat the read effect (factor 10–20 against summed-read arms) and say nothing new |

A lookup that *beats* `tok_streams` by a factor of two on held-out error is not expected and would be treated as a surprise to check (a bug or a tuning asymmetry first), not as a result.

## 7. Stages and stop rules

- **Stage 0 — identities and reproduction.** Tests: `hard` and `lookup` with stored keys and a separate bid net reduce to `tok_streams` exactly (U4); `wmix` equals `tok_streams` at one-hot gates (W1). Re-run `tok_streams` and `tok_mono` on three splits and require the stored Entry 108 numbers to the last digit; only then are stored results reused for the old arms. Stop: any mismatch.
- **Stage 1 — the star at P = 6.** `select` at coverage 9 and 7, `sum` at coverage 9; twelve splits (three teachers); all new arms. Answers A, B, C.
- **Stage 2 — the bank grows.** P = 12 and 24 on `select`; arms `mono`, `tok_mono`, `streams_self`, `tok_streams`, `lookup`, and any single change that was worse or unresolved in stage 1; six splits. Answers D. Stop rule: if L was worse at P = 6, it is replaced by `tok_streams` plus the surviving ingredients.
- **Stage 3 — mechanism**, on stage 1 and 2 models: allocation over training (does the address or the fit move first), merging, key geometry, the effect of hardening.

## 8. Tuning, seeds, statistics

Each new arm's learning rate on splits 100–102 by seen error, from {0.003, 0.01, 0.03, 0.1}, extended until the best is interior (L57); the probe arms also choose between raw and bank-standardised key features. Every contrast is also reported at every pairing of rates whose seen error is within 2× of the best. Twelve splits at P = 6, six at P = 12 and 24; AGENTS.md asks for more seeds and a full bootstrap before any headline claim, so everything stays EXPLORATORY. Each script is committed with its predictions before it runs (L56). The results are reviewed by an agent that wrote none of this.

## 9. Predictions (to be restated in the scripts)

| | Prediction | Confidence |
|---|---|---|
| P1 | `wmix` on `select`: seen error at least 3× `tok_streams` | 0.65 |
| P2 | `wmix` on `sum`, width 36: seen error at least 3× `tok_streams` | 0.6 |
| P3 | `selfbid` not worse on held-out error | 0.6 |
| P4 | `probe` not worse on held-out error; `probe_sg` not worse | 0.5; 0.65 |
| P5 | Under `probe`, in at least one split of twelve two tokens load the same block | 0.5 |
| P6 | `hard` not worse on held-out error | 0.6 |
| P7 | `lookup` not worse than `tok_streams` on held-out error at P = 6 | 0.45 |
| P8 | Hardening after training changes the held-out error of `tok_streams` by less than 10 % | 0.7 |
| P9 | At P = 24, `tok_streams` still recombines (held-out error at most half the blend reference); `mono` is garbage | 0.6; 0.9 |
| P10 | At P = 24, `lookup` has more blocks never loaded than `tok_streams` | 0.6 |

## 10. What this cannot show

- **An accuracy advantage for the lookup over `tok_streams`.** Section 2. The setting in which computed keys could do something stored keys cannot is one where an item is *evidence* about a primitive (an example of its behaviour) instead of a name; that is a different carrier and is not designed here.
- **Anything about several sites.** The block has one. Retrieving sites from different blocks (U5) matters only when a task factors across sites, and on a carrier built to factor that way the outcome is fixed by construction.
- **Where items come from** when the input does not itemise them (Entry 109, point 6). Unchanged.
- The carrier's limits from Entry 108 stand: candidates are named, the `select` teacher has the shape of the per-token arms, a stream contains a teacher primitive exactly.

## 11. Cost

A run takes 15–30 s at P = 6. Stage 1 is about 13 arms × 12 splits × 3 conditions plus tuning: under four hours on two processes. Stage 2 is larger per run (four to eight times) and smaller in runs: about three hours.
