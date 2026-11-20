# Context-dependent weights against static weights — proposed contribution and design

**Status: DESIGN, revision 5, nothing run by the author. EXPLORATORY.** Written 20 Nov 2026. Revision 4 (the researcher's request of 18 Nov: "given our framing, what is our proposed contribution and experimental design?") was reviewed by an agent that did not write it and found **not sound as written** (`review/review_design_rev4.md`): of its three claims one was a replication, one could not come out in its favour on this carrier, and one had closer prior art than I had found. This revision takes the review's corrections and its two proposals. Earlier reviews: `review/review_design.md`, `review/review_design_rev2.md`. Section 11 lists what changed. History Entries 119–121.

## 1. The question

The researcher's (Entry 117): **how do weights that depend on context compare with static weights** — "weight modification based on context itself, … expanding the space that a model stores information in by doing this weight lookup using context of the input and state"; and (Entry 120) that a looped block of this kind, conditioned on its own past, could correct errors and find paths.

## 2. What is known, and so is not ours

| Claim | Where it stands |
|---|---|
| A bank of units retrieved by key stores more at fixed compute | PEER; product-key memory. Not re-measured. |
| Composing weights by attention scores | Multi-head attention across heads (Schug et al. 2025). |
| Averaging weights is exact only where the output is linear in them | The modular-learning survey; our W7–W9 restate it. |
| Context that multiplies beats context that is added, for fit | Our Entry 105 (80–700×); the subject of Jayakumar et al. 2020 (unread). The line between "static" and "context-dependent" weights is this line: a static block whose units are gated by context is, unit for unit, the lookup (U1). |
| Binding learned functions to context items recombines | Neural Production Systems, Neural Interpreters; Entry 108. |
| A fixed way of composing, with no learnable interaction between the parts, generalises to unseen combinations given every part and connected coverage | Wiedemer et al. 2023; Lachapelle et al. 2023 (additive decoders); Schug et al. 2024. Read to their main results only. |
| Choosing among modules by how well each predicts what is observed, trained from scratch | MOSAIC (Haruno, Wolpert & Kawato 1998); modular meta-learning (Alet et al. 2018), which scores the *applied* modules. Fetched by the reviewer, not by me. Jacobs et al. 1991 unread. |
| Experts inside a loop at scale | LoopMoE, Tying the Loop, MoEUT. |

## 3. Proposed contribution

Smaller than revision 4 claimed, and in three parts of unequal standing.

**A. What binding is — derived, to be measured.** Entry 108 found that reading the context once per item recombines and a pooled gate does not, and Entry 109 called this "binding, not routing". The count of what training leaves free says something sharper (tests B0–B4, `tests/test_binding.py`, exact constructions on the teacher):

- In a pooled gate — one softmax over blocks of (excitation from item a) + (excitation from item b) + bid — how much item b excites the block of a enters trained worlds only inside sums that training wants small. So **every held-out pair's decision carries an offset that no trained world sees** (B1). This is a memorising solution in the gate, present at any number of blocks.
- **Normalising each item's address before the items meet removes it** (B2), whether the items then compete or share one softmax. What is left is one bid offset per primitive, and trained worlds that are connected fix those up to a common constant (B3); if the trained worlds fall into two groups the offset between groups is free again (B4).

So the active part of "per item" is the normalisation of each item's address; routing with a normalised address binds; and capacity is irrelevant in both directions. This is an instance of the known principle in section 2 (a composition with no learnable cross terms), not a new theory. What would be ours: the account of *which* part of a learned gate over a bank carries the cross terms, a correction of our own Entry 109, and the measurement. The reviewer's one-seed probes agree (pooled: 0.38–0.45 held-out at 6, 8 and 24 blocks; pooled with normalised loads: 0.056; fitted offsets at most 0.12 trigger units on trained worlds and −1.28 to +1.14 on held-out ones). Prior statements of exactly this have not been searched for.

**B. The applied blocks as the address — a mechanism test, with a narrow claim.** Retrieval by running the bank's blocks on the context's demonstrations and comparing. Given section 2, what is left to claim is narrow: a differentiable soft match of the *applied* blocks against in-context demonstrations, as the address inside a looped block; and the open empirical question the review judged strongest, **whether a bank whose address reads its own unfinished contents sorts itself** (a 2025 paper names this as an unsolved instability). The earlier one-seed probes found that tying the matcher to the applied blocks adds nothing during training; in this part the tie has no job.

**C. Closing the loop — where the tie and the loop get a job. Not designed here.** The researcher's Entry 120. A stored block that can be run on a state and compared with a goal is a forward model of its own move; there the function that is applied and the function that is tested must be the same, and a separate matcher would be a second model that can drift. That is the first place in this program where the tie, and the number of passes, could change an outcome. Section 9 records what a carrier for it needs.

**On static against context-dependent weights, plainly.** On this carrier that contrast is additive against multiplicative conditioning, which is known for fit. One cell is open and kept: what each does when the *code* for an item is off its training distribution — a static block is an unconstrained function of an unfamiliar vector; a selecting bank can only apply a stored primitive, possibly the wrong one.

**Not addressed.** "Expanding the space" as storage — how many primitives at a fixed active width. The number of primitives is six throughout; that question is PEER's.

## 4. Carriers

- **Names** (Entry 108, `worlds.py`): six primitives tanh(A_p h), a trigger each, a world is two primitives named by tokens, four passes, `select` and `sum`, nine of fifteen worlds trained. Stage A runs here, on existing code plus one arm.
- **Evidence** (revision 3): the same, each primitive shown by m = 2 demonstrations (x, f_p(x)), one draw per (world, state). Test conditions fixed in advance: held-out worlds; evidence at 1.5, 2 and 3 times the training scale (2× the pre-named point); m = 1, 4 and noise 0.1 reported only.

Limits, stated now: the `select` teacher has the shape of the per-item arms; a bank block contains a teacher primitive exactly (so every fit contrast against a generic block carries that, and shifted-evidence tests are tilted towards arms that match with blocks); the context arrives already split into two items; no reading depends on the number of passes.

## 5. Measures and the rule for a reading

- Seen and held-out error; regime against the exact references of Entry 108.
- **At the level of the decision:** on held-out worlds, the share of decisions not forced by trained comparisons that match the teacher's (code exists from worlds stage 3). Fit contrasts are also read at matched seen error.
- **The offsets themselves** (stage A): for each world, the trigger offset at which the trained gate switches between the two primitives' blocks, on trained and on held-out worlds.
- **Identification by function:** each block is matched after training to the primitive it computes; an item is identified by the share of its load on blocks that compute its primitive (so duplicates do not count as failure). For an arm without blocks: whether its update is nearer the right primitive's than any other's.
- Failed runs; parameters; multiply-adds per pass and per world.

Rule (revision 3, section 5): twelve splits on twelve teachers; the noise envelope from three other initialisations of the contrast's reference arm (named per stage); *worse* / *better* only if the bootstrap interval of the paired ratio lies outside the envelope at equal rates and at every pairing of rates within 2× of the best seen error; *not worse* needs an upper bound below 1.5; otherwise *unresolved*. Rates and step counts tuned per arm family on splits 100–102, interior (L57). Scripts committed with predictions before running (L56); results reviewed independently.

## 6. Stages

**Stage 0.** Forward identities as tests; `tok_streams` and `tok_mono` against stored numbers; `tok_streams` on the twelve teachers.

**Stage A — what binding is** (names carrier; about two hours with tuning). Reference arm: `tok_streams`.

| Arm | Gate |
|---|---|
| `streams_self` | pooled: one softmax over blocks of excitation_a + excitation_b + bid |
| `pooled_norm` (new) | one softmax over blocks of log(load_a + load_b) + bid, each load a softmax over blocks |
| `tok_streams` | per item: loads normalised, items compete by bids |

At 6 and 24 blocks; `select` and `sum`; the pooled arm also with a sigmoid gate.

| Reading | Condition |
|---|---|
| Account supported | `pooled_norm` recombines (held-out error at most half the blend reference, and not worse than `tok_streams`); `streams_self` does not at either size; held-out offsets of `streams_self` at least five times its trained offsets, and those of `pooled_norm` not |
| Capacity matters after all | `streams_self` recombines at 6 blocks and not at 24, or the reverse |
| Normalisation is not the active part | `pooled_norm` does not recombine |
| Offsets are not the mechanism | `streams_self` fails with held-out offsets no larger than trained ones |

**Stage B — the address, on evidence** (revision 3's stages E0–E1, as reviewed, with two static arms). Reference arm: `bank_item`.

| Arm | What it is |
|---|---|
| `bank_item` (two encoder sizes) | per item; learned encoder against stored keys |
| `bank_match_untied` | per item; a second set of blocks matched against the demonstrations |
| `bank_match` | per item; the applied blocks matched, −‖c·f_k(x) − y‖²/τ; also with the gradient from load to weights cut |
| `static_item` | one shared two-layer block run on [state; code] per item, pooled by bids |
| `static_item_gated` | the same block with the code entering as a gate per hidden unit |
| `static_pooled`, `oracle`, and the names ceilings (`tok_streams` for banks, `tok_mono` for static arms) | |

All bank arms share blocks and bids. Run first: the encoder arms alone, at 4,000, 8,000 and 16,000 steps (the encoder arm was under-trained at 4,000 in the earlier probe).

| Reading | Condition |
|---|---|
| The bank sorts itself | `bank_match` not worse than `bank_item` on seen error, every primitive computed by some block in at least 11 of 12 runs; when it sorts, read from identification over training |
| It does not bootstrap | worse on seen error, or a primitive without a block in more than 3 of 12 runs |
| Matching beats a learned encoder under shift | `bank_match_untied` better than `bank_item` and than its wide form on error at 2× evidence |
| The tie adds nothing here | `bank_match` not better than `bank_match_untied` (expected) |
| An encoder suffices | bank arms mutually not worse in every condition |
| Static against bank, off distribution | `static_item` and `static_item_gated` against `bank_item` (same encoder) at 2× evidence, beside the reference "applies one wrong primitive"; read at the decision level |

**Stage C — adding a primitive after training** (`sum` without bids: a new forward pass, with its own identity tests). Train on worlds over primitives 1–5 with the usual coverage. Adaptation data for every arm: 256 demonstrations of primitive 6, no worlds, no old data. Read at 500 adaptation steps (pre-named) and as a curve.

| Arm | What is trained |
|---|---|
| `bank_match` | one new block, by regression on the demonstrations |
| `bank_item`, frozen encoder | the new block; a new key by cross-entropy that primitive-6 evidence loads it |
| **`bank_item`, trainable encoder, replay from the bank** | the same, with the encoder trained and old primitives' evidence regenerated by running the stored blocks |
| `static_item` | everything, on the demonstrations |

Reference: the same arm trained from the start on six primitives with the same coverage rule. Measures: error on the five new worlds, error on the old worlds.

Named in advance as arithmetic or expected: `bank_match`'s new-world error is the new block's regression error; `static_item` without rehearsal forgets; an untied matcher would copy the new block and equal the tied one. The comparison that carries information is `bank_match` against the trainable encoder with replay (the reviewer's one-seed probe: 8e-6 against 0.21 on new worlds; old worlds unchanged against 0.086).

**Stage D — mechanism.** Identification and block-to-primitive match over training for the match arms; what the encoder confuses under shift.

## 7. Predictions

Not blind: both reviews ran one-seed probes of most arms; their numbers are in the review files and above.

| | Prediction | Confidence |
|---|---|---|
| P1 | Stage A reads "account supported" on `select` | 0.65 |
| P2 | On `sum` with a sigmoid gate the pooled arm recombines at 6 blocks (the review's 0.164 against a bar of 0.266), so the account needs a statement for gates that do not compete | 0.5 |
| P3 | `bank_match` sorts itself: every primitive has a block in at least 11 of 12 runs | 0.7 |
| P4 | `bank_match_untied` better than `bank_item` at 2× evidence | 0.7 |
| P5 | `bank_match` not better than `bank_match_untied` anywhere in stage B | 0.75 |
| P6 | At 2× evidence `static_item` is worse than `bank_item` with the same encoder | 0.5 |
| P7 | `static_item_gated` not worse than `bank_item` on seen error at equal parameters | 0.4 |
| P8 | Stage C: `bank_item` with a trainable encoder and replay is worse than `bank_match` on new-world error at 500 steps | 0.65 |

## 8. What this would and would not establish

Would: which part of a learned gate over a bank decides whether unseen combinations work (A); whether a bank can be addressed by its own contents from scratch, and what that buys under shifted evidence and when a primitive is added (B, C). All on one small synthetic carrier, a feed-forward block with one site, twelve splits: EXPLORATORY.

Would not: a new theory (A is an instance of known results); a new mechanism (B has thirty years of relatives); anything at language-model scale; anything about attention sublayers; storage at fixed compute; error correction or path finding (section 9). Its use to the program is to decide what a lookup block should look like before one is built into a transformer block on the rewrite carrier — normalised per-item address or not, stored keys or matched blocks — and to set up part C.

## 9. Closing the loop: what a design for part C needs (recorded; Entry 120 and the review's finding 9)

- **A signal of error readable in the forward pass.** Distance to a goal the context gives is enough and needs no evidence-matched address; the match score is a second candidate. "The state has stopped changing" is not one (a wrong fixed point is still a fixed point).
- **A choice that has to be revisited.** In this design the load is fixed per world and the bids already re-read the state on every pass, which is the simplest closed loop; nothing is ever wrong and correctable.
- **A carrier where order matters.** At the present step sizes a sequence of primitives is nearly a bag of them: reversing a sequence changes the result by 0.06–0.20 of the total change, swapping one primitive by 0.26–0.40 (exact, from the teacher; the reviewer's computation). The step must be near 1, or the primitives must not nearly commute.
- **Errors in training.** A loop learns to correct only if it is trained from states that contain errors.
- **The tie's job.** Each stored block as the forward model of its own move: run candidate blocks on the current state, compare with the goal, apply the best. Relatives to read first: MOSAIC; modular meta-learning; recurrent networks with recall on mazes; the 2025 recursive reasoning models.

Order: after stages A and B, for cost; the logical dependence on B is weak (only that stored blocks can be run and compared with something in context).

## 10. Cost

Runs are 15–30 s per 4,000 steps at eight blocks and about 70 s at twenty-four. With tuning per arm family: stage A about two hours on two processes, stage B about eight, stage C about four. Roughly fourteen hours in all.

## 11. What changed from revision 4, by review finding

| Finding | Change |
|---|---|
| 1 the "weights" factor bundled four changes; nothing held out at its level | The 2 × 2 is dropped. Static against bank is one cell (off-distribution codes), with a unit-gated static arm, decision-level and matched-seen-error readings |
| 2 the capacity stage did not instantiate the theorem and could not come out for C2 | C2 and its stage are dropped. Part A replaces them: derived (B0–B4), three arms, capacity at two sizes as a check |
| 3 adding a primitive was under-specified and could not separate the tie | Specified (data, losses, reference, the step at which it is read); arithmetic named; trainable encoder with replay added as the comparison that matters; no claim for the tie |
| 4 stage 1's stop rule could not fire; identification undefined for static arms | Stage 1 folded into stage B; identification defined for arms without blocks; static arms get `tok_mono` on names as their ceiling |
| 5 novelty overstated; thesis not well posed | Thesis withdrawn. Prior art added to section 2; claims narrowed (section 3) |
| 6 two readings foregone; matching arithmetic | Removed with the 2 × 2; reference arm named per stage |
| 7 "expanding the space" not measured | Said so (section 3) |
| 8 cost low by three times | Re-estimated |
| 9 section 13 | Rewritten as section 9 with the corrections (signal need not come from the architecture; "four passes could be one" withdrawn; order is by cost; step size) |
