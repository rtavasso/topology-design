# Context-dependent weights against static weights — proposed contribution and design

**Status: DESIGN, revision 6, nothing run by the author. EXPLORATORY.** Written 20 Nov 2026. Reviewed four times by an agent that did not write it (`review/`): revisions 1 and 4 "not sound as written", revisions 2 and 5 "sound with corrections". This revision applies the corrections to revision 5; section 11 lists what changed and why. The researcher's request (18 Nov): "given our framing, what is our proposed contribution and experimental design?" History Entries 119–122.

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
| Choosing among modules by how well each predicts what is observed, trained from scratch | MOSAIC (Haruno, Wolpert & Kawato 1998), which pairs a separate predictor with each controller; modular meta-learning (Alet et al. 2018), which scores the *applied* modules. Fetched by the reviewer, not by me. Jacobs et al. 1991 unread. |
| Experts inside a loop at scale | LoopMoE, Tying the Loop, MoEUT. |

## 3. Proposed contribution

In three parts of unequal standing.

**A. What binding is — derived in part, to be measured.** Entry 108 found that reading the context once per item recombines and a pooled gate does not; Entry 109 called this "binding, not routing". Counting what training leaves free says something sharper. Exact constructions on the teacher, `tests/test_binding.py`:

- *Pooled gate* — one softmax over blocks of (excitation from item a) + (excitation from item b) + bid. How much item a excites the block of b appears in trained worlds only where that block must stay off, so **trained worlds bound each cross term from above and nothing else**. A held-out pair's decision turns on the difference of two cross terms, so it is free (B1: lowering one cross term changes no trained world at any margin and breaks exactly one held-out world). This freedom is in the gate and is there at any number of blocks.
- *Normalised address* — each item's load is a softmax over blocks before the items meet. A load is then a function of one item, and every item is trained; with saturated loads the cross terms have no effect anywhere (B2), whether the items then compete or share one softmax. So **the pair-specific freedom is gone**.
- *What normalisation does not give.* What remains is a bid per primitive, and a bid is a free function of the state. A trained world fixes only the sign of a difference of two bids. If bids have the teacher's form, connected trained worlds fix them up to a constant (B3), and disconnected ones do not (B4). In general, **held-out decisions are pinned only where trained comparisons chain** (B5: a bid change that keeps every trained decision and breaks any chosen held-out world). Transfer beyond the chains is not derived; it is what Entry 108 measured (0.93–0.95 of unforced decisions).

So: the active part of "per item" is the normalisation of each item's address, and routing with a normalised address binds. This is an instance of the known principle in section 2 (a composition with no learnable cross terms), not a new theory. What would be ours is the account of which part of a learned gate over a bank carries the freedom, the correction of Entry 109, and the measurement.

One-seed indications from the reviews (not readings): pooled 0.38–0.45 held-out at 6, 8 and 24 blocks; pooled with normalised loads 0.056, the per-item arm's number; and a causal check on the trained pooled arm — removing its fitted offset takes held-out error from 0.454 to 0.272, masking blocks that belong to neither item to 0.325, both to 0.093, with seen error unchanged. So there are **two routes**: the offset between the two members' blocks, and weight leaking onto third blocks. That capacity does not matter is, so far, an observation at one seed; only the gate-level freedom at every size is derived. Prior statements of exactly this have not been searched for.

**B. The applied blocks as the address — a mechanism test, with a narrow claim.** Retrieval by running the bank's blocks on the context's demonstrations and comparing. Given section 2, what is left to claim is a differentiable soft match of the *applied* blocks against in-context demonstrations as the address inside a looped block, and the empirical question the review judged strongest: **whether a bank whose address reads its own unfinished contents sorts itself** (a 2025 paper names this as a risk). Earlier one-seed probes found that tying the matcher to the applied blocks adds nothing during training; in this part the tie has no job.

**C. Closing the loop — where the tie and the loop could get a job. Not designed here.** The researcher's Entry 120. A stored block that can be run on a state and compared with a goal is a forward model of its own move. One stored block can then be both the move and its predictor; MOSAIC keeps those as two paired models, so the tie is an economy — one model in place of two that must be kept in step — and not a necessity. It is the first place in this program where the tie, and the number of passes, could change an outcome. Section 9 records what a carrier for it needs.

**On static against context-dependent weights, plainly.** On this carrier that contrast is additive against multiplicative conditioning, known for fit. One cell is open and kept: what each does when the *code* for an item is off its training distribution — a static block is an unconstrained function of an unfamiliar vector; a selecting bank can only apply a stored primitive, possibly the wrong one.

**Not addressed.** "Expanding the space" as storage — how many primitives at a fixed active width. The number of primitives is six throughout; that question is PEER's.

## 4. Carriers

- **Names** (Entry 108, `worlds.py`): six primitives tanh(A_p h), a trigger each, a world is two primitives named by tokens, four passes, `select` and `sum`, nine of fifteen worlds trained. Stage A runs here, on existing code plus one arm.
- **Evidence** (revision 3): the same, each primitive shown by m = 2 demonstrations (x, f_p(x)), one draw per (world, state). Test conditions fixed in advance: held-out worlds; evidence at 1.5, 2 and 3 times the training scale (2× the pre-named point); m = 1, 4 and noise 0.1 reported only.

Limits, stated now: the `select` teacher has the shape of the per-item arms; a bank block contains a teacher primitive exactly (every fit contrast against a generic block carries that, and shifted-evidence tests are tilted towards arms that match with blocks); the context arrives already split into two items; no reading depends on the number of passes.

## 5. Measures and the rule for a reading

- Seen and held-out error; regime against the exact references of Entry 108. On `sum` "recombines" means the *full* regime: held-out error under half the smaller reference **and** within three times the arm's own seen error.
- **At the level of the decision** (held-out worlds, `select`): the share of decisions not forced by trained comparisons that match the teacher's. Fit contrasts are also read at matched seen error.
- **Stage A's own:** for each world, the trigger offset at which the trained gate switches between the two members' blocks; the gate mass on blocks that belong to neither item (leak); and the held-out error after each of three removals on the trained model — offset removed, non-member blocks masked, both.
- **Identification, one statistic for every arm:** for an item, take the update the arm attributes to it (a bank: its load-weighted block outputs; a static per-item arm: that branch's output) and ask whether it points, up to scale, nearest the right primitive's update. This counts a primitive split over several blocks as found, and is defined for arms without blocks.
- Failed runs; parameters; multiply-adds per pass and per world.

Rule (revision 3, section 5): twelve splits on twelve teachers; the noise envelope from three other initialisations of the stage's reference arm; *worse* / *better* only if the bootstrap interval of the paired ratio lies outside the envelope at equal rates and at every pairing of rates within 2× of the best seen error; *not worse* needs an upper bound below 1.5; otherwise *unresolved*. Where an arm sits at numerical precision (stage C), absolute criteria replace ratios. Rates and step counts tuned per arm family on splits 100–102, interior (L57), and checked at each block count used. Scripts committed with predictions before running (L56); results reviewed independently.

## 6. Stages

**Stage 0.** Forward identities as tests; `tok_streams` and `tok_mono` against stored numbers; `tok_streams` on the twelve teachers.

**Stage A — what binding is** (names carrier; about three hours with tuning). Reference arm: `tok_streams`.

| Arm | Gate |
|---|---|
| `streams_self` | pooled: one softmax over blocks of excitation_a + excitation_b + bid; also with a sigmoid in place of the softmax |
| `pooled_norm` (new) | one softmax over blocks of log(load_a + load_b) + bid, each load a softmax over blocks, the log computed without a guard |
| `tok_streams` | per item: loads normalised, items compete by bids |

At 6 and 24 blocks; `select` and `sum`.

What the account predicts, written before the run. Softmax pooled gate: fails through both routes. Sigmoid pooled gate: no competition between blocks, so no offset between the two members; leak remains; on `sum` it should fit and not reach the full regime. `pooled_norm` with saturated loads is the per-item arm in the limit, so its recombining is expected by construction once the loads saturate — the measured part is whether training saturates them.

| Reading | Condition |
|---|---|
| Account supported | `streams_self` does not recombine at either size; its held-out offsets are at least five times its trained ones **or** its leak on held-out worlds is at least five times that on trained worlds; and on the trained model the two removals together bring held-out error to at most half the blend reference with seen error unchanged |
| Normalisation suffices | `pooled_norm` recombines |
| …and costs nothing | `pooled_norm` not worse than `tok_streams` on held-out error (read separately; one seed gave 0.083 against 0.045 at 24 blocks) |
| Capacity matters after all | `streams_self` recombines at one size and not the other |
| Normalisation is not the active part | `pooled_norm` does not recombine although its loads saturate |
| The account misses the mechanism | `streams_self` fails and the two removals together leave it above half the blend reference |

**Stage B — the address, on evidence** (revision 3's stages E0–E1 as reviewed, with two static arms). `select` for all arms; `sum` for the bank arms. Reference arm: `bank_item`.

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
| The bank sorts itself | `bank_match` not worse than `bank_item` on seen error, and every primitive identified at 0.95 or better in at least 11 of 12 runs; when it sorts, from identification over training |
| Partly | a primitive below that in 2 or 3 of 12 runs: reported as a failure rate |
| It does not bootstrap | worse on seen error, or more than 3 of 12 |
| Matching beats a learned encoder under shift | `bank_match_untied` better than `bank_item` and than its wide form on trained-world error at 2× evidence |
| The tie adds nothing here | `bank_match` not better than `bank_match_untied` (expected) |
| An encoder suffices | bank arms mutually not worse in every condition |
| Selection fails gracefully off distribution | on trained worlds at 2× evidence, `bank_item` is at or below the reference "applies one wrong primitive" and `static_item` (same encoder) is worse than `bank_item` |
| …and it is selection, not multiplication | `static_item_gated` is worse than `bank_item` there as `static_item` is |
| No difference off distribution | `static_item` not worse than `bank_item` at 2× evidence |

**Stage C — adding a primitive after training** (`sum` without bids: a new forward pass with its own identity tests). Train on seven of the ten worlds over primitives 1–5 (2-connected, not bipartite). Adaptation data for every arm: 256 demonstration pairs of primitive 6, no worlds, no old data. Adaptation at rate 0.01, read at 500 steps (pre-named) and as a curve. Reference and noise envelope: `bank_match`.

| Arm | What is trained, on what |
|---|---|
| `bank_match` | one new block, regressed on the 256 pairs |
| `bank_item`, frozen encoder, positives only | the new block; a new key by cross-entropy that primitive-6 evidence loads the new block |
| `bank_item`, frozen encoder, replay | the same, with old primitives' evidence regenerated by running every stored block on fresh states, each labelled with its block |
| `bank_item`, trainable encoder, replay | the same, the encoder trained too |
| `static_item` | everything; its branch, given primitive-6 evidence drawn from the pairs and a state x_i from another pair, regressed on y_i |

Readings, in absolute terms: an arm **extends** if, at 500 steps, new items and old items are both identified at 0.95 or better and the five new worlds are in the full regime; **extends with damage** if new items are identified and old ones fall below 0.95; **does not extend** otherwise. The comparison is which arms extend, and at how many steps each first does.

Named in advance as arithmetic or expected: `bank_match`'s new-world error is the new block's regression error (about 1e-4 at this rate and step count); `static_item` without rehearsal forgets; a positives-only key captures old evidence; a replay-trained encoder converges given enough steps, so that row measures steps and damage along the way; an untied matcher would copy the new block and equal the tied one, so this stage says nothing for the tie.

**Stage D — mechanism.** Identification over training for the match arms; what the encoder confuses under shift.

## 7. Predictions

Not blind: the reviews ran one-seed probes of most arms; their numbers are in the review files and in section 3.

| | Prediction | Confidence |
|---|---|---|
| P1 | Stage A reads "account supported" on `select` at both sizes | 0.65 |
| P2 | Sigmoid pooled gate on `sum`: fits, is not in the full regime, and its failure is leak (masking non-members restores it) | 0.55 |
| P3 | `pooled_norm` recombines on `select` and `sum`; at 24 blocks it is worse than `tok_streams` | 0.7; 0.5 |
| P4 | `bank_match` sorts itself in at least 11 of 12 runs | 0.7 |
| P5 | `bank_match_untied` better than `bank_item` at 2× evidence | 0.7 |
| P6 | `bank_match` not better than `bank_match_untied` anywhere in stage B | 0.75 |
| P7 | At 2× evidence `static_item` is worse than `bank_item` with the same encoder | 0.5 |
| P8 | `static_item_gated` degrades off distribution as `static_item` does (the property belongs to normalised selection) | 0.55 |
| P9 | Stage C: at 500 steps only `bank_match` extends; the trainable encoder with replay extends with damage or not at all | 0.6 |

## 8. What this would and would not establish

Would: which part of a learned gate over a bank decides whether unseen combinations work, and by which of two routes a pooled gate fails (A); whether a bank can be addressed by its own contents from scratch, and what that buys under shifted evidence and when a primitive is added (B, C). All on one small synthetic carrier, a feed-forward block with one site, twelve splits: EXPLORATORY.

Would not: a new theory (A is an instance of known results); a new mechanism (B has thirty years of relatives); anything at language-model scale; anything about attention sublayers; storage at fixed compute; error correction or path finding (section 9). Its use to the program is to decide what a lookup block should look like before one is built into a transformer block on the rewrite carrier — normalised per-item address or not, stored keys or matched blocks — and to set up part C.

## 9. Closing the loop: what a design for part C needs (recorded; Entry 120 and the reviews)

- **A signal of error readable in the forward pass.** Distance to a goal the context gives is enough and needs no evidence-matched address; the match score is a second candidate. "The state has stopped changing" is not one (a wrong fixed point is still a fixed point).
- **A choice that has to be revisited.** In this design the load is fixed per world and the bids already re-read the state on every pass, which is the simplest closed loop; nothing is ever wrong and correctable.
- **A carrier where order matters.** At the present step sizes a sequence of primitives is nearly a bag of them: reversing a sequence changes the result by 0.06–0.20 of the total change, swapping one primitive by 0.26–0.30 (exact, from the teacher; the reviewer's computation). The step must be near 1, or the primitives must not nearly commute.
- **Errors in training.** A loop learns to correct only if it is trained from states that contain errors.
- **The tie's job.** Each stored block as the forward model of its own move: run candidate blocks on the current state, compare with the goal, apply the best. Relatives to read first: MOSAIC; modular meta-learning; recurrent networks with recall on mazes; the 2025 recursive reasoning models.

Order: after stages A and B, for cost; the logical dependence on B is weak (only that stored blocks can be run and compared with something in context).

## 10. Cost

Runs are 15–30 s per 4,000 steps at eight blocks and about 70 s at twenty-four. With tuning per arm family: stage A about three hours on two processes, stage B about eight, stage C about four. Roughly fifteen hours in all.

## 11. What changed, by review finding

From revision 4 (`review_design_rev4.md`, "not sound as written"):

| Finding | Change |
|---|---|
| 1 the "weights" factor bundled four changes; nothing held out at its level | The 2 × 2 is dropped. Static against bank is one cell (off-distribution codes), with a unit-gated static arm |
| 2 the capacity stage did not instantiate the theorem and could not come out for C2 | C2 and its stage dropped; part A replaces them |
| 3 adding a primitive under-specified; could not separate the tie | Specified; arithmetic named; no claim for the tie |
| 4 stage 1's stop rule could not fire; identification undefined for static arms | Folded into stage B; one identification statistic for all arms; `tok_mono` as the static ceiling |
| 5 novelty overstated; thesis not well posed | Thesis withdrawn; prior art in section 2; claims narrowed |
| 6–9 foregone readings, storage not measured, cost, the closed-loop section | Removed; said; re-estimated; rewritten as section 9 |

From revision 5 (`review_design_rev5.md`, "sound with corrections"):

| Finding | Change |
|---|---|
| 1 "connected trained worlds fix the bids up to a constant" held only for bids of the teacher's form | Section 3A restated: normalisation removes the pair-specific freedom; held-out decisions are pinned only where trained comparisons chain; test B5 added |
| 2 B1 and B2 were exact only through a large margin | B1 made one-sided (bounded from above only); B2's condition (saturated loads) and the margin-free statement given; log-load without a guard |
| 3 stage A: a second route unmeasured; one reading by construction; no criterion for `sum` | Leak measured; the three removals added; by-construction part named; "not worse than `tok_streams`" split off; the full regime on `sum`; the sigmoid prediction written out |
| 4 stage C had no readings; three choices decided its outcome | Absolute readings; rate fixed; frozen-with-replay arm added; expectations named; replay, training worlds, the static arm's loss and the reference specified |
| 5 stage B: the static cell had no condition; identification failed for split primitives | Conditions and a "partly" row added; identification by the direction of the attributed update; tasks stated |
| 6 wording still overstated in places | "No trained world sees", "capacity is irrelevant", "unsolved instability", "must be the same" corrected; the causal check and second route added |
