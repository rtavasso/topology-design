**Verdict: sound with corrections.** Revision 5 resolves the four blocking findings of the last review in its text, and the design can be scripted for stages 0, A and B. One sentence of the contribution is wrong as a derived statement and must change before it is repeated anywhere (finding 1, blocking for that claim, not for any run). Stage C has lost its readings (finding 4). The rest are corrections to wording, measures and specification.

# Confirmation review of `experiments/theory/lookup/DESIGN.md`, revision 5 (commit 559130c), and `tests/test_binding.py`

Same reviewer as `review_design_rev4.md`. `tests/test_binding.py`: 5 passed in 3.4 s. Exact checks are my scratch re-implementation of the test's constructions with the margin and the bids varied (no training). Training probes are **one seed, one split** (split 0, teacher 10, seed 0, names carrier, `worlds.py`), four runs, about 2.5 minutes in total; indications only (L30).

## 1. Are the earlier findings resolved in the text?

| Finding of the rev-4 review | Resolved in the text? | What remains |
|---|---|---|
| 1 the "weights" factor bundled four changes (blocking) | **Yes.** The 2 × 2 is gone; static against bank is one cell on off-distribution codes, with `static_item_gated`; section 2 puts the static/context-dependent line at additive against multiplicative. | The cell's reading has no condition (finding 5). Bids and function class still differ between static and bank arms; section 4 says so. |
| 2 the capacity stage (blocking) | **Yes.** C2 and its stage are dropped; part A replaces them. | Findings 1–3 below, on part A itself. |
| 3 adding a primitive (blocking) | **Mostly.** Data, the step read, the key's loss, replay, the reference and the arithmetic are stated; the trainable encoder with replay is in; no claim for the tie. | No readings at all, and the ratio rule meets the precision trap; `static_item`'s loss is still not stated; four smaller gaps (finding 4). |
| 4 stage 1's stop rule (blocking) | **Yes.** Folded into stage B; `tok_mono` is the static ceiling; identification is defined for arms without blocks. | The definition needs "up to scale" and should be one statistic for all arms (finding 5). |
| 5 novelty and thesis | **Yes.** Thesis withdrawn; MOSAIC, modular meta-learning and the identifiability papers are in section 2 with read depth; claims narrowed. | Two phrases (finding 6). |
| 6 foregone readings, arithmetic, reference arm | **Yes** for A and B. | Stage C's contrasts have no named arm for the noise envelope. |
| 7 "expanding the space" | **Yes** (section 3, "Not addressed"). | — |
| 8 cost | **Yes** (fourteen hours). | Stage A is nearer three hours than two by my count (16 configurations × 12 splits, tuning of four gate families on two tasks, three nulls). Minor. |
| 9 closing the loop | **Yes**, with one number to correct (finding 7). | — |

## Findings

**1. BLOCKING for the claim in section 3A (not for any run) — "what is left is one bid offset per primitive, and trained worlds that are connected fix those up to a common constant (B3)" is true only for bids of the teacher's own form.**
- B3 as coded holds the bids at λ · (the true trigger) + a constant per primitive, and varies the constants. Inside that family the statement is right.
- The arms' bids are free functions of the state (one small net per block). A trained world {a, c} fixes only the **sign** of bid_a − bid_c at each state, not the difference. Signs along a path a–c–b decide a held-out pair {a, b} only where they chain (a over c over b). That is the repo's own `trans` reference, and it is the assumption — "my test had assumed linear triggers" — that the worlds design review corrected in Entry 108.
- *Exact construction* (scratch `p5.py`; the test's exact bank, both normalised gates, split 0). Add a large constant to the bid of every primitive that beats b through a chain of trained comparisons at the current state. Every trained comparison keeps its sign. Result: trained-world error at most 2.2e-3 (the ideal setting's own floor; never above it), while the targeted held-out world's error is 0.31–1.04 in all twelve cases (six held-out worlds, either member targeted), identically for the per-item and the normalised pooled gate. The share of first-pass decisions that trained chains force is 0.18–0.64 per held-out world.
- So normalisation does not make held-out decisions **determined**. It removes the pair-specific parameters. What remains free is one bid *function* per primitive, shared by every world that primitive is in, and trained worlds pin it only on the transitive closure. That the rest transfers (0.93–0.95 of unforced held-out decisions in Entry 108) is the bid nets' inductive bias — measured, not derived.
- B4 has the same scope (constants over the true triggers). It is correct as far as it goes.
- *Fix.* Reword the bullet: "normalisation removes the freedom that belongs to a pair; what is left belongs to a primitive (its bid, a function of the state) and is shared across that primitive's trained worlds; trained worlds fix it only where comparisons chain; beyond that, transfer is empirical". Add the construction above as a test (B5). In stage A's "account supported", say which half is derived (the pair-specific freedom and its removal) and which is measured (that the per-primitive remainder generalises), and read the decision-level measure of section 5 separately on forced and unforced decisions.

**2. SHOULD FIX — B1 and B2 are exact because of the margin built into the construction; the statements that hold without it are different, and stronger.**

*B1.* The coordinator's question: is the perturbation invisible to trained worlds only because of the self-excitation margin? For half of it, yes. Exact, pooled gate, perturbation of 12,000 against bids spanning about ±7,300:

| Self-excitation | Both (as in the test): largest change in a trained world | Raise u[a, b] only | Lower u[b, a] only | Targeted held-out world's error (all three) |
|---|---|---|---|---|
| 1e6 (the test) | 0 | 0 | 0 | 0.89–1.23 |
| 40,000 | 0 | 0 | 0 | 0.89–1.23 |
| 20,000 | 2.4e-3 | 2.4e-3 | 0 | 0.89–1.23 |
| 12,000 | 0.83 | 0.83 | 0 | 0.89–1.23 |

- Raising u[a, b] excites block a in every trained world that contains b. It is unseen only while the self-excitation exceeds it by more than the bids' range. So "an offset that no trained world sees" is too strong for the test as coded.
- Lowering u[b, a] alone is invisible at every margin and breaks the held-out world just as much. It needs nothing but what any fitted solution already has (a block that is not a member of a trained world is not applied there).
- The accurate statement: **trained worlds bound each cross term from above and not from below, and a held-out decision turns on the difference of two of them.** That is not weaker than the design's sentence; it is the same freedom, stated without the margin.
- *Fix.* Make B1's perturbation one-sided (lower only), keep the two-sided case as a second assertion with the margin stated, and reword the first bullet of 3A.

*B2.* With self-excitation 1e6 every load is exactly one-hot, so a cross-term change of 12,000 does nothing anywhere. That is the saturated limit only. With soft loads (my check: self-excitation 8 and 12 in load logits, perturbation ±6):

| Gate | Self-excitation | Largest change in a trained world | Change in the targeted held-out world |
|---|---|---|---|
| per item | 12 | 2.1e-4 to 1.4e-3 | 1.4e-7 to 8.6e-5 |
| per item | 8 | 0.045 to 0.078 | 0.015 to 0.020 |

- The reason normalisation works is therefore not that cross terms vanish. It is that **an item's load is a function of that item alone, so whatever a cross term does in a held-out world it also does in every trained world containing that item** — and every item is trained. Here the trained worlds changed more than the held-out one in every case. That is the statement to test, and it does not need saturation.
- *The normalised pooled gate is not robust at soft loads with sharp bids.* In the same check its ideal setting already fails (seen 1.5, held-out 1.4): a non-member block's logit is log(load) + bid, so it is excluded only if its log-load is below minus the bids' range. The per-item gate multiplies load and bid and has no such condition. The two gates are equal only in the saturated limit. In my trained probes the bids are a few units and it works; the design should state the condition.
- *Fix.* Add the soft-load version of B2 (trained worlds see at least what the held-out world sees). In stage A, compute log(load_a + load_b) as logaddexp of the two log-softmaxes, with no additive guard (the test's own docstring records a guard as a bug; my probe used 1e-9).

**3. SHOULD FIX — Stage A: one mechanism is unmeasured, one reading is by construction and not named, and `sum` has no stated criterion.**

*Is `pooled_norm` the gate I probed?* Yes: one softmax over blocks of log(load_a + load_b) + bid, loads a per-item softmax. The only difference is my guard of 1e-9 inside the log.

*Cross terms act by two routes, and the design measures one.* Post-hoc interventions on the trained pooled arm (`streams_self`, K = 6, `select`; scratch `p6.py`; nothing retrained):

| Intervention on the trained gate | Held-out error | Seen error |
|---|---|---|
| none | 0.454 | 0.0087 |
| fitted offset between the pair's two blocks removed | 0.272 | 0.0088 |
| blocks that are not the pair's masked | 0.325 | 0.0084 |
| both | 0.093 | 0.0086 |

(Per-item arm on the same split: 0.048 held-out. Median absolute offset: 0.039 trigger units on trained worlds, 0.76 on held-out worlds.)
- Removing the cross terms' two effects takes held-out error from 0.45 to 0.09 without touching trained worlds. So the account is causal on this seed, not only correlational.
- About half of it is **leak**: the excitation of a third block by a held-out pair, u[c, a] + u[c, b], is a sum no trained world contains. It shows as gate mass off the pair's blocks (0.70–1.00 on held-out worlds, at least 0.997 on trained), not as an offset.
- The row "offsets are not the mechanism: `streams_self` fails with held-out offsets no larger than trained ones" could be met by a run that fails through leak, which is the same cause.
- *Fix.* Add the measure "gate mass on blocks that are not the pair's, trained against held-out worlds", and the two removals above as the causal check (L56: probe failed runs before naming why they failed). State how offsets are aggregated for "five times" (median absolute offset over worlds, per run).

*"`pooled_norm` recombines" is close to by construction.* In the saturated limit it is the per-item arm, algebraically. What can fail is training reaching that limit. Name it. And split the condition: "at most half the reference" and "offsets not larger on held-out worlds" carry the account; "not worse than `tok_streams`" is a stricter, separate statement. One seed at K = 24: `pooled_norm` seen 0.0090, held-out 0.083, offsets at most 0.08 — recombines, and 1.8 times the per-item arm's 0.045 (K = 8: 0.056 against 0.053).

*`sum`.* The readings name the `blend` reference, which exists for `select` only. On `sum` use Entry 108's *full* regime (under half the smaller reference **and** within 3× of the arm's own seen error), as revision 3 did after R2.4. By the half-reference bar alone, the sigmoid pooled arm at 0.164 "recombines" at 10,000 times its seen error. One seed: `pooled_norm` on `sum`, K = 6, seen 1.9e-6, held-out 4.0e-6, split between the pair's blocks 0.499–0.502 on held-out worlds (un-normalised softmax pooled: 0.328, split 0.04–0.89). That is the cleanest single result for the account so far.

*P2 (sigmoid gate on `sum`).* Recording the uncertainty is right; leaving the statement for later is not necessary. The account already says what a non-competing gate does: with no competition between the pair's blocks there is no offset between them, so that route is closed; the leak onto third blocks remains, because it is a sum of cross terms that trained worlds never contain. Prediction from the account: partial recombination, with error carried by mass on non-member blocks — and the earlier probe adds that the blocks need not be primitives at all under this gate (three of six were). Write that as a test and a prediction before the run (AGENTS.md rule 2), and judge it by the *full* regime, where 0.164 is a failure, not a success.

**4. SHOULD FIX — Stage C is specified but has no readings, and three of its choices decide the outcome.**
- *No readings.* Revision 4 had rows; revision 5 has measures, one prediction (P8) and a sentence. Under section 5's rule "worse" is a ratio. `bank_match`'s new-world error is the regression error of one block (named as arithmetic), about 1e-5. An encoder arm that identified 99.9 % of items would be "worse" by a factor of a hundred. P8 is then met almost whatever happens. Use absolute criteria: identification of new and old items, and the *full* regime against the six-primitive reference.
- *"Read at 500 steps" makes the headline number a function of an untuned rate.* The same block on the same 256 demonstrations (Adam, my check): relative error after 500 steps 7.9e-3 at rate 0.003, 9.3e-5 at 0.01, 1.3e-5 at 0.03. State the adaptation rate and how it is chosen, or read at convergence of the block and report steps to reach it.
- *Two changes at once.* The two `bank_item` rows differ in the encoder being trainable **and** in replay. In my earlier probe replay with the encoder frozen was the best frozen variant (new 0.21, old 0.086). Add frozen + replay.
- *Expected and not named:* the positives-only key captures old evidence (0.58 of old items in my probe; old-world error 5.6e-5 → 1.24). And with a trainable encoder and enough replay the encoder arm must converge in distribution (a supervised encoder reaches 1.000 there, R2), so what the comparison measures is the number of steps and the damage on the way, not whether re-indexing is possible. Under shifted evidence the re-fitted encoder will degrade as in stage B; say whether stage C is read at 2× as well.
- *Replay is under-specified.* Which blocks (duplicates and blocks that compute no primitive exist after training: two of eight in my match-arm probe); what the target is (the old load distribution on replayed evidence handles duplicates; a hard label per block does not); and that the replay inputs are drawn from the known evidence distribution.
- *Smaller.* "The usual coverage" for five primitives is a number of worlds out of ten that the text does not give (the sampler's first seven must pass the graph check). `static_item`: "everything, on the demonstrations" still has no loss; the only one I can construct is a one-item application. The noise envelope's arm for stage C is not named.

**5. SHOULD FIX — Stage B: one reading without a condition; identification still fails for split primitives.**
- "Static against bank, off distribution" lists arms and a reference and no condition. P6 is its only content. State what counts as "the bank degrades to a wrong primitive, the static block to something unconstrained" (for instance: at 2× evidence, error conditional on a misidentified item, against the "applies one wrong primitive" reference).
- "Read at the decision level" is defined in section 5 for held-out worlds. At 2× evidence on trained worlds it needs its own definition.
- *Identification.* "The share of its load on blocks that compute its primitive" handles duplicates. It does not handle a primitive carried by several blocks none of which computes it alone: in my K = 24 per-item probe three of six primitives had no single block within 5 %, at unchanged error. "Every primitive computed by some block in at least 11 of 12 runs" would count that as failure to sort. Use one statistic for every arm: an item is identified when the update it would contribute (the loaded mixture Σ_k load_k f_k for a bank, the branch's output for a static arm) is nearest in direction — up to scale — to its own primitive's.
- "The bank sorts itself" (at least 11 of 12) and "it does not bootstrap" (more than 3 of 12) leave 2–3 failures of 12 unassigned; revision 3 had a "partial" row.
- Which tasks stage B runs on is no longer stated.
- A prediction worth adding, since it ties B to A: a gate per unit computed from an unfamiliar code is as unconstrained as an added code. If `static_item_gated` degrades like `static_item` at 2× evidence and the bank does not, the property belongs to normalised selection among stored units, not to multiplicative conditioning.

**6. SHOULD FIX — Section 3: is the contribution at the right size?** Nearly. Plainly:
- *Still overstated in part A:* the B3 sentence (finding 1); "an offset that no trained world sees" (finding 2); "capacity is irrelevant in both directions" — the derivation shows the gate-level freedom exists at every number of blocks, and three one-seed sizes agree; that extra blocks add nothing further is an observation (no single-world blocks at K = 24), not a consequence.
- *Overstated in part B:* "a 2025 paper names this as an unsolved instability". The paper names a risk ("risks training instability and local minima"). MOSAIC and BounceGrad did train modules and selection together from scratch, as section 2 now says. "Whether it sorts itself" is open for this layer on this carrier, not in general.
- *Overstated in part C:* "the function that is applied and the function that is tested must be the same". They need not be — MOSAIC pairs a separate forward model with each controller. The tie makes them the same at no cost and with no drift. That is an economy unless blocks keep changing during use, which is the condition under which it could be measured.
- *Undersold in part A:* the account now has a causal check on one seed (finding 3: 0.45 → 0.09 by removing cross-term effects post hoc), a second route (leak), and a statement that does not need saturation (finding 2). And section 3 does not say that the per-item softmax is what makes selection among stored units safe off distribution — the one cell kept for static against context-dependent weights is the same idea, and the two could be stated together.
- Part A's sizing against prior work ("an instance of the known principle, not a new theory") is right. "Prior statements of exactly this have not been searched for" is the correct caveat and should stay until the search is done.

**7. MINOR — Section 9.** My corrections are represented accurately: the signal may be a goal distance and needs no evidence-matched address; a stopped state is not an error signal; the bids already close the loop on the state; the order is by cost. One number: "swapping one primitive by 0.26–0.40" mixes in the value at step 1.0. At the present step sizes it is 0.26–0.30 (0.264 at 0.2, 0.295 at 0.4); 0.404 is at step 1.0, where reversal is 0.87. And the "must be the same" wording of part C (finding 6) applies to the last bullet's framing too.

**8. MINOR.**
- Rates are "tuned per arm family". Stage A compares 6 against 24 blocks; say whether the family's rate is checked at both sizes (L57: the ordering in every regime tried).
- Section 2's row on binding still cites Neural Production Systems for "recombines"; `literature.md` says that paper has no held-out-combination test.
- The title still names static against context-dependent weights; the contribution is now about binding and addressing. Section 3 says so plainly, which is enough.

## Anything still by construction and not named

- `pooled_norm` equals the per-item arm in the saturated limit (finding 3).
- B1 and B2 hold exactly because of the margin of 1e6 (finding 2).
- B3 and B4 hold inside the teacher's bid family (finding 1).
- Stage C: the positives-only key's interference, and the convergence of a replay-trained encoder in distribution (finding 4).
- Stage B's static-against-bank cell: a softmax over stored blocks can only output a mixture of stored primitives, so the bank's error off distribution is bounded by construction; what is open is the static arms' behaviour. Section 3 states the idea; the reading should name the bounded half as given.

## Probes

Exact (scratch `p5.py`, no training): tables in findings 1 and 2.

One seed, one split:

| Probe | Result |
|---|---|
| Post-hoc removal of cross-term effects, `streams_self`, K = 6, `select` | table in finding 3 |
| `pooled_norm`, `sum`, K = 6, rate 0.03 | seen 1.9e-6, held-out 4.0e-6; six blocks compute six primitives; split on held-out worlds 0.499–0.502 |
| `pooled_norm`, `select`, K = 24, rate 0.01 | seen 0.0090, held-out 0.083; held-out offsets at most 0.08 trigger units; 18 blocks unused, none serving a single world; mass on the pair's blocks 0.90–1.00 |
| One block fitted to 256 demonstrations (no model training) | relative error at 500 steps: 7.9e-3, 9.3e-5, 1.3e-5 at rates 0.003, 0.01, 0.03 |

Compute: three training runs (34, 30, 71 s) and seconds of exact checks.

## Checked and found correct

- `tests/test_binding.py` passes, and B0's bounds are honest about the floor (the docstring records three runs and what each got wrong).
- B1's "only this held-out world changes" holds, and holds for the one-sided perturbation as well.
- The test's `pooled_norm` is the gate of stage A's table and of my earlier probe.
- B4's construction (two triangles) is right for constants over the true triggers.
- Stage A's readings include three rows that sink the account, and each can occur.
- Section 2's read depths are stated per paper; the reviewer-fetched papers are marked as such.
- Section 9's account of what this design already does (load fixed per world, bids re-read the state) matches the code.
- Section 11's table matches what the text changed, except the two items marked in the table at the top (stage C's readings and `static_item`'s loss).
