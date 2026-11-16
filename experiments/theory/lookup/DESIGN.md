# The lookup block against the other stream configurations and a monolithic block — design

**Status: DESIGN, revision 3, nothing run by the author. EXPLORATORY.** Written 16 Nov 2026 at the researcher's request ("design the experiment to compare this version with the other moe stream configurations and monolithic"). Reviewed twice before any script, by an agent that did not write it: revision 1 "not sound as written" (`review/review_design.md`), revision 2 "sound with corrections" (`review/review_design_rev2.md`). Section 13 lists what each review changed. The version under test is the one in `docs/weight_lookup.md` (Entries 110–112, idea I38). History Entry 113.

## 1. What is being decided, and why the carrier has to change

Which stream configuration the program carries forward: the per-token streams arm that recombined in Entry 108 (`tok_streams`), or the lookup block, in whole or in part.

The lookup differs from `tok_streams` in three ingredients: **keys computed from the blocks' weights**, **bids from a block's own units**, and **a hard load** (only loaded blocks are computed). The same functions are expressible with or without each (U1, U4). That does not bound what training finds, so any of them can help or hurt. But it does decide where the comparison can be informative. On the worlds carrier an item is a *name* — a token with a free embedding — and a name can be addressed by a table. A key computed from the weights then has no job a stored key cannot do: in the first reviewer's one-seed check, the same key head fed random features unrelated to the weights trained as well as the real one.

So the experiment has a main part and a side part.

- **Main part, on evidence.** One change to the carrier: an item is a few demonstrations (x, f_p(x)) of a primitive, not its token. Nothing can be addressed by a table. A stored-key arm needs a learned encoder from demonstrations to an address; a monolithic block has to infer which function is meant; the lookup can compare its stored primitives against what the item shows. This is the form of the comparison in which the configurations can come apart for a reason that belongs to the lookup, and it meets the property the Entry 108 carrier lacked (Entry 107: primitives not named by the input).
- **Side part, on names.** A check for harm: does naive weight mixing break, and does any ingredient damage training where the baseline is known to work. It cannot favour keys computed from weights.

Both parts use a block with one site, where "one key head per site" is the same as "one key head". Retrieving different sites from different blocks (U5) is not tested.

## 2. Carrier

The worlds carrier of Entry 108 (`experiments/theory/worlds/worlds.py`): six primitives f_p(h) = tanh(A_p h) on a six-dimensional state, a trigger w_p each; a world is two primitives; four passes; the block sees the state and the world. Tasks `select` (apply the primitive whose trigger is larger) and `sum` (apply both). Nine of fifteen worlds are trained; every primitive is. Exact references (`blend`, `trans`, own seen error) place a held-out error on a scale.

**The evidence carrier** changes one thing. Each of a world's two primitives is presented as **m demonstrations** (x_i, y_i = f_p(x_i)), x_i ~ N(0, I). Triggers are not demonstrated, so bids are learned as before.

- *Unit of evidence.* One draw of demonstrations per (world, state): every state in a batch comes with its own evidence, in training and in evaluation. (A deployed lookup would compute the load once per world; the cost table counts it that way.)
- Training: m = 2, no noise.
- Test conditions, fixed in advance, none trained on:

| Condition | What is new to the model |
|---|---|
| held-out worlds | the pair of primitives |
| **shifted evidence** | demonstration states at 1.5, 2 and 3 times the training scale; read as a curve, with 2× the pre-named point |
| count and noise | m = 1 and m = 4; noise 0.1 on y. Reported; no reading is attached. |

Held-out worlds hold nothing out from the load (it sees one item at a time). The evidence conditions do, so they are read on trained worlds as well as held-out ones.

## 3. Arms on the evidence carrier

All bank arms use the plain `tok_streams` configuration (soft load, separate bid nets, eight blocks of width 16) with **identical blocks and bids**; they differ only in how an item's load over the blocks is computed. Parameter counts are reported, not equalised by widening blocks.

| Arm | Load of an item on block k | What it stands for |
|---|---|---|
| `enc_streams` | encoder(demonstrations) · stored key. The encoder is a small network on [x; y], averaged over the m demonstrations. | stored keys |
| `enc_streams`, wide | the same, encoder about three times the parameters | capacity |
| `probe_enc` | encoder(demonstrations) · key computed from the block's weights at a learned probe state | the researcher's key head, as worded |
| **`ev_lookup`** | −‖c · f_k(x) − y‖² / τ, averaged over the demonstrations: the applied block itself is evaluated at the item's states and compared with what the item shows. c is one learned scalar (the applied block may be a rescaled primitive, as on `sum`); τ starts so that the initial load logits have the spread of `tok_streams`'. | Entry 112's key head with the probe supplied by the item: **match form + tied to the applied weights** |
| `match_untied` | the same match against a second set of blocks g_k, stored beside the applied ones and trained only through the load | **match form without the tie** |
| `ev_lookup`, gradient from load to weights cut | | whether the addressing gradient shapes the primitives |
| `enc_mono` | one shared block conditioned on encoder(demonstrations), run per item, pooled by bids | monolithic, per-item read |
| `mono` | one block conditioned on the summed encodings | monolithic, summed read |
| `streams_self` | gate on the summed encodings | routing streams |
| `oracle` | load set from the truth (the choice between the two items is still learned bids) | ceiling for identification |
| `tok_streams` on names | | ceiling for the task: what the bank does when identification is free |

`ev_lookup` differs from `enc_streams` in two things, and the arms separate them: `enc_streams` → `match_untied` changes the *form* of the address (prediction-error match in place of a generic network); `match_untied` → `ev_lookup` adds the *tie* to the applied weights, which is the researcher's proposal.

**By construction, and not.** If the bank holds the six primitives exactly, a match identifies a noiseless demonstration perfectly at every scale (checked by the reviewer: 1.000 at scales 1–4). A block also has the teacher's exact form V tanh(B x), so it extrapolates by form; the shift test is tilted towards the match arms for that reason. Not by construction: whether a bank whose addressing depends on its own unfinished contents sorts itself into the primitives; how a trained (not exact) bank behaves off its training states; whether a learned encoder is in fact worse, and whether the tie adds anything to the match.

## 4. Measures

- Seen and held-out error under every test condition; regime against the references.
- **Identification, defined by function.** After training, each block is matched to the teacher primitive it computes (best scalar allowed, within 5 % in function), in every run. An item is identified when its most-loaded block computes its primitive. Reported with: primitives no block computes, primitives computed by more than one block.
- **Failed run:** seen error non-finite or more than 10 times the median of `tok_streams` on names.
- **Compute:** multiply-adds per pass, plus the per-world cost of the load (`ev_lookup`: K · m block evaluations; encoder arms: m encoder evaluations per item).

## 5. Statistics and the rule for a reading

Twelve splits on **twelve teachers** for every newly run arm (the stored Entry 108 runs used three; teacher medians there differed more than chance in two of twelve comparisons). Paired ratios over splits; median, bootstrap 5–95 % interval, and the per-teacher values.

- **Noise floor:** three other initialisations of the reference arm; the envelope of their intervals.
- **Worse / better:** interval entirely above / below the envelope, **at equal learning rates and at every pairing of rates whose seen error is within 2× of the best**. If it holds at some pairings only, the reading is "rate-dependent" and nothing is concluded.
- **Not worse (cleared):** interval's upper bound below 1.5.
- Otherwise **unresolved**, and reported as such.

Tuning: rate from {0.003, 0.01, 0.03, 0.1} on splits 100–102 by seen error, extended until interior (L57), **and step count** from 4,000, 8,000, 16,000 (the encoder arm needed 8,000 in the reviewer's probe); each arm is run at the step count where its seen error has stopped falling.

## 6. Stages on the evidence carrier

- **Stage 0.** Forward identities as tests; `tok_streams` and `tok_mono` re-run on three splits against the stored numbers; `tok_streams` on the twelve teachers. Stop on a mismatch.
- **Stage E0 — the phenomenon first.** `enc_streams` (both widths), `enc_mono`, `mono`, `oracle`. **Stop rule:** if the better encoder arm identifies at 0.99 or more in distribution *and* at 2× evidence, with held-out error not worse than `tok_streams` on names, then a learned encoder already does the job, addressing by content has nothing to fix here, and the experiment ends with that finding. If the encoder arm cannot be trained at any step count (seen error above 0.05), that is recorded as "the encoder arm does not bootstrap" and E1 proceeds.
- **Stage E1.** All arms of section 3, `select`; the bank arms also on `sum`.
- **Stage E2 — the bank grows**, only for arms still standing: twelve primitives, 18 trained worlds, sixteen blocks, constructive split sampler (P = 6 re-run with it so only P changes), references computed once per split in chunks. This is where blocks addressed by content can merge or starve. Reported with the share of forced decisions and the graph distance of held-out pairs, which move with P.
- **Stage E3 — mechanism.** Identification and the block-to-primitive match over training (does the bank sort itself, and when); duplicates among spare blocks.

## 7. Readings on the evidence carrier

Each row names the error it is read on. "Evidence error" is error on trained worlds under 2× evidence; "held-out error" is on held-out worlds with in-distribution evidence.

| Reading | Condition | What follows |
|---|---|---|
| **The tie earns its place** | `ev_lookup` better than `match_untied` on evidence error or held-out error | Keys computed from the applied weights do something a separate matcher does not. The lookup is carried forward. |
| **Match-form addressing, not the tie** | `match_untied` better than `enc_streams` (wide included) on evidence error, and `ev_lookup` not better than `match_untied` | Comparing stored functions with evidence helps; tying them to the applied weights is an economy of parameters, reported as such. |
| **Capacity** | the wide encoder closes the gap to the match arms | A learned encoder does it at a price. |
| **An encoder suffices** | E0's stop rule, or all bank arms not worse than each other in every condition | Keys from weights are dropped. |
| **`ev_lookup` does not bootstrap** | seen error worse than `enc_streams`, or in more than 3 of 12 runs some primitive has no block | Joint training of content and its own addressing fails as posed. No schedule is added after the fact. |
| **Partial** | primitives without a block in 1–3 of 12 runs | Reported as a failure rate beside whichever row above holds. |
| **`ev_lookup` worse on held-out error only** | | The tie costs recombination; `match_untied` or `enc_streams` is preferred. |
| **The key head as worded** | `probe_enc` against `enc_streams` | Expected not worse and not better. |
| **Monolithic** | `enc_mono`, `mono` by regime, and against the bank arms | Whether a single block can infer the function from evidence and recombine. |

## 8. Predictions on the evidence carrier

Not blind: the second reviewer ran one-seed probes of these arms (in the review). On that seed the match arms tied each other and held under shift, the encoder arm needed 8,000 steps and then lost under shift, and cutting the gradient changed nothing.

| | Prediction | Confidence |
|---|---|---|
| E1 | The better encoder arm identifies at 0.99 or more in distribution and below 0.97 at 2× evidence (E0 does not stop) | 0.7 |
| E2 | `ev_lookup`: every primitive has a block in at least 11 of 12 runs | 0.65 |
| E3 | `ev_lookup` better than `enc_streams` on evidence error | 0.7 |
| E4 | `ev_lookup` not better than `match_untied` anywhere on `select` (the reading is "match form, not the tie") | 0.6 |
| E5 | In distribution, held-out error of `ev_lookup`, `match_untied` and `enc_streams` is mutually not worse | 0.5 |
| E6 | `mono` is garbage on held-out worlds; `enc_mono` recombines | 0.85; 0.5 |
| E7 | `probe_enc` not worse and not better than `enc_streams` | 0.7 |
| E8 | Cutting the gradient from load to weights changes neither error of `ev_lookup` | 0.6 |
| E9 | At twelve primitives some match arm has a primitive without a block in at least 2 of 12 runs | 0.4 |

---

# Side part — names: a check for harm

Run after E1. Cut, on the second review's advice, to what can still say something.

## 9. Stage N1 — is naive weight mixing fragile?

The researcher's premise for wanting a lookup. Arms: `wmix` (the pooled gate mixes weights, W̄ = Σ_k g_k W_k, one block applied), `wmix_in` (weights mixed within a token, outputs pooled across tokens), `wmix_sum` (the routing form, summed read), against `tok_streams`. `select` at width 16 (the target gate is one-hot, where the two mixings agree, W1). `sum` at width 36: averaging two blocks' weights adds their functions only if they occupy different hidden units (W4), which with every pair a world needs 6 × 6 units; that is 1.77 times the parameters, so `tok_streams` at width 36 runs beside it. Twelve splits.

| Reading | Condition |
|---|---|
| Premise shown | on `sum`, `wmix` fits (seen error under 1e-3) and is not in Entry 108's *full* regime (held-out under half the smaller reference **and** within 3× of its own seen error); or on `select` it is worse than `tok_streams` on held-out error |
| Premise not shown | full on `sum`, not worse on `select` |

Either way the finding separates weight *averaging* from summing outputs. `tok_streams` and the lookup are on the same side of it (U1).

## 10. Stage N2 — do the ingredients harm training?

A full 2 × 2 × 2 on `select`, coverage 9, twelve splits.

| Factor | Off (as `tok_streams`) | On |
|---|---|---|
| Bid | separate net per block | β_k · tanh(B_k h), the block's own units, no bias |
| Key | stored | computed from the block's output at one learned probe state, projected; initial scale matched to `tok_streams`; features raw |
| Load | soft | hard: soft until the top load probability, averaged over tokens and a batch, passes 0.8 or step 2,000 comes; then top-1, not renormalised |

Controls: three nulls of `tok_streams` and one of `lookup`; a placebo key (the key head on fixed random features); the computed key with gradient cut and with standardised features; a separate bid net without bias; own-unit bids at width 16 and `tok_streams` at the own-unit arm's width (the bid's main effect is read together with these); hard from step 0; hardening after training (not renormalised) of every soft arm.

Pre-fixed contrasts: each factor's main effect; leave-one-out against `lookup`; `lookup` / `tok_streams`; `lookup` / hard-only (equal compute); computed key / placebo.

| Reading | Condition | What follows |
|---|---|---|
| Cleared | not worse on both errors (upper bound below 1.5), no more failed runs or collisions than `tok_streams` | the ingredient may be used |
| Convicted | worse on either error, or more failed runs | dropped, unless a named control form is cleared |
| Unresolved | neither | said so; not carried forward on this evidence |
| The content parts add something on names | `lookup` better than hard-only | stands after checks at equal rates and equal width |
| The content parts add nothing on names | `lookup` not worse and not better than hard-only; computed key not better than the placebo | expected; the hard load's cost saving is kept, the rest is decided on the evidence carrier |

Compute per pass at inference, reported as a table and part of no reading: `mono` 2,400; `tok_mono` 4,880; `tok_streams` 2,328; `wmix` 2,456, or 1,304 with per-token mixtures cached; any hard arm about 580. The saving belongs to the hard load alone.

## 11. Predictions on names

Not blind (first reviewer's one-seed probes).

| | Prediction | Confidence |
|---|---|---|
| N1 | `wmix` on `select` not worse than `tok_streams` on both errors | 0.55 |
| N2 | `wmix` on `sum`, width 36: fits and is not full (premise shown) | 0.7 |
| N3 | `wmix_in` on `sum`, width 36, is full | 0.45 |
| N4 | Every factor is cleared on held-out error | 0.5 |
| N5 | Computed key not better than the placebo | 0.8 |
| N6 | `lookup` not better than hard-only | 0.65 |
| N7 | Hardening a trained soft `tok_streams` raises its seen error at least fivefold | 0.75 |

---

## 12. Cost

Runs take 15–30 s at 4,000 steps with six primitives; the evidence arms need up to four times the steps. Stage 0 and E0: about one hour on two processes. E1: about three. N1 and N2: about three. E2: about three.

## 13. What the reviews changed

| Review finding | Change |
|---|---|
| R1.1 adoption could be met with the lookup's parts idle | compute is in no reading; `lookup` / hard-only added; rows that drop the lookup; readings two-sided |
| R1.2 a computed key is a table on names | placebo control; the evidence carrier became the main part |
| R1.3 "at best equal" did not follow from the identities | withdrawn (section 1) |
| R1.4 24 primitives breaks references and scale | dropped; twelve primitives only, references in chunks, constructive sampler at both sizes |
| R1.5 star, not factorial | full 2 × 2 × 2 |
| R1.6–8 hard load, normalisation, own-unit bid under-specified | specified; controls named |
| R1.9, R2.4 `wmix` | its own stage; 1.77×; Entry 108's *full* regime and an absolute fit condition on `sum`; `wmix_in` |
| R1.10, R2.5, R2.6 noise floor, margin, teachers | three nulls; "not worse" needs an upper bound below 1.5; readings must hold at equal rates; twelve teachers |
| R2.1 `ev_lookup` changed two things | `match_untied` added; the readings separate the match from the tie |
| R2.2 unit of evidence | one draw per (world, state) |
| R2.3 identification circular | defined by function, in every run |
| R2.7 encoder arm under-trained at 4,000 steps | step count tuned; wide encoder in E0; "does not bootstrap" row for the encoder arm |
| R2.8 trivial wins and losses | learned scalar in the match; initial temperature fixed; blocks identical across bank arms; duplicates counted |
| R2 order | evidence first, on the plain `tok_streams` configuration; names cut to N1 and one factorial; bank size moved to the evidence carrier |

## 14. What this cannot show

- Anything about several sites (section 1).
- Where items come from when the input does not itemise them (Entry 109, point 6): the evidence still arrives as two separate items.
- A regime where weight *averaging* is the right operation: primitives in a linear family in weight space, where a new primitive is a mixture of stored ones. There naive weight mixing is exact and output mixing is not; which holds depends on the teacher.
- The carrier's limits from Entry 108 stand: the `select` teacher has the shape of the per-item arms, and a block contains a teacher primitive exactly — which also tilts the shift test (section 3).
- Twelve splits; AGENTS.md asks for more seeds and a full bootstrap before a headline claim.
