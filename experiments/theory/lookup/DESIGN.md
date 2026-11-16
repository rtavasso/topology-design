# The lookup block against the other stream configurations and a monolithic block — design

**Status: DESIGN, revision 2, nothing run by the author. EXPLORATORY.** Written 16 Nov 2026 at the researcher's request ("design the experiment to compare this version with the other moe stream configurations and monolithic"). Revision 1 was reviewed before any script (`review/review_design.md`: "not sound as written", four blocking findings); section 12 lists what changed. The version under test is the one in `docs/weight_lookup.md` (Entries 110–112, idea I38). History Entry 113.

## 1. What is being decided, and why there are two parts

Which stream configuration the program carries forward: the per-token streams arm that recombined in Entry 108 (`tok_streams`), or the lookup block, in whole or in part.

The lookup differs from `tok_streams` in three ingredients: **bids from a block's own units**, **keys computed from the blocks' weights**, and **a hard load** (only the loaded blocks are computed). The same functions are expressible with or without each of them (U1, U4). That does not bound what training finds on held-out worlds, so the lookup can come out better or worse. But it does mean one thing for the design: on the worlds carrier an item is a *name* (a token with a free embedding), and a name can be addressed by a table. A key computed from the weights then has no job a stored key cannot do. The reviewer's one-seed placebo — the same key head fed random features unrelated to the weights — trained as well as the real one.

So the comparison is in two parts.

- **Part I, on names: does the lookup survive training?** Each ingredient is tested for harm, the premise about naive weight mixing is tested, and cost is tabulated. Nothing in Part I can favour keys computed from weights; it can only clear them or convict them.
- **Part II, on evidence: does addressing by content earn its place?** One change to the carrier: an item is a few demonstrations (x, f_p(x)) of a primitive instead of its token. Now nothing can be addressed by a table. A stored-key arm needs a learned encoder from demonstrations to an address, a monolithic block has to infer which function is meant, and the lookup can compare its stored primitives against the evidence. This is the form of the comparison in which the three configurations can come apart for a reason that belongs to the lookup. It also meets the property the Entry 108 carrier lacked (Entry 107: primitives not named by the input).

Both parts use a block with one site, where "one key head per site" and "one key head" are the same thing; retrieval of different sites from different blocks (U5) is not tested here.

## 2. Carrier

The worlds carrier of Entry 108 (`experiments/theory/worlds/worlds.py`): P primitives f_p(h) = tanh(A_p h) on a six-dimensional state, a trigger w_p each; a world is two primitives; four passes; the block sees the state and the world. Tasks `select` (apply the primitive whose trigger is larger) and `sum` (apply both). Some worlds are never trained; every primitive is. Exact references (`blend`, `trans`, own seen error) place a held-out error on a scale. The baseline failure is already reproduced there (summed-read monolith worse than not moving, 12/12).

Changes to the code that both parts need: references computed once per split and in chunks (they do not depend on the arm); a constructive split sampler for P > 6 (random cycle through all primitives plus random chords, checked by `graph_ok`).

---

# Part I — names: does the lookup survive training?

## 3. Arms

**Stage 1a, the premise.** The researcher's reason for a lookup is that naive gate-and-sum in weight space is fragile.

| Arm | Change from `tok_streams` |
|---|---|
| `wmix` | the pooled gate mixes weights, W̄ = Σ_k g_k W_k, and one block is applied |
| `wmix_in` | weights are mixed within a token (Σ_k a_tk W_k); the two tokens' outputs are pooled as before |
| `wmix_sum` | `wmix` with the summed read and a gate on it (the routing form) |

On `select` the target gate is one-hot, where weight and output mixing agree (W1), so width 16 is fair. On `sum`, averaging two blocks' weights adds their functions only if they occupy different hidden units (W4); with every pair a world that needs width 6 × 6 = 36. `wmix` runs on `sum` at width 16 (cannot represent the target; reported, bears on nothing) and at width 36 (can; 1.77 times the parameters, with `tok_streams` at width 36 beside it).

**Stage 1b, the ingredients: a full 2 × 2 × 2.**

| Factor | Off (as `tok_streams`) | On |
|---|---|---|
| Bid | separate net per block, with bias | β_k · tanh(B_k h), the block's own units, no bias |
| Key | stored | computed: the block's output at C = 1 learned probe state (six features for K = 8 blocks, so the projection cannot act as a table), projected; initial scale set so the initial load logits match `tok_streams`; features raw |
| Load | soft over all blocks | hard: soft until the token-to-block load's top probability, averaged over tokens, passes 0.8 or step 2,000 comes; then top-1, not renormalised (the top entry keeps its softmax value, so the address still gets gradient) |

All off is `tok_streams`; all on is `lookup`. Parameters matched at about 2,500 by block width. Controls, each for one named confound:

| Control | For |
|---|---|
| `null_tok_streams` (another initialisation) | the noise floor of every contrast |
| placebo key (the key head on fixed random features) | whether a computed key does anything a table does not |
| computed key with the gradient from key to weights cut | the two-way coupling of Entry 112 |
| computed key, features standardised over the bank | the normalisation, which the review found changes how sharp the address gets |
| own-unit bid at width 16; `tok_streams` at the own-unit arm's width | width against bid source |
| hard from step 0 | whether the soft start is needed (L56) |

Old arms for the table, from stored Entry 108 runs once stage 0 reproduces them: `mono`, `tok_mono`, `streams_self`, `streams_joint`, `unit`, `oracle`.

## 4. Measures

- Seen and held-out error; regime against the references.
- **Failed run:** seen error non-finite or more than 10 times the median of `tok_streams`.
- **Allocation:** number of different blocks the tokens load (argmax); collisions; for hard arms the switch step and collisions at the switch.
- **Hardening after training** of every soft arm (no retraining).
- **Compute per pass** at inference (multiply-adds, one state, two-token world; things that do not read the state are computed once per world). At P = 6: `mono` 2,400; `tok_mono` 4,880; `tok_streams` 2,328; `wmix` 2,456, or 1,304 with per-token mixtures cached; any hard arm about 580. The saving belongs to the hard load alone and is reported as a table, not as a condition of any reading.

## 5. Contrasts and readings

Paired ratios over the same twelve splits; median and bootstrap 5–95 % interval. An arm is **worse** than another if its interval lies above the interval of `null_tok_streams`, **better** if below, **indistinguishable** if they overlap — and then the interval's width is what was ruled out, and is stated. Readings are two-sided. A better result stands if it survives three checks: equal rates, equal width, and the null.

On `sum`, where `tok_streams` is exact to 1e-6 and every ratio is huge, the criterion is the regime (held-out error at most half the smaller reference), not a ratio.

Pre-fixed contrasts: each factor's main effect and the leave-one-out arms against `lookup`; `lookup` / `tok_streams`; **`lookup` / hard-only** (equal compute: do bids and keys from content add anything to a hard `tok_streams`?); `lookup` / `tok_mono`; computed key / placebo key.

| Reading | Condition | What follows |
|---|---|---|
| Premise shown | `wmix` does not recombine on `sum` at width 36 (seen error at most 10× `tok_streams`' but held-out above half the smaller reference), or is worse on `select` held-out error | Weight *averaging* is fragile here. This separates averaging from summing outputs; `tok_streams` is on the good side of it exactly as the lookup is (U1). |
| Premise not shown | `wmix` recombines on `sum` at width 36 and is indistinguishable on `select` | Nothing is wrong with gate-and-sum in weight space on this carrier. |
| An ingredient is cleared | indistinguishable or better on both errors, no more failed runs or collisions than `tok_streams` | It may be carried to Part II. |
| An ingredient is convicted | worse on either error, or more failed runs | Dropped, unless a named control form is cleared (gradient cut; hardening after training). |
| Keys from weights do nothing here | computed key indistinguishable from the placebo | Expected. Part I says nothing for them. |
| Keep `tok_streams` with a hard load | `lookup` indistinguishable from hard-only, or worse | On names, the lookup's content parts are dropped; the cost saving is kept. |
| Nothing survives | hard load convicted in every form | `tok_streams` as it is. |

## 6. Stages, bank size, stop rules

- **Stage 0.** Forward identities as tests (the factorial with every factor off equals `tok_streams`; `wmix` equals it at one-hot gates). Re-run `tok_streams` and `tok_mono` on three splits against the stored numbers. Stop on any mismatch.
- **Stage 1a.** The `wmix` arms, `select` and `sum`, coverage 9, twelve splits. Reported before 1b is run. It does not stop 1b: the ingredients are tested either way.
- **Stage 1b.** The factorial and controls, `select` at coverage 9 and 7, `sum` at coverage 9, twelve splits.
- **Stage 1c, a larger bank.** P = 12 with 18 trained worlds (three per primitive), the constructive sampler, K = 16 blocks and 16-dimensional embeddings (at P = 6 these are 8 and 8, so both are ⌈4P/3⌉), monoliths at matched parameters, every arm tuned again; P = 6 re-run with the constructive sampler so that only P changes. All eight factorial arms, twelve splits. Reported with the share of forced decisions and the graph distance of held-out pairs, which move with P (0.46 → 0.28; 2.0 → 2.5).
  P = 24 is used for one thing only: seen error, failed runs and allocation with 32 blocks (does a hard or computed-key load starve or merge blocks?). No held-out reading is taken there, because the reference scale collapses (`trans` 0.40 against `blend` 0.49; forced share 0.15).

## 7. Predictions for Part I

Not blind: the reviewer's one-seed probes of every arm are in the review, and they tied `tok_streams` on `select` throughout.

| | Prediction | Confidence |
|---|---|---|
| P1 | `wmix` on `select`: indistinguishable from `tok_streams` on both errors | 0.6 |
| P2 | `wmix` on `sum`, width 36: fits (seen error under 1e-3) and does not recombine | 0.65 |
| P3 | `wmix_in` on `sum`, width 36, recombines where `wmix` does not | 0.5 |
| P4 | Each factor's main effect on held-out error is indistinguishable from the null at P = 6 | 0.6 each |
| P5 | Computed key indistinguishable from the placebo | 0.8 |
| P6 | `lookup` indistinguishable from hard-only | 0.6 |
| P7 | Hardening a trained soft `tok_streams` raises its seen error at least fivefold | 0.75 |
| P8 | At P = 24 some hard arm has more collisions than soft `tok_streams` | 0.55 |

---

# Part II — evidence: does addressing by content earn its place?

## 8. The one change to the carrier

A world is still two primitives, but each is presented as **m demonstrations** (x_i, y_i = f_p(x_i)), with fresh states x_i ~ N(0, I) at every step. Nothing names a primitive. Triggers are not demonstrated, so bids are learned as before. Tasks, worlds, held-out worlds and references are unchanged. Training uses m = 2 and no noise.

Three test conditions are fixed in advance, none of them trained on:

| Condition | What is new |
|---|---|
| held-out worlds | the pair of primitives, as in Entry 108 |
| **shifted evidence** | demonstration states at twice the training scale |
| fewer or more, or noisy | m = 1 and m = 4; noise of 0.1 on y |

## 9. Arms

All bank arms share blocks, bids, load and combination (the configuration cleared in Part I); they differ only in **how an item's load over the blocks is computed**.

| Arm | Load of item j on block k | Stands for |
|---|---|---|
| `enc_streams` | encoder(demonstrations) · stored key G_k. The encoder is a small network on [x; y], averaged over the m demonstrations. | stored keys: the per-token streams arm with the token replaced by a learned description |
| `probe_enc` | encoder(demonstrations) · key computed from the block's weights at a learned probe state | the researcher's key head, taken literally |
| **`ev_lookup`** | −‖M (f_k(x) − y)‖², averaged over the demonstrations, with a learned metric M and temperature: the block is evaluated at the item's own states and compared with what the item shows | the key head of Entry 112 with the probe states supplied by the item |
| `ev_lookup`, gradient from load to weights cut | | whether the addressing gradient shapes the primitives, for better or worse |
| `enc_mono` | one shared block conditioned on encoder(demonstrations), run per item, pooled by bids | monolithic with the per-item read |
| `mono` | one block conditioned on the summed encodings | monolithic, summed read |
| `streams_self` | gate on the summed encodings | the routing form |
| `enc_streams`, wide encoder (about three times the parameters) | | capacity against mechanism |
| `oracle` | load set from the truth | floor |

`ev_lookup` is an adaptation of the proposal, not the proposal as worded: the key is still a function of the block's weights, but the probe is the item's state, so query and key meet in the space of what a primitive does. `probe_enc` keeps the learned probe.

What is by construction and what is not. If the bank holds the six primitives exactly, `ev_lookup` identifies a noiseless in-distribution demonstration perfectly; that part is arithmetic. Not by construction: whether the bank becomes the primitives when its addressing depends on its own unfinished contents (at the start every block explains every demonstration equally badly); what happens when the demonstration state is outside what the blocks were trained on, where the blocks themselves are wrong; and whether a learned encoder is in fact worse.

Cost is counted honestly: `ev_lookup` evaluates every block on every demonstration once per world (K · m block evaluations, not per state or pass); the encoder arms run the encoder m times per item.

## 10. Measures, stages, readings

Measures as in Part I, plus **identification**: the share of items whose most-loaded block is the block allocated to the item's true primitive (defined when allocation is one-to-one), under each test condition.

- **Stage E0, the phenomenon first.** `enc_streams`, `enc_mono`, `mono` and `oracle` only; tuned; twelve splits. Stop rule: if `enc_streams` identifies at 0.99 or better in distribution and under shifted evidence, and its held-out error is indistinguishable from the token arm's of Part I, then a learned encoder already does the job, there is nothing for addressing by content to fix on this carrier, and Part II ends with that finding.
- **Stage E1.** All arms, `select`, coverage 9, twelve splits; `sum` for the bank arms.
- **Stage E2, mechanism.** Allocation and identification over training for `ev_lookup` (does the bank sort itself, and when); the bank after training against the teacher's primitives.

| Reading | Condition | What follows |
|---|---|---|
| Content addressing earns its place | `ev_lookup` better than `enc_streams` on held-out error in distribution or under shifted evidence, allocation one-to-one in at least 10 of 12 splits, and the wide encoder does not close the gap | The lookup, with evidence keys, is the configuration to carry forward where items are not named. |
| Capacity, not mechanism | as above, but the wide encoder closes the gap | A learned encoder does it at a price; reported as cost. |
| An encoder suffices | indistinguishable in every condition | Keys from weights are dropped. |
| It does not bootstrap | `ev_lookup` worse on seen error, or allocation collapsed | Joint training of content and content-addressing fails here as posed. Whether a schedule rescues it is a separate question, not answered by adding one after the fact. |
| The literal key head | `probe_enc` against `enc_streams` | Expected indistinguishable, for the reason Part I gives. |
| Monolithic | `enc_mono` and `mono` against the bank arms, by regime | Whether a single block can infer the function from evidence and recombine. |

## 11. Predictions for Part II (blind: nothing has been run on this carrier)

| | Prediction | Confidence |
|---|---|---|
| E1 | `enc_streams` identifies at 0.9 or better in distribution, and below 0.8 under shifted evidence | 0.7; 0.55 |
| E2 | `ev_lookup` reaches one-to-one allocation in at least 9 of 12 splits | 0.5 |
| E3 | Where it does, it identifies at 0.95 or better under shifted evidence | 0.55 |
| E4 | In distribution, held-out error of `ev_lookup` and `enc_streams` is indistinguishable | 0.5 |
| E5 | `mono` is garbage on held-out worlds; `enc_mono` recombines | 0.85; 0.55 |
| E6 | `probe_enc` indistinguishable from `enc_streams` | 0.75 |
| E7 | Cutting the gradient from load to weights makes `ev_lookup` worse on seen error | 0.5 |

---

## 12. What changed from revision 1, by review finding

| Finding | Change |
|---|---|
| 1 adoption could be met with the lookup's parts idle | Compute removed from every reading; `lookup` / hard-only and `lookup` / `tok_mono` added; rows that drop the lookup added; readings two-sided; margin taken from the null of `tok_streams` |
| 2 a computed key is a table on names | Placebo control; one probe state so the key cannot be a table; Part I states it can only test for harm; Part II added |
| 3 "at best equal" did not follow | Withdrawn (section 1) |
| 4 P = 24 breaks references and scale | References once per split, in chunks; constructive sampler at both P; rules in P stated; held-out readings at P = 6 and 12 only |
| 5 star, not factorial | Full 2 × 2 × 2 at both P |
| 6 hard load underspecified | Probability, no renormalisation, fallback step, recorded switch; hard-from-zero control; the old prediction on hardening reversed |
| 7 normalisation a hidden dial | Raw in the factorial with matched initial scale; standardised as a named control |
| 8 own-unit bid changed three things | Width controls; no bias, stated |
| 9 `wmix` | First, as its own stage; 1.77×; regime criterion on `sum`; cached cost; `wmix_in` |
| 10 noise floor and power | `null_tok_streams`; twelve splits everywhere; failed run defined against `tok_streams` |
| 11 tuning at larger P | Every arm tuned at each P |

## 13. Tuning, seeds, statistics, cost

Every arm's learning rate on splits 100–102 by seen error at each P and on each carrier, from {0.003, 0.01, 0.03, 0.1}, extended until the best is interior (L57); contrasts also at every pairing of rates within 2× of the best seen error, and at equal rates. Twelve splits over three teachers (the bootstrap treats splits as independent, as in Entry 108). AGENTS.md asks for more seeds before a headline claim; everything stays EXPLORATORY. Every script is committed with its predictions before it runs (L56); results are reviewed by an agent that wrote none of this.

Runs take 15–30 s at P = 6, about 45 s at P = 12, about 2 minutes at P = 24. Part I: about five hours on two processes. Part II: about three.

## 14. What this cannot show

- Anything about several sites (section 1).
- Where items come from when the input does not itemise them (Entry 109, point 6). In Part II the input still arrives as two separate items.
- A regime where weight *averaging* is the right operation: primitives that lie in a linear family in weight space, where a new primitive is a mixture of stored ones. There naive weight mixing is exact and output mixing is not; which holds depends on the teacher, and a synthetic carrier cannot say which real tasks resemble.
- The carrier's limits from Entry 108 stand: the `select` teacher has the shape of the per-item arms, and a block contains a teacher primitive exactly.
