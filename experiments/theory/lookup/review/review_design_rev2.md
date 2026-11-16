**Verdict: sound with corrections.** Revision 2 resolves the four blocking findings of the first review in its text, not only in section 12. Part I is now a fair survival check. Part II is well posed and is the right experiment, but one control is missing without which its headline reading cannot be attributed to the lookup (finding 1, blocking for stage E1 only), and several things must be specified before E0 can be scripted.

# Review of `experiments/theory/lookup/DESIGN.md`, revision 2 (uncommitted), before any script

Same reviewer as `review_design.md`. Checks are exact computations from the teacher and the stored Entry 108 runs unless marked as a one-seed probe. One-seed probes use my own scratch arms for Part II (my reading of sections 8–9, on the plain `tok_streams` configuration: soft load, separate bid nets), split 0, teacher 10, `select`, coverage 9, rate 0.01, 4,000 steps unless stated. They are indications only (L30). About 5.5 minutes of compute.

## Are the first review's findings resolved?

| Finding (rev. 1) | Resolved in the text? | What remains |
|---|---|---|
| 1 adoption met with the lookup idle | **Yes.** Compute is out of every reading (§4); `lookup` / hard-only and `lookup` / `tok_mono` are pre-fixed; two rows drop the lookup. | No row for `lookup` *better* than hard-only, though §5 says readings are two-sided. `lookup` / `tok_mono` is a contrast with no reading attached. "Cleared" has no upper bound (finding 5). |
| 2 a computed key is a table | **Yes in framing.** Placebo arm; §1 and the row "keys from weights do nothing here" say Part I can only test for harm. | The claim that one probe state means "the projection cannot act as a table" is too strong (finding 9). |
| 3 "at best equal" | **Yes.** Withdrawn in §1. | — |
| 4 P = 24 | **Yes.** Chunked references once per split; constructive sampler at both P; K and embedding at ⌈4P/3⌉; no held-out reading at P = 24. | Arms, splits and tuning at P = 24 are not stated. |
| 5 star | **Yes.** Full 2 × 2 × 2 with leave-one-out contrasts. | The bid factor still carries width with it in every cell (finding 8). |
| 6 hard load | **Yes.** | Say whether "hardening after training" is renormalised. |
| 7 normalisation | **Yes.** | — |
| 8 own-unit bid | **Mostly.** Width controls added, "no bias" stated. | Bias is still bundled with the bid's source (finding 8). |
| 9 `wmix` | **Partly.** Order, 1.77×, cached cost and `wmix_in` are in. | **The new criterion on `sum` is wrong** (finding 4). |
| 10 noise floor and power | **Partly.** `null_tok_streams`, twelve splits, failed-run definition are in. | The rule built on the null is not sound as stated (findings 5, 6). |
| 11 tuning at larger P | **Yes.** | — |

## Findings

**1. BLOCKING for stage E1 — `ev_lookup` differs from `enc_streams` in two things at once, so "content addressing earns its place" cannot be attributed to keys computed from the weights.**
- Section 9 says the bank arms "differ only in how an item's load is computed". Between `enc_streams` and `ev_lookup` that one phrase covers two changes:
  - *the form of the address*: a prediction-error match, −‖M (g(x) − y)‖², in the teacher's own function class, against a generic network on [x; y];
  - *the tie*: the function g being matched is the applied block itself, f_k.
- The researcher's proposal is the tie (keys computed from the blocks' weights). A win for `ev_lookup` could come from the form alone. The wide encoder controls capacity, not form; the gradient-cut arm keeps both.
- L59: when a hypothesis names two things, build the factorial.
- One-seed probe: an untied match trained as well as the tied one (seen / held-out 0.0107 / 0.050 against 0.0085 / 0.049), kept its held-out error at twice the demonstration scale (0.049 in both; seen error 0.0107 → 0.0165 untied, 0.0085 → 0.0085 tied), and its top block was the right one in 0.9998 of items in distribution and 0.9995 at twice the scale. On this seed nearly everything `ev_lookup` did, a match without the tie did.
- *Fix.* Add an **untied match** arm: load −‖M (g_k(x) − y)‖² with g_k a second set of blocks of the same shape, stored beside the applied blocks and trained only through the load. If it equals `ev_lookup`, what earned its place is match-form addressing and the tie is an economy of parameters. If it is worse, the tie matters. Add the corresponding row to §10.

**2. SHOULD FIX — The unit of evidence is undefined, in training and in evaluation.**
- "Fresh states at every step" does not say whether demonstrations are drawn per world or per (world, state). §9's cost line ("once per world") implies per world.
- Evaluated that way with the existing `ev()`, held-out error rests on 6 held-out worlds × 2 items = 12 identification events per run. One misidentification applies the wrong primitive for four passes on 1/6 of the test set. Error would be bimodal across runs.
- The switch rule of the hard load ("top probability averaged over tokens") also needs restating for items whose load varies with the draw.
- *Fix.* State the unit; evaluate with many evidence draws per world (for example one per evaluation state); average the switch criterion over a batch of draws.

**3. SHOULD FIX — Identification is circular as defined, and conditional on success.**
- "The block allocated to the item's true primitive" has no definition on this carrier: there is no token whose argmax fixes the allocation. Taking the modal block per primitive makes identification a measure of consistency.
- "Defined when allocation is one-to-one" drops exactly the runs that fail; E3 ("where it does, it identifies at 0.95") is then a statement about survivors.
- Spare blocks make the argmax version under-count. In my `ev_lookup` probe two blocks computed the same primitive and shared its load: the most-loaded block was "the allocated one" in 0.969 of items (0.920 at twice the scale), while it was *a block computing the true primitive* in 1.000 of items under every condition.
- *Fix.* Match blocks to teacher primitives by function (after the best scalar, since on `sum` a block computes 2 f_p), count an item as identified when its most-loaded block computes its primitive, do this in every run, and report primitives without a block and duplicated primitives.

**4. SHOULD FIX — The `sum` criterion in Part I passes an arm that is 3,000 times worse on held-out worlds than on trained ones.**
- §5 and the premise row use "held-out error at most half the smaller reference". In my one-seed probe of the first review, `wmix` on `sum` at width 36 had seen 6.0e-5 and held-out 0.186 with references 0.57 and 0.53: 0.186 is below 0.266, so it "recombines". The row "premise not shown: nothing is wrong with gate-and-sum in weight space" would then apply to an arm whose held-out error is 70,000 times `tok_streams`'.
- The same row's "seen error at most 10× `tok_streams`'" is a ratio to 1.5e-6 again (1.5e-5; the probe's 6e-5 fails it), and disagrees with P2's "under 1e-3".
- *Fix.* Use Entry 108's *full* regime on `sum` (under the bar **and** within 3× of the arm's own seen error), and an absolute fit condition (seen under 1e-3).

**5. SHOULD FIX — "Cleared" is now failure to reject.** An arm whose interval is wide enough to overlap the null is cleared and "may be carried to Part II", however wide. Stating the width is reporting, not a criterion. *Fix.* Cleared requires the interval's upper bound below a pre-fixed margin; 1.5, the size of the streams-over-monolith gap, is defensible.

**6. SHOULD FIX — The interval-against-null rule (§5): usable, with three repairs.**
- *One null draw is not centred.* The stored same-arm nulls on held-out error have intervals 0.94 [0.86, 0.98] (`tok_mono`, coverage 9), [0.91, 1.10], [0.88, 1.13], [0.93, 1.19]. The first excludes 1. An arm identical to `tok_streams` with interval [0.99, 1.15] would be "worse" against such a null. Run two or three null initialisations (12 runs each) and use their envelope.
- *The margin is now smaller than the rate effect.* `tok_streams` at rate 0.03 against 0.01 is 1.43 [1.32, 1.66] on held-out error at near-equal seen error (0.0071, 0.0081). With a margin of about ±10 %, rate selection alone produces "worse" or "better". §13 reports contrasts at other pairings but does not say which pairing decides. State: a reading must hold at equal rates and at every qualifying pairing; otherwise it is "rate-dependent".
- *Three teachers.* Across teachers the paired ratios are mostly as variable as chance predicts, but not always: the variance of teacher means of the log ratio is 9 times its expected value for `null_mono` (teacher medians 1.16, 0.97, 0.92) and 6 times for the rate contrast (1.87, 1.28, 1.36), two of twelve comparisons beyond the 5 % point of F(2, 9). So the split bootstrap is somewhat too narrow. A cluster bootstrap over three teachers is not possible. Use twelve teachers for every newly run arm (the teacher is a seed; `tok_streams` re-run on them costs twelve runs), and report the three or twelve teacher medians beside every interval.
- Contrasts that do not involve `tok_streams` (`lookup` / hard-only, key / placebo) are judged against the null of `tok_streams`. Add a same-arm null for `lookup`.

**7. SHOULD FIX — Stage E0 can show its phenomenon for a trivial reason: the encoder arm is under-trained at 4,000 steps.**
- One-seed probes of `enc_streams` at 4,000 steps: hidden width 32, seen / held-out 0.149 / 0.535 at rate 0.01 and 0.106 / 0.346 at rate 0.03; width 96, 0.235 / 0.618. Each had two primitives sharing a block or blocks far from any primitive. At **8,000 steps** (width 32, rate 0.01): 0.0115 / 0.083, six primitives on six blocks, identification 0.9993.
- Capacity is not the limit: the same encoder trained with the label reaches 1.000 in distribution at width 32.
- So step count is the dial that decides whether E0 "shows the phenomenon" in distribution, and §13 tunes only the rate. E1 (identification at 0.9 or better, confidence 0.7) would fail at 4,000 steps for this reason.
- The phenomenon that is not trivial is under shift: the 8,000-step encoder arm identifies 0.928 at twice the scale and 0.68 at three times, and its seen error goes from 0.0115 to 0.155; the supervised encoder gives 0.954 and 0.746 (width 32), 0.977 and 0.820 (width 96).
- *Fix.* In E0 compare 4,000, 8,000 and 16,000 steps for every arm (as Entry 108's stage 0 did), put the wide encoder in E0, and apply the stop rule to the best encoder at the step count where its seen error has stopped falling. Add a row "the encoder arm does not bootstrap" beside the one for `ev_lookup`. The rule itself is right in shape (phenomenon first; end Part II if a learned encoder already identifies in and out of distribution). Its ceiling, the token arm of Part I, should be named as such; and say whether `oracle` sets the load only or the choice as well.

**8. SHOULD FIX — Things that can make `ev_lookup` lose or win trivially.**
- *Initial scale.* Neither M nor the temperature has a stated initial value. With 0.1-scale write matrices every block misses every demonstration by about ‖y‖²; differences between blocks are of order 2 y·f_k. Set M = I and the temperature so the initial load logits match `tok_streams`, as §3 already does for the computed key.
- *Scale coupling.* The block's output is both what is applied and what is matched against y. Wherever the load or pool weight on the right block is not 1, the applied block must be a rescaled primitive, and the match is then off. On `sum` the pool is one half, so a block must output 2 f_p: an exact bank at that scale identifies 0.993–0.998 instead of 1.000 (m = 2, no noise), and its residual for the right block is ‖f_p‖², not 0. The un-renormalised top-1 load of Part I has the same effect. One-seed probe on `sum` (rate 0.03): the tied match reached seen / held-out 4.3e-3 / 8.8e-3, the untied match 1.1e-4 / 1.7e-3, and in both the blocks had become 2 f_p. *Fix.* A learned scalar inside the match (c · f_k(x) against y), or renormalise the load for the evidence arms.
- *Spare blocks.* With 8 blocks for 6 primitives and addressing by content, two blocks that come to compute the same primitive split its load. This happened in my `ev_lookup` probe (two blocks on one primitive; mean top probability 0.92; no cost in error with a soft load). With the un-renormalised top-1 load of Part I it would halve that primitive's output. State what is expected of the spares and count duplicates.
- *Parameter matching.* Nothing is said for Part II. `ev_lookup` has no encoder; if its freed parameters widen the blocks, width is confounded as in Part I. Keep the blocks identical across bank arms and report parameter counts.
- *Part I's bid factor* is still bid source + width + bias in every cell of the factorial; say the main effect is read together with the two width controls, and add a separate bid net without bias.

**9. MINOR.**
- *"Cannot act as a table."* Six random features for eight blocks are all extreme points of their hull in every one of 40 draws, so a projection of them can still give each token a block of its own. The placebo will very probably tie again (the design's own P5). At K = 32 about 5 % of blocks are not addressable by any query (all are in only 20 % of draws); with eight spares that is harmless, but it is a property of the feature count, not of the key's source.
- *The shifted-evidence test is sensible and does not break trivially.* At twice the scale 40 % of demonstration outputs are saturated (12 % in training), and an exact bank still identifies 1.000 at scales 1 to 4 (0.996–0.9996 with m = 1 and noise 0.1). The loop itself takes states from rms 1.0 to 1.34 over four passes, so twice the scale is outside the dense training region but not far. Two caveats: the block has the teacher's exact form V tanh(B x), so it extrapolates by form — §9 should list this as tilted by construction; and twice the scale sits at the start of the encoder's decline (supervised encoder: 0.995 at 1.5×, 0.954 at 2×, 0.746 at 3×), so evaluate at 1, 1.5, 2 and 3 (evaluation only) and read the curve.
- *Readings of §10.* They are two-sided and include real negatives. Gaps: one-to-one in 6–9 splits falls between "earns" (10) and "does not bootstrap" ("collapsed"); `ev_lookup` worse than `enc_streams` on held-out error only has no row; "in distribution **or** under shifted evidence" gives two chances; the m = 1, m = 4 and noise conditions have no reading. Say which error each row means: bank arms share blocks and bids, so held-out *worlds* hold nothing out from the load; the evidence conditions do, and should be read on seen worlds too.
- Part I's cost is nearer nine hours than five on two processes if P = 24 is tuned (my count of runs).

## Order, and whether to cut Part I

Run stage 0, then **E0 before Part I**, on the plain `tok_streams` configuration (soft load, separate bid nets) instead of "the configuration cleared in Part I". The case:
- Nothing in Part II depends on Part I once it uses the known-good configuration.
- E0 is a stop-rule stage of four or five arms. If a learned encoder suffices, keys from weights are dropped on both carriers and most of Part I's factorial is moot. If it does not, the lookup has a job and Part I's factorial says how to configure it. E0's outcome changes what of Part I is worth running; Part I's outcome does not change whether E0 is run.
- Part I is predicted to tie by its own author (P4–P6 at 0.6, P5 at 0.8), and my one-seed probes tied.

Cuts to Part I: keep 1a (it is the researcher's premise, it is cheap, and `sum` may show something); run the 2 × 2 × 2 on `select` at coverage 9 only (on `sum` every per-token arm is exact, and coverage 7 adds no question about the ingredients); move the bank-size dial to after E1 and onto the evidence carrier, where merging and starvation under content addressing can actually occur.

## Probes and checks

Exact, no training (scratch `r2_a.py`):
- Exact-bank identification by nearest match, three teachers, 20,000 items each: 1.000 at demonstration scales 1, 1.5, 2, 3, 4 with m = 2 or m = 1 and no noise; 0.996–0.9996 with m = 1 and noise 0.1; 0.993–0.998 when blocks are scaled by 2.
- Saturation: share of |tanh| > 0.9 is 0.12 at scale 1 and 0.40 at scale 2.
- Random features in six dimensions: all K points extreme in 100 % of draws at K = 8, 90 % at K = 16, 20 % at K = 32.
- Stored nulls and teacher medians: as quoted in finding 6.

One-seed training probes of Part II (scratch `lk2.py`, `r2_sup.py`; conditions in the header). Error is seen / held-out worlds; "×2" is the same with demonstration states at twice the scale. "Right block" is the share of items whose most-loaded block computes the item's primitive (within 5 % in function).

| Arm (my reading) | Parameters | Error | Error, ×2 evidence | Right block: in distribution / ×2 / ×3 | Primitives with a block of their own |
|---|---|---|---|---|---|
| `enc_streams`, width 32 | 3,104 | 0.149 / 0.535 | 0.269 / 0.602 | not meaningful (blocks off) | 6 by mode, but two primitives have no block that computes them |
| `enc_streams`, width 32, rate 0.03 | 3,104 | 0.106 / 0.346 | 0.207 / 0.476 | — | 5 |
| `enc_streams`, width 96 | 4,448 | 0.235 / 0.618 | 0.299 / 0.653 | — | 4 |
| `enc_streams`, width 32, 8,000 steps | 3,104 | 0.0115 / 0.083 | 0.155 / 0.213 | 0.9993 / 0.928 / 0.684 | 6 |
| `ev_lookup` (M = I, temperature set for initial logit spread 0.09) | 2,405 | 0.0085 / 0.049 | 0.0085 / 0.049 | 1.000 / 1.000 / 1.000 | 6, one duplicated |
| `ev_lookup`, gradient from load to weights cut | 2,405 | 0.0086 / 0.048 | 0.0086 / 0.048 | 1.000 / 1.000 / 1.000 | 6, two duplicated |
| untied match | 3,941 | 0.0107 / 0.050 | 0.0165 / 0.049 | 0.9998 / 0.9995 / 0.998 | 6 |
| `ev_lookup` on `sum`, rate 0.03 | 2,405 | 4.3e-3 / 8.8e-3 | 5.1e-3 / 8.7e-3 | — | 6 |
| untied match on `sum`, rate 0.03 | 3,941 | 1.1e-4 / 1.7e-3 | 5.0e-3 / 4.2e-3 | — | 6 |

For scale: `tok_streams` on names, same split, 0.0077 / 0.057 (my re-implementation). Supervised encoder alone (told the label, 4,000 steps): width 32, accuracy 1.000 in distribution, 0.995 at 1.5×, 0.954 at 2×, 0.746 at 3×, 0.991 with one demonstration; width 96, 1.000, 0.998, 0.977, 0.820, 0.997.

What these suggest, to be checked on twelve splits and not read as results: `ev_lookup`'s load leaves uniform within 500 steps and the bank sorts itself (the bootstrapping worry did not show on this seed); with m = 1 its seen error was 0.0100. Cutting the gradient changed nothing (against E7). The encoder arm needs about twice the steps and then loses under shift, as E1 predicts. The untied match equals the tied one on `select`.

## Checked and found correct

- The compute table of §4 (2,400; 4,880; 2,328; 2,456 or 1,304; about 580).
- The P rule ⌈4P/3⌉ gives 8 and 8 at P = 6; forced share and graph distance quoted for P = 12 match my computation.
- Part II's held-out conditions are at the level of the component compared (the load): evidence scale, count and noise (L58).
- Section 9's statement of what is arithmetic (an exact bank identifies perfectly) is correct, including under shift.
- Predictions for Part II are honestly marked blind; those for Part I as not blind.
