**Verdict: not sound as written.** The arms are mostly buildable and the carrier arithmetic at P = 6 is right, but the readings that decide adoption can be met without the lookup's own parts contributing anything, the lookup's defining part (keys computed from the weights) has no job on this carrier, and the P = 24 stage breaks in the reference code. Four blocking findings; all can be fixed in a revision, none needs a restart.

# Review of `experiments/theory/lookup/DESIGN.md` (revision 1, commit fe685a1), before any script

Reviewer: an agent that did not write the design. Read: the design, `docs/weight_lookup.md`, `tests/test_weight_lookup.py`, `tests/test_weight_mixing.py`, the worlds design, results and `worlds.py`, ledger L56–L59, I37, I38, history Entries 103–112. Probes are my own scratch arms (my reading of section 4, not the author's code), **one seed, one split, untuned** unless stated; by L30 they are indications, not readings. About 17 minutes of compute in total.

## Answers to the seven questions

| | Question | Verdict |
|---|---|---|
| A | Does it answer the request; is section 2's "at best equal" right? | It compares the arms the researcher named, but the claim is wrong as stated. The identities bound which functions can be expressed, not what training finds on held-out worlds (finding 3). |
| B | Arms well defined, one change each? | `tok_streams`, `probe_sg`, `wmix_sum` yes. `selfbid`, `hard`, `probe` and `wmix` each hide a second choice (findings 6–9). `lookup` = T2 + T3 + T4 does match `weight_lookup.md` restricted to one site. |
| C | Readings sound? | No. Adoption is decided by arithmetic plus a wide margin, there is no row that drops the lookup, and a better result is defined as a bug (finding 1). |
| D | Compute numbers | Correct to within 3 % under the stated rule. The rule is applied unevenly to `wmix`, and the saving belongs to the hard load alone (findings 1, 9). |
| E | The P dial | Sound at P = 12. At P = 24 the reference code does not fit in memory, the reference scale collapses, and several things besides P change (finding 4). |
| F | Worth running? | Only as a survival check, in reduced form. It cannot produce a result in favour of the lookup. A variant with evidence items is more informative (last section). |
| G | Anything else | Star instead of factorial (L56, L59); no identical-arm null for `tok_streams`; ratio criterion meaningless on `sum`; tuning at larger P unspecified (findings 5, 10, 11). |

## Findings

**1. BLOCKING — "Adopt the lookup" can be met with the lookup's own parts contributing nothing.**
- *The compute condition is not a condition.* "A third of its compute or less" is arithmetic: 580 against 2,328 multiply-adds at P = 6, 580 against 7,566 at P = 24 (my count from the configuration). It holds whatever is trained.
- *The saving belongs to T4, not to the lookup.* The design's own table gives `hard` and `lookup` the same cost (about 570). `hard` is `tok_streams` with stored keys and a separate bid net. So `hard` meets the adoption row whenever `lookup` does, with two fewer new parts. The contrast that would decide between them, `lookup` / `hard` at equal compute, is not in section 6.
- *The margin is as wide as the effect Entry 108 could not resolve.* Under the rule "not worse if the upper bootstrap bound is below 2", the stored Entry 108 runs give `tok_mono` / `tok_streams` on held-out error 1.53 [1.19, 1.97] at coverage 9 and 1.18 [1.05, 1.70] at coverage 7: the per-token monolith is "not worse" than streams on the recombination measure. (It fails on seen error, 3.03 [2.84, 3.19].) The same arm at rate 0.03 against 0.01 is 1.43 [1.32, 1.66]. A lookup that lost the whole streams-over-monolith gap on held-out worlds would pass.
- *No outcome drops the lookup.* The rows are: adopt; adopt in part; premise shown ("the lookup is the weight-space form that is not fragile"); premise not shown ("the case rests on cost alone", and cost is the arithmetic above). There is no row for "L worse and nothing survives", none for "unresolved", and a lookup that beats `tok_streams` by two is declared a bug in advance.
- *"Premise shown" over-attributes.* By U1 and U4 the lookup is unit gating, which is output mixing. If `wmix` is worse than `tok_streams`, that separates weight averaging from output summing; `tok_streams` is on the good side of it exactly as the lookup is.
- *Fix.* Make `lookup` / `hard` and `lookup` / `tok_mono` pre-fixed contrasts. Drop the compute clause from the adoption row and report cost as a table. Set the margin from the identical-arm null of `tok_streams` (finding 10), not at 2. Add a row "keep `tok_streams` with a hard load; the lookup's parts are dropped". Make the reading two-sided: a better result stands after a named list of checks (equal rates, equal width, the null).

**2. BLOCKING — On this carrier a key computed from the weights is a table, so "T3 survives" is decided by construction.**
- The query is a free embedding per token, and the key is a learned linear projection of C·d = 18 probe features. With K = 8 blocks (or 14 at P = 12) the features of the blocks are linearly independent (rank K − 1 after standardising: 7 and 13 in my check), so the projection can place every block's key anywhere. That is a stored key table under another parametrisation — the objection `weight_lookup.md` §8.3 makes to the outer-product key (U7).
- One-seed probe of this: a **placebo key** — the same key head applied to fixed random features that have nothing to do with the weights — gave seen 0.0084, held-out 0.0565, against `probe` 0.0083 / 0.061 (raw), 0.0083 / 0.053 (standardised), and `tok_streams` 0.0077 / 0.057. Nothing separates a key that reads the weights from one that does not.
- At P = 24, K = 26 exceeds 18 (feature rank 18 of 26), so the key becomes constrained by content for the first time. Whatever changes for `probe` or `lookup` between P = 12 and P = 24 is then confounded with K crossing C·d.
- The concerns of Entry 112 ("addresses can no longer move ahead of content", merging) are also not instantiated while the query side has a free table.
- *Fix.* Add the placebo-key arm as T3's control. Hold C·d below K at every P (C = 1 gives 6 features for 8 blocks) or make C a dial. Say in section 2 that T3 can only be tested for harm here.

**3. BLOCKING — Section 2's claim "the lookup's held-out error can at best equal `tok_streams`" does not follow from the tests.**
- U1 and U4 are identities between functions at given parameters (I ran all fourteen tests: they pass). They say which functions each arm can express. Held-out error depends on which of those functions training finds.
- On `select`, held-out error is the error of learned bids on comparisons never trained (worlds RESULTS, point 5). T2 replaces the bid's parametrisation: β_k · tanh(B_k h) has no bias, so the bid is an odd function of the state, as the teacher's trigger ⟨w_p, h⟩ is; the separate bid net has a free bias. That is an arm shaped more like the teacher (L58), and it can move held-out error in either direction.
- Held-out error already moves at a fixed function class: 1.43× between two learning rates of `tok_streams`, and 8 % from a floating-point reordering of the same run (finding 10).
- Ways to be worse by construction that the design does not list: a renormalised hard load freezes addresses at the switch (finding 6); K > C·d at P = 24 (finding 2).
- *Fix.* Replace "at best equal" with "the same functions are expressible; held-out error is not bounded by the identities", and remove the clause that treats a better result as a bug.

**4. BLOCKING for stage 2 — P = 24 breaks the reference code and moves the scale every reading sits on.**
- *Memory.* `references()` builds [held-out worlds, 2048 states, P, P] tensors in `Teacher.determined`. At P = 24 that is 240 × 2048 × 576 float64 = 2.1 GiB each, with at least four alive at once; it does not fit in 8 GB, and not twice on two processes. At P = 12 it fits but takes 11 s per call, and `run()` calls it in every run although it depends only on the split and teacher.
- *The sampler.* `world_order(P, split, first=7)` never returns for P > 7 (seven edges cannot 2-connect twelve primitives). The design's constructive sampler works (first draw accepted in 12 of 12 cases I tried), but it draws from a different distribution than the P = 6 sampler (always Hamiltonian; not nested), so P = 6 → 12 changes the sampler as well as P.
- *The references collapse* (exact, from the teacher, six splits, medians):

| P | trained worlds | forced first-pass decisions | `trans` | `blend` | `allmean` | mean graph distance of a held-out pair |
|---|---|---|---|---|---|---|
| 6 | 9 of 15 | 0.46 | 0.20 | 0.47 | 0.80 | 2.0 |
| 12 | 18 of 66 | 0.28 | 0.32 | 0.48 | 0.89 | 2.5 (max 4) |
| 24 | 36 of 276 | 0.15 | 0.40 | 0.49 | 0.96 | 3.1 (max 5–6) |

  (P = 24 with 96 states per world.) At P = 24, "beyond" (half of `trans`, 0.20) and "recombines" (half of `blend`, 0.245) nearly coincide, `allmean` is nearly "not moving", and 85 % of held-out decisions rest on bids extrapolating along chains of three or more trained comparisons. Fixed coverage per primitive does not hold difficulty fixed. It is a defensible choice, but then the forced share and the graph distance have to be reported at each P and P9 restated against them.
- *Other things that change with P and are not stated:* the embedding stays at 8 dimensions for 24 tokens; spare blocks fall from 2 of 8 to 2 of 26; the monolith width at matched parameters (56 at P = 12, 79 at P = 24 by my count, giving `tok_mono` 15,800 multiply-adds against 580) is not given; rates and step count at larger P (finding 11).
- Training itself scales as the design says: `tok_streams` took 44 s at P = 12 and about 28 ms per step at P = 24 (roughly 2 minutes a run), 5× the P = 6 run.
- *Fix.* Rewrite `determined` in chunks and compute references once per split. Use the constructive sampler at P = 6 too for stage 2, or say the sampler changes. Fix mv, K/P, C and monolith width as explicit rules in P.

**5. SHOULD FIX — A star, where L56 and L59 ask for ablations of the candidate and a factorial.**
- The design adds one part at a time to `tok_streams`. L56: "remove each part from the thing actually proposed." L59: when a hypothesis names several things, build the factorial. T2 and T3 both act through B_k; T3 and T4 interact through the switch (finding 6). "Adopt in part" would carry forward a combination, such as T2 + T4, that was never run.
- Stage 2 carries only single changes that were "worse or unresolved" at P = 6. But section 2 says starvation and merging are problems of many blocks. If `lookup` fails at P = 24 and its parts passed at P = 6, nothing at P = 24 says which part failed.
- *Fix.* Three more arms complete the 2 × 2 × 2 at P = 6 (T2+T3, T2+T4, T3+T4). Carry `hard`, `probe` and the three leave-one-out arms to stage 2 regardless of stage 1.

**6. SHOULD FIX — `hard` is underspecified, and the same schedule is a different treatment in `hard` and in `lookup`.**
- Which probability passes 0.8 is not said (I used the token-to-block load, averaged over tokens). Whether the top-1 load is renormalised is not said, and it matters: renormalised, the keys and embeddings get no gradient after the switch (in my run the mean top probability stayed at exactly 0.803 from step 1040 to the end), so a collision at the switch is permanent; un-renormalised (Entry 104's form) the address keeps sharpening (0.937 at the end).
- The switch fired at step 1040 in `hard` and at steps 80–120 in `lookup` (200 at P = 12), because computed keys sharpen faster. Inside the lookup, T4 is nearly "hard from the start".
- `tok_streams` trained soft is not sharp: mean top probability 0.855, minimum 0.577. Hardening it after training took seen error from 0.0077 to 0.115 (renormalised) or 0.166, and held-out from 0.057 to 0.158 or 0.139. P8 (under 10 % change, confidence 0.7) fails on this split by a wide margin, so the soft arm does need all its blocks.
- *Fix.* State the probability, the renormalisation and a fallback switch step; record the switch step and collisions at the switch; add `hard` from step 0 as the L56 control (Entry 104's review found the dense start was not always needed).

**7. SHOULD FIX — `probe`: workable at initialisation, but the normalisation and the projection's scale are hidden dials.**
- Raw keys are workable: with the 0.1-scale write matrices the probe features have rms 0.07 and the load logits a standard deviation of 0.066, against 0.089 for stored keys. The arm trained normally (above).
- Standardising over the bank makes the logits about ten times larger at initialisation (0.84) unless the projection is scaled down, and changes the trained address from soft to one-hot: final mean top probability 0.998 against 0.935 (raw) and 0.855 (stored keys). With standardised keys, hardening after training cost nothing (0.0083 → 0.0082 seen). So the choice decides P8, the switch time of T4 and the allocation measures.
- Choosing raw or standardised by seen error is a second tuned dial that only the new arms get.
- *Fix.* Fix the projection's initial scale so the initial logits match `tok_streams`; run raw and standardised as two named arms.

**8. SHOULD FIX — `selfbid` changes three things: the bid's source, the block width (16 → 23 at matched parameters) and the bid's bias.**
- One-seed numbers (seen / held-out): `selfbid` at width 16 (1,776 parameters) 0.0099 / 0.069; at width 23 (2,504) 0.0089 / 0.050; `tok_streams` at width 23 (3,152) 0.0086 / 0.045; `tok_streams` at width 16, 0.0077 / 0.057. Width alone moves held-out error by as much as the arm does.
- *Fix.* Report `selfbid` at width 16 and `tok_streams` at width 23 beside the matched-parameter arm; state whether the bid has a bias.

**9. SHOULD FIX — `wmix`: one arithmetic error, a criterion that cannot fail on `sum`, and no stop rule on the premise.**
- *Representability.* The W4 argument is right: every pair of primitives is a world, so all six need disjoint units, 6 × 6 = 36. The parameter ratio is not: width 36 with K = 8 is 4,400 parameters against 2,480, **1.77×, not 1.4×** (1.34× would need K = 6).
- *The ratio criterion on `sum`.* `tok_streams` sits at 1.5e-6 there. Any arm that is not exact is "worse": `tok_mono`, which is *full* in 12/12 splits, is 546× [485, 736] on held-out error. P2 is then decided by the baseline's precision. Use the regime (half of the smaller reference) on `sum`.
- *One-seed probes.* `select`, width 16: seen / held-out 0.0084 / 0.058 at rate 0.01, 0.0066 / 0.070 at 0.03, 0.0126 / 0.066 at 0.003 — level with `tok_streams`, with a soft address (mean top probability 0.56–0.73). `sum`, rate 0.03: width 16, 5.4e-4 / 0.46; width 36, 6e-5 / 0.19 (references 0.57 and 0.53; `tok_streams` 2.7e-6 held-out). So the premise may show on `sum` only, as failure to recombine at good fit — which is what W4 predicts.
- *Order.* Entry 107 level 2 puts the phenomenon first. Run `wmix` alone as stage 1a with a stop rule, and state which error "premise shown" refers to (P1 and P2 are about seen error, the reading about held-out).
- *Cost.* 2,456 is right if all K matrices are mixed per state. But the per-token mixtures Σ_k a_tk W_k do not read the state and can be cached per world, as the design allows for computed keys; then it is 1,304.
- `wmix` puts both the load and the competition in weight space. A version that mixes weights within a token and outputs across tokens would say which of the two hurts.

**10. SHOULD FIX — Noise floor, margin and sample size.**
- The quoted floor (0.94–1.05) is `tok_mono` and `mono` against themselves. There is no stored identical-arm null for `tok_streams`, the denominator of every contrast. My re-implementation of `tok_streams`, algebraically identical and differing only in operation order, gave 0.00765 / 0.0570 against the stored 0.00735 / 0.0528 on split 0. Run `null_tok_streams` at each P.
- With six splits the bootstrap cannot declare "not worse" for effects of this size: the stored `tok_mono` / `tok_streams` ratios on the first six splits give 1.66 [1.06, 2.27], "unresolved", where twelve give "not worse". The adoption row needs "not worse at the larger banks" on six splits.
- "Failed run" is defined against the arm's own median; an arm that fails in most runs has none. Define it against `tok_streams`.

**11. SHOULD FIX — Tuning at larger P (L57).** The design tunes new arms at P = 6 and reuses Entry 108's rates for old ones. Nothing is said about rates or the 4,000 steps at P = 12 and 24. Tune every arm at each P or state that none is, and report the ordering at equal rates.

**12. MINOR.**
- Compute counts check out: `mono` 2,400; `tok_mono` 4,880; `tok_streams` 2,328 (design: about 2,280); `hard` 580; `lookup` 586 at width 22 (design: about 570).
- Stage 0's "identical to the last digit" holds on this machine for the repo code (below), but new-arm code can be validated only by forward identities, not by matching trained numbers.
- "Blocks never loaded" (P10) needs a definition under a soft load (argmax).
- "Against the field" is a contrast predicted to say nothing; drop it from the readings.
- Twelve splits share three teachers; the bootstrap treats them as independent (inherited from Entry 108).

## Is it worth running, and what would be more informative

Plainly: as designed, no result of this experiment can favour the lookup. Accuracy is conceded in section 2 (and on `select` my one-seed probes at P = 6 and 12 put every new arm at 0.0066–0.0126 seen and 0.042–0.070 held-out, beside a baseline at 0.0077 / 0.057 at P = 6 and 0.0117 / 0.052 at P = 12). Cost is arithmetic and belongs to the hard load. What it can produce is negatives: a part that breaks training at a large bank, and weight mixing failing to recombine on `sum`. Those are worth a cut-down run — `wmix` first, the 2 × 2 × 2, the placebo key, P = 12 and 24 with fixed references — but under the title "does the lookup survive", not "which configuration is carried forward".

**Section 10, point (i), the evidence carrier: setting it aside is the design's main weakness.** The lesson L58 states the case: "when the main outcome is predicted by construction, change the design", and "ask of a reused testbed what question it was built for". The worlds carrier was built to separate the read from the block, with tokens that name primitives. A name can be addressed by a table, so content keys cannot matter (finding 2). The variant is one change to this carrier: each item of a world is a few demonstrations (x, f_p(x)) of a primitive instead of its token. Then:
- the lookup's key can be the block's own output at the demonstrated states, matched against the demonstrated outputs — the probe states come from the item;
- the stored-key arm needs a learned encoder from demonstrations to an address;
- the per-token monolith has to infer a function from examples, so "monolithic against lookup" is no longer settled by naming.

It meets Entry 107's property (a), unmet in Entry 108, and bears on Entry 109's point 6. What would still be by construction: retrieving a block that was changed or added after training (that is the claimed capability; measuring it is a check). What would not: whether content addressing and content can be trained together at all, how each arm behaves on held-out pairs when the item is evidence, and on demonstrations from states outside training. I think this is valuable enough to design before stage 2 of the present plan, because it is the only version of the comparison in which the three configurations can come apart for a reason tied to the lookup.

**Point (ii), several sites: acceptable to defer, but the reason given is half right.** That K blocks cannot hold K² composites is by construction. Whether per-site key heads find the factorisation from evidence about composites is not. The same "by construction" argument would have ruled out Entry 108, whose finding was the part that was not constructed. Since one key head per site is half of Entry 112's proposal, the design should say in section 1 that it tests the single-site restriction, where "one head per site" and "one head" are the same thing.

## Probes (one seed; my scratch arms; indications only)

`select`, coverage 9, split 0, teacher 10, model seed 0, rate 0.01, 4,000 steps unless stated. References for this split: `allmean` 0.79, `blend` 0.50, `trans` 0.27, forced 0.42.

| Arm (my reading) | Parameters | Seen | Held-out | Final load: mean / min top probability | Note |
|---|---|---|---|---|---|
| `tok_streams`, repo code | 2,480 | 0.007354 | 0.05276 | — | equals the stored run to every printed digit |
| `tok_streams`, my re-implementation | 2,480 | 0.0077 | 0.057 | 0.855 / 0.577 | top-1 after training: 0.115 / 0.158 renormalised, 0.166 / 0.139 not |
| `hard`, not renormalised | 2,480 | 0.0077 | 0.067 | 0.937 / 0.817 | switch at step 1040 |
| `hard`, renormalised | 2,480 | 0.0070 | 0.068 | 0.803 / 0.556, frozen | switch at step 1040 |
| `selfbid`, width 16 | 1,776 | 0.0099 | 0.069 | 0.874 / 0.636 | |
| `selfbid`, width 23 | 2,504 | 0.0089 | 0.050 | 0.893 / 0.737 | |
| `tok_streams`, width 23 | 3,152 | 0.0086 | 0.045 | 0.833 / 0.498 | |
| `probe`, raw | 2,578 | 0.0083 | 0.061 | 0.935 / 0.615 | |
| `probe`, standardised | 2,578 | 0.0083 | 0.053 | 0.998 / 0.994 | top-1 after training: 0.0082 / 0.054 |
| `probe_sg`, raw | 2,578 | 0.0088 | 0.060 | 0.941 / 0.826 | |
| placebo key (random fixed features) | 2,578 | 0.0084 | 0.057 | 0.950 / 0.717 | |
| `lookup`, raw keys, width 22 | 2,498 | 0.0096 | 0.051 | 0.996 / 0.989 | switch at step 120 |
| `lookup`, standardised keys | 2,498 | 0.0092 | 0.056 | 0.999 / 0.998 | switch at step 80 |
| `wmix`, rate 0.01 / 0.03 / 0.003 | 2,480 | 0.0084 / 0.0066 / 0.0126 | 0.058 / 0.070 / 0.066 | 0.73 / 0.56 / 0.78 mean | |
| `wmix` on `sum`, width 16, rate 0.03 | 2,480 | 5.4e-4 | 0.46 | 0.79 / 0.71 | references 0.57, 0.53 |
| `wmix` on `sum`, width 36, rate 0.03 | 4,400 | 6.0e-5 | 0.19 | 0.65 / 0.51 | stored `tok_streams`: 1.5e-6 / 2.7e-6 |

Every arm assigned the six tokens to six different blocks.

P = 12 (constructive sampler, split 0, K = 14, 18 trained worlds; references `blend` 0.48, `trans` 0.34): `tok_streams` 0.0117 / 0.052, twelve tokens on twelve blocks; `hard` 0.0110 / 0.061 (switch at step 1200); `lookup` 0.0114 / 0.042 (switch at step 200). No training was run at P = 24.

One probe (the references at P = 12 and 24) ran 195 s as a single script, over the two-minute limit I was given.

## Checked and found correct

- Tests U1–U8 and W1–W6 pass (14 of 14), and U4 does establish that `tok_streams` is the lookup with stored block-level keys, a separate bid net and a soft load.
- `lookup` = T2 + T3 + T4 is `weight_lookup.md` §4 with §8's block-level probe key, for a block with one site.
- The stored Entry 108 `tok_streams` run reproduces exactly on this machine; its 2,480 parameters and the 12-split contrasts recompute from the JSON files.
- W4 arithmetic: `wmix` on `sum` needs width 36; on `select` the target gate is one-hot.
- Coverage arithmetic: 1.5 P trained worlds is 60 %, 27 %, 13 % of all worlds and three trained worlds per primitive.
- The constructive sampler passes `graph_ok` at P = 12 and 24.
- Compute per pass under the design's rule, and the training-time scaling with P.
- Raw probe keys are usable at initialisation despite the 0.1-scale write matrices.
- Honoured from the lessons: predictions committed with scripts (L56), rates extended until interior and contrasts at every qualifying pairing (L57), the premise's failure is at least included as an arm (L58), review before any run (L59).
