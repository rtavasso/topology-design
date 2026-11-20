**Verdict: not sound as written.** The addressing ladder of stage 2 (encoder → match → match tied to the applied blocks) is sound and was already reviewed. What revision 4 adds — the 2 × 2 as a test of static against context-dependent weights, stage 1's stop rule, the capacity stage (C2), the stage that adds a primitive (C3 iii), and the novelty claims — each has a blocking fault. All can be repaired in a revision; none needs a new carrier. The contribution as stated is weak: one claim is a replication (C1), one cannot come out in its favour on this carrier (C2), and one has closer prior art than the literature notes list and is largely arithmetic here (C3).

# Review of `experiments/theory/lookup/DESIGN.md`, revision 4 (commit 68f2739, sections 1–13), before any script

Reviewer: an agent that wrote none of the design. Read: the design; history Entries 108–120; the last section of `docs/literature.md`; both earlier reviews; revision 3 (`523106a`); `worlds/RESULTS.md`, `worlds.py`, `viz_model.py`; `docs/weight_lookup.md` §8–9; ledger L56–L60, I38, I39 and the direction update of 18 Nov. Web checks are cited where used. Training probes are **one seed, one split** (split 0, teacher 10, model seed 0), on the names carrier with the repo's own `worlds.py` unless stated; by L30 they are indications, not readings. Scratch code is outside the repo.

## Answers to the seven questions

| | Question | Verdict |
|---|---|---|
| A | Is the contribution well posed, supported, new? | Weak as stated. The thesis's dichotomy (compute against recognise) does not hold as worded, and "fixed weights cannot" is false of the proposed arm itself (finding 5). C1 is mostly a replication of Entry 108 with one open cell; C2 is by construction in one half and cannot be met in the other (finding 2); C3 has prior art the notes miss (finding 5), and the match arm's side of its third part is arithmetic (finding 3). Strongest single claim: C3 (i), that a bank can sort itself while its addressing reads its own unfinished contents — the one already probed. The size of the contrast in C3 (iii) did survive the simpler baseline I expected to erase it (one seed). |
| B | Does the design answer Entry 117's question; is the 2 × 2 clean? | The binding factor is clean in parameters among bank arms (same parameter set, different forward pass) but bundles two operations, of which one does the work on one seed (finding 2). The weights factor is four changes at once, and nothing is held out at the level of the weights (finding 1). "Static" is drawn at additive conditioning, which puts a gate per unit on the context-dependent side without an arm for it. |
| C | Stage 1's stop rule and section 8's readings | The stop rule cannot fire and one of its measures is undefined for a static arm (finding 4). Two C1 readings are decided by stored numbers (finding 6). The matching arithmetic is well defined, with two surprises (finding 6). |
| D | Stage 3 | Not the theorem's setting; K = 6 is not exact capacity in the theorem's sense; the pooled gate has a memorising solution at every K, in the gate; one-seed probes at K = 6, 8, 24 show no recombination at any K and no memorising blocks at K = 24; a pooled gate whose per-item loads are normalised first recombines as well as the per-item arm (finding 2). |
| E | Stage 4 | Under-specified in five places; the match arm's and the static arm's outcomes are by construction and the stored-key arm's is set by an unstated loss; it cannot separate the tie. The simpler baselines are needed but, on one seed, do not erase the contrast (finding 3). |
| F | Anything else | Cost is low by about three times; the researcher's "expanding the space" is not measured; L58 and L60 are not honoured for the weights factor and for the tie (findings 1, 3, 7, 8). A more valuable contribution is available (last section). |
| G | Section 13 | The condition is right in substance and too narrow in form; "cannot show it" is correct, one supporting sentence is not; the reason given for the ordering holds for only one of the three candidate signals (finding 9). |

## Findings

**1. BLOCKING — The "weights" factor of the 2 × 2 is four changes at once, and nothing is held out at the level of the weights.**

*What the arms are in the code.* `static_item` is the `tok_mono` form and `bank_item` the `tok_streams` form (`worlds.py`, `Model.forward`). Between them:

| | `static_item` | `bank_item` |
|---|---|---|
| how context reaches the block | concatenated to the state | softmax selection over blocks (the factor intended) |
| function class | generic two-layer block | one-layer blocks V tanh(B h), each **containing a teacher primitive exactly** |
| bid | one read-out of the block's own second layer, a joint function of (state, code), with bias | a separate net per block reading the state only (832 of the bank's 2,432 parameters) |
| normalisation | none | each item's load sums to one |
| compute at equal parameters | 4,680–4,880 multiply-adds per pass | 2,344 |

Section 4 lists the second row as a "limit carried over". Inside this factor it is not a limit, it is a confound: every seen-error contrast between the two arms carries it. On names the stored runs already give `tok_mono` / `tok_streams` = 3.03 [2.84, 3.19] on seen error at equal parameters (recomputed from the JSON files).

*Nothing is held out from a per-item branch* (L58: "hold out at the level of the component being compared"). A branch sees a state and one item; both are covered in training in both arms. On held-out worlds the only new thing is the comparison of two bids, and the bids are parametrised differently in the two arms (third row). So `bank_item` against `static_item` on held-out error measures fit and bid parametrisation. It cannot measure anything about context-dependent weights. Entry 108 said this of its own result ("whether that is fit or recombination is not resolved"): the held-out minus seen difference is 0.034 against 0.031 (recomputed). Revision 4 keeps the same measure, so the reading "weights also recombine" can be met by fit alone. The 3× static block does not remove this: in Entry 108 a 6.8× per-token block reached ×0.64 of its seen error, still twice the streams' (the results reviewer's probe, three splits).

*Where static and context-dependent weights could differ for a reason of their own:* when the **code** is off its training distribution. A static block is then an unconstrained function of an unfamiliar vector (the failure Entry 108 found for the summed read); a selecting bank can only output a stored primitive, possibly the wrong one. The design has the condition (evidence at 1.5–3×) and both arms share the encoder, but section 8 reads shifted evidence only among bank arms.

*Where the line between static and context-dependent is.* Section 1 defines static as "context reaches it only as activations added to its input". Then a static block whose units are gated by the code (FiLM; the `unit` arm of `worlds.py`; what HyperNetworks 2016 fell back to) is context-dependent, and by U1 it is the lookup at unit grain. And `bank_match` has fixed parameters like any network; what depends on context is an activation (the load). So the factor being tested is additive against multiplicative conditioning. That is the subject of [Jayakumar et al. 2020, *Multiplicative Interactions and Where to Find Them*](https://iclr.cc/virtual/2020/poster/1710) (in `literature.md`'s unread list) and of Entry 105 (80–700× on fit). The "fit" third of C1 is therefore not new.

*Fix.*
- Add the arm that sits on the line: `static_item_gated` — the same two-layer block, the code entering as a gate per hidden unit — and say which side of the definition it is on.
- Break the bundle: `bank_item` with a shared bid head, or `static_item` with a separate bid net; and one teacher outside the block's class (two-layer primitives, or width above w) so that no arm contains the teacher.
- Measure recombination where it happens: the share of held-out decisions not forced by trained comparisons that match the teacher (worlds stage 3 has the code), and held-out error at matched seen error (AGENTS.md rule 4).
- Add a reading for `static_item` against `bank_item` under shifted evidence, with the "applies one wrong primitive" reference beside it.

**2. BLOCKING — Stage 3 does not instantiate the theorem; "C2 supported" cannot occur on this carrier, and the outcome that will occur has a cause the design does not look for.**

*The theorem's condition* ([Schug et al. 2024](https://arxiv.org/html/2312.15001v2), Theorem 1, fetched for this review; first 100,000 characters of the page only): student dimensions equal to the teacher's, "M̂ = M ≤ n, ĥ = h" — the number of modules **and the hidden width**. The code is one free vector per task, inferred by optimisation on that task's support data, on held-out tasks too. There is no encoder and no per-item code anywhere in the paper, so C2's distinction (one code against per-item codes) is not already in it.

Against that:
- *K = 6 is not exact.* Six blocks of width 16 are 96 hidden units against the teacher's 36. A width-16 block can hold two primitives (12 units). The dial in K alone starts 2.7× overparameterised in the theorem's own terms.
- *`select` is not the theorem's task.* A state-dependent choice between modules is not a fixed combination of weights. `sum` is (gates as the code, blocks as modules, by W9). Stage 3 is on `select` only.
- *The pooled arm is not the theorem's learner.* Its code is an amortised linear map of the summed item codes into one softmax.

*The pooled gate has a memorising solution at every K — in the gate, not in the blocks.* Its logits are s_k = u_k[a] + u_k[b] + bid_k(h), with u = (keys)(codes)ᵀ a free K × P matrix. A trained world {a, b} constrains only the sums u_k[a] + u_k[b]. The decision between the blocks of a and b in a **held-out** world turns on u_{k_a}[b] − u_{k_b}[a], which enters no trained world's decision; training only asks those cross terms to be low enough inside other sums. So the offset of every held-out decision is free. Per-item loads remove exactly this: a saturated load is one-hot and only the bids are compared. Both halves of C2 are thus structural here, in opposite directions — the design says so for the per-item half and calls the pooled half "measured".

*One-seed probes* (`streams_self` from `worlds.py`; K is a config key; rate 0.03; 4,000 steps):

| Task, gate | K | Seen | Held-out | Bar for "recombines" | Blocks computing a primitive within 5 % | Blocks used by exactly one trained world | Unused blocks |
|---|---|---|---|---|---|---|---|
| `select`, softmax | 6 | 0.0087 | 0.454 | 0.252 (half of `blend`) | 6 of 6 primitives | 0 | 0 |
| `select`, softmax | 8 | 0.0068 | 0.381 | 0.252 | 6 | 0 | 1 |
| `select`, softmax | 24 | 0.0075 | 0.375 | 0.252 | 6 | 0 | 18 |
| `sum`, softmax | 6 | 8.7e-6 | 0.328 | 0.266 (half of `one`) | 6 | 0 | 0 |
| `sum`, softmax | 24 | 6.9e-6 | 0.287 | 0.266 | 5 | 1 | 14 |
| `sum`, sigmoid | 6 | 1.5e-5 | 0.164 | 0.266 | 3 | 0 | 0 |
| `sum`, sigmoid | 24 | 8.2e-6 | 0.236 | 0.266 | 3 | 0 | 14 |

(K = 8 on `select` reproduces the stored run to every digit. Stored `sum`, sigmoid, K = 8, split 0: 0.088; twelve-split median 0.15.)

What they indicate:
- On `select` the pooled arm does not recombine at K = 6, 8 or 24, and capacity does not make it worse. In the stored twelve splits at K = 8 it is under half of `blend` in 1 of 12.
- At K = 6 the bank is exactly what the theorem asks for — six blocks, each computing one primitive to within 0.2 %, each used by two to four trained worlds — and it still fails. The fault is where the derivation puts it. Regressing the gate's log-odds between the pair's two blocks on the teacher's trigger difference gives an offset of at most 0.12 (in trigger units) on every trained world and −1.28 to +1.14 on the six held-out worlds; choice accuracy 0.98–0.99 trained, 0.63–0.91 held-out; 0.70–1.00 of the gate on the pair's blocks. The same at K = 8 and K = 24 (offsets up to ±1.1).
- On `sum` with a softmax the same thing shows as the split between the two blocks: 0.495–0.507 on every trained world, 0.04–0.89 on held-out worlds.
- At K = 24 the "one block per world" solution is not what training finds: 14–18 blocks are unused and at most one block serves a single world.
- On `sum` with a sigmoid gate (the form the repo's own tuning chose) the blocks are **mixtures**: only three of six primitives have a block within 5 %. That is the theorem's "up to a linear map". Stage 3's measures "how many blocks compute a primitive, how many are duplicates" are not defined there.

So the row that will be met is "pooled composition fails at every capacity", the design will conclude "the condition is not capacity", and it has no measurement that says what the condition is.

*A probe of what the condition is.* I changed one line of the pooled arm: each item's load over blocks is normalised first (softmax of code · key per item), the two loads are summed, and the log of that sum replaces the raw key score inside the **same single softmax over blocks**. There is still no competition between items. One seed, K = 8, rate 0.01: seen 0.0092, held-out 0.056 — the per-item arm's numbers (0.0074 / 0.053 stored) — with held-out offsets of at most 0.06 trigger units and choice accuracy 0.92–0.99 on held-out worlds. In the saturated limit this arm equals `tok_streams` algebraically, so its *ability* to recombine is by construction; what the probe adds is that normalising each item's address before items meet is sufficient, and competition between items is not needed. "Pooled against per item" is therefore not one factor either: it bundles normalisation of the address with competition at the state.

*The per-item arm across K* (one seed, rate 0.01): seen / held-out 0.0080 / 0.048 at K = 6 and 0.0075 / 0.045 at K = 24. Flat, as P6 predicts and as the structure forces. At K = 24 a primitive is spread over several softly loaded blocks: only three of six primitives have any single block within 5 % of them, yet the error is unchanged. Section 6's identification ("its most-loaded block computes its primitive") would report failure there. Identify by the loaded mixture Σ_k load_k f_k, not by the top block.

*What the paper itself reports on capacity* (its sections 3.3 and 4.1, fetched; the table in Appendix B.1 was not reached): slight overparameterisation **helps** identification; it "noticeably degrades" only beyond ĥ = 8h and M̂ = 2M, for discrete tasks; and on its larger task "a certain overparameterization is necessary for the optimization to succeed". Section 3's "no spare capacity" is the theorem's sufficient condition, not the paper's empirical finding, and a dial that starts at 2.7× the teacher's width sits inside the range the paper found harmless.

*Is the measurement well defined?* "Used by only one trained world" needs a threshold under a soft gate (I used: more than 10 % of a world's gate mass on the first pass) and a planted case showing it detects a memorising bank (AGENTS.md: "confirm that its measure can detect the target effect on a planted case"). A (world, primitive) block on `select` computes a primitive, so by function it is a duplicate; only the loading pattern tells it apart.

*Other things the dial moves:* parameters ×3.9 from K = 6 to 24 (1,872 → 7,344 on names); the softmax's size; whether rates are tuned per K.

*Fix.* Either drop C2, or test it where the theorem lives: `sum`; exact capacity as K = 6 **and** w = 6, with K and w as separate dials; a pooled arm whose code is a free vector per world re-fitted on a held-out world's data, beside the amortised one; softmax and sigmoid named as two arms; outputs regressed on the six primitives instead of block-by-block matching. Report the gate offset on held-out worlds in every pooled run. Restate C2 as what the carrier can decide (see the last section).

**3. BLOCKING — Stage 4 is under-specified, three of its outcomes are set by the protocol, and it cannot separate the tie.**

*Not specified, and each changes the result:*
- How many demonstrations of primitive 6, and how many steps, are "the same budget".
- The loss for `static_item`. The data are (x, f₆(x)) with no worlds. The only reading I can find is a one-item application — the block on [x; code(other demonstrations)] regressed on f₆(x) — which is a context size never seen in training.
- The loss for `bank_item`'s new key. "Trained so that primitive-6 evidence loads it", from positives only: any key with a large positive projection on the new code satisfies it, and nothing in the data says that old evidence must not load the new block. The arm's interference is produced by the rule "no rehearsal data".
- Which worlds over primitives 1–5 are trained (all ten?), and what the six-primitive reference bank is trained on. If on all fifteen worlds, its "new-world error" is a seen error near 1e-6 on `sum`, and "within 2×" is a ratio to the baseline's precision — the fault both earlier reviews found on `sum` (R1.9, R2.4).
- `sum` "without bids" is not in the code: `tok_streams` always pools items by a softmax over bids and `viz_model.py` always computes a share. It is a new forward pass (update = Σ_items Σ_k load · f_k) and needs its own identity tests. It is well defined; with it the scalar c is 1 and a block is f_p, not 2 f_p.

*By construction.*
- On `sum`, nothing is held out from a per-item branch (Entry 108's own words). Given a block for f₆ and a correct load, every world containing primitive 6 is exact.
- `bank_match`: one block of the teacher's class is fitted to noiseless samples of a function in that class, and an exact block is its own perfect key (R2: 1.000 at every scale). P9 is arithmetic plus the precision of one regression.
- `static_item`, whole block fine-tuned on one primitive with no rehearsal: forgetting. P10 cannot fail except by a small step count.
- `bank_item`: decided by the positives-only rule above.

*Simpler baselines that C3 (iii) has to beat.*
- **A prototype key.** Set the new key to the mean code of the primitive-6 demonstrations; train nothing. This is nearest-class-mean / prototypical classification, the standard few-shot way to add a class to an encoder. I expected it to erase the contrast. On one seed it did not (below).
- **Replay from the bank's own blocks.** Any bank arm can generate (x, f_k(x)) from its stored blocks and use their codes as negatives for the new key, or as a full training set for re-fitting the encoder. That is not rehearsal data. With the encoder trainable this is "re-indexing", and it is the strongest form of the stored-key arm; it is not in the design.
- **`bank_match_untied`.** Copy the new block into the matcher set. It is then identical to the tied arm. Section 8's row "then the tie's case rests on stage 4 alone" rests on nothing: the untied arm is not in stage 4, and if it were it would tie.

Underneath, C3 (iii) is the textbook property of classifying by one model per class: adding a class needs only that class's model, and a discriminative encoder has to be re-fitted. That holds for any analysis-by-synthesis retrieval, tied or not.

*One-seed probe* (my scratch arms: `sum` without bids, evidence items with m = 2, all ten worlds over five primitives trained, eight blocks; then 256 demonstrations of the sixth primitive; details under "Probes"). Error relative to the teacher's change:

| Arm after adding one block | New worlds (5) | Old worlds (10), before → after | New item's top block is the new one | Old items whose top block became the new one |
|---|---|---|---|---|
| match (the new block fitted by regression, nothing else touched) | 8.1e-6 | 2.0e-5 → 2.0e-5 | 1.000 | 0.000 |
| stored key, encoder frozen, key trained on positives only (the design as I read it) | 0.51 | 5.6e-5 → 1.24 | 0.98 | 0.58 |
| stored key, encoder frozen, prototype key, no training | 0.40 | 5.6e-5 → 0.78 | 0.80 | 0.40 |
| stored key, encoder frozen, key trained with replay from the stored blocks | 0.21 | 5.6e-5 → 0.086 | 0.79 | 0.06 |

Reading, as an indication only. The match arm's row is the arithmetic it was said to be. The stored-key rows are not: the encoder trained on five primitives does not give the sixth a code of its own. A nearest-class-mean rule on the frozen encoder (old class means taken from replay of the stored blocks) identifies old items in 0.95–1.00 of cases and the new primitive's items in 0.77, in the 8-dimensional code and in the 32-dimensional hidden layer alike. So with a frozen encoder the contrast is large whatever key rule is used, P11 (0.55) is under-confident, and the positives-only rule adds interference of its own (0.58 of old items captured). What is left open is the trainable encoder with replay, which is the comparison that decides whether "without re-indexing" is a saving or a necessity.

*Fix.* Specify the five items. Add the prototype key, replay from blocks with the encoder frozen and trainable, and the untied arm. Use absolute thresholds on `sum`. Say in section 8 which outcomes are arithmetic. If the tie is to have a test, it is not adding a block but **changing one**: fine-tune a stored block to a new function and ask who still finds it — the tied key follows, an untied matcher and a stored key go stale (`weight_lookup.md` §8.4, "no table to go stale"). That too is by construction, but it is the claim.

**4. BLOCKING for stage 1 — the stop rule cannot fire, and one of its two measures is undefined for a static arm.**
- *Undefined.* Section 6 defines identification through "its most-loaded block". A static arm has no blocks and no load. P1 and the stop rule both read "static_item identifies at 0.99".
- *First clause.* "Not worse than the names ceiling" means the upper bound of the ratio to `tok_streams` on names is below 1.5. With identification free, the static per-item block already fails that: `tok_mono` / `tok_streams` on names is 1.53 [1.19, 1.97] on held-out error (stored, coverage 9; recomputed). The clause fails before evidence enters.
- *Second clause.* R2's probe: an encoder trained **with the label** identifies 0.954 at 2× (0.977 at three times the width). No learned encoder reaches 0.99 there.
- So P1 (0.7) is close to certain and the stage is not a gate. And the clause compares a static block on evidence with a bank on names: two changes. The ceiling for a static arm is `tok_mono` on names.
- What stage 1 would show — identification falling under shift — is a property of the encoder, which `bank_item` shares. It is not a failure of static weights.
- *Fix.* Define identification for a static arm (for instance: the primitive nearest in function to the branch's update at the item's code). Use `tok_mono` on names as the ceiling. Set the threshold from the supervised encoder's ceiling. Or fold stage 1 into stage 2 as rows.

**5. SHOULD FIX — C3's novelty is overstated, and the thesis is not well posed as worded.**

*Prior art not in `literature.md`.*
- [Haruno, Wolpert & Kawato, *Multiple Paired Forward-Inverse Models*](https://papers.neurips.cc/paper_files/paper/1998/file/228499b55310264a8ea0e27b6e7c6ab6-Paper.pdf) (MOSAIC; fetched). Each module's responsibility is a softmax over the modules' prediction errors on what is observed; responsibilities gate both the learning and the output; modules start from random initial conditions and sort themselves. A module is a predictor paired with a separate controller — that is `bank_match_untied`, trained from scratch, in 1998.
- [Alet, Lozano-Pérez & Kaelbling 2018, *Modular meta-learning*](https://ar5iv.arxiv.org/html/1806.10166) (fetched). The structure for a task is chosen by scoring compositions of **the modules that will be applied** on the task's own data; modules are trained from scratch jointly with that search (BounceGrad); at test the weights are frozen and only the structure is searched. That is the tie, with a search in place of a softmax.
- Jacobs et al. 1991 (gating by posterior responsibility) is in the unread list and is the origin of both.

So C3 (i) — "trains from scratch together with the blocks" — has precedent in both the untied and the tied form. "Not found as a trained layer" should be narrowed to what is left: a differentiable soft match of the applied blocks against in-context demonstrations, as the address inside a looped block. C3 (ii) is by form on this carrier (section 4 says so). C3 (iii) is finding 3.

*The thesis.* "Matters less for what a model can compute than for what it can recognise" sets recognising against computing, and recognising is a computation. "Fixed weights cannot [serve as their own index]": `bank_match` has fixed weights, and static transformers select among functions from demonstrations in context (the in-context regression literature; from memory, not re-read). What is being claimed is narrower and should be said that way: when primitives are stored as separately evaluable units, the function that is applied can also be the function that recognises, so there is no second model to keep in step. That is the tie — and the one-seed probes of R2 found the tie adds nothing during training.

*Which claim is strongest.* C3 (i): whether a bank whose address reads its own unfinished contents sorts itself. It is not by construction, a 2025 paper names it as a problem, and the measurement (stage 5) is cheap. It is also the least blind: R2's probe saw it sort within 500 steps.

**6. SHOULD FIX — Section 8, C1: which readings can fail, and the matching arithmetic.**

| Reading | Can it fail? |
|---|---|
| "both per-item arms better than both pooled arms" | Not realistically. Entry 108: 0.05 and 0.10, 12/12. `static_pooled` is the baseline L59 records as failing for a trivial reason. |
| "weights buy fit" (`bank_item` better than compute-matched `static_item` on seen error) | Not realistically. The compute-matched static block has **half** the bank's parameters (below), and at equal parameters the stored ratio is already 3.0. P4 at 0.65 is under-confident. |
| "binding, not weights" against "weights also recombine" | Yes; this is the open cell (stored: 1.53 [1.19, 1.97]). But see finding 1 on what it measures. |
| "static weights suffice" | No (finding 4). |

Arithmetic for d = 6, K = 8, w = 16, code 8, bid width 13 (my count; bank: blocks 1,536 + bids 832 + keys 64 = 2,432, encoder 672 extra; 2,344 multiply-adds per pass with every block evaluated):

| Static arm | Width | Parameters (block) | Multiply-adds per pass |
|---|---|---|---|
| `static_item`, compute-matched | 25 | 1,200 (0.49× the bank) | 2,300 |
| `static_item`, parameter-matched | 39–40 | 2,418–2,520 | 4,680–4,880 (2.0× the bank) |
| `static_item`, 3× parameters | 75 | 7,350 | 14,400 (6.1×) |
| `static_pooled`, compute-matched | 39–40 | 2,379–2,480 | 2,301–2,400 |
| `static_pooled`, parameter-matched | 39–40 | the same arm | |

Both matchings are well defined. For `static_pooled` they coincide, so its "three widths" are two. For `static_item` compute-matched is the *smallest* arm. Say whether the encoder and the bid nets count towards "parameters".

Also: the rule for a reading takes its noise envelope from "the reference arm"; the 2 × 2 has contrasts between two arms neither of which is a reference. Name the arm whose nulls are used for each contrast.

**7. SHOULD FIX — The researcher's "expanding the space" is not measured.** The ledger's direction update names two unmeasured predictions: how many primitives can be held at a fixed active width, and how much learning one disturbs the others. Stage 4 touches the second for one added primitive. The first is nowhere: P is 6 throughout and storage at fixed compute is set aside as PEER's. Section 3 calls C2 "the precise form of the researcher's expanding the space", but K with P fixed is redundancy, not storage. Say that the design does not address it, or add a small dial in P at fixed active width.

**8. SHOULD FIX — Cost, tuning and power.**
- *Cost.* A K = 8 run took 38 s for 4,000 steps here and a K = 24 run 71 s, on names (no encoder, no per-state demonstrations). Section 6 tunes four rates × three step counts on three splits per arm: 36 runs averaging 9,300 steps, about an hour per configuration on one process. Stages 1–3 have roughly 30 configurations (static arms at three widths, bank arms on two tasks, three bank arms at four K, null envelopes). Tuning alone is then about 15 hours on two processes, before the twelve-split runs and the pairings "within 2× of the best". Nine hours is low by about a factor of three if L57 is honoured per arm and per K.
- *Stage 4* reads curves over adaptation steps. Pre-name the step at which P9–P11 are read.
- *Margins.* "Not worse" at an upper bound of 1.5 on twelve splits can be resolved only for ratios near 1 with tight spread; the stored same-parameter contrast has an interval of [1.19, 1.97]. Expect "unresolved" to be common, and say what is done then.

**9. SHOULD FIX — Section 13 (closing the loop; question G).**
- *The condition.* Right in substance: correction needs something to correct against, and without it a history gives sequencing only. Too narrow in form: the signal does not have to be handed in by the architecture. A network given a goal in its context can compute its own distance to it; a learned verifier is the usual case. Of the three candidates, "the state ceasing to change" is not an error signal — a wrong fixed point is still a fixed point.
- *"Static weights have no such list."* Too strong, for the reason in finding 1: a static block conditioned on one of K discrete codes can be tried K ways as well. What a bank adds is that each move is a separately evaluable function — which makes a stored block usable as its own forward model in a search. That is the first place in this program where the **tie** has a job of its own, and section 13 should say so.
- *"This design cannot show it."* Correct: the load is computed once per world (`viz_model.py` computes it before the loop; `tok_streams`' load does not read the state), no choice is revisited, and training contains no errors. But "nothing in stages 1–4 would change if the four passes were one pass with a larger step" is not right for `select`. Exact, from the teacher (all fifteen worlds, 4,096 states): one pass with four times the step has error 0.199 relative to the teacher's four-pass change; the applied primitive changes during the four passes in 0.195 of cases; an executor that keeps its first-pass choice has error 0.100, where the per-item arms reach 0.008. (`sum`: 0.060.) The bids already re-read the state on every pass, which is the simplest closed loop. What the carrier lacks is an error and a choice to revisit. The accurate statement is that no reading depends on the number of passes.
- *Order.* "After stage 2, because the error signal depends on C3 (i)" holds for one candidate only (the match score against demonstrations). On the carrier section 13 itself proposes — start and end states of an unknown sequence — the natural signal is distance to the end state, which needs no evidence-matched address. What it needs from this design is only that stored blocks can be run and compared with something in context, which R2's probes already indicate. So the dependency is weaker than stated; ordering it after stage 2 is defensible on cost, not on logic.
- *A caution for the proposed carrier.* A sequence differs from a bag of primitives only at second order in the step. Exact, from the teacher (fifteen pairs, median): applying (a, a, b, b) against (b, b, a, a) changes the result by 0.056 of the total change at step 0.2 (`sum`'s), 0.204 at 0.4 (`select`'s), 0.870 at 1.0; swapping one primitive for the other changes it by 0.26–0.40. At the present step sizes "find the sequence" is close to "count the primitives", and open and closed loop would hardly differ. The carrier needs a step near 1 or primitives that do not nearly commute.

**10. MINOR.**
- `viz_model.py` uses bids from a block's own units; section 5 inherits revision 3's "plain `tok_streams` configuration" (separate bid nets). Say which the bank arms use.
- `literature.md` records Schug et al. 2024 as "main text; the appendix not read". Its Appendix B.1 is headed "Overparameterization hurts compositional generalization" and Figure 2D reports sensitivity to the factor of overparameterisation. That is the empirical half of what C2 leans on and should be read before C2 is claimed.
- Section 2's row "binding learned functions to items recombines — Neural Production Systems" slightly overstates the prior work: `literature.md` says that paper has no held-out-combination test.
- P5 and the reading it names are the likely outcome; a prediction of failure at 0.55 for a claim listed as a contribution is a reason to redesign the stage before running it (L58: "when the main outcome is predicted by construction, change the design").

## Lessons L56–L60

| Lesson | Honoured? |
|---|---|
| L56 commit scripts with predictions; ablate the final candidate | Stated. The tie is ablated in stage 2; not in stage 4. |
| L57 tune until interior; reading at every pairing | Stated; not costed (finding 8); not stated per K. |
| L58 reproduce the failure first; hold out at the level of the component compared; change the design when the outcome is by construction | Stage 1 intends the first and cannot deliver it (finding 4). Not honoured for the weights factor (finding 1), stage 3 (finding 2) or stage 4 (finding 3). |
| L59 nearest alternative baseline; factorial for a two-part hypothesis | The 2 × 2 is the right shape; its weights factor bundles four things (finding 1). No nearest baseline in stage 4 (finding 3). |
| L60 name the outcome each distinguishing part could change; a placebo for each part; readings that can fail | Not honoured for the tie (no outcome it can change in stages 2 or 4) or for C1's two foregone readings (finding 6). |

## Probes (one seed, one split; indications only)

*Stored numbers, recomputed (no training).* `tok_mono` / `tok_streams` on names, twelve splits: held-out 1.53 [1.19, 1.97] at coverage 9 and 1.18 [1.05, 1.70] at coverage 7; seen 3.03 [2.84, 3.19] and 2.92 [2.56, 3.39]; held-out minus seen 0.034 against 0.031. `streams_self`, K = 8, `select`: held-out under half of `blend` in 1 of 12 splits.

*Stage 3 (pooled gate at K = 6, 8, 24).* Table and reading in finding 2.

*Pooled gate with per-item normalised loads; per-item arm at K = 6 and 24.* In finding 2.

*Stage 4.* Scratch `p4.py`, `p4b.py`. Teacher 10, `sum` with update Σ_items Σ_k load · f_k(h) and step 0.2, four passes, K = 8, w = 16, m = 2 demonstrations per item per (world, state), ten trained worlds over primitives 1–5, rate 0.01 with cosine decay, batch 24. Match arm: 4,000 steps, τ and c learned from 1; seen 2.0e-5 (1.4e-5 at 2× evidence); five primitives each with a block, one duplicated, two blocks computing nothing. Encoder arm (12 → 32 → 8, dot-product keys): 8,000 steps; seen 5.6e-5 (0.029 at 2× evidence, a factor of 500, in line with P7); five primitives with a block. New block: width 16, fitted to 256 demonstrations by 3,000 Adam steps, relative error 1.3e-6 on fresh states. Prototype key: mean code of the new demonstrations, scaled so its own items get the mean top logit of replayed old items. Table in finding 3.

*Section 13.* Exact checks, scratch `p13.py`; numbers in finding 9.

*Compute.* Eleven training runs, 31–81 s each, about eleven minutes in total, one process, one thread. PyTorch was not installed on the machine and went into a virtual environment in the scratchpad.

## Is a different, more valuable contribution available?

Yes, two, both from what the probes above turned up. Neither should displace stage 2.

**1. What binding is — an identifiability statement, in place of C2.** Count what the trained worlds leave undetermined in each form of gate. In a pooled additive gate the cross terms (how much item b excites the block of a) enter only sums that training wants small, so every held-out pair's decision carries a free offset; that is a memorising solution in the gate, present at every capacity. Normalising each item's address before the items meet removes the cross terms, and what is left to compare is a bid per primitive, fixed up to a common constant wherever the trained pairs are connected — the theorem's connected-support condition, arrived at from the gate's side. This predicts the 2 × 2's binding column, it predicts that capacity is irrelevant in both directions, and it says which part of "per item" does the work. It can fail: the one-seed probe says a pooled softmax with normalised per-item loads recombines (0.056 held-out) and the unnormalised one does not at K = 6, 8 or 24 (0.38–0.45), with offsets where the count puts them; twelve splits, the evidence carrier and the sigmoid gate could each contradict it. It is a derive-then-measure claim (AGENTS.md rule 2), it needs three arms on existing code, about an hour, and it corrects Entry 109's "binding, not routing": routing with a normalised address binds. I have not searched for prior statements of it; that search comes before any claim of novelty.

*The case for running it before stage 3, and not before stage 2.* Stage 3 as written costs three of the nine hours and its outcome is indicated in advance (no recombination at any K, for a reason it does not measure). The three-arm test costs an hour and decides what C2 should say. Stage 2 is unaffected and stays first.

**2. Give the tie a job.** In stages 2 and 4 the tie between the applied block and the matched block has no outcome it can change (R2's probe; finding 3). The researcher's own two statements of the aim (Entries 117 and 120) are about lookup that depends on the state and about correction in the loop. A stored block that can be run on a state and compared with a goal is a forward model of its own move; there the applied function and the tested function have to be the same, and an untied matcher would be a second model that can drift. That is section 13's carrier, with the cautions of finding 9. It is a design task, not a run, and it belongs after stage 2 as section 13 already has it.

**What I would cut or shrink.** C1 to its one open cell, measured at the decision level, with the unit-gated static arm added. Stage 1 folded into stage 2. Stage 3 replaced by item 1. Stage 4 kept, with its arithmetic named as such and the trainable-encoder-with-replay arm as the comparison that matters.

## Checked and found correct

- Sections 1–12 are unchanged between commits `fccde39` and `68f2739`; section 13 is the only addition.
- `bank_pooled` and `bank_item` (`streams_self`, `tok_streams`) have identical parameter sets (2,480 on names); they differ only in the forward pass. Among bank arms the binding factor changes no parameter (it does change two operations at once: finding 2).
- K is a config key of `worlds.py`; the stored K = 8 `streams_self` run reproduces exactly.
- The compute figures the design inherits (2,400; 4,880; about 2,330) and the parameter counts of the bank arms.
- Schug et al. 2024 has one code per task and no per-item code: C2's distinction is not already in that paper.
- Section 13's statement that the load is computed once per world.
- Section 8 does include rows that sink each claim, and section 9 marks honestly which predictions are blind.
