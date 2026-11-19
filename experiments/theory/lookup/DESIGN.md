# Context-dependent weights against static weights — proposed contribution and design

**Status: DESIGN, revision 4, nothing run by the author. EXPLORATORY.** Written 18 Nov 2026 at the researcher's request ("given our framing, what is our proposed contribution and experimental design?"), after he stated the question of this line (history Entry 117) and after the literature check (Entry 118). It re-aims revision 3 (Entry 113; in git at `523106a`), whose evidence carrier, arms and rule for a reading it keeps. Reviews of revisions 1 and 2 are in `review/`; this revision is reviewed in `review/review_design_rev4.md`. History Entry 119.

## 1. The question

The researcher's: **how do weights that depend on context compare with static weights?** "Weight modification based on context itself, … expanding the space that a model stores information in by doing this weight lookup using context of the input and state."

Terms. *Static weights*: one block, the same for every input; context reaches it only as activations added to its input. *Context-dependent weights*: the block that is applied is assembled per context from a bank of stored units, so context decides which units are present and how strongly they write.

## 2. What is already known, and so is not our contribution

| Claim | Where it stands |
|---|---|
| A bank of units retrieved by key stores more at fixed compute | Shown at scale (PEER; product-key memory). Not re-measured here. |
| Composing weights by attention scores works | It is ordinary multi-head attention across heads (Schug et al. 2025). |
| Averaging weights is exact only where the output is linear in them | Known (the modular-learning survey; our tests W7–W9 restate it). |
| Binding learned functions to items of the context recombines | Neural Production Systems, Neural Interpreters; our Entry 108 on names. |
| Experts inside a loop train at scale | LoopMoE, Tying the Loop, MoEUT. |

## 3. Proposed contribution

**Thesis.** Keeping primitives as separately stored, retrievable units matters less for what a model can *compute* than for what it can *recognise*. A static block conditioned on context can compute the same functions, and — applied once per context item — recombines them nearly as well (Entry 108). What a bank of units can do and fixed weights cannot is serve as its own index: a stored block can be run on the evidence in the context and compared with it, so the model finds the right primitives by testing its own contents, needs no learned table from context to address, and can be extended by adding a block.

Three claims, each able to fail, none found in prior work as stated:

- **C1 — a decomposition.** The advantage of context-dependent over static weights splits into three effects with different causes: *fit per parameter and compute* (from selecting units instead of conditioning additively), *recombination on unseen combinations* (from binding — one lookup per context item, competition at the state — whether or not the weights are static), and *recognition* (C3). Prior work has each piece at scale on perplexity; none has them separated against exact references on one carrier.
- **C2 — capacity and binding interact.** A theorem (Schug et al. 2024) ties compositional generalisation of composed weights to having no spare capacity. We propose that this is a property of composing with *one code for the whole context*: under per-item loading, spare capacity is harmless, and under a pooled code recombination is present at exact capacity and lost as capacity grows. This is the precise form of the researcher's "expanding the space": expansion is safe only under per-item addressing.
- **C3 — a block can be its own key.** Retrieval by evaluating the applied blocks on the context's evidence (i) trains from scratch together with the blocks, (ii) keeps identifying primitives when the evidence leaves the training distribution, where a learned encoder degrades, and (iii) addresses a block added after training with no change to anything else. Not found as a trained layer (Entry 118); the nearest work does inference over modules by particle filtering and names the bootstrap problem.

What would make this *not* a contribution: C1 if a static per-item block matches the bank on every measure including compute-matched fit (then context-dependent weights buy only compute, which PEER already shows); C2 if capacity has no effect under the pooled code; C3 if a wide encoder matches the match arms under shift and adding a block is as cheap with stored keys.

## 4. Carrier

The evidence carrier of revision 3, unchanged. Six primitives f_p(h) = tanh(A_p h) on a six-dimensional state, a trigger w_p each; a world is two primitives; four passes of one block. Each primitive is **shown, not named**: m = 2 demonstrations (x, f_p(x)), x ~ N(0, I), one draw per (world, state). Tasks `select` (the primitive with the larger trigger applies) and `sum` (both apply). Nine of fifteen worlds trained, every primitive trained. Exact references (`blend`, `trans`, own seen error) and the regimes of Entry 108. Test conditions fixed in advance: held-out worlds; evidence at 1.5, 2 and 3 times the training scale (2× the pre-named point); m = 1 and 4, noise 0.1 (reported only).

Limits carried over and stated now: the `select` teacher has the shape of the per-item arms; a bank block contains a teacher primitive exactly, which tilts shifted-evidence tests towards arms that match with blocks; the context arrives already split into two items.

## 5. Arms

Every arm reads each item through the same encoder where it uses one (a small network on [x; y], averaged over the item's demonstrations), and is reported with its parameter count and its multiply-adds per pass.

| Arm | Weights | Binding | How context reaches the block |
|---|---|---|---|
| `static_pooled` | static | pooled | one two-layer block on [h; code_a + code_b] |
| `static_item` | static | per item | one shared two-layer block run on [h; code_j] for each item j; the two updates pooled by bids from the block |
| `bank_pooled` | context-dependent | pooled | K blocks; one softmax over blocks of (code_a + code_b) · key_k + bid_k(h) (the `streams_self` arm) |
| `bank_item` | context-dependent | per item | K blocks; each item loads blocks by code_j · key_k; items compete by bids; stored keys (`enc_streams` of revision 3) |
| `bank_match_untied` | context-dependent | per item | load = match of a second, separate set of blocks against the item's demonstrations |
| **`bank_match`** | context-dependent | per item | load = match of the **applied** blocks against the item's demonstrations, −‖c·f_k(x) − y‖²/τ (`ev_lookup`) |
| `oracle`; `tok_streams` on names | | | ceilings for identification and for the task |

The first four are a 2 × 2 (weights × binding) with addressing held fixed (a learned encoder). The last three bank arms are a ladder in addressing with weights and binding held fixed: learned encoder → match → match tied to the applied weights.

Sizes. Bank arms: K = 8 blocks of width 16, identical across bank arms. Static arms at three widths: **compute-matched** to the bank with every block evaluated, **parameter-matched**, and three times the parameters. The bank's compute with only loaded blocks evaluated is reported beside these; no static arm is sized to it (that comparison is PEER's).

## 6. Measures and the rule for a reading

Seen and held-out error under each test condition; **identification by function** (each block matched after training to the primitive it computes; an item is identified when its most-loaded block computes its primitive); failed runs; parameters; multiply-adds per pass and per world.

The rule of revision 3, section 5: twelve splits on twelve teachers; three other initialisations of the reference arm give the noise envelope; *worse* or *better* means the bootstrap interval of the paired ratio lies outside the envelope at equal learning rates and at every pairing of rates within 2× of the best seen error; *not worse* means an upper bound below 1.5; otherwise *unresolved*. Rates and step counts (4,000, 8,000, 16,000) tuned per arm on splits 100–102, the best interior (L57). Scripts committed with predictions before they run (L56); results reviewed by an agent that wrote none of this.

## 7. Stages

**Stage 0 — identities and reproduction.** Forward identities as tests; `tok_streams` re-run against stored numbers; `tok_streams` on the twelve teachers.

**Stage 1 — the static block first (C1; the phenomenon before any fix).** `static_pooled` and `static_item` at three widths, `oracle`, the names ceiling. Questions: given evidence instead of names, does a static block fit, and does it recombine when applied per item?
*Stop rule.* If parameter-matched `static_item` is not worse than the names ceiling on held-out error **and** identifies at 0.99 or better at 2× evidence, then on this carrier static weights with per-item binding do everything; C1 reduces to a compute table, C3 has nothing to fix, and only stage 3 (C2) is run.

**Stage 2 — the 2 × 2 and the addressing ladder (C1, C3 i–ii).** All arms of section 5, `select`; bank arms also on `sum`.

**Stage 3 — capacity (C2).** `select`, held-out worlds, in-distribution evidence. Blocks K ∈ {6, 8, 12, 24} (six is exact: one per primitive) for `bank_pooled`, `bank_item`, `bank_match`; width at one, two and four times parameter-matched for `static_pooled` and `static_item`. Reported with, for every bank run, how many blocks compute a primitive, how many are duplicates, and how many are used by only one trained world (the memorising solution, measured by function and by which worlds load the block).

**Stage 4 — adding a primitive after training (C3 iii, and the researcher's "expanding the space").** A `sum` carrier without bids (both items always apply), so that nothing but identification and the new function is needed. Train on worlds over primitives 1–5. Then give every arm the same adaptation data and budget: demonstrations of primitive 6 alone, no worlds.
- `bank_match`: one new block is fitted to the demonstrations by regression; nothing else changes.
- `bank_item`: the same new block, plus a new stored key trained so that primitive-6 evidence loads it; run with the encoder frozen and with it trainable.
- `static_item`: the whole block and encoder fine-tuned on the demonstrations.
Measured as a function of adaptation steps: error on worlds containing primitive 6 (all five are new), and error on the old worlds (interference). No rehearsal data for any arm.

**Stage 5 — mechanism.** Identification and the block-to-primitive match over training for the match arms (does the bank sort itself, and when); what the static block's encoder confuses under shift.

## 8. Readings

"Evidence error" is error on trained worlds at 2× evidence; "held-out error" is on held-out worlds with in-distribution evidence.

| Claim | Reading | Condition |
|---|---|---|
| C1 | **Binding, not weights, recombines** | both per-item arms better than both pooled arms on held-out error, and `bank_item` not better than `static_item` at parameter-matched size |
| C1 | **Weights also recombine** | `bank_item` better than parameter-matched `static_item` on held-out error, and still better than the 3× static block |
| C1 | **Weights buy fit** | `bank_item` better than compute-matched `static_item` on seen error |
| C1 | **Static weights suffice** | stage 1's stop rule |
| C2 | **Supported** | `bank_pooled` recombines (held-out error at most half the blend reference) at K = 6 and not at K = 24, while `bank_item` is not worse at K = 24 than at K = 6 |
| C2 | **Pooled composition fails at every capacity** | `bank_pooled` does not recombine at K = 6 either: the condition is not capacity. C2 fails. |
| C2 | **Capacity is harmless everywhere** | `bank_pooled` recombines at K = 24: C2 fails the other way |
| C3 ii | **The block as its own key earns its place** | `bank_match` better than `bank_item` and than its wide-encoder form on evidence error |
| C3 ii | **Match form, not the tie** | `bank_match_untied` better than `bank_item`, `bank_match` not better than `bank_match_untied`: then the tie's case rests on stage 4 alone |
| C3 i | **Does not bootstrap** | `bank_match` worse than `bank_item` on seen error, or some primitive has no block in more than 3 of 12 runs |
| C3 iii | **Extends without re-indexing** | after adaptation, `bank_match` not worse than a bank trained on all six primitives on new-world error, old-world error unchanged; `bank_item` or `static_item` worse on either |
| C3 iii | **No advantage** | `bank_item` with a new key is not worse on both, at the same adaptation budget |
| any | **An encoder suffices** | the bank arms mutually not worse in every condition |

What is by construction, said in advance. Per-item arms cannot represent a solution specific to a pair, so their immunity to spare capacity in C2 is close to structural; the measured half of C2 is the pooled arm. In stage 4 the other blocks of `bank_match` are untouched, but its loads are not: a new block competes for old evidence, so old-world error can rise. If the bank holds the primitives exactly, matching identifies perfectly; whether a bank that is still forming can address itself, and what a trained, inexact bank does off its training states, are not given.

## 9. Predictions

Not blind for stage 2: the reviewer of revision 2 ran one-seed probes (match arms tied each other and held under shift; the encoder arm needed 8,000 steps and fell to 0.93 identification at 2×). Blind for stages 1, 3 and 4.

| | Prediction | Confidence |
|---|---|---|
| P1 | Stage 1 does not stop: parameter-matched `static_item` identifies below 0.99 at 2× evidence | 0.7 |
| P2 | In distribution, `static_item` (parameter-matched) recombines: held-out error at most half the blend reference | 0.6 |
| P3 | Reading "binding, not weights, recombines" holds | 0.55 |
| P4 | `bank_item` better than compute-matched `static_item` on seen error | 0.65 |
| P5 | `bank_pooled` does not recombine at K = 6 (C2 fails as "the condition is not capacity") | 0.55 |
| P6 | `bank_item` at K = 24 not worse than at K = 6 on held-out error | 0.7 |
| P7 | `bank_match` better than `bank_item` on evidence error | 0.7 |
| P8 | `bank_match` not better than `bank_match_untied` on `select` | 0.6 |
| P9 | Stage 4: `bank_match` reaches new-world error within 2× of the six-primitive bank, old-world error within 1.5× of before | 0.6 |
| P10 | Stage 4: `static_item`'s old-world error rises at least tenfold | 0.7 |
| P11 | Stage 4: `bank_item` with a frozen encoder is worse than `bank_match` on new-world error | 0.55 |

## 10. What this would and would not establish

Would: on one small carrier with exact references, which of the three effects come from context-dependent weights and which from binding; whether the theorem's capacity condition is tied to pooled codes; whether a bank can index itself and be extended.

Would not: anything at the scale where the prior work lives (language models); anything about blocks with attention (the block here is the feed-forward half, one site); where items come from when the context is not already split; storage at fixed compute (PEER's result). A positive outcome is a reason to build the same comparison in a transformer block on the rewrite carrier, which is not designed here.

Twelve splits and a synthetic teacher: EXPLORATORY, and AGENTS.md asks for more seeds and a full bootstrap before any headline claim.

## 11. Cost

About nine hours on two processes: stage 1 one and a half, stage 2 two, stage 3 three, stage 4 two, plus tuning.

## 12. Set aside from revision 3

The check on names (naive weight mixing; the 2 × 2 × 2 of bid source, key source and hard load) answered narrower questions. Naive weight averaging is settled by the rule in section 2. A hard load and bids from a block's own units are engineering choices to be made once the comparison above says whether the bank is worth engineering.
