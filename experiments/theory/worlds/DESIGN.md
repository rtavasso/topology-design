# Do streams recombine where a monolithic block does not? — staged design on a "worlds" carrier

**Status: EXPLORATORY design, revision 1 (15 Nov 2026). Nothing run except 100–300-step smoke tests** (which I saw: a monolith at 300 steps had seen-world error 0.26 and held-out 0.92 on `select`, 0.055 and 0.66 on `sum`). History Entries 106–108; idea I37 in the ledger. Code `worlds.py`; exact statements `tests/test_worlds.py`.

## 1. The question and the logic

I37: does breaking the block into streams of learned primitives, recombined by a gate, let a loop handle combinations it was not trained on where one monolithic block does not? Its null: recombination comes with coverage, not architecture.

The claim is an interaction — arms agree on combinations they were trained on and differ on combinations they were not — so the experiment is staged (Entry 107):

- **Stage 0.** The monolith alone, over a dial of coverage. Does it fail on held-out combinations while fitting seen ones? If not, stop: the carrier cannot test the claim.
- **Stage 1.** A ceiling: streams whose gate is set from the truth. Is modularity rewarded here at all?
- **Stage 2.** The comparison, at the coverage stage 0 fixes.
- **Stage 3.** Mechanism checks on whichever arm closes the gap.

Entry 105's comparison failed on four counts (Entry 106): the baseline never failed; programs were held out but nothing was held out from the block; the task named one primitive per pass; a shared controller did the recombining. This design is built against those four.

## 2. The carrier

Six primitives f_p(h) = tanh(A_p h) on a six-dimensional state, each with a trigger direction w_p. A **world** is a set of two primitives, stated in the input as two tokens. Fifteen worlds exist. A block is applied four times; on every pass it receives the state h and one read of the world, r = E[a] + E[b] (E learned by each arm). There is no clock, no register and no attention: the same input every pass, and nothing outside the block decides anything.

Two task families (two notions of "combination"):

- **select.** Each pass applies *one* of the world's two primitives — the one whose trigger ⟨w_p, h⟩ is larger. Which one is not stated; it depends on the state and changes from pass to pass. (Step 0.4.)
- **sum.** Each pass applies *both*, summed. (Step 0.2.)

**Held-out combinations.** Training uses a subset of the worlds; the rest are never trained on. The training set always satisfies: every primitive is in at least two training worlds, the graph of training worlds is connected, and it contains an odd cycle. Under these conditions the right behaviour in a held-out world is *determined* by the training worlds (test `test_training_worlds_determine…`): in `select`, the rule in world {a, b} is the sign of ⟨w_a − w_b, h⟩, and w_a − w_b is a signed sum of trained differences; in `sum`, each f_p is recoverable from the trained sums. **Coverage dial:** 7, 9, 11, 13 or all 15 worlds in training.

This is rung 4's split in miniature: there, worlds were sets of stated axioms, fifteen of them, and the instances that failed (0.47 against 0.94) were those needing a pair of axioms never together in a training world.

**The properties Entry 107 asked for.**

| Property | Here | Shortfall |
|---|---|---|
| Primitives not named by the input | `select`: the input names two candidates; which applies is decided by the state through triggers that are never shown. `sum`: no primitive is ever seen alone. | Tokens map one-to-one to primitives. The *identity* of the candidates is given; what must be discovered is the per-primitive function and (select) the per-primitive trigger, from pairs only. |
| Every primitive seen, combinations held out, a dial | yes | fifteen combinations is small |
| A held-out combination is new at the block's own input | yes: r = E[a] + E[b] is a vector the block has never received | — |
| A memorising solution exists | a block can fit each trained world with its own boundary or its own summed function | not shown until stage 0 |
| Both arms can represent the truth | streams: exact (test `test_the_modular_solution_is_exact…`). Monolith: by approximation; **checked by the full-coverage column of stage 0** | the monolith's fit may be poorer; see §6 |
| Exact scoring, true primitive per step known | yes | — |
| Several task instances | three teachers | three |
| No shared controller | nothing but the block | — |

**A limit stated now.** The teacher of `select` — an available set, per-primitive bids, the larger bid wins — has the same shape as the "each stream bids" arm below. That is deliberate (the claim is conditional: *if* the data have modular structure, does a matching architecture use it where a monolith does not), and it bounds the conclusion: this carrier cannot show that streams help when the data are not of this shape.

## 3. Arms

All have the same number of trainable parameters to within 3 % (about 2,500; test). All get h and r and nothing else.

| Arm | Block | Isolates |
|---|---|---|
| `mono` | two hidden layers of 40 on [h; r] | the baseline |
| `streams_joint` | 8 streams V_k tanh(B_k h) of width 16; gate logits G r + MLP([r; h]) | the candidate's form (Entry 104) |
| `streams_self` | the same streams; gate logit of stream k = G_k·r + s_k(h): each stream bids from the state alone, the world only adds a bias | a gate that is itself modular |
| `unit` | 60 units, each with a soft gate from C r + MLP([r; h]), all summed | gating without grouping |
| `mono` + weight decay | as `mono` | any regulariser, as against modularity |
| `oracle` | six streams; gate set from the truth (the chosen primitive, or both) | the ceiling |

Streams arms run every stream (no sparsity): the question here is recombination, and Entry 104 treats cost. Scores: softmax or sigmoid is tuned per arm (softmax can represent both tasks; an additive sigmoid bid cannot represent `select`).

**References, exact from the teacher**, for held-out worlds: `blend` (select: apply the average of the two primitives — knows which two, cannot choose) and `one` (apply only one of the two).

## 4. Outcomes

Error = squared error of the change h_4 − h_0 relative to the teacher's change, on fresh inputs. **Seen** = training worlds; **held-out** = worlds never trained on. Primary outcome of stage 2: the held-out error itself, on the scale given by the seen error (floor) and the references (ceiling of not composing), and paired against the monolith's.

## 5. Tuning rules (from L57)

- Every arm gets its own learning rate from {0.003, 0.01, 0.03, 0.1} with cosine decay; if the best sits at an edge the grid is extended until it does not.
- Selection is by **seen-world error** on separate seeds — the only legitimate criterion, since held-out worlds cannot be used. Because that criterion cannot see the primary outcome, the held-out error is **reported for every configuration in the grid**, so that the reading can be checked against the selection.
- 4,000 steps; if an arm's seen error is still falling by more than a factor 1.5 over the last quarter, steps are doubled for all arms.

## 6. Stages, criteria, stop rules

**Stage 0** — `mono` only. (i) Rates: at coverage 9, four rates × three teachers, both tasks. (ii) Dial: coverage 7, 9, 11, 13, 15 × six runs (three teachers × two seeds), both tasks, at the chosen rate. (iii) A monolith twice as wide, at coverage 7.

For each task family:
- **Passes** if, at the lowest coverage where the seen error is within 2× of the full-coverage seen error, the held-out error is at least 5× the seen error (median of per-run ratios), and the ratio is lower at coverage 13 than at 7.
- **Weak** if that ratio is between 2 and 5: carry on, flagged as low power.
- **Dropped** if the ratio is below 2 at every coverage. If both families are dropped, **stop and report**: this carrier cannot test the claim.
- The coverage used in stage 2 is the lowest one that passes the fit condition.

**Stage 1** — `oracle`, six runs at the stage-2 coverage. Modularity "is rewarded" if its held-out error is within 2× of its own seen error and at least 10× below the monolith's held-out error. If not, stop: even a perfectly modular block does not recombine here.

**Stage 2** — all arms, twelve paired runs (three teachers × four seeds) at the stage-2 coverage, each arm tuned as in §5; `mono`, `streams_joint`, `streams_self`, `unit` also at one higher coverage.
- **Support** for I37 on this carrier: a streams arm's held-out error is at most a third of the monolith's in at least 10 of 12 paired runs *and* below half the non-composing reference (median), while the regularised monolith is not (within 2× of the plain one).
- **Gating, not streams:** `unit` meets the same bar.
- **Null:** every arm's held-out error is within 2× of the monolith's (median of paired ratios).
- Anything else is reported as mixed, by arm and task.

**Stage 3** — for arms that meet the bar: agreement between the gate's largest stream and the true primitive on held-out worlds; removing one stream and checking which primitives suffer.

Independent review of this design before stage 0; of the results after stage 2 (or at a stop).

## 7. Predictions (before any run beyond the smoke tests named above)

Stage 0. P0a `select` passes: held-out ≥ 5× seen at coverage 7, falling with coverage, and the held-out error at coverage 7 is at least half the `blend` reference. P0b `sum` passes too, and is still ≥ 2× at coverage 13. P0c seen error at coverage 7 within 2× of that at 15, both tasks. P0d doubling the monolith's width changes the ratio at coverage 7 by less than 2×.

Stage 1. P1 the oracle meets both conditions on both tasks (a check, not a bet).

Stage 2 (to be re-stated, not changed in direction, once stage 0 fixes the coverage). P2a `select`: `streams_self` meets the support bar. P2b `select`: `streams_joint` is better than the monolith but does not meet the bar (median paired ratio between 1/3 and 1). P2c `select`: `unit` is within 2× of `streams_joint`. P2d the regularised monolith is within 2× of the plain one, both tasks. P2e `sum`: all three gated arms meet the bar, and `streams_self` and `streams_joint` are within 2× of each other.

My confidence in P2a is about 0.6 and in P2b about 0.5; if P2a fails with the oracle passing, the reading is that a matching architecture does not find the modular solution by gradient descent on this carrier.

## 8. Budget

One run is 10–40 s on one core. Stage 0: about 100 runs. Stage 1: 12. Stage 2: tuning about 200 short runs, main about 250. Two processes; roughly two hours of wall time in all.
