# Review 1 of the rung-1 design (`experiments/route/DESIGN.md`, revision 1)

Independent adversarial review, 28 Oct 2026, before the pilot. Read: DESIGN.md; Entries 33, 44–53, 55–60; AGENTS rules 3–5; `route/*.py`, `derive/*`, `tests/test_route.py`. CPU checks (scratchpad `review_route/`): the derive scripts; the index policy fitted on the exact marginal; the planted head at small gains; the probe ceiling; RM and RM-T with the repository trainer (batch 16, 600 and 1,800 steps, one seed).

**Verdict: run with changes.** Six of eight predictions are decided before any GPU step. **The 1,800-step CPU run (≈ 29k episodes, ≈ 7 % of the planned budget) already passes P1, P2 and P3:** route accuracy 0.93–1.00 at k = 1–8, 0.95 / 0.93 at k = 16 / 32, **1.00 / 0.98 / 0.93 on `pp4` / `pp6` / `pp8`** (PP runs never seen), 0.93 / 0.88 on `pp16` / `pp32`, 0.65 on `alt32`; steps = k; p(HALT) 1.00 at distance 0, 0.00 at ≥ 1. P6 and P8's halting line hold by construction; P7 and P8's probe line are mis-specified and would be scored wrong for uninteresting reasons. Only P5 is open.

## A. Is the question tested?

**(i) RM vs RM-T: a foregone tie, up to speed.** The state is one of 66 symbols, π is Markov, the readout is exact and answers absorb, so the marginal is a forward algorithm on a 66-state HMM-like model whose MLE is the deterministic local rule; EM with an exact E-step on Zipf-dominated short chains converges to it. P1's falsifier can fire only as a speed difference. At 600 steps the two arms are indistinguishable (HALT-at-answer and HOP-at-key learned, route accuracy 1.00 / 0.70 / 0.60 at k = 1–3 = the all-H share, PREV not yet flipped in either); the slow part is the content match in the cross-attention, not the E-step. One real difference: RM-T never trains off-route states, so its committed route runs to T_eval on any error (steps 14–40 here). RM-T is a speed reference, labeled optimization.

**(ii) "New orders" is in-distribution for a Markov router.** The trigger sits in the value slot of the matched pair (← → PREV; key/answer → HOP; answer in register → HALT). The router in `router.py` has *no* self-attention among pairs (§4 says it has one), so its receptive field is the pair(s) its two heads select; step, history and word never reach it. Two of the three "new orders within length" strata are training words: `alt4` = PHPH has probability 4.1e-3 per training query (≈ 9,300 draws in 6k steps), `alt6` ≈ 1,300, `alt8` ≈ 200; `deepP8` is 1.5 % of training queries. Only PP runs are novel, and only as a context neighbourhood the router cannot see; the module-network outcome (a memorized layout) is unavailable to this architecture. Order tests that can fail:
- **trigger on the neighbour**: mark pair(s_{i+1}) ("reachable by PREV") instead of pair(s_i), so the router must read one token past the match (give it the self-attention layer §4 describes); PP runs become new neighbourhoods;
- **a deferred trigger**: a marker on pair(s_i) meaning "the step after next is PREV", so the action at s_{i+1} depends on route history; the exact marginal runs on symbol × flag (132 states) and the question becomes whether the router learns a richer state (F4);
- **a third circuit** whose condition is a count of markers along the chain.
Also, P4's pooling contrast becomes yes/no if training has **zero** P-steps at depth ≥ 4 (F).

**(iii) RM vs LF/LR is apples to oranges.** With one-hot states and exact circuits, b_{t+1} = Σ_s b_t(s)[π(HOP|s)·HOP + π(PREV|s)·PREV] *is* superposed execution with a per-state soft gate (Entry 33): RM-marginal is a looped block with internal soft gating over exact circuits, RM-committed its argmax. RM vs LF therefore measures the cost of *learning the circuits* (rung 2–3), not selection vs gating, and P5's falsifier text ("explicit selection has a measured advantage over internal gating") does not follow from an LF failure. The arm that isolates selection is the one §6 defers: a looped block with HOP and PREV **planted** and only the gate learned. Add it.

## B. Confounds, leaks, artefacts

- **Word leaks:** none beyond the trigger. Blocks are shuffled and the prev slot is given, so the adjacency layout tells a static model nothing the token does not. RM-idx has no context input, so P7's "the strata leak the word" cannot fire.
- **P7 is wrong on the marginal reading.** Fitting the 12×3 index table on the exact marginal (λ = 0.02, 400 Adam steps, 2,000 episodes): committed in-range accuracy 0.747 (≈ coverage), **marginal 0.936** (1.00 at k ≤ 3; P-words 0.74; `pp4` 0.78, `alt4` 1.00, `len16` 0.16). A state-blind Bayes mixture gets short P-words right because wandering routes spread over wrong answers while the oracle route's mass lands on the target. The L22 gap of 0.23 holds for committed routes only; for the marginal it is ≈ 0.06, at k ≥ 5. P7 must name the reading; 0.94 is the state-blind reference for every `marg_acc` table.
- **λ and the index router:** the fitted table halts at t = 5–8 (p(HALT) 0.51 / 0.58 / 0.68 / 1.00) because λ competes with the Zipf tail; committed accuracy at k ≥ 6 is 0.00 even on all-H words; "within 0.05 of coverage" passed (0.747) by a near-tie. Report RM-idx by k.
- **λ on RM:** harmless. A wrong answer costs 6.6 nats (ε/15), ⊥ 2.77, so PREV is globally aversive at init (88 % of states are H-states) until the ←-feature rescues it; PREV is the last action learned (flip at ≈ 600–1,200 steps here). T_train = 12 adds no horizon artefact.
- **§5 numbers reproduce exactly.** Defects: `wander.py`'s "mass halting on any answer" is identically 1.000 (uniform rows of R sum to 1); the "oracle share" counts only HOP-then-HALT although all mass reaching the absorbing answer halts on it, so the share driving the gradient is 3^-k (≈ 0.90 at k = 1, not 0.30).
- **Design/code mismatches:** the self-attention layer (absent); "evaluated only at states with mass > 10⁻⁶" (all 66 are evaluated once per episode, better); d = 328 (§2) vs 344 (code, §6); "planted router within one unit of the init": the code plants gain 3 / bias 4; a committed-exact plant exists down to gain 0.1, but the marginal-sharp plant (p(target) ≥ 0.95 at k = 32) needs gain 6 / bias 8 against Σ lr = 2.75, reachable only because the LayerNormed value-slot feature has magnitude ≈ 13 (D91). Say which plant and why.

## C. Predictions

- **P1:** passes at 7 % of budget; the information is PREV's time-to-flip. Keep, as optimization.
- **P2:** foregone given P1 (no step input; the states at k = 32 are the same symbols). Label "by construction".
- **P3:** foregone given P1 (A.ii); alternation and deep-P are not new orders. Add an A.ii variant or demote to a sanity check.
- **P4:** S8's reach is F26; the new content is one line ("the ← gate is learned per layer at d ≤ 3"), not 2 T4-hours.
- **P5:** the open one. Its falsifier conflates block and exit rule: LR is blind when the readout does not move, and PREV changes nothing in the exact prediction. Report LF at **oracle depth** (loop k and k+1, from the stored per-step readouts) beside LR, so a failure is attributable.
- **P6:** degenerate. With absorbing answers and λ = 0 the HALT logit at answer states gets zero gradient; the committed route halts or runs to T by the sign of an untrained logit per answer symbol. "Halts late, accuracy unchanged" is guaranteed, and "minimal routes generalize better" has no content: the oracle route is the unique successful route up to post-answer dithering. Restate: λ identifies the halting time; B1 is not testable at rung 1.
- **P7:** restate per reading (B).
- **P8:** the halting line is fine. The probe line cannot pass: the router's features are a function of the matched pair, which fixes remaining distance only as {0, 1, ≥ 2}; the R² ceiling of that information is **0.61** on the k = 8 stratum, 0.70 on the Zipf stream, ≈ 0.31 at k = 16, ≈ 0.17 at k = 32; an *untrained* router scores 0.62, the trained one 0.11 on the code's probe set (dominated by k = 16–32). "R² ≥ 0.8 / ≥ 0.5" cannot hold and "R² < 0.3 out of range" must fire. Drop it or predict "reaches the ceiling"; here the state is the symbol and carries no progress (Entry 47 is not answerable at this rung).
- **Outcome table** lacks ¬P1 ∧ ¬RM-T (architecture), P1 ∧ ¬P5 with LF correct at oracle depth (exit rule, not block), and the marginal/committed split. "Reading the outcomes" oversells P1 ∧ P2 ∧ P3 as "Q1 and Q2 answered yes": a pass shows an exact E-step learns a locally triggered rule, which NPI/NRAM predict (E).

## D. Budget, power, fairness

- Seeds 10/5/3 suffice (effects are 0/1); the shortage is information, not power. Move S8's 2.0 T4-hours (5 × 25k, for F26 again; 3 seeds or 12k steps suffice) to the planted-gate block and an A.ii variant.
- **Pool with replacement:** 60k episodes, seen 6.4× by routed/LF arms and **26.7× by S8**; time-to-criterion is measured on training batches (`hits` in both trainers), so S8's k ≥ 4 curves can rise by memorization (360k query–label pairs, 12.4M parameters). Use a fresh generator or a held-out set scored every 100 steps.
- LF expressivity: one head can copy both value and prev slots (disjoint subspaces) and the 4d MLP can gate 2 × 66 directions on ←, so one loop per step suffices. Compute is matched in count (k+1 calls vs T loops); S2 (E[k]+1 ≈ 3.3) is uninformative.

## E. Prior work

Not new, and the design should say so. (a) A controller over exact primitives that generalizes in length once it conditions on the local frame is NPI with recursion (Cai 2017): RM-T is that; the Markov router is the recursion insight. (b) Soft selection over exact modules trained from outputs, tested in length, is NRAM (Kurach 2016; its fuzzy execution *is* this forward recursion) and Neural Programmer. (c) Marginalizing over latent traces with a forward lattice is Neural Program Lattices (Li 2017), the closest form of the exact E-step. (d) A controller composing a library serially with a halt action, tested on longer and novel compositions, is CRL (Chang 2019; missing from §14). New here: the carrier (context circuits on a token sequence), exact oracle routes, the looped-block contrast. P1–P3 read as replication; P5, the planted-gate arm and an A.ii variant are the contribution.

## F. Changes

**Blocking:** (1) P7 restated per reading, with the marginal state-blind reference 0.94 in §5. (2) P8's probe line dropped or re-based on the ceiling. (3) P6 restated as untestable here. (4) P2/P3 relabeled by-construction, alternation/deep-P moved out of "new orders", an A.ii variant added or P3 demoted. (5) A held-out criterion set for time-to-criterion. (6) Design text matched to the code (B).

**Non-blocking:** P5 outcome rows with the oracle-depth reading; `wander.py` fixed and §5 merged; S8 cut.

**Cheap additions (< 1 GPU-hour):** LF at oracle depth (free from `ok_all`); the planted-circuit block with a learned gate (≈ 10 × 6k ≈ 0.4 h); the zero-count manipulation (no P at depth ≥ 4 in training) on RM and LF (≈ 0.5 h); PREV flip time in the pilot; RM-idx by k.

**Verdict: run with changes.** With them the rung costs ≈ 3 T4-hours and yields two results not already known: LF against the planted-gate block in unseen orders, and what a trigger-inferring router does.
