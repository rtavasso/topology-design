# Independent review of DESIGN.md revision 3 (27 Oct 2026)

Read: DESIGN.md rev 3 (L1–176), PILOT1/2.md, history Entries 55–57, AGENTS.md, `grids/rev3.json` (83 runs; counts, steps, lr verified against `make_grids.py`), and the code the design relies on (`model.py`, `train_stacked.py`, `analyze.py`). Line numbers are DESIGN.md's.

## A. Predictions

**P1 (L137).** Falsifiable, with a dead zone: confirmation needs LH-Δ ≥ 0.9 at k = 16, the falsifier fires at < 0.6. Pilot seeds at the failure edge are bimodal (LH-s at k = 12: 0.56 ± 0.36, i.e. ≈ 0.2 and 0.9), so a 10-seed mean of 0.7 is the likely "neither" outcome. State per-seed fractions (≥ 8/10 seeds ≥ 0.9). Same for P2 (0.6–0.9) and P1's k = 32 line.

**P3 (L139).** Nested in P1 (¬P1 ⇒ ¬P3) and binary where the outcome is a frontier: "within 0.02 **and** cheaper" vs "no more accurate **and** no cheaper" leaves the trade-off cases unread. The cost anchor is wrong out of range: Entry 56 gives LR excess 1.6/1.1/1.3/**1.9/2.1/2.3** at k = 1/4/8/16/24/32, so "≈ k + 1.5" (L79, L139) holds in range only, and **P7's "LR ≥ 0.95 at every k ≤ 32 at ≤ k + 2 steps" (L143) is already violated by the pilot traces at k = 24, 32.** A sanity line that fails by construction reopens a pilot result for nothing.

**P4 (L140).** "LH-x constant in k" is true by construction; not a prediction.

**P5 (L141).** Falsifiable; the LO-vs-S8 line replicates pilot 2 (4k vs 9–11k). Schedules differ (cosine over 6k vs 25k), so cross-arm time-to-criterion is an optimization observation — labelled correctly.

**P6 (L142).** "S8's t95(3) at 10⁻¹ ≥ 3× factor 1 (censoring at 25k counts)" is unconfirmable unless factor-1 t95(3) ≤ 8.3k (pilot: 9–11k). `analyze.py` L242 honestly prints "≥ budget/base×", which would read "≥ 2.5×" — contradicting the text. The LH-s halted line has no falsifier: if halted t95(3) does **not** move, is that "the policy pools too" or a failure? Both outcomes read as fine. Under the inverse-share hypothesis (72× fewer k ≥ 3 examples at 10⁻²) it will be censored at 8k; state the test one-sided.

**Outcome table (L147).** Missing: ¬P1 ∧ ¬P2; P1 ∧ ¬P2 (the ¬P2 line says "the drift explanation is wrong", but with P1 holding, drift in the state that a change head ignores is *confirmed*; only T-exposure failed to fix it); and every "fails in range → no test" case for the two untrained arms.

## B. Confounds

**(i) LH-Δ's features are block-shaped.** `delta_features` (model.py L178–183) is not detached: the halting loss back-propagates into the block through KL(r_t‖r_{t−1}) and ‖Δh‖/‖h‖. The block can make chasing features large (jumping intermediate readouts cost nothing where p_t ≈ 0) and settled features small; whether a shaped signal persists at loops 11–32 is exactly the open question, and the falsifier ("LH-Δ < 0.6 while **LR** ≥ 0.9", L137) tests LR on **LF's** block, not LH-Δ's. A ¬P1 cannot then be attributed to head vs block. `kl_exit` runs only for `halting == "fixed"` (train_stacked.py L86). Add a forced fixed-loop evaluation of every looped arm's block with `kl_exit` (≈ 10 lines, +5% eval compute). LR on LO's block is also the planted caveat's live test.

**The head's optimizer budget.** AdamW moves each scalar ≤ lr per step; Σ lr over 8k steps (500 warm-up, cosine) is **1.13**. LH-Δ's bias can reach ≈ −3.1 from −2; the planted head (w₂ = −2, b = −10, L94) is unreachable. With log-features ≤ 0 during chasing (KL < 1 nat, relative Δh < 1) and w ≤ 0 (needed for "halt when change is small"), the chasing logit is ≥ b, so λ_chase ≳ 0.04. A constant chasing λ of 0.05/0.04/0.03/0.02 reaches cumulative p = 0.5 at loop 14/17/23/35. In range nothing shows (cum p over 8 loops = 0.28); k = 16 is marginal and k = 32 fails — **a flat-λ premature halt that mimics the horizon signature**, which the table would read as "¬P1: the free rule's threshold does not emerge from the PonderNet objective" (a mechanism conclusion from a learning-rate artefact). LH-s has no such floor (264 unit-scale features). Escape routes exist (the block drives chasing features above 1), but whether they are found in 8k steps is unknown: **LH-Δ has never been trained.** §9's "learning rates per arm from the pilots (3 values × 2 seeds for every arm)" (L151) is false for LH-Δ and LH-s24.

**(ii) LH-s24 changes more than the horizon.** (a) The truncated geometric prior retains 0.65 of its mass at T = 10, 0.92 at T = 24; (b) the forced λ_T = 1 anchor carries 0.32 of the initial mass at T = 10 and 0.05 at T = 24 — the deep-hop signal that anchored pilot 2's narrow basin is 6× weaker; (c) loops 9–24 are only ever post-answer idle states for k ≤ 8, so the block is additionally trained to idle for 16 loops (LF's block cannot: overthinking), and the head sees drift(9..24) only with register = answer, so a linear head can learn drift-as-halt-cue. ¬P2 is therefore compatible with drift being the mechanism, and P2 passing may work through the block. Preregister: in-range gate; LH-s24's step-probe vs LH-s's 0.35 (lower ⇒ block changed); λ-by-loop at k = 32. A complementary test: LH-s with k_max = 4, T = 10 (0.7 h) — does failure track T (≈ 15 loops) or 1.5 × k_max (≈ 6)?

**(iii) LR threshold.** 10⁻² was the first value tried on 2 pilot-1 runs × 64 traces (Entry 56); 10⁻²–10⁻⁴ all worked, 10⁻¹ failed. Fixing it before 10 new seeds of the same configuration is legitimate as "preregistered for rev 3", not as derived. Call LR a one-hyperparameter rule tuned on pilot traces, not "zero-parameter". Report the threshold curve (10⁻¹ … 10⁻⁴, free) and make P3's "cheaper" claim against that frontier; at one threshold the 0.3-loop margin is an artefact of the threshold.

**(iv) Windowed criterion.** At factor 10⁻², n ≈ 54 per window, so "≥ 0.95" means ≤ 2 errors: P(pass | true 0.95) = 0.49, two consecutive 0.24; P(false pass | true 0.90) = 0.08, pair 0.007 (≈ 35% over 60 windows of a 0.90 plateau). The late bias at small n is one-sided: it inflates LO's 10⁻²/1 ratio toward the ≥ 3× falsifier. Define windows by example count (≥ 250 k-examples per window) rather than 100 steps. Add k = 2 as the within-arm control (its share barely changes). Further: at 10⁻² the LH-s policy halts by loop 3 for 99.6% of queries, so the block's gradient at depth is policy-gated (L23); a moved T-loop t95 would be misread as "the block is data-hungry".

## C. Budget and power

Recomputed from `SEC × STEPS`: 2.17 + 1.08 + 2.89 + 1.49 + 1.73 + 0.54 + 2.89 + 0.14 = **12.9 h**; longest-first assignment of the 10 stacks gives 6.49/6.37 h as stated. Assumes per-run-step cost flat in stack size (compute-bound at 6; plausible at 10) and linear in T for LH-s24 (an upper bound). Pilot timings included evaluation. Credible; a 50% overrun still fits; resumability covers a one-GPU fallback.

Seeds: 10/5 suffice for the 0.97-vs-0.16 lines; not for P3's "within 0.02 at every one of 14 k" (LO is 0.89 ± 0.11 at k = 32), for 5-seed LH-s24 with bimodal outcomes (s.e. ≈ 0.17), or for 2× ratios from 3–5 censored S8 runs. Per-seed statements fix most of this free. The paired bootstrap named in §8 is not in `analyze.py`; write it before unblinding.

Budgets: looped 6k/8k fine. S8 25k at 1e-4: the best lr was the lowest tried — not bracketed (rule 4). LH-s24's 8k is a guess. Cut first: LO main 10 → 5 (0.54 h; spread 0.02), LH-x 5 → 3 — not the 10⁻² LH-s cell.

## D. Fairness and labels

- LH-Δ vs LR compute: both count loops including the settle loop; fair, except LR sits on LF's block (lr 1e-3, 6k, full-depth loss) — fixed by B(i).
- S8: 3× steps, 7.8× parameters, stated; lr not bracketed; schedule length differs. P5 is correctly "optimization"; **P6 is labelled mechanism but is the same measure** — the inference is fair only for the LO line and only with B(iv) fixed.
- P3 "mechanism (compute allocation)" is a performance comparison about whether a 2-parameter head detects the final jump; label it so.
- P1 is a mechanism claim only conditional on both heads passing in range and on the λ-trace signature (ramp after T vs flat floor). The L17 pair shows the *measure* detects the mechanism, not that the trained arm will.

## E. Literature

- Geiping 2025's zero-shot KL exit **is** LR; Q2's "the free rule suffices" and P7's LR line are a reproduction at exact demand (Entry 56 says so; the write-up must too).
- Injection necessity (G2) and LF overthinking (P7's T = 40 line) are Bansal 2022 / Schwarzschild 2021; foregone.
- CALM (Schuster et al. 2022) exits on softmax response and hidden-state saturation with a calibrated threshold — LH-Δ's features, in distribution. Cite it; the open part is extrapolation of a *learned* change head beyond its training depth.
- PonderNet reported parity extrapolation with a state-reading head; P1's LH-s line is a setup-specific replication, not a general fact.
- Mixture-of-Recursions routers are never evaluated beyond training depth; Q1 remains open.

## F. Issues

**Blocking.**
1. Gate LH-Δ (and LH-s24) before the grid: 2 seeds × head-lr {3e-4, 3e-3} × 8k ≈ 0.6 h. Preregister the in-range precondition ("no test" below 0.95 at k ≤ 8) and the flat-λ-floor reading; record the head's (w₁, w₂, b) and mean chasing λ per run (not saved today).
2. `kl_exit` on every looped arm's own block.
3. Fix P7's LR line (≤ k + 2.5, or by k) and P3's anchor (≈ k + 2 out of range).
4. Restate P6's S8 and LH-s lines censoring-aware and one-sided; give the LH-s halted line a falsifier.
5. Name the 0.6–0.9 dead zones (per-seed fractions); add ¬P1∧¬P2, P1∧¬P2 and no-test rows to the table.

**Non-blocking.** Example-count windows; k = 2 control; LH-s24 step-probe/λ predictions; LR threshold frontier; "one-parameter" wording; P3/P6 labels; bootstrap code; S8 lr bracket; CALM citation.

**Add if < 1 GPU-h each.** LH-Δ at factor 10⁻² × 5 seeds (0.74 h): the sharpest Q5 contrast — a change head's halted t95(3) should *not* move while LH-s's does; deferring it (L175) costs a second session. LH-s with k_max = 4 (0.7 h). S8 at lr 3e-5 × 2 (0.7 h). Free: LR thresholds, ‖h‖ in traces, head parameters.

**Verdict: run with the listed changes.** The grid is cheap, balanced and fits one session, and the questions are real and mostly unanswered by the literature. But the headline arm has never been trained, its head sits inside an optimizer-displacement bound that can reproduce the very failure signature the design wants to explain, and its falsifier is measured on a different block. Spend 0.6 h on the gate and the code change in (2) first; otherwise a ¬P1 is uninterpretable and the 13 hours get repeated.
