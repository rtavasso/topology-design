# Re-check of revision 2 (HEAD 5fb517d), before stage 0

**Verdict: fix first — three small edits to `stage0.py` (points 1–3), then run stage 0.** Five tests pass. Checks used 86 s; runs below are untuned (rate 0.01, 4,000 steps, coverage 9, split 0, one seed).

**Stage 0**

1. **The control's regime column is wrong.** With `same=True` the references are 0 (`allmean`, `blend`, `trans`), so `regime()` returns "garbage" for a run with held-out error 1e-6. Skip the label there; give R0b's "within 1.5×" an absolute floor.
2. **Width 120 runs at width 40's rate** (the rate file is per task and arm, `stage0.py:48`), so `mono` can be called under-fitting for lack of tuning. Give `rates` a width.
3. **The R² column misreads on `select`.** The true rule scores 0.49 and `blend` 1.00; share of weight on the pair is 0.95 and 1.00. Print those references, or regress on chosen/other.
4. "no-choice" spans 0.25–0.79 here (half of `blend` up to `allmean`). Split it at `blend`.

Implemented correctly: `determined` and `trans` (forced share 0.42, matching my 0.29–0.57), nested 2-connected splits, evaluation inputs fixed per split, checkpoints.

**Arms and contrasts (before stage 2)**

5. `tok_mono` is the arm I probed in effect: 0.020 / 0.078 on `select`, 0.0025 / 0.0047 on `sum`. `tok_streams` is a fair per-token streams arm: 0.0074 / 0.053 and <1e-4 / <1e-4.
6. **The support reading for C1 is mis-specified.** It does not require the monolith to recombine less. On `sum`, C1 is far below 1/2 with both arms *full*: streams have the teacher's exact form, so that is fit. On `select` the held-out ratio was 0.67 and the seen ratio 0.37. Require the held-out contrast to be at most half the same contrast on seen worlds, and the two arms to be in different regimes.
7. C3 ≤ 1/5 is foregone (1.5 against 0.08); label R2a a confirmation.
8. The identical-arm null uses `tok_mono` only; add `mono`.

**Your note 4** is incorrect: revision 1's default was already softmax (`worlds.py:21` at 808286e), so my streams numbers were softmax runs.

Check script (not in the repository): `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/review_w1/rev2_check.py`
