# Review of RESULTS.md (worlds carrier, stages 0–3)

**Verdict: sound with corrections.** Every number reproduces from the JSON, the pre-fixed readings are applied correctly, and I found no bug that biases a contrast. The verdict on I37 is worded more strongly than the data and the carrier allow.

Method: recomputation and code read (`review/scripts/res_*.py`); three probes, 247 s of training (7 s over the allowance).

## Errors (all small)

1. **R2c is scored "held".** At coverage 7 its first clause fails: C2 ≤ 1/2 in 2/12 splits (11/12 at coverage 9).
2. **`sum`, "again worse than not moving".** Median 1.06, above 1 in 8/12 (above the references 12/12).
3. **Control, "no gap at all".** `tok_mono` is 1.3–1.7× its seen error and passes R0b on the 1e-3 floor; three splits.
4. "History Entry 108" does not exist yet. Deviations from DESIGN §6 are not listed (coverages 7 and 9 for 9 and 11; `sum` at 9 only, without nulls or the wide and weight-decay monoliths).

Readings: I get the same four. "Gating, not grouping" is met and should be said. Under §5's every-rate rule C1 is 0.49 at one pairing; rate 0.003 qualifies for both per-token arms (1.6–1.7× best seen error) and was not run. Probe, `tok_streams` at 0.003, splits 0–4: 0.55 against stored `tok_mono` (0.54 at 0.01 on those splits).

## Overstatements

1. **"No, as posed" — the most important correction.**
   - The ledger words I37 as parallel learned transformations with a learned soft recombination and cites Neural Interpreters as the architecture. `tok_mono` is that, with shared weights. On that reading C3 = 0.05 is support; "no" holds only for "separate weights per stream".
   - The per-token success is close to constructed. Each branch sees a state and one named primitive, all covered in training; on `sum` nothing is held out from the block (Entry 106, point 2), on `select` only the bid comparison. Entry 107's property (a) is not met. Limits says this; the headline does not.
   - "Separate weights without that shape did not recombine" contradicts item 4: on `sum`, `streams_self` is in the *recombines* regime in 8/12 with `mono` above its references 12/12.
   - Change: use DESIGN §7's closing sentence. The informative results are that separate weights were not needed and that a gate reading a pooled summary stays at blend level on `select`.

2. **"Streams do not supply recombination" (C1).**
   - "The gap is larger with streams (4.7× against 2.4×)" is the held-out/seen ratio that DESIGN §4 excludes because it grows with fit.
   - A sound version exists: held − seen is equal, 0.031 against 0.034 (per-split ratio 1.08 [0.67, 1.37]; 0.99 at rate 0.03).
   - "Fit explains it" is untested. Better fit barely lowers `tok_mono`'s held-out error: 8,000 steps give seen ×0.76, held ×0.91; my width-120 probe (17,208 parameters, splits 0–2) gives seen ×0.64, held ×1.02, and `tok_streams` is at 0.48–0.59 of it.
   - The advantage is sign-consistent at equal rates: 0.65, 0.80, 0.85, 0.69 (9, 11, 9, 11 of 12); 0.69 and 0.74 against the second initialisation. The 0.49 and 1.08 are cross-rate pairings, and held-out error rises with rate in both arms (×1.2–1.4).
   - Against it: the null also "wins" 9/12, and at coverage 7 the intervals overlap.
   - Change: "both per-token arms recombine; streams lower held-out error by 15–35 %, short of the pre-fixed factor of two; fit or recombination is unresolved."

3. **"They know which two are available and do not choose."** One pass: `streams_self` is nearer the true update than the blend in 11/12 (0.46 against 0.56); 0.76 of the regression weight is on the pair (0.94 per-token). That is an error level, not a mechanism. At coverage 7 it is 1.5× `blend`.

4. **Mechanism, 0.999 against 0.98.** Selection by margin: mean |margin| is 1.43 forced, 1.06 seen, 0.67 unforced (probe, `tok_mono`, split 0), and a scalar bid guarantees forced decisions once trained ones are right. Margin-matched: 0.984 seen, 0.947 unforced. The learned bid difference is linear in the true margin with the same slope on both (correlation 0.96, 0.95): the teacher's form, recovered.

5. **"At any width or weight decay."** Two widths; decay at width 40 only, best at the grid's edge (1.30, 1.23, 0.92), not extended.

6. **Missing from Limits.** A stream is V·tanh(Bh) of width 16, which contains the teacher's primitive exactly; the single block must approximate six through additive conditioning. Every "ratio on trained worlds" reflects that.

## Checked, fine

- Both tables, intervals, nulls, and the figures in items 1–5.
- The control claim holds narrowly; it is weak, since a net fitted to a world-independent target need not read the world.
- `trans` and `determined` are correct and show extension beyond chaining (unforced 0.93).
- Code: evaluation inputs shared per split, pairing by split, rate overrides, bootstrap. Null arms change initialisation and minibatch stream (conservative).
- Commit order against file times. The extra rates were chosen after C1 was known.
