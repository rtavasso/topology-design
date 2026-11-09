# Review 5: the amended pass rules (DESIGN §17) and their implementation, 9 Nov 2026

**Reviewer:** independent (a fresh agent with no part in the design, the audit or the amendment), read-only, told not to open the pilot; it worked from the design text, the code and the dataset, with its own search engine (`eng.py`) for the cross-checks. Scripts in [review5_scripts/](review5_scripts/) as it left them (absolute paths replaced by `REPO_ROOT`; `mutate_first_version.py` is its mutation check of the first planted tests, `mutate.py` the same check brought up to the code as amended after this review). Its report follows verbatim; the handling is below it and in DESIGN §17a.

---

**Verdict: sound with changes.** Two changes block the launch (text / derive / analyze only; no run changes). Otherwise `verdicts()` implements §17 as written.

**Blocking**

1. **P3 clause 2's must-fail set uses the wrong positions.** At cap 5 no action is ever valid at prefix positions 7–8, nor an A-rule at 5–6 (brute force over all 47,155 terms of ≤ 5 leaves; eval5 canonical proofs never exceed position 6). The 105-way head's never-labelled rows are 7–12, not 9–12 (§12a.5, derive §5/§9c, §17). On labelled slots, **986** eval7 training-world instances have no route within T (D 361, I 222, AD 197, ID 182, A 20, CD 2, C 1, AC 1), not 318; **2,007** have every shortest proof on an unlabelled slot (0.15–0.63 per training world, A 0.39), not 583. Planted through `verdicts()`: an oracle-strength RT-absolute failing exactly the 986 gets R 0.851 against bar 0.953 → **falsified 0/5**, "a table beyond the artefact", for the artefact itself. With the corrected set it holds 5/5; failing all 2,007 gives R 0.73–0.78 → falsified. Fix: rebuild `eval7_mustfail` from the slots labelled in RT-absolute's stream; read "no more" on the 2,793 instances with a shortest proof on labelled slots; report the 1,021 detour-only instances apart; correct §16's unaddressable shares.

2. **P2's non-composing floor is not robust, and one variant passes P2** (wording plus one derived row). The same policy with σ-greedy as last resort instead of first-by-(rule, path): 0.727 / 0.458 (pairs / triples); with σ-greedy whenever no training-world move exists: **0.751 / 0.497** (cross-checked with my own engine) → `verdicts()` returns **pass**. Restricted to training worlds contained in W: 0.354 / 0.122; tie-break variants 0.48–0.54 / 0.19–0.21. So 0.531 is mid-range, and a masked pass does not separate planning over the unseen pair from single-world plans glued by greedy moves. §17 should say so before launch, print this reference beside lookahead-3 (0.727 / 0.507), and state what a pass claims.

**Decide before launch (thresholds)**

- P3 clause 1: above the lookahead-3 base (0.923) ρ stays 1.057 although R ≤ 1/base, so the bar is a flat 0.951: an oracle-strength router keeping 93 % at sizes 6–7 is falsified 0/10 (§6: 10/10). The nearest rung flips at base 0.870: keeping 97 % is falsified at base 0.86 and passes at 0.88. Suggest capping ρ at 1/base.
- P7 clause 1 can hardly fail: the band is about 0.83–1.08 × RT-canon, "above" is unreachable once RT-canon's unmasked median ≥ 0.943, and the auditor's example-blind planner (0.61 against 0.69) sits inside. Print RT-shuffled unmasked beside it.

**Accounting (last column)** — accurate for P2, P3 clause 1, P5. Missing: P8 clause 1's withdrawal is a loosening (a §6 failure becomes unscored; if the masked gap exceeded 0.05, P8 would hold where §6 failed); P8's unmasked "holds" and its 0.05 are not blind (the sanity table puts the unmasked gap ≥ 0.14 above the masked one); P7's masked → unmasked switch; clause 2's new tolerance. P2's dropped RT-all clause is the only route from a §6 non-pass to a pass; the falsifier is not weakened (RT-all ≥ 0.9 on 608 implies ≥ 149/209). §17 should say whether RT-all's pilot average was seen and list PILOT.md's four items inline (P6's bar is twice the auditor's 0.05). "Undecided" also labels ten seeds at 0.60; call that "prediction not met, not falsified".

**Code against text** — strata, readings, inequalities, counts and both reading matrices match. Differences: P2 waives the triples clause if that stratum is empty; P3's summary verdict is clause 1 only; P5 gives "falsified" precedence where both apply; the §6 sections still print "P2 passes when … RT-all ≥ 0.9" unlabelled.

**Tests** — every rule has a planted pass and fail, but 42 of 60 mutants survive: every seed-count boundary (6/10, 3/10, 7/10, 4/5, 1/5, 2/3), every threshold except P3's 0.9 factor, median → max, and swapped readings for P5, RT-all's gate and clause 2's rest. Swaps are caught for P2, P7, P8 and the must-fail rate.

**Confirmed (own engine)** — comb flag on all 1,673 held-out binding instances: 0 mismatches, 209 (AI 47, CI 162) / 509. Must-fail as defined (positions ≤ 8) on all 583 flagged: 0 mismatches, 318. Matched ratio 1.126 / 1.106 / 1.057 (pairs 1.317 / 1.237 / 1.116). Reference 0.531 / 0.212, re-run in reversed instance and dict order: 0 mismatches; total-order key, deterministic. Planted P2: greedy, lookahead-2, the reference → falsified; lookahead-3, oracle → pass; greedy without RT-all → undecided. P3 clause 1: greedy kept at 85 % → §6 10/10, §17 falsified. Strata 1,408 (0.404 / 0.686) and 1,673 (0.325 / 0.644); P6 n 2,213, uniform reference 0.102.

**Not checked** — anything neural; the slots actually labelled in the training stream (I used validity at cap 5, a lower bound on the artefact); the stored ladder outcomes; the auditor's † figures; the pilot (not opened).

---

## Handling (9 Nov, same day, before the launch; DESIGN §17a)

**Blocking 1 — taken in full.** The slots the training stream labels were counted from the eight training files (`derive.labelled_slots`): 52 — all eight rules at positions 0–4, six (no A rule) at positions 5–6, none at 7–12 — exactly the reviewer's validity bound. `derive.py` §9c now classifies each of the 4,800 eval7 instances of the training worlds by those slots: a shortest proof on them (2,793), only a detour within T (1,021), no route (986: D 361, I 222, AD 197, ID 182, A 20, CD 2, C 1, AC 1) — the reviewer's counts to the instance. P3's second clause reads "no more" on the 2,793, the artefact on the 986, and reports the 1,021 apart. The design's "rows 9–12" (§12a.5) and §16's unaddressable shares are corrected in §17a; the error is L41.

**Blocking 2 — taken as asked.** The glued reference (`trainworld_policy(..., fallback="greedy")`) is in the library, stored per instance and printed beside the rungs: 0.751 / 0.497 (the reviewer's figures reproduced). §17 now says what a masked pass claims and what it does not, and the analysis prints, unscored, the two rows that bear on the difference: the unmasked reading, and RT-canon on the 52 / 256 instances the glued reference does not solve. The falsifier's floor stays the training-world reference with the nearest-in-metric fallback (0.531 / 0.212), described as mid-range among the variants, with the reviewer's range quoted.

**Thresholds.** P3 clause 1: ρ is now interpolated between the rungs' (size-≤ 5 success, ratio) points and capped at 1 / success (`analyze.rho_for`): continuous across rungs, and 0.9 exactly for an oracle-strength router. P7 clause 1: said to be a consistency band that can hardly fail; RT-shuffled's unmasked rate on the same stratum is printed beside it, unscored (a scored comparison was not added — the author had seen the pilot's unmasked rates of both arms).

**Accounting.** Every item added to §17's table and to its statement of what was seen of the pilot (RT-all's all-world average included). The dropped RT-all clause: a §17 pass that would not have been a §6 pass is labelled so in the verdict line itself ("§6's further requirement (RT-all ≥ 0.9 on the same strata) is not met").

**Code against text.** P3's summary verdict now names both clauses; "undecided" between a pass and a falsifier is "not met, not falsified"; the §6 sections are labelled "printed, not scored"; the waived triples clause (empty stratum only) and P5's precedence are stated in §17.

**Tests.** The planted tests were rewritten to pin every count, threshold, statistic and reading at its boundary; threshold comparisons are rounding-safe (`_ge` / `_gt`). The mutation check brought up to the amended code (`review5_scripts/mutate.py`): 76 mutants — every count, threshold, statistic and reading of `verdicts()`, with those for what this review added — 76 killed (75 on the first run; the survivor, the 0.9 factor moved to 0.89, needed a finer-grained cell and now has one).
