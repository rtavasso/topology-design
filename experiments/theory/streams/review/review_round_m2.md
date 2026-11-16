# Independent review of the rounds M / M2 section of `docs/moe_in_loops.md` (15 Nov 2026)

**Verdict: sound with corrections.** Almost every number recomputes, the predictions precede their runs, and the three "errors" of the results review are answered. But the out-of-range paragraph states the opposite of what the data show for doubled inputs, and that changes how far "no accuracy difference that survives tuning" reaches. Two table entries and one count are wrong.

Scope: recomputation from the six result JSON files and reading of code only (`scripts/revm2_recompute.py`, no training). p-values are exact two-sided sign-flip tests on per-seed log ratios, ten seeds. The 128 multiply-adds of the attention read and the pruning figures (8.8e-5 / 2.5e-5) are taken from the earlier review, not rechecked.

## Checked and correct

- Second table: every cell (93, 398, 727; 79; 36; 174; 0.8 / 4.0 / 0.63; 1.1 / 19.5 / 0.62; 16, 12.7; 2.1, 2.1; 7.8, 18, 15) and every seed count.
- First table: candidate errors (6.5e-5, 6.8e-5, 1.1e-3; 1.8e-5, 1.4e-5, 6.3e-4; 3.4e-6, 7.1e-6, 3.2e-4).
- Point 1: medians 0.91–1.22 over the three regimes; per seed 0.74–1.74 in round M; no-attention block 2.4e-2, 10/10, 1.10–1.61.
- Point 3: 7/10 twice; step 1,000: 10/10, 10/10, 10/10 and 9/10, 9/10, 5/10.
- Point 5: 176/580, 208/1,160, 3.3, 5.6, 2.3, 3.8; training cost 194 and 186 (four streams), 293 and 342 (eight), medians; 608/1,032.
- Point 6: 8–10 of 10, 5–10 of 10, 0–2 of 10 (decaying rates).
- Predictions: S1–S8 seven held, S1 failed (3.98); X4, X5 held, X6 partly (10/10 and 5/10), X1–X3 failed. R1–R9: 7 held, 2 failed. Scoreboard rows and totals: 68 = 46 + 6 + 16 before M; 98 = 66 + 10 + 22.
- Commit order: S1–S8 committed 16:57, first M2 file 17:07; X1–X6 committed 17:37, first extension file 17:47. `arch_m2.py` unchanged since.
- Results review: error 1 (not best against best), error 2 ("rises least", length-8 jumps and the rate), error 3 (the no-attention gap) are each answered in the text.

## Errors

1. **"Outside the trained range … nothing separates the blocks consistently: with inputs twice as large the candidate's error rises about as much as the per-unit gate's."** It does not. In every decaying cell the candidate's error rises more than the per-unit gate's in 10/10 seeds on the program task (10.9 against 6.4, 11.0 against 6.8 at 0.01; 11.4 against 4.3, 12.0 against 5.7 in the extension) and in 10/10 and 8/10 on the branch task. Best against best (the extension), the per-unit gate's error at doubled inputs is 0.23 times the candidate's with four streams (range 0.08–2.4) and 0.30 with eight (0.15–1.3); the candidate is better in 1/10 seeds in each (p = 0.008, 0.004). This is the best-resolved difference between the two blocks in the tuned regime and it goes against the candidate. It also makes point 4's "the one accuracy difference between streams and gated units that held up" false as written. **Change:** report it; restrict point 3 and §0 7(c) to "at the trained input scale"; in point 4 say "the one difference in the candidate's favour".
2. **First table, "Rates chosen", extension column.** "Candidates 0.03; per-unit gate and additive 0.1" is wrong twice: the three-stream candidate (N1s) chose 0.1 and the branch per-unit gate (N14) chose 0.03. The two-layer block chose 0.1.
3. **Point 6, "in the two seeds probed".** One seed (0), two runs: the candidate at 0.03 and the per-unit gate at 0.1 (`arch_m2.md`).
4. **First table, "seeds in which the candidate still addresses the sequence at 8 passes: 9/10, 9/10, 1/10".** These are ten minus the count of length-8 error above three times length-4 error. The address at eight passes was measured in one run per regime; the stored `addr` is at length 4. Two of the nine extension seeds counted as lost have length-8 errors of 4e-5 and 9e-5. Relabel as the error ratio, or give seeds below 1e-2 at length 8: 10, 10, 3.

## Overstatements

5. **Point 6, "a property of the shared controller, not of the block".** Not shown. The damage follows the rate, and the rate follows the block: rows at 0.1 break in 10/10 seeds at length 8 and 7–10/10 already at length 6 (additive: 8/10 at both); rows at 0.03 in 5–9/10 at length 8 and 0–3/10 at length 6. Seeds below 1e-2 at length 8: candidates 3, 5 and 6 of 10; per-unit gates 0, 0 and 3. Paired length-8 ratios, per-unit gate over candidate: 2.8 (7/10), 346 (9/10), and on the branch task, where both ran at 0.03, 3.9 (9/10; 2.1 at length 4). **Change:** "in every block, more at 0.1 than at 0.03; no program-task row compares two blocks at one high rate".
6. **Point 3, "grouping into streams does not cost accuracy".** In the extension the per-unit gate is ahead in 7/10 seeds in both rows, geometric means 0.68 and 0.69; p = 0.15 and 0.16 each, 0.04 pooled over the twenty. "Nominally ahead" is fair for one cell. "Does not cost" is not: say "costs at most about 1.6 times at the trained scale, unresolved at ten seeds". The rate grid steps by three, and one step moves either block's three-seed training error 3–9 times, more than the effect.
7. **Point 3, "trains faster … in all three regimes".** Rounds M and M2-as-designed are the same runs up to step 1,000 apart from the cosine factor (same seeds, same 0.01, identical switch steps; step-1,000 errors differ by a uniform 0.8). That is two regimes. With eight streams at the best rates the candidate is not faster: 0.91 (5/10) at step 1,000 and 0.74 (3/10) at step 2,000. Unreported: in 4/10 of those seeds the confidence trigger never fired and the switch was forced by the cap at step 1,501 (block cost 712 in those seeds, against a quoted 290–340). The claim holds for four streams, and for eight only at 0.01. More exact wording: the per-unit gate's best rate is three times the candidate's, so a shared 0.01 under-trains it more.
8. **Point 4 and §0 7(d).** The 2.1 is robust over seeds (1.3–3.2 and 1.4–2.9, 10/10, both blocks on a flat optimum between 0.03 and 0.1). But only one per-unit-gate configuration was run: 27 units with a gate MLP of 12 hidden units, against the candidate's 24. Attribution to "streams against gated units" is not isolated. Say "this configuration".
9. **Point 7.** Every 18-unit seed above 1e-2 has lost the address in range (`addr` 0.25, 0.25, 0.0): controller failures, not "no spare stream". The per-unit gate in the extension is bimodal (four seeds near 1e-6, five near 1e-3). Its 475 parameters against 337 (41 % more) are not stated, although "within 3 % unless stated".
10. **Point 1, "by construction".** The additive block's held-out error exceeds its training error in 10/10 seeds in M2 as designed (1.02–1.23) and 9/10 in round M — the criterion the text uses for the no-attention block. "By construction" holds to about 10 %, not exactly.
11. **Point 2.** 80–700 is the program task; on the branch task ungated blocks are 8–21 times worse. The five teachers are one seed each, constant rate. Point 5's "sorted itself into groups" is one network.
12. **§0 7(c), "once each has its own learning rate".** Each row had its own rate in M2 as designed too; what changed was the top of the grid.

## Suggestions

13. The section cites this review as having shaped it before it existed; state what was changed.
14. X1–X6 were written after seeing six three-seed numbers at 0.03 (`arch_m2.py` says so; the note does not). X3 failed on its letter although the note's own reading ("unresolved") is its spirit.
15. Still unanswered from the results review: the per-unit gate is the better block for four of the five other teachers; the 24-unit rows were not run.
16. AGENTS.md rule 4 asks about 29 seeds and a bootstrap P(A > B) for headline claims; §0 7(c) and 7(d) rest on ten-seed medians, one teacher and rates chosen on three seeds. Keep them labelled exploratory.

## The single most important correction

Finding 1. With both blocks tuned, the per-unit gate is 3–4 times more accurate at doubled inputs in 9/10 seeds at both sizes, and the text says nothing separates them. The supportable summary is: at tuned rates the per-unit gate is level or slightly ahead at the trained scale and clearly ahead at doubled inputs; the candidate is ahead on the branch task (one configuration) and at eight passes (confounded with the rate); what one stream per pass buys is compute.
