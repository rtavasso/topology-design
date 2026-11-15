# Independent review of the round M design (`arch_m.py`), before it is run (15 Nov 2026)

**Verdict: run after changes.** No bug found and the matching is honest, but the per-unit-gate verdict depends on an untuned learning rate, and three predictions (Q1, Q5, Q6) would hold for reasons that say nothing about streams.

Checked: the new code paths line by line; every row instantiated and counted; ten single runs (seed 0, teacher 10, about 3.5 minutes of CPU; scripts in the scratch folder `review_m/`). Every probe number below is one seed.

| Probe, 3,000 steps | train, length 4 | held out, length 4 | length 8 | length 4, inputs x2 |
|---|---|---|---|---|
| M1 candidate, lr 3e-3 | 2.9e-5 | 2.6e-5 | 3.3e-5 | 2.2e-4 |
| M1, lr 1e-2 | 2.3e-5 | 2.2e-5 | 3.9e-5 | 8.2e-5 |
| M3 additive, lr 3e-3 | 4.5e-3 | 4.6e-3 | 4.8e-3 | 4.4e-2 |
| M3, lr 1e-2 | 6.5e-3 | 6.7e-3 | 6.3e-3 | 4.9e-2 |
| M3, lr 3e-3, 6,000 steps | 2.1e-3 | 2.2e-3 | 2.2e-3 | 2.5e-2 |
| M4 per-unit gate, lr 3e-3 | 1.1e-4 | 1.1e-4 | 1.1e-4 | 6.5e-4 |
| M4, lr 1e-2 | 2.8e-5 | 3.0e-5 | 4.8e-5 | 6.2e-5 |
| M4, 18 units, lr 3e-3 | 2.2e-3 | 2.1e-3 | 2.4e-3 | 8.2e-3 |
| Two-layer block on the program task (16+16 units, 608 block parameters) | 4.8e-3 | 4.6e-3 | 5.4e-3 | 2.7e-2 |
| M7 no attention, 14 units | 0.104 | 0.109 | 0.52 | 0.30 |
| M7, 29 units | 0.046 | 0.048 | 0.84 | 0.15 |

## Must fix before running

1. **Learning rate is not tuned per arm, and it flips the M4 verdict.** At 3e-3, M4 is 4.4 times the candidate; at 1e-2 it is 3.0e-5 against the candidate's best 2.2e-5, and equal or better at doubled inputs (6.2e-5 against 8.2e-5). M3 stays about 200 times worse at either rate. "Neither side is tuned" (docstring line 19) is not accurate: rate, steps and schedule were chosen over eight rounds on the candidate with this teacher. Change: three rates (1e-3, 3e-3, 1e-2), three seeds, for M1, M3, M4, M12, M13; run ten seeds at each arm's best; report best against best (AGENTS rule 4).
2. **Nothing has converged at 3,000 steps, so ratios are snapshots.** Between steps 2,000 and 3,000 the error falls 3.4 times (M1), 4.1 times (M4), 1.4 times (M3); M3 halves again by 6,000. At 1e-2 the candidate read 2.6e-4 at step 2,000 and 2.2e-5 at 3,000. Change: record errors at 1,000, 2,000, 3,000 steps, plus 6,000 for three seeds, or decay the rate to zero.
3. **The table has no training error, so Q1 and Q6 cannot be read.** Recombination is held-out over training error. That ratio was 0.9 to 1.05 in every probe, M7 included. Q1 holds by construction: the block sees only (h, r_t), and every token occurs at every position in training. M7 fails to fit (0.104), not to recombine. Change: print error on training programs of length 4 (`out["train"]` pools lengths) and the ratio; restate Q1 on the ratio.
4. **M7 is confounded twice.** Its 14 units are below the 18 directions that Q3 says are needed, and at lengths 6 and 8 it reads input columns that training never touched (change exactly 0.0, positions 5 to 7). Change: drop Q6's length clause; run M7 at length 4 only, at a width that is not the limit (36 or more, labelled unmatched), or drop it.

## Should fix

5. **Score Q4 and Q5 as within-model ratios per seed, not by the 1e-2 threshold.** Length 8 over length 4: 1.26 (M1), 1.03 (M3), 1.01 (M4). Doubled inputs over own error: 8.5, 9.5, 5.7. Additive conditioning showed no special fragility; Q5's "at most 5/10" would hold only because M3 starts 180 times higher. Report paired log-ratios and seeds won. `med` (line 72) is the sixth of ten.
6. **M4 is the streams family**: 29 width-one streams, sigmoid scores, dense, not renormalised. Label it so; I37's comparison rests on M3, M9 and a two-layer block, which should be added on the program task.
7. **Compute.** Multiply-adds per pass, block only: M1 176, M2 608, M3/M4 580, M5 240, M8 208, M9/M10 1,160, M12 608, M13 989, M14 684, M15 1,000. Units-matched is generous to the monolith (FLOP-matched is 9 units). M5 and M11 are arithmetic (12 < 18, 12 < 30): replace them with M3 and M4 at 18 and 24 units.
8. **Record the address.** Beta and M train; under M4 beta fell to 2.5. `aux["attn"]` is not recorded on the monolith path (`continue`, line 227).

## Note in the write-up

- Counts verified: 705/706/703, 1,329/1,331, 1,169/1,140/1,155; 97 are shared; one of four streams is spare, so matching favours the monolith. Initial scales do not handicap it (output rms 0.057 M3, 0.028 M4, 0.029 M1).
- Seeds are paired (programs, embeddings, inputs). The register advances before `continue`; the early return is safe. `arch_m.py 5` crashes on the unused row.
- Ten seeds suffice for 5x and 10x (candidate seed range 3.5x); 2x and 3x only as paired ratios.
- Cost: about 15 s per monolith run and 22 s per candidate run, so roughly 50 minutes in one process as designed.

## The single change

Finding 1, paid for by dropping M2, M5, M11, M14 and M7's extra lengths.
