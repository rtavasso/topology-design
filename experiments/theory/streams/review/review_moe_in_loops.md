# Independent review of `docs/moe_in_loops.md` (15 Nov 2026)

**Verdict: sound with corrections.** I found no bug in the testbed that changes a table, and every number in the note matches `arch_*.md`. Three of its conclusions do not survive a check: that sparse-from-the-start cannot be trained, that A7 explains when the dense phase can end, and that softmax was the right choice over sigmoid for the candidate.

Checked: `loopmoe.py` line by line; the seven tests (7 pass); every number in §0, §4, §5, §6 against the tables; all 60 predictions against outcomes; about 11.5 minutes of my own runs (scripts in `review/scripts/`, one process, 3,000 steps or fewer).

## My runs

| Run | Result |
|---|---|
| R1. Candidate register (exact shift), gate reads the token only, **top-1 from the start**, program task, 4 streams | solved 1/5 (2.0e-5, 0.21, 0.39, 0.22, 0.19); failures use 1–2 streams, purity 1.00 |
| R2. The candidate on the **branch** task (gate reads the data), **top-1 from the start, no dense phase** | **solved 5/5**: 1.0e-3, 1.1e-3, 1.6e-3, 1.7e-3, 1.1e-3 (note: candidate 1.1e-3, dense 1.4e-3) |
| R3. Program task, candidate register + the data-reading gate, top-1 from the start | solved 3/3: 1.8e-5, 3.3e-5, 8.8e-5 (seeds 1, 2 failed in R1) |
| R4. As R3, 8 streams, 5 primitives | solved 1/2 (2.6e-5; 7.2e-2 with 4 streams for 5 primitives) |
| R5. **Sigmoid** scores, register `reg0`, dense 10 % then top-1 | solved 3/3: 3.8e-5, 4.8e-5, 4.0e-5 (softmax, same schedule: 2.9e-5) |
| R6. Round B's network trained dense, probed at steps 90 and 300 | Step 90: attention on position t in 25 % of passes (4 of 5 seeds), token→top-stream map not one-to-one (5/5). Step 300: attention 100 %, map one-to-one (5/5), mean top probability 0.76–0.81, top-1 error 0.05–0.11 against dense 0.018–0.023. Eight streams at step 300 (3 seeds): top probability 0.48, 0.62, 0.72; top-1 error 0.13–0.28 against dense 0.024–0.053; map one-to-one 3/3 |
| R7. Candidate, branch task, probed at its switch (steps 110, 138, 111) | top probability on IF passes 0.59–0.68 (0.84–0.90 elsewhere); branch choice right in 82–94 % of cases; top-1 error 0.17–0.18. At the end: 99.7 %, 8e-4–1.1e-3. `M` moved 9–11 % from the exact shift; attention still on position t |
| R8. Round B's B2 (`reg0`, top-1 from the start), seed 0, at the end | attention on position t in 0 % of passes; all three tokens on one stream |
| R9. Candidate, branch, a different teacher (seed 11), one run | 7.0e-4 |
| R10. Running-mean stand-in on an ordinary (not symmetrised) batch | 0.088 against a typical stream output of 0.45 |

## Errors

1. **"Gating before the streams cannot be trained sparse from the start here" (§0.2, §4, §5) is false for the candidate's own data-reading gate.** Top-1 from the first step solved the branch task 5/5 at the candidate's error (R2) and the program task 3/3 (R3); with eight streams 1/2 (R4). Round B tested only a gate that reads the token, with a register that must also learn to move. The §5 row "every stream until the gate has decided" has no evidence on the branch task, and the "Dense steps" column of round E's branch rows measures something that was not needed. Change: restrict the claim to token-only gates; add "candidate without the dense phase" to round E at ten seeds, both sizes.
2. **A7 is not the reason the dense start works, and "the dense phase is needed exactly until the gate has decided" is contradicted by the note's own tables.** The fixed 300-step start solved 10/10 (E5, F1), yet the 0.8 rule had fired in no seed by step 300 (earliest 306 and 531). At step 300 the top-1 error is 2.5–8 times the dense error (R6), so the A7 bound is not small there. What changed between step 90 (fails) and step 300 (works) is that the address was learned and the token→top-stream map became one-to-one. On the branch task the gate is undecided on IF passes at the switch and the remaining 99 % of the error is removed under top-1 (R7). Change: state A7 as a sufficient condition for a harmless switch; the working hypothesis is "dense until the argmax assignment is one-to-one", which needs its own test.
3. **The stated mechanism of round B's failure (A3: unrun streams starve) was not measured, and the note's own diagnostic points elsewhere.** Purity 0.6–0.7 means the gate is not keyed on the current token; in B2 the register never learned to address (R8). "Tokens share them (purity 0.6)" misreads purity: sharing gives purity 1.00 with fewer streams used, which is what R1 shows once the address is given. Two failure modes are mixed.
4. **Round C's "normalised against raw" comparison is confounded.** `ln=False` also removes normalisation of the register query (`loopmoe.py` lines 177 and 185). "3/5 against 5/5 was noise" is not shown: it recurred at the higher learning rate (C9 3/5, C10 5/5, same seeds), and round E tested a different network.
5. **Scoreboard:** C10 fails its own criterion (1.7e-3 against C9's 5.6e-3 is 3.3×, not within 3×): 40 held, 6 partly, 14 failed. A6 held numerically for the wrong reason (training error 2.6e-3, purity 0.68: an optimisation failure, not fitting through the state).
6. **Smaller mismatches.** "In every solved variant … each token has its own stream" is false for A6 (5/5, purity 0.68). "Error throughout on programs never trained on" is false for the length-4 column of rounds D and E: test programs are drawn independently (lines 221–222), about 39 % coincide with training programs (lengths 6 and 8 are clean). "History Entry 104" does not exist; the test docstring says A1–A6. §0.2 "failed in 0 or 1 of 5" means "solved". A5 holds to 16 passes (`LMAX`), not "any number".

## Overstatements

7. **Softmax over sigmoid** rests on top-2-from-the-start at five seeds (5/5 against 2/5), a regime the candidate does not use. Under the candidate's schedule sigmoid solved 3/3 (R5). σ-MoE's own margin for sigmoid is small (10.37 against 10.54).
8. **"Did not help"** for explorers (0/5 → 2/5, 3/5) and expert dropout (median 0.18 → 0.074) cannot be read from five seeds. The Default-MoE stand-in is untestable here rather than unhelpful (A6; R10).
9. **"Running top-1 lowers the largest gain (7.7)"**: the Jacobian of a hard top-1 map does not see the jump at the switch (line 318 evaluates with top-k on). The 2.7× is a maximum over 96 random points.
10. **All seeds share one teacher** (`Teacher(cfg, seed=10)`, line 64). Not disclosed. R9 is one supporting run.
11. **Literature.** "One router per loop is not needed": the register is a clock, so a tied gate fed a clock is a per-loop router by another name; it says nothing about per-token routers. "A tied router needs no pass signal" is not a claim MoEUT or SUT make. EvoMoE already decides which experts run from the gate's own scores (kept above 0.001, per `literature.md`), so "fixed schedule" undersells it. The σ-MoE/ST-MoE "disagreement" on expert dropout should be checked: from memory, ST-MoE's noise was aimed at fine-tuning, not exploration. §5's "not found elsewhere": `literature.md` line 302 already places one block per loop for the whole state in Routing Networks and Modular Networks; a controller attending over the instruction at each step and emitting softmax module weights is Stack-NMN (Hu et al. 2018) and MAC (Hudson & Manning 2018), neither cited.

## Tests

A3 (all four cases), A2's first half, A5 and the A7 bound are real tests. A1 is a tautology. A2's formula for the gate term is tested only as "non-zero". A4's "sum up to K" is tested only as "greater than 1". A7's proportionality holds for any gate, and both halves are single-pass: they do not support the multi-pass or training-dynamics reading. A6 is not vacuous (R10) but is exact only on a symmetrised batch.

## No fault found

No leak from teacher or held-out programs; fixed-length splits are disjoint; at run time only the selected stream reaches the output; the confidence switch, schedules and explorer do what the note says; seeds are paired; the IF token is correct; streams-per-pass figures reproduce.

## The single most important point

The note's central lesson — sparse from the start fails, a dense start fixes it, A7 says why — is a finding about a gate that reads only the token. With the gate the candidate actually uses for data-dependent choices, top-1 from the first step already works at four streams, and where a dense start does matter, what it supplies is a one-to-one assignment, not a confident gate.
