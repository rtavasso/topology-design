# Independent review of the round M results and their write-up (15 Nov 2026)

**Verdict: sound with corrections.** Every number recomputes and the predictions were committed before the runs (a737c25, b56f337 precede 5e853ee; `arch_m.py` unchanged since). The additive-against-gated result stands. The conclusion "not streams — grouping buys only compute" does not: it depends on the candidate's learning rate, and it fails with eight streams and at 18 units.

Checked: all figures in the note's table, bullets, point 7 and the scoreboard from the two JSON files (`scripts/revm_recompute.py`); multiply-adds by hand; twelve extra runs, 414 s, one process (`scripts/revm_runs.py`). Extra runs are three seeds (0–2) or one, so they are leads, not results.

| Extra run | Held-out error, length 4 | Against |
|---|---|---|
| M8 (eight-stream candidate) at rate 0.003 | 1.1e-5, 2.2e-5, 1.6e-5 | same seeds at 0.01 (reported): 2.7e-5, 1.1e-4, 8.0e-5; M10 is 5.5, 3.5, 7.0 times worse |
| M10 (eight-stream per-unit gate) at 0.003 | 8.2e-4, 7.0e-4, 1.3e-3 | at 0.01: 6.3e-5, 7.7e-5, 1.2e-4 — 0.01 is its better rate |
| Candidate, 3 streams of 6 units (18 units, 337 parameters, 96 multiply-adds), 0.01 | 2.5e-6, 4.8e-5, 3.0e-5 | M4a (18 units, per-unit gate): 2.6e-5, 9.1e-4, 9.5e-4 |
| Branch: per-unit gate with logits C r + MLP(r, h); 27 units, gate hidden 12; 1,176 parameters, 1,032 multiply-adds; 0.01 | 2.0e-3, 2.1e-3, 1.5e-3 | candidate M12: 3.0, 1.9, 1.1 times; two-layer M13: 0.13, 0.19, 0.12 |
| same, 0.003 | 1.7e-3, 2.5e-3, 2.2e-3 | M12: 2.4, 2.3, 1.7 times |
| M4 seed 9 at 0.01 (reproduces the stored 2.5e-5, 7.9e-2 at length 8) | attention on position t, passes 6–8: 0.9, 0.4, 0.04 | at 0.003: 1.0 at all eight passes, length 8 / length 4 = 1.0 |
| Gates of that trained M4 | 10–12 of 29 units open per token, 16–19 below 0.05; 25 units serve one token, 4 serve two | only the 12 most open units per token computed: 8.8e-5 (all 29: 2.5e-5; 9: 6.6e-4) |

## Errors

1. **"The per-unit gate matches the candidate … not streams" (bullet 2, point 7) is not best against best.** The selection put the candidate at 0.01, worse than 0.003 (the note says so for four streams); the eight-stream row inherits it. At 0.003 the eight-stream candidate is 2.3–4.8 times better than reported and the per-unit gate is 3.5–7 times worse than it (3/3 seeds; every M8 run is below M10's ten-seed minimum of 2.4e-5); R7's "within 3x" clause would fail. With four streams it survives (round E G8, the same rows at 0.003: 2.7e-5 against M4's 2.8e-5). The stored checkpoints agree: at step 1,000 the candidate is ahead in 10/10 and 9/10 seeds (3.3 and 24 times); 3,000 steps is where the curves cross. At 18 units grouping buys accuracy: the no-slack candidate, with fewer parameters, beats M4a by 11–32 times in 3/3 seeds. **Change:** rerun M1 and M8 at 0.003 (ten seeds), sweep the eight-stream family itself, and restate: gating buys about two orders of magnitude; grouping buys compute *and* faster training, visible when streams are many, units few or steps short.
2. **Out-of-range paragraph.** "The candidate's error rises least (2 times)" is the 0.01 noise floor inflating the denominator: at 0.003 M8 rises 9–12 times to the same absolute error (1.3e-4–1.9e-4); the design review's probe had 8.5. The length-8 jumps are tied to the rate: M4 seed 9 keeps the address at 0.003, and round E G8 (candidate, same ten seeds, 0.003) has no jump (maximum 6.1e-5). The selection rule (training error at length 4) cannot see this. **Change:** say "at rate 0.01"; drop "rises least".
3. **"No block has a recombination gap."** The no-attention block's held-out error exceeds its training error in 10/10 seeds (1.10–1.61). It under-fits *and* has a gap. The 0.98–1.22 are medians; per seed 0.74–1.74.

## Overstatements

4. **"Streams do not recombine better" (point 7 headline); §7 "round M says what to expect".** "By construction" is right — the block sees only (h, r_t) and the register never reads data — which means recombination was not tested. Say "cannot be tested here"; delete the forecast.
5. **Branch, "6–8 times better".** No gated row. A per-unit gate that reads the data is 5–8 times better than the two-layer block and 1.1–3.0 times worse than the candidate (candidate ahead 6/6). Add the row at ten seeds; the lead attributable to streams is about 2, not 6–8.
6. **Compute.** Arithmetic correct but block-only: with the shared read (128 multiply-adds at length 4) the pass ratios are 2.3 and 3.8, not 3.3 and 5.6; branch 0.66, not 0.6. Training forward cost averages 194 (M1) and 292, at most 580 (M8). M4a is bimodal (three seeds at 6e-6–4e-5, seven at 9e-4–1.5e-3; ratio of medians 13.5 against the quoted 7.4). The 24-unit rows the design review asked for were not run.
7. **"41 times at twice the training"** is a ratio of three-seed medians at one checkpoint; the same table gives 490 at step 4,000 and 32 at 5,000. Point 7 omits "three seeds".
8. **Scoreboard.** Revised set: 7 held, 2 failed — agreed. First set: I get held Q2, Q5, Q6, Q9; partly Q1 (M9 solves 2/10), Q4 (the additive clause, 1.05 against at least 5, failed), Q7 (length-8 clause failed); the note's 4/2/1 fits if Q1 or Q4 counts as failed — list them. Q5 held on the letter only. Dropping Q3 and Q8 is fair (disclosed, decided before the runs, both would have held). But Q2/R2/R7 and Q9/R8 score the same data twice, and R1, R2 held non-blind.

## Suggestions

9. Design-review must-fixes 1–4 are addressed; `arch_m.md` still prints M7's length-8 columns unflagged, and the note's "every block … 0.9–1.6" excludes M7 (12.9) silently.
10. Caveats to add: accuracy gaps are optimisation claims at 3,000 steps with no rate decay (the 1e-5 floor is Adam noise); M7 runs at another architecture's rate; "address kept 1.00" is a median (M15 seed 6 is 0.0 — its 5.9e-2); 3/10 and 5/10 cannot separate 0.8 from 1.1; in four of five other teachers the per-unit gate is the better one; multiplicative against additive conditioning is prior art (FiLM; Jayakumar et al. 2020 — from memory, not re-read).

## The single most important correction

Finding 1: do not conclude that grouping into streams buys only compute. Rerun the candidates at 0.003 and report best against best.
