# Review of the worlds design (DESIGN.md rev. 1, commit 808286e), before stage 0

**Verdict: run after changes.** Stage 0 passes for a reason that is not rung 4's, and the comparison lacks the baseline that separates streams from how the world is read.

Probes (`scripts/`, 250 s; **untuned: rate 0.01, 4,000 steps, teacher 10, coverage 9, one seed unless two values are shown** — indications only). `tok` is my own arm: the same monolith applied to each token separately ([h; E[p]]), pooled by a fixed rule (sum, or a softmax over one bid per token). No streams.

| arm | select: seen / held out | sum: seen / held out |
|---|---|---|
| mono | 0.088 / 1.55, 0.087 / 1.57 | 0.006 / 1.31, 0.005 / 1.00 |
| tok | 0.024 / 0.076, 0.031 / 0.084 | 0.002 / 0.004, 0.002 / 0.004 |
| streams_self | 0.013 / 0.69 | <1e-4 / 0.19 |
| streams_joint | 0.011 / 0.89 | <1e-4 / 0.40 |
| unit | 0.028 / 0.65 | <1e-4 / 0.35 |

Held-out references (select / sum): no change 1.0; a random training world's rule 1.2–1.4 / 0.9–1.1; mean of all six primitives 0.76–0.82 / 0.55–0.65; `blend` 0.46–0.50; `one` 0.93–1.03 / 0.48–0.52.

## Must fix before stage 0

1. **Stage 0 is foregone, and its failure is not rung 4's.** The monolith's held-out error is above "no change" and above ignoring the world; regressed on the six teacher primitives, its held-out update has R² 0.2–0.4. Rung 4's model sat *at* its non-composing reference (0.47 against 0.53). A control — all primitives identical, so the world is irrelevant — gave held-out 0.0000: an unseen r alone does not break it; an unconstrained function of r does. This is F52's confound; §2 lists no shortfall for it. *Change:* replace the held/seen ratio (it grows with fit, and the rate is chosen by fit) with position against the references above; add the regression and the irrelevant-world control.
2. **Add a per-token monolith.** It recombines without streams, so the planned comparison would measure the summed read r = E[a] + E[b], not streams. Rung 4's baseline read its axioms by attention.
3. **`select` is not determined in the sense tested.** `tests/test_worlds.py:26–43` treats trigger differences as known vectors; training shows only their sign. Across a cut vertex of the training graph, rescaled triggers reproduce every trained choice (agreement 1.000) and agree on 51–85 % of crossing held-out decisions (error 0.45). 7 of 12 seeds have a cut vertex at coverage 7 (seeds 2, 3; not 0, 1). If bids are free functions of the state, only 29–57 % of held-out decisions follow by transitivity. *Change:* require a 2-connected training graph; add a transitivity reference.
4. **Splits.** `sample_worlds` depends on (seed, coverage) only (`worlds.py:54`): six stage-0 runs are two splits, twelve stage-2 runs four; coverages are not nested. Separate the split seed from the initialisation seed; nest the dial.
5. **The fit condition is vacuous.** The monolith's seen error on `select` is 0.116 at full coverage and 0.088 at 9 (3–8 times the gated arms'; over 50 times on `sum`) and 0.064 at rate 0.03: the optimum may be at the grid's edge. Add an absolute condition (within 2× of a gated arm's seen error) and the width and steps to meet it; tune the wide monolith separately.
6. **The step rule cannot be applied:** no curve is recorded (`worlds.py:191`), and cosine decay flattens every curve. Compare 4,000 with 8,000 steps directly.

## Must fix before stage 2

7. **Support bar.** "The non-composing reference" is unnamed. With `one` on `select`, an arm at the `blend` level (0.5, never choosing) can pass both clauses: a third of the monolith's is 0.52. Name `blend`; require the verdict at every rate whose seen error is within 2× of the best (L57); report each held-out world (one-pass error 0.02–0.47 within one run).
8. **Weight decay cannot be selected by seen error.** Fix a grid; take its best held-out error.
9. **The coverage rule will pick 7**, where eight streams outnumber the training worlds and cut vertices occur. Compare at 9 and 11 too.
10. **Power.** Bars of 2–3× on four splits, and no identical-arm null (AGENTS.md rule 4). Run that null first; use twelve distinct splits.

## Note in the write-up

- The gated arms' success is not forced: none met the bar on `select`, only `streams_self` on `sum`.
- Checked: the oracle gate at the student's state is the right ceiling; parameters match (2,476–2,528; oracle 1,200); arms share worlds and inputs per seed; four tests pass.
- Stage 1 is foregone: keep it as a floor.

## The single change

Make it a 2 × 2: read (summed, per token) × block (monolith, streams). Drop `one` for `select` and the held/seen ratio.
