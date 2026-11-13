# Reference numbers for design revision 3 — summary and reading

Model-free; no model has been trained. Script: [`refs3.py`](refs3.py). Full per-draw tables: [refs3_heldout.md](refs3_heldout.md) (150 held-out-world instances × 5 draws of the world sets), [refs3_gate.md](refs3_gate.md) (60 training-world instances × 5 sizes × 3 draws), [refs3_cheap.md](refs3_cheap.md) (80 instances × 3 draws). H8, ρ = 0.5, Δ = 2, k ∈ {2, 3, 4}. Ranges below are over draws.

## The four mechanisms

| | what it outputs at a decision |
|---|---|
| **M** memorising posterior q_T | Bayes mixture over the *training* worlds that agree with the evidence — the optimum of the training objective wherever one exists ("in support") |
| **G** generalising posterior q | the same with all 255 worlds |
| **O** optimistic planner o | no belief: the optimal actions of the one world made of every rule not yet rejected |
| **D** deletion | the previous answer with the rejected action or rule removed (includes any belief that is never revised) |

A statistic here is a set of least-squares coefficients: the model's output (or its change after a counterfactual rejection, with the rejected rule's actions deleted from both outputs) regressed on the mechanisms' outputs (or changes), pooled over decisions (or probes). It is linear in the model, so independent output noise does not bias it.

## What each planted model reads (ranges over five draws)

**Levels, in-support decisions, held-out worlds** — coefficients on (G − M, O − M).

| training worlds | decisions in support | signal per instance | G | O | product-of-marginals posterior | training prior, never revised | full prior, never revised | noisy stand-in half-way M→G | sd at 150 instances |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 33–72 % | 0.16–0.40 | (1, 0) | (0, 1) | (0.52–0.84, −0.24–0.13) | (0.00–0.05, −0.02–0.07) | (0.91–0.99, 0.02–0.07) | (0.49–0.61, −0.07–0.03) | 0.02–0.04, 0.02–0.05 |
| 8 | 49–86 % | 0.14–0.32 | (1, 0) | (0, 1) | (0.75–0.99, −0.12–0.10) | (0.03–0.13, −0.02–0.20) | (0.91–0.98, 0.04–0.11) | (0.54–0.62, −0.05–0.01) | 0.02–0.04, 0.02–0.03 |
| 16 | 77–94 % | 0.10–0.22 | (1, 0) | (0, 1) | (0.82–1.06, −0.19–0.09) | (0.07–0.22, 0.03–0.08) | (0.90–0.94, 0.07–0.10) | (0.50–0.59, −0.06–−0.02) | 0.02–0.04, 0.02–0.03 |
| 32 | 89–98 % | 0.05–0.16 | (1, 0) | (0, 1) | (0.72–1.03, −0.13–0.10) | (0.09–0.25, 0.05–0.15) | (0.86–0.95, 0.07–0.10) | (0.55–0.64, −0.05–−0.01) | 0.03–0.05, 0.02–0.03 |
| 64 | 97–100 % | 0.03–0.06 | (1, 0) | (0, 1) | (0.66–1.09, −0.02–0.03) | (0.09–0.33, 0.06–0.12) | (0.85–0.94, 0.07–0.11) | (0.43–0.60, −0.10–−0.01) | 0.04–0.07, 0.02–0.03 |

**Changes, out-of-support rejection probes, held-out worlds** — coefficients on (dG, dO) with the two base switches controlled for.

| training worlds | probes per instance, crossing / beyond | signal per instance | G | O | M in support, G outside | M in support, O outside | memoriser → nearest training worlds | product-of-marginals posterior | half-way updater | noisy stand-in at 0 / 0.5 / 1 (coefficient on dG) | sd at 150 instances |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.7–7.4 / 3.1–7.9 | 0.27–0.34 | (1, 0) | (0, 1) | (0.98–1.01, −0.03–0.01) | (0.01–0.08, 0.89–0.99) | (−0.01–0.01, −0.01–0.01) | (1.03–1.31, −0.19–0.09) | (0.49–0.50, 0.00) | −0.05–0.01 / 0.41–0.58 / 0.96–1.07 | 0.04–0.05, 0.02–0.03 |
| 8 | 2.2–4.8 / 1.3–5.9 | 0.18–0.28 | (1, 0) | (0, 1) | (0.98–1.00, −0.01–0.01) | (0.00–0.08, 0.90–0.99) | (−0.02–0.06, −0.03–0.04) | (0.99–1.30, −0.21–0.06) | (0.49–0.50, 0.00) | −0.04–0.03 / 0.42–0.57 / 0.88–1.09 | 0.03–0.06, 0.02–0.04 |
| 16 | 1.8–2.8 / 0.6–2.2 | 0.13–0.23 | (1, 0) | (0, 1) | (0.97–1.00, −0.01–0.02) | (0.00–0.15, 0.84–1.00) | (−0.11–0.01, 0.01–0.08) | (0.96–1.29, −0.15–0.08) | (0.48–0.50, 0.00–0.01) | −0.08–0.12 / 0.36–0.49 / 0.91–1.04 | 0.05–0.11, 0.04–0.06 |
| 32 | 0.5–2.2 / 0.2–0.9 | 0.02–0.14 | (1, 0) | (0, 1) | (0.87–1.02, −0.02–0.08) | (0.00–0.14, 0.85–1.00) | (−0.22–0.11, −0.05–0.25) | (0.91–1.18, −0.12–0.10) | (0.44–0.51, −0.01–0.04) | −0.13–0.23 / 0.43–0.62 / 0.92–1.33 | 0.07–0.25, 0.04–0.19 |
| 64 | 0.2–0.6 / 0.0–0.2 | 0.00–0.03 | unreadable | | | | | | | | |

**Changes, in-support rejection probes, held-out worlds** — coefficients on (dM, dG, dO): M reads (1, 0, 0), G (0, 1, 0), O (0, 0, 1), the half-way memorising updater (0.5, 0, 0), every deleting model (0, 0, 0), at every size and draw. The product-of-marginals posterior reads (0.03–0.34, 0.40–0.92, −0.03–0.23) at 4 training worlds and (−0.02–0.16, 0.77–1.01, −0.10–0.12) from 8 up. 3–6 probes per instance at 4 training worlds, 10–13 at 32; sd at 150 instances ≤ 0.09.

**The gate, training worlds** — sum of the three coefficients on in-support probes: M, G and O each read 1.00; the product-of-marginals posterior 0.84–1.08; the half-way updater 0.50; noisy stand-ins at 0 / 0.5 / 1 read −0.17–0.12 / 0.35–0.56 / 0.76–1.07; sd at 60 instances 0.03–0.12. At 4 training worlds only 58–92 % of problems have a consistent training world, and 2–3 probes per instance stay in support.

## Reading

1. **Why a planner had to be named.** Under the statistics as first re-aimed (one generalising direction), a model with *no belief at all* — plan in the largest world not yet contradicted — reads as the generalising posterior or beyond it: γ 0.9–1.3 on two draws (0.1–0.8 on the third), out-of-support β_cf 1.0–1.4 at up to 16 training worlds (0.8–1.4 at 32), in-support changes (−0.2–0.2, 1.0–1.4) ([refs3_cheap.md](refs3_cheap.md)). With the planner as its own direction, every planted model is recovered at its own coordinates and a half-posterior, half-planner mixture reads (0.5, 0.5).
2. **Why two base-switch directions.** At a probe whose rejection takes the evidence out of support, a model that sat at M before must move its whole answer, not only reroute. With one direction, "M in support, G outside" read 1.4–1.8 on those probes, so a memoriser moving 30 % of the way would have read about 0.5. With the switches (M→G and M→O) as controlled directions, it reads (1, 0) and "M in support, O outside" reads (≤ 0.15, ≥ 0.84).
3. **Why the rejected rule is deleted from both outputs.** With deletion applied only to the earlier output, a model that ignores the verdict reads 0.09 instead of 0.
4. **What levels cannot say.** A full prior that is never revised reads (0.85–0.99, 0.02–0.11) on levels: "G" by level alone. On changes it reads zero. Levels say which prior; changes say whether, and by which mechanism, the answer is revised.
5. **A posterior with estimated marginals reads as G.** Coefficients on G of 0.5–1.3; on O within ±0.25. It is a generalising posterior with a slightly different prior, and is classed with G.
6. **A fitted 8 × 8 rule table** ("after r is rejected, multiply each remaining action by a factor that depends on r and on the action's rule"), fitted on the exact posterior's own probes, removes 18 % of the squared error of plain deletion and reads 0.12–0.22 out of support at up to 16 training worlds and up to 0.41 at 32 (single-direction statistic). It stays under 0.5 everywhere.
7. **Readable range.** Both scored statistics have signal at 4, 8, 16 and 32 training worlds and none at 64. Scaling the sampling sd from 150 to 4,000 instances (÷ 5.2): levels ≤ 0.01; out-of-support changes ≤ 0.02 up to 16 training worlds and 0.01–0.05 at 32. The 32 cell is the thin one and is checked at stage 0.
8. **One inexactness.** When the planner (or a model) has all its mass on the probed rule's actions, deleting that rule leaves nothing to renormalise and the probe is dropped for that direction; this is why "M in support, O outside" reads up to 0.15 off its coordinates at 16–32 training worlds.

Earlier files: [power.md](power.md) (first power analysis), [review1_checks.txt](review1_checks.txt) (my recount of two of review 1's numbers).
